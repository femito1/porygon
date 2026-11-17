#!/usr/bin/env python3
"""Porygon thesis compute report: job accounting, compute totals and training-lineage segments.

Re-runnable from the laptop (standard library only; needs `ssh gpucluster2`). Read-only on the cluster except the
raw-dump directory /vol/bitbucket/fr1226/porygon-thesis/raw/.

Sources, in order of preference (the method is recorded per row):
  1. `sacct` (slurmdbd)                                  -> method "M-sacct"
  2. the Slurm job-completion store (jobcomp/elasticsearch, JobCompLoc in `scontrol show config`): one record per
     finished job with start, end, elapsed, allocated CPUs/GPUs, node, state   -> method "M-jobcomp"
     (used when slurmdbd is down, as on 2026-11-17: "Connection refused" on localhost:6819)
  3. `squeue` for jobs still running at the snapshot (elapsed so far)         -> method "M-squeue-running"
GPU type per job comes from the node's gres (`sinfo -N -o "%N %G"`); nodes no longer in sinfo use NODE_GPU_EXTRA
(each entry cites the log line that proves it).

Training lineages: `parse_trainlogs.py` (copied to the cluster) summarises every train.log; v2 segments carry
timestamps and step counters, v1 (Stable-Baselines) logs only step counters, so v1 wall time = the job's elapsed.

Usage:
  python3 scripts/thesis/compute_report.py                 # fetch everything, rewrite CSVs + AUTO blocks
  python3 scripts/thesis/compute_report.py --offline       # rebuild from the newest cached dumps
Outputs: docs/thesis/data/{jobs,compute_by_gpu,compute_by_purpose,compute_by_month,compute_by_stream,
lineage_segments,lineages}.csv and the blocks between <!-- AUTO:BEGIN … --> / <!-- AUTO:END … --> in
docs/thesis/compute.md and docs/thesis/runs.md (text outside the blocks is never touched).
"""
import argparse
import collections
import csv
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys

HOST = 'gpucluster2'
CUSER = 'fr1226'
RAW = '/vol/bitbucket/fr1226/porygon-thesis/raw'
CLUSTER_PY = '/vol/bitbucket/fr1226/python/cpython-3.12.16/bin/python3'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THESIS = os.path.join(ROOT, 'docs', 'thesis')
DATA = os.path.join(THESIS, 'data')
CACHE = os.path.expanduser('~/.cache/thesis-raw')

# Nodes that are no longer listed by sinfo (or to pin a type): node -> (GPU type, evidence)
NODE_GPU_EXTRA = {
    'vm-kn-a16': ('A16', 'porygon-ws-fastenv/logs/slurm-687853.out "GPU 0: NVIDIA A16"; '
                         'porygon-ws-netv2/logs/slurm-710000.out "cuda ok NVIDIA A16"'),
}
GRES_NAME = {'nvidia_a30': 'A30', 'nvidia_a40': 'A40', 'nvidia_a16': 'A16', 'tesla_t4': 'T4',
             'nvidia_a100_80gb_pcie': 'A100-80GB'}

# [E] board power limits (vendor spec TDP / max board power). A16: one board = 4 GPUs at 250 W => 62.5 W per GPU.
GPU_TDP_W = {'A30': 165, 'A40': 300, 'A16': 62.5, 'T4': 70, 'A100-80GB': 300, 'RTX3090': 350}
CPU_W_PER_CORE = (3.0, 6.0)  # [E] server socket TDP 150-280 W over 32-64 cores, incl. a share of uncore

# Purpose rules (first match wins). Keep in sync with the table in compute.md.
PURPOSE_RULES = [
    ('RL training (PPO)', r'^(gloria|porygon|arm-.*|ws-netv2-(r0|r1|r1long|g9|g4|train))$'),
    ('BC pretraining / distillation', r'^(ws-netv2-pre|ws-data-bc|porygon-distill|porygon-expert)$'),
    ('benchmarks on Showdown (vs Foul Play / our search)',
     r'^(porygon-foulplay|porygon-search|porygon-oracle|cm-[AB]\d|or-(all|revealed)-\d|g9-(fit|stock|sx2)-\d|'
     r'g9-fp-smoke|ws-netv2-(fp-.*|fp9-.*|rawfp|conf\d+|foulplay|vs-search|vs9-.*|slottest2)|ws-wsp-fpcal|'
     r'ws-fastenv-(la2slot|netsearch|scale|smoke|fpsmoke|refreeze1)|ws-data-(bceval9d?|teacher9|bcbench))$'),
    ('head-to-heads (in-process / engine)',
     r'^(porygon-h2h(-nets)?|gen9-h2h-.*|ws-wsp-(h2h|spd)|ws-fastenv-(eval|cmp|helper)|ws-netv2-h2h)$'),
    ('data (crawl, re-simulation, labels, self-play data, eval fits)',
     r'^(ws-data-(host|searchlabels\d*|normalise|convert|labels-ab)|gen9-data|porygon-selfplay|porygon-fit|gen9-fit)$'),
    ('diagnostics / dev / microbenchmarks',
     r'^(ws-wsp-(v2diag|v2pos|box|selfplay|spdbench|spdfp.*)|ws-fastenv-(diag2?|bench|dbench|dev|v3gates|v3i1|'
     r'eng-spike|nnueparity|dtest|train4)|ws-netv2-prof)$'),
    ('builds / tests / probes', r'^(probe|a100-probe|porygon-build|porygon-engine|porygon-pycheck|gloria-test|'
                                r'compact|wrap|ws-netv2-smoke)$'),
]


def purpose_of(name):
    for p, rx in PURPOSE_RULES:
        if re.match(rx, name or ''):
            return p
    return 'unclassified'


def stream_of(name):
    m = re.match(r'^ws-(netv2|data|fastenv|wsp)-', name or '')
    return m.group(1) if m else 'pre-campaign (gloria repo)'


# ----------------------------------------------------------------------------------------------- fetching
def ssh(cmd, timeout=900):
    """Run CMD on the cluster; return stdout between markers (the login banner never leaks in)."""
    full = f'echo @@BEGIN@@; {cmd}; echo @@END@@ $?'
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', HOST, full], capture_output=True, text=True, timeout=timeout)
    out = r.stdout
    if '@@BEGIN@@' not in out:
        raise RuntimeError(f'ssh failed: {r.stderr[-500:]}')
    body = out.split('@@BEGIN@@', 1)[1]
    body, _, tail = body.rpartition('@@END@@')
    return body.lstrip('\n'), int((tail.strip() or '1').split()[0]), r.stderr


def fetch(stamp):
    os.makedirs(CACHE, exist_ok=True)
    ssh(f'mkdir -p {RAW}')
    got = {}
    sacct_cmd = (f'sacct -u {CUSER} -S 2025-01-01 -E now -X -P -n '
                 '-o JobID,JobName,Partition,AllocTRES,ElapsedRaw,State,Start,End,NodeList')
    out, rc, err = ssh(f'{sacct_cmd} 2>&1 | tee {RAW}/sacct-{stamp}.txt')
    if rc == 0 and out.strip() and 'error' not in out.lower()[:400]:
        got['sacct'] = out
        open(os.path.join(CACHE, f'sacct-{stamp}.txt'), 'w').write(out)
        print(f'sacct: {len(out.splitlines())} rows', file=sys.stderr)
    else:
        print(f'sacct unavailable ({out.strip().splitlines()[:1]}); using jobcomp', file=sys.stderr)
        loc, _, _ = ssh("scontrol show config | awk -F'= ' '/^JobCompLoc/{print $2}'")
        loc = loc.strip()
        base = loc.rsplit('/', 1)[0] if loc.endswith('/_doc') else loc
        q = json.dumps({'query': {'term': {'username': CUSER}}, 'sort': [{'jobid': 'asc'}]})
        out, rc, err = ssh(f"curl -s -m 120 -H 'Content-Type: application/json' '{base}/_search?size=10000' "
                           f"-d '{q}' | tee {RAW}/jobcomp-{stamp}.json")
        d = json.loads(out)
        n = d['hits']['total']['value']
        if n > len(d['hits']['hits']):
            raise RuntimeError(f'jobcomp returned {len(d["hits"]["hits"])} of {n}: page it')
        got['jobcomp'] = d
        open(os.path.join(CACHE, f'jobcomp-{stamp}.json'), 'w').write(out)
        print(f'jobcomp: {n} records from {base}', file=sys.stderr)
    out, _, _ = ssh(f'squeue -u {CUSER} -h -o "%i|%j|%P|%C|%b|%T|%S|%N|%M|%l" | tee {RAW}/squeue-{stamp}.txt')
    open(os.path.join(CACHE, f'squeue-{stamp}.txt'), 'w').write(out)
    got['squeue'] = out
    out, _, _ = ssh(f'sinfo -h -N -o "%N %G" | sort -u | tee {RAW}/nodes-gres-{stamp}.txt')
    open(os.path.join(CACHE, f'nodes-gres-{stamp}.txt'), 'w').write(out)
    got['nodes'] = out
    # training-log segments
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parse_trainlogs.py')
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', here, f'{HOST}:/vol/bitbucket/fr1226/porygon-thesis/'],
                   check=True, timeout=300)
    out, rc, _ = ssh(
        'cd /vol/bitbucket/fr1226 && (find porygon-ws-netv2/runs porygon-ws-fastenv/runs -maxdepth 5 -name train.log '
        '-size +1k 2>/dev/null; find gloria/runs -maxdepth 3 -name train.log -size +1k 2>/dev/null) '
        f'> {RAW}/trainlogs-{stamp}.txt && {CLUSTER_PY} -I porygon-thesis/parse_trainlogs.py {RAW}/trainlogs-{stamp}.txt '
        f'| tee {RAW}/trainsegs-{stamp}.jsonl', timeout=1800)
    open(os.path.join(CACHE, f'trainsegs-{stamp}.jsonl'), 'w').write(out)
    got['segs'] = out
    return got


def newest(pattern):
    fs = sorted(glob.glob(os.path.join(CACHE, pattern)))
    return fs[-1] if fs else None


def load_cached():
    got = {}
    f = newest('sacct-*.txt')
    j = newest('jobcomp-*.json')
    # prefer whichever dump is newer
    if f and (not j or os.path.basename(f)[6:] > os.path.basename(j)[8:]):
        got['sacct'] = open(f).read()
    elif j:
        got['jobcomp'] = json.load(open(j))
    got['squeue'] = open(newest('squeue-*.txt')).read()
    got['nodes'] = open(newest('nodes-gres-*.txt')).read()
    s = newest('trainsegs*.jsonl')
    got['segs'] = open(s).read() if s else ''
    got['stamp'] = re.sub(r'^.*?-(\d{8}T\d{4}Z).*$', r'\1', os.path.basename(newest('squeue-*.txt')))
    return got


# ----------------------------------------------------------------------------------------------- parsing
def ts(s):
    if not s or s in ('Unknown', 'None', 'N/A'):
        return None
    return dt.datetime.fromisoformat(s.replace(' ', 'T')[:19])


def elapsed_squeue(s):  # [D-]HH:MM:SS or MM:SS
    d = 0
    if '-' in s:
        d, s = s.split('-')
    parts = [int(x) for x in s.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(d) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def node_gpu_map(nodes_txt):
    m = {}
    for line in nodes_txt.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        g = re.search(r'gpu:([a-z0-9_]+):', parts[1])
        if g:
            m[parts[0]] = GRES_NAME.get(g.group(1), g.group(1))
    for n, (t, _) in NODE_GPU_EXTRA.items():
        m.setdefault(n, t)
    return m


def jobs_from(got, stamp_dt):
    rows = {}
    if 'sacct' in got:
        for line in got['sacct'].splitlines():
            f = line.split('|')
            if len(f) < 9 or not f[0].isdigit():
                continue
            tres = f[3]
            cpus = int((re.search(r'cpu=(\d+)', tres) or [0, 0])[1])
            gpus = int((re.search(r'gres/gpu=(\d+)', tres) or [0, 0])[1])
            rows[int(f[0])] = dict(jobid=int(f[0]), name=f[1], partition=f[2], cpus=cpus, gpus=gpus,
                                   elapsed_s=int(f[4] or 0), state=f[5].split()[0], start=ts(f[6]), end=ts(f[7]),
                                   node=None if f[8] in ('None assigned', '') else f[8], method='M-sacct')
    else:
        for h in got['jobcomp']['hits']['hits']:
            x = h['_source']
            tres = x.get('tres_alloc') or ''
            cpus = int((re.search(r'cpu=(\d+)', tres) or [0, x.get('total_cpus') or 0])[1])
            gpus = int((re.search(r'gres/gpu=(\d+)', tres) or [0, 0])[1])
            rows[x['jobid']] = dict(jobid=x['jobid'], name=x.get('job_name'), partition=x.get('partition'),
                                    cpus=cpus if x.get('@start') else 0, gpus=gpus if x.get('@start') else 0,
                                    elapsed_s=int(x.get('elapsed') or 0), state=x.get('state'),
                                    start=ts(x.get('@start')), end=ts(x.get('@end')), node=x.get('nodes'),
                                    method='M-jobcomp', work_dir=x.get('work_dir'))
    for line in got['squeue'].splitlines():
        f = line.split('|')
        if len(f) < 10 or f[5] != 'RUNNING':
            continue
        jid = int(f[0])
        if jid in rows and rows[jid]['method'] != 'M-squeue-running' and rows[jid]['state'] != 'RUNNING':
            continue
        el = elapsed_squeue(f[8])
        gpus = int((re.search(r'gpu:(?:[a-z0-9_]+:)?(\d+)', f[4]) or [0, 1])[1])
        rows[jid] = dict(jobid=jid, name=f[1], partition=f[2], cpus=int(f[3]), gpus=gpus, elapsed_s=el,
                         state='RUNNING', start=ts(f[6]), end=stamp_dt, node=f[7], method='M-squeue-running')
    return rows


def month_split(start, end, seconds):
    """Split SECONDS over calendar months in [start, end] (proportional to wall time)."""
    if not start or not end or seconds <= 0 or end <= start:
        return {start.strftime('%Y-%m') if start else 'n/a': seconds}
    out, t = collections.Counter(), start
    while t < end:
        nxt = (t.replace(day=1) + dt.timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0)
        seg = min(end, nxt) - t
        out[t.strftime('%Y-%m')] += seconds * seg.total_seconds() / (end - start).total_seconds()
        t = min(end, nxt)
    return out


# ----------------------------------------------------------------------------------------------- lineages
LIN_NAMES = {'G9': 'G9p', 'R1bF': 'R1b-F', 'G9P4': 'G9-P4', 'G9q-cd300': 'G9q-cd@305', 'G9q-cd600': 'G9q-cd@605',
             'G9q-cd800': 'G9q-cd@805', 'G9p-cd': 'G9p-cd@104.7', 'G9r-cd': 'G9r-cd@50', 'G9t-cd': 'G9t-cd@50',
             'G9q-cd': 'G9q-cd@50 (control)', 'g9qent-cd': 'G9qent-cd@880', 'g9qent-g9qcd': 'G9q-cd@885 (G9qent control)',
             'G9qent': 'G9qent', 'G9q2': 'G9q2', 'G9q': 'G9q', 'G9p': 'G9p', 'G9s': 'G9s', 'G9r': 'G9r', 'G9t': 'G9t',
             'R1b': 'R1b'}


def top_level_lineage(jobid, seg):
    """Job-level train.logs (before lineages had their own subdirectory) mix lineages; label by job and steps."""
    gen = (seg.get('header') or {}).get('gen')
    s0 = seg.get('s0') or 0
    if jobid in (706657, 710000):
        return 'R1'
    if jobid == 718701:
        if gen == 'gen9':
            return 'G9p'
        return 'R1' if s0 < 5e6 else 'R1b'
    if jobid == 741849:
        return 'R1b' if gen == 'gen4' else 'G9p'
    return f'job{jobid}'


def lineage_segments(got, jobs):
    segs = []
    for line in got['segs'].splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        p = r['path']
        if r['kind'] == 'v2':
            m = re.search(r'runs/(?:r1|g9|g4r1bf)/\d{8}-\d{6}-(\d+)/(?:([\w\-]+)/)?train\.log$', p)
            if m:
                jid = int(m.group(1))
                lin = LIN_NAMES.get(m.group(2), m.group(2)) if m.group(2) else top_level_lineage(jid, r)
            elif re.search(r'runs/as-test(-b)?/', p):  # actor-stream A/B runs, inline in G9p's A30 job
                jid, lin = 718701, 'as-test (actor-stream A/B)'
            elif re.search(r'porygon-ws-fastenv/runs/', p):
                jid, lin = None, 'fastenv dev: ' + p.split('/')[-2]
            else:
                continue  # handover-* / *-parked copies duplicate job-dir logs
            if not r.get('t0'):
                continue  # header only (saved/exited at once)
            segs.append(dict(lineage=lin, version='v2', jobid=jid, path=p, th=ts(r.get('th') or r['t0']),
                             t0=ts(r['t0']), t1=ts(r['t1']), s0=r['s0'], s1=r['s1'], n_upd=r['n_upd'],
                             gen=(r.get('header') or {}).get('gen'), params=(r.get('header') or {}).get('params'),
                             collector=(r.get('header') or {}).get('collector'), saved=r.get('saved')))
        elif r['kind'] == 'v1':
            m = re.search(r'-(\d{6,7})(?:-expert)?/train\.log$', p)
            jid = int(m.group(1)) if m else None
            d = p.split('/')[-2]
            if d.startswith('arm-'):
                lin = 'v1 ' + re.sub(r'-\d{8}-\d{6}-\d+$', '', d)
            elif 'porygon-ws-fastenv' in p:
                lin = 'fastenv dev: ' + d
            else:
                lin = 'v1 main'
            j = jobs.get(jid, {})
            pm = re.search(r'params=([\d,]+)', r.get('header') or '')
            segs.append(dict(lineage=lin, version='v1', jobid=jid, path=p, th=j.get('start'), t0=j.get('start'),
                             t1=j.get('end'), s0=r['s0'], s1=r['s1'], n_upd=r.get('n_fps'), gen='gen4',
                             params=int(pm.group(1).replace(',', '')) if pm else None,
                             collector='showdown', saved=None, fps_mean=r.get('fps_mean')))
    # de-duplicate identical segments (same lineage, s0, s1)
    seen, out = set(), []
    for s in sorted(segs, key=lambda s: (s['lineage'], s['t0'] or dt.datetime.min, s['s0'])):
        k = (s['lineage'], s['s0'], s['s1'])
        if k in seen:
            continue
        seen.add(k)
        j = jobs.get(s['jobid'], {})
        s['node'] = j.get('node')
        s['gpu_type'] = j.get('gpu_type')
        if s['version'] == 'v2':
            s['wall_h'] = (s['t1'] - s['th']).total_seconds() / 3600
            dtu = (s['t1'] - s['t0']).total_seconds()
        else:
            s['wall_h'] = (j.get('elapsed_s') or 0) / 3600
            dtu = j.get('elapsed_s') or 0
        s['steps'] = s['s1'] - s['s0']
        s['sps'] = s['steps'] / dtu if dtu > 0 else None
        out.append(s)
    return out


# ----------------------------------------------------------------------------------------------- output
def h(x):
    return f'{x:,.1f}'


def write_csv(path, rows, cols):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) if not isinstance(r.get(c), dt.datetime) else r[c].strftime('%Y-%m-%dT%H:%M:%SZ')
                        for c in cols])


def replace_block(path, tag, text):
    begin, end = f'<!-- AUTO:BEGIN {tag} -->', f'<!-- AUTO:END {tag} -->'
    src = open(path).read() if os.path.exists(path) else ''
    block = f'{begin}\n{text.rstrip()}\n{end}'
    if begin in src and end in src:
        pre, rest = src.split(begin, 1)
        _, post = rest.split(end, 1)
        src = pre + block + post
    else:
        src = src.rstrip() + '\n\n' + block + '\n'
    open(path, 'w').write(src)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--offline', action='store_true', help='use the newest cached dumps in ~/.cache/thesis-raw')
    ap.add_argument('--no-docs', action='store_true', help='write CSVs only')
    a = ap.parse_args()
    if a.offline:
        got = load_cached()
        stamp = got['stamp']
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%MZ')
        got = fetch(stamp)
    stamp_dt = dt.datetime.strptime(stamp, '%Y%m%dT%H%MZ')
    gmap = node_gpu_map(got['nodes'])
    jobs = jobs_from(got, stamp_dt)
    for j in jobs.values():
        j['gpu_type'] = gmap.get(j['node']) if j['node'] else None
        j['purpose'] = purpose_of(j['name'])
        j['stream'] = stream_of(j['name'])
        j['elapsed_h'] = j['elapsed_s'] / 3600
        j['gpu_h'] = j['elapsed_h'] * j['gpus']
        j['cpu_core_h'] = j['elapsed_h'] * j['cpus']
        j['months'] = month_split(j['start'], j['end'], j['elapsed_s'])
    os.makedirs(DATA, exist_ok=True)
    jl = sorted(jobs.values(), key=lambda j: j['jobid'])
    write_csv(os.path.join(DATA, 'jobs.csv'), jl,
              ['jobid', 'name', 'stream', 'purpose', 'partition', 'node', 'gpu_type', 'gpus', 'cpus', 'start', 'end',
               'elapsed_h', 'gpu_h', 'cpu_core_h', 'state', 'method'])

    def agg(key):
        c = collections.defaultdict(lambda: collections.Counter())
        for j in jl:
            k = key(j)
            c[k]['jobs'] += 1
            c[k]['gpu_h'] += j['gpu_h']
            c[k]['cpu_core_h'] += j['cpu_core_h']
            c[k]['elapsed_h'] += j['elapsed_h']
            if j['method'] == 'M-squeue-running':
                c[k]['running_gpu_h'] += j['gpu_h']
        return c

    tot_gpu = sum(j['gpu_h'] for j in jl)
    tot_cpu = sum(j['cpu_core_h'] for j in jl)
    run_gpu = sum(j['gpu_h'] for j in jl if j['method'] == 'M-squeue-running')
    started = [j for j in jl if j['elapsed_s'] > 0]
    src = 'sacct' if 'sacct' in got else 'jobcomp'
    lines = []
    lines.append(f'_Generated by `scripts/thesis/compute_report.py` at {stamp} (source: {src} + squeue; '
                 f'{len(jl)} job records, {len(started)} with non-zero elapsed). Every row is [M] from the job '
                 'records; the totals and splits are [D] (sums of allocated time; month splits pro rata by wall '
                 'time); running jobs count their elapsed time up to the snapshot._\n')
    lines.append('**Grand total (Imperial DoC Slurm):** '
                 f'{h(tot_gpu)} GPU-hours allocated, {h(tot_cpu)} CPU-core-hours allocated [D]; of these '
                 f'{h(run_gpu)} GPU-h belong to jobs still running at the snapshot.\n')
    # by GPU type
    lines.append('### By GPU type [D]\n')
    lines.append('| GPU | jobs | GPU-hours | CPU-core-hours | TDP per GPU [E] | GPU energy at TDP, kWh [E, upper bound] |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    g = agg(lambda j: j['gpu_type'] or '(never started)')
    egpu = 0.0
    rows = []
    for k, v in sorted(g.items(), key=lambda kv: -kv[1]['gpu_h']):
        tdp = GPU_TDP_W.get(k)
        e = v['gpu_h'] * tdp / 1000 if tdp else 0
        egpu += e
        rows.append(dict(gpu_type=k, jobs=v['jobs'], gpu_h=round(v['gpu_h'], 2), cpu_core_h=round(v['cpu_core_h'], 1),
                         tdp_w=tdp, energy_kwh_at_tdp=round(e, 1)))
        lines.append(f"| {k} | {v['jobs']} | {h(v['gpu_h'])} | {h(v['cpu_core_h'])} | {tdp or '—'} W | {h(e)} |")
    lines.append(f'| **total** | {len(jl)} | **{h(tot_gpu)}** | **{h(tot_cpu)}** | | **{h(egpu)}** |')
    write_csv(os.path.join(DATA, 'compute_by_gpu.csv'), rows, list(rows[0].keys()))
    lo, hi = CPU_W_PER_CORE
    lines.append(f'\nCPU energy [E]: {h(tot_cpu)} core-h × {lo:g}–{hi:g} W per core = '
                 f'{h(tot_cpu * lo / 1000)}–{h(tot_cpu * hi / 1000)} kWh. Total [E]: ≤ {h(egpu + tot_cpu * hi / 1000)} kWh '
                 '(GPU at TDP is an upper bound: most benchmark, data and diagnostic jobs hold a GPU only because every '
                 'partition requires one, and leave it near idle).\n')
    # by purpose
    lines.append('### By purpose [D]\n')
    lines.append('| purpose | jobs | GPU-hours | share | CPU-core-hours | share | A30 | A40 | A16 | T4 | A100 |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    rows = []
    for k, v in sorted(agg(lambda j: j['purpose']).items(), key=lambda kv: -kv[1]['gpu_h']):
        bt = collections.Counter()
        for j in jl:
            if j['purpose'] == k:
                bt[j['gpu_type']] += j['gpu_h']
        rows.append(dict(purpose=k, jobs=v['jobs'], gpu_h=round(v['gpu_h'], 2), cpu_core_h=round(v['cpu_core_h'], 1),
                         **{f'gpu_h_{t}': round(bt[t], 2) for t in ('A30', 'A40', 'A16', 'T4', 'A100-80GB')}))
        lines.append(f"| {k} | {v['jobs']} | {h(v['gpu_h'])} | {100 * v['gpu_h'] / tot_gpu:.1f}% | "
                     f"{h(v['cpu_core_h'])} | {100 * v['cpu_core_h'] / tot_cpu:.1f}% | " +
                     ' | '.join(h(bt[t]) for t in ('A30', 'A40', 'A16', 'T4', 'A100-80GB')) + ' |')
    write_csv(os.path.join(DATA, 'compute_by_purpose.csv'), rows, list(rows[0].keys()))
    # by stream
    lines.append('\n### By workstream [D]\n')
    lines.append('| stream | jobs | GPU-hours | CPU-core-hours |')
    lines.append('|---|---:|---:|---:|')
    rows = []
    for k, v in sorted(agg(lambda j: j['stream']).items(), key=lambda kv: -kv[1]['gpu_h']):
        rows.append(dict(stream=k, jobs=v['jobs'], gpu_h=round(v['gpu_h'], 2), cpu_core_h=round(v['cpu_core_h'], 1)))
        lines.append(f"| {k} | {v['jobs']} | {h(v['gpu_h'])} | {h(v['cpu_core_h'])} |")
    write_csv(os.path.join(DATA, 'compute_by_stream.csv'), rows, list(rows[0].keys()))
    # by month
    lines.append('\n### By month (pro rata by wall time) [D]\n')
    types = ['A30', 'A40', 'A16', 'T4', 'A100-80GB']
    lines.append('| month | GPU-hours | ' + ' | '.join(types) + ' | CPU-core-hours | RL training GPU-h |')
    lines.append('|---|---:|' + '---:|' * len(types) + '---:|---:|')
    mon = collections.defaultdict(collections.Counter)
    for j in jl:
        for m, sec in j['months'].items():
            frac = sec / j['elapsed_s'] if j['elapsed_s'] else 0
            mon[m]['gpu_h'] += j['gpu_h'] * frac
            mon[m]['cpu_core_h'] += j['cpu_core_h'] * frac
            mon[m][j['gpu_type']] += j['gpu_h'] * frac
            if j['purpose'].startswith('RL'):
                mon[m]['rl'] += j['gpu_h'] * frac
    rows = []
    for m in sorted(mon):
        v = mon[m]
        if v['gpu_h'] == 0 and v['cpu_core_h'] == 0:
            continue
        rows.append(dict(month=m, gpu_h=round(v['gpu_h'], 2), cpu_core_h=round(v['cpu_core_h'], 1),
                         rl_gpu_h=round(v['rl'], 2), **{f'gpu_h_{t}': round(v[t], 2) for t in types}))
        lines.append(f"| {m} | {h(v['gpu_h'])} | " + ' | '.join(h(v[t]) for t in types) +
                     f" | {h(v['cpu_core_h'])} | {h(v['rl'])} |")
    write_csv(os.path.join(DATA, 'compute_by_month.csv'), rows, list(rows[0].keys()))
    # states
    st = collections.Counter(j['state'] for j in jl)
    lines.append('\nJob states [M]: ' + ', '.join(f'{k} {v}' for k, v in st.most_common()) + '.')
    lines.append(f"\nLargest jobs [M]: " + '; '.join(
        f"{j['jobid']} {j['name']} ({j['gpu_type']}, {h(j['elapsed_h'])} h)"
        for j in sorted(jl, key=lambda j: -j['gpu_h'])[:8]) + '.')
    if not a.no_docs:
        replace_block(os.path.join(THESIS, 'compute.md'), 'compute', '\n'.join(lines))

    # lineages
    segs = lineage_segments(got, jobs)
    write_csv(os.path.join(DATA, 'lineage_segments.csv'), segs,
              ['lineage', 'version', 'jobid', 'node', 'gpu_type', 'th', 't0', 't1', 's0', 's1', 'steps', 'n_upd',
               'wall_h', 'sps', 'gen', 'params', 'collector', 'path'])
    L = collections.OrderedDict()
    for s in sorted(segs, key=lambda s: (s['t0'] or dt.datetime.min)):
        d = L.setdefault(s['lineage'], dict(lineage=s['lineage'], version=s['version'], gen=s['gen'],
                                            params=s['params'], first=s['th'], last=s['t1'], s_first=s['s0'],
                                            s_max=s['s1'], steps_trained=0, wall_h=0.0, upd_s=0.0, jobs=[],
                                            gpus=collections.Counter(), segments=0))
        d['params'] = d['params'] or s['params']
        d['gen'] = d['gen'] or s['gen'] or ('gen4' if s['lineage'].startswith('R1') else
                                            'gen9' if s['lineage'].startswith('G9') else None)
        d['last'] = max(d['last'], s['t1']) if d['last'] and s['t1'] else (s['t1'] or d['last'])
        d['s_first'] = min(d['s_first'], s['s0'])
        d['s_max'] = max(d['s_max'], s['s1'])
        d['steps_trained'] += s['steps']
        d['wall_h'] += s['wall_h']
        if s['sps']:
            d['upd_s'] += s['steps'] / s['sps']
        if s['jobid'] and s['jobid'] not in d['jobs']:
            d['jobs'].append(s['jobid'])
        d['gpus'][s['gpu_type'] or '?'] += s['wall_h']
        d['segments'] += 1
    lrows = []
    for d in L.values():
        d['mean_sps'] = d['steps_trained'] / d['upd_s'] if d['upd_s'] else None
        d['gpu_mix'] = ', '.join(f'{k} {v:.1f} h' for k, v in d['gpus'].most_common())
        d['job_ids'] = ' '.join(str(j) for j in d['jobs'])
        lrows.append(d)
    write_csv(os.path.join(DATA, 'lineages.csv'), lrows,
              ['lineage', 'version', 'gen', 'params', 'first', 'last', 's_first', 's_max', 'steps_trained', 'wall_h',
               'mean_sps', 'segments', 'gpu_mix', 'job_ids'])
    out = [f'_Generated by `scripts/thesis/compute_report.py` at {stamp} from the train.log files (v2: timestamps '
           'and step counters in each `upd` line [M]; v1: step counters [M], wall time = job elapsed [M]). '
           'Wall = Σ segment (last upd − trainer start) [D]; mean steps/s = Σ steps / Σ (last − first upd) [D]; '
           '"steps trained" counts re-done steps after a resume from an older checkpoint._\n',
           '| lineage | gen | params | first start | last log | steps first → max | steps trained | wall h | '
           'mean steps/s | GPU (wall h) | jobs |', '|---|---|---:|---|---|---|---:|---:|---:|---|---|']
    for d in lrows:
        if d['lineage'].startswith('fastenv dev'):
            continue
        f = d['first'].strftime('%m-%d %H:%M') if d['first'] else '?'
        l_ = d['last'].strftime('%m-%d %H:%M') if d['last'] else '?'
        out.append(f"| {d['lineage']} | {d['gen'] or ''} | {d['params']:,} | {f} | {l_} | "
                   f"{d['s_first'] / 1e6:,.2f}M → {d['s_max'] / 1e6:,.2f}M | {d['steps_trained'] / 1e6:,.1f}M | "
                   f"{d['wall_h']:.1f} | {d['mean_sps']:,.0f} | {d['gpu_mix']} | {d['job_ids']} |"
                   if d['params'] else
                   f"| {d['lineage']} | {d['gen'] or ''} | ? | {f} | {l_} | {d['s_first'] / 1e6:,.2f}M → "
                   f"{d['s_max'] / 1e6:,.2f}M | {d['steps_trained'] / 1e6:,.1f}M | {d['wall_h']:.1f} | "
                   f"{(d['mean_sps'] or 0):,.0f} | {d['gpu_mix']} | {d['job_ids']} |")
    if not a.no_docs:
        replace_block(os.path.join(THESIS, 'runs.md'), 'lineages', '\n'.join(out))
    print(f'jobs {len(jl)}  GPU-h {tot_gpu:.1f}  CPU-core-h {tot_cpu:.1f}  lineages {len(lrows)}', file=sys.stderr)


if __name__ == '__main__':
    main()
