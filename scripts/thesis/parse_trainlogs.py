#!/usr/bin/env python3
"""Cluster-side helper for compute_report.py: summarise Porygon train.log files.

Reads a list of train.log paths (one per line, relative to --root) and prints one JSON object per trainer segment
(a segment starts at each "v2 trainer:" header for v2 logs; a v1 Stable-Baselines log is one segment).
Fields: path, kind (v2|v1|watchdog), t0/t1 (first/last 'upd' timestamp, UTC as logged), s0/s1 (steps), n_upd,
header (format/gen/collector/params), saved (last 'saved final_model.pt at N'), plus for watchdog logs the
'first start on <node>' / 'MODE' / 'resumes from' lines. Standard library only; read-only.
"""
import json, re, sys, os, argparse

UPD = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) upd (\d+) steps ([\d,]+) \(\+\d+\) sps (\d+)')
HDR = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) v2 trainer: (\S+) (gen\d+), collector (\S+), (.*?)params=([\d,]+)')
SAVED = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) saved final_model\.pt at ([\d,]+) steps')
V1TS = re.compile(r'total_timesteps\s*\|\s*(\d+)')
V1FPS = re.compile(r'\|\s*fps\s*\|\s*(\d+)')
V1HDR = re.compile(r'(.*?)params=([\d,]+)(.*)')
WD = re.compile(r'^(\d\d:\d\d:\d\d) (watchdog: first start on (\S+?):.*?MODE (\S+): (\S+)|fresh job: (\S+) resumes from (\S+)|fresh job: (.*)|restart in place.*|.*cuda ok.*)')


def parse(path):
    segs, cur = [], None
    v1 = {'path': path, 'kind': 'v1', 's0': None, 's1': None, 'fps': [], 'header': None}
    wd = []
    with open(path, errors='replace') as f:
        for line in f:
            if ' upd ' in line:
                m = UPD.match(line)
                if m:
                    if cur is None:
                        cur = {'path': path, 'kind': 'v2', 'header': None}
                        segs.append(cur)
                    s = int(m.group(3).replace(',', ''))
                    if 't0' not in cur:
                        cur.update(t0=m.group(1), s0=s, u0=int(m.group(2)), n_upd=0, sps_sum=0)
                    cur.update(t1=m.group(1), s1=s, u1=int(m.group(2)))
                    cur['n_upd'] += 1
                    cur['sps_sum'] += int(m.group(4))
                    continue
            if 'v2 trainer:' in line:
                m = HDR.match(line)
                cur = {'path': path, 'kind': 'v2', 'th': m.group(1) if m else line[:19],
                       'header': {'format': m.group(2), 'gen': m.group(3), 'collector': m.group(4),
                                  'battles': m.group(5).strip(' ,'), 'params': int(m.group(6).replace(',', ''))}
                       if m else {'raw': line[:300]}}
                segs.append(cur)
                continue
            if line.startswith('20') and 'saved final_model.pt' in line:
                m = SAVED.match(line)
                if m and cur is not None:
                    cur['saved'] = int(m.group(2).replace(',', ''))
                    cur['t_saved'] = m.group(1)
                continue
            if 'total_timesteps' in line:
                m = V1TS.search(line)
                if m:
                    s = int(m.group(1))
                    v1['s0'] = s if v1['s0'] is None else v1['s0']
                    v1['s1'] = s
                continue
            if '|    fps' in line:
                m = V1FPS.search(line)
                if m: v1['fps'].append(int(m.group(1)))
                continue
            if 'params=' in line and v1['header'] is None and not line.startswith('20'):
                v1['header'] = line.strip()[:400]
                continue
            m = WD.match(line)
            if m:
                wd.append(line.strip()[:400])
    out = segs
    if v1['s0'] is not None:
        fps = v1.pop('fps')
        v1['n_fps'] = len(fps)
        v1['fps_mean'] = sum(fps) / len(fps) if fps else None
        v1['fps_last'] = fps[-1] if fps else None
        out = out + [v1]
    if wd:
        out = out + [{'path': path, 'kind': 'watchdog', 'lines': wd[:200]}]
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('list')
    ap.add_argument('--root', default='/vol/bitbucket/fr1226')
    a = ap.parse_args()
    for p in open(a.list):
        p = p.strip()
        if not p: continue
        try:
            for seg in parse(os.path.join(a.root, p)):
                seg['path'] = p
                st = os.stat(os.path.join(a.root, p))
                seg['mtime'] = int(st.st_mtime)
                seg['size'] = st.st_size
                print(json.dumps(seg))
        except Exception as e:  # keep going; report the failure
            print(json.dumps({'path': p, 'kind': 'error', 'error': repr(e)}))
        sys.stdout.flush()
