#!/bin/bash
# Laptop dry run of a FRESH job of scripts/slurm_train_v2.sbatch (a requeue, a chained or an early-submitted successor)
# against a copy of the cluster's netv2.env: a fake ROOT with the run state after the 2026-11-09 12:40 swap (G9p on the
# A30 = r1 MODE g9p, lineage dir G9p; G9s on the A16 = g9 MODE g9s; the pre-swap lineage dirs, the 12:40 handovers, the
# stale 04:02 top-level files, chain_left / marker files), stub python (trainer + import checks), sbatch, squeue,
# scontrol, srun, pkill, sleep and nproc first on PATH; the job script runs unmodified. The stub trainer logs the args that
# win (argparse: last wins), the pool it sees and what it resumed from, writes final_model.pt + a train.log line where the
# real one would and exits (MOCK_WAIT=1: waits for SIGINT first); with runs/RUN/restart armed the restart loop runs once.
#   bash scripts/netv2_job_mock.sh SBATCH_SCRIPT ENVFILE [old]   (old = a spooled pre-fix job script: no chain-flag case)
# Expectations are the CORRECT behaviour: FAIL lines are gaps.
set -u
SB=$(readlink -f "$1"); ENVF=$(readlink -f "$2"); OLD=${3:-}
M=${MOCK_ROOT:-/tmp/netv2-job-mock}
pass=0; fail=0
ok() { pass=$((pass + 1)); echo "  PASS $1"; }
bad() { fail=$((fail + 1)); echo "  FAIL $1"; }
expect() { if printf '%s\n' "$3" | grep -qE -- "$2"; then ok "$1"; else bad "$1: /$2/ not in: $3"; fi; }
stubs() {
  rm -rf "$M"; mkdir -p "$M/bin" "$M/site/torch"
  cat > "$M/bin/python" <<'EOF'
#!/usr/bin/env python3
# stub "python": the job script's -c checks, and porygon.v2.train (Python, like the real trainer: SIGINT is ignored in
# a background child of a non-interactive bash, and only signal.signal re-enables it -- a bash stub cannot trap it)
import os, signal, sys, time
a = sys.argv[1:]; M = os.environ["MOCK"]
if a and a[0] == "-c":
    print("cuda ok (mock)" if "assert" in a[1] else f"{M}/site/torch" if "dirname" in a[1] else ""); sys.exit(0)
if a[:3] != ["-u", "-m", "porygon.v2.train"]:
    sys.exit(f"mock python: unexpected {a}")
v = {}
for i, x in enumerate(a):
    if x in ("--resume", "--run-dir", "--pool-dir", "--anchor") and i + 1 < len(a):
        v[x] = a[i + 1]  # argparse: last wins
resume, rundir, pooldir = v.get("--resume", ""), v.get("--run-dir", ""), v.get("--pool-dir", "")
time.sleep(1)  # a background pool copy lands meanwhile
def n(d): return len([f for f in os.listdir(d) if f.endswith(".pt")]) if os.path.isdir(d) else 0
def rd(f):
    try: return open(f).read().strip()
    except OSError: return "MISSING"
yn = lambda f: "y" if os.path.isfile(f) else "n"
log = open(f"{M}/calls.log", "a")
log.write(f"call resume={resume} [{rd(resume)}] run_dir={rundir} pool_dir={pooldir} pool={n(pooldir)} "
          f"frozen={n(pooldir + '/frozen')} best={yn(rundir + '/best_model.pt')} ema={yn(rundir + '/ema_model.pt')} "
          f"anchor={v.get('--anchor', '-')} ps={os.environ.get('PORYGON_PS', '-')} pp={os.environ.get('PYTHONPATH', '')}\n"); log.flush()
jid = os.environ.get("SLURM_JOB_ID", "")
for pend in (f"{M}/pending-inline-{jid}", f"{M}/pending-inline"):  # S8 / S10: an INLINE_CMD queued while this trainer
    if os.path.exists(pend) and os.path.exists(pend + ".dest"):  # runs (as the driver does), moved into place now
        os.replace(pend, open(pend + ".dest").read().strip()); os.remove(pend + ".dest")
brk = f"{M}/break-{jid}"  # S10c: files this job's trainer makes unreadable (a copy that fails midway)
if os.path.exists(brk):
    for f in open(brk).read().split():
        try: os.chmod(f, 0)
        except OSError: pass
    os.remove(brk)
if os.environ.get("MOCK_WAIT") == "1":
    got = []
    signal.signal(signal.SIGINT, lambda *_: got.append(1))
    for _ in range(300):
        if got: break
        time.sleep(0.1)
    log.write("trainer: " + ("got SIGINT" if got else "no SIGINT within 30 s") + "\n"); log.flush()
os.makedirs(rundir, exist_ok=True); os.makedirs(pooldir, exist_ok=True)
with open(rundir + "/final_model.pt.tmp", "w") as f: f.write("trained-from:" + rd(resume) + "\n")
os.replace(rundir + "/final_model.pt.tmp", rundir + "/final_model.pt")
with open(rundir + "/train.log", "a") as f: f.write("2026-11-11 21:40:00 upd 9 steps 99,000,000 (+43000) sps 2200\n")
EOF
  cat > "$M/bin/sbatch" <<'EOF'
#!/bin/bash
echo "sbatch $*" >> "$MOCK/sbatch.log"; echo "Submitted batch job 999999"
EOF
  for c in squeue scontrol srun pkill; do printf '#!/bin/bash\necho "%s $*" >> "$MOCK/other.log"\nexit 0\n' "$c" > "$M/bin/$c"; done
  printf '#!/bin/bash\n# long sleeps fail at once (ends the job script'"'"'s background while-sleep loops)\ncase $1 in 0*|1|2|3|4) exec /bin/sleep "$1";; esac\nexit 1\n' > "$M/bin/sleep"
  printf '#!/bin/bash\necho "${SLURM_CPUS_ON_NODE:-30}"\n' > "$M/bin/nproc"
  # nvidia-smi: the GPU memory a scenario writes to $MOCK/gpu_mib (none: fails, like a node without a working GPU)
  printf '#!/bin/bash\nf=$MOCK/gpu_mib-$SLURM_JOB_ID; [ -f "$f" ] || f=$MOCK/gpu_mib\n[ -f "$f" ] || exit 9\ncat "$f"\n' > "$M/bin/nvidia-smi"
  chmod +x "$M"/bin/*
}
at() { touch -d "2026-11-$1" "${@:2}"; }  # at 'DD HH:MM' files...
lineage() {  # lineage DIR TAG MTIME_FINAL SNAPSHOTS... : a mirrored lineage dir with final/ckpt/best/ema/pool/frozen
  local d=$1 tag=$2 t=$3; shift 3; mkdir -p "$d/checkpoints" "$d/pool/frozen"
  echo "$tag-ckpt" > "$d/checkpoints/ckpt_x.pt"; echo "$tag-final" > "$d/final_model.pt"; at "$t" "$d/final_model.pt"
  touch -d "$(date -d "2026-11-$t" '+%F %T') 30 minutes ago" "$d/checkpoints/ckpt_x.pt"
  echo "$tag-best" > "$d/best_model.pt"; echo "$tag-ema" > "$d/ema_model.pt"
  echo P1 > "$d/pool/frozen/P1-g9h-dmg-s2.pt"; echo "$tag-bestnet" > "$d/pool/frozen/best-x.pt"
  for s in "$@"; do echo "$tag-$s" > "$d/pool/$s.pt"; done
}
tree() {  # tree post|pre : the cluster state after the 12:40 swap (projected to the time limit), or MODE files as before
  rm -rf "$M/nv2" "$M/tmp" "$M"/*.log; mkdir -p "$M/tmp"
  N=$M/nv2; mkdir -p "$N/scripts" "$N/pokemon-showdown" "$N/runs/pretrain" "$N/runs/g9s" "$N/runs/g9a30"
  cp "$SB" "$N/scripts/slurm_train_v2.sbatch"; cp "$ENVF" "$N/netv2.env"
  printf '#!/bin/bash\nexit 0\n' > "$N/scripts/start_servers.sh"; chmod +x "$N/scripts/start_servers.sh"; echo '{}' > "$N/pokemon-showdown/package.json"
  echo P1 > "$N/runs/pretrain/P1-g9h-dmg-s2.pt"; echo init-g9s > "$N/runs/g9s/init-g9s.pt"
  R=$N/runs/r1; J=$R/20261108-220401-718701; G=$N/runs/g9; K=$G/20261109-021316-741849
  mkdir -p "$R/pool" "$G/pool" "$J/checkpoints" "$K" "$R/handover-g9p/pool" "$G/handover-g9p-a30/pool" "$R/handover-g9s-a16/pool"
  echo 0 > "$R/chain_left"; echo 0 > "$G/chain_left"
  # R1b-F's parked state (2026-11-09 05:00) and follow-up #1's start point
  mkdir -p "$N/runs/r1b-parked/pool"; echo R1bF-parked-final > "$N/runs/r1b-parked/final_model.pt"; echo R1bF-parked-best > "$N/runs/r1b-parked/best_model.pt"
  echo R1bF-parked-ema > "$N/runs/r1b-parked/ema_model.pt"; echo best75 > "$R/restart-best-7.5M.pt"
  for s in 00002540836 00007500005 00020009055; do echo "R1bF-$s" > "$N/runs/r1b-parked/pool/$s.pt"; done
  touch "$R/G9_STARTED" "$R/LINEAGE_DIRS" "$R/R1B_PARKED" "$G/G9P_STARTED" "$G/LINEAGE_DIRS" "$G/R1B_STARTED" "$G/pre_done"
  echo stale-top > "$J/final_model.pt"; at '09 04:02' "$J/final_model.pt"; echo stale-R1b-top > "$K/final_model.pt"; at '09 04:03' "$K/final_model.pt"
  echo G9p-handover-13M > "$R/handover-g9p/final_model.pt"; at '09 05:00' "$R/handover-g9p/final_model.pt"
  echo G9p-handover-26M > "$G/handover-g9p-a30/final_model.pt"; echo G9s-handover-56M > "$R/handover-g9s-a16/final_model.pt"
  at '09 12:40' "$G/handover-g9p-a30/final_model.pt" "$R/handover-g9s-a16/final_model.pt"
  mkdir -p "$J/G9"; echo stale-G9 > "$J/G9/final_model.pt"; at '09 05:00' "$J/G9/final_model.pt"
  # pre-swap segments (frozen at 12:40) and post-swap segments (newest, near the time limit)
  lineage "$J/G9s" G9s-preswap '09 12:39' 00002507561 00050037964 00055002732
  lineage "$K/G9" G9p-preswap '09 12:39' 00002509811 00025009642
  lineage "$J/G9p" G9p-postswap '11 21:47' 00002509811 00025009642 00150000000
  lineage "$K/G9s" G9s-postswap '12 01:45' 00002507561 00050037964 00080000000
  if [ "$1" = post ]; then
    echo g9p > "$R/MODE"; echo g9s > "$G/MODE"; touch "$R/G9P_A30_STARTED" "$G/G9S_A16_STARTED"
  else  # the swap back: MODE files as before 12:40, the pre-swap lineage dirs are the newest again
    echo g9s > "$R/MODE"; echo g9p > "$G/MODE"; rm -rf "$J/G9p" "$K/G9s"
    at '11 21:47' "$J/G9s/final_model.pt"; at '12 01:45' "$K/G9/final_model.pt"
  fi
}
job() {  # job RUN_NAME JOBID CPUS PARTITION NAME [restart] -> runs the job script as Slurm would; prints its log
  local run=$1 id=$2 cpus=$3 part=$4 name=$5
  [ "${6:-}" = restart ] && touch "$M/nv2/runs/$run/restart"
  ( cd "$M/nv2" && env -i HOME="$HOME" USER="${USER:-nand}" PATH="$M/bin:/usr/bin:/bin" MOCK="$M" NV2="$M/nv2" PY="$M/bin/python" \
      TMPDIR="$M/tmp" SLURM_SUBMIT_DIR="$M/nv2" SLURM_JOB_ID="$id" SLURM_JOB_NAME="$name" SLURM_JOB_PARTITION="$part" \
      SLURM_CPUS_ON_NODE="$cpus" SLURM_CPUS_PER_TASK=6 SLURM_MEM_PER_NODE="$([ "$part" = a16 ] && echo 65536 || echo 98304)" \
      NODE_BIN=/nonexistent NV2_SWAP_SLEEP=1 NV2_SWAP_ITERS="${MOCK_SWAP_ITERS:-240}" ${MOCK_END:+SLURM_JOB_END_TIME=$MOCK_END NV2_DEADLINE_POLL_S=1 NV2_DEADLINE_PAT=bin/python.-u.-m.porygon.v2.train} MOCK_WAIT="${MOCK_WAIT:-}" \
      setsid -w bash -c 'echo $$ > "$MOCK/sid-$SLURM_JOB_ID"; exec bash "$MOCK/nv2/scripts/slurm_train_v2.sbatch" "$1"' _ "$run" > "$M/job-$id.out" 2>&1 )
  local rc=$? sid; sid=$(cat "$M/sid-$id" 2>/dev/null)
  if [ -n "$sid" ]; then /bin/sleep 1; pkill -s "$sid" 2>/dev/null; fi  # leftovers of this mock job only (its own session)
  echo "  exit $rc ; job log (trimmed):"; grep -v -E "pre-read|find: |^$" "$M/job-$id.out" | cut -c1-200 | sed 's/^/    | /' | head -n 12
}
calls() { c=$(cat "$M/calls.log" 2>/dev/null); printf '  trainer calls:\n%s\n' "$(sed 's/^/    /' <<< "$c")"; }
stubs; export MOCK=$M
T=$M/tmp
echo "== S1: r1 fresh job after 718701, post-swap (MODE g9p = G9p on the A30), restart flag armed"
tree post; echo 24576 > "$M/gpu_mib"; job r1 900001 30 a30 ws-netv2-r1 restart; calls
expect "S1 resumes the newest mirrored G9p file (post-swap segment)" "^call resume=.*/20261108-220401-718701/G9p/final_model.pt \[G9p-postswap-final\]" "$(sed -n 1p <<< "$c")"
expect "S1 G9p lineage dirs" "run_dir=$T/ws-netv2-900001/run/G9p pool_dir=$T/ws-netv2-900001/run/G9p/pool" "$(sed -n 1p <<< "$c")"
expect "S1 pool (3) + frozen (2) + best/ema re-seeded" "pool=3 frozen=2 best=y ema=y" "$(sed -n 1p <<< "$c")"
expect "S1 restart in place relaunches from the lineage's own save" "^call resume=.*/run/G9p/final_model.pt \[trained-from:G9p-postswap-final\]" "$(sed -n 2p <<< "$c")"
expect "S1 G9p recipe on the A30 (KL anchor, 28 workers, cache 192, stacked-opp)" "kl-anchor-coef 0.1 .*fastenv-workers 28 .*--snapshot-cache 192 --stacked-opp" "$(cat "$M/job-900001.out")"
expect "S1 no resubmission (chain_left 0)" "^$" "$(cat "$M/sbatch.log" 2>/dev/null)"
echo "== S2: g9 fresh job after 741849, post-swap (MODE g9s = G9s on the A16), restart flag armed, chain_left 1"
tree post; echo 15356 > "$M/gpu_mib"; echo 1 > "$M/nv2/runs/g9/chain_left"; job g9 900002 10 a16 ws-netv2-g9 restart; calls
expect "S2 resumes the newest mirrored G9s file (post-swap, not the pre-swap A30 segment)" "^call resume=.*/20261109-021316-741849/G9s/final_model.pt \[G9s-postswap-final\]" "$(sed -n 1p <<< "$c")"
expect "S2 G9s lineage dirs" "run_dir=$T/ws-netv2-900002/run/G9s pool_dir=$T/ws-netv2-900002/run/G9s/pool" "$(sed -n 1p <<< "$c")"
expect "S2 pool (3) + frozen (2) + best/ema re-seeded" "pool=3 frozen=2 best=y ema=y" "$(sed -n 1p <<< "$c")"
expect "S2 restart in place relaunches from the lineage's own save" "^call resume=.*/run/G9s/final_model.pt \[trained-from:G9s-postswap-final\]" "$(sed -n 2p <<< "$c")"
expect "S2 G9s recipe on the A16 (8 workers, cache 72, no KL anchor)" "guard-anchor-drop 0.10 .*fastenv-workers 8 .*--snapshot-cache 72 " "$(cat "$M/job-900002.out")"
s=$(cat "$M/sbatch.log" 2>/dev/null); echo "  sbatch: ${s:-none}"
if [ -z "$OLD" ]; then
  expect "S2 resubmission carries name / a16 / gres / 10 CPUs / 64G" "--job-name=ws-netv2-g9 --partition=a16 --gres=gpu:1 --cpus-per-task=10 --mem=65536M --time=72:00:00 scripts/slurm_train_v2.sbatch g9" "$s"
else
  echo "  (old spooled script: its bare chain sbatch cannot carry flags; keep chain_left 0 and queue the successor by hand)"
fi
echo "== S2b: the successors on an a40 (46068 MiB): G9p cache 640; G9s cache 512 + stacked-opp; no nvidia-smi: A30 / A16 values"
tree post; echo 46068 > "$M/gpu_mib"; job r1 900011 30 a40 ws-netv2-r1 restart; calls
expect "S2b G9p on an a40: cache 640 + stacked-opp" "fastenv-workers 28 .*--snapshot-cache 640 --stacked-opp" "$(cat "$M/job-900011.out")"
tree post; echo 46068 > "$M/gpu_mib"; job g9 900012 10 a40 ws-netv2-g9 restart; calls
expect "S2b G9s on an a40: cache 512 + stacked-opp, 8 workers" "fastenv-workers 8 .*--snapshot-cache 512 --stacked-opp" "$(cat "$M/job-900012.out")"
tree post; rm -f "$M/gpu_mib"; job g9 900013 10 a16 ws-netv2-g9 restart; calls
expect "S2b G9s, GPU unknown: the A16 value (72, no stacked-opp)" "fastenv-workers 8 .*--snapshot-cache 72 --resume" "$(cat "$M/job-900013.out")"
if [ -z "$OLD" ]; then  # R1b-F is submitted fresh with the current job script, never under an old spooled one
  echo "== S5: R1b-F resume: a fresh g4r1bf job on a dove T4 (training, 10 CPUs), restart flag armed; then its chained job"
  tree post; mkdir -p "$M/nv2/runs/g4r1bf"; echo 15360 > "$M/gpu_mib"; job g4r1bf 900021 10 training ws-netv2-g4 restart; calls
  expect "S5 first start resumes the parked R1b-F state with its pool + best / ema" "^call resume=$M/nv2/runs/r1b-parked/final_model.pt \[R1bF-parked-final\] run_dir=$T/ws-netv2-900021/run/R1bF pool_dir=$T/ws-netv2-900021/run/R1bF/pool pool=3 frozen=0 best=y ema=y" "$(sed -n 1p <<< "$c")"
  expect "S5 restart in place relaunches from the lineage's own save" "^call resume=$T/ws-netv2-900021/run/R1bF/final_model.pt \[trained-from:R1bF-parked-final\]" "$(sed -n 2p <<< "$c")"
  expect "S5 gen4 R1b-F recipe (8 workers, cache 40, strength watch 0.0999)" "fastenv-workers 8 .*--snapshot-cache 40 --guard-pg-mean 20 --guard-anchor-drop 0.0999" "$(cat "$M/job-900021.out")"
  s=$(cat "$M/sbatch.log" 2>/dev/null); echo "  sbatch: ${s:-none}"
  {
    expect "S5 chain resubmission carries name / training / gres / 10 CPUs / 72 h" "--job-name=ws-netv2-g4 --partition=training --gres=gpu:1 --cpus-per-task=10 --mem=98304M --time=72:00:00 scripts/slurm_train_v2.sbatch g4r1bf" "$s"
  }
  job g4r1bf 900022 10 training ws-netv2-g4; calls
  expect "S5b the chained job resumes the mirrored R1bF save (not the parked state)" "^call resume=$M/nv2/runs/g4r1bf/[0-9-]+-900021/R1bF/final_model.pt \[trained-from:trained-from:R1bF-parked-final\]" "$(sed -n 3p <<< "$c")"
else
  echo "== S5 skipped: R1b-F (g4r1bf) is a fresh submission of the current job script"
fi
wrapper() {  # the v5 wrapper in the fake NV2: src-24c633f first on PYTHONPATH, then the mock python
  mkdir -p "$M/nv2/bin"
  printf '#!/bin/bash\nexport PYTHONPATH=%s/nv2/src-24c633f${PYTHONPATH:+:$PYTHONPATH}\nexec "$MOCK/bin/python" "$@"\n' "$M" > "$M/nv2/bin/py-src-24c633f"
  chmod +x "$M/nv2/bin/py-src-24c633f"
}
if [ -z "$OLD" ]; then  # the successors were spooled with the current job script (d913fdbc)
  echo "== S6: successor src (v5, 2026-11-10): a fresh r1 job trains through \$NV2/bin/py-src-24c633f (src-24c633f first, --snapshot-cache-cpu 400)"
  tree post; echo 24576 > "$M/gpu_mib"; wrapper
  job r1 900031 30 a30 ws-netv2-r1 restart; calls
  expect "S6 the successor's trainer runs with src-24c633f first" "^call resume=.*/G9p/final_model.pt .* pp=$M/nv2/src-24c633f:$M/nv2/src" "$(sed -n 1p <<< "$c")"
  expect "S6 its restart in place too" "pp=$M/nv2/src-24c633f:" "$(sed -n 2p <<< "$c")"
  expect "S6 successor args: cache 192 + stacked-opp + cpu tier 400" "--snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400" "$(cat "$M/job-900031.out")"
  tree post; echo 15356 > "$M/gpu_mib"; wrapper
  job g9 900032 10 a16 ws-netv2-g9 restart; calls
  expect "S6 the G9s successor: src-24c633f first, cache 72 + cpu tier 200" "pp=$M/nv2/src-24c633f:" "$(sed -n 1p <<< "$c")"
  expect "S6 G9s successor args" "--snapshot-cache 72 --snapshot-cache-cpu 200 --resume" "$(cat "$M/job-900032.out")"
  echo "== S6b: deadline watch through the wrapper: the job ends in 15 min + 3 s; the trainer gets SIGINT"
  tree post; echo 24576 > "$M/gpu_mib"; wrapper
  MOCK_END=$(( $(date +%s) + 903 )) MOCK_WAIT=1 job r1 900033 30 a30 ws-netv2-r1 restart; calls
  expect "S6b the wrapped trainer got SIGINT from the watcher" "^trainer: got SIGINT" "$(sed -n 2p <<< "$c")"
  # S5c (v7, 2026-11-10): R1b-F with the wrapper present: job 829660 (spooled = the current script) and its chained
  # jobs train through src-24c633f with --stacked-opp, otherwise the S5 recipe (no cpu tier)
  echo "== S5c: R1b-F (job 829660 and a chained job) through \$NV2/bin/py-src-24c633f with --stacked-opp"
  tree post; mkdir -p "$M/nv2/runs/g4r1bf"; echo 15360 > "$M/gpu_mib"; wrapper
  job g4r1bf 829660 10 training ws-netv2-g4 restart; calls
  expect "S5c 829660 first launch: src-24c633f first" "pp=$M/nv2/src-24c633f:$M/nv2/src$" "$(sed -n 1p <<< "$c")"
  expect "S5c 829660 restart in place: its own save, src-24c633f first" "^call resume=$T/ws-netv2-829660/run/R1bF/final_model.pt .* pp=$M/nv2/src-24c633f:$M/nv2/src$" "$(sed -n 2p <<< "$c")"
  expect "S5c args: cache 40 + stacked-opp, then the strength watch" "fastenv-workers 8 .*--snapshot-cache 40 --stacked-opp --guard-pg-mean 20 --guard-anchor-drop 0.0999" "$(cat "$M/job-829660.out")"
  if grep -q "snapshot-cache-cpu" "$M/job-829660.out"; then bad "S5c R1b-F got the cpu tier"; else ok "S5c R1b-F without the cpu tier"; fi
  job g4r1bf 900023 10 training ws-netv2-g4; calls
  expect "S5c the chained job: mirrored R1bF save, src-24c633f first" "^call resume=$M/nv2/runs/g4r1bf/[0-9-]+-829660/R1bF/final_model.pt .* pp=$M/nv2/src-24c633f:$M/nv2/src$" "$(sed -n 3p <<< "$c")"
fi
# S7 (v5, both the current and the spooled scripts): job 718701 (G9p) trains through the wrapper (src-24c633f first)
# with the args it had before (no CPU tier); job 741849 (G9s) keeps $ROOT/src only; restarts included
echo "== S7: job 718701 runs src-24c633f first with unchanged args; job 741849 keeps \$ROOT/src (restarts included)"
tree post; echo 24576 > "$M/gpu_mib"; wrapper
job r1 718701 30 a30 ws-netv2-r1 restart; calls
expect "S7 718701 first launch: src-24c633f first" "pp=$M/nv2/src-24c633f:$M/nv2/src$" "$(sed -n 1p <<< "$c")"
expect "S7 718701 restart in place: src-24c633f first" "^call resume=.*/run/G9p/final_model.pt .* pp=$M/nv2/src-24c633f:$M/nv2/src$" "$(sed -n 2p <<< "$c")"
expect "S7 718701 args: cache 192 + stacked-opp, no cpu tier" "--snapshot-cache 192 --stacked-opp --resume" "$(cat "$M/job-718701.out")"
if grep -q "snapshot-cache-cpu" "$M/job-718701.out"; then bad "S7 718701 got the cpu tier"; else ok "S7 718701 without the cpu tier"; fi
tree post; echo 15356 > "$M/gpu_mib"; wrapper
job g9 741849 10 a16 ws-netv2-g9 restart; calls
expect "S7 741849 first launch: \$ROOT/src only" " pp=$M/nv2/src$" "$(sed -n 1p <<< "$c")"
expect "S7 741849 restart in place: \$ROOT/src only" "^call resume=.*/run/G9s/final_model.pt .* pp=$M/nv2/src$" "$(sed -n 2p <<< "$c")"
if grep -q -E "snapshot-cache-cpu|py-src" "$M/job-741849.out"; then bad "S7 741849 switched"; else ok "S7 741849 unchanged"; fi
# S7b (v8, 2026-11-10; both the current and the spooled scripts): with $NV2/bin/py-src-4f18108 present, job 718701
# (G9p) and an r1 successor train through it (src-4f18108 first) with --actor-stream --decoupled in place of
# --no-overlap; job 741849 (G9q / G9s) keeps $ROOT/src and --no-overlap
echo "== S7b: G9p (718701, r1 successor) through \$NV2/bin/py-src-4f18108 with --actor-stream --decoupled; 741849 unchanged"
aswrapper() {
  printf '#!/bin/bash\nexport PYTHONPATH=%s/nv2/src-4f18108${PYTHONPATH:+:$PYTHONPATH}\nexec "$MOCK/bin/python" "$@"\n' "$M" > "$M/nv2/bin/py-src-4f18108"
  chmod +x "$M/nv2/bin/py-src-4f18108"
}
tree post; echo 24576 > "$M/gpu_mib"; wrapper; aswrapper
job r1 718701 30 a30 ws-netv2-r1 restart; calls
expect "S7b 718701 first launch: src-4f18108 first" "pp=$M/nv2/src-4f18108:$M/nv2/src$" "$(sed -n 1p <<< "$c")"
expect "S7b 718701 restart in place: its own save, src-4f18108 first" "^call resume=.*/run/G9p/final_model.pt .* pp=$M/nv2/src-4f18108:$M/nv2/src$" "$(sed -n 2p <<< "$c")"
expect "S7b 718701 args: actor-stream + decoupled, cache 192 + stacked-opp" "--fastenv-battles 48 --actor-stream --decoupled .*--snapshot-cache 192 --stacked-opp --resume" "$(cat "$M/job-718701.out")"
if grep -q -E "no-overlap|snapshot-cache-cpu" "$M/job-718701.out"; then bad "S7b 718701 kept --no-overlap or got the cpu tier"; else ok "S7b 718701 without --no-overlap / cpu tier"; fi
tree post; echo 24576 > "$M/gpu_mib"; wrapper; aswrapper
job r1 900041 30 a30 ws-netv2-r1 restart; calls
expect "S7b r1 successor: src-4f18108 first, actor-stream + cpu tier 400" "pp=$M/nv2/src-4f18108:$M/nv2/src$" "$(sed -n 1p <<< "$c")"
expect "S7b r1 successor args" "--actor-stream --decoupled .*--snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400" "$(cat "$M/job-900041.out")"
tree post; echo 15356 > "$M/gpu_mib"; wrapper; aswrapper
job g9 741849 10 a16 ws-netv2-g9 restart; calls
expect "S7b 741849: \$ROOT/src only, --no-overlap" "pp=$M/nv2/src$" "$(sed -n 2p <<< "$c")"
if grep -q -E "actor-stream|py-src" "$M/job-741849.out"; then bad "S7b 741849 switched"; else ok "S7b 741849 unchanged"; fi
# S8 (2026-11-10, user decision G9q option A; both the current and the spooled scripts): in job 741849 the park
# INLINE_CMD (as written for the cluster: G9s's state to runs/g9s-parked/, then MODE g9q) runs at a restart in place;
# the relaunch is G9q's first start from P3 (G9q lineage dirs, November sets, $ROOT/src only, args with P3); its next
# restart in place resumes G9q's own save
echo "== S8: 741849 parks G9s through the INLINE_CMD and relaunches as G9q (first start from P3), then restarts G9q in place"
tree post; echo 15356 > "$M/gpu_mib"; wrapper; echo P3 > "$M/nv2/runs/pretrain/P3-g9-s2.pt"
G=$M/nv2/runs/g9
echo "$G/INLINE_CMD" > "$M/pending-inline.dest"  # the stub trainer queues it during G9s's run (the driver's timing)
cat > "$M/pending-inline" <<'INL'
set -u
C=$NV2; L=$LOCAL/run/G9s; P=$C/runs/g9s-parked
[ -f "$L/final_model.pt" ] || { echo "no G9s final_model.pt: not parking"; exit 1; }
rm -rf "$P.tmp"; mkdir -p "$P.tmp"; cp -p "$L/final_model.pt" "$P.tmp/"; cp -rp "$L/pool" "$P.tmp/pool"; mv "$P.tmp" "$P"
cmp -s "$L/final_model.pt" "$P/final_model.pt" && echo g9q > "$C/runs/g9/MODE" && touch "$C/runs/g9/restart"
INL
job g9 741849 10 a16 ws-netv2-g9 restart; calls
expect "S8 first launch: G9s (its newest mirrored file)" "^call resume=.*/G9s/final_model.pt \[G9s-postswap-final\] run_dir=$T/ws-netv2-741849/run/G9s .* ps=- pp=$M/nv2/src$" "$(sed -n 1p <<< "$c")"
expect "S8 G9s parked with its last save" "^trained-from:G9s-postswap-final$" "$(cat "$M/nv2/runs/g9s-parked/final_model.pt" 2>/dev/null)"
expect "S8 relaunch = G9q's first start: from P3, G9q dirs, November sets, \$ROOT/src only" "^call resume=$M/nv2/runs/pretrain/P3-g9-s2.pt \[P3\] run_dir=$T/ws-netv2-741849/run/G9q pool_dir=$T/ws-netv2-741849/run/G9q/pool .* ps=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11 pp=$M/nv2/src$" "$(sed -n 2p <<< "$c")"
expect "S8 G9q eval anchor P1 (the job script's --anchor)" " anchor=$M/nv2/runs/pretrain/P1-g9h-dmg-s2.pt ps=" "$(sed -n 2p <<< "$c")"
expect "S8 G9q args: P3 anchor + frozen, 8 workers, cache 72" "--kl-anchor $M/nv2/runs/pretrain/P3-g9-s2.pt --kl-anchor-coef 0.1 .*--frozen $M/nv2/runs/pretrain/P3-g9-s2.pt .*--fastenv-workers 8 .*--snapshot-cache 72 --resume $M/nv2/runs/pretrain/P3-g9-s2.pt" "$(cat "$M/job-741849.out")"
expect "S8 G9q restart in place: its own save" "^call resume=$T/ws-netv2-741849/run/G9q/final_model.pt \[trained-from:P3\] run_dir=$T/ws-netv2-741849/run/G9q .* ps=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11 pp=$M/nv2/src$" "$(sed -n 3p <<< "$c")"
expect "S8 G9Q_STARTED + MODE g9q" "^g9q [0-9T:-]+$" "$(cat "$G/MODE") $(cat "$M/nv2/runs/g9q/G9Q_STARTED" 2>/dev/null)"
if grep -q -E "snapshot-cache-cpu|py-src|stacked-opp" "$M/job-741849.out"; then bad "S8 741849 got successor / a40 settings"; else ok "S8 741849 keeps \$ROOT/src, no cpu tier, no stacked-opp"; fi
if [ -z "$OLD" ]; then  # 810367 was spooled with the current job script
  echo "== S9: the successor 810367 (fresh job) after G9q started: the newest MIRRORED G9q file, not G9s (A16, then an a40)"
  for gpu in 15356 46068; do
    tree post; echo $gpu > "$M/gpu_mib"; wrapper; echo P3 > "$M/nv2/runs/pretrain/P3-g9-s2.pt"; echo g9q > "$G/MODE"; mkdir -p "$M/nv2/runs/g9q"; date > "$M/nv2/runs/g9q/G9Q_STARTED"
    K=$G/20261109-021316-741849; mkdir -p "$K/G9q/pool/frozen" "$K/G9q/checkpoints"
    echo G9q-mirror-ckpt > "$K/G9q/checkpoints/ckpt_00005000000.pt"; at '12 01:00' "$K/G9q/checkpoints/ckpt_00005000000.pt"
    echo G9q-mirror-final > "$K/G9q/final_model.pt"; at '12 01:44' "$K/G9q/final_model.pt"; at '12 01:46' "$K/G9s/final_model.pt"
    echo b > "$K/G9q/best_model.pt"; echo e > "$K/G9q/ema_model.pt"; echo P3 > "$K/G9q/pool/frozen/P3-g9-s2.pt"; echo bb > "$K/G9q/pool/frozen/best-00002500000.pt"
    echo s1 > "$K/G9q/pool/00002500000.pt"; echo s2 > "$K/G9q/pool/00005000000.pt"
    job g9 810367 10 "$([ $gpu = 15356 ] && echo a16 || echo a40)" ws-netv2-g9 restart; calls
    expect "S9 ($gpu MiB) resumes the mirrored G9q save with its pool / frozen / best / ema, wrapper, November sets" "^call resume=$K/G9q/final_model.pt \[G9q-mirror-final\] run_dir=$T/ws-netv2-810367/run/G9q pool_dir=$T/ws-netv2-810367/run/G9q/pool pool=2 frozen=2 best=y ema=y anchor=$M/nv2/runs/pretrain/P1-g9h-dmg-s2.pt ps=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11 pp=$M/nv2/src-24c633f:$M/nv2/src$" "$(sed -n 1p <<< "$c")"
    expect "S9 ($gpu MiB) its restart in place: its own save" "^call resume=$T/ws-netv2-810367/run/G9q/final_model.pt \[trained-from:G9q-mirror-final\]" "$(sed -n 2p <<< "$c")"
    if [ $gpu = 15356 ]; then
      expect "S9 A16 args: P3, 8 workers, cache 72 + cpu tier 200" "--kl-anchor $M/nv2/runs/pretrain/P3-g9-s2.pt .*--fastenv-workers 8 .*--snapshot-cache 72 --snapshot-cache-cpu 200 --resume" "$(cat "$M/job-810367.out")"
    else
      expect "S9 a40 args: cache 512 + stacked-opp + cpu tier 200" "--fastenv-workers 8 .*--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --resume" "$(cat "$M/job-810367.out")"
    fi
  done
fi
# S10-S11 (v9, 2026-11-10): the G9q <-> G9p GPU swap (two-phase, all-or-nothing; kit $SWAP_KIT, default scripts/netv2_swap).
# S10 (the running jobs 718701 / 741849, current and spooled scripts): each job's phase-1 INLINE_CMD (the kit file, plus
# a mock-only `touch restart` for one more restart in place) is queued during its first trainer call; both jobs then
# restart in place at about the same time: COMMIT -> the A30 relaunches as G9q from runs/g9q/handover-a30, the A16 as
# G9p from runs/r1/handover-g9p-a16, then each restarts in place from the new lineage's own save. S10b: only the A30
# answers (phase 2 times out -> ABORT: G9p continues from its own local save; the late A16 follows ABORT). S10c: the
# A16's copy fails midway -> FAIL-g9 + ABORT: both continue their own lineages. S11 (fresh jobs): successors 810366 /
# 810367 after the swap resume the swapped lineages' newest mirrored files; S11b: requeued jobs after an interrupted
# swap (both phase 1 DONE, no decision, MODE files untouched) decide COMMIT from the files alone.
KIT=${SWAP_KIT:-$(dirname "$(readlink -f "$0")")/netv2_swap}; P1=$M/nv2/runs/pretrain/P1-g9h-dmg-s2.pt; P3=$M/nv2/runs/pretrain/P3-g9-s2.pt; NOV=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11
swaptree() {  # the 2026-11-10 state: r1 MODE g9p (A30: G9p), g9 MODE g9q (A16: G9q, started from P3), both wrappers
  tree post; rm -rf "$M"/pending-inline* "$M"/gpu_mib* "$M"/break-* "$M"/calls.log "$M"/tmp/ws-netv2-*; wrapper; aswrapper
  echo 24576 > "$M/gpu_mib-718701"; echo 15356 > "$M/gpu_mib-741849"; echo 24576 > "$M/gpu_mib-810366"
  echo P3 > "$P3"; echo g9q > "$M/nv2/runs/g9/MODE"; mkdir -p "$M/nv2/runs/g9q" "$M/nv2/runs/swap-50M"; date > "$M/nv2/runs/g9q/G9Q_STARTED"
  K=$M/nv2/runs/g9/20261109-021316-741849; J=$M/nv2/runs/r1/20261108-220401-718701
  lineage "$K/G9q" G9q-mirror '11 10:00' 00002500000 00005000000; echo P3 > "$K/G9q/pool/frozen/P3-g9-s2.pt"; rm -f "$K/G9q/pool/frozen/P1-g9h-dmg-s2.pt"
}
queue() {  # queue JOBID RUN KITFILE
  { cat "$KIT/$3" 2>/dev/null || echo 'echo "mock: no swap kit file"; exit 1'; echo "touch \"\$C/runs/$2/restart\"  # mock only"; } > "$M/pending-inline-$1"
  echo "$M/nv2/runs/$2/INLINE_CMD" > "$M/pending-inline-$1.dest"
}
pair() {  # both running jobs restart in place at about the same time; job logs in $M/s-r1.txt / $M/s-g9.txt
  ( job r1 718701 30 a30 ws-netv2-r1 restart > "$M/s-r1.txt" 2>&1 ) & local p=$!
  /bin/sleep 2; job g9 741849 10 a16 ws-netv2-g9 restart > "$M/s-g9.txt" 2>&1; wait $p
  c1=$(grep "ws-netv2-718701/" "$M/calls.log" 2>/dev/null); c2=$(grep "ws-netv2-741849/" "$M/calls.log" 2>/dev/null)
  printf '  718701 calls:\n%s\n  741849 calls:\n%s\n' "$(sed 's/^/    /' <<< "$c1" | cut -c1-330)" "$(sed 's/^/    /' <<< "$c2" | cut -c1-330)"
  echo "  STATE:"; sed 's/^/    /' "$M/nv2/runs/swap-50M/STATE" 2>/dev/null | cut -c1-200; echo "  DECISION: $(cat "$M/nv2/runs/swap-50M/DECISION" 2>/dev/null)"
}
SW=$M/nv2/runs/swap-50M
echo "== S10: the swap in the running jobs: phase 1 in both, COMMIT, relaunch as the other lineage, restart in place"
swaptree; queue 718701 r1 inline-718701-g9p.sh; queue 741849 g9 inline-741849-g9q.sh; pair
expect "S10 718701 first: G9p (its mirrored file), src-4f18108" "^call resume=$J/G9p/final_model.pt \[G9p-postswap-final\] run_dir=$T/ws-netv2-718701/run/G9p .* ps=- pp=$M/nv2/src-4f18108:$M/nv2/src$" "$(sed -n 1p <<< "$c1")"
expect "S10 718701 relaunch = G9q's A30 first start from the A16's handover (pool / frozen / best / ema, P1 eval anchor, November sets)" "^call resume=$M/nv2/runs/g9q/handover-a30/final_model.pt \[trained-from:G9q-mirror-final\] run_dir=$T/ws-netv2-718701/run/G9q pool_dir=$T/ws-netv2-718701/run/G9q/pool pool=2 frozen=2 best=y ema=y anchor=$P1 ps=$NOV pp=$M/nv2/src-4f18108:$M/nv2/src$" "$(sed -n 2p <<< "$c1")"
expect "S10 718701 restart in place: G9q's own save" "^call resume=$T/ws-netv2-718701/run/G9q/final_model.pt \[trained-from:trained-from:G9q-mirror-final\] run_dir=$T/ws-netv2-718701/run/G9q .* ps=$NOV pp=$M/nv2/src-4f18108:" "$(sed -n 3p <<< "$c1")"
expect "S10 718701 G9q args: P3, 28 workers, actor stream, cache 192 + stacked-opp, no cpu tier" "with TRAIN_ARGS='.*--kl-anchor $P3 .*--frozen $P3 .*--fastenv-workers 28 --fastenv-battles 48 --actor-stream --decoupled .*--run-dir $T/ws-netv2-718701/run/G9q .*--snapshot-cache 192 --stacked-opp --resume $M/nv2/runs/g9q/handover-a30/final_model.pt'" "$(cat "$M/job-718701.out")"
expect "S10 741849 first: G9q (its mirrored file), \$ROOT/src, November sets" "^call resume=$K/G9q/final_model.pt \[G9q-mirror-final\] run_dir=$T/ws-netv2-741849/run/G9q .* ps=$NOV pp=$M/nv2/src$" "$(sed -n 1p <<< "$c2")"
expect "S10 741849 relaunch = G9p's A16 first start from the A30's handover (October sets, 24c633f wrapper)" "^call resume=$M/nv2/runs/r1/handover-g9p-a16/final_model.pt \[trained-from:G9p-postswap-final\] run_dir=$T/ws-netv2-741849/run/G9p pool_dir=$T/ws-netv2-741849/run/G9p/pool pool=3 frozen=2 best=y ema=y anchor=$P1 ps=- pp=$M/nv2/src-24c633f:$M/nv2/src$" "$(sed -n 2p <<< "$c2")"
expect "S10 741849 restart in place: G9p's own save" "^call resume=$T/ws-netv2-741849/run/G9p/final_model.pt \[trained-from:trained-from:G9p-postswap-final\] run_dir=$T/ws-netv2-741849/run/G9p .* ps=- pp=$M/nv2/src-24c633f:" "$(sed -n 3p <<< "$c2")"
expect "S10 741849 G9p args: P1, 8 workers, --no-overlap, cache 72 + cpu tier 200" "with TRAIN_ARGS='.*--kl-anchor $P1 .*--frozen $P1 .*--fastenv-workers 8 --fastenv-battles 48 --no-overlap .*--run-dir $T/ws-netv2-741849/run/G9p .*--snapshot-cache 72 --snapshot-cache-cpu 200 --resume $M/nv2/runs/r1/handover-g9p-a16/final_model.pt'" "$(cat "$M/job-741849.out")"
expect "S10 COMMIT, MODE files flipped, both applied" "^COMMIT by (r1 718701|g9 741849) .* g9q g9p2 yy$" "$(cat "$SW/DECISION" 2>/dev/null) $(cat "$M/nv2/runs/r1/MODE") $(cat "$M/nv2/runs/g9/MODE") $([ -f "$M/nv2/runs/r1/SWAP_APPLIED" ] && echo y)$([ -f "$M/nv2/runs/g9/SWAP_APPLIED" ] && echo y)"
expect "S10 STATE: both phase 1 DONE, both COMMIT applied" "r1 718701 phase1 DONE.*g9 741849 phase1 DONE.*applied COMMIT.*applied COMMIT|g9 741849 phase1 DONE.*r1 718701 phase1 DONE.*applied COMMIT.*applied COMMIT" "$(tr '\n' ' ' < "$SW/STATE" 2>/dev/null)"
echo "== S10b: only the A30 answers: phase 2 times out -> ABORT; G9p continues from its own save; the late A16 follows ABORT"
swaptree; queue 718701 r1 inline-718701-g9p.sh
MOCK_SWAP_ITERS=3 job r1 718701 30 a30 ws-netv2-r1 restart > "$M/s-r1.txt"; c1=$(grep "ws-netv2-718701/" "$M/calls.log"); sed 's/^/    /' <<< "$c1" | cut -c1-250
expect "S10b 718701 relaunch: G9p from its own local save (not G9q, not a stale dir)" "^call resume=$T/ws-netv2-718701/run/G9p/final_model.pt \[trained-from:G9p-postswap-final\] run_dir=$T/ws-netv2-718701/run/G9p .* ps=- pp=$M/nv2/src-4f18108:" "$(sed -n 2p <<< "$c1")"
expect "S10b 718701 next restart: still G9p, own save" "^call resume=$T/ws-netv2-718701/run/G9p/final_model.pt \[trained-from:trained-from:G9p-postswap-final\] run_dir=$T/ws-netv2-718701/run/G9p " "$(sed -n 3p <<< "$c1")"
expect "S10b ABORT by the A30, MODE g9p, handover published (DONE)" "^ABORT by r1 718701 .* g9p y$" "$(cat "$SW/DECISION" 2>/dev/null) $(cat "$M/nv2/runs/r1/MODE") $([ -f "$M/nv2/runs/r1/handover-g9p-a16/DONE" ] && echo y)"
queue 741849 g9 inline-741849-g9q.sh; job g9 741849 10 a16 ws-netv2-g9 restart > "$M/s-g9.txt"; c2=$(grep "ws-netv2-741849/" "$M/calls.log")
expect "S10b late 741849: phase 1 refused, G9q continues from its own save" "^call resume=$T/ws-netv2-741849/run/G9q/final_model.pt \[trained-from:G9q-mirror-final\] run_dir=$T/ws-netv2-741849/run/G9q .* ps=$NOV pp=$M/nv2/src$" "$(sed -n 2p <<< "$c2")"
expect "S10b MODE g9 g9q, no G9q handover" "^g9q n$" "$(cat "$M/nv2/runs/g9/MODE") $([ -e "$M/nv2/runs/g9q/handover-a30" ] && echo y || echo n)"
echo "== S10c: the A16's copy fails midway (an unreadable pool net): FAIL-g9 -> ABORT; both continue their own lineages"
swaptree; queue 718701 r1 inline-718701-g9p.sh; queue 741849 g9 inline-741849-g9q.sh; echo "$T/ws-netv2-741849/run/G9q/pool/00005000000.pt" > "$M/break-741849"
pair; chmod -R u+rw "$T" "$M/nv2/runs/g9q" 2>/dev/null
expect "S10c 718701 relaunch: G9p, own save" "^call resume=$T/ws-netv2-718701/run/G9p/final_model.pt \[trained-from:G9p-postswap-final\] run_dir=$T/ws-netv2-718701/run/G9p " "$(sed -n 2p <<< "$c1")"
expect "S10c 741849 relaunch: G9q, own save" "^call resume=$T/ws-netv2-741849/run/G9q/final_model.pt \[trained-from:G9q-mirror-final\] run_dir=$T/ws-netv2-741849/run/G9q .* ps=$NOV pp=$M/nv2/src$" "$(sed -n 2p <<< "$c2")"
expect "S10c ABORT, FAIL-g9, no G9q handover, MODE files unchanged" "^ABORT by .* y n g9p g9q$" "$(cat "$SW/DECISION" 2>/dev/null) $([ -f "$SW/FAIL-g9" ] && echo y) $([ -e "$M/nv2/runs/g9q/handover-a30" ] && echo y || echo n) $(cat "$M/nv2/runs/r1/MODE") $(cat "$M/nv2/runs/g9/MODE")"
echo "== S11b: requeued jobs after an interrupted swap (both phase 1 DONE, then both jobs died; no decision, MODE untouched)"
swaptree; queue 718701 r1 inline-718701-g9p.sh; queue 741849 g9 inline-741849-g9q.sh; pair > /dev/null  # build both handovers
rm -rf "$SW/DECISION" "$SW/STATE" "$M/nv2/runs/r1/SWAP_APPLIED" "$M/nv2/runs/g9/SWAP_APPLIED" "$M/nv2/runs/r1/G9Q_A30_STARTED" "$M/nv2/runs/g9/G9P_A16_STARTED" "$M/calls.log"
echo g9p > "$M/nv2/runs/r1/MODE"; echo g9q > "$M/nv2/runs/g9/MODE"; rm -rf "$J"/G9q "$K"/G9p "$M"/nv2/runs/r1/2026111[0-9]-* "$M"/nv2/runs/g9/2026111[0-9]-* "$T"/ws-netv2-*
job r1 718701 30 a30 ws-netv2-r1 > /dev/null; c1=$(grep "ws-netv2-718701/" "$M/calls.log")
expect "S11b requeued 718701 decides COMMIT and first-starts G9q from the handover" "^call resume=$M/nv2/runs/g9q/handover-a30/final_model.pt \[trained-from:G9q-mirror-final\] run_dir=$T/ws-netv2-718701/run/G9q .*pool=2 frozen=2 best=y ema=y .*ps=$NOV pp=$M/nv2/src-4f18108:" "$(sed -n 1p <<< "$c1")"
job g9 741849 10 a16 ws-netv2-g9 > /dev/null; c2=$(grep "ws-netv2-741849/" "$M/calls.log")
expect "S11b requeued 741849 applies that COMMIT and first-starts G9p from the handover" "^call resume=$M/nv2/runs/r1/handover-g9p-a16/final_model.pt \[trained-from:G9p-postswap-final\] run_dir=$T/ws-netv2-741849/run/G9p .*pool=3 frozen=2 best=y ema=y .*ps=- pp=$M/nv2/src-24c633f:" "$(sed -n 1p <<< "$c2")"
expect "S11b COMMIT by the A30 side, MODE g9q / g9p2" "^COMMIT by r1 718701 .* g9q g9p2$" "$(cat "$SW/DECISION" 2>/dev/null) $(cat "$M/nv2/runs/r1/MODE") $(cat "$M/nv2/runs/g9/MODE")"
if [ -z "$OLD" ]; then  # 810366 / 810367 were spooled with the current job script
  echo "== S11: successors 810366 (a30) and 810367 (a16, a40) after the swap: the swapped lineages' newest mirrored files"
  for g in a30 a16 a40; do
    swaptree; R1=$M/nv2/runs/r1; G=$M/nv2/runs/g9
    echo "COMMIT by g9 741849 2026-11-10T16:00:00" > "$SW/DECISION"; echo g9q > "$R1/MODE"; echo g9p2 > "$G/MODE"; touch "$R1/SWAP_APPLIED" "$G/SWAP_APPLIED" "$R1/G9Q_A30_STARTED" "$G/G9P_A16_STARTED"
    for h in "$M/nv2/runs/g9q/handover-a30" "$R1/handover-g9p-a16"; do mkdir -p "$h/pool"; echo handover > "$h/final_model.pt"; echo d > "$h/DONE"; done
    lineage "$J/G9q" G9q-a30 '11 21:20' 00002500000 00005000000 00010000000; lineage "$K/G9p" G9p-a16 '12 01:40' 00002509811 00025009642 00150000000 00160000000
    at '10 16:00' "$K/G9q/final_model.pt"  # the A16's G9q segment ends at the swap; the A30's G9p segment (11 21:47 in tree post -> swap time)
    at '10 16:00' "$J/G9p/final_model.pt"
    if [ $g = a30 ]; then
      echo 24576 > "$M/gpu_mib"; job r1 810366 30 a30 ws-netv2-r1 restart > /dev/null; calls
      expect "S11 810366: the A30's mirrored G9q (not the A16's pre-swap G9q), pool / frozen / best / ema, src-4f18108, November sets" "^call resume=$J/G9q/final_model.pt \[G9q-a30-final\] run_dir=$T/ws-netv2-810366/run/G9q pool_dir=$T/ws-netv2-810366/run/G9q/pool pool=3 frozen=2 best=y ema=y anchor=$P1 ps=$NOV pp=$M/nv2/src-4f18108:$M/nv2/src$" "$(sed -n 1p <<< "$c")"
      expect "S11 810366 restart in place: its own save" "^call resume=$T/ws-netv2-810366/run/G9q/final_model.pt \[trained-from:G9q-a30-final\]" "$(sed -n 2p <<< "$c")"
      expect "S11 810366 args: P3, actor stream, cache 192 + stacked-opp + cpu tier 400" "--kl-anchor $P3 .*--fastenv-workers 28 --fastenv-battles 48 --actor-stream --decoupled .*--snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400 --resume" "$(cat "$M/job-810366.out")"
    else
      echo $([ $g = a16 ] && echo 15356 || echo 46068) > "$M/gpu_mib"; job g9 810367 10 $g ws-netv2-g9 restart > /dev/null; calls
      expect "S11 810367 ($g): the A16's mirrored G9p (not the A30's pre-swap G9p), October sets, 24c633f wrapper" "^call resume=$K/G9p/final_model.pt \[G9p-a16-final\] run_dir=$T/ws-netv2-810367/run/G9p pool_dir=$T/ws-netv2-810367/run/G9p/pool pool=4 frozen=2 best=y ema=y anchor=$P1 ps=- pp=$M/nv2/src-24c633f:$M/nv2/src$" "$(sed -n 1p <<< "$c")"
      expect "S11 810367 ($g) restart in place: its own save" "^call resume=$T/ws-netv2-810367/run/G9p/final_model.pt \[trained-from:G9p-a16-final\]" "$(sed -n 2p <<< "$c")"
      x="--snapshot-cache 72 --snapshot-cache-cpu 200"; [ $g = a40 ] && x="--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200"
      expect "S11 810367 ($g) args: P1, 8 workers, --no-overlap, $x" "--kl-anchor $P1 .*--fastenv-workers 8 --fastenv-battles 48 --no-overlap .*$x --resume" "$(cat "$M/job-810367.out")"
    fi
  done
fi
rm -f "$M"/pending-inline* "$M"/gpu_mib-* "$M"/break-*
echo "== S3: the swap back (MODE files as before 12:40): fresh r1 and g9 jobs"
tree pre; job r1 900003 30 a30 ws-netv2-r1; calls
expect "S3 r1 resumes the newest G9s file" "^call resume=.*/20261108-220401-718701/G9s/final_model.pt \[G9s-preswap-final\] run_dir=$T/ws-netv2-900003/run/G9s .*pool=3 frozen=2 best=y ema=y" "$(sed -n 1p <<< "$c")"
tree pre; job g9 900004 10 a16 ws-netv2-g9; calls
expect "S3 g9 resumes the newest G9 (G9p) file" "^call resume=.*/20261109-021316-741849/G9/final_model.pt \[G9p-preswap-final\] run_dir=$T/ws-netv2-900004/run/G9 .*pool=2 frozen=2 best=y ema=y" "$(sed -n 1p <<< "$c")"
echo "== S4: deadline watch: the job ends in 15 min + 3 s; the trainer is stopped by SIGINT, the job ends"
tree post; MOCK_END=$(( $(date +%s) + 903 )) MOCK_WAIT=1 job r1 900005 30 a30 ws-netv2-r1 restart; calls
expect "S4 watcher armed, then SIGINT sent" "deadline watch armed .*deadline watch: SIGINT" "$(grep -h "deadline watch" "$M"/nv2/runs/r1/*/train.log 2>/dev/null | tr '\n' ' ')"
expect "S4 the trainer got SIGINT" "^trainer: got SIGINT" "$(sed -n 2p <<< "$c")"
expect "S4 the restart flag is removed and the job does not relaunch" "^$" "$(sed -n 3p <<< "$c"; ls "$M/nv2/runs/r1/restart" 2>/dev/null)"
echo "passed $pass, failed $fail"
