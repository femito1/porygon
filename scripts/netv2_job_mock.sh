#!/bin/bash
# Laptop dry run of a FRESH job of scripts/slurm_train_v2.sbatch (a requeue, a chained or an early-submitted successor)
# against a copy of the cluster's netv2.env: a fake ROOT with the run state after the 2026-11-09 12:40 swap (G9p on the
# A30 = r1 MODE g9p, lineage dir G9p; G9s on the A16 = g9 MODE g9s; the pre-swap lineage dirs, the 12:40 handovers, the
# stale 04:02 top-level files, chain_left / marker files), stub python (trainer + import checks), sbatch, squeue,
# scontrol, srun, pkill, sleep and nproc first on PATH; the job script runs unmodified. The stub trainer logs the args that
# win (argparse: last wins), the pool it sees and what it resumed from, writes final_model.pt + a train.log line where the
# real one would and exits (MOCK_WAIT=1: waits for SIGINT first); with runs/RUN/restart armed the restart loop runs once.
#   bash scripts/netv2_job_mock.sh SBATCH_SCRIPT ENVFILE [old]   (old = a spooled pre-fix job script: no chain-flag case)
# Expectations are the CORRECT behaviour: FAIL lines are gaps. S12-S13 (v10): G9r and the park INLINE_CMD. S14 (v12
# draft): G9t (MODE g9t, runs/g9t/RECIPE). S16 (v14): the G9r park -> G9t on the A16, the parked-HOLD guard.
set -u
SB=$(readlink -f "$1"); ENVF=$(readlink -f "$2"); OLD=${3:-}
M=${MOCK_ROOT:-/tmp/netv2-job-mock}
WDB=$(dirname "$(readlink -f "$0")")/netv2_wd_stubs
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
open(f"{M}/argv.log", "a").write(os.environ.get("SLURM_JOB_ID", "") + " " + " ".join(a) + "\n")  # S17: the full argv
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
ss, q = "99,000,000", f"{M}/save-steps-{jid}"  # S19 (v21): a queue of save steps per job (one line per trainer call)
if os.path.exists(q):
    ls = open(q).read().split()
    if ls: ss = ls[0]; open(q, "w").write("\n".join(ls[1:]))
with open(rundir + "/train.log", "a") as f:  # the real trainer's start / update / save lines (the park script reads them)
    f.write("2026-11-11 21:30:00 v2 trainer: mock, resume=" + resume + "\n2026-11-11 21:40:00 upd 9 steps " + ss + " (+43000) sps 2200\n"
            "2026-11-11 21:40:10 saved final_model.pt at " + ss + " steps\n")
EOF
  # sbatch / squeue / scontrol: as before (log only) unless $MOCK/squeue.txt exists (the v12 watchdog scenarios S15):
  # then the scripts/netv2_wd_stubs versions (a queue in squeue.txt; sbatch prints the cluster's banner and registers
  # its job PENDING); df: the stub (100 GB available unless $MOCK/squeue.txt.df says otherwise)
  printf '#!/bin/bash\necho "sbatch $*" >> "$MOCK/sbatch.log"\nif [ -f "$MOCK/squeue.txt" ]; then WDQ=$MOCK/squeue.txt exec %s/sbatch "$@"; fi\necho "Submitted batch job 999999"\n' "$WDB" > "$M/bin/sbatch"
  for c in squeue scontrol; do printf '#!/bin/bash\necho "%s $*" >> "$MOCK/other.log"\nif [ -f "$MOCK/squeue.txt" ]; then WDQ=$MOCK/squeue.txt exec %s/%s "$@"; fi\nexit 0\n' "$c" "$WDB" "$c" > "$M/bin/$c"; done
  for c in srun pkill; do printf '#!/bin/bash\necho "%s $*" >> "$MOCK/other.log"\nexit 0\n' "$c" > "$M/bin/$c"; done
  printf '#!/bin/bash\nWDQ=$MOCK/squeue.txt exec %s/df "$@"\n' "$WDB" > "$M/bin/df"
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
      NODE_BIN=/nonexistent NV2_PARK_PY="$M/bin/parkpy" NV2_SWAP_SLEEP=1 NV2_SWAP_ITERS="${MOCK_SWAP_ITERS:-240}" ${MOCK_END:+SLURM_JOB_END_TIME=$MOCK_END NV2_DEADLINE_POLL_S=1 NV2_DEADLINE_PAT=bin/python.-u.-m.porygon.v2.train} MOCK_WAIT="${MOCK_WAIT:-}" ${MOCK_EXTRA_ENV:-} ${MOCK_PARK_ROUNDS:+NV2_PARK_HOLD_ROUNDS=$MOCK_PARK_ROUNDS} ${MOCK_HOLD_ROUNDS:+NV2_G9T_HOLD_ROUNDS=$MOCK_HOLD_ROUNDS NV2_HOLD_ITERS=0} NV2_TMPWATCH_POLL_S="${MOCK_POLL:-0}" NV2_WD_RETRY_S=0 ${MOCK_POLL:+NV2_DEADLINE_PAT=bin/python.-u.-m.porygon.v2.train} \
      setsid -w bash -c 'echo $$ > "$MOCK/sid-$SLURM_JOB_ID"; exec bash "$MOCK/nv2/scripts/slurm_train_v2.sbatch" "$1"' _ "$run" > "$M/job-$id.out" 2>&1 )
  local rc=$? sid; sid=$(cat "$M/sid-$id" 2>/dev/null); echo "exit $rc" > "$M/rc-$id"
  if [ -n "$sid" ]; then /bin/sleep 1; pkill -s "$sid" 2>/dev/null; fi  # leftovers of this mock job only (its own session)
  echo "  exit $rc ; job log (trimmed):"; grep -v -E "pre-read|find: |^$" "$M/job-$id.out" | cut -c1-200 | sed 's/^/    | /' | head -n 12
}
job_rc() { cat "$M/rc-$1" 2>/dev/null; }
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
  expect "S5c args: cache 40 + stacked-opp, then the strength watch" "fastenv-workers 8 .*--snapshot-cache 40 --stacked-opp --snapshot-cache-cpu 200 --guard-pg-mean 20 --guard-anchor-drop 0.0999" "$(cat "$M/job-829660.out")"
  if grep -q "snapshot-cache-cpu 200" "$M/job-829660.out"; then ok "S5c R1b-F has the cpu tier (v11)"; else bad "S5c R1b-F without the cpu tier (v11 adds it)"; fi
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
# S12-S13 (v10, 2026-11-11; G9r, approved .orchestra/netv2/G9R-PROPOSAL.md): the park INLINE_CMD ($KIT/../netv2_g9r/
# inline-a16-park-g9r.sh, queued during the A16 trainer's run as the level-1 agent installs it, plus a
# mock-only `touch restart`) at a restart in place of 741849 (current and spooled scripts): S12 MODE g9q (after a swap
# ABORT) parks G9q in runs/g9q-parked, S12b MODE g9p2 (after a committed + applied swap) parks G9p in
# runs/g9p-parked-a16; both relaunch as G9r's first start from P3 (G9r lineage dirs, November sets, the
# py-src-ff32b71 wrapper, --lr-schedule power --lr-horizon 300000000, entropy floor 0.35), then restart in place from
# G9r's own save. S12c: an injected copy failure -> MODE untouched, the same lineage continues. S13 (fresh jobs: 810367
# on an a16 / a40): G9r's newest MIRRORED file (never the parked dirs or a newer G9q mirror); S13b: a swap COMMIT
# applied late by the fresh job leaves MODE g9r.
G9R_SHA=ff32b71; WT=$(dirname "$(dirname "$(readlink -f "$0")")")
printf '#!/bin/bash\necho "counters: {'"'"'steps'"'"': 99000000, '"'"'updates'"'"': 9}"; echo "keys: ['"'"'counters'"'"', '"'"'optimizer'"'"']"\n' > "$M/bin/parkpy"; chmod +x "$M/bin/parkpy"
g9rtree() {  # swaptree + the G9r wrapper (src-ff32b71 first, then the mock python), its src (train.py), PREREG
  swaptree; mkdir -p "$M/nv2/runs/g9r" "$M/nv2/src-$G9R_SHA/porygon/v2"; echo prereg > "$M/nv2/runs/g9r/PREREG.txt"
  git -C "$WT" show "$G9R_SHA:src/porygon/v2/train.py" > "$M/nv2/src-$G9R_SHA/porygon/v2/train.py" 2>/dev/null || cp "$WT/src/porygon/v2/train.py" "$M/nv2/src-$G9R_SHA/porygon/v2/train.py"
  printf '#!/bin/bash\nexport PYTHONPATH=%s/nv2/src-%s${PYTHONPATH:+:$PYTHONPATH}\nexec "$MOCK/bin/python" "$@"\n' "$M" "$G9R_SHA" > "$M/nv2/bin/py-src-$G9R_SHA"; chmod +x "$M/nv2/bin/py-src-$G9R_SHA"
}
g9rargs() {  # g9rargs JOB OPP RESUME: the G9r TRAIN_ARGS pattern
  echo "--lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.35 --guard-ptop-max 0.6 --kl-anchor $P3 --kl-anchor-coef 0.1 --kl-anchor-decay-steps 50000000 --critic-warmup-steps 1000000 --frozen $P3 --p-frozen 0.67 --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir $T/ws-netv2-$1/run/G9r --pool-dir $T/ws-netv2-$1/run/G9r/pool $2 --lr-schedule power --lr-horizon 300000000 --resume $3"
}
PP9R="pp=$M/nv2/src-$G9R_SHA:$M/nv2/src$"
echo "== S12: 741849 (MODE g9q, swap ABORTed) parks G9q through the INLINE_CMD and relaunches as G9r (first start from P3), then restarts G9r in place"
g9rtree; echo "ABORT by r1 718701 2026-11-11T16:00:00" > "$SW/DECISION"; queue 741849 g9 ../netv2_g9r/inline-a16-park-g9r.sh
job g9 741849 10 a16 ws-netv2-g9 restart > "$M/s-g9.txt"; c2=$(grep "ws-netv2-741849/" "$M/calls.log"); sed 's/^/    /' <<< "$c2" | cut -c1-260
expect "S12 first: G9q (its mirrored file), \$ROOT/src, November sets" "^call resume=$K/G9q/final_model.pt \[G9q-mirror-final\] run_dir=$T/ws-netv2-741849/run/G9q .* ps=$NOV pp=$M/nv2/src$" "$(sed -n 1p <<< "$c2")"
expect "S12 relaunch = G9r's first start from P3: G9r dirs, P1 eval anchor, November sets, the ff32b71 wrapper" "^call resume=$P3 \[P3\] run_dir=$T/ws-netv2-741849/run/G9r pool_dir=$T/ws-netv2-741849/run/G9r/pool .*anchor=$P1 ps=$NOV $PP9R" "$(sed -n 2p <<< "$c2")"
expect "S12 G9r restart in place: its own save" "^call resume=$T/ws-netv2-741849/run/G9r/final_model.pt \[trained-from:P3\] run_dir=$T/ws-netv2-741849/run/G9r .* ps=$NOV $PP9R" "$(sed -n 3p <<< "$c2")"
expect "S12 G9r args (first start)" "with TRAIN_ARGS='--no-bf16 --n-epochs 3 --batch-size 4096 $(g9rargs 741849 "--snapshot-cache 72 --snapshot-cache-cpu 200" "$P3")'" "$(cat "$M/job-741849.out")"
expect "S12 G9r args (restart in place)" "with TRAIN_ARGS='--no-bf16 --n-epochs 3 --batch-size 4096 $(g9rargs 741849 "--snapshot-cache 72 --snapshot-cache-cpu 200" "$T/ws-netv2-741849/run/G9r/final_model.pt")'" "$(cat "$M/job-741849.out")"
expect "S12 G9q parked with its last save (+ pool, frozen, best, ema, checksums), MODE g9r, G9R_STARTED" "^trained-from:G9q-mirror-final 2 2 G9q-mirror-best y g9r y$" "$(cat "$M/nv2/runs/g9q-parked/final_model.pt" 2>/dev/null) $(ls "$M/nv2/runs/g9q-parked/pool" 2>/dev/null | grep -c pt$) $(ls "$M/nv2/runs/g9q-parked/pool/frozen" 2>/dev/null | grep -c pt$) $(cat "$M/nv2/runs/g9q-parked/best_model.pt" 2>/dev/null) $(cd "$M/nv2/runs/g9q-parked" 2>/dev/null && md5sum -c --quiet MD5SUMS > /dev/null 2>&1 && echo y) $(cat "$M/nv2/runs/g9/MODE") $([ -f "$M/nv2/runs/g9r/G9R_STARTED" ] && echo y)"
expect "S12 PARKED.txt: counters, source, MODE" "^G9q PARKED .* in job 741849 .*MODE g9q -> g9r.*counters: \{'steps': 99000000" "$(tr '\n' ' ' < "$M/nv2/runs/g9q-parked/PARKED.txt" 2>/dev/null)"
echo "== S12b: 741849 (MODE g9p2 after a committed swap: G9p on this A16) parks G9p in runs/g9p-parked-a16 and relaunches as G9r"
g9rtree; G=$M/nv2/runs/g9; echo "COMMIT by g9 741849 2026-11-11T16:00:00" > "$SW/DECISION"; echo g9p2 > "$G/MODE"; touch "$G/SWAP_APPLIED" "$G/G9P_A16_STARTED" "$M/nv2/runs/r1/SWAP_APPLIED"
lineage "$K/G9p" G9p-a16 '12 01:40' 00250000000 00260000000 00270000000; queue 741849 g9 ../netv2_g9r/inline-a16-park-g9r.sh
job g9 741849 10 a16 ws-netv2-g9 restart > "$M/s-g9.txt"; c2=$(grep "ws-netv2-741849/" "$M/calls.log"); sed 's/^/    /' <<< "$c2" | cut -c1-260
expect "S12b first: G9p (its mirrored A16 file), October sets, the 24c633f wrapper" "^call resume=$K/G9p/final_model.pt \[G9p-a16-final\] run_dir=$T/ws-netv2-741849/run/G9p .* ps=- pp=$M/nv2/src-24c633f:$M/nv2/src$" "$(sed -n 1p <<< "$c2")"
expect "S12b relaunch = G9r's first start from P3 (November sets, ff32b71 wrapper)" "^call resume=$P3 \[P3\] run_dir=$T/ws-netv2-741849/run/G9r .*anchor=$P1 ps=$NOV $PP9R" "$(sed -n 2p <<< "$c2")"
expect "S12b G9r restart in place: its own save" "^call resume=$T/ws-netv2-741849/run/G9r/final_model.pt \[trained-from:P3\] .* ps=$NOV $PP9R" "$(sed -n 3p <<< "$c2")"
expect "S12b G9p parked (3 pool + 2 frozen), MODE g9r, no G9q park" "^trained-from:G9p-a16-final 3 2 g9r n$" "$(cat "$M/nv2/runs/g9p-parked-a16/final_model.pt" 2>/dev/null) $(ls "$M/nv2/runs/g9p-parked-a16/pool" 2>/dev/null | grep -c pt$) $(ls "$M/nv2/runs/g9p-parked-a16/pool/frozen" 2>/dev/null | grep -c pt$) $(cat "$G/MODE") $([ -e "$M/nv2/runs/g9q-parked" ] && echo y || echo n)"
echo "== S12c: the park's copy fails midway (an unreadable G9q pool net): MODE untouched, G9q continues from its own save"
g9rtree; echo "ABORT by r1 718701 2026-11-11T16:00:00" > "$SW/DECISION"; queue 741849 g9 ../netv2_g9r/inline-a16-park-g9r.sh; echo "$T/ws-netv2-741849/run/G9q/pool/00005000000.pt" > "$M/break-741849"
job g9 741849 10 a16 ws-netv2-g9 restart > "$M/s-g9.txt"; chmod -R u+rw "$T" 2>/dev/null; c2=$(grep "ws-netv2-741849/" "$M/calls.log"); sed 's/^/    /' <<< "$c2" | cut -c1-260
expect "S12c relaunch: G9q from its own save (no G9r)" "^call resume=$T/ws-netv2-741849/run/G9q/final_model.pt \[trained-from:G9q-mirror-final\] run_dir=$T/ws-netv2-741849/run/G9q .* ps=$NOV pp=$M/nv2/src$" "$(sed -n 2p <<< "$c2")"
expect "S12c MODE g9q, no park dir, no G9R_STARTED, FAIL logged" "^g9q n n y$" "$(cat "$M/nv2/runs/g9/MODE") $([ -e "$M/nv2/runs/g9q-parked" ] && echo y || echo n) $([ -e "$M/nv2/runs/g9r/G9R_STARTED" ] && echo y || echo n) $(grep -q 'park FAIL (MODE untouched: g9q): copy' "$M/nv2/runs/g9r/park.log" 2>/dev/null && echo y)"
if grep -q -E "lr-schedule|py-src-$G9R_SHA" "$M/job-741849.out"; then bad "S12c G9r args after a failed park"; else ok "S12c no G9r args"; fi
if [ -z "$OLD" ]; then  # 810367 was spooled with the current job script
  echo "== S13: the successor 810367 (fresh job) after G9r started: the newest MIRRORED G9r file (a16, a40), never the parked dirs"
  for g in a16 a40; do
    g9rtree; G=$M/nv2/runs/g9; echo g9r > "$G/MODE"; date > "$M/nv2/runs/g9r/G9R_STARTED"; echo "ABORT by r1 718701 x" > "$SW/DECISION"
    lineage "$K/G9r" G9r-mirror '12 01:44' 00001000000 00002000000; echo P3 > "$K/G9r/pool/frozen/P3-g9-s2.pt"; rm -f "$K/G9r/pool/frozen/P1-g9h-dmg-s2.pt"
    for d in "$M/nv2/runs/g9q-parked" "$M/nv2/runs/g9p-parked-a16"; do mkdir -p "$d/pool"; echo parked > "$d/final_model.pt"; at '12 01:50' "$d/final_model.pt"; done; at '12 01:50' "$K/G9q/final_model.pt"
    echo $([ $g = a16 ] && echo 15356 || echo 46068) > "$M/gpu_mib"; job g9 810367 10 $g ws-netv2-g9 restart > /dev/null; calls
    expect "S13 810367 ($g): the mirrored G9r save with its pool / frozen / best / ema, wrapper, November sets" "^call resume=$K/G9r/final_model.pt \[G9r-mirror-final\] run_dir=$T/ws-netv2-810367/run/G9r pool_dir=$T/ws-netv2-810367/run/G9r/pool pool=2 frozen=2 best=y ema=y anchor=$P1 ps=$NOV $PP9R" "$(sed -n 1p <<< "$c")"
    expect "S13 810367 ($g) restart in place: its own save" "^call resume=$T/ws-netv2-810367/run/G9r/final_model.pt \[trained-from:G9r-mirror-final\] .*$PP9R" "$(sed -n 2p <<< "$c")"
    x="--snapshot-cache 72 --snapshot-cache-cpu 200"; [ $g = a40 ] && x="--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200"
    expect "S13 810367 ($g) args: $x" "with TRAIN_ARGS='--no-bf16 --n-epochs 3 --batch-size 4096 $(g9rargs 810367 "$x" "$T/ws-netv2-810367/run/G9r/final_model.pt")'" "$(cat "$M/job-810367.out")"
  done
  echo "== S13b: a swap COMMIT the A16 never applied (DECISION COMMIT, no runs/g9/SWAP_APPLIED) reaches the fresh 810367 after MODE g9r: G9r continues"
  g9rtree; G=$M/nv2/runs/g9; echo g9r > "$G/MODE"; date > "$M/nv2/runs/g9r/G9R_STARTED"; echo "COMMIT by r1 718701 2026-11-11T16:00:00" > "$SW/DECISION"
  for h in "$M/nv2/runs/g9q/handover-a30" "$M/nv2/runs/r1/handover-g9p-a16"; do mkdir -p "$h/pool"; echo handover > "$h/final_model.pt"; echo d > "$h/DONE"; done
  lineage "$K/G9r" G9r-mirror '12 01:44' 00001000000; echo 15356 > "$M/gpu_mib"; job g9 810367 10 a16 ws-netv2-g9 restart > /dev/null; calls
  expect "S13b 810367 resumes G9r (not G9p from the handover), MODE g9r, the COMMIT recorded as applied" "^call resume=$K/G9r/final_model.pt \[G9r-mirror-final\] run_dir=$T/ws-netv2-810367/run/G9r .*$PP9R" "$(sed -n 1p <<< "$c")"
  expect "S13b MODE g9r, SWAP_APPLIED, no G9P_A16_STARTED" "^g9r y n$" "$(cat "$G/MODE") $([ -f "$G/SWAP_APPLIED" ] && echo y) $([ -e "$G/G9P_A16_STARTED" ] && echo y || echo n)"
  # S14 (g9t DRAFT, not deployed; skipped with MOCK_SKIP_G9T=1; NEXT-RECIPE-PROPOSAL 3 / 5): MODE g9t = C's recipe + --gae-lambda (runs/g9t/RECIPE
  # "C LAMBDA", recorded in runs/g9t/G9T_STARTED at the first start). The tree is the 2026-11-12 state (swap COMMITted
  # and applied, G9r on the A16, 4f18108 disabled). S14: the A16 job 810367 trains G9r; an INLINE_CMD queued during
  # its trainer's run flips MODE g9t (+ a mock-only `touch restart`) -> the restart in place is G9t's first start from
  # P3 (G9t dirs, P1 eval anchor, November sets, the ff32b71 wrapper, + --gae-lambda), then G9t restarts in place from
  # its own save. S14b: the same without runs/g9t/RECIPE -> refused, G9r continues from its own save. S14c: a fresh
  # successor (a16 / a40) after G9t started: its newest MIRRORED G9t file, never a newer G9r mirror or a parked dir.
  # S14d: the A30 (r1 MODE g9t, a fresh job): G9t's first start with 28 workers, cache 192 + stacked-opp + cpu 400.
  g9ttree() {  # g9rtree + the 2026-11-12 state; MODE g9r on the A16 with G9r started and mirrored
    g9rtree; G=$M/nv2/runs/g9; rm -f "$M/nv2/bin/py-src-4f18108"; mkdir -p "$M/nv2/runs/g9t/lam"
    echo "COMMIT by r1 718701 2026-11-11T11:05:44" > "$SW/DECISION"; touch "$G/SWAP_APPLIED" "$M/nv2/runs/r1/SWAP_APPLIED" "$G/G9P_A16_STARTED" "$M/nv2/runs/r1/G9Q_A30_STARTED"
    echo g9r > "$G/MODE"; echo g9q > "$M/nv2/runs/r1/MODE"; date > "$M/nv2/runs/g9r/G9R_STARTED"
    lineage "$K/G9r" G9r-mirror '12 01:44' 00001000000 00002000000; echo P3 > "$K/G9r/pool/frozen/P3-g9-s2.pt"; rm -f "$K/G9r/pool/frozen/P1-g9h-dmg-s2.pt"
    echo 15356 > "$M/gpu_mib-810367"
  }
  g9targs() {  # g9targs C LAMBDA WORKERS JOB OPP RESUME: the G9t TRAIN_ARGS pattern after --batch-size 4096
    local fl=0.5 cx=""; if [ "$1" = g9r ]; then fl=0.35; cx=" --lr-schedule power --lr-horizon 300000000"; fi
    echo "--lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min $fl --guard-ptop-max 0.6 --kl-anchor $P3 --kl-anchor-coef 0.1 --kl-anchor-decay-steps 50000000 --critic-warmup-steps 1000000 --frozen $P3 --p-frozen 0.67 --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers $3 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir $T/ws-netv2-$4/run/G9t --pool-dir $T/ws-netv2-$4/run/G9t/pool $5$cx --gae-lambda $2 --resume $6"
  }
  flip() { printf 'echo g9t > runs/%s/MODE\ntouch runs/%s/restart  # mock only\n' "$2" "$2" > "$M/pending-inline-$1"; echo "$M/nv2/runs/$2/INLINE_CMD" > "$M/pending-inline-$1.dest"; }
  if [ -z "${MOCK_SKIP_G9T:-}" ]; then
  echo "== S14: 810367 (A16, G9r) is flipped to MODE g9t by an INLINE_CMD (RECIPE 'g9r 0.5'): G9t's first start from P3, then its restart in place"
  g9ttree; echo "g9r 0.5" > "$M/nv2/runs/g9t/RECIPE"; flip 810367 g9
  job g9 810367 10 a16 ws-netv2-g9 restart > "$M/s-g9.txt"; c2=$(grep "ws-netv2-810367/" "$M/calls.log"); sed 's/^/    /' <<< "$c2" | cut -c1-260
  expect "S14 first: G9r (its mirrored file), the ff32b71 wrapper" "^call resume=$K/G9r/final_model.pt \[G9r-mirror-final\] run_dir=$T/ws-netv2-810367/run/G9r .* ps=$NOV $PP9R" "$(sed -n 1p <<< "$c2")"
  expect "S14 relaunch = G9t's first start from P3: G9t dirs, P1 eval anchor, November sets, the ff32b71 wrapper" "^call resume=$P3 \[P3\] run_dir=$T/ws-netv2-810367/run/G9t pool_dir=$T/ws-netv2-810367/run/G9t/pool .*anchor=$P1 ps=$NOV $PP9R" "$(sed -n 2p <<< "$c2")"
  expect "S14 G9t restart in place: its own save" "^call resume=$T/ws-netv2-810367/run/G9t/final_model.pt \[trained-from:P3\] run_dir=$T/ws-netv2-810367/run/G9t .* ps=$NOV $PP9R" "$(sed -n 3p <<< "$c2")"
  expect "S14 G9t args (first start)" "with TRAIN_ARGS='--no-bf16 --n-epochs 3 --batch-size 4096 $(g9targs g9r 0.5 8 810367 "--snapshot-cache 72 --snapshot-cache-cpu 200" "$P3")'" "$(cat "$M/job-810367.out")"
  expect "S14 G9t args (restart in place)" "with TRAIN_ARGS='--no-bf16 --n-epochs 3 --batch-size 4096 $(g9targs g9r 0.5 8 810367 "--snapshot-cache 72 --snapshot-cache-cpu 200" "$T/ws-netv2-810367/run/G9t/final_model.pt")'" "$(cat "$M/job-810367.out")"
  expect "S14 G9T_STARTED records the recipe and the run; G9r's save left as it was; MODE g9t; 3 trainer calls" "^[0-9T:-]+ g9r 0.5 g9 810367 trained-from:G9r-mirror-final g9t 3$" "$(cat "$M/nv2/runs/g9t/G9T_STARTED" 2>/dev/null) $(cat "$G"/*-810367/G9r/final_model.pt 2>/dev/null) $(cat "$G/MODE") $(grep -c . <<< "$c2")"
  echo "== S14b: the same flip without runs/g9t/RECIPE: G9t refused, G9r continues from its own save"
  g9ttree; flip 810367 g9
  job g9 810367 10 a16 ws-netv2-g9 restart > "$M/s-g9.txt"; c2=$(grep "ws-netv2-810367/" "$M/calls.log"); sed 's/^/    /' <<< "$c2" | cut -c1-260
  expect "S14b relaunch: G9r from its own save, no G9t" "^call resume=$T/ws-netv2-810367/run/G9r/final_model.pt \[trained-from:G9r-mirror-final\] run_dir=$T/ws-netv2-810367/run/G9r .* ps=$NOV $PP9R" "$(sed -n 2p <<< "$c2")"
  expect "S14b refusal logged, no G9T_STARTED, MODE file kept" "MODE g9t refused: no runs/g9t/RECIPE: G9r continues .* n g9t$" "$(cat "$G"/*-810367/train.log 2>/dev/null | tr '\n' ' ') $([ -e "$M/nv2/runs/g9t/G9T_STARTED" ] && echo y || echo n) $(cat "$G/MODE")"
  if grep -q -E "gae-lambda|/G9t" "$M/job-810367.out"; then bad "S14b G9t args after a refusal"; else ok "S14b no G9t args"; fi
  echo "== S14c: a fresh successor (1100002, a16 / a40) after G9t started (record 'g9q 0.6 g9'): the newest MIRRORED G9t file"
  for g in a16 a40; do
    g9ttree; echo g9t > "$G/MODE"; echo "g9q 0.6" > "$M/nv2/runs/g9t/RECIPE"; echo "2026-11-13T09:00:00 g9q 0.6 g9 810367" > "$M/nv2/runs/g9t/G9T_STARTED"
    K2=$G/20261112-015035-810367; lineage "$K2/G9t" G9t-mirror '13 11:00' 00001000000 00002000000 00003000000; echo P3 > "$K2/G9t/pool/frozen/P3-g9-s2.pt"; rm -f "$K2/G9t/pool/frozen/P1-g9h-dmg-s2.pt"
    at '13 12:00' "$K/G9r/final_model.pt"; for d in "$M/nv2/runs/g9r-parked" "$M/nv2/runs/g9t-parked"; do mkdir -p "$d/pool"; echo parked > "$d/final_model.pt"; at '13 12:30' "$d/final_model.pt"; done
    echo $([ $g = a16 ] && echo 15356 || echo 46068) > "$M/gpu_mib-1100002"; job g9 1100002 10 $g ws-netv2-g9 restart > /dev/null; calls
    expect "S14c 1100002 ($g): the mirrored G9t save with its pool / frozen / best / ema, wrapper, November sets" "^call resume=$K2/G9t/final_model.pt \[G9t-mirror-final\] run_dir=$T/ws-netv2-1100002/run/G9t pool_dir=$T/ws-netv2-1100002/run/G9t/pool pool=3 frozen=2 best=y ema=y anchor=$P1 ps=$NOV $PP9R" "$(sed -n 1p <<< "$c")"
    expect "S14c 1100002 ($g) restart in place: its own save" "^call resume=$T/ws-netv2-1100002/run/G9t/final_model.pt \[trained-from:G9t-mirror-final\] .*$PP9R" "$(sed -n 2p <<< "$c")"
    x="--snapshot-cache 72 --snapshot-cache-cpu 200"; [ $g = a40 ] && x="--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200"
    expect "S14c 1100002 ($g) args: $x --gae-lambda 0.6" "with TRAIN_ARGS='--no-bf16 --n-epochs 3 --batch-size 4096 $(g9targs g9q 0.6 8 1100002 "$x" "$T/ws-netv2-1100002/run/G9t/final_model.pt")'" "$(cat "$M/job-1100002.out")"
  done
  echo "== S14d: the A30 (a fresh r1 job 1100003, r1 MODE g9t, RECIPE 'g9q 0.5'): G9t's first start from P3, then its restart in place"
  g9ttree; echo g9t > "$M/nv2/runs/r1/MODE"; echo "g9q 0.5" > "$M/nv2/runs/g9t/RECIPE"; echo 24576 > "$M/gpu_mib-1100003"
  job r1 1100003 30 a30 ws-netv2-r1 restart > /dev/null; calls
  expect "S14d first start from P3 in the G9t dirs, wrapper, November sets" "^call resume=$P3 \[P3\] run_dir=$T/ws-netv2-1100003/run/G9t pool_dir=$T/ws-netv2-1100003/run/G9t/pool .*anchor=$P1 ps=$NOV $PP9R" "$(sed -n 1p <<< "$c")"
  expect "S14d restart in place: its own save" "^call resume=$T/ws-netv2-1100003/run/G9t/final_model.pt \[trained-from:P3\] .*$PP9R" "$(sed -n 2p <<< "$c")"
  expect "S14d args: 28 workers, cache 192 + stacked-opp + cpu 400, --gae-lambda 0.5" "with TRAIN_ARGS='--no-bf16 --n-epochs 3 --batch-size 4096 $(g9targs g9q 0.5 28 1100003 "--snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400" "$T/ws-netv2-1100003/run/G9t/final_model.pt")'" "$(cat "$M/job-1100003.out")"
  expect "S14d G9T_STARTED records run r1" "^[0-9T:-]+ g9q 0.5 r1 1100003$" "$(cat "$M/nv2/runs/g9t/G9T_STARTED" 2>/dev/null)"
  fi
  # S15 (v12, 2026-11-12, orchestrator-approved): the successor watchdog through the real job script (ids >= 1135476;
  # $MOCK/squeue.txt switches the sbatch / squeue / scontrol stubs to scripts/netv2_wd_stubs). Tree: g9ttree (the
  # 2026-11-12 state, G9r on the A16, MODE g9r, mirrored). S15a: a fresh g9 job queues ONE successor (afterany, its
  # resources), restarts in place without a second one, and chain_left 2 -> 0 stops the end-of-job chain. S15b: /tmp <
  # 20 GB at the start: exit 1 before any trainer call, resubmitted with --exclude=<node>, MODE kept; the resubmitted
  # job resumes the same lineage. S15c: the /tmp watcher (< 3 GB): mirror, the pending successor excludes the node,
  # restart flag removed, SIGINT; the trainer's final save is mirrored before the job ends; no relaunch. S15d: HOLD ->
  # exit 0 without training or chaining (HOLD kept). S15e: another job of the name is RUNNING (started earlier) -> exit 0.
  HN=$(hostname -s); WQ=$M/squeue.txt
  cnt() { local n; n=$(grep -c -- "$1" "$2" 2> /dev/null); echo "${n:-0}"; }
  echo "== S15a: a fresh g9 job (1200010, A16) queues one successor, restarts in place, chain_left 2 -> 0 (no end-of-job chain)"
  g9ttree; rm -f "$WQ"*; : > "$WQ"; echo 2 > "$G/chain_left"; echo 15356 > "$M/gpu_mib-1200010"; job g9 1200010 10 a16 ws-netv2-g9 restart > /dev/null; calls
  expect "S15a trainer: the mirrored G9r save, then its restart in place" "^call resume=$K/G9r/final_model.pt \[G9r-mirror-final\] run_dir=$T/ws-netv2-1200010/run/G9r .*call resume=$T/ws-netv2-1200010/run/G9r/final_model.pt " "$(tr '\n' ' ' <<< "$c")"
  expect "S15a exactly one sbatch: the successor afterany:1200010 with this job's resources" "^sbatch --job-name=ws-netv2-g9 --partition=a16 --gres=gpu:1 --cpus-per-task=10 --mem=65536M --time=72:00:00 --dependency=afterany:1200010 scripts/slurm_train_v2.sbatch g9 1$" "$(cat "$M/sbatch.log") $(grep -c . "$M/sbatch.log")"
  expect "S15a queued 1300001 (id parsed past the banner), JOBSTARTS, chain_left 0" "queued successor 1300001 \(afterany:1200010\).* 1 0$" "$(tr '\n' ' ' < "$G/WATCHDOG.log") $(grep -c 1200010 "$G/JOBSTARTS") $(cat "$G/chain_left")"
  expect "S15a the job's mirrored train.log has the watchdog lines" "watchdog: queued successor 1300001" "$(cat "$G"/*-1200010/train.log 2>/dev/null)"
  echo "== S15b: /tmp has 10 GB at the start of 1200011: exit 1 before training, resubmitted excluding $HN; the resubmitted job resumes G9r"
  g9ttree; rm -f "$WQ"*; : > "$WQ"; echo $((10 * 1048576)) > "$WQ.df"; echo 15356 > "$M/gpu_mib-1200011"; job g9 1200011 10 a16 ws-netv2-g9 restart > /dev/null; calls
  expect "S15b no trainer call, exit 1" "^0 exit 1$" "$(cnt ws-netv2-1200011 "$M/calls.log") $(grep -o 'exit [0-9]*' <<< "$(job_rc 1200011)")"
  expect "S15b resubmitted with --exclude=$HN, no dependency; MODE g9r kept" "^sbatch --job-name=ws-netv2-g9 --partition=a16 --gres=gpu:1 --cpus-per-task=10 --mem=65536M --time=72:00:00 --exclude=$HN scripts/slurm_train_v2.sbatch g9 g9r$" "$(cat "$M/sbatch.log") $(cat "$G/MODE")"
  expect "S15b WATCHDOG.log + TMP_ALERT" "/tmp has 10 GB < 20 GB: resubmitted as 1300001 excluding $HN; exit before training \(MODE stays g9r\).*y$" "$(tr '\n' ' ' < "$G/WATCHDOG.log") $([ -f "$G/TMP_ALERT" ] && echo y)"
  echo $((60 * 1048576)) > "$WQ.df"; echo 15356 > "$M/gpu_mib-1300001"; job g9 1300001 10 a16 ws-netv2-g9 > /dev/null; calls
  expect "S15b the resubmitted 1300001: the same lineage (G9r's mirror), its own successor 1300002" "^call resume=$K/G9r/final_model.pt \[G9r-mirror-final\] run_dir=$T/ws-netv2-1300001/run/G9r .*queued successor 1300002 \(afterany:1300001\)" "$(grep ws-netv2-1300001 <<< "$c" | tr '\n' ' ')$(tr '\n' ' ' < "$G/WATCHDOG.log")"
  echo "== S15c: the /tmp watcher in 1200012: 50 GB at the start, then 2 GB: mirror, exclude $HN, no restart, SIGINT, final save mirrored"
  g9ttree; rm -f "$WQ"*; : > "$WQ"; printf '%s\n%s\n%s\n' $((50 * 1048576)) $((50 * 1048576)) $((2 * 1048576)) > "$WQ.df"; echo 15356 > "$M/gpu_mib-1200012"
  MOCK_POLL=1 MOCK_WAIT=1 job g9 1200012 10 a16 ws-netv2-g9 restart > /dev/null; calls
  w=$(tr '\n' ' ' < "$G/WATCHDOG.log")
  expect "S15c one trainer call (no relaunch), which got SIGINT" "^call resume=$K/G9r/final_model.pt .* trainer: got SIGINT $" "$(tr '\n' ' ' <<< "$c")"
  expect "S15c order: /tmp low -> mirror done -> the successor excludes $HN -> restart flag removed -> SIGINT" "/tmp has 2048 MB left .*mirror to $G/[0-9-]+-1200012 done .*pending 1300001: ExcNodeList=$HN .*restart flag removed, stopping the trainer .*SIGINT [0-9]+" "$w"
  expect "S15c scontrol update on the pending successor; runs/g9/restart gone; TMP_ALERT" "scontrol update jobid=1300001 ExcNodeList=$HN n y$" "$(cat "$WQ.scontrol" 2>/dev/null | tr '\n' ' ')$([ -e "$G/restart" ] && echo y || echo n) $([ -f "$G/TMP_ALERT" ] && echo y)"
  expect "S15c the final save (after SIGINT) is in the job's mirror, LOCAL removed" "^trained-from:G9r-mirror-final n$" "$(cat "$G"/*-1200012/G9r/final_model.pt 2>/dev/null) $([ -d "$T/ws-netv2-1200012" ] && echo y || echo n)"
  echo "== S15d: runs/g9/HOLD at the start of 1200013: exit 0, no training, no successor, HOLD kept"
  g9ttree; rm -f "$WQ"*; : > "$WQ"; touch "$G/HOLD"; echo 15356 > "$M/gpu_mib-1200013"; job g9 1200013 10 a16 ws-netv2-g9 restart > /dev/null; calls
  expect "S15d no trainer call, no sbatch, exit 0, HOLD kept, reason logged" "^0 0 exit 0 y .*exit without training or chaining: runs/g9/HOLD exists" "$(cnt ws-netv2-1200013 "$M/calls.log") $(cnt . "$M/sbatch.log") $(grep -o 'exit [0-9]*' <<< "$(job_rc 1200013)") $([ -f "$G/HOLD" ] && echo y) $(tr '\n' ' ' < "$G/WATCHDOG.log")"
  echo "== S15e: 1200001 (ws-netv2-g9) is RUNNING since earlier: 1200014 exits 0 without training; 1300005 is its successor already"
  g9ttree; rm -f "$WQ"*; printf '1200001 ws-netv2-g9 RUNNING 2026-11-12T00:00:00 (null)\n1300005 ws-netv2-g9 PENDING N/A afterany:1200001\n' > "$WQ"; echo 15356 > "$M/gpu_mib-1200014"
  job g9 1200014 10 a16 ws-netv2-g9 restart > /dev/null; calls
  expect "S15e no trainer call, no sbatch, reason logged" "^0 0 .*job 1200001 of run g9 is already running \(started 2026-11-12T00:00:00\); pending: 1300005" "$(cnt ws-netv2-1200014 "$M/calls.log") $(cnt . "$M/sbatch.log") $(tr '\n' ' ' < "$G/WATCHDOG.log")"
  rm -f "$WQ"*
  # S16 (v14, 2026-11-12; G9t takes the A16, G9r PARKED resumable; skipped with MOCK_SKIP_V14=1 or MOCK_SKIP_G9T=1):
  # S16: 810367 (G9r) gets the park INLINE_CMD (scripts/netv2_g9t/inline-a16-park-g9r-start-g9t.sh, queued during its
  # trainer's run, + a mock-only `touch restart`): G9r parked whole in runs/g9r-parked, MODE g9t, the relaunch = G9t's
  # first start from P3 (G9q's recipe + --gae-lambda 0.5), then its restart in place. S16b: the pending successor 1137377
  # (watchdog on) starts fresh after it: not blocked by the park, one successor queued, G9t's newest MIRRORED file.
  # S16c: MODE g9t refused with G9r parked (RECIPE removed, G9T_STARTED corrupted) in a fresh job -> HOLD rounds (mock
  # cap MOCK_HOLD_ROUNDS), no trainer call, never G9r. S16d: refused without a park -> G9r continues (v13). S16e: the park
  # refuses (RECIPE 'g9q 0.6'): MODE g9r, no park dir, G9r relaunches from its own save.
  if [ -z "${MOCK_SKIP_G9T:-}" ] && [ -z "${MOCK_SKIP_V14:-}" ]; then
  t16tree() {  # g9ttree + RECIPE "g9q 0.5" + a read-only runs/g9t/PREREG.txt (the park's preconditions)
    g9ttree; rm -f "$WQ"* "$M/sbatch.log"; echo "${1:-g9q 0.5}" > "$M/nv2/runs/g9t/RECIPE"; echo prereg > "$M/nv2/runs/g9t/PREREG.txt"; chmod 444 "$M/nv2/runs/g9t/PREREG.txt"
  }
  Z=$M/nv2/runs/g9r-parked; A16X="--snapshot-cache 72 --snapshot-cache-cpu 200"
  echo "== S16: 810367 (A16, G9r) parks G9r through the INLINE_CMD and relaunches as G9t (first start from P3, 'g9q 0.5'), then restarts G9t in place"
  t16tree; queue 810367 g9 ../netv2_g9t/inline-a16-park-g9r-start-g9t.sh
  job g9 810367 10 a16 ws-netv2-g9 restart > "$M/s-g9.txt"; c2=$(grep "ws-netv2-810367/" "$M/calls.log"); sed 's/^/    /' <<< "$c2" | cut -c1-260
  expect "S16 first: G9r (its mirrored file), the ff32b71 wrapper, November sets" "^call resume=$K/G9r/final_model.pt \[G9r-mirror-final\] run_dir=$T/ws-netv2-810367/run/G9r .* ps=$NOV $PP9R" "$(sed -n 1p <<< "$c2")"
  expect "S16 relaunch = G9t's first start from P3: G9t dirs, P1 eval anchor, November sets, the ff32b71 wrapper" "^call resume=$P3 \[P3\] run_dir=$T/ws-netv2-810367/run/G9t pool_dir=$T/ws-netv2-810367/run/G9t/pool .*anchor=$P1 ps=$NOV $PP9R" "$(sed -n 2p <<< "$c2")"
  expect "S16 G9t restart in place: its own save" "^call resume=$T/ws-netv2-810367/run/G9t/final_model.pt \[trained-from:P3\] run_dir=$T/ws-netv2-810367/run/G9t .* ps=$NOV $PP9R" "$(sed -n 3p <<< "$c2")"
  expect "S16 G9t args (first start): G9q's recipe + --gae-lambda 0.5, 8 workers, cache 72 + cpu 200" "with TRAIN_ARGS='--no-bf16 --n-epochs 3 --batch-size 4096 $(g9targs g9q 0.5 8 810367 "$A16X" "$P3")'" "$(cat "$M/job-810367.out")"
  expect "S16 G9t args (restart in place)" "with TRAIN_ARGS='--no-bf16 --n-epochs 3 --batch-size 4096 $(g9targs g9q 0.5 8 810367 "$A16X" "$T/ws-netv2-810367/run/G9t/final_model.pt")'" "$(cat "$M/job-810367.out")"
  expect "S16 G9r parked with its last save (+ 2 pool, 2 frozen, best, ema, checksums), MODE g9t, G9T_STARTED, no runs/g9/PARKED, 3 calls" "^trained-from:G9r-mirror-final 2 2 G9r-mirror-best G9r-mirror-ema y g9t [0-9T:-]+ g9q 0.5 g9 810367 n 3$" "$(cat "$Z/final_model.pt" 2>/dev/null) $(ls "$Z/pool" 2>/dev/null | grep -c pt$) $(ls "$Z/pool/frozen" 2>/dev/null | grep -c pt$) $(cat "$Z/best_model.pt" 2>/dev/null) $(cat "$Z/ema_model.pt" 2>/dev/null) $(cd "$Z" 2>/dev/null && md5sum -c --quiet MD5SUMS > /dev/null 2>&1 && echo y) $(cat "$G/MODE") $(cat "$M/nv2/runs/g9t/G9T_STARTED" 2>/dev/null) $([ -e "$G/PARKED" ] && echo y || echo n) $(grep -c . <<< "$c2")"
  expect "S16 PARKED.txt: steps, job, source; park.log DONE" "^G9r PARKED .* in job 810367 .*MODE g9r -> g9t.*steps: 99000000 .*source: $T/ws-netv2-810367/run/G9r .*counters: \{'steps': 99000000.* DONE: G9r parked in $Z at 99000000 steps" "$(cat "$Z/PARKED.txt" "$M/nv2/runs/g9t/park.log" 2>/dev/null | tr '\n' ' ')"
  echo "== S16b: the pending successor 1137377 (fresh, watchdog on) after S16: G9t's newest mirrored file, one successor, not blocked by runs/g9r-parked"
  printf '1137377 ws-netv2-g9 RUNNING 2026-11-15T01:46:00 (null)\n' > "$WQ"; rm -f "$M/sbatch.log"; echo 15356 > "$M/gpu_mib-1137377"
  job g9 1137377 10 a16 ws-netv2-g9 restart > /dev/null; c3=$(grep "ws-netv2-1137377/" "$M/calls.log"); sed 's/^/    /' <<< "$c3" | cut -c1-260
  expect "S16b first: G9t's mirrored save from 810367 (never a G9r file or the parked dir), pool / frozen / best / ema, wrapper, November sets" "^call resume=$G/[0-9]+-[0-9]+-810367/G9t/final_model.pt \[trained-from:trained-from:P3\] run_dir=$T/ws-netv2-1137377/run/G9t pool_dir=$T/ws-netv2-1137377/run/G9t/pool .*anchor=$P1 ps=$NOV $PP9R" "$(sed -n 1p <<< "$c3")"  # (the stub trainer leaves no G9t pool / best to re-seed: S14c checks that)
  expect "S16b restart in place: its own save" "^call resume=$T/ws-netv2-1137377/run/G9t/final_model.pt \[trained-from:trained-from:trained-from:P3\] .*$PP9R" "$(sed -n 2p <<< "$c3")"
  expect "S16b args (restart in place): G9q's recipe + --gae-lambda 0.5" "with TRAIN_ARGS='--no-bf16 --n-epochs 3 --batch-size 4096 $(g9targs g9q 0.5 8 1137377 "$A16X" "$T/ws-netv2-1137377/run/G9t/final_model.pt")'" "$(cat "$M/job-1137377.out")"
  expect "S16b exactly one sbatch: its successor afterany:1137377 (a16, 10 CPUs, 64G)" "^sbatch --job-name=ws-netv2-g9 --partition=a16 --gres=gpu:1 --cpus-per-task=10 --mem=65536M --time=72:00:00 --dependency=afterany:1137377 scripts/slurm_train_v2.sbatch g9 1$" "$(cat "$M/sbatch.log") $(grep -c . "$M/sbatch.log")"
  expect "S16b WATCHDOG.log: first start (MODE g9t), queued 1300001; never blocked; no G9r call" "MODE g9: g9t .*queued successor 1300001 \(afterany:1137377\) +n 0$" "$(tr '\n' ' ' < "$G/WATCHDOG.log") $(grep -q 'exit without training' "$G/WATCHDOG.log" && echo y || echo n) $(grep -c '/G9r ' <<< "$c3")"
  echo "== S16c: a fresh g9 job 1137379 with G9r parked and MODE g9t refused (RECIPE removed, G9T_STARTED corrupt): HOLD rounds, no trainer call, never G9r"
  rm -f "$M/nv2/runs/g9t/RECIPE"; echo corrupt > "$M/nv2/runs/g9t/G9T_STARTED"; printf '1137379 ws-netv2-g9 RUNNING 2026-11-15T03:00:00 (null)\n' > "$WQ"; rm -f "$M/sbatch.log"; echo 15356 > "$M/gpu_mib-1137379"
  MOCK_HOLD_ROUNDS=2 job g9 1137379 10 a16 ws-netv2-g9 > /dev/null; l=$(cat "$T/ws-netv2-1137379/run/train.log" 2>/dev/null)
  expect "S16c no trainer call, exit 3 (the mock cap), HOLD, MODE g9t, record untouched" "^0 exit 3 y g9t corrupt$" "$(cnt ws-netv2-1137379 "$M/calls.log") $(grep -o 'exit [0-9]*' <<< "$(job_rc 1137379)") $([ -f "$G/HOLD" ] && echo y) $(cat "$G/MODE") $(cat "$M/nv2/runs/g9t/G9T_STARTED")"
  expect "S16c train.log: refused (G9T_STARTED unreadable) with G9r parked -> HOLD rounds 1, 2, then the mock cap; never 'G9r continues'" "MODE g9t refused: runs/g9t/G9T_STARTED unreadable .*HOLD: G9r is parked .*\(round 1\): HOLD .*\(round 2\): HOLD .*held 2 rounds \(mock cap\): exit +n$" "$(tr '\n' ' ' <<< "$l") $(grep -q 'G9r continues' <<< "$l" && echo y || echo n)"
  rm -f "$G/HOLD"
  echo "== S16d: a fresh g9 job 1137380, MODE g9t refused (no RECIPE, no record) and G9r NOT parked: G9r continues (v13)"
  t16tree; rm -f "$M/nv2/runs/g9t/RECIPE"; echo g9t > "$G/MODE"; printf '1137380 ws-netv2-g9 RUNNING 2026-11-15T03:00:00 (null)\n' > "$WQ"; echo 15356 > "$M/gpu_mib-1137380"
  job g9 1137380 10 a16 ws-netv2-g9 > /dev/null; c4=$(grep "ws-netv2-1137380/" "$M/calls.log")
  expect "S16d G9r's mirrored save, wrapper, November sets; refusal logged; no HOLD" "^call resume=$K/G9r/final_model.pt \[G9r-mirror-final\] run_dir=$T/ws-netv2-1137380/run/G9r .*ps=$NOV ${PP9R%$} MODE g9t refused: no runs/g9t/RECIPE: G9r continues n$" "$(sed -n 1p <<< "$c4") $(grep -h 'MODE g9t refused' "$G"/*-1137380/train.log 2>/dev/null | cut -d' ' -f2-) $([ -f "$G/HOLD" ] && echo y || echo n)"
  echo "== S16e: the park refuses in 810367 (RECIPE 'g9q 0.6'): MODE g9r, no park dir, G9r relaunches from its own save"
  t16tree "g9q 0.6"; queue 810367 g9 ../netv2_g9t/inline-a16-park-g9r-start-g9t.sh
  job g9 810367 10 a16 ws-netv2-g9 restart > /dev/null; c2=$(grep "ws-netv2-810367/" "$M/calls.log")
  expect "S16e relaunch: G9r from its own save" "^call resume=$T/ws-netv2-810367/run/G9r/final_model.pt \[trained-from:G9r-mirror-final\] run_dir=$T/ws-netv2-810367/run/G9r .*$PP9R" "$(sed -n 2p <<< "$c2")"
  expect "S16e MODE g9r, no park dir, no G9T_STARTED, FAIL logged" "^g9r n n y$" "$(cat "$G/MODE") $([ -e "$Z" ] && echo y || echo n) $([ -e "$M/nv2/runs/g9t/G9T_STARTED" ] && echo y || echo n) $(grep -q "park-g9r FAIL (MODE untouched: g9r): runs/g9t/RECIPE is not exactly 'g9q 0.5'" "$M/nv2/runs/g9t/park.log" 2>/dev/null && echo y)"
  if grep -q -E "gae-lambda|/G9t" "$M/job-810367.out"; then bad "S16e G9t args after a refused park"; else ok "S16e no G9t args"; fi
  chmod -R u+w "$M/nv2/runs/g9t" 2>/dev/null; rm -f "$WQ"*
  fi
fi
# S17 (v15, 2026-11-15; USER decision ~14:05, runs/g9q2/PREREG.txt; skipped with MOCK_SKIP_G9Q2=1 or MOCK_SKIP_G9T=1):
# the fresh A16 job (watchdog on) with g9 MODE g9q2 after G9t's park (runs/g9/HOLD + runs/g9/PARKED removed at deploy):
# S17: G9q2's first start from P3 = G9q's recipe + --seed 7272027, the 24c633f wrapper, November sets, cache 72 + cpu
# 200, one successor queued; its restart in place: its own save, no --seed. S17b: the successor: G9q2's newest mirrored
# file (never G9q's / G9t's / G9r's), no --seed. S17c: runs/g9/PARKED still there -> exit 0, no trainer call, no sbatch.
if [ -z "${MOCK_SKIP_G9T:-}" ] && [ -z "${MOCK_SKIP_G9Q2:-}" ]; then
  q2args() {  # q2args JOB EXTRA RESUME: the G9q2 TRAIN_ARGS pattern after --batch-size 4096
    echo "--lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5 --guard-ptop-max 0.6 --kl-anchor $P3 --kl-anchor-coef 0.1 --kl-anchor-decay-steps 50000000 --critic-warmup-steps 1000000 --frozen $P3 --p-frozen 0.67 --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir $T/ws-netv2-$1/run/G9q2 --pool-dir $T/ws-netv2-$1/run/G9q2/pool --snapshot-cache 72 --snapshot-cache-cpu 200$2 --resume $3"
  }
  q2tree17() {
    g9ttree; wrapper; rm -f "$WQ"* "$M/sbatch.log"; : > "$WQ"; echo g9q2 > "$G/MODE"
    mkdir -p "$M/nv2/runs/g9t-parked"; echo parked > "$M/nv2/runs/g9t-parked/PARKED.txt"; echo "2026-11-13T09:11:25 g9q 0.5 parked 810367" > "$M/nv2/runs/g9t/G9T_STARTED"
    lineage "$G/20261112-015035-810367/G9t" G9t-mirror '14 18:39' 00055000000; lineage "$K/G9q" G9q-mirror '17 12:00' 00025038951
    echo 15356 > "$M/gpu_mib-1300100"; echo 15356 > "$M/gpu_mib-1300101"
  }
  PP24="pp=$M/nv2/src-24c633f:$M/nv2/src$"
  echo "== S17: a fresh A16 job 1300100 with MODE g9q2: G9q2's first start from P3 (+ --seed), then its restart in place (no --seed)"
  q2tree17; job g9 1300100 10 a16 ws-netv2-g9 restart > /dev/null; c5=$(grep "ws-netv2-1300100/" "$M/calls.log"); sed 's/^/    /' <<< "$c5" | cut -c1-260
  expect "S17 first start: P3, G9q2 dirs, P1 eval anchor, November sets, the 24c633f wrapper" "^call resume=$P3 \[P3\] run_dir=$T/ws-netv2-1300100/run/G9q2 pool_dir=$T/ws-netv2-1300100/run/G9q2/pool .*anchor=$P1 ps=$NOV $PP24" "$(sed -n 1p <<< "$c5")"
  expect "S17 restart in place: its own save" "^call resume=$T/ws-netv2-1300100/run/G9q2/final_model.pt \[trained-from:P3\] run_dir=$T/ws-netv2-1300100/run/G9q2 .*ps=$NOV $PP24" "$(sed -n 2p <<< "$c5")"
  expect "S17 args (first start, the trainer's argv): G9q's recipe + --seed 7272027" "^1300100 -u -m porygon.v2.train .* --max-hours 70.5 --no-bf16 --n-epochs 3 --batch-size 4096 $(q2args 1300100 " --seed 7272027" "$P3")$" "$(grep '^1300100 ' "$M/argv.log" | sed -n 1p)"
  expect "S17 restart argv: no --seed" "^1300100 .* --snapshot-cache-cpu 200 --resume $T/ws-netv2-1300100/run/G9q2/final_model.pt$" "$(grep '^1300100 ' "$M/argv.log" | sed -n 2p)"
  expect "S17 args (restart in place): no --seed" "with TRAIN_ARGS='--no-bf16 --n-epochs 3 --batch-size 4096 $(q2args 1300100 "" "$T/ws-netv2-1300100/run/G9q2/final_model.pt")'" "$(cat "$M/job-1300100.out")"
  expect "S17 G9Q2_STARTED, MODE g9q2, 2 calls, exactly one sbatch (afterany:1300100, a16, 10 CPUs, 64G)" "^[0-9T:-]+ seed 7272027 g9 1300100 g9q2 2 sbatch --job-name=ws-netv2-g9 --partition=a16 --gres=gpu:1 --cpus-per-task=10 --mem=65536M --time=72:00:00 --dependency=afterany:1300100 scripts/slurm_train_v2.sbatch g9 1$" "$(cat "$M/nv2/runs/g9q2/G9Q2_STARTED" 2>/dev/null) $(cat "$G/MODE") $(grep -c . <<< "$c5") $(cat "$M/sbatch.log") $(grep -c . "$M/sbatch.log")"
  echo "== S17b: the successor 1300101 (fresh): G9q2's newest mirrored save, no --seed"
  rm -f "$M/sbatch.log"; : > "$WQ"; job g9 1300101 10 a16 ws-netv2-g9 > /dev/null; c6=$(grep "ws-netv2-1300101/" "$M/calls.log"); sed 's/^/    /' <<< "$c6" | cut -c1-260
  expect "S17b G9q2's mirrored save from 1300100 (not the newer G9q mirror), wrapper, November sets" "^call resume=$G/[0-9]+-[0-9]+-1300100/G9q2/final_model.pt \[trained-from:trained-from:P3\] run_dir=$T/ws-netv2-1300101/run/G9q2 .*ps=$NOV $PP24" "$(sed -n 1p <<< "$c6")"
  if grep -q -- "--seed" "$M/job-1300101.out"; then bad "S17b --seed in a later start"; else ok "S17b no --seed"; fi
  echo "== S17c: runs/g9/PARKED left in place (deploy step skipped): the fresh job exits 0 without training or chaining"
  q2tree17; touch "$G/PARKED"; job g9 1300100 10 a16 ws-netv2-g9 > /dev/null
  expect "S17c no trainer call, exit 0, no sbatch, no G9Q2_STARTED" "^0 exit 0 0 n$" "$(cnt ws-netv2-1300100 "$M/calls.log") $(job_rc 1300100) $(grep -c . "$M/sbatch.log" 2>/dev/null || echo 0) $([ -e "$M/nv2/runs/g9q2/G9Q2_STARTED" ] && echo y || echo n)"
fi
# S18 (v19, 2026-11-16; runs/p4/PREREG.txt; skipped with MOCK_SKIP_V19=1 / MOCK_SKIP_G9Q2=1 / MOCK_SKIP_G9T=1): fresh A40
# jobs (watchdog on) with g9 MODE g9p4. S18: G9-P4's first start in a fresh job = BC-P4 in all three roles, the 24c633f
# wrapper, November sets, cache 512 + --stacked-opp + cpu 200, no --seed, one successor; its restart in place: its own
# save. S18b: the successor: G9-P4's newest mirrored save (never G9q2's / G9q's). S18c: G9-P4 parked (PARKED.txt + the
# park guard's runs/g9/HOLD): the fresh job exits 0 without a trainer call or a successor.
if [ -z "${MOCK_SKIP_G9T:-}" ] && [ -z "${MOCK_SKIP_G9Q2:-}" ] && [ -z "${MOCK_SKIP_V19:-}" ]; then
  P4=$M/nv2/runs/pretrain/P4-g9-s2.pt
  p4args() {  # p4args JOB RESUME: the G9-P4 TRAIN_ARGS pattern after --batch-size 4096 (A40)
    echo "--lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5 --guard-ptop-max 0.6 --kl-anchor $P4 --kl-anchor-coef 0.1 --kl-anchor-decay-steps 50000000 --critic-warmup-steps 1000000 --frozen $P4 --p-frozen 0.67 --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir $T/ws-netv2-$1/run/G9P4 --pool-dir $T/ws-netv2-$1/run/G9P4/pool --snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --resume $2"
  }
  p4tree18() {
    q2tree17; echo g9p4 > "$G/MODE"; mkdir -p "$M/nv2/runs/g9p4"; rm -f "$P4"; echo P4 > "$P4"; chmod a-w "$P4"; md5sum < "$P4" | cut -c1-8 > "$M/nv2/runs/g9p4/P4.md5"
    mkdir -p "$M/nv2/runs/g9q2-parked"; echo parked > "$M/nv2/runs/g9q2-parked/PARKED.txt"; lineage "$G/20261115-170149-1173274/G9q2" G9q2-mirror '17 13:00' 00050000000
    echo 46068 > "$M/gpu_mib-1300300"; echo 46068 > "$M/gpu_mib-1300301"
  }
  echo "== S18: a fresh A40 job 1300300 with MODE g9p4: G9-P4's first start from BC-P4, then its restart in place"
  p4tree18; job g9 1300300 10 a40 ws-netv2-g9 restart > /dev/null; c7=$(grep "ws-netv2-1300300/" "$M/calls.log"); sed 's/^/    /' <<< "$c7" | cut -c1-260
  expect "S18 first start: BC-P4, G9P4 dirs, P1 eval anchor, November sets, the 24c633f wrapper" "^call resume=$P4 \[P4\] run_dir=$T/ws-netv2-1300300/run/G9P4 pool_dir=$T/ws-netv2-1300300/run/G9P4/pool .*anchor=$P1 ps=$NOV $PP24" "$(sed -n 1p <<< "$c7")"
  expect "S18 restart in place: its own save" "^call resume=$T/ws-netv2-1300300/run/G9P4/final_model.pt \[trained-from:P4\] run_dir=$T/ws-netv2-1300300/run/G9P4 .*ps=$NOV $PP24" "$(sed -n 2p <<< "$c7")"
  expect "S18 args (first start, the trainer's argv): G9q's recipe with BC-P4, no --seed" "^1300300 -u -m porygon.v2.train .* --max-hours 70.5 --no-bf16 --n-epochs 3 --batch-size 4096 $(p4args 1300300 "$P4")$" "$(grep '^1300300 ' "$M/argv.log" | sed -n 1p)"
  if grep '^1300300 ' "$M/argv.log" | grep -q -- "--seed\|P3-g9-s2"; then bad "S18 --seed or P3 in G9-P4's argv"; else ok "S18 no --seed, no P3"; fi
  expect "S18 G9P4_STARTED, MODE g9p4, 2 calls, exactly one sbatch (afterany:1300300, a40, 10 CPUs, the job's mem)" "^[0-9T:-]+ $(cat "$M/nv2/runs/g9p4/P4.md5") g9 1300300 g9p4 2 sbatch --job-name=ws-netv2-g9 --partition=a40 --gres=gpu:1 --cpus-per-task=10 --mem=[0-9]+M --time=72:00:00 --dependency=afterany:1300300 scripts/slurm_train_v2.sbatch g9 1$" "$(cat "$M/nv2/runs/g9p4/G9P4_STARTED" 2>/dev/null) $(cat "$G/MODE") $(grep -c . <<< "$c7") $(cat "$M/sbatch.log") $(grep -c . "$M/sbatch.log")"
  echo "== S18b: the successor 1300301 (fresh): G9-P4's newest mirrored save"
  rm -f "$M/sbatch.log"; : > "$WQ"; job g9 1300301 10 a40 ws-netv2-g9 > /dev/null; c8=$(grep "ws-netv2-1300301/" "$M/calls.log"); sed 's/^/    /' <<< "$c8" | cut -c1-260
  expect "S18b G9-P4's mirrored save from 1300300 (not G9q2's / G9q's), wrapper, November sets" "^call resume=$G/[0-9]+-[0-9]+-1300300/G9P4/final_model.pt \[trained-from:trained-from:P4\] run_dir=$T/ws-netv2-1300301/run/G9P4 .*ps=$NOV $PP24" "$(sed -n 1p <<< "$c8")"
  echo "== S18c: G9-P4 parked (runs/g9p4-parked/PARKED.txt + runs/g9/HOLD): a fresh job exits 0 without training or chaining"
  p4tree18; echo "2026-11-17T04:00:00 x g9 1300300" > "$M/nv2/runs/g9p4/G9P4_STARTED"; mkdir -p "$M/nv2/runs/g9p4-parked"; echo parked > "$M/nv2/runs/g9p4-parked/PARKED.txt"; touch "$G/HOLD"
  job g9 1300300 10 a40 ws-netv2-g9 > /dev/null
  expect "S18c no trainer call, exit 0, no sbatch" "^0 exit 0 0$" "$(cnt ws-netv2-1300300 "$M/calls.log") $(job_rc 1300300) $(grep -c . "$M/sbatch.log" 2>/dev/null || echo 0)"
  rm -f "$G/HOLD"; chmod u+w "$P4"; rm -rf "$P4" "$M/nv2/runs/g9p4" "$M/nv2/runs/g9p4-parked" "$M/nv2/runs/g9q2-parked"
  # S19 (v21, 2026-11-20; runs/g9e/PREREG.txt; skipped with MOCK_SKIP_V21=1): the G9E stage machine end to end through
  # the real job script in ONE fresh A40 job 1175374 with MODE g9e (no restart flag at the start: v21 re-arms it at every
  # launch): park (stub) -> G9E from cd@1350 to S0 + 100M -> cooldown G9Ecd100 from the branch point -> reading r100 (stub
  # CONTINUE) -> G9E on to S0 + 250M -> cooldown G9Ecd250 -> reading r250 (stub ROBUST) -> DONE => HOLD rounds (mock cap: exit 3)
  if [ -z "${MOCK_SKIP_V21:-}" ]; then
    echo "== S19: a fresh A40 job 1175374 with MODE g9e: the whole G9E pipeline in one allocation (stub saves / scripts)"
    p4tree18; echo g9e > "$G/MODE"; echo "2026-11-16T21:10:00 x g9 1173274" > "$M/nv2/runs/g9p4/G9P4_STARTED"; echo 46068 > "$M/gpu_mib-1175374"
    GI=$M/nv2/runs/g9q-cd1350/final/01355036714.pt; mkdir -p "$(dirname "$GI")" "$M/nv2/scripts/netv2_g9e" "$M/nv2/runs/g9e"; echo cd1350 > "$GI"; chmod a-w "$GI"
    SE=$M/nv2/scripts/netv2_g9e
    printf '#!/bin/bash\necho "park $SLURM_JOB_ID" >> "$NV2/runs/g9e/stub.calls"; mkdir -p "$NV2/runs/g9p4-parked"; echo parked > "$NV2/runs/g9p4-parked/PARKED.txt"\n' > "$SE/park-p4.sh"
    printf '#!/bin/bash\nk=$1; echo "read $k" >> "$NV2/runs/g9e/stub.calls"; R=$NV2/runs/g9e/r$k; mkdir -p "$R"; v=CONTINUE; [ "$k" = 250 ] && v=ROBUST\nprintf "hdr\\nE1@%%sM VERDICT %%s\\n" "$k" "$v" > "$R/READING.txt"; printf "hdr\\nFINAL: %%s\\n" "$v" > "$NV2/runs/g9e/READING.txt"; touch "$R/DONE"; [ "$v" = CONTINUE ] || touch "$NV2/runs/g9e/DONE"\n' > "$SE/read.sh"
    chmod a-w "$SE"/*.sh; printf '1,455,040,000\n1,460,081,000\n1,605,050,000\n1,610,090,000\n' > "$M/save-steps-1175374"
    rm -f "$M/sbatch.log" "$G/restart"
    MOCK_HOLD_ROUNDS=1 MOCK_PARK_ROUNDS=1 MOCK_EXTRA_ENV="NV2_G9E_INIT_MD5=$(md5sum < "$GI" | cut -c1-8) NV2_G9E_PARK_MD5=$(md5sum < "$SE/park-p4.sh" | cut -c1-8) NV2_G9E_READ_MD5=$(md5sum < "$SE/read.sh" | cut -c1-8)" job g9 1175374 10 a40 ws-netv2-g9 > /dev/null
    c9=$(grep "ws-netv2-1175374/" "$M/calls.log"); sed 's/^/    /' <<< "$c9" | cut -c1-200; L9=$T/ws-netv2-1175374/run; E9=$M/nv2/runs/g9e
    expect "S19 4 trainer calls; the job ends in the finished HOLD (mock cap: exit 3)" "^4 exit 3$" "$(cnt ws-netv2-1175374 "$M/calls.log") $(job_rc 1175374)"
    expect "S19 call 1: G9E from cd@1350, wrapper, November sets" "^call resume=$GI \[cd1350\] run_dir=$L9/G9E pool_dir=$L9/G9E/pool .*anchor=$P1 ps=$NOV $PP24" "$(sed -n 1p <<< "$c9")"
    expect "S19 call 2: G9Ecd100 from the branch point (G9E's save at the cap)" "^call resume=$E9/cd100/branch-point-01455040000.pt \[trained-from:cd1350\] run_dir=$L9/G9Ecd100 " "$(sed -n 2p <<< "$c9")"
    expect "S19 call 3: G9E on from its own save at the 100M cap" "^call resume=$L9/G9E/final_model.pt \[trained-from:cd1350\] run_dir=$L9/G9E " "$(sed -n 3p <<< "$c9")"
    expect "S19 call 4: G9Ecd250 from the 250M branch point" "^call resume=$E9/cd250/branch-point-01605050000.pt \[trained-from:trained-from:cd1350\] run_dir=$L9/G9Ecd250 " "$(sed -n 4p <<< "$c9")"
    a9=$(grep '^1175374 ' "$M/argv.log")
    expect "S19 argv 1: the exploiter recipe, --total-steps S0 + 100M" "^1175374 -u -m porygon.v2.train .* --total-steps 1455036714 --max-hours 70.5 --no-bf16 --n-epochs 3 --batch-size 4096 --lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.35 --guard-ptop-max 0.6 --kl-anchor $M/nv2/runs/pretrain/P3-g9-s2.pt --kl-anchor-coef 0 .* --frozen $GI --p-frozen 1.0 --p-baseline 0 --collector fastenv --fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 0 --pool-battles 16 --run-dir $L9/G9E --pool-dir $L9/G9E/pool --snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --snapshot-every 25000000 --resume $GI$" "$(sed -n 1p <<< "$a9")"
    expect "S19 argv 2: cooldown 5M cosine x0.1 from the branch step" " --total-steps 1460040000 .* --cooldown-start 1455040000 --final-lr-frac 0.1 --resume $E9/cd100/branch-point-01455040000.pt$" "$(sed -n 2p <<< "$a9")"
    expect "S19 argv 3 / 4: cap S0 + 250M, then its cooldown" "--total-steps 1605036714 .* --resume $L9/G9E/final_model.pt \| .*--total-steps 1610050000 .* --cooldown-start 1605050000 --final-lr-frac 0.1 --resume $E9/cd250/branch-point-01605050000.pt$" "$(sed -n 3p <<< "$a9") | $(sed -n 4p <<< "$a9")"
    expect "S19 stubs: one park, the two readings in order; finals published r--; DONE + HOLD; one sbatch" "^park 1175374 read 100 read 250 \| r-- r-- \| DONE HOLD \| 1$" "$(tr '\n' ' ' < "$E9/stub.calls")| $(stat -c %A "$E9"/cd100/final/01460081000.pt | cut -c2-4) $(stat -c %A "$E9"/cd250/final/01610090000.pt | cut -c2-4) | $([ -f "$E9/DONE" ] && echo DONE) $([ -f "$G/HOLD" ] && echo HOLD) | $(grep -c . "$M/sbatch.log" 2>/dev/null || echo 0)"
    expect "S19 train.log: the finished HOLD round" "MODE g9e is finished \(FINAL: ROBUST\): the A40 is for main, round 1: HOLD" "$(cat "$G"/*-1175374/train.log 2>/dev/null; cat "$L9/train.log" 2>/dev/null)"
    chmod -R u+w "$M/nv2/runs/g9e" "$M/nv2/scripts/netv2_g9e" "$M/nv2/runs/g9q-cd1350"; rm -rf "$M/nv2/runs/g9e" "$M/nv2/scripts/netv2_g9e" "$M/nv2/runs/g9q-cd1350" "$M/nv2/runs/g9p4-parked" "$M/save-steps-1175374" "$G/HOLD" "$G/restart"
  fi
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
