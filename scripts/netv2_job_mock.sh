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
    if x in ("--resume", "--run-dir", "--pool-dir") and i + 1 < len(a):
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
          f"frozen={n(pooldir + '/frozen')} best={yn(rundir + '/best_model.pt')} ema={yn(rundir + '/ema_model.pt')}\n"); log.flush()
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
      NODE_BIN=/nonexistent ${MOCK_END:+SLURM_JOB_END_TIME=$MOCK_END NV2_DEADLINE_POLL_S=1 NV2_DEADLINE_PAT=bin/python.-u.-m.porygon.v2.train} MOCK_WAIT="${MOCK_WAIT:-}" \
      setsid -w bash -c 'echo $$ > "$MOCK/sid"; exec bash "$MOCK/nv2/scripts/slurm_train_v2.sbatch" "$1"' _ "$run" > "$M/job-$id.out" 2>&1 )
  local rc=$? sid; sid=$(cat "$M/sid" 2>/dev/null)
  if [ -n "$sid" ]; then /bin/sleep 1; pkill -s "$sid" 2>/dev/null; fi  # leftovers of this mock job only (its own session)
  echo "  exit $rc ; job log (trimmed):"; grep -v -E "pre-read|find: |^$" "$M/job-$id.out" | cut -c1-200 | sed 's/^/    | /' | head -n 12
}
calls() { c=$(cat "$M/calls.log" 2>/dev/null); printf '  trainer calls:\n%s\n' "$(sed 's/^/    /' <<< "$c")"; }
stubs; export MOCK=$M
T=$M/tmp
echo "== S1: r1 fresh job after 718701, post-swap (MODE g9p = G9p on the A30), restart flag armed"
tree post; job r1 900001 30 a30 ws-netv2-r1 restart; calls
expect "S1 resumes the newest mirrored G9p file (post-swap segment)" "^call resume=.*/20261108-220401-718701/G9p/final_model.pt \[G9p-postswap-final\]" "$(sed -n 1p <<< "$c")"
expect "S1 G9p lineage dirs" "run_dir=$T/ws-netv2-900001/run/G9p pool_dir=$T/ws-netv2-900001/run/G9p/pool" "$(sed -n 1p <<< "$c")"
expect "S1 pool (3) + frozen (2) + best/ema re-seeded" "pool=3 frozen=2 best=y ema=y" "$(sed -n 1p <<< "$c")"
expect "S1 restart in place relaunches from the lineage's own save" "^call resume=.*/run/G9p/final_model.pt \[trained-from:G9p-postswap-final\]" "$(sed -n 2p <<< "$c")"
expect "S1 G9p recipe on the A30 (KL anchor, 28 workers, cache 96, stacked-opp)" "kl-anchor-coef 0.1 .*fastenv-workers 28 .*--snapshot-cache 96 --stacked-opp" "$(cat "$M/job-900001.out")"
expect "S1 no resubmission (chain_left 0)" "^$" "$(cat "$M/sbatch.log" 2>/dev/null)"
echo "== S2: g9 fresh job after 741849, post-swap (MODE g9s = G9s on the A16), restart flag armed, chain_left 1"
tree post; echo 1 > "$M/nv2/runs/g9/chain_left"; job g9 900002 10 a16 ws-netv2-g9 restart; calls
expect "S2 resumes the newest mirrored G9s file (post-swap, not the pre-swap A30 segment)" "^call resume=.*/20261109-021316-741849/G9s/final_model.pt \[G9s-postswap-final\]" "$(sed -n 1p <<< "$c")"
expect "S2 G9s lineage dirs" "run_dir=$T/ws-netv2-900002/run/G9s pool_dir=$T/ws-netv2-900002/run/G9s/pool" "$(sed -n 1p <<< "$c")"
expect "S2 pool (3) + frozen (2) + best/ema re-seeded" "pool=3 frozen=2 best=y ema=y" "$(sed -n 1p <<< "$c")"
expect "S2 restart in place relaunches from the lineage's own save" "^call resume=.*/run/G9s/final_model.pt \[trained-from:G9s-postswap-final\]" "$(sed -n 2p <<< "$c")"
expect "S2 G9s recipe on the A16 (8 workers, cache 32, no KL anchor)" "guard-anchor-drop 0.10 .*fastenv-workers 8 .*--snapshot-cache 32 " "$(cat "$M/job-900002.out")"
s=$(cat "$M/sbatch.log" 2>/dev/null); echo "  sbatch: ${s:-none}"
if [ -z "$OLD" ]; then
  expect "S2 resubmission carries name / a16 / gres / 10 CPUs / 64G" "--job-name=ws-netv2-g9 --partition=a16 --gres=gpu:1 --cpus-per-task=10 --mem=65536M --time=72:00:00 scripts/slurm_train_v2.sbatch g9" "$s"
else
  echo "  (old spooled script: its bare chain sbatch cannot carry flags; keep chain_left 0 and queue the successor by hand)"
fi
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
