# ONE self-resubmitting tracker job for all runs (PLAN rule: at most one wsp job on long + training at a time,
# because every dove job holds a GPU and the per-user GPU cap is shared with the benchmark slot). Each pass runs the
# per-run configs in v2diag/conf/ sequentially (v2diag_watch_any.sh in NO_CHAIN mode), then ends. The next pass is
# queued first, held until this job ends (afterany) and >= 20 min from now, and only if no other ws-wsp-v2diag job is
# pending. Stop everything: touch v2diag/STOP_ALL. Since 2026-11-09 ~12:14 long = dove03-04 only; training = dove01-04
# (same QoS tempgpu and per-user caps, higher priority tier): submit to training (the submit plugin refuses a list).
[ -e v2diag/STOP_ALL ] && { echo "STOP_ALL"; exit 0; }
others=$(squeue -h -u $USER -n ws-wsp-v2diag -t PENDING -o %i | grep -v "^${SLURM_JOB_ID}$" | wc -l)
if squeue -h -u $USER -o %j | grep -q -E "^ws-netv2-(g9|r1)" && [ "$others" = "0" ]; then
  sbatch --begin=now+20minutes --dependency=afterany:$SLURM_JOB_ID --partition=training scripts/wsp/v2diag_run.sbatch \
    scripts/wsp/v2diag_watch_all.sh | tail -1
else
  echo "not resubmitting (netv2 jobs gone, or $others other wsp tracker job(s) pending)"
fi
# one-off analyses ride along in the same job (one wsp job on long): v2diag/oneshot/*.sh run once, then are moved
for o in $(ls v2diag/oneshot/*.sh 2>/dev/null); do
  echo "#### oneshot $o $(date -u +%H:%M)"; mkdir -p v2diag/oneshot/done; mv $o v2diag/oneshot/done/; bash v2diag/oneshot/done/$(basename $o)
done
# the per-run watcher also runs as a /tmp copy: bash reads scripts incrementally, and replacing a file another NFS
# client has open gives that reader a stale handle
# G9p first since the 12:40 GPU swap (it is the A30 run now, with a post-swap sharpening watch item)
for c in v2diag/conf/g9p.conf v2diag/conf/g9s.conf v2diag/conf/r1b.conf; do
  echo "#### $c $(date -u +%H:%M)"
  A=$(mktemp /tmp/v2diag-any-XXXX.sh); cp scripts/wsp/v2diag_watch_any.sh $A
  NO_CHAIN=1 bash $A $c; rm -f $A
done
