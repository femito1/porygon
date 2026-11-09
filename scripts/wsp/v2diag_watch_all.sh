# ONE self-resubmitting tracker job for all runs (PLAN rule: at most one wsp job on the long partition at a time,
# because every long job holds a GPU and the per-user GPU cap is shared with the benchmark slot). Each pass runs the
# per-run configs in v2diag/conf/ sequentially (v2diag_watch_any.sh in NO_CHAIN mode), then ends. The next pass is
# queued first, held until this job ends (afterany) and >= 20 min from now, and only if no other ws-wsp-v2diag job is
# pending. Stop everything: touch v2diag/STOP_ALL.
[ -e v2diag/STOP_ALL ] && { echo "STOP_ALL"; exit 0; }
others=$(squeue -h -u $USER -n ws-wsp-v2diag -t PENDING -o %i | grep -v "^${SLURM_JOB_ID}$" | wc -l)
if squeue -h -u $USER -o %j | grep -q -E "^ws-netv2-(g9|r1)" && [ "$others" = "0" ]; then
  sbatch --begin=now+20minutes --dependency=afterany:$SLURM_JOB_ID --partition=long scripts/wsp/v2diag_run.sbatch \
    scripts/wsp/v2diag_watch_all.sh | tail -1
else
  echo "not resubmitting (netv2 jobs gone, or $others other wsp tracker job(s) pending)"
fi
for c in v2diag/conf/g9s.conf v2diag/conf/g9p.conf v2diag/conf/r1b.conf; do
  echo "#### $c $(date -u +%H:%M)"
  NO_CHAIN=1 bash scripts/wsp/v2diag_watch_any.sh $c
done
