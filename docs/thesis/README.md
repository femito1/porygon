# Thesis record (Porygon)

One place for everything the thesis will cite: compute, runs, results (positive and negative), datasets, timeline,
methods, and how AI assistance was used. Started 2026-11-17; kept up to date by the orchestrator as results land.

## Files
| file | what |
|---|---|
| `compute.md` | GPU-hours / CPU-hours by GPU type, by purpose (training, benchmarks, data, diagnostics), by month; cloud (RunPod) and laptop use; totals |
| `runs.md` | every training run / lineage: hardware, start-end, steps, throughput, wall time, init, recipe, outcome |
| `results.md` | every evaluation result: question, arms, harness, n, estimate + 95% interval, pre-registration hash, verdict, file path; negative and null results included |
| `datasets.md` | human replay corpus and every frozen dataset (size, rating filter, months, splits), self-play data volumes |
| `timeline.md` | dated chronology: phases, decisions, pivots, incidents |
| `methods.md` | network architecture(s), observation spec, PPO recipe and hyperparameters, search algorithms, evaluation methodology (duplicate pairs, sequential tests, pre-registration) |
| `codebase.md` | code size by area, tests, languages, external components and licences |
| `ai-use.md` | how AI agents were used (for the academic-integrity declaration): roles, sessions, token volumes |
| `data/` | machine-readable tables behind the above (CSV / JSON) for figures |

## Conventions
- **Every number carries its provenance:** `[M]` measured (with the source file, job id or command), `[D]` derived
  from measured numbers (show the arithmetic), `[E]` estimate (state the basis and a range). Never mix them silently.
- Intervals are 95% unless stated; win rates from duplicate pairs give pair-mean intervals (see `methods.md`).
- Dates and times are UTC. Step counts are environment steps (decisions) unless stated.
- Cluster paths are written in full (`/vol/bitbucket/fr1226/...`); large raw data stays on the cluster, summaries here.
- Append, don't rewrite history: when a number is corrected, keep the old one struck through with the reason.

## Upkeep (orchestrator)
- Each NEW pre-registered result at a check-in ⇒ one row in `results.md` (fact-checked from the file).
- Each new run, restart, hardware move or incident ⇒ `runs.md` / `timeline.md`.
- Weekly (or before a write-up) ⇒ re-run the compute report (`scripts/thesis/`) and refresh `compute.md`.
