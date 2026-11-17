# Results (every evaluation the project produced)

Compiled 2026-11-17 from a sweep of the cluster (`result.txt`, `RESULT*`, `RESULTS.txt`, `VERDICT*`, `READING.txt`,
`OVERALL.txt`, `analysis.txt`, `SKIP.txt`, `summary.txt` under `/vol/bitbucket/fr1226/gloria/runs` and every
`/vol/bitbucket/fr1226/porygon-ws-*`; list in `/vol/bitbucket/fr1226/porygon-thesis/resultfiles.txt`, 222 files, and
`aux-files.txt`), plus results quoted in `README.md` (and its git history), the stream HANDOFFs, `docs/netsearch/`,
`docs/paper/paper.tex` and memory. Machine-readable: `data/results.csv` (one row per result, same ids).

**Totals:** 238 rows, 71 pre-registered, 76 negative / null / inconclusive (26 negative, 45 null, 5 inconclusive). "Pre-registered" = the row names a PREREG file written before its first game (md5 as on the
cluster on 2026-11-17; `/vol/bitbucket/fr1226/porygon-thesis/prereg-md5.txt`). Rows marked `info` are measurements
without a decision rule, constituents of pooled rows, or diagnostics.

**Conventions.** `[M]` measured, read from the named file; `[D]` derived (pooled or rearranged from measured
records, arithmetic shown); `[E]` estimate or assumption. Win rates are arm A's, in percent. Interval types:
- *pair-mean 95%* = mean of duplicate-pair scores ± 1.96·sd/√pairs (`methods.md` §Evaluation);
- *MUE* = median-unbiased estimate with its 95% stage-wise CI from a group-sequential test (Lan–DeMets; the naive mean is
  biased by up to ±0.9 points); these are internal go/no-go tests (user rule, 2026-11-09);
- *binomial per game* = p ± 1.96·√(p(1−p)/n) (early runs before the duplicate server, 2026-10-27).
Raw networks play at temperature 0.5 unless stated. "Pure gen9 search" = `search:none`, 4 × 1000 ms, stock evaluation.
Foul Play = pmariglia/foul-play at its default (2 sequential 1 s searches on one core; commit 8f541f8 for every
porygon-ws run). Our search spends 4 core-seconds per move, Foul Play 2.

## Headline standing (2026-11-17 12:00 UTC)

| generation | bot | vs Foul Play (95%, games) | vs our pure gen9 search | id |
|---|---|---|---|---|
| gen4 | search + fitted leaf eval (eval_r1), v1 prior 0.25, 4 × 1 s | **59.19 [56.95, 61.43]** (1599) [D] | — | fp4-04 |
| gen4 | same, compute matched: ours 2 × 1 s / Foul Play 4 × 1 s | 57.84 [55.67, 60.02] / 58.36 [56.14, 60.59] [D] | — | fp4-05/06 |
| gen4 | search, stock eval (baseline) | 54.50 [52.96, 56.04] (3198) [D] | — | fp4-02 |
| gen4 | raw v2 network R1b-F@300M, no search | 55.6 [51.5, 59.7] (400, not confirmed at 1000) [M] | — | v2g4-10 |
| gen4 | raw v1 network @600M (T 0.5) | 46.2 [42.4, 50.0] (600) [M] | — | v1-25 |
| gen9 | raw v2 network G9q-cd@605M | **58.8 [55.9, 61.7]** (1000, CONFIRMED) [M] | 52.5 [49.6, 55.4] (1000, not confirmed) | g9q-17/16 |
| gen9 | raw v2 network G9q-cd@805M (current deployment net) | **57.8 [55.0, 60.6]** (1000, CONFIRMED) [M]; 56.90 [53.94, 59.86] on the Stage 3 harness | **55.5 [52.7, 58.3]** (1000, Milestone 1 CONFIRMED) | g9q-24/23, ns-19 |
| gen9 | lookahead2 on G9q-cd@805M (δ 0.005, K 8) | **58.20 [55.29, 61.11]** (1000) [M]; paired gain over its raw net +1.30 [−2.71, +5.31] NOT SHOWN | — (vs its raw net on cd@605: 53.17 [50.59, 55.73], Bet 1 GO) | ns-18/20/14 |
| gen9 | hybrid H(δ 0.0212) + G9p-cd@104.71M (benchmark bot since 11-11) | 56.2 [53.4, 59.0] (1000) [M] | 55.0 [52.0, 58.0] (1000) | ns-07/05 |
| gen9 | pure engine search (stock eval) | 51.6 [48.8, 54.4] (1000, Stage 3 harness) [M]; 51.72 [49.36, 54.08] (1508, gloria harness) [D] | — | ns-08, g9s-01 |

Context, not ours: MissingNo.Zero's raw net beats Foul Play 61.6% (gen4) / 66.7% (gen9); a Jaxcalibur-level bot ≈ 75%
(`.orchestra/research/REPORT.md`). **Pending at compilation:** LA2-H (lookahead2 vs the benchmark hybrid, prereg md5
ee8f1bd6, `$F/runs/la2-h/VERDICT.txt`), fpcal (`/vol/bitbucket/fr1226/porygon-ws-wsp/fpcal/READING.txt`, md5
5a68f926), scale gate T1 (deepen D4 vs D1, md5 9fd3960d, `$F/runs/scale-gate/VERDICT.txt`), G9-P4 later readings.
`$F` = `/vol/bitbucket/fr1226/porygon-ws-fastenv`, `$C` = `/vol/bitbucket/fr1226/porygon-ws-netv2`.

## Results by theme

### Benchmarks against Foul Play (all generations) (71 rows)

| id | date | question | A vs B | harness | games / pairs | A's win % [95%] | prereg (md5) | verdict | src |
|---|---|---|---|---|---|---|---|---|---|
| v1-06 | 2026-10-21 | search (v1@91M prior) vs Foul Play gen4 | SearchPlayer 4x1 s, v1@91M vs Foul Play gen4 | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 204 | 49.0 [M] | - | ~even (baseline) | `README.md @ 7020fa8 (48% when Foul Play searched on 4 cores)` |
| v1-07 | 2026-10-21 | raw v1 net vs Foul Play gen4 | raw v1@91M vs Foul Play gen4 | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 200 | 16.5 [M] | - | raw net far weaker | `README.md @ 7020fa8` |
| fp4-01 | 2026-10-27 | default search pooled (5 early runs, 400M-500M prior, no duplicate server yet) | SearchPlayer 4x1 s, prior 0.25, stock eval vs Foul Play gen4 | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 1992 | 52.6 [50.4, 54.8] [D] | - | 4x Foul Play's compute; ~even (baseline) | `jobs 300964 301118 301192 301389 301388: 1048-944 (scan); README @ 02c5e57` |
| fp4-10 | 2026-10-27 | oracle revealed, old inference, stock eval | search + oracle revealed vs Foul Play gen4 | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 400 / 200 | 61.25 [56.83, 65.67] [D] | - | old inference: +8.8 | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); job 302020` |
| fp4-11 | 2026-10-27 | oracle all, old inference, stock eval | search + oracle all vs Foul Play gen4 | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 399 / 200 | 65.38 [61.26, 69.49] [D] | - | old inference: +12.9 | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); job 302021` |
| fp4-09 | 2026-10-31 | root PUCT c 2 pooled (2 runs) | search, prior 0, c_puct 2 vs Foul Play gen4 | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 798 / 400 | 55.0 [51.76, 58.24] [D] | - | no change from default | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); jobs 303471 304133` |
| fp4-02 | 2026-11-01 | stock-eval search baseline, new inference (8 runs) | search 4x1 s, prior 0.25, stock eval vs Foul Play gen4 | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 3198 / 1600 | 54.5 [52.96, 56.04] [D] | - | baseline | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); jobs 302802 303464-8 307044-5` |
| fp4-03 | 2026-11-03 | stock weights rescaled x2 | search, stock x2 vs Foul Play gen4 | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 1594 / 800 | 54.47 [52.17, 56.76] [D] | - | rescaling alone: no gain | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); jobs 309102-5` |
| fp4-04 | 2026-11-03 | fitted leaf evaluation (eval_r1 = fe x2) | search, fitted eval vs Foul Play gen4 | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 1599 / 800 | 59.19 [56.95, 61.43] [D] | - | HEADLINE gen4: +4.7 +- 2.7 over stock | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); jobs 308845-8 (README first reported 59.2 +- 1.1 @ c6b56b8)` |
| fp4-05 | 2026-11-06 | fitted eval, compute matched: ours 2x1 s | search 2x1 s, fitted vs Foul Play gen4 default | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 1597 / 800 | 57.84 [55.67, 60.02] [D] | - | gain holds at Foul Play's compute | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); jobs 380872/4/6/8` |
| fp4-06 | 2026-11-06 | fitted eval, Foul Play given 4x1 s (--fp-parallelism 2) | search 4x1 s, fitted vs Foul Play 4x1 s | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 1605 / 804 | 58.36 [56.14, 60.59] [D] | - | doubling Foul Play does not close the gap | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); jobs 380873/5/7/9` |
| fp4-07 | 2026-11-08 | oracle: true sets of revealed mons (fitted eval) | search + oracle revealed vs Foul Play gen4 | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 800 / 400 | 62.12 [58.88, 65.37] [D] | - | inference ceiling +2.9 +- 3.9 | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); jobs 592909 592911` |
| fp4-08 | 2026-11-08 | oracle: whole true team (fitted eval) | search + oracle all vs Foul Play gen4 | Foul Play gen4 (default: 2 states x 1 s, 1 core), gloria slurm_foulplay.sbatch | 799 / 400 | 64.44 [61.39, 67.48] [D] | - | +5.2 +- 3.7 | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); jobs 592910 592912` |
| v1-24 | 2026-11-08 | raw v1 net vs Foul Play gen4 (default T) | raw v1@600M (301120) vs Foul Play gen4 (8f541f8) | netv2 foulplay-net harness | 600 / 300 | 44.8 [41.1, 48.5] [M] | - | raw v1 below Foul Play | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/foulplay-net-20261108-171126-688719/result.txt` |
| v1-25 | 2026-11-08 | raw v1 net vs Foul Play gen4 at T 0.5 | raw v1@600M T 0.5 vs Foul Play gen4 (8f541f8) | netv2 foulplay-net harness | 600 / 300 | 46.2 [42.4, 50.0] [M] | - | raw v1 below Foul Play | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/foulplay-net-20261108-194529-701384/result.txt` |
| v2g9-16 | 2026-11-10 | G9p@100M vs Foul Play gen9 | G9p@100M T 0.5 vs Foul Play gen9 | Foul Play gen9 (8f541f8, ms 1000, 1 core), netv2 fp9 harness | 400 / 200 | 47.0 [42.1, 51.9] [M] | - | info | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/foulplay-net-20261110-042109-843443/result.txt` |
| v2g9-17 | 2026-11-10 | G9p-cd vs Foul Play gen9 | G9p-cd T 0.5 vs Foul Play gen9 | Foul Play gen9 (8f541f8, ms 1000, 1 core), netv2 fp9 harness | 400 / 200 | 46.2 [41.6, 50.8] [M] | - | info | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/foulplay-net-20261110-163933-943190/result.txt` |
| ns-07 | 2026-11-11 | Stage 3b: hybrid H(0.0212) vs Foul Play gen9 | H(0.0212) + G9p-cd vs Foul Play gen9 (8f541f8) | fastenv Stage 3 slot harness (11 engine procs/side, 3 concurrent) vs Foul Play (NETSEARCH-3B-HARNESS) | 1000 / 500 | 56.2 [53.4, 59.0] [M] | docs/netsearch/NETSEARCH-PREREG.md + NETSEARCH-3B-HARNESS.md (5b2d8f16 (harness)) | hybrid = gen9 benchmark bot | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/netsearch-stage3b-20261111-025822-1026613/result.txt` |
| ns-08 | 2026-11-11 | Stage 3b: pure gen9 search vs Foul Play gen9 | pure gen9 search vs Foul Play gen9 | fastenv Stage 3 slot harness (11 engine procs/side, 3 concurrent) vs Foul Play | 1000 / 500 | 51.6 [48.8, 54.4] [M] | docs/netsearch/NETSEARCH-PREREG.md (db59ec88 (file now incl. results)) | Q2: H - search +4.6 +- 3.9 [+0.7, +8.5] SUCCESS | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/netsearch-stage3b-20261111-025822-1026613/result.txt` |
| g9q-14 | 2026-11-14 | raw G9q-cd@605M vs Foul Play gen9 (400 g) | G9q-cd@605M T 0.5 vs Foul Play gen9 | Foul Play gen9 (8f541f8, ms 1000, 1 core), netv2 fp9 harness | 400 / 200 | 57.5 [53.1, 61.9] [M] | rawfp/PREREG.txt (f0a508e3) | first raw net > Foul Play | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/rawfp/RESULTS.txt` |
| v2g4-10 | 2026-11-14 | raw R1b-F@300M vs Foul Play gen4 | raw R1b-F@300M T 0.5 vs Foul Play gen4 (8f541f8) | netv2 foulplay-net harness | 400 / 200 | 55.6 [51.5, 59.7] [M] | rawfp/PREREG.txt (f0a508e3) | raw gen4 net beats Foul Play (400 g, not confirmed at 1000) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/rawfp/RESULTS.txt` |
| g9q-17 | 2026-11-15 | raw G9q-cd@605M vs Foul Play gen9 (1000 g) | G9q-cd@605M T 0.5 vs Foul Play gen9 | Foul Play gen9 (8f541f8, ms 1000, 1 core), netv2 fp9 harness | 1000 / 500 | 58.8 [55.9, 61.7] [M] | conf605/PREREG.txt (517b679c) | CONFIRMED (lower > 50%) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/conf605/RESULTS.txt` |
| g9q-24 | 2026-11-16 | raw G9q-cd@805M vs Foul Play gen9 (1000 g, seed 8800222) | G9q-cd@805M T 0.5 vs Foul Play gen9 | Foul Play gen9 (8f541f8, ms 1000, 1 core), netv2 fp9 harness | 1000 / 500 | 57.8 [55.0, 60.6] [M] | conf800/PREREG.txt (c6a38724) | CONFIRMED (= cd@605 within noise) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/conf800/RESULTS.txt` |
| ns-18 | 2026-11-17 | LA2-FP: lookahead2(cd@805, d 0.005) vs Foul Play gen9 | lookahead2 on G9q-cd@805M vs Foul Play gen9 (8f541f8) | fastenv Stage 3 slot harness (11 engine procs/side, 3 concurrent) vs Foul Play | 1000 / 500 | 58.2 [55.29, 61.11] [M] | LA2-FP-PREREG.md (1173284f) | Q1 headline | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/la2-fp/VERDICT.txt` |
| ns-19 | 2026-11-17 | LA2-FP arm 2: raw cd@805 vs Foul Play gen9 (same pairs) | raw G9q-cd@805M T 0.5 vs Foul Play gen9 | fastenv Stage 3 slot harness (11 engine procs/side, 3 concurrent) vs Foul Play | 1000 / 500 | 56.9 [53.94, 59.86] [M] | LA2-FP-PREREG.md (1173284f) | positive | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/la2-fp/VERDICT.txt` |

Per-run constituents (46 gloria runs; each [M] from its `analysis.txt` unless marked [D]; the arm's network is not logged, see the note below):

| job | date | config (non-default `SearchPlayer` options) | games / pairs | win % [95%] | outcome |
|---|---|---|---|---|---|
| 300964 | 2026-10-25 | v1 prior (net not logged), default | 398 | 55.0 [50.1, 59.9] [D] | info |
| 301118 | 2026-10-26 | v1 prior (net not logged), default | 398 | 51.8 [46.9, 56.7] [D] | info |
| 301119 | 2026-10-26 | v1 prior (net not logged), value_weight=0.5 | 398 | 49.7 [44.8, 54.6] [D] | null: no gain |
| 301192 | 2026-10-26 | v1 prior (net not logged), default | 400 | 51.2 [46.3, 56.1] [D] | info |
| 301388 | 2026-10-26 | v1 prior (net not logged), default | 400 / 200 | 54.4 [49.8, 59.0] [M] | info |
| 301389 | 2026-10-26 | v1 prior (net not logged), default | 400 / 200 | 50.6 [45.9, 55.3] [M] | info |
| 301735 | 2026-10-27 | v1 prior (net not logged), prior_weight=1.0 | 400 / 200 | 49.6 [45.3, 53.9] [M] | null: no gain |
| 302020 | 2026-10-27 | v1 prior (net not logged), oracle=revealed | 400 / 200 | 61.3 [56.9, 65.7] [M] | info |
| 302021 | 2026-10-27 | v1 prior (net not logged), oracle=all | 400 / 200 | 65.4 [61.3, 69.5] [M] | info |
| 302022 | 2026-10-27 | v1 prior (net not logged), prior_weight=0.5 oracle=None | 400 / 200 | 54.2 [49.7, 58.7] [M] | null: no gain |
| 302802 | 2026-10-28 | v1 prior (net not logged), oracle=None | 400 / 200 | 50.7 [46.5, 54.9] [M] | info |
| 302877 | 2026-10-28 | v1 prior (net not logged), samples=8 oracle=None | 400 / 200 | 53.6 [49.3, 57.9] [M] | null: no gain |
| 303464 | 2026-10-29 | v1 prior (net not logged), oracle=None | 400 / 200 | 52.2 [47.9, 56.5] [M] | info |
| 303465 | 2026-10-29 | v1 prior (net not logged), oracle=None | 400 / 200 | 57.5 [53.2, 61.8] [M] | info |
| 303466 | 2026-10-29 | v1 prior (net not logged), oracle=None | 400 / 200 | 50.7 [46.5, 54.9] [M] | info |
| 303467 | 2026-10-29 | v1 prior (net not logged), oracle=None | 400 / 200 | 56.2 [51.9, 60.5] [M] | info |
| 303468 | 2026-10-29 | v1 prior (net not logged), oracle=None | 400 / 200 | 57.8 [53.0, 62.6] [M] | info |
| 303471 | 2026-10-29 | v1 prior (net not logged), prior_weight=0.0 c_puct=2.0 oracle=None | 400 / 200 | 56.4 [51.8, 61.0] [M] | info |
| 303472 | 2026-10-29 | v1 prior (net not logged), prior_weight=0.0 c_puct=8.0 oracle=None | 400 / 200 | 53.2 [49.4, 57.0] [M] | null: no gain |
| 304133 | 2026-10-30 | v1 prior (net not logged), prior_weight=0.0 c_puct=2.0 oracle=None | 400 / 200 | 53.6 [49.0, 58.2] [M] | info |
| 307044 | 2026-10-31 | v1 prior (net not logged), oracle=None | 400 / 200 | 53.0 [48.7, 57.3] [M] | info |
| 307045 | 2026-10-31 | v1 prior (net not logged), oracle=None | 400 / 200 | 57.8 [53.5, 62.1] [M] | info |
| 307072 | 2026-10-31 | v1 prior (net not logged), root_game=sample oracle=None | 400 / 200 | 40.8 [36.3, 45.3] [M] | negative: -13 points: harmful |
| 307073 | 2026-10-31 | v1 prior (net not logged), root_game=max oracle=None | 400 / 200 | 52.9 [48.5, 57.3] [M] | null: no gain |
| 307076 | 2026-11-01 | v1 prior (net not logged), rm_gamma=0.1 oracle=None | 400 / 200 | 53.4 [49.1, 57.7] [M] | null: no gain |
| 307077 | 2026-11-01 | v1 prior (net not logged), root_game=sample rm_gamma=0.1 oracle=None | 400 / 200 | 41.8 [37.3, 46.3] [M] | negative: harmful |
| 308845 | 2026-11-03 | v1 prior (net not logged), eval_config=fe_x2.json oracle=None | 400 / 200 | 55.5 [50.9, 60.1] [M] | info |
| 308846 | 2026-11-03 | v1 prior (net not logged), eval_config=fe_x2.json oracle=None | 400 / 200 | 60.8 [56.1, 65.5] [M] | info |
| 308847 | 2026-11-03 | v1 prior (net not logged), eval_config=fe_x2.json oracle=None | 400 / 200 | 60.0 [55.6, 64.4] [M] | info |
| 308848 | 2026-11-03 | v1 prior (net not logged), eval_config=fe_x2.json oracle=None | 400 / 200 | 60.5 [56.3, 64.7] [M] | info |
| 309102 | 2026-11-03 | v1 prior (net not logged), eval_config=stock_x2.json oracle=None | 400 / 200 | 56.1 [51.7, 60.5] [M] | info |
| 309103 | 2026-11-03 | v1 prior (net not logged), eval_config=stock_x2.json oracle=None | 400 / 200 | 52.6 [48.0, 57.2] [M] | info |
| 309104 | 2026-11-03 | v1 prior (net not logged), eval_config=stock_x2.json oracle=None | 400 / 200 | 54.6 [49.9, 59.3] [M] | info |
| 309105 | 2026-11-03 | v1 prior (net not logged), eval_config=stock_x2.json oracle=None | 400 / 200 | 54.5 [49.8, 59.2] [M] | info |
| 380872 | 2026-11-06 | v1 prior (net not logged), samples=2 eval_config=eval_r1.json oracle=None | 400 / 200 | 54.6 [50.3, 58.9] [M] | info |
| 380873 | 2026-11-06 | v1 prior (net not logged), eval_config=eval_r1.json oracle=None | 402 / 201 | 61.4 [57.0, 65.8] [M] | info |
| 380874 | 2026-11-06 | v1 prior (net not logged), samples=2 eval_config=eval_r1.json oracle=None | 400 / 200 | 59.2 [54.7, 63.7] [M] | info |
| 380875 | 2026-11-06 | v1 prior (net not logged), eval_config=eval_r1.json oracle=None | 402 / 201 | 57.2 [52.8, 61.6] [M] | info |
| 380876 | 2026-11-06 | v1 prior (net not logged), samples=2 eval_config=eval_r1.json oracle=None | 400 / 200 | 56.8 [52.7, 60.9] [M] | info |
| 380877 | 2026-11-06 | v1 prior (net not logged), eval_config=eval_r1.json oracle=None | 402 / 201 | 58.7 [54.0, 63.4] [M] | info |
| 380878 | 2026-11-06 | v1 prior (net not logged), samples=2 eval_config=eval_r1.json oracle=None | 400 / 200 | 60.8 [56.3, 65.3] [M] | info |
| 380879 | 2026-11-06 | v1 prior (net not logged), eval_config=eval_r1.json oracle=None | 402 / 201 | 56.1 [51.8, 60.4] [M] | info |
| 592909 | 2026-11-08 | v1 prior (net not logged), eval_config=eval_r1.json oracle=revealed | 400 / 200 | 60.5 [55.9, 65.1] [M] | info |
| 592910 | 2026-11-08 | v1 prior (net not logged), eval_config=eval_r1.json oracle=all | 400 / 200 | 63.0 [58.6, 67.4] [M] | info |
| 592911 | 2026-11-08 | v1 prior (net not logged), eval_config=eval_r1.json oracle=revealed | 400 / 200 | 63.7 [59.1, 68.3] [M] | info |
| 592912 | 2026-11-08 | v1 prior (net not logged), eval_config=eval_r1.json oracle=all | 400 / 200 | 65.9 [61.7, 70.1] [M] | info |

### Gen4 search and the fitted leaf evaluation (engine and search-vs-search) (25 rows)

| id | date | question | A vs B | harness | games / pairs | A's win % [95%] | prereg (md5) | verdict | src |
|---|---|---|---|---|---|---|---|---|---|
| s4-01 | 2026-10-25 | one-turn value lookahead (v1 value) alone vs raw net | lookahead (v1@400M value) vs raw v1@400M | Showdown (slurm_search) | 300 | 53.3 [47.7, 58.9] [M] | - | not better than search | `README.md @ 1d4e2d6 (run dirs not kept)` |
| s4-02 | 2026-10-25 | value lookahead mixed 0.5 into the search vs the search | search + value_weight 0.5 vs search | Showdown | 300 | 53.7 [48.1, 59.3] [M] | - | no clear gain | `README.md @ 1d4e2d6` |
| s4-03 | 2026-10-25 | value lookahead replacing the engine search | lookahead only vs search | Showdown | 300 | 42.7 [37.1, 48.3] [M] | - | worse | `README.md @ 1d4e2d6` |
| s4-04 | 2026-10-27 | patched (durations) vs stock poke-engine, same search | search, patched engine vs search, stock engine (.venv-stock) | Showdown duplicate (slurm_search) | 400 / 200 | 49.1 [45.0, 53.2] [M] | - | neutral: a correctness fix | `/vol/bitbucket/fr1226/gloria/runs/search-20261027-010752-301447/analysis.txt` |
| s4-14a | 2026-11-01 | first leaf-eval fit arms vs stock (full) | evalfit_r1_full vs stock eval | engine self-play h2h | 5466 / 2733 | 50.02 [48.8, 51.2] [M] | - | exploratory scan; label mapping assumed = dir order [E] | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261101-183553-307921/summary.txt` |
| s4-14b | 2026-11-01 | first leaf-eval fit arms vs stock (full_abs) | evalfit_r1_full_abs vs stock eval | engine self-play h2h | 5370 / 2685 | 54.34 [53.1, 55.5] [M] | - | exploratory scan; label mapping assumed = dir order [E] | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261101-183553-307921/summary.txt` |
| s4-14c | 2026-11-01 | first leaf-eval fit arms vs stock (full_double) | evalfit_r1_full_double vs stock eval | engine self-play h2h | 5378 / 2689 | 46.08 [44.9, 47.3] [M] | - | exploratory scan; label mapping assumed = dir order [E] | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261101-183553-307921/summary.txt` |
| s4-14d | 2026-11-01 | first leaf-eval fit arms vs stock (full_fe) | evalfit_r1_full_fe vs stock eval | engine self-play h2h | 5402 / 2701 | 46.43 [45.2, 47.6] [M] | - | exploratory scan; label mapping assumed = dir order [E] | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261101-183553-307921/summary.txt` |
| s4-14e | 2026-11-01 | first leaf-eval fit arms vs stock (full_half) | evalfit_r1_full_half vs stock eval | engine self-play h2h | 5422 / 2711 | 33.01 [31.8, 34.2] [M] | - | exploratory scan; label mapping assumed = dir order [E] | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261101-183553-307921/summary.txt` |
| s4-14f | 2026-11-01 | first leaf-eval fit arms vs stock (stock_k) | evalfit_r1_stock_k vs stock eval | engine self-play h2h | 5470 / 2735 | 59.87 [58.7, 61.1] [M] | - | exploratory scan; label mapping assumed = dir order [E] | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261101-183553-307921/summary.txt` |
| s4-08 | 2026-11-02 | fitted eval scale k in engine self-play (300k iterations) | fitted fe x2 vs stock eval | engine self-play h2h (gloria h2h) | 1848 / 864 | 58.1 [55.9, 60.3] [M] | - | k 2 chosen | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261102-120033-308347/summary.txt` |
| s4-09 | 2026-11-02 | fitted eval scale k (300k it.) | fitted fe x4 vs stock eval | engine self-play h2h | 1849 / 860 | 56.0 [53.8, 58.2] [M] | - | flat in k | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261102-120033-308347/summary.txt` |
| s4-10 | 2026-11-02 | fitted eval scale k (300k it.) | fitted fe x8 vs stock eval | engine self-play h2h | 1863 / 876 | 56.8 [54.6, 59.0] [M] | - | flat in k | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261102-120033-308347/summary.txt` |
| s4-13a | 2026-11-02 | leaf-eval arm vs stock at 20k iterations (fe_x2) | evalfit_r1_fe_x2 vs stock eval (scale 1) | engine self-play h2h (20k it.) | 5474 / 2737 | 59.61 [58.4, 60.8] [M] | - | arm label mapping assumed = dir order [E] | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261101-214953-308079/summary.txt` |
| s4-13b | 2026-11-02 | leaf-eval arm vs stock at 20k iterations (fe_x4) | evalfit_r1_fe_x4 vs stock eval (scale 1) | engine self-play h2h (20k it.) | 5456 / 2728 | 62.01 [60.8, 63.2] [M] | - | arm label mapping assumed = dir order [E] | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261101-214953-308079/summary.txt` |
| s4-13c | 2026-11-02 | leaf-eval arm vs stock at 20k iterations (full_x4) | evalfit_r1_full_x4 vs stock eval (scale 1) | engine self-play h2h (20k it.) | 5464 / 2732 | 64.48 [63.3, 65.6] [M] | - | arm label mapping assumed = dir order [E] | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261101-214953-308079/summary.txt` |
| s4-13d | 2026-11-02 | leaf-eval arm vs stock at 20k iterations (stock_x2) | evalfit_r1_stock_x2 vs stock eval (scale 1) | engine self-play h2h (20k it.) | 5454 / 2727 | 62.41 [61.2, 63.6] [M] | - | arm label mapping assumed = dir order [E] | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261101-214953-308079/summary.txt` |
| s4-13e | 2026-11-02 | leaf-eval arm vs stock at 20k iterations (stock_x4) | evalfit_r1_stock_x4 vs stock eval (scale 1) | engine self-play h2h (20k it.) | 5512 / 2756 | 65.02 [63.8, 66.2] [M] | - | arm label mapping assumed = dir order [E] | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261101-214953-308079/summary.txt` |
| s4-05 | 2026-11-03 | fitted leaf eval (fe x2) vs stock eval, search vs search on Showdown | search, fitted eval fe_x2 vs search, stock eval | Showdown duplicate (slurm_search) | 800 / 400 | 55.0 [51.8, 58.2] [M] | - | fit not opponent-specific (paper quotes 54.9 +- 3.4 per game) | `/vol/bitbucket/fr1226/gloria/runs/search-20261102-220256-308659/analysis.txt` |
| s4-06 | 2026-11-04 | 4 determinizations vs 1 (fitted eval) | search 4 samples vs search 1 sample | Showdown duplicate | 400 / 200 | 53.1 [48.7, 57.5] [M] | - | info | `/vol/bitbucket/fr1226/gloria/runs/search-20261103-175503-309120/analysis.txt` |
| s4-07 | 2026-11-04 | network prior 0.25 vs none (gen4, fitted eval) | search prior 0.25 (v1) vs search prior 0 | Showdown duplicate | 800 / 400 | 53.8 [50.5, 57.1] [D] | - | prior helps (+3.8 +- 3.3) | `/vol/bitbucket/fr1226/gloria/runs/search-20261103-233119-309116/analysis.txt (prior-0 side 46.2 +- 3.3; 100 - 46.2)` |
| s4-11 | 2026-11-04 | luck-adjusted vs raw-outcome fit targets | raw-outcome fit (full fe) vs luck-adjusted fit | engine self-play h2h (300k it.) | 1834 / 857 | 52.1 [49.9, 54.3] [M] | - | luck adjustment brings nothing | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261104-032420-309106/summary.txt` |
| s4-12 | 2026-11-04 | boost edits to the fitted eval (2 arms) | evalfit_wsp_boost_half / _matched vs fitted eval | engine self-play h2h | 3000 / 1500 | 49.3 [47.7, 50.9] [M] | - | no gain (second arm 49.8) | `/vol/bitbucket/fr1226/gloria/runs/h2h-20261104-064021-309108/summary.txt (arm order assumed = dir order [E])` |
| v2g4-11 | 2026-11-12 | gen4 search prior: R1b-F@150M vs v1@500M (prior test #1) | search + R1b-F@150M prior vs search + v1@500M prior | netv2 vs-search (Showdown, 21 workers) | 2000 / 1000 | 51.1 [49.1, 53.1] [M] | g4r1bf/SLOTTEST-PREREG.txt (b734530e) | inconclusive => keep v1 prior | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261112-112250-1127715/result.txt` |
| v2g4-12 | 2026-11-16 | gen4 search prior test #2: R1b-F@300M vs v1@500M | search + R1b-F@300M prior vs search + v1@500M prior | netv2 vs-search (Showdown) | 2000 / 1000 | 51.9 [50.0, 53.8] [M] | g4r1bf/SLOTTEST2-PREREG.txt (5ee05392) | INCONCLUSIVE => keep v1@500M prior | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g4r1bf/slottest2/VERDICT.txt` |

### v1 PPO network (gen4, SB3, Oct 2026) (23 rows)

| id | date | question | A vs B | harness | games / pairs | A's win % [95%] | prereg (md5) | verdict | src |
|---|---|---|---|---|---|---|---|---|---|
| v1-01 | 2026-10-20 | raw v1 net vs poke-env baseline bots | v1 PPO @60M vs SimpleHeuristicsPlayer | training eval, Showdown | 200 | 83.5 [M] | - | ~thesis level (Wang: ~80% at 40M) | `README.md @ ab10bee (2026-10-20); no run file` |
| v1-02 | 2026-10-20 | raw v1 net vs poke-env baseline bots | v1 PPO @60M vs MaxBasePowerPlayer | training eval, Showdown | 200 | 95.0 [M] | - | positive | `README.md @ ab10bee` |
| v1-03 | 2026-10-20 | search player (v1@60M prior, 300 ms) vs raw net | SearchPlayer 300 ms vs raw v1@60M | Showdown | 200 | 75.0 [M] | - | search >> raw net | `README.md @ 9481474` |
| v1-04 | 2026-10-21 | later checkpoint vs earlier | v1@121M vs v1@101M | Showdown | 200 | 56.5 [M] | - | still improving | `README.md @ 11a925f` |
| v1-05 | 2026-10-21 | later checkpoint vs earlier | v1@121M vs v1@60M | Showdown | 200 | 60.0 [M] | - | positive | `README.md @ 11a925f` |
| v1-08 | 2026-10-25 | official ladder (gen4randombattle) | raw v1@60M vs ladder opponents | Showdown official ladder | 20 | 65.0 [D] | - | 13-7; accounts then bot-flagged | `README.md @ ef7884d (13-7; date = first README mention [E]); 13/20 = 65%` |
| v1-09 | 2026-10-25 | later checkpoint vs earlier | v1@300M vs v1@60M | Showdown | 400 | 71.5 [M] | - | positive | `README.md @ bb439b7` |
| v1-10 | 2026-10-25 | v1 vs heuristic | v1@300M vs SimpleHeuristicsPlayer | Showdown | 400 | 89.5 [M] | - | positive | `README.md @ bb439b7` |
| v1-11 | 2026-10-25 | later checkpoint vs earlier | v1@400M vs v1@300M | Showdown | 600 | 54.5 [M] | - | ~+30 Elo per 100M | `README.md @ bb439b7` |
| v1-12 | 2026-10-25 | expert iteration round 1 (9k search battles distilled) | distilled net (KL beta 3) vs its source net | Showdown |  | 50.5 [M] | - | at best tied its source | `README.md @ ef7884d` |
| v1-13 | 2026-11-05 | +40M PPO at the lr floor from 600M (control arm) | arm-ctrl 377837 final vs v1@600M (301120) | Showdown h2h (gloria slurm_h2h_nets) | 4000 | 50.0 [48.45, 51.55] [M] | - | no gain | `/vol/bitbucket/fr1226/gloria/runs/h2h-nets-20261106-011250-380681/summary.txt` |
| v1-14 | 2026-11-05 | embedding lr x30 (+40M from 600M) | arm-x30 371651 final vs v1@600M | Showdown h2h | 4000 | 50.6 [49.05, 52.15] [M] | - | no gain | `/vol/bitbucket/fr1226/gloria/runs/h2h-nets-20261105-161840-380661/summary.txt` |
| v1-15 | 2026-11-05 | embedding lr x300 (+40M from 600M) | arm-x300 371541 final vs v1@600M | Showdown h2h | 4000 | 48.5 [46.95, 50.05] [M] | - | no gain | `/vol/bitbucket/fr1226/gloria/runs/h2h-nets-20261105-161840-380661/summary.txt` |
| v1-16 | 2026-11-08 | v1 600M vs 500M | v1@600M vs v1@500M | wsp h2h_dup (Showdown) | 400 / 200 | 51.1 [46.7, 55.5] [M] | - | no measurable gain | `/vol/bitbucket/fr1226/porygon-ws-wsp/out/h2h/676503/summary.txt` |
| v1-17 | 2026-11-08 | harness self-check (500M vs itself) | v1@500M vs v1@500M | wsp h2h_dup | 400 / 200 | 49.9 [45.5, 54.3] [M] | - | sanity | `/vol/bitbucket/fr1226/porygon-ws-wsp/out/h2h/676503/summary.txt` |
| v1-18a | 2026-11-08 | weight-space edit of v1@500M (switch-matchup key edit s1) | v1@500M + switch-matchup key edit s1 vs v1@500M | wsp h2h_dup | 4000 / 2000 | 49.9 [48.5, 51.3] [M] | - | no gain | `/vol/bitbucket/fr1226/porygon-ws-wsp/out/h2h/698101/summary.txt` |
| v1-18b | 2026-11-08 | weight-space edit of v1@500M (switch-matchup key edit s2) | v1@500M + switch-matchup key edit s2 vs v1@500M | wsp h2h_dup | 4000 / 2000 | 49.2 [47.8, 50.6] [M] | - | no gain | `/vol/bitbucket/fr1226/porygon-ws-wsp/out/h2h/698101/summary.txt` |
| v1-18c | 2026-11-08 | weight-space edit of v1@500M (switch-matchup key edit s3) | v1@500M + switch-matchup key edit s3 vs v1@500M | wsp h2h_dup | 4000 / 2000 | 45.9 [44.5, 47.3] [M] | - | harm | `/vol/bitbucket/fr1226/porygon-ws-wsp/out/h2h/698101/summary.txt` |
| v1-19 | 2026-11-08 | play temperature 0.5 vs default (v1@500M) | v1@500M T 0.5 vs v1@500M | wsp h2h_dup | 4000 / 2000 | 52.3 [50.9, 53.7] [M] | - | T 0.5 adopted for raw nets | `/vol/bitbucket/fr1226/porygon-ws-wsp/out/h2h/688986/summary.txt` |
| v1-20 | 2026-11-08 | SWA of late checkpoints | SWA(500M..600M) vs v1@600M | wsp h2h_dup | 4000 / 2000 | 50.0 [48.6, 51.4] [M] | - | no gain | `/vol/bitbucket/fr1226/porygon-ws-wsp/out/h2h/688986/summary.txt` |
| v1-21 | 2026-11-08 | SWA of late checkpoints | SWA(500M..600M) vs v1@500M | wsp h2h_dup | 4000 / 2000 | 50.8 [49.3, 52.3] [M] | - | no gain | `/vol/bitbucket/fr1226/porygon-ws-wsp/out/h2h/688986/summary.txt` |
| v1-22 | 2026-11-08 | L2H6 mean-ablation edit | v1@500M edited vs v1@500M | wsp h2h_dup | 4000 / 2000 | 50.6 [49.2, 52.0] [M] | - | no gain | `/vol/bitbucket/fr1226/porygon-ws-wsp/out/h2h/688986/summary.txt` |
| v1-23 | 2026-11-08 | L2H6 mean-ablation edit | v1@600M edited vs v1@600M | wsp h2h_dup | 4000 / 2000 | 50.4 [49.0, 51.8] [M] | - | no gain | `/vol/bitbucket/fr1226/porygon-ws-wsp/out/h2h/688986/summary.txt` |

### Gen9 port: pure engine search (20 rows)

| id | date | question | A vs B | harness | games / pairs | A's win % [95%] | prereg (md5) | verdict | src |
|---|---|---|---|---|---|---|---|---|---|
| g9s-04 | 2026-11-07 | gen9 fitted eval vs stock in engine self-play | fitted x2 vs stock | engine self-play h2h | 4000 | 60.2 [58.8, 61.6] [M] | - | engine-only over-predicts Showdown gain | `docs/paper/paper.tex (Generation 9 paragraph)` |
| g9s-05 | 2026-11-07 | gen9 stock x2 vs stock in engine self-play | stock x2 vs stock | engine self-play h2h | 4000 | 58.4 [57.0, 59.8] [M] | - | positive | `docs/paper/paper.tex` |
| g9s-01 | 2026-11-08 | gen9 pure search, stock eval (4 runs) | search:none 4x1 s, stock vs Foul Play gen9 | gloria slurm_foulplay --gen 9 | 1508 / 754 | 51.72 [49.36, 54.08] [D] | - | ~even | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); jobs 515841/4/7/50` |
| g9s-02 | 2026-11-08 | gen9 pure search, stock x2 | search:none, stock x2 vs Foul Play gen9 | gloria slurm_foulplay --gen 9 | 1600 / 800 | 53.31 [51.04, 55.58] [D] | - | positive | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); jobs 515842/5/8/51` |
| g9s-03 | 2026-11-08 | gen9 pure search, fitted eval_gen9_r1 x2 | search:none, fitted vs Foul Play gen9 | gloria slurm_foulplay --gen 9 | 1494 / 746 | 53.28 [50.93, 55.64] [D] | - | fit +1.6 +- 3.4 over stock: not measurable | `/vol/bitbucket/fr1226/porygon-thesis/pooled.txt (pool.py over results.jsonl/battles.jsonl); jobs 515843/6/9/52 (runs predate the sleep-count fix 006ff76)` |
| g9s-06 | 2026-11-09 | label teacher: G9p@17.5M prior in gen9 search vs pure search (500 ms) | search G9p prior 0.25, 4x500 ms vs search:none 4x500 ms | Showdown duplicate server (netv2 h2h) | 400 / 200 | 51.5 [47.0, 56.0] [M] | - | no gain from a weak prior | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261109-083010-770666/result.txt` |
| g9s-07 | 2026-11-09 | label teacher: more search time (1000 vs 500 ms) | search:none 4x1000 ms vs search:none 4x500 ms | Showdown duplicate server (netv2 h2h) | 400 / 200 | 53.5 [49.0, 58.0] [M] | - | not clearly better | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261109-100931-770666/result.txt` |

Per-run constituents (13 gloria runs; each [M] from its `analysis.txt` unless marked [D]; the arm's network is not logged, see the note below):

| job | date | config (non-default `SearchPlayer` options) | games / pairs | win % [95%] | outcome |
|---|---|---|---|---|---|
| 492059 | 2026-11-07 | prior_weight=0.0 eval_config=eval_gen9_r1.json oracle=None | 8 / 4 | 37.5 [13.0, 62.0] [M] | info |
| 515841 | 2026-11-07 | prior_weight=0.0 eval_config=stock oracle=None | 338 / 169 | 51.2 [46.0, 56.4] [M] | info |
| 515842 | 2026-11-07 | prior_weight=0.0 eval_config=gen9_stock_x2.json oracle=None | 400 / 200 | 55.0 [50.7, 59.3] [M] | info |
| 515843 | 2026-11-07 | prior_weight=0.0 eval_config=eval_gen9_r1.json oracle=None | 400 / 200 | 49.0 [44.7, 53.3] [M] | info |
| 515844 | 2026-11-07 | prior_weight=0.0 eval_config=stock oracle=None | 400 / 200 | 50.2 [45.5, 54.9] [M] | info |
| 515845 | 2026-11-07 | prior_weight=0.0 eval_config=gen9_stock_x2.json oracle=None | 400 / 200 | 53.5 [49.0, 58.0] [M] | info |
| 515846 | 2026-11-07 | prior_weight=0.0 eval_config=eval_gen9_r1.json oracle=None | 400 / 200 | 55.8 [51.4, 60.2] [M] | info |
| 515847 | 2026-11-07 | prior_weight=0.0 eval_config=stock oracle=None | 370 / 185 | 53.2 [48.6, 57.8] [M] | info |
| 515848 | 2026-11-07 | prior_weight=0.0 eval_config=gen9_stock_x2.json oracle=None | 400 / 200 | 53.5 [49.1, 57.9] [M] | info |
| 515849 | 2026-11-07 | prior_weight=0.0 eval_config=eval_gen9_r1.json oracle=None | 329 / 164 | 52.1 [46.9, 57.3] [M] | info |
| 515850 | 2026-11-07 | prior_weight=0.0 eval_config=stock oracle=None | 400 / 200 | 52.2 [47.8, 56.6] [M] | info |
| 515851 | 2026-11-08 | prior_weight=0.0 eval_config=gen9_stock_x2.json oracle=None | 400 / 200 | 51.2 [46.3, 56.1] [M] | info |
| 515852 | 2026-11-08 | prior_weight=0.0 eval_config=eval_gen9_r1.json oracle=None | 365 / 182 | 56.3 [51.3, 61.3] [M] | info |

### v2 network, gen4 (R1b / R1b-F) (10 rows)

| id | date | question | A vs B | harness | games / pairs | A's win % [95%] | prereg (md5) | verdict | src |
|---|---|---|---|---|---|---|---|---|---|
| v2g4-01 | 2026-11-09 | v2 net @17M (R1b handover) vs old v1 @46M | R1b @17M vs v1 @46M | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 56.9 [54.0, 59.8] [M] | - | v2 ahead at 1/3 the steps | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261109-030957-718701/result.txt` |
| v2g4-02 | 2026-11-10 | R1b-F strength check @50M vs v1@46M | R1b-F @50M vs v1 @46M | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 78.7 [76.3, 81.1] [M] | - | continue | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261110-060733-829660/result.txt` |
| v2g4-03 | 2026-11-10 | R1b-F @50M vs v1@500M (best v1) | R1b-F @50M vs v1 @500M | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 49.8 [46.9, 52.7] [M] | - | null | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261110-061309-829660/result.txt` |
| v2g4-04 | 2026-11-11 | R1b-F strength check @100M vs v1@46M | R1b-F @100M vs v1 @46M | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 77.0 [74.5, 79.5] [M] | - | continue | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261110-235543-829660/result.txt` |
| v2g4-05 | 2026-11-11 | R1b-F @100M vs v1@500M (best v1) | R1b-F @100M vs v1 @500M | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 52.6 [49.7, 55.5] [M] | - | null | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261111-000022-829660/result.txt` |
| v2g4-06 | 2026-11-11 | R1b-F strength check @150M vs v1@46M | R1b-F @150M vs v1 @46M | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 80.9 [78.6, 83.2] [M] | - | continue | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261111-183536-829660/result.txt` |
| v2g4-07 | 2026-11-11 | R1b-F @150M vs v1@500M (best v1) | R1b-F @150M vs v1 @500M | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 56.4 [53.6, 59.2] [M] | - | passed >= 55% rule | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261111-184315-829660/result.txt` |
| v2g4-08 | 2026-11-14 | R1b-F strength check @300M vs v1@46M | R1b-F @300M vs v1 @46M | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 84.6 [82.5, 86.7] [M] | g4r1bf/NOTES.txt (300M watch) (6ae5d289) | continue | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261114-154510-1136569/result.txt` |
| v2g4-09 | 2026-11-14 | R1b-F @300M vs v1@500M (best v1) | R1b-F @300M vs v1 @500M | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 60.7 [58.1, 63.3] [M] | g4r1bf/NOTES.txt (300M watch) (6ae5d289) | >= 60% => candidate prior | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261114-154746-1136569/result.txt` |
| v2g4-13 | 2026-11-16 | R1b-F hold #1: guard trip @390.2M vs R1b-F@300M | R1b-F trip @390.2M vs R1b-F @300M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 49.9 [47.9, 51.9] [M] | g4r1bf/HOLD-PREREG.txt (af0441c6) | CONTINUE (not clearly worse; not better) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g4r1bf-hold1/VERDICT.txt` |

### v2 network, gen9 (G9s, G9p, G9q milestones) (28 rows)

| id | date | question | A vs B | harness | games / pairs | A's win % [95%] | prereg (md5) | verdict | src |
|---|---|---|---|---|---|---|---|---|---|
| v2g9-01 | 2026-11-09 | G9 (=G9p, P1-init PPO) @25M vs P1, real Showdown | G9p @25M vs P1 | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 65.7 [63.0, 68.4] [M] | - | RL improves on BC; validates in-process (65.9 +- 1.1) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261109-115722-741849/result.txt` |
| v2g9-02 | 2026-11-09 | G9p handover (~13M) vs pure gen9 search | G9p ~13M vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 400 / 200 | 35.8 [31.0, 40.6] [M] | - | info | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261109-062535-770331/result.txt` |
| v2g9-03 | 2026-11-09 | G9s (from scratch) @25M vs pure gen9 search (seed 738164) | G9s (from scratch) @25M T 0.5 vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 400 / 200 | 26.8 [22.6, 31.0] [M] | - | below 50% | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261109-124045-784935/result.txt` |
| v2g9-04 | 2026-11-09 | G9p @25M vs pure gen9 search (seed 738164) | G9p @25M T 0.5 vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 400 / 200 | 39.8 [35.2, 44.4] [M] | - | below 50% | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261109-133546-784936/result.txt` |
| v2g9-05 | 2026-11-09 | G9s @50M vs pure gen9 search (seed 738164) | G9s @50M T 0.5 vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 400 / 200 | 36.8 [31.8, 41.8] [M] | - | below 50% | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261109-143347-801702/result.txt` |
| v2g9-06 | 2026-11-09 | G9p @50M vs pure gen9 search (seed 738164) | G9p @50M T 0.5 vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 400 / 200 | 42.5 [38.2, 46.8] [M] | - | below 50% | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261109-161806-824044/result.txt` |
| v2g9-07 | 2026-11-09 | G9p @100M vs pure gen9 search (seed 738164) | G9p @100M T 0.5 vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 400 / 200 | 43.5 [39.1, 47.9] [M] | - | below 50% | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261109-223729-839408/result.txt` |
| v2g9-09 | 2026-11-09 | G9s vs G9p step-matched @10M | G9s @10M vs G9p @10M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 200 / 100 | 31.0 [25.29, 36.71] [M] | - | G9p > G9s SUCCESS (P1 init helps) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/cmp/10M/RESULT.txt` |
| v2g9-10 | 2026-11-09 | G9s vs G9p @25M | G9s @25M vs G9p @25M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 200 / 100 | 33.0 [27.39, 38.61] [M] | - | G9p > G9s SUCCESS | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/cmp/25M/RESULT.txt` |
| v2g9-11 | 2026-11-09 | G9s vs G9p @50M | G9s @50M vs G9p @50M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 200 / 100 | 35.5 [29.07, 41.93] [M] | - | G9p > G9s SUCCESS | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/cmp/50M/RESULT.txt` |
| v2g9-12 | 2026-11-09 | G9s vs G9p, latest snapshots 11-09 12:19 | G9s latest vs G9p latest | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 1200 / 600 | 45.26 [42.56, 48.05] [D] | cmp/latest-20261109-1219/PREREG.txt (f341b9e0) | G9p > G9s SUCCESS (MUE 54.74 [51.95, 57.44]); G9s parked | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/cmp/latest-20261109-1219/RESULT.txt (A = 100 - B's MUE)` |
| v2g9-18 | 2026-11-09 | EMA weights vs raw weights (G9p @65.97M) | G9p EMA vs G9p raw | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 51.4 [49.4, 53.4] [M] | g9p-ema/PREREG.txt (63c9776d) | no clear gain | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9p-ema/h2h/RESULT.txt` |
| v2g9-08 | 2026-11-10 | G9s @75M vs pure gen9 search (seed 738164) | G9s @75M T 0.5 vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 400 / 200 | 39.8 [35.3, 44.3] [M] | - | below 50% | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261110-024824-843442/result.txt` |
| v2g9-13 | 2026-11-10 | lr-cooldown branch vs main run at matched steps | G9p-cd @104.71M vs G9p @105.01M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 3400 / 1700 | 52.05 [50.42, 53.65] [M] | g9p-cd/PREREG-CD.txt (82bf7250) | SUCCESS: cooldown branches become deployment nets | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9p-cd/cmp/RESULT.txt` |
| v2g9-14 | 2026-11-10 | G9p-cd vs pure gen9 search (netsearch gate, seed 738164) | G9p-cd T 0.5 vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 400 / 200 | 50.5 [46.0, 55.0] [M] | docs/netsearch/NETSEARCH-PREREG.md (gate) (db59ec88 (file now incl. results)) | gate PASSED (>= 45%) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261110-004830-845624/result.txt` |
| v2g9-15 | 2026-11-10 | G9p-cd vs pure gen9 search, 1000-g confirmation (new seed 738201) | G9p-cd T 0.5 vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 1000 / 500 | 43.4 [40.6, 46.2] [M] | - | 400-g 50.5 not confirmed | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261110-045726-864111/result.txt` |
| v2g9-19 | 2026-11-10 | actor-stream speed test (a) | --actor-stream --decoupled vs --no-overlap (2 arms) | trainer sps (A30) |  | 138.0 [M] | as-test/PREREG-A.txt (b0f82ddc) | GO to (b): 1.380x >= 1.25 | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/as-test/RESULT.txt` |
| v2g9-20 | 2026-11-10 | actor-stream per-sample non-inferiority (b), +5M from G9p@183M | arm B (actor-stream) vs arm A (sync) | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 4000 / 2000 | 50.0 [48.6, 51.4] [M] | as-test-b/PREREG-B.txt (f01783d2) | B PASSES (lower > 48%) => adopted | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/as-test-b/RESULT.txt` |
| v2g9-21 | 2026-11-10 | actor-stream long-horizon drift: G9p after +25M with actor-stream vs its start | D25 (actor-stream) vs D (start) | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 46.0 [44.0, 48.0] [M] | as-test-b/PREREG-DRIFT.txt (12043732) | drift (control 49.6 +- 2.0) => actor-stream REVERTED | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/as-test-b/drift/h2h/RESULT.txt` |
| v2g9-22 | 2026-11-10 | drift control: synchronous run over the same span | D vs Dm25 | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 49.6 [47.6, 51.6] [M] | as-test-b/PREREG-DRIFT.txt (12043732) | control | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/as-test-b/drift/control-h2h/RESULT.txt` |
| g9q-03 | 2026-11-11 | G9q @112.5M vs G9p @338.6M (3x steps) | G9q @112.53M vs G9p @338.55M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 49.4 [47.4, 51.4] [M] | g9q-vs-g9p/PREREG.txt (7d65a559) | INCONCLUSIVE (G9q matches G9p at 1/3 the steps) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9q-vs-g9p/READING.txt` |
| v2g9-23 | 2026-11-11 | G9p plateau: @265M vs @100M | G9p @265.03M vs G9p @100M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 55.2 [53.2, 57.2] [M] | g9p-plateau/PREREG.txt (656af9ba) | STILL IMPROVING | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9p-plateau/READING.txt` |
| v2g9-24 | 2026-11-11 | G9p plateau: @265M vs @186M | G9p @265.03M vs G9p @186.08M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 51.8 [49.8, 53.8] [M] | g9p-plateau/PREREG.txt (656af9ba) | INCONCLUSIVE | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9p-plateau/READING.txt` |
| g9q-13 | 2026-11-13 | re-freeze trigger: G9q-cd@305M vs G9p-cd (raw) | G9q-cd @305.04M vs G9p-cd @104.71M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 400 / 200 | 59.55 [54.94, 63.96] [M] | REFREEZE-PREREG.md s2 + g9q-cd300/PREREG.txt (ad1911d2 / af9315a1) | trigger passes (>= 55%) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9q-cd300/cmp/READING.txt` |
| g9q-15 | 2026-11-14 | raw G9q-cd@605M vs pure gen9 search (400 g, seed 738164) | G9q-cd@605M vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 400 / 200 | 57.0 [52.5, 61.5] [M] | rawfp/PREREG.txt (f0a508e3) | >= 50% => propose 1000-g confirmation | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/rawfp/RESULTS.txt` |
| g9q-16 | 2026-11-15 | Milestone 1 attempt: raw G9q-cd@605M vs pure gen9 search (1000 g) | G9q-cd@605M vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 1000 / 500 | 52.5 [49.6, 55.4] [M] | conf605/PREREG.txt (517b679c) | NOT CONFIRMED (+9.1 +- 3.9 over G9p-cd on the same pairs) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/conf605/RESULTS.txt` |
| g9q-21 | 2026-11-15 | G9q still improving: cd@805 vs cd@605 | G9q-cd @805.03M vs G9q-cd @605.04M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2800 / 1400 | 52.46 [50.67, 54.22] [M] | g9q-cd800/PREREG.txt (1a1b9fa4) | IMPROVING | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9q-cd800/READING.txt` |
| g9q-23 | 2026-11-16 | Milestone 1: raw G9q-cd@805M vs pure gen9 search (1000 g, seed 8800111) | G9q-cd@805M T 0.5 vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 1000 / 500 | 55.5 [52.7, 58.3] [M] | conf800/PREREG.txt (c6a38724) | CONFIRMED: raw net beats our search | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/conf800/RESULTS.txt` |

### v2 PPO recipe arms (one change each) (13 rows)

| id | date | question | A vs B | harness | games / pairs | A's win % [95%] | prereg (md5) | verdict | src |
|---|---|---|---|---|---|---|---|---|---|
| g9q-04 | 2026-11-12 | G9r (power-law lr decay) 25M harm guard vs G9q | G9r @25M vs G9q @25M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 1800 / 900 | 49.86 [47.68, 52.17] [M] | g9r/PREREG.txt (ee69eb24) | no harm => continue | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9r/cmp/25M/READING.txt` |
| g9q-05 | 2026-11-12 | G9r-cd vs G9q-cd @50M (rule 2, decision of record) | G9r-cd @50M vs G9q-cd @50M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 3200 / 1600 | 48.12 [45.04, 51.37] [M] | g9r/PREREG.txt + PREREG-CD50.txt (ee69eb24 / c0aa6045) | UNDECIDED (both FUTILITY) => no gain; G9r parked | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9r/cmp/cd50/READING.txt (all pairs 49.5 +- 1.6)` |
| g9q-06 | 2026-11-12 | G9r drift: @50M vs @25M | G9r @50M vs G9r @25M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 1000 / 500 | 55.18 [52.22, 58.04] [M] | g9r/PREREG.txt (ee69eb24) | not degrading | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9r/cmp/drift50/READING.txt` |
| g9q-07 | 2026-11-12 | G9r raw @50M vs G9q raw @50M (informational) | G9r @50M vs G9q @50M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 3200 / 1600 | 52.16 [50.52, 53.77] [M] | g9r/PREREG.txt (ee69eb24) | raw-vs-raw favours the lower-lr arm; cooled vs cooled decides | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9r/cmp/info50/READING.txt` |
| g9q-08 | 2026-11-12 | GAE lambda pre-check (noise/bias on G9q@45M rollouts) | lambda 0.5 vs lambda 0.754 | offline rollout statistics |  | lambda 0.5 picked (0.6 fails gain) [M] | g9t/lam/PREREG.txt (b37c83a6) | lambda 0.5 picked (0.6 fails gain) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9t/lam/VERDICT.txt` |
| g9q-09 | 2026-11-14 | G9t (GAE lambda 0.5) 25M vs G9q | G9t @25M vs G9q @25M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2400 / 1200 | 49.9 [48.0, 51.8] [M] | g9t/PREREG.txt (e57b8ab4) | CONTINUE (both FUTILITY) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9t/cmp/25M/READING.txt` |
| g9q-10 | 2026-11-14 | G9t-cd vs G9q-cd @50M (rule 2) | G9t-cd @50M vs G9q-cd @50M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 4000 / 2000 | 48.1 [46.7, 49.5] [M] | g9t/PREREG.txt (e57b8ab4) | STOP: G9q-cd > G9t-cd SUCCESS (MUE 51.71 [50.25, 53.16]); G9t parked | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9t/cmp/cd50/READING.txt` |
| g9q-11 | 2026-11-14 | G9t drift: @50M vs @25M | G9t @50M vs G9t @25M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 3800 / 1900 | 51.8 [50.4, 53.2] [M] | g9t/PREREG.txt (e57b8ab4) | CONTINUE | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9t/cmp/drift50/READING.txt` |
| g9q-12 | 2026-11-14 | G9t raw @50M vs G9q raw @50M (info) | G9t @50M vs G9q @50M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 3800 / 1900 | 50.8 [49.3, 52.3] [M] | g9t/PREREG.txt (e57b8ab4) | info | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9t/cmp/info50/READING.txt` |
| g9q-18 | 2026-11-15 | seed variance: G9q2 (same recipe, new seed) vs G9q @25M | G9q2 @25M vs G9q @25M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2800 / 1400 | 50.2 [48.5, 51.9] [M] | g9q2/PREREG.txt (5d218a36) | both FUTILITY | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9q2/cmp/25M/READING.txt` |
| g9q-19 | 2026-11-16 | seed variance: G9q2 vs G9q @50M | G9q2 @50M vs G9q @50M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 1600 / 800 | 50.7 [48.5, 52.9] [M] | g9q2/PREREG.txt (5d218a36) | both FUTILITY: seed variance < ~2 points | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9q2/cmp/OVERALL.txt` |
| g9q-20 | 2026-11-16 | bf16 autocast speed (A40, G9q recipe) | bf16 vs fp32 | trainer sps microbench (24 updates each) |  | 101.8 [M] | g9qbf/PREREG.txt (a77115fe) | SKIP (needs >= 1.15x): stay fp32 | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9qbf/SKIP.txt` |
| g9q-22 | 2026-11-16 | entropy schedule 0.01 -> 0.001 (G9qent), cooled vs cooled | G9qent-cd @880M vs G9q-cd @885M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 3000 / 1500 | 50.64 [48.92, 52.48] [M] | g9qent/PREREG.txt (ecc8fe39) | NO GAIN (both FUTILITY) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9qent/VERDICT.txt` |

### Human behaviour cloning and the data-quality lever (P1-P4, U7) (19 rows)

| id | date | question | A vs B | harness | games / pairs | A's win % [95%] | prereg (md5) | verdict | src |
|---|---|---|---|---|---|---|---|---|---|
| bc-01 | 2026-11-09 | v1-architecture BC vs v2 BC on the same human data (P1) | v1 BC (bc-gen9-p1-v1) vs v2 BC P1 (g9h-dmg-s2) | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 22.4 [20.0, 24.8] [M] | - | v2 architecture far better | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261109-051040-738164/result.txt` |
| bc-02 | 2026-11-09 | v1 BC vs pure gen9 search | v1 BC P1 vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 400 / 200 | 13.0 [9.8, 16.2] [M] | - | info | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261109-051536-738164/result.txt` |
| bc-03 | 2026-11-09 | P1 (v2 BC, >=1900) vs pure gen9 search | P1 T 0.5 vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 400 / 200 | 31.0 [26.7, 35.3] [M] | - | info | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261109-034403-738164/result.txt` |
| bc-04 | 2026-11-09 | P1 vs Foul Play gen9 | P1 T 0.5 vs Foul Play gen9 | Foul Play gen9 (8f541f8, ms 1000, 1 core), netv2 fp9 harness | 400 / 200 | 33.8 [29.1, 38.5] [M] | - | info | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/foulplay-net-20261109-072606-763531/result.txt` |
| bc-05 | 2026-11-09 | data size: BC on 0.92M decisions (P1 size) vs P1 | BC vr-1pg (0.92M) vs P1 | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 48.7 [46.6, 50.8] [M] | p2cmp/PREREG.txt (bca99aeb) | info | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/p2cmp/vr1pg-vs-p1/RESULT.txt` |
| bc-06 | 2026-11-09 | data size: BC on P2 (2.29M) vs P1 | BC P2 vs P1 | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 52.8 [50.8, 54.8] [M] | p2cmp/PREREG.txt (bca99aeb) | more data helps | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/p2cmp/p2-vs-p1/RESULT.txt` |
| bc-07 | 2026-11-09 | data size: BC on P3 (7.05M) vs P1 | BC P3 vs P1 | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 59.6 [57.6, 61.6] [M] | p2cmp/PREREG-P3.txt (e7793ce9) | more data helps: P3 becomes the init | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/p2cmp/p3-vs-p1/RESULT.txt` |
| bc-09 | 2026-11-09 | BC-P3 vs G9p@100M (RL from P1) | BC P3 vs G9p@100M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 36.9 [34.9, 38.9] [M] | p3-vs-g9p100/PREREG.txt (a2dadb1a) | RL >> BC | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/p3-vs-g9p100/h2h/RESULT.txt` |
| bc-12 | 2026-11-09 | search labels as BC targets (P2 fine-tuned on 4x500 ms search labels) vs P1 | BC P2ft (search labels) vs P1 | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 34.9 [32.1, 37.7] [M] | - | search labels hurt | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261109-050133-718701/result.txt` |
| bc-13 | 2026-11-09 | human + search-label mix vs P1 | BC P2-mix vs P1 | Showdown duplicate server (netv2 h2h) | 1008 / 504 | 46.0 [43.1, 48.9] [M] | - | no gain | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261109-065627-718701/result.txt` |
| bc-14 | 2026-11-09 | search-label net at sharper T 0.28 vs P1 (sequential) | BC P2ft T 0.28 vs P1 T 0.5 | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2800 / 1400 | 47.34 [45.49, 49.24] [D] | - | sequential design runs/cmp/seq-gain.json; P1 > P2ft SUCCESS (MUE 52.65 [50.76, 54.51]) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/d1b-p2ft-t028-vs-p1/RESULT.txt (A est = 100 - 52.65)` |
| bc-08 | 2026-11-10 | BC-P3 vs pure gen9 search | BC P3 T 0.5 vs pure gen9 search | pure gen9 search (search:none, 4 x 1000 ms, stock eval), netv2 vs-search harness (21 workers, 5 concurrent) | 400 / 200 | 38.5 [34.2, 42.8] [M] | - | info | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/vs-search-20261110-015023-843802/result.txt` |
| g9q-01 | 2026-11-10 | human-init scale: G9q (P3 init) vs G9p (P1 init) @25M | G9q @25M vs G9p @25M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 400 / 200 | 61.06 [55.77, 65.9] [M] | g9q/PREREG.txt (a64af3b2) | SUCCESS: bigger BC init carries through RL | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9q/cmp/25M/RESULT.txt` |
| g9q-02 | 2026-11-11 | human-init scale: G9q vs G9p @50M | G9q @50M vs G9p @50M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 600 / 300 | 58.17 [53.78, 62.33] [M] | g9q/PREREG.txt (a64af3b2) | SUCCESS => G9q becomes the main gen9 run | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9q/cmp/50M/RESULT.txt` |
| bc-10 | 2026-11-12 | U7: rating quality at equal size (>=1500 uniform vs P3 >=1900) | BC U7 vs BC P3 | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 6000 / 3000 | 47.2 [46.0, 48.4] [M] | u7/PREREG.txt (067f4b39) | QUALITY MATTERS (P3 > U7) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/u7/READING.txt` |
| bc-11 | 2026-11-12 | U7 vs P1 (informational) | BC U7 vs P1 | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 57.6 [55.6, 59.6] [M] | u7/PREREG.txt (067f4b39) | info | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/u7/u7-vs-p1/RESULT.txt` |
| bc-15 | 2026-11-16 | P4 data lever: BC-P4 (12.0M, own rating >=1900) vs BC-P3 | BC-P4 vs BC-P3 | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 49.95 [47.81, 52.23] [M] | p4/PREREG.txt (2479eb4f) | B1 both FUTILITY: within ~2 points (informational) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/p4/bc/READING.txt` |
| g9q-25 | 2026-11-17 | P4 init: G9-P4 vs G9q step-matched @25M | G9-P4 @25M vs G9q @25M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 4800 / 2400 | 51.45 [50.06, 52.87] [M] | p4/PREREG.txt (2479eb4f) | SUCCESS-P4 (informational) | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/p4/cmp/OVERALL.txt` |
| g9q-26 | 2026-11-17 | P4 init: G9-P4 vs G9q step-matched @50M (decides) | G9-P4 @50M vs G9q @50M | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 3800 / 1900 | 51.81 [50.31, 53.29] [M] | p4/PREREG.txt (2479eb4f) | CANDIDATE next main | `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/p4/cmp/OVERALL.txt` |

### Network + search: hybrid, V3, lookahead2, NNUE, deepen (19 rows)

| id | date | question | A vs B | harness | games / pairs | A's win % [95%] | prereg (md5) | verdict | src |
|---|---|---|---|---|---|---|---|---|---|
| ns-01 | 2026-11-10 | Stage 1 offline: raw G9p-cd vs search, root-Q regret R | raw G9p-cd vs search (R per decision) | offline re-score, 501 positions |  | -1.72 [-1.97, -1.47] [M] | - | margin arms d 0.0212 / 0.0261 / 0.0358 | `docs/netsearch/NETSEARCH-PREREG.md Stage 1 ($F/netsearch/stage1-g9pcd-v2/)` |
| ns-02 | 2026-11-10 | Stage 2 tuning: hybrid H(d 0.0212) vs pure search | H(d 0.0212) + G9p-cd vs pure gen9 search | fastenv Stage 3 slot harness (11 engine procs/side, 3 concurrent) | 400 / 200 | 53.25 [48.69, 57.81] [M] | docs/netsearch/NETSEARCH-PREREG.md (db59ec88 (file now incl. results)) | d* = 0.0212, GO | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/netsearch-stage2-20261110-102106-943438/result.txt (overrides 22.5%)` |
| ns-03 | 2026-11-10 | Stage 2 tuning: hybrid H(d 0.0261) vs pure search | H(d 0.0261) + G9p-cd vs pure gen9 search | fastenv Stage 3 slot harness (11 engine procs/side, 3 concurrent) | 400 / 200 | 48.75 [44.28, 53.22] [M] | docs/netsearch/NETSEARCH-PREREG.md (db59ec88 (file now incl. results)) | not selected | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/netsearch-stage2-20261110-102106-943438/result.txt (overrides 19.6%)` |
| ns-04 | 2026-11-10 | Stage 2 tuning: hybrid H(d 0.0358) vs pure search | H(d 0.0358) + G9p-cd vs pure gen9 search | fastenv Stage 3 slot harness (11 engine procs/side, 3 concurrent) | 400 / 200 | 51.5 [47.06, 55.94] [M] | docs/netsearch/NETSEARCH-PREREG.md (db59ec88 (file now incl. results)) | not selected | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/netsearch-stage2-20261110-102106-943438/result.txt (overrides 16.3%)` |
| ns-05 | 2026-11-11 | Stage 3a: hybrid H(0.0212) vs pure gen9 search (fixed n) | H(0.0212) + G9p-cd vs pure gen9 search | fastenv Stage 3 slot harness (11 engine procs/side, 3 concurrent) | 1000 / 500 | 55.0 [52.0, 58.0] [M] | docs/netsearch/NETSEARCH-PREREG.md (db59ec88 (file now incl. results)) | Q1 SUCCESS | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/netsearch-stage3ac-20261110-171632-985078/result.txt` |
| ns-06 | 2026-11-11 | Stage 3c: raw G9p-cd vs pure search, same pairs | raw G9p-cd T 0.5 vs pure gen9 search | fastenv Stage 3 slot harness (11 engine procs/side, 3 concurrent) | 1000 / 500 | 48.05 [45.09, 51.01] [M] | docs/netsearch/NETSEARCH-PREREG.md (db59ec88 (file now incl. results)) | Q3 paired 3a - 3c = +7.0 +- 4.2 | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/netsearch-stage3ac-20261110-171632-985078/result.txt` |
| ns-10 | 2026-11-12 | V3 gate P1: leaf-encoder parity | v3leaf encoder vs reference encoder | offline, 300 pairs |  | 100.0 [M] | docs/netsearch/V3-GATES-PREREG.md (3407619d (pre-results)) | PASS (A agree 1.000, mean dP 1.9e-4) | `docs/netsearch/V3-GATES-PREREG.md Results (job 1127567)` |
| ns-11 | 2026-11-12 | V3 gate P2: leaves per decision on the slot T4 | Python V3, B 64 vs target >= 2,000 | bench, 90 decisions |  | 2048 [M] | docs/netsearch/V3-GATES-PREREG.md (3407619d (pre-results)) | PASS (forward 47% of wall) | `docs/netsearch/V3-GATES-PREREG.md Results` |
| ns-12 | 2026-11-12 | V3 I1: batched GPU leaf inference under slot contention | best: E16 B 384 vs target 35,300 leaves/s | T4 bench, 5,904 leaves |  | 7465 [M] | docs/netsearch/V3-I1-PREREG.md (b1d4fe17 (pre-results)) | FAIL => V3 parked | `docs/netsearch/V3-I1-PREREG.md Results (job 1135630)` |
| ns-09 | 2026-11-14 | re-freeze #1: H(G9q-cd@305, d 0.0172) vs H(G9p-cd) | H(G9q-cd@305) vs H(G9p-cd) benchmark bot | fastenv Stage 3 slot harness (11 engine procs/side, 3 concurrent) | 1000 / 500 | 50.7 [47.91, 53.49] [M] | docs/netsearch/REFREEZE-PREREG.md (ad1911d2) | NOT ADOPTED | `/vol/bitbucket/fr1226/porygon-ws-fastenv/refreeze1/VERDICT.txt` |
| ns-13 | 2026-11-15 | lookahead2 margin d* by the precision rule (cd@605) | lookahead2 vs raw net | offline study, 1,403 positions |  | d* = 0.01 [M] | docs/netsearch/LA2-BET1-PREREG.md (97c9b4e9) | d* = 0.01 | `git e874baf; $F/lookahead2/study-20261115/bench-all.txt` |
| ns-14 | 2026-11-16 | Bet 1: lookahead2(cd@605, d 0.01, K 8) vs the same raw net | lookahead2 on G9q-cd@605M vs raw G9q-cd@605M T 0.5 | fastenv slot harness on Showdown (lookahead2_slot), sequential 50 -> 54% | 1200 / 600 | 53.17 [50.59, 55.73] [M] | LA2-BET1-PREREG.md (97c9b4e9) | GO (overrides 16.3%; 0.85 s/decision) | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/la2-bet1/VERDICT.txt` |
| ns-15 | 2026-11-16 | lookahead2 margin d* for cd@805 | lookahead2 on cd@805 vs raw cd@805 | offline study, 1,403 positions |  | d* = 0.005 [M] | LA2-DELTA800-PREREG.md (77ae985a) | d* = 0.005 | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/la2-delta800/VERDICT.txt` |
| ns-16 | 2026-11-16 | NNUE gate 1: 155k CPU student vs teacher and evaluate() (set H, human, K 4 dets) | student AUC 0.714 vs evaluate() AUC 0.718 / teacher 0.748 | offline, 3,000 positions |  | FAIL (Bet 2 as designed stops) [M] | NNUE-GATE1-PREREG.md v2 (43f045da) | FAIL (Bet 2 as designed stops) | `/vol/bitbucket/fr1226/porygon-ws-fastenv/nnue/gate1/VERDICT.txt` |
| ns-17 | 2026-11-16 | NNUE gate 1 set S (self-play, true State) | student AUC 0.796 vs evaluate() 0.773 / teacher 0.834 | offline, 200,000 positions |  | c2 met, c1 not met [M] | NNUE-GATE1-PREREG.md v2 (43f045da) | c2 met, c1 not met | `/vol/bitbucket/fr1226/porygon-ws-fastenv/nnue/gate1/VERDICT.txt` |
| ns-20 | 2026-11-17 | LA2-FP Q2: paired gain of lookahead2 over the raw net vs Foul Play | lookahead2 - raw (points) vs  | paired per team pair | 2000 / 500 | 1.3 [-2.71, 5.31] [M] | LA2-FP-PREREG.md (1173284f) | NOT SHOWN | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/la2-fp/VERDICT.txt` |
| ns-21 | 2026-11-17 | scale gate T2: lookahead2 vs deepen D1 (1,024 extra leaves) | lookahead2 (cd@805) vs deepen D1 | fastenv Stage 3 slot harness (11 engine procs/side, 3 concurrent) (helper, chain B), sequential 50 -> 54% | 600 / 300 | 49.13 [45.41, 53.21] [M] | SCALE-GATE-PREREG.md (9fd3960d) | FUTILITY: 1x not shown worse than lookahead2 (naive 48.50 [45.03, 51.97]) | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/scale-gate/VERDICT-B.txt` |
| ns-22 | 2026-11-17 | scale gate T3 (info): raw net vs deepen D1 | raw cd@805 vs deepen D1 | fastenv Stage 3 slot harness (11 engine procs/side, 3 concurrent) (helper) | 400 / 200 | 46.17 [40.99, 52.01] [M] | SCALE-GATE-PREREG.md (9fd3960d) | FUTILITY (naive 44.00 [39.80, 48.20]) | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/scale-gate/VERDICT-B.txt` |
| ns-23 | 2026-11-17 | deepen microbench: wall time per decision at budget 1x / 4x / 8x | deepen b1024 / b4096 / b8192 vs lookahead2 0.94 s | 120 positions |  | 5.0 [M] | - | info | `/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/dbench2-20261117/RESULT` |

### Calibration (harnesses, opponent strength, value) (5 rows)

| id | date | question | A vs B | harness | games / pairs | A's win % [95%] | prereg (md5) | verdict | src |
|---|---|---|---|---|---|---|---|---|---|
| cal-01 | 2026-11-09 | in-process vs real Showdown: G9p@25M vs P1 | in-process 65.9 +- 1.1 (6000 g) vs Showdown 65.7 +- 2.7 (1008 g) | spd_h2h vs netv2 h2h |  | 0.2 [-2.8, 3.2] [M] | - | in-process valid (detects biases >= ~4 points); pre-registered by wsp (file not located) | `memory benchmark-method.md; Showdown side /vol/bitbucket/fr1226/porygon-ws-netv2/runs/h2h-20261109-115722-741849/result.txt` |
| cal-02 | 2026-11-09 | in-process vs real Showdown: P1 vs heuristic | in-process 78.8 +- 1.7 vs Showdown 80.0 +- 3.2 | spd_h2h vs Showdown |  | agree [M] | - | agree | `memory benchmark-method.md` |
| cal-03 | 2026-11-11 | harness offset: raw G9p-cd vs pure search, Stage 3 harness vs netv2 harness | 48.05 (Stage 3, 500 pairs) vs 43.4 (netv2, 500 pairs) | two harnesses, different pairs |  | 4.65 [D] | - | compare arms within one harness | `ns-06 minus v2g9-15` |
| cal-04 | 2026-11-16 | value calibration: v2 net P(win) vs evaluate() on held-out human positions | v2 net AUC 0.752 vs evaluate() AUC 0.718 | offline (NNUE step 2) |  | net is the better leaf [M] | - | net is the better leaf | `docs/netsearch/NNUE-LEAVES-NOTE.md line 16` |
| cal-05 | 2026-11-16 | fpcal: raw cd@805 vs Foul Play at fixed 50k / 800k iterations | raw cd@805 vs Foul Play L 50k / H 800k it. | Foul Play gen9, seed 8800222 | 2000 / 1000 | pending (job 1173956) [E] | wsp fpcal/PREREG.txt (5a68f926) | pending (job 1173956) | `/vol/bitbucket/fr1226/porygon-ws-wsp/fpcal/READING.txt (not yet written)` |

### Diagnostics (5 rows)

| id | date | question | A vs B | harness | games / pairs | A's win % [95%] | prereg (md5) | verdict | src |
|---|---|---|---|---|---|---|---|---|---|
| dg-04 | 2026-11-07 | oracle labels cannot judge the network (gen4) | oracle sided with search vs in 82% of net/search disagreements | offline oracle labels |  | methodological trap [M] | - | methodological trap | `docs/paper/paper.tex 'A methodological trap'` |
| dg-05 | 2026-11-08 | opponent set inference accuracy (gen4, fitted search runs) | whole set right 56-58% vs hidden item 82-88%, move 79-80%, ability 96-97% | benchmark decision logs (4 runs, ~200k samples each) |  | near ceiling [M] | - | near ceiling | `README.md / docs/paper/paper.tex` |
| dg-01 | 2026-11-16 | loss patterns: win rate vs Foul Play by game length (< 25 vs >= 25 turns) | raw cd@805 < 25 turns: 48.4% (353 g) vs >= 25 turns: 62.9% (647 g) | conf800 FP games | 1000 | edge is in long games; tactics weakness (<= +5 pts if fixed) [M] | - | edge is in long games; tactics weakness (<= +5 pts if fixed) | `/vol/bitbucket/fr1226/porygon-ws-wsp/losses/REPORT.md (md5 336c0f99)` |
| dg-02 | 2026-11-16 | loss patterns M1: cd@805 vs itself (logged) | cd@805 vs cd@805 | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 3000 / 1500 | 49.1 [47.6, 50.6] [M] | - | sanity | `/vol/bitbucket/fr1226/porygon-ws-wsp/losses/REPORT.md` |
| dg-03 | 2026-11-16 | loss patterns M2: cd@805 vs BC-P3 | cd@805 vs BC-P3 | in-process fastenv simulator (spd_h2h / netv2_cmp_step), duplicate pairs | 2000 / 1000 | 72.8 [71.0, 74.6] [M] | - | info | `/vol/bitbucket/fr1226/porygon-ws-wsp/losses/REPORT.md` |

**Note on the gloria benchmark networks.** Runs in `/vol/bitbucket/fr1226/gloria/runs/foulplay-*` do not log the prior
network or the commit (memory `benchmark-method.md`). [E] from the README history and the paper: the 400M v1 net
(`runs/20261024-135245-300664`) for job 300964 (2026-10-25); the 500M net (`runs/20261025-195826-301040`, the benchmark
default) from 2026-10-26 to 11-03; the 600M net (`runs/20261026-082356-301120`) for the compute-matched rows 380872–9
(paper.tex: "the 600M-step one in the compute-matched rows"; the paper itself flags the stock-eval rows as `\verify`).
The gen9 runs (515841–52) use no network (MODEL none, prior 0).

**Corrections (append-only).** ~~gen4 fitted eval vs Foul Play 59.2% ± 1.1% (README @ c6b56b8, 2026-11-03)~~ → 59.2% ±
2.2% (pair-mean 95% over 800 pairs; the ±1.1 was a standard error). ~~fitted search vs stock search 54.9% ± 1.6%~~
(README @ a571e75) → 55.0% ± 3.2% pair-mean (54.9 ± 3.4 per game in the paper; `search-20261102-220256-308659`).
Early pooled default search "52.6% ± 2.2%" (2026-10-27) is binomial per game: those five runs predate the duplicate
server. The README rows "R1b-F @150M vs v1@500M 56.4% ± 2.8%" and "R1b-F @100M 52.6%" match the files; G9r-cd "49.5% ±
1.6%" and G9t-cd "48.1% ± 1.4%" in the README are the all-pairs naive numbers, the decisions of record are the
sequential tests in rows g9q-05 / g9q-10.

## Voided, failed or smoke runs (not results)
- Smokes (2–8 games, harness checks only): `$F/runs/netsearch-smoke-20261109-*-841429` (3), `netsearch-fpsmoke-20261110-
  171918-986503`, `netsearch-rfsmoke-20261113-104539-1141881`, `la2-presmoke-20261115`, `la2-bet1/smoke-run`,
  `la2-fp/smoke-run`, `la2-h/smoke-run`, `scale-gate/smoke-B-run`; `$C/runs/vs-search-20261112-021143-1127714` and
  `vs-search-20261115-175226-1172540` (8 games each); gloria `foulplay-20261107-025934-492059` (gen9, 8 games);
  `/vol/bitbucket/fr1226/porygon-ws-data/runs/evalnet-*-677193` (4-game dry runs of the data stream's eval wrappers);
  NNUE `gate1-smoke` (FAIL on 20k positions, superseded by gate 1).
- Void: `$C/runs/h2h-20261114-152241-1136569` ("no games": websocket-login hang, killed by the hang guard and replayed
  as `h2h-20261114-154510/154746-1136569`); R1b-F hold #1 T2 (Showdown, informational) "void".
- Failed / incomplete: gloria `foulplay-*-301121`, `-301734`, `-302004` (no results.jsonl); wsp harness tests
  `porygon-ws-wsp/spd/fproot/runs/*` (757215 53.0%/100 g, 762714 37.0%/100, 762977 36.7%/60, 828361 47.5%/60, 757217
  failed: Foul Play harness readiness tests for wsp's poll fix, not strength results).
- netv2 RL guard trips and park decisions are training events, recorded in `runs.md` / `timeline.md`.

## Negative and null results (part of the contribution)
Everything below either lost, tied within its pre-registered resolution, or failed its gate. Read with the
resolution in mind: 400-game runs are ±4.4–4.9 points; sequential gain tests resolve ~2 points (θ₁ 52%).

Main lessons, each backed by rows here:
1. **Search-side ideas that do not beat plain determinized MCTS against Foul Play (gen4):** 8 determinizations, root
   PUCT (c 2 / 8), regret matching, Bayesian root game (sampled: −13 points), prior weight 0.5 / 1.0, v1 value
   lookahead. Only the fitted leaf evaluation moved the number (+4.7 ± 2.7).
2. **Fitting the evaluation helps in gen4 but not measurably in gen9** (+1.6 ± 3.4 vs Foul Play although engine
   self-play said +10): engine-only head-to-heads over-predict Showdown gains under hidden information.
3. **v1 network training plateaued:** +40M at the lr floor, boosted embedding learning (×30, ×300), SWA, weight-space
   edits, expert iteration (distilled net tied its source).
4. **Search labels as targets hurt** (34.9% vs human BC); neither a weak prior nor 2× search time improved the teacher.
5. **v2 recipe arms that did not help (cooled vs cooled):** power-law lr anneal (G9r), GAE λ 0.5 (G9t, worse), entropy
   schedule (G9qent), bf16 (1.02× speed), EMA weights, actor-stream overlap (passed short non-inferiority, failed the
   25M drift check: 46.0 vs 49.6 control). Seed-to-seed variance at matched steps < ~2 points (G9q2).
6. **Raw-net gains did not always carry over:** G9p-cd's 50.5% vs search over 400 games fell to 43.4% over 1000; G9q-cd's
   in-process gain did not carry into the hybrid (re-freeze #1: 50.7%); cd@805 beat cd@605 net-vs-net but not vs Foul
   Play (57.8 vs 58.8); BC-P4 ≈ BC-P3.
7. **Depth routes that failed their gates:** V3 (GPU net in an MCTS tree: ~2,115 leaves/decision vs 10,000 needed),
   NNUE gate 1 (CPU student no better than evaluate() at determinized human leaves), lookahead2's paired gain vs Foul
   Play not shown (+1.3 ± 4.0), and deepen at 1× not better than lookahead2 (T2 FUTILITY).
8. **Gen4 search prior:** the stronger v2 net (R1b-F) as prior did not clearly beat the v1@500M prior (twice inconclusive).

| id | what | A's win % [95%] | outcome | verdict |
|---|---|---|---|---|
| v1-07 | raw v1 net vs Foul Play gen4 (raw v1@91M vs Foul Play gen4) | 16.5 | negative | raw net far weaker |
| s4-01 | one-turn value lookahead (v1 value) alone vs raw net (lookahead (v1@400M value) vs raw v1@400M) | 53.3 [47.7, 58.9] | null | not better than search |
| s4-02 | value lookahead mixed 0.5 into the search vs the search (search + value_weight 0.5 vs search) | 53.7 [48.1, 59.3] | null | no clear gain |
| s4-03 | value lookahead replacing the engine search (lookahead only vs search) | 42.7 [37.1, 48.3] | negative | worse |
| v1-12 | expert iteration round 1 (9k search battles distilled) (distilled net (KL beta 3) vs its source net) | 50.5 | null | at best tied its source |
| fpr-301119 | one-turn v1 value lookahead in the search (0.5) (SearchPlayer 4x1 s, v1 prior (net not logged), value_weight=0.5 vs Foul Play gen4) | 49.7 [44.8, 54.6] | null | no gain |
| fpr-301735 | network prior weight 1.0 (SearchPlayer 4x1 s, v1 prior (net not logged), prior_weight=1.0 vs Foul Play gen4) | 49.6 [45.3, 53.9] | null | no gain |
| fpr-302022 | network prior weight 0.5 (SearchPlayer 4x1 s, v1 prior (net not logged), prior_weight=0.5 oracle=None vs Foul Play gen4) | 54.2 [49.7, 58.7] | null | no gain |
| s4-04 | patched (durations) vs stock poke-engine, same search (search, patched engine vs search, stock engine (.venv-stock)) | 49.1 [45.0, 53.2] | null | neutral: a correctness fix |
| fpr-302877 | 8 determinizations instead of 4 (2x compute) (SearchPlayer 4x1 s, v1 prior (net not logged), samples=8 oracle=None vs Foul Play gen4) | 53.6 [49.3, 57.9] | null | no gain |
| fpr-303472 | root PUCT c 8 on the prior (SearchPlayer 4x1 s, v1 prior (net not logged), prior_weight=0.0 c_puct=8.0 oracle=None vs Foul Play gen4) | 53.2 [49.4, 57.0] | null | no gain |
| fp4-09 | root PUCT c 2 pooled (2 runs) (search, prior 0, c_puct 2 vs Foul Play gen4) | 55.0 [51.76, 58.24] | null | no change from default |
| fpr-307072 | Bayesian root game, sampled (SearchPlayer 4x1 s, v1 prior (net not logged), root_game=sample oracle=None vs Foul Play gen4) | 40.8 [36.3, 45.3] | negative | -13 points: harmful |
| fpr-307073 | Bayesian root game, argmax (SearchPlayer 4x1 s, v1 prior (net not logged), root_game=max oracle=None vs Foul Play gen4) | 52.9 [48.5, 57.3] | null | no gain |
| fpr-307076 | regret matching in the tree (rm_gamma 0.1) (SearchPlayer 4x1 s, v1 prior (net not logged), rm_gamma=0.1 oracle=None vs Foul Play gen4) | 53.4 [49.1, 57.7] | null | no gain |
| fpr-307077 | regret matching + sampled root game (SearchPlayer 4x1 s, v1 prior (net not logged), root_game=sample rm_gamma=0.1 oracle=None vs Foul Play gen4) | 41.8 [37.3, 46.3] | negative | harmful |
| fp4-03 | stock weights rescaled x2 (search, stock x2 vs Foul Play gen4) | 54.47 [52.17, 56.76] | null | rescaling alone: no gain |
| s4-11 | luck-adjusted vs raw-outcome fit targets (raw-outcome fit (full fe) vs luck-adjusted fit) | 52.1 [49.9, 54.3] | null | luck adjustment brings nothing |
| s4-12 | boost edits to the fitted eval (2 arms) (evalfit_wsp_boost_half / _matched vs fitted eval) | 49.3 [47.7, 50.9] | null | no gain (second arm 49.8) |
| v1-13 | +40M PPO at the lr floor from 600M (control arm) (arm-ctrl 377837 final vs v1@600M (301120)) | 50.0 [48.45, 51.55] | null | no gain |
| v1-14 | embedding lr x30 (+40M from 600M) (arm-x30 371651 final vs v1@600M) | 50.6 [49.05, 52.15] | null | no gain |
| v1-15 | embedding lr x300 (+40M from 600M) (arm-x300 371541 final vs v1@600M) | 48.5 [46.95, 50.05] | null | no gain |
| g9s-01 | gen9 pure search, stock eval (4 runs) (search:none 4x1 s, stock vs Foul Play gen9) | 51.72 [49.36, 54.08] | null | ~even |
| g9s-03 | gen9 pure search, fitted eval_gen9_r1 x2 (search:none, fitted vs Foul Play gen9) | 53.28 [50.93, 55.64] | null | fit +1.6 +- 3.4 over stock: not measurable |
| v1-16 | v1 600M vs 500M (v1@600M vs v1@500M) | 51.1 [46.7, 55.5] | null | no measurable gain |
| v1-18a | weight-space edit of v1@500M (switch-matchup key edit s1) (v1@500M + switch-matchup key edit s1 vs v1@500M) | 49.9 [48.5, 51.3] | null | no gain |
| v1-18b | weight-space edit of v1@500M (switch-matchup key edit s2) (v1@500M + switch-matchup key edit s2 vs v1@500M) | 49.2 [47.8, 50.6] | null | no gain |
| v1-18c | weight-space edit of v1@500M (switch-matchup key edit s3) (v1@500M + switch-matchup key edit s3 vs v1@500M) | 45.9 [44.5, 47.3] | negative | harm |
| v1-20 | SWA of late checkpoints (SWA(500M..600M) vs v1@600M) | 50.0 [48.6, 51.4] | null | no gain |
| v1-21 | SWA of late checkpoints (SWA(500M..600M) vs v1@500M) | 50.8 [49.3, 52.3] | null | no gain |
| v1-22 | L2H6 mean-ablation edit (v1@500M edited vs v1@500M) | 50.6 [49.2, 52.0] | null | no gain |
| v1-23 | L2H6 mean-ablation edit (v1@600M edited vs v1@600M) | 50.4 [49.0, 51.8] | null | no gain |
| v1-24 | raw v1 net vs Foul Play gen4 (default T) (raw v1@600M (301120) vs Foul Play gen4 (8f541f8)) | 44.8 [41.1, 48.5] | negative | raw v1 below Foul Play |
| v1-25 | raw v1 net vs Foul Play gen4 at T 0.5 (raw v1@600M T 0.5 vs Foul Play gen4 (8f541f8)) | 46.2 [42.4, 50.0] | negative | raw v1 below Foul Play |
| bc-12 | search labels as BC targets (P2 fine-tuned on 4x500 ms search labels) vs P1 (BC P2ft (search labels) vs P1) | 34.9 [32.1, 37.7] | negative | search labels hurt |
| bc-13 | human + search-label mix vs P1 (BC P2-mix vs P1) | 46.0 [43.1, 48.9] | negative | no gain |
| bc-14 | search-label net at sharper T 0.28 vs P1 (sequential) (BC P2ft T 0.28 vs P1 T 0.5) | 47.34 [45.49, 49.24] | negative | sequential design runs/cmp/seq-gain.json; P1 > P2ft SUCCESS (MUE 52.65 [50.76, 54.51]) |
| g9s-06 | label teacher: G9p@17.5M prior in gen9 search vs pure search (500 ms) (search G9p prior 0.25, 4x500 ms vs search:none 4x500 ms) | 51.5 [47.0, 56.0] | null | no gain from a weak prior |
| g9s-07 | label teacher: more search time (1000 vs 500 ms) (search:none 4x1000 ms vs search:none 4x500 ms) | 53.5 [49.0, 58.0] | null | not clearly better |
| v2g9-03 | G9s (from scratch) @25M vs pure gen9 search (seed 738164) (G9s (from scratch) @25M T 0.5 vs pure gen9 search) | 26.8 [22.6, 31.0] | negative | below 50% |
| v2g9-04 | G9p @25M vs pure gen9 search (seed 738164) (G9p @25M T 0.5 vs pure gen9 search) | 39.8 [35.2, 44.4] | negative | below 50% |
| v2g9-05 | G9s @50M vs pure gen9 search (seed 738164) (G9s @50M T 0.5 vs pure gen9 search) | 36.8 [31.8, 41.8] | negative | below 50% |
| v2g9-06 | G9p @50M vs pure gen9 search (seed 738164) (G9p @50M T 0.5 vs pure gen9 search) | 42.5 [38.2, 46.8] | negative | below 50% |
| v2g9-07 | G9p @100M vs pure gen9 search (seed 738164) (G9p @100M T 0.5 vs pure gen9 search) | 43.5 [39.1, 47.9] | negative | below 50% |
| v2g9-09 | G9s vs G9p step-matched @10M (G9s @10M vs G9p @10M) | 31.0 [25.29, 36.71] | negative | G9p > G9s SUCCESS (P1 init helps) |
| v2g9-10 | G9s vs G9p @25M (G9s @25M vs G9p @25M) | 33.0 [27.39, 38.61] | negative | G9p > G9s SUCCESS |
| v2g9-11 | G9s vs G9p @50M (G9s @50M vs G9p @50M) | 35.5 [29.07, 41.93] | negative | G9p > G9s SUCCESS |
| v2g9-12 | G9s vs G9p, latest snapshots 11-09 12:19 (G9s latest vs G9p latest) | 45.26 [42.56, 48.05] | negative | G9p > G9s SUCCESS (MUE 54.74 [51.95, 57.44]); G9s parked |
| v2g9-18 | EMA weights vs raw weights (G9p @65.97M) (G9p EMA vs G9p raw) | 51.4 [49.4, 53.4] | null | no clear gain |
| v2g4-03 | R1b-F @50M vs v1@500M (best v1) (R1b-F @50M vs v1 @500M) | 49.8 [46.9, 52.7] | null |  |
| v2g9-08 | G9s @75M vs pure gen9 search (seed 738164) (G9s @75M T 0.5 vs pure gen9 search) | 39.8 [35.3, 44.3] | negative | below 50% |
| v2g9-15 | G9p-cd vs pure gen9 search, 1000-g confirmation (new seed 738201) (G9p-cd T 0.5 vs pure gen9 search) | 43.4 [40.6, 46.2] | negative | 400-g 50.5 not confirmed |
| v2g9-21 | actor-stream long-horizon drift: G9p after +25M with actor-stream vs its start (D25 (actor-stream) vs D (start)) | 46.0 [44.0, 48.0] | negative | drift (control 49.6 +- 2.0) => actor-stream REVERTED |
| g9q-03 | G9q @112.5M vs G9p @338.6M (3x steps) (G9q @112.53M vs G9p @338.55M) | 49.4 [47.4, 51.4] | inconclusive | INCONCLUSIVE (G9q matches G9p at 1/3 the steps) |
| ns-08 | Stage 3b: pure gen9 search vs Foul Play gen9 (pure gen9 search vs Foul Play gen9) | 51.6 [48.8, 54.4] | null | Q2: H - search +4.6 +- 3.9 [+0.7, +8.5] SUCCESS |
| v2g4-05 | R1b-F @100M vs v1@500M (best v1) (R1b-F @100M vs v1 @500M) | 52.6 [49.7, 55.5] | null |  |
| v2g9-24 | G9p plateau: @265M vs @186M (G9p @265.03M vs G9p @186.08M) | 51.8 [49.8, 53.8] | inconclusive | INCONCLUSIVE |
| g9q-04 | G9r (power-law lr decay) 25M harm guard vs G9q (G9r @25M vs G9q @25M) | 49.86 [47.68, 52.17] | null | no harm => continue |
| g9q-05 | G9r-cd vs G9q-cd @50M (rule 2, decision of record) (G9r-cd @50M vs G9q-cd @50M) | 48.12 [45.04, 51.37] | null | UNDECIDED (both FUTILITY) => no gain; G9r parked |
| ns-12 | V3 I1: batched GPU leaf inference under slot contention (best: E16 B 384 vs target 35,300 leaves/s) | 7465 | negative | FAIL => V3 parked |
| v2g4-11 | gen4 search prior: R1b-F@150M vs v1@500M (prior test #1) (search + R1b-F@150M prior vs search + v1@500M prior) | 51.1 [49.1, 53.1] | inconclusive | inconclusive => keep v1 prior |
| g9q-09 | G9t (GAE lambda 0.5) 25M vs G9q (G9t @25M vs G9q @25M) | 49.9 [48.0, 51.8] | null | CONTINUE (both FUTILITY) |
| g9q-10 | G9t-cd vs G9q-cd @50M (rule 2) (G9t-cd @50M vs G9q-cd @50M) | 48.1 [46.7, 49.5] | negative | STOP: G9q-cd > G9t-cd SUCCESS (MUE 51.71 [50.25, 53.16]); G9t parked |
| ns-09 | re-freeze #1: H(G9q-cd@305, d 0.0172) vs H(G9p-cd) (H(G9q-cd@305) vs H(G9p-cd) benchmark bot) | 50.7 [47.91, 53.49] | null | NOT ADOPTED |
| g9q-16 | Milestone 1 attempt: raw G9q-cd@605M vs pure gen9 search (1000 g) (G9q-cd@605M vs pure gen9 search) | 52.5 [49.6, 55.4] | inconclusive | NOT CONFIRMED (+9.1 +- 3.9 over G9p-cd on the same pairs) |
| g9q-18 | seed variance: G9q2 (same recipe, new seed) vs G9q @25M (G9q2 @25M vs G9q @25M) | 50.2 [48.5, 51.9] | null | both FUTILITY |
| bc-15 | P4 data lever: BC-P4 (12.0M, own rating >=1900) vs BC-P3 (BC-P4 vs BC-P3) | 49.95 [47.81, 52.23] | null | B1 both FUTILITY: within ~2 points (informational) |
| g9q-19 | seed variance: G9q2 vs G9q @50M (G9q2 @50M vs G9q @50M) | 50.7 [48.5, 52.9] | null | both FUTILITY: seed variance < ~2 points |
| g9q-20 | bf16 autocast speed (A40, G9q recipe) (bf16 vs fp32) | 101.8 | null | SKIP (needs >= 1.15x): stay fp32 |
| g9q-22 | entropy schedule 0.01 -> 0.001 (G9qent), cooled vs cooled (G9qent-cd @880M vs G9q-cd @885M) | 50.64 [48.92, 52.48] | null | NO GAIN (both FUTILITY) |
| ns-16 | NNUE gate 1: 155k CPU student vs teacher and evaluate() (set H, human, K 4 dets) (student AUC 0.714 vs evaluate() AUC 0.718 / teacher 0.748) | FAIL (Bet 2 as designed stops) | negative | FAIL (Bet 2 as designed stops) |
| ns-17 | NNUE gate 1 set S (self-play, true State) (student AUC 0.796 vs evaluate() 0.773 / teacher 0.834) | c2 met, c1 not met | negative | c2 met, c1 not met |
| v2g4-12 | gen4 search prior test #2: R1b-F@300M vs v1@500M (search + R1b-F@300M prior vs search + v1@500M prior) | 51.9 [50.0, 53.8] | inconclusive | INCONCLUSIVE => keep v1@500M prior |
| v2g4-13 | R1b-F hold #1: guard trip @390.2M vs R1b-F@300M (R1b-F trip @390.2M vs R1b-F @300M) | 49.9 [47.9, 51.9] | null | CONTINUE (not clearly worse; not better) |
| ns-20 | LA2-FP Q2: paired gain of lookahead2 over the raw net vs Foul Play (lookahead2 - raw (points) vs ) | 1.3 [-2.71, 5.31] | null | NOT SHOWN |
| ns-21 | scale gate T2: lookahead2 vs deepen D1 (1,024 extra leaves) (lookahead2 (cd@805) vs deepen D1) | 49.13 [45.41, 53.21] | null | FUTILITY: 1x not shown worse than lookahead2 (naive 48.50 [45.03, 51.97]) |
