# Results (every evaluation the project produced)

Compiled 2026-11-17 from a sweep of the cluster (`result.txt`, `RESULT*`, `RESULTS.txt`, `VERDICT*`, `READING.txt`,
`OVERALL.txt`, `analysis.txt`, `SKIP.txt`, `summary.txt` under `/vol/bitbucket/fr1226/gloria/runs` and every
`/vol/bitbucket/fr1226/porygon-ws-*`; list in `/vol/bitbucket/fr1226/porygon-thesis/resultfiles.txt`, 222 files, and
`aux-files.txt`), plus results quoted in `README.md` (and its git history), the stream HANDOFFs, `docs/netsearch/`,
`docs/paper/paper.tex` and memory. Machine-readable: `data/results.csv` (one row per result, same ids).

**Totals:** @@COUNTS@@. "Pre-registered" = the row names a PREREG file written before its first game (md5 as on the
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
@@TABLES@@

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
@@NEG@@
