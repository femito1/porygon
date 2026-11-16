# Scale gate: does the deepened lookahead get stronger with thinking time? Pre-registration (fastenv, 2026-11-17; DRAFT until main signs off, then binding and read-only)
Main's decision (11-16 ~23:10Z, with the user's go-ahead): the next depth route is a net-guided search that gets
stronger with more thinking time. The design is `SCALE-SEARCH-DESIGN.md` (algorithm, budgets, microbench); the
prototype is `porygon.deepen` (best-first selective deepening on top of lookahead2; budget 0 = lookahead2 exactly).
This file is written BEFORE any game with deepen. Nothing is launched until main signs off this file's md5.
Records and deviations after sign-off go in `$C/runs/scale-gate/NOTES.txt`, never here. The cluster copy
`$C/runs/scale-gate/PREREG.txt` will be byte-identical to this file.

## Questions
- **T1 (the decision): does more budget buy strength?** D4 (extra budget 8,192 leaves) vs D1 (2,048), head to head.
- **T2: is the 1× search at least not worse than lookahead2?** lookahead2 (L) vs D1, head to head.
- **T3 (anchor, informational, no rule):** the raw net vs D1.
- Not in this gate (a follow-up after a GO, its own prereg): D0.5 (1,024) and D2 (4,096) vs D1 to draw the curve,
  and the vs-Foul-Play / vs-benchmark-bot confirmations.

## Players (one net, one frozen tree)
- Net `$C/nets/g9q-cd-805.03M.pt`, sha256 `e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f`, loaded
  through `$C/frozen/netv2-fba02fb-20261110` for every player and every leaf. October server and October in-tree
  encoder library (md5 324d8e97), as every Stage 3 / re-freeze / Bet 1 / LA2 run (G9q trains on the November sets:
  the recorded train/serve deviation, the same for all arms).
- **D_B** = `deepen:$NET,budget=B,margin=δ_B,temperature=0.5,worlds=8,workers=10,leaf_device=cuda` with the
  `porygon.deepen.Config` defaults: wave 1024; our options by the net's policy until 0.9 of the mass, ≤ 3; theirs by
  the opponent-action head until 0.9, ≤ 4, ≥ 2; branches ≥ 3% below the root (all at the root); max 4 turns; 200
  regret-matching iterations between waves, 300 for inner nodes at the end, 1000 at the root; **inner-node backup
  nash = 0** (the expected value under the net's priors over the kept options; see "Backup" below); fp32 leaves.
- **L** = `lookahead:$NET,margin=0.005,temperature=0.5,worlds=8,workers=10,leaf_device=cuda` (LA2-FP's arm with 10
  workers; δ* = 0.005 from `$C/runs/la2-delta800/DELTA`). It is D_0: the microbench checked that deepen at budget 0
  reproduces lookahead2's rows, Q and x exactly (max |ΔQ| = 0, 120 / 120 positions).
- **raw** = `$NET,temperature=0.5` (netv2's NetPlayer).
- Code: ws/fastenv commit (filled at sign-off) with `src/porygon/deepen.py` sha256 (filled at sign-off).

## δ_B (Stage 0 of the chain, no games; the lookahead2 rule unchanged)
The margin rule of `LA2-BET1-PREREG.md`, applied to each deepen arm: on the 501 positions of S2 + S2c
(`$C/netsearch/study-20261109/{s2-sl-ARP,s2c-hu-AR}.jsonl`), per position A = deepen(K 8, B) and judge R =
deepen(K 32, 4B) on fresh worlds (`deepen_bench.py delta`), both with the same a_net (the net's T 0.5 draw, seeded per
position). **δ_B = the smallest grid margin ≥ 0 from which on (every larger finite grid margin too) ≥ 90% of A's
overrides are confirmed by R** (R's Q of A's pick > R's Q of a_net); grid as Bet 1's (0, 0.0025, …, 0.15). The rule
removes world-sampling and selection noise only; the games decide strength.
- If no grid margin passes, δ_B = 0.15 (the largest finite grid margin) and it is reported.
- D1 and D4 each get their own δ_B (2,048 and 8,192). L keeps δ* = 0.005, derived by the same rule.
- Why per budget: deeper trees change Q's noise and selection bias (v1 microbench: overrides 15% → 23-26% at the
  same δ), so one δ for all budgets would confound budget with gate calibration. Each arm is "the same search with
  its gate calibrated by one rule", the way δ is re-derived for a new net.

## Backup (fixed before any game)
Inner nodes back up the expected value under the net's priors over the kept options (nash = 0), not the Nash value.
Reason (v1 microbench, job 1173957, 120 positions, code 7164e48): with the Nash backup, Q(a_net) fell vs lookahead2
by −0.018 (mean) at 1,024 extra leaves and −0.023 at 2,048, and the override rate at δ 0.005 rose from 15% to 26% /
23%: min / max over pruned options of noisy leaf values biases the deepened rows, and the gate reads that as a gap.
The priors backup is linear in the leaf values (no selection inside the tree); the root keeps lookahead2's exhaustive
equilibrium, so the adversarial choice where it matters (this decision) is unchanged. The microbench v2 (job
1173960) reports both backups at 2,048 and 8,192 for information; the choice above does not depend on it.

## Harness (Showdown, the slot)
- **Why not in-process:** every searching player reads poke-env `Battle` objects off the Showdown protocol
  (`search.convert`, `History`, `v3leaf.LeafRoot`); fastenv's JS tracker gives no such object, and PLAN's rule is that
  anything involving our search runs on Showdown. Both sides are deterministic functions of their inputs and seeds
  (budgets are in leaves, no clock), so the harness's speed changes hours, not decisions.
- `scripts/fastenv/netsearch_slot.sh` (Stage 3 harness; `netsearch_h2h.py` accepts `deepen:`): OPP = D1, ARMS =
  (D4, L, raw). 3 concurrent battles, blocks of 100 duplicate pairs, arms rotated per round, block j on a fresh server
  seeded `scale20261117b<j>`, so every arm plays the same team pairs. No battle timer (our server, our players).
- **The benchmark slot**: one 24-CPU `--partition=training` job with one T4 (each search side its own 10-worker pool;
  the two leaf nets share the T4), claimed in `ledger.md` after LA2-H (≈ 15Z 11-17). If the slot stays busy, the
  same chain may run on fastenv's helper (12 CPUs, 5 workers per side, same decisions, slower) with main's OK.

## Test (internal go/no-go: sequential, as Bet 1)
- **Design:** Bet 1's group-sequential design, byte copy `scripts/fastenv/plans/seq-la2-bet1.json` (md5
  f0911c195e850d1e11ea11359c0b0100): onesided, θ0 50%, θ1 54%, α 0.025 one-sided, β 0.10, looks every 100 pairs,
  n_max 800 pairs; HSD(−4) / HSD(−2) spending; futility non-binding; plug-in σ. After every block, `vendor/seq_monitor.py
  --order sequential --final --a-prefix pa --design …` on that arm's blocks in block order; an arm stops at its own
  decision (SUCCESS or FUTILITY); at most 9 blocks per arm (the 9th only if pairs were lost before 800).
- Each arm's statistic is its player's pair score vs D1.
- **T1 (arm 1, D4 vs D1):** SUCCESS = D4 beats D1 (repeated 95% CI excludes 50% from below).
- **T2 (arm 2, L vs D1):** SUCCESS = lookahead2 beats D1 ⇒ "1× WORSE than lookahead2"; FUTILITY or the last look
  without SUCCESS ⇒ "1× not shown worse".
- **Arm 3 (raw vs D1):** informational; it stops by the same monitor; its median-unbiased estimate is reported.

## Decision of record
- **GO** iff T1 SUCCESS and T2 not SUCCESS: strength rises with budget and the 1× search is not worse than
  lookahead2 ⇒ propose (separate preregs) the curve arms (D0.5, D2), the vs-Foul-Play / vs-benchmark-bot runs at the
  deployable budget, the CUDA-graph / fp16 speed work, and deepen as the expert-iteration teacher.
- **MIXED** iff T1 SUCCESS and T2 SUCCESS: depth scales but the 1× search is worse than lookahead2 ⇒ report; main
  decides (e.g. D4 vs L directly).
- **FLAT** iff T1 FUTILITY, or the last look without SUCCESS ⇒ **stop this route**; report the diagnostics (override
  rate and gap per side, depth, where D4 and D1 disagree, outcome by override gap).
- **VOID** iff HARNESS_FAIL, > 2% fallbacks on any searching side (an arm or the opponent D1), no decision on T1, or
  the δ stage FAILED ⇒ the chain writes FAILED; replay with the same seeds after the fix (main's OK).

## Reported with the verdict
Per arm: the median-unbiased estimate with its 95% CI (stage-wise ordering) at the stopping look, the naive
mean ± 1.96 sd/√n, pairs, won-both / split / lost-both. Per searching side: override rate, median gap of overrides,
fallbacks, and per searched decision the leaves / extra leaves / depth / wall seconds / forward seconds (median, q90,
max). δ_2048 and δ_8192 with their override rates and precision. Hours. Informational: the composite D4-vs-L reading
(T1 and T2 on the same team pairs), not a test.

## Chain (one cluster-side job, no agent; written by the launch agent after sign-off from `lookahead2_slot.sbatch`)
`scripts/fastenv/scale_gate.sbatch`, code frozen into `$C/runs/scale-gate/code` at submission:
1. checks: PREREG md5 = `PREREG.md5`; net sha256; ≥ 24 CPUs; a GPU; δ* of L from `la2-delta800/DELTA` = 0.005;
   design md5 f0911c19…;
2. **δ stage** (Stage 0): `deepen_bench.py delta` at B 2,048 and 8,192 on S2 + S2c; `DELTA_2048`, `DELTA_8192`
   (one number each) + `delta-*.txt`; a failure (> 2% failed positions) writes FAILED;
3. **smoke** (tag `scalesmoke20261117`, 2 pairs per arm, read mechanically): 4 games per arm, no Traceback, no
   HARNESS_FAIL, both searching sides searched with leaves > 0 and gate rows > 0, extra leaves > 0 on the deepen
   sides, 0 failed worlds, fallbacks ≤ max(2, 2%) per side; FAIL stops the chain;
4. main run `runs/scale-gate/main-run` with `SEQ_DESIGN` (the copy above), BLOCKS 9;
5. `VERDICT.txt` (r--, line 1 = GO / MIXED / FLAT / VOID: …), `RESULT`, the ledger RELEASE line in
   `$C/LEDGER-OUTBOX.md`.
Stage files `runs/scale-gate/{STATUS, DELTA_2048, DELTA_8192, SMOKE.txt, VERDICT.txt, RESULT, FAILED}`.

## Expected time
TIME_PLACEHOLDER

## Caveats (known before sign-off)
- All arms use the same net (self-play family): a GO is a scaling claim, not a strength claim vs other bots; the
  vs-Foul-Play and vs-benchmark-bot confirmations are separate fixed-n slot runs.
- Budgets are in leaves; the wall time they cost on the slot (microbench, `SCALE-SEARCH-DESIGN.md` §6) maps them to
  thinking time. The deployable budget under Showdown's timer (+10 s per turn) is chosen after the gate.
- Two searching sides share one T4 (and the node's CPUs): slower than deployment, same decisions.
- T2 at θ1 54% does not detect a small loss of 1× vs lookahead2 (≲ 2 points); its estimate is reported.
- The opponent head used for pruning was trained on G9q's own games' opponents (self-play); nothing models Foul Play.
- 10 workers per side instead of the hybrid's 11 (two pools on 24 CPUs); speed only.
