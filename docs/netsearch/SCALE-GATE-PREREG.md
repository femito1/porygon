# Scale gate: does the deepened lookahead get stronger with thinking time? Pre-registration (fastenv, 2026-11-17; signed off by main 11-17 ~00:30Z with the chain-A dependency amended; binding and read-only)
Main's decision (11-16 ~23:10Z, with the user's go-ahead): the next depth route is a net-guided search that gets
stronger with more thinking time. The design is `SCALE-SEARCH-DESIGN.md` (algorithm, budgets, microbench); the
prototype is `porygon.deepen` (best-first selective deepening below lookahead2's root game; budget 0 = lookahead2
exactly). This file is written BEFORE any game with deepen; nothing is launched until main signs off its md5.
Records and deviations after sign-off go in `$C/runs/scale-gate/NOTES.txt`, never here. The cluster copy
`$C/runs/scale-gate/PREREG.txt` will be byte-identical to this file.

## Questions
- **T1 (the decision): does more budget buy strength?** D4 (4× = 4,096 extra leaves) vs D1 (1× = 1,024), head to head.
- **T2: is the 1× search not worse than lookahead2?** lookahead2 (L) vs D1, head to head.
- **T3 (anchor, informational, no rule):** the raw net vs D1.
- Not in this gate (follow-ups after a GO, each with its own prereg): D0.5 (512) and D2 (2,048) vs D1 for the curve;
  8× (8,192); vs Foul Play and vs the benchmark bot at the deployable budget.

## Players (one net, one frozen tree)
- Net `$C/nets/g9q-cd-805.03M.pt`, sha256 `e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f`, loaded
  through `$C/frozen/netv2-fba02fb-20261110` for every player and every leaf. October server and October in-tree
  encoder library (md5 324d8e97), as every Stage 3 / re-freeze / Bet 1 / LA2 run (G9q trains on the November sets:
  the recorded train/serve deviation, the same for all arms).
- **D_B** = `deepen:$NET,budget=B,margin=0.005,temperature=0.5,worlds=8,workers=W,leaf_device=cuda` with the
  `porygon.deepen.Config` defaults: wave 1024; our options by the net's policy until 0.9 of the mass, ≤ 3; theirs by
  the opponent-action head until 0.9, ≤ 4, ≥ 2; branches ≥ 3% below the root (all at the root); at most 4 turns;
  inner-node backup nash = 0 (the expected value under the net's priors over the kept options); 200 regret-matching
  iterations at the root between waves, 1000 at the end; fp32 leaves. D1: B = 1,024; D4: B = 4,096.
- **L** = `lookahead:$NET,margin=0.005,temperature=0.5,worlds=8,workers=W,leaf_device=cuda` (LA2-FP's arm; δ* 0.005
  from `$C/runs/la2-delta800/DELTA`). It is deepen at budget 0: the microbench reproduced lookahead2's rows, Q, x,
  leaves and picks exactly on 120 / 120 positions (max |ΔQ| = 0).
- **raw** = `$NET,temperature=0.5` (netv2's NetPlayer).
- W = 10 on the slot (two pools on 24 CPUs), 5 on the helper (two pools on 12 CPUs). The worker count changes speed,
  not the player.
- Code: ws/fastenv commit and `src/porygon/deepen.py` sha256 filled in `NOTES.txt` at sign-off (the chain freezes
  them into its run dir; the manifest records both).

## Fixed before any game, and why
- **One margin, δ = 0.005, for every search arm** (lookahead2's δ* for this net). The microbench (v2, 120 positions)
  found the override rate at δ 0.005 at 12.5-15.8% for every budget from 0.5× to 8× (lookahead2 15.0%), with Q(a_net)
  and Q(pick) shifting together (relative ≤ 0.002): the gate's calibration does not drift with budget, so the budget is
  the only thing that changes between D1 and D4. (`deepen_bench.py delta`, the lookahead2 δ rule for deepen, exists
  and is not part of this gate.)
- **Backup nash = 0.** In the v1 microbench (job 1173957) the Nash backup lowered Q(a_net) vs lookahead2 (mean
  −0.018 / −0.023 at 1× / 2×) and raised the override rate to 26% / 23%; in v2 it gave 22-27% while nash = 0 stayed
  at 14-16%. The choice was made from v1, before v2 and before any game.
- **Budgets 1× = 1,024 and 4× = 4,096** extra leaves: 2.44 s and 5.00 s median per decision (max 4.8 / 10.3 s) in the
  microbench, i.e. inside Showdown's +10 s per turn, so the gate tests budgets a deployed bot can use; 4× is the
  ratio main asked for.

## Harness (Showdown)
- **Why not in-process:** every searching player reads poke-env `Battle` objects off the Showdown protocol
  (`search.convert`, `History`, `v3leaf.LeafRoot`); fastenv's JS tracker gives no such object, and PLAN's rule is that
  anything involving our search runs on Showdown. The search's budget is in leaves (no clock), so the hardware changes
  hours, not the player.
- `scripts/fastenv/netsearch_slot.sh` (Stage 3 harness; `netsearch_h2h.py` accepts `deepen:`), OPP = D1, 3 concurrent
  battles, blocks of 100 duplicate pairs, arms rotated per round, block j on a fresh server seeded
  `scale20261117b<j>`, so every arm in both chains plays the same team pairs. No battle timer (our server, our
  players). Two searching sides share the job's T4.
- **Chain A (the slot):** arm 1 = D4. One 24-CPU `--partition=training` job with a T4, `--dependency=afterany:1173956`
  (wsp's fpcal, which holds the slot claim after LA2-H; two slot jobs never run at once), claimed in `ledger.md` on
  sign-off.
- **Chain B (fastenv's helper):** arms 2 = L and 3 = raw. One 12-CPU `--partition=training` job with a T4, started on
  sign-off (it needs no slot).

## Test (internal go/no-go: sequential, Bet 1's design)
- **Design:** `scripts/fastenv/plans/seq-la2-bet1.json` (md5 f0911c195e850d1e11ea11359c0b0100): onesided, θ0 50%,
  θ1 54%, α 0.025 one-sided, β 0.10, looks every 100 pairs, n_max 800 pairs; HSD(−4) / HSD(−2) spending; futility
  non-binding; plug-in σ. OC at the planning mix: P(SUCCESS) 0.024 at 50%, 0.37 at 52%, 0.91 at 54%.
- After every block, `vendor/seq_monitor.py --order sequential --final --a-prefix pa --design …` on that arm's blocks
  in block order (`SEQ_DESIGN` in the plan); an arm stops at its own decision; at most 9 blocks per arm (the 9th only
  if pairs were lost before 800).
- Each arm's statistic is its player's pair score vs D1.
- **T1 (arm 1, D4 vs D1):** SUCCESS = D4 beats D1 (the repeated 95% CI excludes 50% from below).
- **T2 (arm 2, L vs D1):** SUCCESS = lookahead2 beats D1 ⇒ "1× WORSE than lookahead2"; FUTILITY or the last look
  without SUCCESS ⇒ "1× not shown worse".
- **Arm 3 (raw vs D1):** stops by the same monitor; informational.

## Decision of record
- **GO** iff T1 SUCCESS and T2 not SUCCESS: strength rises with budget and the 1× search is not shown worse than
  lookahead2 ⇒ propose (separate preregs) the curve arms, the vs-Foul-Play / vs-benchmark-bot runs at the deployable
  budget, the speed work (fp16, CUDA graphs, max_turns), and deepen as the expert-iteration teacher.
- **MIXED** iff T1 SUCCESS and T2 SUCCESS: depth scales but the 1× search is worse than lookahead2 ⇒ report; main
  decides (e.g. D4 vs L directly).
- **FLAT** iff T1 FUTILITY, or T1's last look without SUCCESS ⇒ **stop this route**; report the diagnostics (override
  rate and gap per side, depth, where D4 and D1 disagree, outcome by override gap). A FLAT means "4× does not beat
  1× by the ≈ 3-4 points this design detects", not "no gain at all" (P(SUCCESS) 0.37 at a true 52%).
- **VOID** iff HARNESS_FAIL, > 2% fallbacks on any searching side (an arm or the opponent D1), or no decision on T1
  ⇒ FAILED; replay with the same seeds after the fix (main's OK).
- If T1 decides FLAT before chain B has decided, chain A stops chain B (touches B's `arm-K/STOPPED`; the harness
  skips stopped arms) and the verdict says so.

## Reported with the verdict
Per arm: the median-unbiased estimate with its 95% CI (stage-wise ordering) at the stopping look, the naive
mean ± 1.96 sd/√n, pairs, won-both / split / lost-both. Per searching side: override rate, median gap of overrides,
fallbacks; per searched decision leaves / extra leaves / depth / wall seconds / forward seconds (median, q90, max).
Hours. Informational: the composite D4-vs-L reading (T1 and T2 on the same team pairs), not a test.

## Chains (cluster-side, no agent; written by the launch agent after sign-off from `lookahead2_slot.sbatch` and Bet
1's sequential stages)
`scripts/fastenv/scale_gate.sbatch A|B`, code frozen into `$C/runs/scale-gate/code` once (both chains run that copy):
1. checks: PREREG md5 = `PREREG.md5`; net sha256; CPUs (A ≥ 24, B ≥ 12); a GPU; the design md5; the frozen code;
2. smoke (tag `scalesmoke20261117`, 2 pairs per arm, read mechanically): 4 games per arm, no Traceback, no
   HARNESS_FAIL, every searching side searched with leaves > 0 and gate rows > 0, extra leaves > 0 on the deepen sides,
   0 failed worlds, fallbacks ≤ max(2, 2%) per side; FAIL stops that chain (FAILED);
3. main run (`runs/scale-gate/{A,B}-run`, tag `scale20261117`, BLOCKS 9, `SEQ_DESIGN` the copy above);
4. A: after T1's decision, FLAT ⇒ stop B; then wait ≤ 48 h for B's `RESULT-B` (unless FLAT) and write the combined
   `VERDICT.txt` (r--, line 1 = GO / MIXED / FLAT / VOID: …) + `RESULT`; B writes `VERDICT-B.txt` + `RESULT-B`;
5. A's ledger RELEASE line in `$C/LEDGER-OUTBOX.md`; B cancels the helper at its end.
Stage files `runs/scale-gate/{STATUS-A, STATUS-B, SMOKE-A.txt, SMOKE-B.txt, VERDICT-B.txt, RESULT-B, VERDICT.txt,
RESULT, FAILED}`.

## Expected time [e: from the microbench and Bet 1's 34.8 searched decisions per game per side]
- **Chain A (D4 vs D1):** per battle-turn ≈ 5.0 s (D4's median; D1's 2.4 s runs alongside) plus the shared T4 (the
  forwards of both sides ≈ 2.0 s of GPU per battle-turn, i.e. ≥ 71 s of GPU per game) ⇒ ≈ 45-55 games/h at 3
  concurrent battles. E[pairs] 376 at 50%, 546 at 52%, 490 at 54% ⇒ ≈ 15-24 h; n_max (1,600 games) ≈ 30-35 h.
  `--time 60:00:00`.
- **Chain B (L vs D1, raw vs D1, 5 workers per side):** ≈ 35 × 3.5 s / 3 ≈ 40-45 s per game ⇒ ≈ 80 games/h; E ≈ 800-1,100
  games per arm at most ⇒ ≈ 15-25 h for both; worst case ≈ 40 h. `--time 60:00:00`.
- The two chains run in parallel; the verdict is expected ≈ 1-1.5 days after the slot frees (after fpcal ≈ 20Z 11-17).

## Caveats (known before sign-off)
- All arms use the same net: a GO is a scaling claim, not a strength claim vs other bots; the vs-Foul-Play and
  vs-benchmark-bot confirmations are separate fixed-n slot runs.
- Budgets are in leaves; the microbench maps them to seconds (`SCALE-SEARCH-DESIGN.md` §6) on one T4 with 3 decisions
  in flight. Two searching sides on one T4 are slower than that; the players are unchanged.
- T2 at θ1 54% does not detect a small loss of 1× vs lookahead2 (≲ 2 points); its estimate is reported.
- Backed-up values drift up with depth (a common shift across rows, +0.034 at 0.5× to +0.065 at 4×); harmless for
  the decision, noted for later expert iteration.
- The opponent head used below the root was trained on G9q's own games' opponents; nothing models Foul Play. The root
  stays exhaustive and adversarial (lookahead2's equilibrium).
- 10 workers per side on the slot instead of the hybrid's 11; 5 on the helper. Speed only.
