# LA2-2030: lookahead2 on G9q-cd@2030 vs the gen9 benchmark bot, then LA2-FP200: pre-registration (fastenv, 2026-11-23; FINAL, binding and read-only from the ledger claim on)
Main's design (2026-11-23, below verbatim): LA2-1350's adoption test one net later, then (only if it adopts) the new
benchmark bot's Foul Play headline at a fixed 200,000 iterations per state, as one cluster-side slot chain. Written
before any data of this test. Cluster copy `$F/runs/la2-2030/PREREG.txt` (r--, byte-identical), md5 in `PREREG.md5`.
Records and deviations after the claim go in `$F/runs/la2-2030/NOTES.txt`, never here.
`$F` = `/vol/bitbucket/fr1226/porygon-ws-fastenv`.

## Main's design (verbatim)
CONTEXT: netv2's F2030 CONFIRMED (`/vol/bitbucket/fr1226/porygon-ws-netv2/runs/conf2030/RESULTS.txt`): raw G9q-cd@2030 (`/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9q-cd2030/final/02035041391.pt`, md5 55940b3f) scores 60.8% [57.9, 63.7] against Foul Play at a FIXED 200k iterations. It is the deployment net. The benchmark bot is `lookahead:$F/nets/g9q-cd-1355.04M.pt,margin=0.0075,temperature=0.5,worlds=8`.

ONE ITEM: arm ONE slot chain (training partition, 24 CPUs, no agent needed to finish) with these two stages. Main fixes the design; write it verbatim into a new read-only PREREG, `$F/runs/la2-2030/PREREG.txt`, plus a repo copy.
1. **LA2-2030 adoption:** exactly the LA2-1350 design, one net later.
   - Copy the net into `$F/nets/` read-only and record its sha256. Re-read δ inside the chain by Bet 1's precision rule.
   - Then lookahead2(cd@2030, δ*, K 8, T 0.5) vs the current benchmark bot lookahead2(cd@1350, δ 0.0075), 500 pairs fixed n, new seed tag.
   - Reading: lower 95% > 50 ⇒ ADOPTED (BETTER); > 47 ⇒ ADOPTED (NON-INFERIOR, margin 3); else NOT ADOPTED. Output `$F/runs/la2-2030/VERDICT.txt`.
   - The gate is F2030 CONFIRMED (already true; check the file and md5 55940b3f).
2. **LA2-FP200 (only if stage 1 ADOPTS):** the new benchmark bot vs Foul Play gen9 at a FIXED 200,000 iterations per state, 1000 games fixed n, new seed. This is the headline number.
   - Use wsp's fixed-iteration launcher with its iteration check: `/vol/bitbucket/fr1226/porygon-ws-wsp/fpcal/code/fp9_iters.sbatch` 82a207a6, `fp_iters_check.py`. See netv2's `scripts/netv2_conf2030/conf2030.sbatch` on ws/netv2 for how it is driven with our harness. If our lookahead2 player cannot run under that harness, use your slot harness with a fixed-iteration Foul Play launch and the same check, and say so.
   - Report the pair-mean 95% interval. No decision rule (measurement).
   - A Foul Play websocket handshake timeout at start with 0 games is a VOID: replay once with the same seed, automatically.
   - Output `$F/runs/la2-2030/FP200.txt`.

Further rules:
- CLAIM line in `$F/LEDGER-OUTBOX.md` (flock) when you submit, and one RELEASE line at the end. Size `--time` from LA2-1350's measured 6.8 h plus the δ study plus about 2.5 h for 1000 Foul Play games plus margin, and state the basis.
- Laptop mock for the gate and reading logic. At most an in-slot ≤ 8-game smoke, as before.

## Nets
- **Candidate** `$F/nets/g9q-cd-2035.04M.pt` = netv2 `runs/g9q-cd2030/final/02035041391.pt` (G9q's cooldown branch
  from 2030M, ending at 2,035,041,391 steps), copied 2026-11-23 14:59Z, r--; md5 `55940b3f79245514c8444bf57972487c`,
  sha256 `de82e0dd75b0ca79e35938d4085599ffc37c881cb60ea8985ddea198db5df7a6`, equal for both files. It loads through the
  frozen netv2 tree `$F/frozen/netv2-fba02fb-20261110` (checked 11-23 on the login node: gen 9, 5,809,977 parameters,
  as cd@1350).
- **Benchmark bot's net** `$F/nets/g9q-cd-1355.04M.pt`, sha256 `23593a3138cd17a5a010e42ca9cca1604b4ee98081e64df1c4ee7f49533886cd`
  (LA2-1350 ADOPTED (NON-INFERIOR), `$F/runs/la2-1350/VERDICT.txt`).

## Gate
- `lookahead2_adopt.py gate --name F2030 --md5 55940b3f`, read once at the chain's start: PLAY iff line 2 of
  `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/conf2030/RESULTS.txt` contains "CONFIRMED (lower 95% bound > 50%)" and
  not "NOT CONFIRMED", and line 1 names "md5 55940b3f". (Checked 11-23 15:00Z: line 1 names md5 55940b3f, line 2 reads
  60.8% ± 2.9% [57.9, 63.7] => CONFIRMED.) Anything else ⇒ `WITHDRAWN`, `RESULT`, one RELEASE line; nothing played.

## Stage 1: LA2-2030 (LA2-1350's design, one net later)
- **δ\*:** Bet 1's δ rule, unchanged (`LA2-BET1-PREREG.md`, section "δ*"), run exactly as LA2-1350 ran it:
  `lookahead2_delta.sh` with `LA2D_STAGE=$F/runs/la2-2030`, the candidate, `LA2D_OUT=$F/lookahead2/study-cd2035-20261123`,
  `LA2D_PREFIX=DELTA-`, `LA2D_CANCEL=0`; the 1,403 study positions `$F/netsearch/study-20261109`, seed 11,
  `--worlds 8 --retest --ref 32 --workers 10 --concurrency 3`. δ\* = the smallest grid margin ≥ 0 from which on at
  least 90% of the overrides (weighted by the net's T 0.5 draw probability) are confirmed by judge R (K 32), pooled.
  Fewer than 98% usable or no margin meets the rule ⇒ FAILED, nothing played. `DELTA` is written before any game.
- **A** (candidate) = `lookahead:$F/nets/g9q-cd-2035.04M.pt,margin=δ*,temperature=0.5,worlds=8,workers=11,leaf_device=cuda`.
- **OPP** (the benchmark bot) = `lookahead:$F/nets/g9q-cd-1355.04M.pt,margin=0.0075,temperature=0.5,worlds=8,workers=11,leaf_device=cuda`
  (LA2-1350's candidate line).
- **Harness** = LA2-1350's: `netsearch_slot.sh`, 5 blocks of 100 duplicate pairs = 1000 games, fixed n; block j on a
  fresh duplicate server seeded `8233571b<j>` (seed tag **8233571**, unused under `porygon-ws-*`, checked 11-23);
  3 concurrent battles; one client holding both players, each with its own 11 lookahead workers, both leaf nets on the
  job's GPU; October server `$F/.showdown-eval`; frozen netv2 tree fba02fb for both players.
- **Reading** (`lookahead2_adopt.py verdict`, mechanical): A's pair-mean interval over the 500 pairs (a pair scores
  (s0 + s1) / 2; mean ± 1.96 sd / √n), lower bound `lo`, unrounded, strict >. `lo` > 50% ⇒ **ADOPTED (BETTER)**; else
  `lo` > 47% ⇒ **ADOPTED (NON-INFERIOR, margin 3 points)**; else **NOT ADOPTED** (the benchmark bot stays lookahead2 on
  cd@1355.04M, δ 0.0075). VERDICT.txt line 1. ADOPTED (either) ⇒ the gen9 benchmark bot becomes A at δ\*.
- **Void** (LA2-H's rule): HARNESS_FAIL, > 2% fallbacks on either side, or < 450 pairs ⇒ `VOID: …`, no reading,
  FAILED + RELEASE; nothing more is played.
- Operating characteristics (LA2-1350's half-width 2.84 points at 500 pairs): true 47% ⇒ P(ADOPTED) ≈ 2.5%; true 50% ⇒
  P(NON-INFERIOR or better) ≈ 55%; true 52% ⇒ ≈ 94%, P(BETTER) ≈ 28%.

## Stage 2: LA2-FP200 (only if stage 1 reads ADOPTED (BETTER) or ADOPTED (NON-INFERIOR …))
- **Question (measurement, no decision rule):** the new benchmark bot A (stage 1's spec, δ\*) vs Foul Play gen9 at a
  FIXED 200,000 MCTS iterations per state. Reported: A's pair-mean 95% interval over 500 duplicate pairs (1000 games).
- **Harness (deviation from the suggested launcher, as main allowed):** wsp's `fp9_iters.sbatch` plays through netv2's
  `scripts/vs_foulplay.py`, whose `--player` is only `policy` (raw net) or `search`: our lookahead2 player cannot run
  under it. So this stage uses **our slot harness** (`netsearch_slot.sh`, LA2-FP's Stage 3b pattern: our
  `scripts/vs_foulplay.py --player spec`, one player per Foul Play instance on ONE lookahead pool) with Foul Play started
  exactly as wsp's launcher starts it: `$FP/.venv-gen9/bin/python fp_fixed_iters.py --iters-per-s 200000 run.py …
  --search-time-ms 1000 --search-parallelism 1` (PYTHONPATH `$G/.engine-gen9:$G/src`), i.e. 2 states × 200,000
  iterations normally, 4 × 100,000 early. The opponent spec is `foulplay:ms=1000,parallelism=1,iters=200000`.
- **The same iteration check:** wsp's `fp_iters_check.py` on every block's Foul Play logs every 300 s and at the
  block's end (`ITERS_OK` per block). A mismatch stops the block and voids the run (`ITERS_MISMATCH`, HARNESS_FAIL).
  The chain checks the md5s of wsp's files before playing: `fp_fixed_iters.py` f9c32cdf, `fp_iters_check.py` a1ff4886,
  `fp9_iters.sbatch` 82a207a6 (the files netv2's F2030 used).
- **Design:** 5 blocks of 100 duplicate pairs, fixed n, block j on a fresh duplicate server seeded `8233792b<j>` (seed
  tag **8233792**, new, unused); **8 Foul Play instances** (8 concurrent battles; LA2-FP used 3). With Foul Play at a
  fixed iteration count and lookahead2 at a fixed K, CPU contention changes speed, not play; 8 instances + 11 lookahead
  workers + server + client fit 24 CPUs. Block timeout 300 min.
- **Infrastructure VOID, replayed once automatically:** the run has HARNESS_FAIL, 0 games, and a Foul Play instance
  "did not start" with "timed out during handshake" (in run.log or its logs) ⇒ `VOID-1`, the run moved to
  `fp-run-void1`, replay once with the same seed and settings after 60 s. A second such void ⇒ FAILED. Any other
  harness failure is not replayed.
- **FP200.txt** (r--), line 1 `LA2-FP200: lookahead2(G9q-cd@2035.04M, δ* …, K 8, T 0.5) vs Foul Play gen9 at a FIXED
  200000 iterations per state: m ± h [lo, hi] (500 duplicate pairs, 1000 games); measurement, no decision rule`; then
  won-both / split / lost-both, the summed iteration-check counts, overrides, fallbacks, hours per block, the specs,
  the seed, the voids replayed. **Void** (no number): HARNESS_FAIL, > 2% fallbacks, < 450 pairs, or a block without a
  passing iteration check ⇒ line 1 `VOID: …`, FAILED (stage 1's VERDICT stands).
- Context only (not paired, different harnesses): raw cd@2030 60.8% [57.9, 63.7] at 200k (netv2 fp9 harness, 20
  instances); lookahead2(cd@805) 58.2% ± 2.9 vs time-based Foul Play (this harness, 3 instances, LA2-FP).

## Smoke (in the slot, 8 games, mechanical)
After δ\*, before any main game: (a) 2 pairs A vs OPP (tag `8233571smoke`): 4 games, no Traceback, no HARNESS_FAIL, one
arm; for A and OPP searched and leaves > 0, gate rows > 0, 0 failed worlds, fallbacks ≤ max(2, 2%). (b) 2 pairs A vs
Foul Play at 200k with 8 instances (tag `8233792smoke`): the same checks for A, plus the iteration check passed with
full-budget states seen. Not PASS ⇒ FAILED, nothing more is played. No games outside the slot.

## Slot job
- `cd $F && sbatch runs/la2-2030/code/scripts/fastenv/la2_2030.sbatch`: job `ws-fastenv-la2x2030`,
  `--partition=training`, 24 CPUs, 1 GPU, 48 GB, **`--time=24:00:00`**, no dependency (the slot is free since netv2's
  F2030 RELEASE 14:10Z).
- **Time basis.** LA2-1350's chain took 7.84 h on dove01: δ study 0.90 h, smoke 0.08 h, 1000 games 6.86 h. FP200: Foul
  Play at 200k takes ≈ 78 s of thinking per game (F2030: 1000 games over 20 instances in 65 min) and lookahead2 ≈ 1.3 s
  per decision × ≈ 37 decisions; with 8 concurrent battles ≈ 1000 / 8 × ≈ 90 s ≈ 3.1 h (+ 0.2 h for the FP smoke).
  Expected ≈ 0.9 + 0.3 + 6.9 + 3.1 ≈ 11.2 h. Slow case: the h2h at 1.3× LA2-1350's rate (8.9 h) + FP200 at LA2-FP's
  3-instance rate (7.6 h) + a 1–2 h cold-node start ≈ 19 h. Limit 24 h ≈ 2.1× the expectation. If the limit cuts FP200,
  stage 1's VERDICT stands and FAILED records it.
- **Stages**, each logged to `STATUS`: prereg → gate → checks (wsp launcher md5s, both nets' sha256, ≥ 24 CPUs, a GPU,
  the frozen trees, `lookahead2.py` / `netsearch.py` / `v3leaf.py` byte-identical to the benchmark bot's frozen code
  `$F/runs/la2-1350/code`, a fresh bench dir) → delta → smoke (a, b) → main → verdict → fp200 (if ADOPTED) → FP200.txt.
  A failed stage (or TERM) appends to `FAILED` and writes the RELEASE line; nothing later runs. A stage dir that already
  holds a chain is refused.
- **Code** frozen at submission into `$F/runs/la2-2030/code` (git archive of the ws/fastenv commit that adds this file;
  hash in `code/COMMIT` and NOTES.txt). Since LA2-1350's code (d44c739) the diff is: `la2_2030.sbatch` (new: this
  chain), `lookahead2_adopt.py` (gate `--name`, `smoke --fp`, `fp200`), `netsearch_slot.sh` and `foulplay_instances.sh`
  (the `iters=N` Foul Play option: wsp's launcher and check; nothing changes without it, so stage 1's harness is
  LA2-1350's), `tests/test_la2_2030.py`.
- **Outputs** in `$F/runs/la2-2030/`: `STATUS`, `WITHDRAWN` | `FAILED`, `DELTA`, `DELTA-VERDICT.txt`, `DELTA-RESULT`,
  `SMOKE.txt`, `smoke-run/`, `smoke-fp-run/`, `main-run/`, `VERDICT.txt` (r--), `fp-run/` (+ `fp-run-void1/`,
  `VOID-1`), `FP200.txt` (r--), `RESULT` (VERDICT line 1, then FP200 line 1 or "FP200 not run"). One RELEASE line in
  `$F/LEDGER-OUTBOX.md` at the end (done, WITHDRAWN or FAILED).
- **Laptop mock** (before submission): `tests/test_la2_2030.py` (the gate on conf2030's real format: CONFIRMED / NOT
  CONFIRMED / FAILED / missing / another net; the reading at lower bounds 52.9 / 50.1 / 48.0 / 46.9% through the chain;
  FP200 only after ADOPTED; the VOID replay once, two voids FAILED, a non-void harness failure not replayed; FP200's void
  rules; the smoke's Foul Play checks; the fixed-iteration launch line of `foulplay_instances.sh`), plus an `NS_DRY=1`
  dry run of the harness with the FP200 plan.

## Caveats
- **Two things change at once in stage 1:** the net (cd@1350 → cd@2030) and its δ\*. The adopted unit is (net, δ\*).
- **Power:** stage 1 is not powered for superiority at +2 points (≈ 28%); the non-inferiority branch is main's answer.
- **FP200 vs history:** FP200 is the first lookahead2 number at the fixed-iteration yardstick; compare it with raw
  cd@2030's 60.8% (another harness, 20 instances; not a paired test) and not with LA2-FP's time-based 58.2%.
- **Judge:** the δ rule's judge R shares A's value function (Bet 1's caveat).
