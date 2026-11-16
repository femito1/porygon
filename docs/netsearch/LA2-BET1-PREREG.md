# Bet 1: one-turn lookahead with network-valued leaves vs the same raw network: pre-registration (fastenv, 2026-11-15 ~21:36 UTC; DRAFT until main signs off, then binding and read-only)
Main's decision (2026-11-15): fastenv's project is Bet 1. Steps 0-2 and the smoke were approved; the main test runs only
after main signs off this file's md5. Records and deviations after sign-off go in a separate file
(`runs/la2-bet1/NOTES.txt`), never here. The cluster copy `$C/runs/la2-bet1/PREREG.txt` is byte-identical to this file.

## Question
Does a one-turn lookahead, with its leaves scored by the v2 network, add strength over the same raw network?
- **A** = `porygon.lookahead2` on G9q-cd@605M (below), margin δ* = 0.01, K = 8 worlds.
- **B** = the raw G9q-cd@605M at T 0.5 (netv2's NetPlayer: how every raw-net number is measured).
- One network for both sides and for A's root and leaves: `$C/nets/g9q-cd-605.04M.pt`, sha256
  `b7957be4c74dd37fc41892706c250d47f6a110558621e7f813e0ad9a9124001b` (netv2 `runs/g9q-cd600/final/00605037107.pt`,
  copied 11-15, hashes equal). It loads through ONE frozen netv2 tree, `$C/frozen/netv2-fba02fb-20261110`. That tree's
  `porygon/v2/{encoder,spec,model,damage,tables,evaluate,player}.py` are byte-identical (cmp) to G9q's training source
  `src-24c633f`.
- Set library: the October server and the in-tree October encoder library (md5 324d8e97), as in every Stage 3 and
  re-freeze run. G9q's training games used the November true sets (±1 level on 4 gen9 species). This is the recorded
  train/serve deviation of re-freeze #1, and it is the same for A and B.

## A: lookahead2 (`src/porygon/lookahead2.py`, ws/fastenv e874baf, sha256 8b68b626325b9029b39268ae3ef04e41120011071fd9c3f15602da58a0962cfe)
At every decision with ≥ 2 legal actions:
1. **Root.** The net on the live battle (`netsearch.net_forward`, CPU) gives T-1 probabilities p and P(win). a_net is
   drawn from softmax(logits / 0.5), exactly as B draws its move.
2. **Worlds.** K = 8 determinizations from `search.convert`, our search's sampler of the opponent's hidden sets and
   unseen mons.
3. **Expansion.** In each world, every pair (our legal action, their engine option) is simulated for one turn with
   poke-engine's `generate_instructions`. Every chance branch is kept (min_prob 0).
   - A finished battle scores 1 / 0 / 0.5.
   - Otherwise the branch scores the net's P(win) of the leaf observation. The leaf is `v3leaf.LeafEncoder`: the
     root's real v2 observation patched with the engine state, turns = 1 (parity PASS in the V3 gates).
   - If only the opponent has to switch (a faint, their U-turn), the branch takes their switch-in that is worst for
     us: the min over switch-ins of the expected score of `none` vs that switch.
   - If we have to switch, the state is scored directly.
   - Engine panics on a pair score the row's mean. A world with > 10% failed cells is dropped.
4. **Game.** The K matrices M_k[a, j] of our expected P(win) form a zero-sum Bayesian game: the opponent knows the
   world, we play one mixed strategy x. It is solved by regret matching+ with linear averaging, 1000 iterations.
   Q(a) = mean_k M_k[a] · y_k is our win probability with a against the opponent's equilibrium play.
5. **Gate.** a_L = argmax x (first on ties). Play a_L iff a_L ≠ a_net and Q(a_L) − Q(a_net) > δ*; else play a_net.
   - If Q(a_net) is undefined (a_net is not an engine option in every world), play a_net.
   - Any failure plays a_net and counts as a fallback.
- **Value without potential.** G9q's value head is BCE(sigmoid(win_logit), (λ-return + 1)/2). The rewards are sparse
  ±1: G9q-cd600's trainer argv has no `--shaping`, and the default is 0 0 0. So P(win) is used as is, with no V + Φ.
- **No time budget.** The same position and worlds give the same decision on any machine; contention only slows A.
  - Worlds expand in 10 worker processes (`python -m porygon.lookahead2 --worker`).
  - All leaves of a decision are scored in one batched forward on the T4.
  - Measured on the 1,403 study positions (helper, 12 CPUs, 3 positions in flight): per searched decision, leaves median 1,711 (q10 40, q90 4,683, max 19,290); wall s median 1.09 (q90 2.19, max 7.0); worker CPU s median 1.03 (q90 2.77); the leaf forward median 0.26 s, the solve 0.09 s. No failed world or engine error in 3 × 1,403 lookaheads.
  - The hybrid's budget is 4 × 1000 ms engine search (4 core-s) on 11 workers, ≈ 1.2 s wall.

## δ* (rule fixed after a 14-position probe, before the 1,403-position bench)
- **Data.** On the study positions `$C/netsearch/study-20261109` (S1 501, S2 250, S2c 251, S3 401), the lookahead was
  run three times per position: A with K 8 worlds, a fresh retest B with K 8, and a fresh judge R with K 32.
- **Rule.** δ* is the smallest grid margin ≥ 0 from which on (every larger finite grid margin too) at least 90% of the
  overrides are confirmed by R. An override is weighted by the net's T 0.5 draw probability. It is confirmed when
  R's Q of A's pick exceeds R's Q of the net's draw.
  - Grid: 0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.075, 0.1,
    0.15.
  - The rule only removes overrides that are world-sampling noise. R shares A's value function, so it cannot say
    whether the lookahead or the policy is right; the games decide that.
- **Result** (`$C/lookahead2/study-20261115/bench-all.txt`): 
  - **δ\* = 0.01.**
    - Override rate (weighted by the T 0.5 draw) 15.3%.
    - R confirms 90.4% of the overrides at δ\*, and ≥ 90.4% at every larger grid margin up to 0.15.
    - At 0.0075 the precision is 0.899, so δ\* is the next grid margin.
  - Pooled curve, from `bench-all.txt`. Gains are R-judged, P(win) per searched decision:

    | margin | override | R: gain vs the raw draw | R: vs the pure lookahead | R: precision |
    |---|---|---|---|---|
    | -inf | 0.486 | +0.00539 | 0 | 0.717 |
    | 0.0025 | 0.215 | +0.00532 | −0.00007 | 0.891 |
    | 0.005 | 0.191 | +0.00525 | −0.00014 | 0.893 |
    | 0.01 | 0.153 | +0.00500 | −0.00039 | 0.904 |
    | 0.02 | 0.095 | +0.00443 | −0.00096 | 0.959 |
    | 0.05 | 0.029 | +0.00273 | −0.00266 | 0.975 |

  - Also measured:
    - the lookahead's pick equals the net's argmax on 52.9% of positions;
    - A's pick equals B's (a fresh K 8) on 80.4%;
    - the selected-gap inflation vs R is +0.0020 (sd 0.0107);
    - the gap of the T 0.5 draw is 0 on 59.6% (weighted), q90 0.019, q95 0.038.
  - Per subset (informational; the rule is the pooled one): S1 0.015, S2 0.0025, S2c 0.0075, S3 0.05, S2+S2c 0.0075.
  - `bench-all.jsonl` sha256 f785f1338fedf30aa76140be1ab2897423b4dd8ea5fff80872384570129d840c (the analysis ran
    on code c41979a; the bench ran on the pre-commit sync `a4a218b+lookahead2-wip`, whose `lookahead2.py` has the
    sha256 above).

## Harness
- **Stage-3 slot harness, on fastenv's helper job.** The slot is busy, and the next claim is netv2's.
  - `scripts/fastenv/netsearch_slot.sh` → netv2's `porygon.v2.evaluate` via `netsearch_h2h.py` (`lookahead:` spec).
  - Server: fastenv's `.showdown-eval` (smogon 7b02eb3 + duplicate patch, code sha256 95a0414e26cfd185, October sets).
  - 3 concurrent battles, blocks of 100 duplicate pairs. Block j runs on a fresh server seeded `la2b120261115b<j>`.
  - Helper: dove, `--partition=training`, 12 CPUs, one T4. B runs on the CPU; A's leaf net runs on the T4.
- **Specs.**
  - A = `lookahead:$C/nets/g9q-cd-605.04M.pt,margin=0.01,temperature=0.5,worlds=8,workers=10,leaf_device=cuda`
  - B = `$C/nets/g9q-cd-605.04M.pt,temperature=0.5`
- Code: the chain freezes `$C/src` + `$C/scripts` into `runs/la2-bet1/code` at its start, and every run copies its
  code from there. Manifest: code commit, net sha256, server / engine hashes, design md5.

## Test (internal go/no-go, sequential: user rule allows it for internal tests)
- **Design.** wsp's group-sequential design, `scripts/fastenv/vendor/seq_design.py` (byte copy of main's
  `scripts/wsp/seq_design.py`).
  - Kind onesided: θ0 50%, θ1 54%, α 0.025 one-sided, β 0.10 (power 0.90 at 54%).
  - Looks every 100 pairs, n_max 800 pairs (1,600 games).
  - Spending: HSD(−4) for α, HSD(−2) for β. Futility is non-binding; σ is a plug-in re-estimated at every look.
  - Design JSON: `scripts/fastenv/plans/seq-la2-bet1.json`, md5 f0911c195e850d1e11ea11359c0b0100.
- **Bounds** (win rate of A by pair means):

  | look | pairs | SUCCESS if ≥ | FUTILITY if ≤ |
  |---|---|---|---|
  | 1 | 100 | 60.92% | 45.67% |
  | 2 | 200 | 57.34% | 48.47% |
  | 3 | 300 | 55.66% | 49.79% |
  | 4 | 400 | 54.60% | 50.62% |
  | 5 | 500 | 53.83% | 51.21% |
  | 6 | 600 | 53.24% | 51.67% |
  | 7 | 700 | 52.74% | 52.06% |
  | 8 | 800 | 52.31% | 52.31% |

  These are the values at the planning σ 0.318; the monitor uses the observed σ.
  - OC, 4000 simulations at the planning pair mix: P(SUCCESS) is 0.024 at 50%, 0.37 at 52%, 0.91 at 54%.
  - E[pairs] is 376 at 50%, 546 at 52%, 490 at 54%.
- **Looks.** After every block, `vendor/seq_monitor.py` runs on arm 1's blocks' `battles.jsonl` in block order:
  `--order sequential --final --a-prefix pa --design runs/la2-bet1/seq-design.json`. It plays at most 9 blocks; the
  9th is only needed if pairs were lost before 800 complete pairs. The data behind look j are fixed (nested).
- **Decision of record.**
  - **GO** iff the SUCCESS bound is crossed: the repeated 95% CI excludes 50% from below at the stopping look. This is
    main's "lower 95% > 50%".
  - **NO-GO** iff the FUTILITY bound is crossed, or the last look (800 pairs) is reached without SUCCESS.
  - **VOID** iff HARNESS_FAIL, or > 2% fallbacks on either side, or no decision. The chain writes FAILED, and the run
    is replayed with the same seeds after the fix.
- **Reported with the verdict** (`scripts/fastenv/lookahead2_verdict.py`):
  - the median-unbiased estimate with its 95% CI (stage-wise ordering);
  - the naive mean ± 1.96 sd/√n of the pair means (informational, not the decision);
  - won-both / split / lost-both;
  - A's override rate and the median gap of its overrides;
  - fallbacks on both sides;
  - leaves, wall seconds, forward seconds and worker CPU seconds per searched decision (median, q90, max);
  - hours.

## Chain (one cluster-side job chain, no agent)
- **Script.** `scripts/fastenv/lookahead2_bet1.sh` (ws/fastenv e874baf, sha256 cd16fa2649ae0946e9727c276a870ed5ae093e9600527f2367799a5cd484efe1), run as a step of
  fastenv's helper job (1173378).
- **Stages.**
  1. Checks: this file's md5 = `runs/la2-bet1/PREREG.md5` (written on sign-off); net sha256; design md5; ≥ 12 CPUs;
     a GPU.
  2. Smoke: 2 pairs, tag `la2smoke20261115`, read mechanically.
     - PASS iff 4 games, A searched > 0 with leaves > 0, gate log rows, fallbacks ≤ max(2, 2%), no Traceback, the
       sequential look ran, no HARNESS_FAIL.
     - FAIL stops the chain.
  3. Main run `runs/la2-bet1/main-run`.
  4. `VERDICT.txt` (r--, line 1 = GO / NO-GO / VOID: …) and `RESULT`.
  5. The helper job is cancelled.
- Stage files: `runs/la2-bet1/{STATUS, SMOKE.txt, VERDICT.txt, RESULT, FAILED}`.
- **Expected time.** About 0.6 h per block of 200 games (from the smoke), so ≤ 5.5 h at
  n_max. E[pairs] ≈ 400-550 pairs, i.e. about 2.5-3.5 h.

## Pre-signoff smoke (mechanical, 3 pairs, margin 0.01, tag la2presmoke20261115; `$C/runs/la2-presmoke-20261115`)
**PASS (mechanical; the result is not used).** Job 1173378, 21:30-21:32Z, code c41979a with margin 0.01.
- 3 pairs / 6 games in 59 s of client time, 185 searched decisions.
- 0 fallbacks, 0 failed worlds, 0 engine errors, 0 Traceback lines.
- Overrides 14.6%, against the bench's 15.3%.
- The gate log was written and the sequential look ran (DECISION continue at 3 pairs). The verdict script printed its
  report (VOID, as expected without a decision).
- Per searched decision:
  - leaves median 1,283 (q90 4,436, max 8,657);
  - wall s median 0.68 (q90 1.47, max 5.28);
  - forward 0.19 s;
  - worker CPU 0.72 s (q90 2.43).
- Score 66.7% on 3 pairs (not used).
- Run dir `$C/runs/la2-presmoke-20261115`.

## Deviations and caveats (known before sign-off)
- **10 lookahead workers instead of the hybrid's 11** (a 12-CPU helper). The lookahead has no time budget, so the
  worker count changes speed, not decisions.
- **A against the same network.** A's leaf values come from the network that B plays with. No opponent model is used:
  Q is taken against the opponent's equilibrium play, not B's policy head. Still, a GO here is a self-improvement
  claim, and it has to be confirmed against other opponents (Foul Play, the benchmark hybrid) in fixed-n slot runs
  (step 4, separate pre-registrations).
- The δ* rule uses a judge that shares A's value function. It sets δ* by noise only.
- **Harness differences** (CPU count, the helper instead of the slot) affect only speed: both sides are
  deterministic functions of their inputs and draws, with no clocks.

## What follows
- **GO:** propose to main the fixed-n slot follow-ups:
  - lookahead2 vs Foul Play gen9, paired with the raw net on the same team pairs;
  - lookahead2 vs the benchmark hybrid H(0.0212) + G9p-cd;
  - a design note for net-valued leaves at depth: a small distilled value net on CPU inside poke-engine's MCTS.
- **NO-GO:** report the override diagnostics: the outcome of games by override gap bin, and where the overrides
  concentrate (forced switches, tera, KO turns). Nothing else is tested without a new decision.
