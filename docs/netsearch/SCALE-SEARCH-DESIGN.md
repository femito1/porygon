# A net-guided search that scales with thinking time: design (fastenv, 2026-11-16; prototype `porygon.deepen`)
Main's decision (11-16 ~23:10Z, with the user's go-ahead): the next depth route is a search, guided by the network,
that gets stronger with more thinking time; later it is the teacher for expert iteration. This note chooses the
algorithm, states its leaf budget per decision and why that budget can grow, and records the prototype and its
microbench. The scaling test is pre-registered separately (`SCALE-GATE-PREREG.md`). Tags: [m] measured, [e] estimated.

## 0. What we know (the constraints the choice has to respect)
- **The leaf value is the lever, not the root combination.** Over poke-engine's MCTS with `evaluate()` leaves, root
  PUCT on the net, regret matching, a Bayesian root game (`sample`, −13 points) and net priors in the search gave no
  gain (`benchmark-method`). lookahead2, the first search with **network** leaves, beat its own raw net 53.17%
  [50.59, 55.73] (Bet 1) [m]. The NNUE route (cheap CPU leaves) failed: a small student was no better than
  `evaluate()` at determinized human leaves (gate 1) [m]. So the leaves stay the full v2 net.
- **Network leaves are expensive.** 10.5M parameters, 158 MFLOP and 12.3 KB of observation per leaf (V3 gates) [m].
  The slot's T4 (I1 bench, eager, under contention): 7.4k leaves/s at B 384 (fp32 and fp16), 9.8k (fp32) / 13.2k
  (fp16) at B 1024; uncontended 11.2k / 15.3k at B 1024 [m]. Python CPU cost of a leaf (engine step + LeafEncoder):
  ≈ 0.8 ms (V3 gates: engine 369 µs + encode 440 µs) [m].
- **lookahead2 per searched decision** (Bet 1 bench, 1,403 positions): leaves median 1,711 (q90 4,683, max 19,290),
  wall median 1.09 s, worker CPU 1.03 s, GPU forward 0.26 s [m]. Its pick equals a fresh rerun's on 80.4% of positions
  (world-sampling noise) and its selected-gap inflation is small (+0.0020 vs a K 32 judge) [m].
- **The optimizer's curse needs a margin gate** (hybrid H, lookahead2): we only override the net's T 0.5 draw when the
  search's Q beats it by δ. For G9q-cd@805M δ* = 0.005 (override 19.9%) [m].
- **Showdown's move timer** (`server/room-battle.ts`, our server's build): a 150 s bank, +10 s per turn (capped at
  150 s; +5 s for a mid-turn request such as a forced switch), at most 150 s per turn; the bank shrinks after turn 100.
  A bot that averages ≤ ~8 s per decision never runs the bank down; single decisions can take far longer.
- **The gap to close** is ≈ 135-165 Elo (Jaxcalibur ≈ FP + 190-220; our raw cd@805 ≈ FP + 55; lookahead2 ≈ +22 over
  its raw net). One more ply of the same kind is unlikely to close it alone; the point of this route is a search whose
  strength keeps rising with compute, so that more time, a faster leaf, or a better net each buy strength.

## 1. The choice: best-first selective deepening of lookahead2's game ("deepen")
**Keep lookahead2's root exactly** (K 8 worlds, every joint action, every chance branch, the Bayesian matrix game,
the margin gate) and spend every extra leaf below it, one turn at a time, on the leaves whose value matters most to
the decision. Budget 0 is lookahead2 itself, so the scaling curve starts at a measured point.

Why not simultaneous-move MCTS with net priors and values (Jaxcalibur's pUCT, our parked V3):
1. **At our leaf budget MCTS cannot even cover one turn.** lookahead2's exact one-turn game already uses ≈ 1.7k leaves
   (≈ 8 worlds × 9 × 9 joint actions × 2.6 chance branches). An MCTS with 2-10k simulations over 8 world trees has
   250-1,250 per tree for ≈ 80 joint actions with sampled chance: noisier root statistics than the exact matrix it
   replaces, before it buys any depth. MCTS's advantage (deep, narrow lines) appears at budgets (≫ 10⁴ leaves per
   decision) that the T4 cannot reach (§3).
2. **Simultaneous selection at the nodes.** Decoupled UCB / PUCT does not converge to the Nash equilibrium of a
   simultaneous-move node; regret-matching selection does, but needs many visits per node. deepen keeps the exact
   matrix game at the root, where the decision is made, and backs up the inner nodes linearly (§2): a min / max over
   pruned options of noisy leaves biased the gate in the v1 microbench (§6).
3. **Batching.** MCTS evaluates leaves sequentially; batching them on the GPU needs virtual loss with many leaves in
   flight (V3 needed K 64 per tree), which distorts selection most at small budgets. deepen expands in waves of ≈ 1k
   leaves, which is the T4's efficient batch size (B ≥ 1024: 9.8-13.2k leaves/s vs 7.4k at B 384 under contention).
4. **Our negative results were about root combination over a weak evaluator.** deepen changes neither the root game
   nor the gate that worked; it replaces leaf values by deeper ones.

Why not simply a full 2-ply expansion: one more full turn multiplies the tree by ≈ 200 (≈ 80 joint actions × 2.6
branches), i.e. ≈ 350k leaves per decision: two orders of magnitude over budget. Selective expansion is necessary.

## 2. The algorithm (`src/porygon/deepen.py`)
Per decision with ≥ 2 legal actions:
1. **Root (wave 0) = lookahead2.** a_net is drawn from the net at T 0.5 (the raw player). K = 8 worlds from
   `search.convert` in lookahead2's order; every (our legal action, their engine option) pair is simulated one turn
   with `generate_instructions`, every chance branch kept; leaves scored by the net (`v3leaf.LeafEncoder`, turns = 1);
   a forced opponent switch takes their switch-in worst for us. The same forward also returns each leaf's **policy
   logits** and **opponent-action logits** (`aux=True`), the priors for a later expansion. The root observation's
   opponent-action head gives the root's opponent prior.
2. **Waves** until the extra leaves reach the budget B (or nothing is worth expanding):
   - **Solve.** Inner nodes bottom-up (deepest first): a node's value is the expected value of its kept matrix under
     the net's priors over the kept options, both sides (`nash = 0`, the gate's setting; `nash = 1` is the Nash value,
     all nodes of a depth solved in one batched regret-matching+ call; values in between mix the two). Then the root's
     Bayesian game (`lookahead2.solve_bayes`, 200 iterations between waves).
   - **Reach** of every leaf = the product along its path of: the world's weight 1/K; a row weight — at the root
     (x(a) + π(a) + [a = a_net]) / 3 (the lookahead's equilibrium, the net's prior, and the move the gate compares
     against), below the root (x + π) / 2; a column weight (y + q) / 2 (the opponent's equilibrium and the net's
     opponent-action head); the chance probability. A forced opponent switch passes its reach to the switch-in worst
     for us.
   - **Priority** = reach × 4v(1 − v), v the leaf's P(win) (decided positions are not refined). Leaves at
     `max_turns` = 4, finished battles and failed expansions are never expanded.
   - **Expand** the highest-priority leaves until the wave's estimated new leaves reach `wave` = 1024 (or the
     remaining budget). An expansion is one more turn from the leaf's engine state: our options = the net's policy
     at that state (from the leaf's own forward), most probable first until 0.9 of the mass, at most 3; theirs = the
     net's opponent-action head at that state (one small aux forward of the wave's picked leaves), until 0.9 of the
     mass, at most 4, at least 2; every kept pair simulated with all chance branches ≥ 3% (renormalised); new leaves
     encoded with turns + 1.
   - **Transpositions.** A new leaf with the same (world, depth, engine state string) as an existing one is that leaf
     (a DAG): no forward, the reach adds up.
   - **Score** the wave's new leaves in one batched forward.
3. **Final solve**: inner nodes (300 iterations if nash > 0), the root game with lookahead2's 1000; Q(a) = our
   value with a against the opponent's root equilibrium, x our strategy.
4. **Gate** (`lookahead2.gate`, unchanged): play a_L = argmax x iff a_L ≠ a_net and Q(a_L) − Q(a_net) > δ; else a_net.
   Any failure plays a_net and counts as a fallback.

**Determinism.** Given the position, the worlds and B, the decision is a deterministic function (no clock): the
budget is in leaves, not seconds, so arms are machine-independent and contention only slows them. Wall time per
budget is measured (§6) and maps budgets onto thinking time.

**Optimizer's curse.** The gate is lookahead2's; δ stays 0.005 at every budget in the gate test (one change at a
time). Selection is decision-focused: both a_net's row and the lookahead's own pick carry reach, so deepening refines
both sides of the gap the gate reads rather than only the favourite. Inner nodes back up linearly (no selection
inside the tree), so the only max is the root's, as in lookahead2. Measured (§6): with this backup the override rate
at δ 0.005 stays 14-16% at every budget (lookahead2 15%) and Q(a_net) and Q(pick) move together; with a Nash backup
it rose to 22-27%.

## 3. Leaf budget per decision, and why it can grow
The budget unit is **1× = 1,024 extra leaves** beyond lookahead2's tree. Measured per searched decision (§6; 120
study positions, 3 decisions in flight, 10 workers, one T4, fp32):

| arm | extra leaves B | total leaves median (q90) | wall s median (q90, max) | depth reached |
|---|---|---|---|---|
| lookahead2 (= deepen 0) | 0 | 1,344 (4,478) | 0.94 (1.57, 5.7) | 1 |
| 0.5× | 512 | 2,062 (5,208) | 2.13 (2.96, 5.5) | 2-4 (mostly 2-3) |
| 1× | 1,024 | 2,678 (5,659) | 2.44 (3.63, 4.8) | 2-4 (mostly 3) |
| 2× | 2,048 | 3,534 (6,672) | 3.22 (4.67, 6.2) | 3-4 |
| 4× | 4,096 | 5,564 (8,597) | 5.00 (6.17, 10.3) | 4 (118 / 120) |
| 8× | 8,192 | 9,594 (12,694) | 8.99 (11.1, 13.0); fp16 7.58 | 4 (cap) |
Measured leaves, seconds and decisions/s per arm: §6. The levers that make larger budgets affordable:
1. **Batching across worlds and nodes.** Every wave is one forward of ≈ 1k leaves from all 8 worlds and every node
   expanded in it (the T4's efficient batch); worker processes expand worlds in parallel.
2. **Selective expansion.** Width by prior mass (≤ 3 × ≤ 4 pairs per node instead of ≈ 80) and branch pruning below
   3% below the root: an expansion costs ≈ 11 new leaves (measured: 8,211 extra leaves over 704 expansions at 8×)
   instead of ≈ 200.
3. **Transpositions.** Identical states share a leaf: 27-28% of the new leaves below the root repeat a state and
   cost no forward (measured).
4. **The opponent head only where needed.** It is computed in one small forward of the leaves picked for expansion
   (≈ 100 per wave), not for every leaf (computing it for every leaf made the whole forward ≈ 60% slower in v1).
5. **fp16 leaves.** I1's parity passed for fp16 autocast (mean |ΔP| 9.8·10⁻⁵, policy top-1 flips 0.04%); it buys
   ≈ +35% leaves/s at B 1024 in I1; in deepen at 8× it cuts the wall time 8.99 → 7.58 s with the same pick on
   99.2% of positions. Off in the gate (lookahead2's anchor is fp32), available per spec (`half=1`).
6. **Not in the prototype, the next levers:** a higher `max_turns` (the cap of 4 binds from 4×), pinned host
   buffers and one forward per wave for all decisions in flight, CUDA graphs (I1's capture failed on one unpinned host→device copy in the
   model's forward: a model-side fix), a second GPU or the A40-class GPUs for deployment, TensorRT, reuse of the tree
   across turns, a cheaper leaf for the deepest nodes (an NNUE-2b student, if it ever passes gate 1).
So the budget scales with wall time at the rate the GPU and the CPU workers sustain, and every lever above raises it.
Where the time goes at 8× (median per decision, 3 in flight): leaf forwards 2.39 s (inside the GPU lock; ≈ 4k
leaves/s, below I1's 9.8k at B 1024 because waves are ≈ 1k leaves plus a small aux forward each), worker expansion
3.71 s wall (14.2 core-s, ≈ 1.5 ms per new leaf: engine step, encode, state string, pickling), solves 0.74 s.

## 4. The opponent model (general strength only)
- **At the root nothing models the opponent**: their every option is in the matrix and the Bayesian game takes their
  equilibrium (they know the world). This is lookahead2's, and what our gate tests measure.
- **Below the root** the opponent is the net's opponent-action head (trained on the opponent's actions in G9q's own
  training games, i.e. a model of a player like us): it picks which of their options are expanded and weighs them in
  the node's value (nash = 0); our own options are weighed by the policy. This is the value of "both sides play like
  the net for one more turn", a refinement of the net's own value with exact dynamics and chance. Nothing is fitted
  to Foul Play or any benchmark opponent (`no-opponent-overfitting`), and the gate test is against our own players.
- Pruning their replies to ≤ 4 keeps 65% of the head's mass on average (§6; tera doubles their move options). The
  root stays exhaustive and adversarial. `nash = 1` (adversarial inner nodes) is kept as an option for later tests.

## 5. Hardware and compute per decision
- **Gate and deployment hardware:** the slot (24 CPUs + one T4; dove, `--partition=training`) or fastenv's helper
  (12 CPUs + one T4). CPU worker processes (`python -m porygon.deepen --worker`, the lookahead2 pool protocol) expand
  worlds and nodes; the player's process holds the tree, solves it and runs the leaf net on the GPU (`leaf_device`).
  A head-to-head with two searching sides puts two pools and two leaf nets on one T4.
- **Compute per decision vs the timer:** measured in §6. With the timer's +10 s per turn, every arm whose median is
  ≤ ~5 s and q99 ≤ ~30 s is safe; in the gate games the timer is off (our harness vs our own bots), and the
  vs-Foul-Play follow-up keeps it on.

## 6. Prototype and microbench (jobs 1173957 v1, 1173960 v2, 1173963 0.5×)
Code (ws/fastenv): `src/porygon/deepen.py` (module docstring = §2), `tests/test_deepen.py` (pure parts: option
selection, priors, batched solver = `solve_bayes` per game, the tree's root game = `lookahead2.combine`, values,
reach, selection, transpositions, spec; laptop: 10 pass with `test_lookahead2.py`'s 8),
`scripts/fastenv/deepen_bench.py` (+ `.sbatch`; `run` / `report` / `delta`), `deepen:` specs in the slot harness.

**v1** (job 1173957, code 7164e48: Nash backup, the opponent head and option lists computed for every leaf;
cancelled after 4 passes, `$C/runs/dbench-20261116/NOTES.txt`): at 1,024 / 2,048 extra leaves the pick equalled
lookahead2's on 53% / 50%, the override rate at δ 0.005 rose from 15% to 26% / 23%, and Q(a_net) fell by 0.018 /
0.023 (mean) vs lookahead2. ⇒ the backup became the priors' expected value (nash = 0), and the priors became lazy.

**v2** (job 1173960, code 5aba4e0, dove04, 12 CPUs, one T4; `$C/runs/dbench2-20261117/`): 120 study positions (S2 +
S1 rows of `$C/netsearch/study-20261109`), G9q-cd@805.03M, K 8, 3 decisions in flight on 10 workers, fp32 unless
noted; one pass per arm, the same worlds per position in every pass (†: job 1173963, `$C/runs/dtest-20261117/bench512`, the same positions and seeds; its lookahead2 / deepen-0 passes reproduced v2's); a_net = one seeded T 0.5 draw per position, the
override column at δ 0.005. No failed decision, world, expansion or engine error in any pass.

| arm | leaves median (q90) | extra | expansions / waves | wall s median / q90 / q99 / max | decisions/s | forward / expand / solve s | worker CPU s | override | pick = lookahead2's | pick = 8×'s | ΔQ(a_net) / ΔQ(pick) vs lookahead2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lookahead2 | 1,344 (4,478) | – | – | 0.94 / 1.57 / 5.19 / 5.69 | 2.72 | 0.27 / 0.27 / 0.11 | 1.05 | 0.150 | 1 | 0.658 | 0 |
| deepen 0 | 1,344 (4,478) | 0 | 0 | 1.03 / 1.69 / 4.94 / 5.78 | 2.51 | 0.30 / 0.36 / 0.11 | 1.26 | 0.150 | **1 (parity)** | 0.658 | 0 |
| 0.5× (512) † | 2,062 (5,208) | 582 | 51 / 2 | 2.13 / 2.96 / 4.45 / 5.47 | 1.37 | 0.59 / 0.62 / 0.17 | 2.39 | 0.125 | 0.783 | 0.742 | +0.034 / +0.034 |
| 1× (1,024) | 2,678 (5,659) | 1,126 | 100 / 2 | 2.44 / 3.63 / 4.22 / 4.76 | 1.18 | 0.68 / 0.84 / 0.18 | 3.19 | 0.158 | 0.717 | 0.767 | +0.039 / +0.041 |
| 2× (2,048) | 3,534 (6,672) | 2,104 | 190 / 3 | 3.22 / 4.67 / 6.04 / 6.24 | 0.87 | 0.89 / 1.23 / 0.22 | 4.64 | 0.142 | 0.683 | 0.850 | +0.051 / +0.052 |
| 4× (4,096) | 5,564 (8,597) | 4,120 | 363 / 5 | 5.00 / 6.17 / 8.17 / 10.34 | 0.58 | 1.35 / 2.05 / 0.36 | 8.12 | 0.142 | 0.675 | 0.892 | +0.065 / +0.066 |
| 8× (8,192) | 9,594 (12,694) | 8,211 | 704 / 9 | 8.99 / 11.11 / 12.54 / 12.95 | 0.32 | 2.39 / 3.71 / 0.74 | 14.19 | 0.150 | 0.658 | 1 | +0.076 / +0.078 |
| 8× fp16 | 9,592 (12,682) | 8,209 | 710 / 9 | 7.58 / 9.57 / 10.58 / 10.98 | 0.39 | 1.96 / 3.21 / 0.66 | 11.97 | 0.158 | 0.667 | 0.992 | +0.076 / +0.078 |
| 2× Nash backup | 3,626 (6,679) | 2,120 | 192 / 3 | 3.98 / 5.38 / 7.77 / 7.97 | 0.71 | 0.99 / 1.52 / 0.44 | 4.98 | 0.217 | 0.467 | 0.508 | −0.020 / −0.015 |
| 8× Nash backup | 9,597 (12,674) | 8,212 | 709 / 9 | 12.27 / 15.62 / 19.85 / 22.82 | 0.24 | 2.35 / 5.39 / 1.96 | 14.76 | 0.267 | 0.508 | 0.508 | −0.030 / −0.025 |

Also measured: parity of deepen 0 with lookahead2 on all 120 positions (rows, Q, x, leaves and picks equal; max |ΔQ|
= 0); 27-28% of the new leaves below the root are transpositions; 2.26 leaves per expanded cell; the kept opponent
options cover 65% of the head's mass per expansion on average (the cap of 4 binds: with tera they often have > 8
options); depth: 1× mostly 3 turns, 4× and 8× all at the cap of 4.

Reading (no strength claim: the games decide):
- **Anytime behaviour.** The pick moves monotonically toward the deepest search's: agreement with 8× is 66% for
  lookahead2, then 74% / 77% / 85% / 89% at 0.5× / 1× / 2× / 4×. Deepening changes a third of lookahead2's picks by 4×.
- **The gate stays calibrated with the priors backup.** The override rate at δ 0.005 is 14-16% at every budget
  (lookahead2 15.0%), and Q(a_net) and Q(pick) shift together (relative difference ≤ 0.002). With the Nash backup the
  override rate rises to 22-27% and a_net's Q falls 0.005-0.006 more than the pick's: the selection bias the gate would
  read as a gap. ⇒ the gate uses nash = 0 and one δ (0.005) at every budget.
- **Values drift up with depth** (+0.04 at 1× to +0.08 at 8×, the same for every row): backed-up values are not
  calibrated win probabilities (both sides' options are pruned to the top of the priors). Harmless for the decision
  (it is a common shift), but an expert-iteration value target must not use them raw.
- **Cost:** wall ≈ 0.94 s + ≈ 1.0 ms per extra leaf per decision with 3 in flight (fp32); ≈ 3.3-3.9k leaves/s per
  decision stream, ≈ 10-11k leaves/s for the three. The GPU lock is busy ≈ 80% at 8×, so two searching sides on one
  T4 are GPU-bound; CPU ≈ 1.5 ms per new leaf.
- **Timer:** 4× (5.0 s median, 10.3 s max) fits Showdown's +10 s per turn; 8× (9.0 s median) is at the edge (7.6 s
  with fp16).

## 7. Reuse of lookahead2
Unchanged and called: `search.convert` (worlds), `v3leaf.LeafRoot` / `LeafEncoder` / `side_options` /
`needs_switch` / `terminal`, `lookahead2.solve_bayes`, `ref_value`, `gate`, `engine_choice`, `LookaheadPool` (the
worker protocol; `DeepenPool` only changes the worker module), the `LookaheadPlayer` (DeepenPlayer overrides
`choose_move`), the spec options (`deepen:` = `lookahead:` + the deepening's), the slot harness (`netsearch_slot.sh`,
`netsearch_h2h.py`, `netsearch_slot.py` accept `deepen:`). The root expansion (`op_root`) is lookahead2's
`expand_world` with each leaf's engine state and options recorded; the microbench checks b0 against lookahead2.

## 8. Expert iteration (later)
The search's root strategy x, its Q and its root value are the targets for training the net on the search's choices
(policy distillation toward x or the gated move, value toward the root game's value). Past ExIt attempts failed
because poke-engine's search was weaker than the net; the gate test's "4× beats 1×" and "1× ≥ lookahead2" are
the precondition for using deepen as a teacher (a teacher must beat the student at the budget it labels with).

## 9. Known limitations
- Below the root each world is played with full knowledge of that world (strategy fusion); only the root's strategy
  is shared across worlds. Deeper turns in the same world inherit its sampled sets.
- Leaf observations keep the root's knowledge fields (V3 P1-B: mean |ΔP| 0.021 vs a fresh encode); deeper leaves
  are further from what the net saw in training.
- K = 8 worlds at every budget: depth does not reduce world-sampling noise (lookahead2's pick = a rerun's on 80%).
  More worlds are the other use of time; an arm that adds worlds instead of depth is a separate question.
- Best-first minimax can tunnel into one line; the reach mixes in priors and the uncertainty term to spread it.
- No time limit inside a decision (budget in leaves): a pathological position takes longer; the q99 / max wall time
  is measured.
