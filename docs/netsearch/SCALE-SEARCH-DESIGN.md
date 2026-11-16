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
   simultaneous-move node; regret-matching selection does, but needs many visits per node. A matrix game solved
   exactly over the expanded cells has neither problem.
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
   - **Solve.** Inner nodes bottom-up (deepest first, all nodes of a depth solved in one batched regret-matching+
     call, 200 iterations), then the root's Bayesian game (`lookahead2.solve_bayes`, 200 iterations between waves).
   - **Reach** of every leaf = the product along its path of: the world's weight 1/K; a row weight — at the root
     (x(a) + π(a) + [a = a_net]) / 3 (the lookahead's equilibrium, the net's prior, and the move the gate compares
     against), below the root (x + π) / 2; a column weight (y + q) / 2 (the opponent's equilibrium and the net's
     opponent-action head); the chance probability. A forced opponent switch passes its reach to the switch-in worst
     for us.
   - **Priority** = reach × 4v(1 − v), v the leaf's P(win) (decided positions are not refined). Leaves at
     `max_turns` = 4, finished battles and failed expansions are never expanded.
   - **Expand** the highest-priority leaves until the wave's estimated new leaves reach `wave` = 1024 (or the
     remaining budget). An expansion is one more turn from the leaf's engine state: our options = the net's policy
     at that state, most probable first until 0.9 of the mass, at most 3; theirs = the opponent-action head, until
     0.9 of the mass, at most 4, at least 2; every kept pair simulated with all chance branches ≥ 3% (renormalised);
     new leaves encoded with turns + 1.
   - **Transpositions.** A new leaf with the same (world, depth, engine state string) as an existing one is that leaf
     (a DAG): no forward, the reach adds up.
   - **Score** the wave's new leaves in one batched forward.
3. **Final solve**: inner nodes with 300 iterations, the root game with lookahead2's 1000; Q(a) = our win
   probability with a against the opponent's equilibrium, x our strategy.
4. **Gate** (`lookahead2.gate`, unchanged): play a_L = argmax x iff a_L ≠ a_net and Q(a_L) − Q(a_net) > δ; else a_net.
   Any failure plays a_net and counts as a fallback.

**Determinism.** Given the position, the worlds and B, the decision is a deterministic function (no clock): the
budget is in leaves, not seconds, so arms are machine-independent and contention only slows them. Wall time per
budget is measured (§6) and maps budgets onto thinking time.

**Optimizer's curse.** The gate is lookahead2's; δ stays 0.005 at every budget in the gate test (one change at a
time). Selection is decision-focused: both a_net's row and the lookahead's own pick carry reach, so deepening refines
both sides of the gap the gate reads rather than only the favourite. Deeper values are min-over-their-replies of
noisy leaves (pessimistic) inside max-over-ours (optimistic); the gate absorbs the residual bias, and the gate test's
override rate per budget shows whether it grows.

## 3. Leaf budget per decision, and why it can grow
| arm | extra leaves B | total leaves per decision [e: base median 1.7k + B] |
|---|---|---|
| lookahead2 (= deepen 0) | 0 | ≈ 1.7k |
| 0.5× | 1,024 | ≈ 2.7k |
| 1× | 2,048 | ≈ 3.7k |
| 2× | 4,096 | ≈ 5.8k |
| 4× | 8,192 | ≈ 9.9k |
Measured leaves, seconds and decisions/s per arm: §6. The levers that make larger budgets affordable:
1. **Batching across worlds and nodes.** Every wave is one forward of ≈ 1k leaves from all 8 worlds and every node
   expanded in it (the T4's efficient batch); worker processes expand worlds in parallel.
2. **Selective expansion.** Width by prior mass (≤ 3 × ≤ 4 pairs per node instead of ≈ 80) and branch pruning below
   3% below the root: an expansion costs ≈ 10-30 leaves instead of ≈ 200.
3. **Transpositions.** Identical states share a leaf (measured share in §6).
4. **fp16 leaves.** I1's parity passed for fp16 autocast (mean |ΔP| 9.8·10⁻⁵, policy top-1 flips 0.04%); it buys
   ≈ +35% leaves/s at B 1024. Off in the gate (lookahead2's anchor is fp32), available per spec (`half=1`).
5. **Not in the prototype, the next levers:** CUDA graphs (I1's capture failed on one unpinned host→device copy in the
   model's forward: a model-side fix), a second GPU or the A40-class GPUs for deployment, TensorRT, reuse of the tree
   across turns, a cheaper leaf for the deepest nodes (an NNUE-2b student, if it ever passes gate 1).
So the budget scales with wall time at the rate the GPU and the CPU workers sustain, and every lever above raises it.

## 4. The opponent model (general strength only)
- **At the root nothing models the opponent**: their every option is in the matrix and the Bayesian game takes their
  equilibrium (they know the world). This is lookahead2's, and what our gate tests measure.
- **Below the root** the net's opponent-action head (trained on the opponent's actions in G9q's training games, i.e.
  self-play) decides only *which* of their options are expanded and half of the reach weight; the value of a node is
  still the min over the kept replies (Nash of the kept matrix). Nothing is fitted to Foul Play or any benchmark
  opponent (`no-opponent-overfitting`), and the gate test is against our own players.
- Pruning their replies is optimistic for us (a pruned best reply is missed). Mitigations: coverage of 0.9 of the
  head's mass, at least 2 replies, and the root is exhaustive. The share of opponent mass kept is logged.

## 5. Hardware and compute per decision
- **Gate and deployment hardware:** the slot (24 CPUs + one T4; dove, `--partition=training`) or fastenv's helper
  (12 CPUs + one T4). CPU worker processes (`python -m porygon.deepen --worker`, the lookahead2 pool protocol) expand
  worlds and nodes; the player's process holds the tree, solves it and runs the leaf net on the GPU (`leaf_device`).
  A head-to-head with two searching sides puts two pools and two leaf nets on one T4.
- **Compute per decision vs the timer:** measured in §6. With the timer's +10 s per turn, every arm whose median is
  ≤ ~5 s and q99 ≤ ~30 s is safe; in the gate games the timer is off (our harness vs our own bots), and the
  vs-Foul-Play follow-up keeps it on.

## 6. Prototype and microbench (job 1173957)
RESULTS_PLACEHOLDER

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
