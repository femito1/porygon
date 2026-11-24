# Council brief — Porygon (shared, identical for every member)

Facts only, as of 2026-11-24. Every number cites a row id in `docs/thesis/data/results.csv` / `docs/thesis/results.md`
(same ids). Read the sources yourself; this brief is a map, not an argument.

## The goal
Be the strongest Pokémon Showdown random-battle bot in **gen9** (primary) and **gen4** (secondary, validation bed).
- **Bar:** Jaxcalibur, closed source; self-play transformer + pUCT; gen9 ladder peak 2557 Elo vs Foul Play's 2341.
  - That implies ≈ 75–78% vs Foul Play [estimate from Elo].
  - Its author says the net alone ≈ 2400 Elo and search adds 100–150.
- **Second bar:** MissingNo.Zero, ≈ 3M-param PPO, no search, raw net vs Foul Play 66.7% gen9 / 61.6% gen4. Foul Play
  ran on an M4 Pro core, an unknown budget.
- **Yardstick:** Foul Play, the strongest open-source bot, gen9 at a FIXED 200k MCTS iterations per state. The score
  depends on Foul Play's budget: −3.2 points per 4× (fpc-04).

## Where we stand
| bot | vs Foul Play | ids |
|---|---|---|
| gen9 benchmark bot: lookahead2 (one-turn net-valued lookahead, margin gate δ 0.0025) on G9q-cd@2030M | 61.60 [58.80, 64.40] (fixed 200k) | ns-32, ns-31 |
| gen9 raw net G9q-cd@2030M (T 0.5, no search) | 60.8 [57.9, 63.7] (fixed 200k) | g9q-32 |
| gen4: poke-engine MCTS + fitted leaf eval + v1 PPO prior | 59.50 [56.69, 62.31] (fixed 200k; 59.2 ± 2.2 time-based) | fp4-12, fp4-04 |
| gen4 raw v2 net R1b-F-cd@640M, same Foul Play pairs | 59.85 [56.90, 62.80] (fixed 200k); paired Δ +0.35 [−3.57, +4.27] | v2g4-16, v2g4-17 |

**Trajectory of the gen9 raw net (G9q, cooled snapshots):**
- In-process h2h, each ≈ 550–700M steps later beats the previous: cd@805 vs cd@605 52.46 (g9q-21); cd@1350 vs cd@805
  52.05 (g9q-29); cd@2030 vs cd@1350 51.99 (g9q-31).
- vs Foul Play: 58.8 / 57.8 / 59.1 (time-based) / 60.8 (fixed 200k) for cd@605 / 805 / 1350 / 2030 (g9q-17, -24, -30, -32).
- **Known weakness:** games < 25 turns. cd@805 48.4% vs 62.9% in longer games (dg-01); cd@2030 53.4% vs 64.6% (dg-08).
  The same nets show no length effect in self-play. Loss patterns: setting up at < 50% HP while slower ≈ 1.6 pts;
  attacking into a resist when a neutral ≥ 70 BP move exists ≈ 0.9 pt (`evidence/loss-patterns-report.md`).

## What the system is (details: `docs/thesis/methods.md`, `docs/thesis/codebase.md`)
- **Network v2:** pre-RMSNorm transformer, d_model 256, 4 layers, 8 heads, per-token-type FFNs, ≈ 5.8M params.
  - Entity tokens (field, Pokémon, moves) with damage features.
  - Heads: pointer policy, win-probability value, aux heads (opponent action, hidden-set heads).
  - fp32 + TF32.
- **Training:**
  - Init: behaviour cloning on gen9 human replays (rated ≥ 1500; corpus via inputlog re-simulation).
  - PPO self-play, in-process Showdown simulator ("fastenv").
  - Opponents ≈ 49% self, 36% frozen (BC net + own best models), 10% snapshots, 5% baseline.
  - Constant lr 7e-5; batch 4096, 3 epochs, clip 0.1, ent 0.01, vf 0.5, max_grad_norm 0.5.
  - Deployment nets = short cosine "cooldown" branches (5M steps, lr ×0.1).
  - ≈ 2.0k steps/s on one A30 (30 CPUs), with the update ≈ 50% of wall time.
- **Search:**
  - poke-engine (Rust, open source) MCTS over determinized worlds.
  - lookahead2 = K 8 sampled worlds × every joint action one turn ahead (all chance branches), v2-net leaves, regret-matching
    matrix game; it overrides the net's sampled move only if Q beats it by δ.
  - ≈ 0.5–0.9 s per decision on 11 workers.

## What has been tried and did not help (do not re-propose without a new reason; read the rows)
- **Deeper / bigger search:**
  - v1-net leaves and root games: s4-*, fpr-*.
  - Root PUCT, more worlds, regret matching: fpr-302877, -303472, -307076.
  - V3, a GPU net inside MCTS (ns-12: leaf inference caps ≈ 2k leaves/decision).
  - NNUE CPU student as the leaf eval (ns-16/17).
  - `deepen`, best-first deepening on lookahead2: 4× budget = 51.72 vs 1×, FLAT (ns-25, ns-21).
- **Search → net (expert iteration):**
  - search labels from a teacher weaker than the net (bc-12…14, v1-12);
  - ExIt-LA2, fine-tuning toward lookahead2's played policy: HARMFUL, 48.17 vs control (ns-27, ns-28).
  - ExIt with a soft completed-Q target on the same data (council P2, 11-24): no harm, no gain: 50.20 vs control (ns-33),
    50.09 vs parent (ns-34) ⇒ the ExIt-LA2 line is closed in both forms.
- **Recipe arms, one change each, cooled vs cooled:**
  - lr annealing G9r (g9q-04/05); GAE λ 0.5 G9t (g9q-09/10); entropy schedule G9qent (g9q-22);
  - bf16 (g9q-20); EMA weights (v2g9-18); actor-stream overlap (v2g9-21).
- **Capacity / data:**
  - 2× wider BC net = BC-P4 (bn-01; data-limited at 12M decisions; never tried under RL);
  - higher-rated human data (bc-15; G9-P4's head start washed out, g9q-28).
- **Diagnostics that closed hypotheses:**
  - value-gradient swamping: the value gradient into the trunk is ≈ 3% of the policy's (dg-06);
  - exploitability: an RL exploiter trained 100M steps only against frozen cd@1350 gets 50.47 [49.05, 51.90] (ns-30)
    ⇒ no blind spot a dedicated agent finds.
- **Probe side findings (untested):** global-norm clipping is active in 100% of minibatches (coefficient ≈ 0.4);
  policy-gradient minibatch coherence ≈ 0.005 (noise scale ≈ 2–7·10⁵ transitions vs batch 4096) (dg-06).
- **Seed variance** at matched steps < ~2 points (g9q-18/19).
- **Lesson:** net-vs-net gains transfer only partly to Foul Play. Lookahead2's gain over its raw net vs Foul Play is
  +1.3 [−2.7, +5.3] (ns-20), and absent in short games (la2-len1).

## Resources (measured unless marked)
- **Imperial Slurm, per-user caps:** multigpu 48 CPUs / 3 GPUs; tempgpu 64 CPUs / 4 GPUs.
  - A30 (30 CPUs): G9q, the main gen9 run, ≈ 2.2B steps on 11-24.
  - A40 (10 CPUs): G9-P4 keep-alive (≈ 1.55k sps); **free for a new experiment**.
  - T4 (10 CPUs): R1b-F, gen4 v2, ≈ 400 sps, ≈ 630M steps.
  - Data host: A16, 8 CPUs; replay crawler; in-process net-vs-net h2h ≈ 5k games/CPU-h, a 3,400-game test in ≈ 15 min.
  - One benchmark slot (24 CPUs + T4) for anything with search or Foul Play: 1000 raw-net games ≈ 2 h; 1000 lookahead2
    games ≈ 6 h.
  - The shared disk was 99% full: keep artifacts small.
- **GPU Dojo VM:** 1 T4, 4 vCPU, ≈ 88 free GPU-h, counted while running; good for in-process h2h (≈ 25k games/h).
- **Laptop:** sleeps, ≈ 5 GB free disk; no heavy compute.
- **Humans:** one person, who delegates; their Showdown account (bot-flagged ladder accounts) needs them present.
- **Time:** the thesis runs to Sept 2027; the campaign continues.

## Hard rules (violating any one disqualifies a proposal)
- General strength only: never model or exploit Foul Play's decision rule; gains must hold against other opponents too.
- No training on Jaxcalibur's games (challenge games for evaluation only). Human replay data only from rated ≥ 1500.
- Foul Play is GPL-3.0: never copy its code. poke-engine is open source and usable.
- Pre-register the question and the decision rule before data, change one thing at a time, report 95% intervals from
  duplicate pairs. Sequential tests are for internal go/no-go only; headline numbers are fixed-n.
- Never release a GPU allocation; cluster-side pipelines; no heavy compute on the laptop or login node.

## Source map (all read-only, relative to the repo root)
`docs/thesis/results.md` (every result, by theme) · `docs/thesis/data/results.csv` · `docs/thesis/methods.md` ·
`docs/thesis/codebase.md` · `docs/thesis/timeline.md` · `docs/thesis/runs.md` · `docs/thesis/datasets.md` ·
`docs/thesis/compute.md` · `docs/netsearch/*.md` (search designs, preregs) · `docs/netv2/*.md` ·
`docs/council/evidence/literature-report.md` (MissingNo.Zero, Jaxcalibur, Metagross, PokeJAX…) ·
`docs/council/evidence/loss-patterns-report.md` · `docs/council/evidence/campaign-plan-and-rules.md` ·
`src/porygon/` (code: `v2/` network + trainer, `lookahead2.py`, `search.py`, fastenv bindings) · `README.md`.
