# Methods

Compiled 2026-11-17 from the code on `main` (fd6fd06) and the cluster checkpoints. Provenance: `[M]` read from the
named code / file, `[D]` derived, `[E]` estimate. Code paths are relative to the repo root.

## 1. Environment and action space
- Format: Pokémon Showdown `gen4randombattle` and `gen9randombattle` (singles, random teams from Showdown's generator).
  The generation is process-wide: `PORYGON_GEN=9` selects inference, conversion, `N_ACTIONS` (26) and the engine build
  [M] README Setup, memory `gen9-port.md`.
- Action space = poke-env 0.16.1's `SinglesEnv`: gen4 10 actions (switch 0–5, move 6–9); gen9 26 (22–25 = move +
  terastallize; 10–21 always masked) [M] CONTRACTS.md "Observation spec v2".
- Two simulators: (a) **Showdown** (real server, websocket, poke-env players; `scripts/start_servers.sh`) and (b)
  **fastenv**, an in-process Showdown (`src/porygon/fastenv*.py`, `fastenv/js/`: Node workers running Showdown's
  `BattleStream`, a JS port of poke-env 0.16.1's first-person tracker, JS ports of the v1/v2 encoders and baseline
  bots; exact Showdown mechanics; ≈ 590 transitions/s per CPU in training, env alone 18.1k gen9 / 20.8k gen4
  decisions/s on 24 EPYC CPUs ≈ 8–9× the Showdown collector) [M] fastenv HANDOFF. Parity checks: tracker/encoder
  parity vs poke-env (`scripts/fastenv/parity_v1.py`, `parity_v2.py`), heuristic-bot port 0 mismatches on 11.8k
  decisions [M] git a55591f.
- (c) **poke-engine 0.0.48** (pmariglia, Rust, MIT) as the search simulator, patched (§5.1).

## 2. Observation spec v2.1 (owner netv2; `src/porygon/v2/spec.py`, `encoder.py`, `parity.py`)
Generation-parametric dict of arrays per decision (field table in `.orchestra/CONTRACTS.md`):
- `field_num` (31): weather one-hot (8) + elapsed, terrain one-hot (4) + elapsed, trick room + elapsed, gravity, turn/100,
  unseen/alive counts, force switch, trapped flags, tera flags, wait, no active, available switches/moves.
- `side_num` (2 × 18): screens, safeguard, mist, tailwind, lucky chant and their elapsed turns; hazards.
- Per mon (12 slots: ours 0–5 in `battle.team` order = switch actions, opponent 6–11 in order of first appearance):
  `species`, `types` (2), `tera`, `tera_p` (20), `item`/`item_p` (8), `ability`/`ability_p` (4), `moves`/`move_p` (24),
  `move_type` (24), `move_num` (24 × 3), raw `stats` (6), `hp` (2), `boosts` (7), `mon_num` (74 flags incl. 49 volatile
  effects), plus `action_mask` (10 or 26).
- **Opponent set posterior** from revealed information only: the Showdown-generator set library
  (`gen{N}randombattle_library.json.gz`, October file md5 324d8e97) filtered by every revealed move / tera / ability /
  item, weighted by generator count; marginals for moves, item, ability, tera and expected stats.
- **Aux targets** (training only): opponent's true item / ability / tera / moves per slot, move-in-set, unseen species,
  next opponent action class (55 classes) and its soft target `aux_opp_target` (v2.1 addition).
- Parity: fixtures `/vol/bitbucket/fr1226/porygon-ws-netv2/fixtures/gen{4,9}` (30 battles each, 944 / 663 decisions); pass =
  ints exact, floats ± 1e-4 [M] CONTRACTS.md.
- v1 observation (for contrast): `src/porygon/encoder.py`, field + 12 mon tokens, flat; movepool prior for unrevealed sets.

## 3. Networks
### 3.1 v1 (`src/porygon/model.py`, SB3 2.9.0)
`BattleTransformer` features extractor: field and mon tokens projected to d 256, learned positional embeddings, a
`nn.TransformerEncoder` with 3 layers, 8 heads, FFN 2·d, GELU; species / move embeddings 64-d; `MaskedPolicy` with a
per-action pointer head and a value head [M] model.py defaults, train.py `--d-model 256 --n-layers 3`. Parameter count
not recorded (unknown).

### 3.2 v2 (`src/porygon/v2/model.py`, custom trainer)
Entity transformer [M] model.py docstring + checkpoint cfg of `runs/g9q-cd800/final/00805029873.pt`:
- Tokens: field (1) | our mons 0–5 | opponent mons 6–11 | our active's moves 0–3 (= move actions) | (gen9) the same moves
  with tera (= actions 22–25).
- Every id embedding = learned id vector (std 0.02) + an MLP of the id's static features (`porygon.v2.tables`);
  dims: species 128, move 96, item 64, ability 64, type 32. Move vectors for all 12 × 24 move slots mix the move entity,
  its type, dynamic numbers and **damage summaries** vs the other side (`porygon.v2.damage`: table-driven, MAE vs
  poke-engine 0.33% gen4 / 0.90% gen9 of max HP [M] git cc6d8dd); mon tokens pool their move vectors (d_move 128).
- Trunk: pre-RMSNorm transformer, **d_model 256, 4 layers, 8 heads, FFN multiplier 1**, QK-norm, **per-token-type FFNs**
  (field / own mons / opponent mons / actions), shared attention with a per-head additive bias from mon-pair damage /
  speed features (pair_hidden 32), fainted keys masked. (The code default is 6 layers × FFN 2, "10.4M" in git ceb0875;
  every production v2 lineage — R1b, G9*, P1–P4 BC — runs n_layers 4, ffn_mult 1 [M] checkpoint cfg; netv2.env passes
  `--n-layers 4 --ffn-mult 1` for R1b.)
- Heads: pointer policy over the action tokens; win-probability value (MLP 256; its input gradient into the trunk scaled by
  `value_trunk_grad`, 1.0 in production); aux heads for the opponent's item, ability, tera, moves, move-in-set, unseen
  species and next action (55 classes).
- ID dropout 0.1 (id-embedding path only, seeded per transition so acting and update forwards agree).
- **Parameter count:** trainable (sum of `net.parameters()`, as the trainer reports): **5,809,977 (gen9) / 5,702,969
  (gen4)** [M] `scripts/wsp/v2diag_conf/{g9q,r1b}.conf` `WANT=` (the tracker's checkpoint gate); state-dict tensor
  elements incl. non-trainable buffers (static tables): 6,020,439 (G9q-cd@805) / 5,913,431 (R1b-F@300M) [M] torch.load
  on the cluster 2026-11-17.
- Precision: fp32 with TF32 matmuls (`--no-bf16`); bf16 autocast was only 1.018× faster on an A40 (result g9q-20).

## 4. Training
### 4.1 v1 PPO (`src/porygon/train.py`; `--async-envs --overlap-update`)
Self-play PPO after Jett Wang, *Winning at Pokémon Random Battles Using Reinforcement Learning*, MIT MEng 2024
(hdl.handle.net/1721.1/153888) [M] memory `wang-thesis-recipe.md`:
| hyperparameter | Porygon v1 (train.py default) | Wang | note |
|---|---|---|---|
| γ / GAE λ | 0.9999 / 0.754 | same | |
| clip / value clip | 0.0829 / 0.0184 | same | |
| entropy coef | **0.01** | 0.0588 | 0.0588 kept the policy near-uniform with ≤ 9 legal actions |
| value coef / max grad norm | 0.4375 / 0.543 | same | |
| epochs / batch | 7 / 1024 | same | |
| lr | 10^-4.23 / (8x + 1)^1.5, x = steps / total (`thesis_lr`) | same | "annealing = biggest factor" (Wang) |
| reward | sparse ±1 + potential-based shaping (0.05 fainted, 0.05 HP) | sparse ±1 | |
| opponents | latest policy both sides + pool of past snapshots (every 2.5M) and baseline bots (`--pool-battles`) | self-play | |
Total 600M steps (+40M arms). Throughput ~2.1–2.2k steps/s sustained on the cluster [M] memory.

### 4.2 v2 PPO (`src/porygon/v2/train.py`; cluster wrapper `scripts/slurm_train_v2.sbatch` + `netv2.env`)
G9q's recipe, read from the args stored in `runs/g9q-cd800/final/00805029873.pt` [M] (cooldown-specific values marked):
| setting | value | source of the choice |
|---|---|---|
| collector | fastenv in-process, strictly on-policy (`--no-overlap`), 28 workers × 48 battles on the A30 | fastenv stream; actor-stream overlap reverted (v2g9-21) |
| rollout / epochs / batch / micro-batch | 40,960 transitions / 3 / 4096 / 2048 | netv2 (fewer epochs, larger batch than v1) |
| γ / GAE λ / clip | 0.9999 / 0.754 / 0.1 | Wang (λ 0.5 tested: worse, g9q-10) |
| lr | 7·10⁻⁵ constant after a 1M warm-up (WSD); main run never cools | netv2; lr 2·10⁻⁴ degraded after 7.5M (README); power-law anneal no gain (G9r) |
| entropy / zero-avoid / value / aux coefs | 0.01 / 0.003 / 0.5 / 0.1 | netv2 (entropy schedule: no gain, g9q-22) |
| grad norm / weight decay / EMA | 0.5 / 0.01 / 0.999 (EMA saved, not deployed: v2g9-18) | netv2 |
| reward | sparse ±1, no shaping; value head BCE on λ-returns ⇒ sigmoid(win_logit) = P(win) | MissingNo.Zero-style |
| init / anchor | BC net P3 (`runs/pretrain/P3-g9-s2.pt`) as init, KL anchor coef 0.1 decaying to 0 at 50M, critic warm-up 1M, frozen opponent | netv2 (G9q PREREG a64af3b2) |
| opponents | self-play 16 battles + pool 16; pool snapshot every 2.5M; p_frozen 0.67, p_baseline 0.08 of pool battles; `--frozen-best`; `--stacked-opp`; snapshot cache 192 (+ CPU tier) | netv2 |
| guards | eval anchor drop 0.15, entropy rel 1.5, entropy min 0.5 (0.35 in cooldowns), P(top ≥ 0.9) max 0.6, pool alarm | netv2 (`porygon.v2.train` guards) |
| steps cap | `TOTAL` in netv2.env (3·10⁹ for G9q since v20, 2026-11-17) | |
Gen4 R1b-F: the same trainer, gen4, `--n-layers 4 --ffn-mult 1 --lr 7e-5 --guard-anchor-drop 0.15 --guard-entropy-max 1.0`
plus its own follow-ups (netv2.env `nv2_cfg_r1b`) [M] `scripts/netv2_p4/netv2.env.v19`.

### 4.3 lr-cooldown branches (deployment nets)
From a main-run checkpoint, a 5M-step branch with cosine lr decay to ×0.1 (`--cooldown-start S --total-steps S+5M
--final-lr-frac 0.1`, e.g. cd@805: cooldown_start 800,000,710, total 805,000,710) and entropy floor 0.35, run INLINE in
the lineage's job (`scripts/netv2_g9q/cd800-driver.sh`); the main run keeps its lr. G9p-cd@104.71M beat G9p@105.01M
52.05% [50.42, 53.65] (v2g9-13), which set the practice; cooled-vs-cooled comparisons decide recipe arms because
raw-vs-raw favours a lower-lr arm (g9q-07) [M].

### 4.4 Behaviour cloning (`src/porygon/v2/pretrain.py`)
Loss = policy CE (human one-hot) over legal actions + 0.5 · value BCE (win prob vs (z + 1)/2) + 0.1 · the RL trainer's
aux losses. Defaults: AdamW lr 3·10⁻⁴, 1000 warm-up steps, weight decay 0.01, batch 512, grad norm 1.0 [M] argparse.
Production wrapper (`logs/nv2agent-p2-20261109.sh NAME SHARD_DIR --stream-gb 1.0 --eval-every 5000`) adds `--epochs 6 --seed 2
--value-sample one-per-game --val-frac 0.06 --stream-parts shards` [M] data HANDOFF: the value loss sees one position
per game per epoch (AlphaGo's fix for outcome memorisation); a streaming loader permutes rows per epoch. BC-P4 took
≈ 3 h [M] ORCHESTRATOR.md. A v1 BC (`porygon.bc`) exists for the v1 architecture (bc-01 compares).

## 5. Search
### 5.1 Engine patches (`scripts/build_engine.sh`, `patches/poke-engine-0.0.48-*.patch`)
- **gen4 durations:** stock 0.0.48 gen4 compiles out taunt/encore duration logic and never ends confusion; the patch
  branches before each move on every timed volatile with P(D = n | D ≥ n) for gen4's uniform durations (taunt 3–5,
  encore 4–8, confusion ends on attempt 2–5). Neutral for strength (49.1% vs stock, s4-04).
- **search options:** root PUCT on priors, joint root statistics (pair matrix), regret matching (`rm_gamma`).
- **tunable evaluation:** the leaf evaluation's constants from a JSON (`--eval-config`).
- `evaluate()` (src/genx/evaluate.rs): per-mon alive/HP/status/item terms, hazards, volatiles, boosts, side conditions;
  MCTS scores a leaf as sigmoid(0.0125 · (eval − root eval)) [M] memory `engine-patch.md`.

### 5.2 Determinized MCTS (`src/porygon/search.py`, `engine_worker.py`, `inference.py`)
Per decision: K = 4 determinizations (opponent sets sampled from the set posterior: library ∩ revealed moves / item /
ability, item evidence, speed order, **damage evidence** per hit vs poke-engine's damage range); poke-engine's
simultaneous-move MCTS (decoupled UCB1) for `ms` = 1000 ms on each, in parallel worker processes; action score = mean
visit share + prior_weight · π(a) (prior 0.25 with a v1 net; 0 and MODEL `none` in gen9); argmax. Options tested:
c_puct, root_game (Bayesian root game), rm_gamma, value_weight (v1 lookahead). Tracked beyond poke-env: Wish / Future
Sight timers, Rest sleep, last moves, substitute HP [M] search.py docstring. Inference accuracy: whole set right 56–58%,
hidden item 82–88%, move 79–80%, ability 96–97% (gen4) [M] paper.

### 5.3 Fitted leaf evaluation (`scripts/fit_eval.py`, `src/porygon/data/eval_r1.json`)
f(s) ∈ ℝ⁶⁴ (side one − side two): the stock score/100 plus corrections (alive counts, HP bins, status × HP, items, sash,
hazards per mon, sub, leech seed, confusion, taunt, yawn, locks, boosts × attacking category, speed advantage, screens,
tailwind, wish, future sight, toxic counter). P(win | s) = σ(wᵀ f(s)), squared-probability loss, L2 towards the stock
score (cross-validated), splits by team pair, and a fixed effect per team pair so that only within-pair variation
identifies w. The search uses σ(k · wᵀ(f(s) − f(s_root))) with k = 2. Data: 39,823 gen4 / 39,970 gen9 engine self-play
games (stock eval, 20k iterations/move). Held-out log loss gen4 0.526 (stock) → 0.511 (all features); gen9 0.536 → 0.514
[M] paper.tex §3, Table 1.

### 5.4 Hybrid H(δ) (`src/porygon/netsearch.py`)
The raw v2 net samples a_net at T 0.5; our pure search runs unchanged; its pick a_s replaces a_net only if
Q(a_s) − Q(a_net) > δ, Q = mean over determinizations of total_score / visits. δ chosen offline by a pre-registered rule
on re-scored study positions (Stage 1), then tuned on Showdown (Stage 2) [M] netsearch.py docstring, NETSEARCH-PREREG.md.

### 5.5 lookahead2 (`src/porygon/lookahead2.py`, `v3leaf.py`)
At each decision with ≥ 2 legal actions: K = 8 worlds from `search.convert`; every (our action, their option) pair simulated
one turn with poke-engine's `generate_instructions` over all chance branches (low-probability ones dropped); leaves scored by
the v2 net's P(win) on the root observation patched with the engine state (`LeafEncoder`); forced opponent switches take
the switch-in worst for us; the K payoff matrices form a zero-sum Bayesian game solved by regret matching+ with linear
averaging; Q(a) = mean_k M_k[a]·y_k; the lookahead's move replaces a_net iff Q(a_L) − Q(a_net) > δ (δ 0.01 for cd@605,
0.005 for cd@805, chosen by a pre-registered precision rule: the smallest margin from which ≥ 90% of overrides are
confirmed by a K = 32 judge). No time budget (deterministic given worlds); ~0.85 s per searched decision, ~1.6–1.7k
leaves, one batched GPU forward [M] docstring, la2-bet1 VERDICT, la2-delta800 VERDICT.

### 5.6 deepen (`src/porygon/deepen.py`; design `docs/netsearch/SCALE-SEARCH-DESIGN.md`)
lookahead2's tree (budget 0 = lookahead2 exactly; parity 120/120) plus best-first selective deepening in waves: leaf
priority = reach × 4v(1 − v), reach = product of world weight, row weight from (x + π + [a = a_net]) and column weight from
(y + q), q = the net's opponent-action head; expansions keep our options by the policy's mass and theirs by the opponent
head; identical states share a leaf (DAG). Budgets 1,024 (D1) / 4,096 (D4) extra leaves; median 2.44 / 5.00 / 8.99 s per
decision at 1× / 4× / 8× [M] docstring, dbench2 RESULT.

### 5.7 Routes that failed their gates (methods only)
- **V3** (v2 net inside an MCTS tree, Python then Rust): gates P1 (leaf-encoder parity) and P2 (≥ 2k leaves/decision)
  passed; I1 (≥ 35,300 batched leaves/s on the slot T4) failed at 7,465 (`docs/netsearch/V3-*`).
- **NNUE** (a 154,609-parameter CPU student on poke-engine State features distilled from G9q-cd@805, `src/porygon/nnue/`):
  gate 1 compares AUC / calibrated log loss vs teacher and evaluate() on self-play (true State) and human (K 4
  determinizations) sets; failed (NNUE-GATE1-PREREG.md v2).

## 6. Evaluation methodology
### 6.1 Duplicate pairs
A patched Showdown server (`patches/showdown-duplicate-battles.patch`, `scripts/build_eval_server.sh`) plays battles 2i
and 2i+1 with the same seed and the teams swapped and logs both true teams. A pair scores (s₀ + s₁)/2 ∈ {0, ¼, ½, ¾, 1};
the estimate is the mean pair score and the 95% interval is mean ± 1.96·sd/√pairs [M] CONTRACTS.md. In random battles
pairing is worth only 1.1–1.4× the games (most variance is in play, not the team draw); σ ≈ 0.319 per pair at 50%, ≈ 19%
less variance than unpaired games [M] paper §Method, wsp seqtest-plan.md. A 400-game run is ±4.4–4.9 points; a 3-point
difference needs ~2000 games per arm. Runs from before 2026-10-27 have no pairs (binomial intervals).
In-process duplicate pairs (`scripts/wsp/spd_h2h.py`, `porygon.fastenv_eval`) use pair-indexed seeds; ~205–222k games/h on
8 CPUs; validated against Showdown (G9p@25M vs P1: 65.9 ± 1.1 in-process vs 65.7 ± 2.7 real; P1 vs heuristic 78.8 ± 1.7 vs
80.0 ± 3.2) [M] memory `benchmark-method.md`. Rule: in-process games count for net-vs-net and net-vs-bot only; anything
involving our search or Foul Play runs on Showdown.

### 6.2 Group-sequential tests (`scripts/wsp/seq_design.py`, `seq_monitor.py`; design `runs/cmp/seq-gain.json`)
Unit = a duplicate pair. Gain design: H0 θ ≤ 50% vs θ₁ = 52%, α 0.025 one-sided, power 0.80, looks every 100 pairs,
n_max 2400 pairs; Lan–DeMets error spending with Hwang–Shih–DeCani γ −4 (α, ≈ O'Brien–Fleming, final bound 2.10) and γ −2
(β); non-binding futility; SE re-estimated at each look. Simulated α 0.021, power 0.81; expected 1041 pairs at θ 50%
(−48% vs fixed 2000) and 1516 at 52%. Reports: the median-unbiased estimate with a 95% stage-wise CI (the naive mean is
biased by up to ±0.9 points) and the 95% repeated CI. Two one-sided tests ("A > B", "B > A") on the same pairs; the
nested pair-id order is the decision of record. Bet 1 / scale gate use the same machinery with θ₁ 54% and power 0.9
[M] seqtest-plan.md, la2-bet1 VERDICT, scale-gate VERDICT-B. **Scope (user, 2026-11-09):** internal go/no-go only;
headline numbers stay fixed-n.

### 6.3 Pre-registration practice
Every pipeline writes `PREREG.txt` (read-only, md5 sent to the orchestrator) before its first game, then `STATUS`, the
result file(s) and `FAILED` on failure; a cluster-side job chain reads the result by the registered rule ("read
mechanically by …") [M] CONTRACTS.md, PLAN.md. Rules: follow the rule as written; add a parallel arm rather than
override; never lower a gate after data exist (fixing a rule before any data is a new version, e.g. NNUE gate 1 v2);
confirm 400-game results with 1000; change one thing at a time; seeds are grep-checked as unused; the same seed replays
the same team pairs, so confirmations use a new seed; void rules (e.g. HARNESS_FAIL, > 2% search fallbacks, < 450
pairs ⇒ replay with the same seeds) [M] ORCHESTRATOR.md, fastenv HANDOFF. md5s of every PREREG on the cluster:
`/vol/bitbucket/fr1226/porygon-thesis/prereg-md5.txt`.

### 6.4 Harnesses and opponents
| harness | used for | settings |
|---|---|---|
| gloria `slurm_foulplay.sbatch` / `vs_foulplay.py` | gen4/gen9 search vs Foul Play (Oct–Nov 8) | 4 Foul Play instances, 24 CPUs, ~3.5 h per 400 games; readiness poll instead of a 3 s sleep from 11-09 |
| netv2 `netv2_foulplay.sbatch` (fp9) | raw nets vs Foul Play | K = 20 series, Foul Play 8f541f8, fp-parallelism 1 |
| netv2 `netv2_vs_search.sbatch` | raw nets vs pure search | 1 shard × 21 workers, 5 concurrent; seed 738164 replays 200 team pairs |
| fastenv `netsearch_slot.sbatch` / `lookahead2_slot.sbatch` (Stage 3) | hybrid, lookahead2, deepen | 11 engine processes per side, 3 concurrent, blocks of pairs on a fresh server seeded `<tag>b<j>`, code frozen in the run dir |
| in-process spd_h2h / netv2_cmp_step | net vs net | fastenv simulator, pair-indexed seeds |
- **Foul Play** (pmariglia, GPL-3.0; cluster clone `/vol/bitbucket/fr1226/foul-play`, commit 8f541f8 in every porygon-ws run,
  gen9 venv `.venv-gen9` with our `.engine-gen9` on PYTHONPATH) at default settings: 2 sequential 1 s searches on one core
  = 2 core-s/move; ours 4 × 1 s parallel = 4 (`--fp-parallelism P` gives it 2P states on P cores) [M] memory.
- **Harness offset:** the Stage 3 harness gives raw nets ≈ 5 points more vs search than netv2's (48.05 vs 43.4 for G9p-cd,
  cal-03): compare arms within one harness.
- Raw networks are always evaluated at temperature 0.5 (cooler gained nothing for v1; v1-19).
- General strength only: never model Foul Play's decision rule; confirm Foul Play gains against our own bots too
  [M] PLAN.md "Experiment discipline".
