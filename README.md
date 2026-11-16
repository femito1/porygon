# Porygon
A Pokémon Showdown agent for random battles (gen4, gen9): determinized MCTS on poke-engine with a fitted leaf
evaluation and set inference, plus a PPO self-play network; in gen9 the benchmark bot is a network + search hybrid.

**Goal:** the best bot in gen4 and gen9 random battles, and a network strong enough to rival or beat search on
its own. The bar is Jaxcalibur (gen9, closed source: self-play transformer + pUCT, peak 2557 Elo / 95.3 GXE, #1
on the ladder); Foul Play (pmariglia, open source, peak 2341 / 88 in gen9) is the strongest bot we can benchmark.
Metagross (a 142M-parameter policy on Foul Play's search) cannot be one: its weights are unreleased, its repo has no
licence, and its author's own head-to-head against plain Foul Play is 44.0% [37.3, 50.9].

## Where it stands (95% intervals from duplicate pairs)
| | vs Foul Play |
|---|---|
| gen4, search with fitted eval (default), 4 × 1 s, prior 0.25 | **59.2% ± 2.2%** (1599 games) |
| gen4, same, compute matched (us 2 × 1 s / Foul Play 4 × 1 s) | 57.8% ± 2.2% / 58.4% ± 2.2% |
| gen4, stock eval (and stock weights at ×2 scale) | 54.5% ± 1.5% (54.5% ± 2.3%) |
| gen4, oracle: true sets of revealed mons / whole team | 62.1% ± 3.2% / 64.4% ± 3.0% |
| gen4, **raw network** R1b-F@300M at T 0.5, no search (old v1 network: 46.2% ± 3.8%) | 55.6% ± 4.1% (400 games) |
| gen9, **raw network** G9q-cd@605M at T 0.5, no search | **58.8% ± 2.9%** (1000 games, confirmed > 50%) |
| gen9, **network + search** (the benchmark bot): G9p-cd net at T 0.5, overridden by our search when its pick is > 0.0212 better in root Q (22% of moves) | 56.2% ± 2.8% (1000 games) |
| gen9, pure engine search on the same team pairs | 51.6% ± 2.8% (1000 games); hybrid − search +4.6 ± 3.9 points |
| gen9, pure engine search (no network), stock / stock ×2 / fitted ×2 | 51.7% / 53.3% / 53.3% (± 2.4%) |

Opponent inference is near its ceiling (whole set right 56–58%, hidden item 82–88%, move 79–80%, ability
96–97%). The gen9 fit beat stock 60.2% in engine self-play but not measurably against Foul Play. The hybrid
(pre-registered, fixed n: `docs/netsearch/NETSEARCH-PREREG.md`) also beats our pure gen9 search head to head,
55.0% ± 3.0% (1000 games). Our raw gen9 network now beats Foul Play on its own, and its point estimate is above the
hybrid's (the two have not met head to head); re-freezing the hybrid on an earlier G9q net (G9q-cd@305M) did not
beat the current one, 50.7% ± 2.8%. The target stays 65–75% against Foul Play; MissingNo.Zero's raw net scores
66.7% in gen9 (61.6% in gen4).

**Networks (2026-11-16).** v2 networks (`src/porygon/v2/`, `fastenv*.py`, `netsearch.py`, the replay corpus), raw
at temperature 0.5:
| | result |
|---|---|
| gen4, v2 network @17M steps vs the old v1 network @46M | 56.9% ± 2.9% (1008 games) |
| gen4, v2 network R1b-F @150M / @300M vs v1 @46M | 80.9% ± 2.3% / 84.6% ± 2.1% (≈ 1000 games each) |
| gen4, R1b-F @50M / 100M / 150M / 300M vs v1 @500M (our best v1 net) | 49.8% / 52.6% / 56.4% / 60.7% (300M: ± 2.6%, 1008 games) |
| gen9, v1-style BC vs v2 BC (P1), same human data | 22.4% ± 2.4% (1008 games) |
| gen9, human BC vs P1: 0.92M / 2.29M / 7.05M decisions (P3, all ≥ 1900-rated) | 48.7% ± 2.1% / 52.8% ± 2.0% / 59.6% ± 2.0% (2000 games each) |
| gen9, BC on P3's size rated ≥ 1500 (U7) vs P3 (≥ 1900) | 47.2% ± 1.2% (6000 games) |
| gen9, P1 (v2 BC on ≥1900-rated replays) vs our gen9 search / vs Foul Play | 31.0% ± 4.3% / 33.8% ± 4.7% |
| gen9, G9p-cd (PPO from P1, lr cooldown branch, 104.7M) vs our gen9 search / vs Foul Play | 43.4% ± 2.8% (1000 games) / 46.2% ± 4.6% (400) |
| gen9, G9q (PPO from P3) vs G9p, step matched at 25M / 50M | 61.1% [55.8, 65.9] / 58.2% [53.8, 62.3] (sequential tests) |
| gen9, G9q @112.5M vs G9p @338.6M | 49.4% ± 2.0% (2000 games) |
| gen9, G9q-cd@605M vs our gen9 search / vs Foul Play | 52.5% ± 2.9% / 58.8% ± 2.9% (1000 games each) |
| gen9, cooled at 50M vs G9q-cd: G9r (power-law lr decay) / G9t (GAE λ 0.5) | 49.5% ± 1.6% / 48.1% ± 1.4% (3200 / 4000 games) |
| gen9, cooled at +50M vs G9q-cd: entropy-coefficient schedule 0.01 → 0.001 (G9qent) | 50.6% [48.9, 52.5] (3000 games) |
| gen9, G9q2 (G9q's recipe, a new seed) vs G9q, step matched at 25M / 50M | 50.2% ± 1.7% / 50.7% ± 2.2% (2800 / 1600 games) |
| gen9, G9q-cd@805M vs G9q-cd@605M | 52.5% [50.7, 54.2] (2800 games, sequential) |
| gen9, **lookahead2** on G9q-cd@605M vs the same raw network | **53.2% [50.6, 55.7]** (1200 games, sequential) |
| gen4, search with the R1b-F@150M / @300M prior vs the v1@500M prior | 51.1% ± 2.0% / 51.9% [50.0, 53.8] (2000 games each) |

On the same team pairs as G9p-cd's 43.4%, G9q-cd@605M gains +9.1 ± 3.9 points against our gen9 search, but its
52.5% [49.6, 55.4] is not yet a confirmed ≥ 50%; the stronger G9q-cd@805M is now being measured against our search
and Foul Play (1000 games each). Scoring the search's leaves with the network works: **lookahead2** simulates every
pair of moves for one turn in sampled worlds, scores each outcome with the network's win probability, and overrides
the raw network's pick only by a clear margin; it beats its own raw network 53.2%. Its tests against Foul Play and
against the benchmark hybrid are queued. Two seeds of G9q's recipe agree within ~2 points. Human-data quality
matters at equal size (U7), so the next BC init, P4, uses all ≥ 1900-rated data (1.8× P3). G9q is the main gen9
run; G9p, G9q2, G9r, G9t and G9qent are parked. In gen4 the R1b-F prior did not clearly help our search, so the
v1@500M prior stays.

**Tried, no gain — don't retry as is:**
- **Search:** 8 determinizations instead of 4; root PUCT on the prior (`--c-puct`); regret matching in the tree
  (`--rm-gamma`); Bayesian root game (`--root-game max`; `sample` is −13 points); prior weight 0.5 / 1.0; one-turn
  value lookahead (`--value-weight`); a G9p prior in the gen9 search (51.5% ± 4.5% vs none).
- **Leaf evaluation and engine:** luck-adjusted fit targets (raw outcomes are as good); patched vs stock engine
  (a correctness fix, neutral).
- **v1 network training:** PPO 400M → 600M and +40M steps at the learning-rate floor; constant lr 3·10⁻⁵ from 600M
  (worse); boosted embedding learning (`--embedding-lr-mult`); one round of expert iteration (9k battles); SWA of
  late checkpoints; direct weight edits (weight-space programming).
- **Search labels as BC targets** (4 × 500 ms search, 108k decisions): 34.9% ± 2.8% vs human BC.
- **PPO settings:** bf16 autocast (broke SB3 PPO); lr 2·10⁻⁴ for v2 (degraded after 7.5M).
- **v2 recipe arms (gen9, one change each from G9q's recipe):** power-law lr decay over 300M (G9r) and GAE λ 0.5
  (G9t, the value a pre-registered pre-check picked) both lost cooled-vs-cooled to G9q-cd at 50M; an entropy
  schedule (G9qent) tied (table). bf16 autocast was only 1.02× faster on an A40, so v2 stays fp32 (TF32 on).
- **A small CPU value net inside the engine (NNUE route):** a 155k-parameter student distilled from G9q, on
  poke-engine state features, was no better than the hand evaluation at determinized human positions (AUC 0.714 vs
  0.718; the teacher 0.748) and cost ~90 µs per forward; it failed its pre-registered offline gate.
- **Actor-stream overlap** (`--actor-stream`, one update of policy lag): +38% steps/s and non-inferior over 5M
  steps, but after 25M steps G9p scored 46.0% ± 2.0% against its own start (a synchronous control over the same
  span: 49.6% ± 2.0%). Reverted. Pair every recipe change with a long-horizon drift check.
- **Judging networks:** oracle labels from the same engine side with the search, so they cannot judge the network.

The 0.25 network prior does help the gen4 search (+3.8 ± 3.5 vs none, 800 games).

## Roadmap
1. **A stronger raw network.** G9q keeps training and still improves (cd@805 beat cd@605); confirm the raw net past
   our gen9 search (≥ 50%) and grow it toward MissingNo.Zero's 66.7% against Foul Play. P4 (all ≥ 1900-rated data,
   1.8× P3) gives BC-P4 and a P4-init run read against G9q at 25M / 50M; the better init seeds the next main run
   with the December sets. The recipe levers tried so far (lr annealing, GAE λ, entropy schedule, bf16,
   actor-stream) did not help. gen4: R1b-F keeps training under its strength guard.
2. **Network-quality leaves in the search (the lever against Jaxcalibur, whose search adds 100–150 Elo).** Done:
   the margin-rule hybrid at the root (the gen9 benchmark bot) and lookahead2 (one ply of network-valued leaves,
   +3 points over its own raw net). Queued: lookahead2 vs Foul Play and vs the benchmark hybrid (adopted iff better).
   Next, chosen by those results: more depth, either a deeper selective GPU lookahead or a second attempt at a fast
   CPU student inside the engine. V3 (the network inside an MCTS tree on the benchmark GPU) stays parked: ~2k
   leaves per decision (`docs/netsearch/V3-GATES-PREREG.md`, `V3-I1-PREREG.md`, `V3-RUST-DESIGN.md`).
3. **Distil the stronger search into the network:** lookahead2 now beats the net, so it is a teacher worth
   copying (search labels from the weaker hand-evaluation search did not help).
4. **Leaf evaluation, round 2:** refit on self-play played with the fitted evaluation; TreeStrap-style targets
   from deeper search; recheck the leaf scale whenever the time per move changes. Engine-only head-to-heads
   over-predict Showdown gains (gen9: 60% in self-play, +1.6 ± 3.4 vs Foul Play): confirm on Showdown.
5. **Paper** (`docs/paper/`): IEEE CoG / workshop on the fitted evaluation; see `docs/paper/README.md` (which also
   notes what the network results could and could not claim as new).
6. **Search fidelity left:** locked moves always last 3 turns (gen4: 2–3); a taunt applied after its target moved
   misses that turn; encore needs the engine's last-move tracking; damage evidence ignores crits, stat stages and
   screens; gen9 team sampling lacks the Fire / Freeze-Dry weakness caps.
7. **A public rating:** Showdown's bot registry (2026-10-13) and opt-in bot matchmaking (2026-10-20) let a
   registered bot ladder against other bots and opted-in humans, from home IPs only. pkmn.bot is a bot-only
   ladder with a gen9 random-battles table. Both need the owner's logins.

## Layout
| | |
|---|---|
| `src/porygon/encoder.py`, `model.py` | battle → observation (field + 12 mon tokens); `BattleTransformer`, `MaskedPolicy` |
| `src/porygon/v2/` | v2 network: observation spec, encoder, model, self-play PPO (`train`), BC pretraining (`pretrain`), head-to-head harness (`evaluate`) |
| `src/porygon/fastenv*.py`, `fastenv/` | in-process Showdown battles (Node workers, no server) for collection and net-vs-net head-to-heads |
| `src/porygon/netsearch.py`, `scripts/fastenv/netsearch_*` | the network + search hybrid (margin rule) and its benchmark jobs |
| `src/porygon/replay.py`, `bc.py`, `scripts/fetch_replays.py`, `resim_inputlog.js`, `convert_replays.py` | human replay corpus: crawl, re-simulate input logs (exact first-person views), convert, BC |
| `src/porygon/env.py`, `async_env.py`, `pool.py`, `train.py` | self-play PPO: envs, async rollouts, opponent pool, training |
| `src/porygon/search.py`, `engine_worker.py` | `SearchPlayer`: poke-engine MCTS over sampled opponent sets (gen4, gen9) |
| `src/porygon/inference.py` | opponent set posterior: generator set library + evidence from the battle log |
| `src/porygon/selfplay.py`, `scripts/gen_teams.js`, `scripts/fit_eval.py` | engine-only self-play; fit the leaf evaluation |
| `src/porygon/data/eval_r1.json`, `eval_gen9_r1.json` | fitted leaf evaluations (gen4: the default; gen9: not yet the default) |
| `src/porygon/lookahead.py`, `expert.py`, `tune.py` | value lookahead; expert iteration; SPSA on the evaluation |
| `src/porygon/decisions.py`, `scripts/analyze_benchmark.py` | benchmark decision logs; pair-scored win rate, inference accuracy |
| `src/porygon/evaluate.py`, `tournament.py`, `play.py` | vs baseline bots; round-robin Elo; play on the official server |
| `src/gloria/` | alias of `porygon` (old name) so old checkpoints load |
| `scripts/build_engine.sh`, `patches/poke-engine-*` | poke-engine 0.0.48: gen4 durations, search options, tunable evaluation |
| `scripts/build_eval_server.sh`, `patches/showdown-*` | benchmark Showdown: duplicate battles, true teams logged |
| `scripts/build_set_library.js`, `score_inference.py` | set libraries (`data/<format>_library.json.gz`); offline inference scoring |
| `scripts/vs_foulplay.py`, `slurm_foulplay.sbatch` | benchmark vs Foul Play |
| `scripts/slurm_*.sbatch` | cluster jobs: train (and `--arm` experiments), search / net head-to-heads, self-play, fit, expert |
| `scripts/oracle_labels.py`, `probe_embeddings.py` | network diagnostics (oracle-labelled decisions; embedding probes) |
| `docs/paper/` | paper draft, figure, literature review and search tool |

## Setup
```bash
git submodule update --init            # smogon/pokemon-showdown
cd pokemon-showdown && npm install && cd ..
python -m venv .venv && .venv/bin/python -m pip install -e .
scripts/build_engine.sh                # poke-engine 0.0.48 gen4 + patches (needs Rust)
WHEEL_DIR=/tmp/w GEN=9 scripts/build_engine.sh && .venv/bin/python -m pip install --no-deps --target .engine-gen9 /tmp/w/*.whl
node scripts/build_set_library.js 2000000 4 gen9randombattle   # only when pokemon-showdown/ changes (also gen4)
```
The generation is process-wide: `PORYGON_GEN=9 PYTHONPATH=$PWD/.engine-gen9:$PWD/src` selects gen9 (inference,
conversion, action space, set library, engine build). gen9 search runs without a network: MODEL `none`; the gen9
benchmark bot is the hybrid (Benchmark).

## Train
```bash
scripts/start_servers.sh 10 8100        # stop: scripts/stop_servers.sh
python -m porygon.train --async-envs --overlap-update --n-battles 60 --pool-battles 30 \
  --ports 8100-8109 --resume runs/<run>/final_model.zip --total-steps 500_000_000
```
Always `--async-envs --overlap-update` (lockstep decays to ~800 steps/s). `--pool-battles` adds battles against
past snapshots and baseline bots; `--anchor CKPT` evaluates against a fixed checkpoint and picks `best_model` by
it. The learning rate is `10^-4.23 / (8x + 1)^1.5` with `x` = steps / `--total-steps`.

v2 networks train with `porygon.v2.train` (cluster: `scripts/slurm_train_v2.sbatch RUN [args]`): lr 7·10⁻⁵ constant,
fp32, the in-process collector with `--no-overlap`, `--stacked-opp` (pool opponents batched; +13% steps/s after
its padding fix). Keep the snapshot cache (`--snapshot-cache`, plus `--snapshot-cache-cpu`) at least the pool size:
cache misses cut R1b-F's speed by about half. Deployment and evaluation nets are short lr-cooldown branches
(G9p-cd@104.7M beat G9p@105.0M 52.05% [50.42, 53.65]); the main run keeps its lr.

Cluster (Imperial DoC Slurm): settings in `slurm.env` (`SHA`, `RESUME`, `POOL`, `TOTAL`, `CHAIN`, `ANCHOR`; never
`sbatch --export`), then `sbatch scripts/slurm_train.sbatch --async-envs --overlap-update`. The job resumes from
its newest checkpoint, mirrors its run to the shared disk, pushes results, and requeues itself if its log is
silent for 45 min. Side-by-side experiment arms that cannot touch the main lineage:
`sbatch -c 16 scripts/slurm_train.sbatch --arm NAME START_ZIP TOTAL [train args]`. Raw-network head-to-heads:
`sbatch scripts/slurm_h2h_nets.sbatch N_BATTLES REF MODEL...`.

## Benchmark
```bash
sbatch --partition=long -c 24 scripts/slurm_foulplay.sbatch MODEL 100 4 [--fp-parallelism P] [vs_foulplay args]
sbatch --partition=long -c 24 scripts/slurm_foulplay.sbatch none 100 4 --gen 9 --eval-config FILE|stock
sbatch scripts/slurm_search.sbatch MODEL --opponent search [porygon.search args]
```
Games are duplicate pairs (same seeds, teams swapped) on a patched Showdown that logs the true teams. Output in
`runs/{foulplay,search}-<date>-<job>/` with `analysis.txt`. Pairing is worth only ~1.1–1.4× the games: a
400-game run is ±4.8 points and a 3-point difference needs ~2000 games per arm. Report 95% intervals from pair
means. Foul Play's default uses 2 core-seconds per move, our search 4; `--fp-parallelism 2` matches them.
Judge changes by general strength: confirm a Foul Play gain head-to-head against our previous search too.
Options: `--oracle revealed|all` (diagnosis), `--engine-python` (another engine build),
`--eval-config FILE|stock`, `--prior-weight` (default 0.25; 0 with MODEL none), `--log-oracle`.

The gen9 hybrid is the player spec `netsearch:NET,margin=0.0212,temperature=0.5,ms=1000,samples=4,workers=11`
(`vs_foulplay.py SPEC --player spec`); the pre-registered runs used `scripts/fastenv/netsearch_slot.sbatch PLAN`
with the plans in `scripts/fastenv/plans/` (`stage3b.env`: hybrid and pure search vs Foul Play).

## Credits
- PPO recipe: Jett Wang, *Winning at Pokémon Random Battles Using Reinforcement Learning* (MIT MEng thesis, 2024).
- [poke-engine](https://github.com/pmariglia/poke-engine) (MIT) by pmariglia: the simulator behind the search.
- [Foul Play](https://github.com/pmariglia/foul-play) (GPL-3.0) by pmariglia: reference and benchmark; no Foul Play
  code is included here.
- [poke-env](https://github.com/hsahovic/poke-env) and [Pokémon Showdown](https://github.com/smogon/pokemon-showdown).
