# Porygon
A Pokémon Showdown agent for random battles (gen4, gen9): determinized MCTS on poke-engine with a fitted leaf
evaluation and set inference, plus a PPO self-play network (gen4).

**Goal:** the best bot in gen4 and gen9 random battles, and a network strong enough to rival or beat search on
its own. The bar is Jaxcalibur (gen9, closed source: self-play transformer + pUCT, peak 2557 Elo / 95.3 GXE, #1
on the ladder); Foul Play (pmariglia, open source, peak 2341 / 88 in gen9) is the strongest bot we can benchmark.

## Where it stands (95% intervals from duplicate pairs)
| | vs Foul Play |
|---|---|
| gen4, search with fitted eval (default), 4 × 1 s, prior 0.25 | **59.2% ± 2.2%** (1599 games) |
| gen4, same, compute matched (us 2 × 1 s / Foul Play 4 × 1 s) | 57.8% ± 2.2% / 58.4% ± 2.2% |
| gen4, stock eval (and stock weights at ×2 scale) | 54.5% ± 1.5% (54.5% ± 2.3%) |
| gen4, oracle: true sets of revealed mons / whole team | 62.1% ± 3.2% / 64.4% ± 3.0% |
| gen9, pure engine search (no network), stock / stock ×2 / fitted ×2 | 51.7% / 53.3% / 53.3% (± 2.4%) |

Opponent inference is near its ceiling (whole set right 56–58%, hidden item 82–88%, move 79–80%, ability
96–97%). The gen9 fit beat stock 60.2% in engine self-play but not measurably against Foul Play.

**Networks (2026-11-09).** These come from the network-first campaign; its code lives on the `ws/*` branches (not
merged yet), and its state and handoffs are in `.orchestra/`. Raw networks at temperature 0.5:
| | result |
|---|---|
| gen4, v2 network @17M steps vs the old v1 network @46M | 56.9% ± 2.9% (1008 games) |
| gen9, v1-style BC vs v2 BC (P1), same human data | 22.4% ± 2.4% (1008 games) |
| gen9, P1 (v2 BC on ≥1900-rated replays) vs our gen9 search / vs Foul Play | 31.0% ± 4.3% / 33.8% ± 4.7% |
| gen9, P1 + 13M RL steps (G9p) vs our gen9 search | 35.8% ± 4.8% |
The old gen4 v1 network alone scores 46.2% ± 3.8% vs Foul Play (MissingNo.Zero's raw net: 61.6%).

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
- **Judging networks:** oracle labels from the same engine side with the search, so they cannot judge the network.

The 0.25 network prior does help the gen4 search (+3.8 ± 3.5 vs none, 800 games).

## Roadmap
1. **A strong network (the main lever).** v2 is built: damage inputs, aux heads for opponent sets and next action,
   small-init embeddings, gen9 tera actions. Two gen9 RL runs are in progress, one from scratch and one from human
   BC. Next: grow the raw network past our search head to head, then past Foul Play (gen9 target ~75%).
2. **Network in the search:** prior and leaf value inside the tree. The gap to the engine is network FLOPs, not
   Python (`.orchestra/SPEEDUPS.md`). Prototype in Python first; a batched-leaf Rust tree only if it gains on
   Showdown.
3. **Stronger distillation targets:** a longer or network-guided search as teacher, with sharpened targets.
4. **Leaf evaluation, round 2:** refit on self-play played with the fitted evaluation; TreeStrap-style targets
   from deeper search; recheck the leaf scale whenever the time per move changes. Engine-only head-to-heads
   over-predict Showdown gains (gen9: 60% in self-play, +1.6 ± 3.4 vs Foul Play): confirm on Showdown.
5. **Paper** (`docs/paper/`): IEEE CoG / workshop on the fitted evaluation; see `docs/paper/README.md`.
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
conversion, action space, set library, engine build). gen9 search runs without a network: MODEL `none`.

## Train
```bash
scripts/start_servers.sh 10 8100        # stop: scripts/stop_servers.sh
python -m porygon.train --async-envs --overlap-update --n-battles 60 --pool-battles 30 \
  --ports 8100-8109 --resume runs/<run>/final_model.zip --total-steps 500_000_000
```
Always `--async-envs --overlap-update` (lockstep decays to ~800 steps/s). `--pool-battles` adds battles against
past snapshots and baseline bots; `--anchor CKPT` evaluates against a fixed checkpoint and picks `best_model` by
it. The learning rate is `10^-4.23 / (8x + 1)^1.5` with `x` = steps / `--total-steps`.

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

## Credits
- PPO recipe: Jett Wang, *Winning at Pokémon Random Battles Using Reinforcement Learning* (MIT MEng thesis, 2024).
- [poke-engine](https://github.com/pmariglia/poke-engine) (MIT) by pmariglia: the simulator behind the search.
- [Foul Play](https://github.com/pmariglia/foul-play) (GPL-3.0) by pmariglia: reference and benchmark; no Foul Play
  code is included here.
- [poke-env](https://github.com/hsahovic/poke-env) and [Pokémon Showdown](https://github.com/smogon/pokemon-showdown).
