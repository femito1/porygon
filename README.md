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

The gen4 network alone is far weaker than the search (it loses ~75% to it). The gen9 fit beat stock 60.2% in
engine self-play but not measurably against Foul Play. Opponent inference is near its ceiling (whole set right
56–58%, hidden item 82–88%, move 79–80%, ability 96–97%).

**Tried, no gain — don't retry as is:** 8 determinizations instead of 4; root PUCT on the prior (`--c-puct`);
regret matching in the tree (`--rm-gamma`); Bayesian root game (`--root-game max`; `sample` is −13 points);
prior weight 0.5 / 1.0; one-turn value lookahead (`--value-weight`); luck-adjusted fit targets (raw outcomes are
as good); patched vs stock engine (correctness fix, neutral); PPO 400M → 600M and +40M steps at the learning-rate
floor; boosted embedding learning (`--embedding-lr-mult`); one round of expert iteration (9k battles); oracle
labels from the same engine (side with the search, cannot judge the network). The 0.25 network prior does help
(+3.8 ± 3.5 vs none, 800 games).

## Roadmap
1. **A strong network (the main lever).** Likely gaps vs Jaxcalibur, by expected value: damage calculations as
   inputs (per pair of mons; ours gets only power, type effectiveness and stats); auxiliary heads predicting the
   opponent's hidden sets and next action; ~5× more games and a bigger model (8.5M vs our 3.5M); a learning-rate
   schedule that does not decay to ~2·10⁻⁶ (ours did by 400M steps, so the "plateau" may be no learning).
   A gen9 network needs tera actions (poke-env actions 22–25). Measure every network against our search head to
   head (target: beat it) and against Foul Play (target in gen9: ~75%, Jaxcalibur's level by Elo).
   Undecided: gen9 first (where the bar is) or gen4 (existing baseline network).
2. **A faster training environment:** self-play inside poke-engine (Rust) with a partial-observation encoder
   instead of Showdown servers (~2k steps/s now). Worth it once the network shows it responds to data.
3. **Network in the search:** the network as prior and leaf value inside the tree needs Rust-side inference
   (Python evaluates ~1.3k states/s per thread vs the engine's ~430k iterations/s); then expert iteration at scale.
4. **Leaf evaluation, round 2:** refit on self-play played with the fitted evaluation; TreeStrap-style targets
   from deeper search; recheck the leaf scale whenever the time per move changes. Engine-only head-to-heads
   over-predict Showdown gains (gen9: 60% in self-play, +1.6 ± 3.4 vs Foul Play): confirm on Showdown.
5. **Paper** (`docs/paper/`): IEEE CoG / workshop on the fitted evaluation; see `docs/paper/README.md`.
6. **Search fidelity left:** locked moves always last 3 turns (gen4: 2–3); a taunt applied after its target moved
   misses that turn; encore needs the engine's last-move tracking; damage evidence ignores crits, stat stages and
   screens; gen9 team sampling lacks the Fire / Freeze-Dry weakness caps.
7. **A public rating:** the official ladder is closed to us (both accounts flagged as bots, datacenter IPs locked);
   the PokéAgent Challenge ladder has no random battles.

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
