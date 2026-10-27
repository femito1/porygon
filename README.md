# Porygon
A Pokémon Showdown agent trained with self-play reinforcement learning (gen4 random battles).

The training recipe follows Jett Wang, *Winning at Pokémon Random Battles Using Reinforcement
Learning* (MIT MEng thesis, 2024 — [dspace](https://hdl.handle.net/1721.1/153888)), whose PPO
self-play agent reached 1693 Elo (rank 8) on the gen4randombattle ladder:

- **Self-play PPO** (stable-baselines3) where both players of every battle are the learner and
  both trajectories are training data; sparse ±1 win reward; γ = 0.9999, λ = 0.754, clip 0.0829,
  7 epochs, ~40k transitions per update, and the annealed learning rate
  `10^-4.23 / (8x + 1)^1.5` that the thesis found to be the single biggest factor.
- **Markov state encoding**: every multi-turn condition (weather, screens, hazards, boosts,
  status counters, volatile effects) is encoded with its elapsed turns.
- **Action masking**: illegal actions get probability 0 in rollouts and in the PPO update.

Differences from the thesis: a transformer over field + 12 Pokémon tokens instead of an MLP,
learned embeddings for species/moves/items/abilities, a movepool prior for unrevealed opponent
sets (from Showdown's random-battle data), and poke-env's slot-based 10-action space.

## Results (gen4randombattle)
- **Raw network** (PPO, 60M steps): 83.5% vs `SimpleHeuristicsPlayer`, 95% vs `MaxBasePowerPlayer`
  (200 battles each) — on par with the thesis (~80% at 40M, ~85% at 150M steps). Continued
  training with the opponent pool keeps improving: 300M beats 60M 71.5% and the heuristic bot
  89.5% (400 battles each); 400M beats 300M 54.5% (600 battles).
- **Checkpoint tournament** (`porygon.tournament`, 100 battles per pairing, Bradley-Terry Elo):
  60M 1711 > 46M 1663 > 31M 1635 > 10M 1493 > heuristic 1457 > 2.5M 1346 > max_power 1194.
  Every checkpoint beats every earlier one; gains slow after ~30M steps.
- **Search** (`porygon.search`, defaults 1000 ms/move, 4 sampled opponent sets, prior weight
  0.25): beats the raw network ~75% head-to-head and the heuristic bot 97%.
- **vs Foul Play** (pmariglia's search bot, 1000 ms/move, `scripts/slurm_foulplay.sbatch`): the search
  player wins **52.6% ± 2.2%** over every run of the default search (1048-944: 400M, 400M with the
  fidelity fixes, 500M three times; single 400-battle runs ranged 50-55%, 49% with the 91M network);
  the raw network alone 16.5% (91M, 200). None of the changes since the 400M network (fidelity fixes,
  500M, value lookahead 0.5: 49.7%, prior weight 1.0: 49.6%) moved it measurably; the engine patch is
  neutral head-to-head too (49.1% ± 4.1% vs the stock engine, 200 duplicate pairs).
- **Benchmark method**: benchmarks run on a patched Showdown (`scripts/build_eval_server.sh`) that plays
  duplicate pairs (same seeds, teams swapped) and logs the true teams; `scripts/analyze_benchmark.py`
  scores the pairs, our opponent inference and the decisive turns. Pairing is worth only ~1.1x the games
  vs Foul Play (1.4x search vs search): outcomes are mostly decided in-battle, not by the teams. Our
  determinizations get a hidden ability right ~95%, an item ~60%, a move ~65%, a whole set ~35%.
- **Value lookahead** (`porygon.lookahead`: one turn ahead, every outcome scored by the network's value,
  400M network, 300 battles each, `scripts/slurm_search.sbatch`): on its own it beats the raw network
  only 53.3% ± 5.6%; mixed half-and-half into the search it beats the search 53.7% ± 5.6%, and replacing
  the engine search with it loses 42.7% ± 5.6%. One turn of the network's value is not yet a better
  evaluation than poke-engine's 1 s search.
- **Ladder**: 13-7 with the raw 60M network. Both accounts used (femito, porygon-agent) are now
  flagged as bots by Showdown, which matches bots only with players who opted in to bot games;
  datacenter IPs are locked too, so ladder games could only run from a home connection anyway.

## Layout
| | |
|---|---|
| `src/porygon/encoder.py` | battle → flat observation (field token + 12 mon tokens) |
| `src/porygon/model.py` | `BattleTransformer` features extractor, `MaskedPolicy` (per-action pointer head) |
| `src/porygon/env.py` | `SelfPlayEnv` (poke-env PettingZoo env), vectorised self-play, `PolicyPlayer` |
| `src/porygon/async_env.py` | asynchronous rollouts (`--async-envs`, `--overlap-update`) |
| `src/porygon/pool.py` | opponent pool: past policy snapshots + baseline bots |
| `src/porygon/train.py` | PPO training with eval-vs-baselines, checkpoints, pool snapshots |
| `src/porygon/search.py` | `SearchPlayer`: network-guided poke-engine MCTS over sampled opponent sets |
| `src/porygon/engine_worker.py` | poke-engine search subprocesses used by `search.py` |
| `src/porygon/lookahead.py` | one-turn lookahead scored by the network's value (`--value-weight`), leaf encoder |
| `src/porygon/expert.py` | expert iteration: search-vs-search training data, distillation into a checkpoint |
| `src/porygon/evaluate.py`, `tournament.py` | win rates vs baseline bots; round-robin Elo |
| `src/porygon/play.py` | play on the official server (ladder / challenges) |
| `src/gloria/` | alias of `porygon` (the old name) so checkpoints saved before the rename load |
| `scripts/slurm_train.sbatch` | PPO training job for the Imperial DoC Slurm cluster |
| `scripts/slurm_foulplay.sbatch` | Foul Play benchmark job (local Showdown server on the node) |
| `scripts/slurm_search.sbatch` | `porygon.search` head-to-heads on a node (fits next to a training job) |
| `scripts/slurm_expert.sbatch`, `slurm_distill.sbatch` | expert-iteration jobs: generate + distil, or re-distil existing data |
| `scripts/sync_results.sh`, `fetch_results.sh` | push a run's results to a git repo / download and archive them |
| `scripts/pod_*.sh`, `pod_retry_deploy.py`, `vm_setup.sh` | fallbacks: RunPod pods, a plain GPU VM |
| `scripts/vs_foulplay.py` | benchmark vs Foul Play (setup in its docstring) |
| `src/porygon/decisions.py`, `scripts/analyze_benchmark.py` | benchmark decision logs; duplicate scoring, inference accuracy, swings |
| `scripts/build_eval_server.sh`, `patches/showdown-*` | benchmark Showdown: duplicate battles, true teams logged |
| `scripts/build_engine.sh`, `patches/` | poke-engine 0.0.48 with gen4 taunt / encore / confusion durations |
| `scripts/ladder_watchdog.sh` | long ladder sessions |

## Setup
```bash
git submodule update --init            # official smogon/pokemon-showdown
cd pokemon-showdown && npm install && cd ..
python -m venv .venv && .venv/bin/pip install -e .
scripts/build_engine.sh                # poke-engine 0.0.48 gen4 + patches/ (for search; needs Rust)
node scripts/build_data.js             # only if the Showdown checkout changes
```

## Train
Locally (throughput is bound by battle latency, ~3 cores per Showdown server):
```bash
scripts/start_servers.sh 10 8100        # stop: scripts/stop_servers.sh
python -m porygon.train --async-envs --overlap-update --n-battles 60 --pool-battles 30 \
  --ports 8100-8109 --resume runs/<run>/final_model.zip --total-steps 500_000_000
```
Use `--async-envs --overlap-update`. In the default lockstep mode every step waits for the slowest of
~150 battles, and throughput decays within a job (1.4k → ~800 steps/s over 20 h). `--async-envs`
steps every battle independently and `--overlap-update` keeps battles running during the PPO
update (training data at most one update old); together they held 1.5–2.0k steps/s for 17 h on a
30-core node. The PPO update (~17 s per 40k steps) is now the bottleneck, and evals stall
collection for ~200 s every 2.5M steps.

`--pool-battles` adds battles against frozen snapshots (one added every `--snapshot-every` steps)
and baseline bots; seed a pool with `python -m porygon.pool CKPTS... --out DIR`. `--anchor CKPT`
also evaluates against a fixed earlier checkpoint and picks `best_model` by that score: the
baseline bots saturate around 85%, while head-to-head win rates keep showing progress.

On the Imperial DoC cluster, from the repo copy on `/vol/bitbucket`: put `SHA=<showdown commit>`,
`RESUME=<zip>`, `POOL=<pool dir>`, `TOTAL=<steps>`, `CHAIN=<follow-up jobs>`, optionally
`ANCHOR=<zip or .pt>` (default `RESUME`) in `slurm.env`
(never `sbatch --export`, which gets jobs held), then `sbatch scripts/slurm_train.sbatch`. The job
sets itself up, resumes from the newest checkpoint, uses all cores it gets (30 on a 32-core
request), stops after 70.5 h, keeps its run directory and pool on the node's local disk (mirrored
to the shared disk every 10 min, at the end and on `scancel`) and pushes results every 20 min;
extra arguments go to `porygon.train`: `sbatch scripts/slurm_train.sbatch --async-envs --overlap-update`.
`TOTAL` sets the learning-rate schedule (`x` = steps / `TOTAL`), so raise it to continue a
finished run. Fetch results anywhere with
`scripts/fetch_results.sh` (into `runs/remote/`). The push holds only the newest checkpoint and
pool snapshots, so the fetch also copies them into `runs/archive/` (never overwritten); the full
history stays on the cluster (`runs/<run>/checkpoints/` every 5M steps, `$POOL/*.pt` every 2.5M)
for `rsync`.

## Expert iteration
Distil the search player into the network: search-vs-search games record every searched decision
(observation, the move distribution search played, and the shaped return PPO would assign), then
the checkpoint is trained toward them. The output is a normal PPO checkpoint that PPO can resume.
```bash
python -m porygon.expert generate MODEL --games 9000 --out data/round1 --ports 8100-8103   # 300 ms x 4 samples
python -m porygon.expert distill MODEL data/round1 --out runs/round1/distilled.zip      # --temperature 0.25
```
Raw visit shares are much flatter than the move search plays (entropy ~1.2 vs the network's
~0.33 nats), so `distill` sharpens them (`--temperature`, default 0.25) and keeps the policy near
its source (`--beta`, KL weight). On the cluster: `EXPERT_MODEL=<zip>` (optionally `EXPERT_GAMES`,
`EXPERT_MS`) in `slurm.env`, then `sbatch scripts/slurm_expert.sbatch` (9,000 battles ≈ 640k
decisions, ~11 h on 30 cores); re-distil existing data with `scripts/slurm_distill.sbatch`.

**Result so far: no gain.** One round from 300M: the distilled raw network at best tied its source
(50.5% with `--beta 3`; worse with weaker anchoring), and search with it as prior tied search with
the source. A 300 ms search scored by poke-engine's evaluation is no better a teacher than PPO;
it needs the network's value inside the search first.

## Evaluate / play
```bash
python -m porygon.evaluate MODEL --port 8000 --n-battles 500 [--anchor OLD_MODEL]
python -m porygon.tournament MODEL_A MODEL_B ... --n-battles 100   # .zip checkpoints or pool .pt
python -m porygon.search MODEL --opponent {policy,heuristic,max_power,search} --n-battles 100
python -m porygon.search MODEL --opponent search --value-weight 0.5 --device cuda   # mix in the value lookahead
python -m porygon.play MODEL --username USER --accept 5 --search                # search player; --ladder N for the ladder
python scripts/vs_foulplay.py MODEL --player search --opponents fpa0 fpa1 ...   # setup in its docstring
sbatch scripts/slurm_foulplay.sbatch MODEL 100 4   # the same on a cluster node: 4 Foul Play instances x 100 games
PS_PASSWORD=... PLAY_ARGS=--search scripts/ladder_watchdog.sh USER MODEL TOTAL_GAMES 4   # from a home connection
```

## Roadmap
1. **Beat Foul Play convincingly** (now 52.6% ± 2.2%, mostly from using 4x its compute). The value
   lookahead, the fidelity fixes, more PPO (600M trained) and a higher prior weight all left it where it
   was, so the next step is diagnosis: an oracle run that gives the search the true sets of revealed
   mons bounds what better opponent inference (items ~60%, moves ~65% right) can be worth. The network
   in the engine's tree (priors, then values) is the other candidate.
2. **A public rating**: the [PokéAgent Challenge](https://pokeagentchallenge.com/battling.html)
   runs a live leaderboard for AI agents (bots welcome; Foul Play won its Gen 9 OU final). It has
   no random battles: Gen 1–4 OU, Gen 9 OU and VGC, headline boards Gen 1 OU and Gen 9 OU. Entering
   needs team handling and usage-stat opponent priors instead of random-battle sets, and retraining.
3. **Search fidelity**: done: taunt / encore / confusion end after their random gen4 durations
   (patched engine; stock 0.0.48 gen4 keeps them forever), Substitute HP is ¼ max HP minus an estimate
   of each hit it took, charging two-turn moves and our recharge come from the request. Left: locked
   moves (Outrage) always last 3 turns (gen4: 2-3), taunt applied after its target moved misses that
   turn, encore needs the last move; occasional `NonFinite` engine panics (sample skipped). Tune time
   per move (1 s beat 0.3 s; more may help).
4. **Training speed**: make the PPO update cheaper (bf16, fewer/larger minibatches) and stop evals
   from stalling collection; log per-rollout collect / update / callback time to find the rest.
5. **gen9randombattle**: encoder, action space (tera) and vocab already support it; start from the
   gen4 model.

## Credits
- Training recipe: Jett Wang, *Winning at Pokémon Random Battles Using Reinforcement Learning*
  (MIT MEng thesis, 2024).
- [poke-engine](https://github.com/pmariglia/poke-engine) (MIT) by pmariglia is the battle
  simulator behind `porygon.search`.
- [Foul Play](https://github.com/pmariglia/foul-play) (GPL-3.0) by pmariglia was a reference
  for how a Showdown battle maps to a poke-engine state and is our strongest benchmark; no
  Foul Play code is included in this repository.
- [poke-env](https://github.com/hsahovic/poke-env) and
  [Pokémon Showdown](https://github.com/smogon/pokemon-showdown).
