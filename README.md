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
- **vs Foul Play** (pmariglia's search bot, 1000 ms/move): search player 49% (204 battles), raw
  network 16.5% (200).
- **Ladder**: 13-7 with the raw 60M network. Showdown locks datacenter IPs, so ladder games
  must run from a home connection.

## Layout
| | |
|---|---|
| `src/porygon/encoder.py` | battle → flat observation (field token + 12 mon tokens) |
| `src/porygon/model.py` | `BattleTransformer` features extractor, `MaskedPolicy` (per-action pointer head) |
| `src/porygon/env.py` | `SelfPlayEnv` (poke-env PettingZoo env), vectorised self-play, `PolicyPlayer` |
| `src/porygon/pool.py` | opponent pool: past policy snapshots + baseline bots |
| `src/porygon/train.py` | PPO training with eval-vs-baselines, checkpoints, pool snapshots |
| `src/porygon/search.py` | `SearchPlayer`: network-guided poke-engine MCTS over sampled opponent sets |
| `src/porygon/engine_worker.py` | poke-engine search subprocesses used by `search.py` |
| `src/porygon/expert.py` | expert iteration: search-vs-search training data, distillation into a checkpoint |
| `src/porygon/evaluate.py`, `tournament.py` | win rates vs baseline bots; round-robin Elo |
| `src/porygon/play.py` | play on the official server (ladder / challenges) |
| `src/gloria/` | alias of `porygon` (the old name) so checkpoints saved before the rename load |
| `scripts/slurm_train.sbatch`, `slurm_expert.sbatch` | training / expert-iteration jobs for the Imperial DoC Slurm cluster |
| `scripts/sync_results.sh`, `fetch_results.sh` | push a run's results to a git repo / download and archive them |
| `scripts/pod_*.sh`, `pod_retry_deploy.py` | the same on RunPod |
| `scripts/vs_foulplay.py`, `slurm_foulplay.sbatch` | benchmark vs Foul Play, locally or as a cluster job |
| `scripts/ladder_watchdog.sh` | long ladder sessions |

## Setup
```bash
git submodule update --init            # official smogon/pokemon-showdown
cd pokemon-showdown && npm install && cd ..
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/pip install poke-engine --no-cache-dir \
  --config-settings="build-args=--features poke-engine/gen4 --no-default-features"   # for search; needs Rust
node scripts/build_data.js             # only if the Showdown checkout changes
```

## Train
Locally (throughput is bound by battle latency, ~3 cores per Showdown server):
```bash
scripts/start_servers.sh 10 8100        # stop: scripts/stop_servers.sh
python -m porygon.train --n-battles 60 --pool-battles 30 --ports 8100-8109 \
  --resume runs/<run>/final_model.zip --total-steps 400_000_000
```
`--async-envs` steps every battle independently (`porygon.async_env`) instead of in lockstep, where
each step waits for the slowest of ~150 battles; on a cluster node 48 battles went from 1,776 to
2,940 steps/s. `--overlap-update` (with `--async-envs`) also keeps battles running during the PPO
update, which otherwise takes ~10 of every ~21 s; the training data is then at most one update
old. `--pool-battles` adds battles against frozen snapshots (one added every `--snapshot-every` steps)
and baseline bots; seed a pool with `python -m porygon.pool CKPTS... --out DIR`. `--anchor CKPT`
also evaluates against a fixed earlier checkpoint and picks `best_model` by that score: the
baseline bots saturate around 85%, while head-to-head win rates keep showing progress.

On the Imperial DoC cluster, from the repo copy on `/vol/bitbucket`: put `SHA=<showdown commit>`,
`RESUME=<zip>`, `POOL=<pool dir>`, `TOTAL=<steps>`, `CHAIN=<follow-up jobs>`, optionally
`ANCHOR=<zip or .pt>` (default `RESUME`) in `slurm.env`
(never `sbatch --export`, which gets jobs held), then `sbatch scripts/slurm_train.sbatch`. The job
sets itself up, resumes from the newest checkpoint, uses all cores it gets (30 on a 32-core
request), stops after 70.5 h, and pushes results every 20 min; extra arguments go to
`porygon.train` (e.g. `sbatch scripts/slurm_train.sbatch --async-envs`). Fetch them anywhere with
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
~0.25 nats), so `distill` sharpens them (`--temperature`). On the cluster: `EXPERT_MODEL=<zip>`
(and optionally `EXPERT_GAMES`, `EXPERT_MS`) in `slurm.env`, then `sbatch scripts/slurm_expert.sbatch`;
9000 battles (~1M decisions) take ~12 h on 30 cores. Search scores positions with poke-engine's own
evaluation, so gains won't compound over rounds the way AlphaZero's do; that would need the
network's value inside the search.

## Evaluate / play
```bash
python -m porygon.evaluate MODEL --port 8000 --n-battles 500 [--anchor OLD_MODEL]
python -m porygon.tournament MODEL_A MODEL_B ... --n-battles 100   # .zip checkpoints or pool .pt
python -m porygon.search MODEL --opponent {policy,heuristic,max_power,search} --n-battles 100
python -m porygon.play MODEL --username USER --ladder 50 --search                # search player on the ladder
python scripts/vs_foulplay.py MODEL --player search --opponents fpa0 fpa1 ...   # setup in its docstring
sbatch scripts/slurm_foulplay.sbatch MODEL 100 4   # the same on a cluster node: 4 Foul Play instances x 100 games
PS_PASSWORD=... PLAY_ARGS=--search scripts/ladder_watchdog.sh USER MODEL TOTAL_GAMES 4   # from a home connection
```

## Roadmap
1. **Re-run the search player vs Foul Play** with the 400M network (last measured 49% with 91M).
2. **Throughput decay**: lockstep steps/s fell within a job (1.4k → ~770 over 20 h) while CPU per
   step and memory stayed flat, i.e. growing waits on the slowest battle. Check whether
   `--async-envs` (first used from 278M) removes it; if not, restart jobs every ~12 h.
3. **Ladder the search player** from a home connection for a real rating (the thesis bot: 1693
   Elo / 79.5% GXE over 200 games): `play.py --search`, on the dedicated account `porygon-agent`.
   Its first searches found no match for 40+ min while gen4 games were running; find out why.
4. **Search fidelity**: poke-engine's gen4 build doesn't model taunt/encore/confusion durations;
   Substitute HP is assumed ¼; occasional `NonFinite` engine panics (sample skipped). Tune time
   per move further (1 s beat 0.3 s; more may help).
5. **Expert iteration** (`porygon.expert`): one round from 300M (9,000 battles, 640k decisions)
   did not help: the distilled raw network at best tied its source (50.5% with a strong KL to it),
   and search with it as prior tied search with the source. A 300 ms search scored by poke-engine's
   evaluation is no better a teacher than PPO; revisit once search uses the network's value.
6. **Throughput**: async collection runs at ~4k steps/s but a rollout cycle is ~21 s, ~10 of them
   the PPO update. Measure `--overlap-update` on a job (untested on the cluster: the shared disk was
   too slow for an A/B run), watching for GIL contention between the collector thread and the
   update; then try more battles per job and make the best combination the default.
7. **gen9randombattle**: encoder, action space (tera) and vocab already support it; start from the
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
