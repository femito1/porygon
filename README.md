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
- **vs Foul Play** (pmariglia's search bot, 1000 ms/move on one core): our search wins **52.6% ± 2.2%**
  (all runs of the default search with the 400M–500M networks; 49% with 91M; the raw network alone
  16.5%), using 4x its compute (4 parallel 1 s searches). It did not move with the patched engine
  (49.1% ± 4.1% head-to-head vs the stock engine), 500M vs 400M, prior weight 0.5 (54.2%) or 1.0
  (49.6%), or the value lookahead.
- **Oracle** (the search given the opponent's true sets, 500M, 400 battles each): true sets of the
  revealed mons **61.3% ± 4.4%**, the whole true team 65.4% ± 4.1%, so opponent inference was the
  bottleneck (unrevealed species are at chance, as they must be).
- **Opponent inference** (`porygon.inference`, scored offline on 800 logged Foul Play battles with
  `scripts/score_inference.py`): sampling whole sets from Showdown's own generator, conditioned on
  what the log shows and rules out, gets a revealed mon's whole set right 58% of the time (the old
  re-implemented set logic: 35%), a hidden item 83–86% (58–60%), a hidden move 78% (65%); the same live.
  Against Foul Play it scored 50.7% ± 4.2% (500M, 400 games): no measurable gain yet. A third of the way
  from 35% to exact sets is worth ~+3 points if the oracle's +9 scales, which needs ~2000 games to see.
- **Value lookahead** (`porygon.lookahead`, one turn ahead scored by the network's value, 400M): alone it
  beats the raw network only 53.3% ± 5.6%; mixed 0.5 into the search 53.7% ± 5.6% vs the search; as
  a replacement for the engine search 42.7%. The value is not yet sharper than poke-engine's search.
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
| `src/porygon/inference.py` | opponent set posterior: generator set library + evidence from the battle log |
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
| `scripts/build_set_library.js`, `score_inference.py` | build the gen4 set library (`data/*library.json.gz`); score inference offline |
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
PS_PASSWORD=... PLAY_ARGS=--search scripts/ladder_watchdog.sh USER MODEL TOTAL_GAMES 4   # from a home connection
```

## Benchmark
```bash
sbatch -c 24 scripts/slurm_foulplay.sbatch MODEL 100 4 [vs_foulplay args]   # 4 Foul Play instances x 100 games
sbatch scripts/slurm_search.sbatch MODEL --opponent search [porygon.search args]   # head-to-heads
```
Both run a patched Showdown (`scripts/build_eval_server.sh` → `.showdown-eval/`, never the training
server's checkout) that plays duplicate pairs (same seeds, teams swapped) and writes the true teams,
and they log every searched decision (`porygon.decisions`). Output in `runs/{foulplay,search}-<date>-<job>/`:
`battles.jsonl` (true teams, winners), `decisions.jsonl` (probs, shares, action, value, each
determinization's opponent team), `replays/`, `teams/`, and `analysis.txt` from
`scripts/analyze_benchmark.py` (pair-scored win rate, inference accuracy against the true teams,
largest value drops of lost-both pairs). Pairing is worth only ~1.1–1.4x the games, so a 400-game run
is ±4.5%; compare against the pooled 52.6% ± 2.2%.
- `--oracle revealed|all`: search with the true sets of the revealed mons / the whole team (diagnosis).
- `--engine-python .venv-stock/bin/python` (`--opp-engine-python` for the opponent): engine workers
  under another poke-engine build (`STOCK=1 VENV=.venv-stock scripts/build_engine.sh`), for engine A/Bs.
- Locally: `scripts/build_eval_server.sh`, then `PS_DIR=$PWD/.showdown-eval PORYGON_DUPLICATE=<seed>
  PORYGON_DUPLICATE_LOG=<file> PORYGON_DUPLICATE_TEAMS=<dir> scripts/start_servers.sh 1 8001`.

## Roadmap
1. **Opponent inference, what is left**: the remaining item misses are mostly Choice Band vs Life Orb /
   Silk Scarf / Choice Specs, which only damage tells apart: compare the damage each hit did with the
   engine's damage range for every candidate set (`pe.calculate_damage`; crits, screens, boosts and
   weather at the time). Score on the logged benchmarks first (`scripts/score_inference.py`; new
   evidence must stay 100% consistent with the true teams), then a 400-game Foul Play run. Rebuild the
   library (`node scripts/build_set_library.js`) whenever `pokemon-showdown/` is updated.
2. **The network inside the search tree**: priors at every node (PUCT) instead of the 0.25 post-hoc
   mix, later its value at the leaves. This is where Rust pays: poke-engine's MCTS already runs in
   Rust (~430k iterations/s), and Python is only per-decision glue (tens of ms against a fixed 1 s
   budget), so porting that buys nothing; but network evaluations inside the tree must run in Rust
   (e.g. ONNX through `ort` / `tract` in the patched engine, `patches/`) since Python callbacks cap at
   ~1k evaluations/s. The value needs to be sharper first (see Value lookahead).
3. **A public rating**: the [PokéAgent Challenge](https://pokeagentchallenge.com/battling.html)
   runs a live leaderboard for AI agents (bots welcome; Foul Play won its Gen 9 OU final). It has
   no random battles: Gen 1–4 OU, Gen 9 OU and VGC, headline boards Gen 1 OU and Gen 9 OU. Entering
   needs team handling and usage-stat opponent priors instead of random-battle sets, and retraining.
4. **Search fidelity left**: locked moves (Outrage) always last 3 turns (gen4: 2–3), a taunt applied
   after its target moved misses that turn, encore needs the engine's last-move tracking, occasional
   `NonFinite` engine panics (sample skipped). Time per move: 1 s beat 0.3 s; more may help.
5. **Training**: PPO reached 600M (`slurm.env` `TOTAL` = 600M; raise it to continue); 400M → 600M did
   not move the Foul Play result, so resume only once the network weighs more in the search (2).
   Cheaper PPO updates (bf16, larger minibatches) and evals that don't stall collection would help.
   Moving training off Showdown onto a Rust simulator would multiply throughput but risks training
   on mechanics that differ from Showdown's.
6. **gen9randombattle**: encoder, action space (tera) and vocab already support it; start from the
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
