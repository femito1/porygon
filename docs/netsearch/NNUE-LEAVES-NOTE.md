# Net-valued leaves at depth ("NNUE for Pokémon"): design note (fastenv, 2026-11-15; DRAFT, to propose only if Bet 1 is GO)
Status: a proposal, not approved. It answers the question V3 left open: how to get network values at many search leaves
when GPU inference caps a Python or Rust tree at ~2k leaves per decision (V3 I1: about 7.5k leaves/s on the slot T4).

## Idea
Distil G9q's value head into a small value network that runs on the CPU inside poke-engine's Rust MCTS, in place of
`evaluate()`. The search keeps its speed class (≥ 10⁴–10⁵ leaves per decision instead of ~10³) and gains a leaf score
that knows what the network knows.
- **Throughput.** With the stock eval, the benchmark search does ≈ 266k iterations per world per 1000 ms on one dove
  core (study median), i.e. a few µs per leaf. A student costing ~10 µs per leaf keeps ≈ 70-90k leaves/s per core.
  At the hybrid's 4 × 1000 ms that is ≈ 300k net-valued leaves per decision, about 150× the GPU-bound V3 ceiling
  (~2k) and about 175× lookahead2's one-ply median (1.7k).
- Bet 1 tests whether one ply of net-valued leaves helps.
- Step 2 showed the net predicts outcomes better than `evaluate()`: human games, AUC 0.752 vs 0.718, calibrated
  log-loss 0.579 vs 0.611. The one-turn change carries more outcome information too: log-loss gain 0.015 vs 0.009.

## Student network (full information, as the engine sees a determinized world)
- **Inputs.** Built from the poke-engine `State` only, so the Rust and Python feature extractors can be tested for
  exact parity.
  - Per mon (12): species id, types, item id, ability id, 4 move ids, level, HP fraction, status (+ counters), stats.
  - The active's boosts and volatiles.
  - Per side: hazards and screens. Field: weather and terrain (+ turns), trick room, the turn number.
- **Architecture.** Small embeddings (8–16 dims per id, tables exported from G9q's id embeddings as an init). A shared
  per-mon MLP (~64 wide) feeds sum/max pooling per side, plus the two actives' vectors and the field. Then a 2-layer
  MLP (~128) outputs a logit.
  - ~50–200k parameters, f32.
  - Target cost ≤ 10 µs per leaf on one dove core (plain matmuls, no batching).
  - NNUE-style incremental updates are possible later: most leaf states differ from their parent in a few mons.
- **Targets.**
  - The teacher's P(win) on the corresponding partial-information observation of the side to move; this gives the
    expectation over hidden info that a determinized leaf should carry.
  - Blended with the final outcome (e.g. 0.5 / 0.5).
  - Loss: BCE on logits.

## Data
- Positions from G9q self-play on fastenv (in-process Showdown; ~590 transitions/s per CPU).
  - Per decision: the poke-engine state of the true full-information position, built from fastenv's battle state by
    a converter shared with `search.convert`'s state builder; the teacher's value from that side's v2 observation; the
    final result.
  - 10–50M positions, i.e. a few CPU-days on the data host / helper.
- Held-out games for the gates below; the training sets never include them.

## Gates (each pre-registered; stop at the first failure)
1. **Offline.** On held-out positions, the student's AUC and log-loss lie within 0.01 of the teacher's and beat
   `evaluate()` by the Step 2 margins (the same `lookahead2_study.py outcome` metrics).
2. **Parity and speed.** The Rust forward equals the PyTorch forward to 1e-5 on 10,000 states. The Rust search with the
   student reaches ≥ 30% of stock `evaluate()`'s iterations per second.
3. **Engine strength.** Pure engine search (MODEL none, 4 × 1000 ms) with the student vs with the stock eval, duplicate
   pairs on Showdown, sequential (wsp design). This is the direct "better leaves" test.
4. **Integration.** The margin-rule hybrid with the student-eval search vs the benchmark hybrid. Fixed-n slot run.

## Engineering
- A `poke-engine` patch adds `set_value_net(weights_path)` and the student forward in `evaluate()`, behind the existing
  tunable-eval switch (`pe.set_eval`). It is built by `scripts/build_engine.sh` inside a job.
- Feature spec in one Rust + one Python file with a shared test fixture, the same scheme as fastenv's v2-encoder
  parity.
- Risks:
  - Determinized leaves leak hidden info into the student's inputs. The student is trained on full-information
    states and sees a determinized world, as `evaluate()` does.
  - Speed below gate 2: smaller student, incremental updates, int8.
  - Train/serve drift from the November sets: train on the server's sets.

## Cost
About 1–2 weeks of stream time:
- data export and training: days of CPU;
- the Rust patch, parity and speed: a few days;
- gates 3–4: one helper run plus one slot run.
