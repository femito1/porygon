# Bet 2 gate 1 (offline): can a small CPU student on poke-engine State features predict as well as G9q's value? Pre-registration (fastenv, 2026-11-16; binding before any training, read-only)
Main's decision (2026-11-16, item B): gate 1 of `NNUE-LEAVES-NOTE.md` is approved; gates 2–4 are NOT. This file fixes
the data, the student, the metrics and the PASS rule before any data is generated or any student is trained. Records
and deviations go in `$C/nnue/gate1/NOTES.txt`, never here. Cluster copy `$C/nnue/gate1/PREREG.txt` (r--), md5 in
`PREREG.md5`.

## Question
On held-out positions, does a ~50–200k-parameter f32 student whose inputs are built from the poke-engine `State`
alone predict game outcomes within 0.01 of the teacher (G9q-cd@800M's value head) and better than poke-engine's stock
`evaluate()` by at least Step 2's margins?

## Teacher
`$C/nets/g9q-cd-805.03M.pt` (sha256 e93ac3f8…b11f; netv2 `runs/g9q-cd800/final/00805029873.pt`), loaded through the
frozen tree `$C/frozen/netv2-fba02fb-20261110`. P(win) = sigmoid(win_logit) (sparse ±1 rewards, no V + Φ).

## Data (G9q-cd@800M self-play on fastenv)
- **Games.** In-process Showdown (fastenv host, gen9randombattle, the server's set library), both sides the teacher at
  T 1.0 (its training policy; diversity for leaf-like states), seeds fixed by worker, no duplicate pairs. Target
  ≥ 10M decision positions (≈ 0.25–0.3M games); the generator stops at 10.5M.
- **Per decision** (both sides, every decision row with ≥ 1 legal action, forced switches included):
  1. the poke-engine `State` of the TRUE position from that side's point of view (side one = the side to move), built
     by `porygon.nnue.convert_full` from the sim's full battle state. It shares `search.convert`'s builders
     (`make_mon`, side / field construction, engine move ids);
  2. the teacher's P(win) from that side's v2 observation (the same forward that picks the move);
  3. the game's final result for that side (1 / 0; ties and games cut at the turn cap are dropped).
- **Converter parity (before data generation).** On ≥ 2,000 decisions from ≥ 50 fastenv games replayed through
  poke-env (netsearch_study's replayer), `convert_full` equals `search.convert(battle, rng, truth=…, oracle_all=True)`
  field by field on everything both know exactly: all of side one, side two's species / items / abilities / moves /
  stats / status / boosts / active index, hazards, screens, weather, terrain, trick room. Mismatch rate > 0.5% of
  decisions on any field ⇒ fix before generating; the final rates go in NOTES.
- **Split by game** (sha1 of the game id): 90% train, 2% validation (early stopping, the blend weight is fixed),
  8% held-out. Held-out games are never used for fitting anything except the 2-fold calibration below.

## Student
- **Inputs** from the `State` only (one Python feature extractor, to be mirrored in Rust at gate 2): per mon (12):
  species, item, ability, 4 move ids, tera type (ids, embeddings 8–16 dims), level, HP fraction, status (+ sleep /
  toxic counters), the 5 stats / 100, types, terastallized, fainted, active; the actives' boosts and volatiles; per side
  hazards, screens, side conditions; field weather / terrain (+ turns), trick room.
- **Architecture.** A shared per-mon MLP (~64 wide) → sum and max pooling per side, plus the two actives' vectors and
  the field vector → a 2-layer MLP (~128) → one logit. 50k–200k parameters, f32. Embeddings may be initialised from
  G9q's id embeddings.
- **Target.** y = 0.5 · teacher P(win) + 0.5 · outcome; loss BCE on the logit. AdamW, early stopping on validation
  BCE. One fit; up to 2 additional fits only for crashes / divergence (not for gate tuning), recorded in NOTES.

## Metrics (`lookahead2_study.py`'s outcome metrics, unchanged)
AUC; calibrated log-loss = the 2-fold, game-cross-fitted affine logit recalibration of `analyze_outcome`, applied
the same way to the student, the teacher and `evaluate()` (the latter on eval / 100). Outcome = the game's result.
- **Set S (self-play held-out):** all held-out positions (cap 200k, a fixed seeded subsample). Student and
  `evaluate()` score the true-position State (one State, no determinization); the teacher its own P(win).
- **Set H (human, Step 2):** the same 3,000 positions of Step 2 (`outcome-human`: records part-0070[0-3], 1,500
  pairs, per-record 2, seed 11, K 4 determinizations of `search.convert`). The teacher is re-scored with cd@800 on
  them; `evaluate()` and the student are each averaged over the same K 4 determinizations (for the student the mean
  of P(win)), exactly as `evaluate()` was in Step 2.

## PASS rule (all four, point estimates; otherwise FAIL)
On BOTH set S and set H:
1. AUC(student) ≥ AUC(teacher) − 0.01 and LL_cal(student) ≤ LL_cal(teacher) + 0.01;
2. AUC(student) − AUC(evaluate) ≥ 0.015 and LL_cal(evaluate) − LL_cal(student) ≥ 0.017.
Reported with game-clustered bootstrap 95% intervals of the differences (1000 resamples), by-turn tables and decile
calibration (informational). A PASS lets fastenv propose gate 2 (Rust parity and speed); it decides nothing else.

## Also reported (information, not part of the rule)
- The student's per-leaf CPU cost: a PyTorch-free numpy forward of one State on one core (dove), median over
  10,000 held-out states: (a) forward only from precomputed features; (b) feature extraction from the `State`
  object + forward. Plus the parameter count.
- Data generation throughput, positions per game, the converter's parity rates.

## Chain (one cluster-side chain on fastenv's helper job, no agent)
`scripts/fastenv/nnue_gate1.sh`: checks (PREREG md5, teacher sha256) → parity (above; FAIL stops) → generate →
fit → set-H re-score → report. Stage files `$C/nnue/gate1/{STATUS,FAILED,VERDICT.txt (r--, line 1 PASS / FAIL:
…),RESULT}`.
