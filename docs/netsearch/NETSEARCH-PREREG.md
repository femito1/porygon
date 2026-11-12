# Network + search: Stage 1 result, Stage 2 and Stage 3 pre-registration (fastenv, registered 2026-11-10)
Design and rationale: `NETSEARCH-DESIGN.md` (§2 the rule, §3 the margin, §6 the test plan). This file fixes what the
design left open, before any Stage 2 or Stage 3 game. Orchestrator approval: 2026-11-09 22:45 (Stage 1 and Stage 2 if a
G9p net reaches ≥ 45% vs our gen9 search on netv2's standard harness); the gate moved to the cooldown deployment net
on 2026-11-10 00:1x (netv2's PREREG-CD: G9p-cd@104.71M beat G9p@105.01M 52.05% [50.42, 53.65], 3400 games).

## The network (frozen for every stage)
- **G9p-cd@104.71M**: netv2's `runs/g9p-cd/final/00104714038.pt` (= `runs/g9p-cd/vs9-104.7M.pt`, a trainer
  checkpoint; `porygon.v2.model.load` reads its state_dict as netv2's harness does), copied to
  `$C/nets/g9p-cd-104.71M.pt`, sha256 35cd1007bb90bdf939cedf877f5fcdd89ef640628f77007b3b6638d5add4ad1c (= the gate
  run's manifest).
- **Gate: PASSED.** 845624, raw T 0.5 vs our gen9 search on the standard harness (seed 738164, 21 workers, 5
  concurrent): **50.5% ± 4.5%** (200 pairs, 400 games; won both 43, split 116, lost both 41; search fallbacks 0 of
  14,254).
- For context (not part of the gate): the pre-registered 1000-game confirmation 864111 (new seed 738201, same
  harness) gave 43.4% ± 2.8% (500 pairs). The gate rests on 845624, as registered; the orchestrator kept Stage 2
  approved (2026-11-10 09:5x).
- Failed checks before it: G9p@100M 43.5% ± 4.4% (839408); G9p@50M 42.5% ± 4.3%.
- netv2's code (`porygon.v2`: encoder, model loader, NetPlayer, the evaluate harness) is frozen at registration:
  `$C/frozen/netv2-fba02fb-20261110/src/porygon` (netv2 cluster copy, COMMIT fba02fbd…; package sha256
  8a1f7f824ea2e026 = the live copy at 10:05 and the smoke runs' copy). Plans name it (`NETV2_DIR`).

## Stage 1 (offline, done 2026-11-10 10:10; `$C/netsearch/stage1-g9pcd-v2/`)
- `NET=nets/g9p-cd-104.71M.pt OUT=netsearch/stage1-g9pcd-v2 bash scripts/fastenv/netsearch_stage1.sh` (helper 943274).
- It re-scored the study's positions and searches with G9p-cd: S1 501, S2 250 (its stale PUCT searches dropped), S2c 251
  and S3 401 rows (all rows).
- **Pooled R curve** (S2 + S2c, 501 positions), as override rate at T 0.5 / hybrid − pure search (R) ± SE, Q per
  decision:

  | δ | 0 | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.1 |
  |---|---|---|---|---|---|---|---|
  | override | 47.0% | 33.3% | 24.2% | 18.0% | 14.3% | 11.4% | 4.6% |
  | hybrid − search | +0.0004 ± 0.0003 | +0.0004 ± 0.0006 | −0.0004 ± 0.0007 | −0.0014 ± 0.0007 | −0.0024 ± 0.0007 | −0.0035 ± 0.0008 | −0.0077 ± 0.0010 |

- The raw net is −0.0172 ± 0.0025 vs the search (R); 50M gave −0.0165.
- The search's pick equals the net's argmax on 50.9% of positions; A = R on 82.6%.
- Selected-gap inflation: +0.0099 ± 0.0022.
- Value head: BCE 0.541, Brier 0.186.
- **Arms by the §3.1 rule** (`netsearch_arms.py`, linear interpolation):
  - **δ₁ = 0.0212** (L = 0.0005, override ≈ 23.4%);
  - **δ₂ = δ_free = 0.0261** (L = 0.001, ≈ 20.5%);
  - **δ₃ = 0.0358** (L = 0.002, ≈ 15.9%).
  - The provisional 50M values were 0.0232 / 0.0285 / 0.0369.
- **Bug found and fixed before registration.** The first Stage 1 run (02:29, `stage1-g9pcd/`, rc 1) stopped on the
  re-score's mask check for the human records S2c and S3. `rescore` matched rows by record id alone, and the human
  records of both sides of a battle share that id, so one side's decision indices were replayed in the other side's
  stream. Its pooled curve therefore had 251 positions, and those margins were never used. Fix (ws/fastenv 9ac4785):
  match by (record id, side). The re-run is the one above, all rows matched.

## Stage 2 (tuning, slot; an internal go/no-go, no claim)
**Players.**
- H(δ) = `netsearch:NET,margin=δ,temperature=0.5,ms=1000,samples=4,workers=11`: the net's T 0.5 draw
  unless our search's pick beats it by more than δ in root Q (§2). Its 11 engine processes are its own.
- Opponent: `search:none,ms=1000,samples=4,eval_config=stock,prior_weight=0,workers=11`, our pure gen9 search as
  benchmarked (MODEL none, stock eval, 4 × 1000 ms), with its own 11 engine processes.

**Arms.** H(0.0212), H(0.0261), H(0.0358) (arms 1, 2, 3), the Stage 1 values above.

**Harness** (`scripts/fastenv/netsearch_slot.sbatch` with `scripts/fastenv/plans/stage2.env` from `stage2.env.in`).
- One 24-CPU `--partition=training` job (the benchmark slot, claimed in the ledger). One harness process at a time
  (netv2's `porygon.v2.evaluate` via `netsearch_h2h.py`), 3 concurrent battles.
- fastenv's copy of the duplicate-battle benchmark server: `$C/.showdown-eval`, smogon 7b02eb3 + the duplicate patch,
  code sha256 95a0414e26cfd185 = netv2's `.showdown-eval` (gloria's differs only in a chat-plugin state file).
- The run copies its code (ours and netv2's `src/porygon`) into the run directory at the start.
- Blocks of 50 duplicate pairs: 4 rounds × 3 arms; round j plays the arms in the order rotated by j − 1 (round 1:
  1, 2, 3; round 2: 2, 3, 1; ...).
- Block j of every arm runs on a fresh server seeded `20261110b<j>`, so all arms play the same 200 team pairs.
  Tag 20261110 has never been used, and Stage 3 does not use it.

**Looks and futility.**
- A look after each of an arm's blocks: at 50, 100 and 150 pairs, over that arm's blocks so far, in block order
  (nested).
- Futility: the arm stops if its upper 95% bound (pair means) is below 50%. 200 pairs ends an arm.
- Stage 2 makes no claim, so futility-only looks need no α spending.

**Selection and go/no-go** (design §6).
- δ\* = the arm with the best point estimate at the end, among the arms not stopped for futility.
- Arms within 1 point of the best go to the larger margin (closer to the net, fewer overrides).
- Go to Stage 3 iff δ\*'s point estimate ≥ 51.5%.
- No-go: report the arms; V1 or V2 arms may follow with a new pre-registration on the same tuning seeds.

**Void.** A harness failure (the run stops itself: HARNESS_FAIL) or > 2% search fallbacks on either side voids the
run. It is replayed with the same seeds after the fix.

**Reported:** each arm's win rate ± 95% CI, pairs, override rate, fallback rates (both sides), hours; the looks
(`looks.jsonl`).

**Deviations from the design text, and why.**
1. **Interleaved blocks instead of 3 parallel shards.** Each arm then plays on exactly the harness the design fixes
   for all games (11 engine processes per side, 3 concurrent battles, one harness process). Rotation balances the arms
   in time, and each arm's looks are nested. Three parallel shards on 24 CPUs would have needed 1 concurrent battle
   and ≈ 4 engine processes per side per shard.
2. **11 engine processes per side, really.** `porygon.search.engine_pool` keeps one pool per worker count per
   process. So data's teacher test, whose harness the design copies, ran both sides on ONE shared 11-process pool.
   The hybrid now has its own pool (`own_pool`, default on). Each search still gets its full 1000 ms per
   determinization. CPU load is up to 22 busy engine processes on 24 CPUs, close to netv2's vs-search harness (21),
   instead of ≤ 11.
3. **Duration:** ≈ 6 h expected (≈ 45–55 s per game per battle slot, 3 slots), not 7.5 h; the cap stays 1200 games.
4. **Stage 1's re-score bug** (above): fixed and re-run before any game; the procedure and the rule are unchanged.
5. **netv2's code is frozen** at registration (`NETV2_DIR`), not copied from its live tree at job start.

## Stage 3 (headline, fixed n; registered now, before Stage 2's first game)
**Test seed tag 20261111** (blocks seeded `20261111b<j>`), drawn on 2026-11-09 with the design and unused since. No
Stage 3 game is read by any tuning tool. Same network, same opponent spec, same harness, same server copy as Stage 2.

**3a + 3c, one job** (`plans/stage3ac.env`, written when δ\* is known; nothing else in it may change).
- Arm 1 = H(δ\*). Arm 2 = the raw snapshot at T 0.5 (`NET,temperature=0.5`, netv2's NetPlayer), vs
  the same opponent with its 11 engine processes and 3 concurrent battles.
- 5 rounds × 2 arms × 100 pairs: 500 pairs (1000 games) per arm, rotated like Stage 2, on the same team pairs.
- Fixed n: no looks, no futility.
- **Q1 (3a): success iff the lower 95% bound of H(δ\*) vs pure search > 50%.** Report the estimate ± 95% CI and the
  override rate.
- **Q3:** the paired difference 3a − 3c per team pair (block j, pair i), mean ± 1.96 sd / √500.

**3b (only if 3a succeeds).** H(δ\*) vs Foul Play gen9, and pure search vs Foul Play, 500 pairs each on the same team
pairs (tag 20261111). Foul Play config as the gen9 benchmark.
- **Q2: success iff the 95% CI of the paired difference (H − pure search, per team pair) excludes 0.**
- The Foul Play harness (vs_foulplay with `netsearch:` specs) gets its own short pre-registration of the harness
  details only (no new choices) before 3b's first game.

**Decisions** as the design §6: 3a and 3b succeed → the hybrid becomes the benchmark bot and V3 is next; 3a only →
report the vs-search gain, no Foul Play-specific tuning; 3a fails → no headline claim, override analysis, V2 / V6 at
Stage 2.

## Results (fixed n, as registered; written 2026-11-12 by fastenv)
**Stage 2** (943438, 6.3 h, not void): arm 1 δ 0.0212 53.25% ± 4.56%, arm 2 δ 0.0261 48.75% ± 4.47%, arm 3 δ 0.0358
51.5% ± 4.44% (200 pairs each) ⇒ δ\* = 0.0212, GO (≥ 51.5%).

**Stage 3a + 3c** (985078, 9.6 h, not void; `$C/runs/netsearch-stage3ac-20261110-171632-985078/result.txt`).
- 3a H(0.0212) vs pure search: **55.0% ± 3.0% [52.0, 58.0]**, 500 pairs; overrides 23.4% ⇒ **Q1 SUCCESS**.
- 3c raw T 0.5 vs pure search: 48.05% ± 2.96%, 500 pairs.
- Q3 paired 3a − 3c: **+7.0 ± 4.2 points** (500 common pairs).

**Stage 3b** (1026613, 2026-11-11 03:02–19:37Z, 16.6 h; `$C/runs/netsearch-stage3b-20261111-025822-1026613/result.txt`).
- Not void: no HARNESS_FAIL; 10/10 blocks with 100 full pairs (0 incomplete); our fallbacks 6/34,116 = 0.02% (arm 1)
  and 0.01% (arm 2); 0 battle timeouts, 0 no-accepts, 0 re-challenges in every block; Foul Play restarts 1 in 10
  blocks (a2b1 instance x0, a Foul Play ZeroDivisionError; reported, not voiding).
- Arm 1 H(0.0212) vs Foul Play gen9: **56.2% ± 2.8% [53.4, 59.0]**, 500 pairs (won both 137, split 288, lost both 75);
  overrides 22.0% of 34,116 searched decisions; 7.98 h.
- Arm 2 pure gen9 search vs Foul Play gen9: **51.6% ± 2.8% [48.8, 54.4]**, 500 pairs (110 / 296 / 94); 8.46 h.
- **Q2:** `netsearch_slot.py paired RUN 1 2`: H − pure search **+4.6 ± 3.9 points [+0.7, +8.5]** on 500 common team
  pairs (unmatched 0) ⇒ the CI excludes 0 ⇒ **Q2 SUCCESS**.
- Context only (not a test): H vs Foul Play 56.2% next to H vs pure search 55.0% (3a).

**Decision (design §6, 3a and 3b succeed):** "network + search beats our search and Foul Play by more than our
search does". The hybrid H(0.0212) with G9p-cd@104.71M becomes the gen9 benchmark bot; V3 (net value at the leaves,
Python first) is next, with its own pre-registration (draft: `NEXT-PROPOSAL.md`).
