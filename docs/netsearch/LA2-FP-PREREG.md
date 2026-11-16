# Bet 1 follow-up FP: lookahead2 on G9q-cd@800M vs Foul Play gen9, paired with the raw net: pre-registration (fastenv, 2026-11-16; FINAL, binding and read-only from the ledger claim on)
Main's decision (2026-11-16, Bet 1 GO, item A.2): run lookahead2 on G9q-cd@800M (our best net) vs Foul Play gen9,
paired with the raw net on the same team pairs, fixed n, 500 pairs per arm, Stage 3b harness. Main approved the slot
claim (after netv2's conf800, job 1173380). Bet 1 (`LA2-BET1-PREREG.md`, md5 97c9b4e9…) returned GO: lookahead2(0.01,
K 8) on G9q-cd@605M vs the same raw net, median-unbiased 53.17% [50.59, 55.73] at look 6 (600 pairs).
Cluster copy `$C/runs/la2-fp/PREREG.txt` (r--, byte-identical), md5 in `PREREG.md5`. Records and deviations after
the claim go in `runs/la2-fp/NOTES.txt`, never here.

## Questions (fixed n, headline numbers)
- **Q1 (headline).** lookahead2's win rate vs Foul Play gen9, with its 95% interval from duplicate pairs.
- **Q2 (does Bet 1's gain carry over to another opponent?).** Paired difference, arm 1 − arm 2, on the same team pairs
  (`netsearch_slot.py paired RUN 1 2`).
  - **CARRIES OVER** iff the 95% CI's lower bound is > 0.
  - **REVERSED** iff its upper bound is < 0.
  - **NOT SHOWN** otherwise.
- Q2 is the decision of record. Q1 is reported next to the benchmark hybrid's Stage 3b 56.2 ± 2.8 and netv2's raw
  G9q-cd@800 confirmation (conf800); those come from other team pairs or harnesses, so the comparison is informational.

## Net and δ*
- `$C/nets/g9q-cd-805.03M.pt` = netv2 `runs/g9q-cd800/final/00805029873.pt`, sha256
  `e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f` (md5 d0c5b6cf…). Frozen netv2 tree fba02fb.
- **δ\*** = the number in `$C/runs/la2-delta800/DELTA`, written by the δ chain of `LA2-DELTA800-PREREG.md` (Bet 1's
  δ rule, unchanged, applied to this net). The chain reads it at its start; it waits up to 3 h for it, and if the file
  is missing or the δ chain FAILED, it writes FAILED and plays nothing.

## Arms (both vs `foulplay:ms=1000,parallelism=1`, the Stage 3b opponent)
1. `lookahead:$C/nets/g9q-cd-805.03M.pt,margin=δ*,temperature=0.5,worlds=8,workers=11,leaf_device=cuda`: Bet 1's A at
   the hybrid's 11 workers. The three Foul Play instances' players share one lookahead pool and one copy of each net
   (`vs_foulplay.py` spec mode; first used in games here: the smoke tests it).
2. `$C/nets/g9q-cd-805.03M.pt,temperature=0.5`: the raw net at T 0.5.
- October server and library (as every Stage 3 / re-freeze / Bet 1 run). G9q trains on the November sets (±1 level on
  4 gen9 species): the recorded train/serve deviation, the same for both arms.

## Harness (Stage 3b's, `NETSEARCH-3B-HARNESS.md`)
- `netsearch_slot.sh` with OPP=foulplay. Foul Play 8f541f80f91f with the login patch, venv `.venv-gen9`, gloria's
  `.engine-gen9`. 3 instances per block, one battle each; fresh instances and a fresh server per block.
- 5 rounds × 2 arms × 100 pairs, rotated. Block j is seeded `la2fp20261116b<j>`, so both arms play the same team pairs.
- One 24-CPU `--partition=training` slot job with a GPU, `--dependency=afterany:1173380`.
- **Chain** `sbatch scripts/fastenv/lookahead2_slot.sbatch fp` (code frozen into `runs/la2-fp/code` at submission):
  1. checks: PREREG md5, net sha256, 24 CPUs, a GPU, δ\* from `DELTA`;
  2. smoke (tag `la2fpsmoke20261116`): 2 pairs per arm (4 games per arm over the 3 instances, so the shared pool is
     used concurrently), read mechanically: 4 games per arm, no Traceback, no HARNESS_FAIL; arm 1 searched and leaves
     > 0, gate rows > 0, 0 failed worlds, fallbacks ≤ max(2, 2%). FAIL stops the chain (FAILED + outbox RELEASE);
  3. main run, `paired`, `VERDICT.txt` (r--, line 1 = CARRIES OVER / REVERSED / NOT SHOWN / VOID), `RESULT`, the ledger
     RELEASE line in `$C/LEDGER-OUTBOX.md`.
- **Void.** HARNESS_FAIL, > 2% fallbacks on our side, or < 450 pairs on an arm ⇒ replay with the same seeds after the
  fix.
- **Expected time.** ≈ 8 h per 1000 games for arm 1 (Stage 3b's hybrid rate) and ≈ 5-6 h for arm 2 ⇒ ≈ 13-14 h;
  limit 30 h.

## Reported
Both arms ± CI, won-both / split / lost-both, the paired difference, arm 1's override rate, fallbacks, hours.

## Caveats
- **The battle timer is on vs Foul Play** (`vs_foulplay.py`). lookahead2 needed 0.85 s per decision at the median and
  15.5 s at most in Bet 1, far below the timer.
- General strength only: nothing in lookahead2 models Foul Play. The opponent is the equilibrium of the one-turn game.
- Two things differ from Bet 1: the net (cd@800 for cd@605) and its δ\* (re-derived by the same rule).
