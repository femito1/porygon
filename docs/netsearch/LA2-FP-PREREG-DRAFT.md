# Bet 1 follow-up FP: lookahead2 vs Foul Play gen9, paired with the raw net: pre-registration DRAFT (fastenv, 2026-11-15; not binding)
Write this only if Bet 1 (`LA2-BET1-PREREG.md`, md5 97c9b4e9…) returns GO. It runs only after main signs off its md5
and the slot is claimed in the ledger. It is drafted before any Bet 1 result is known; on sign-off it is copied
unchanged to `$C/runs/la2-fp/PREREG.txt` (r--), with its md5 in `PREREG.md5`.

## Questions (fixed n, headline numbers)
- **Q1 (headline).** lookahead2's win rate vs Foul Play gen9, with its 95% interval from duplicate pairs.
- **Q2 (does Bet 1's gain carry over to another opponent?).** Paired difference, arm 1 − arm 2, on the same team pairs.
  - **CARRIES OVER** iff the 95% CI's lower bound is > 0.
  - **REVERSED** iff its upper bound is < 0.
  - **NOT SHOWN** otherwise.
- Q2 is the decision of record. Q1 is reported next to the benchmark hybrid's Stage 3b 56.2 ± 2.8. That number comes
  from other team pairs (tag 20261111), so the comparison is informational.

## Arms (both vs `foulplay:ms=1000,parallelism=1`, the Stage 3b opponent)
1. `lookahead:$C/nets/g9q-cd-605.04M.pt,margin=0.01,temperature=0.5,worlds=8,workers=11,leaf_device=cuda`: Bet 1's A
   at the hybrid's 11 workers. The three Foul Play instances' players share one lookahead pool and one copy of each
   net (`vs_foulplay.py` spec mode).
2. `$C/nets/g9q-cd-605.04M.pt,temperature=0.5`: the raw net (Bet 1's B).
- Net sha256 b7957be4…001b. Frozen netv2 tree fba02fb. October server and library.

## Harness (Stage 3b's, `NETSEARCH-3B-HARNESS.md`)
- `netsearch_slot.sh` with OPP=foulplay. Foul Play 8f541f80f91f with the login patch, venv `.venv-gen9`, gloria's
  `.engine-gen9`. 3 instances per block, one battle each; fresh instances and a fresh server per block.
- 5 rounds × 2 arms × 100 pairs, rotated. Block j is seeded `la2fp20261116b<j>`, so both arms play the same team pairs.
- One 24-CPU `--partition=training` slot job with a T4.
- Chain: `sbatch scripts/fastenv/lookahead2_slot.sbatch fp` (smoke of 1 pair per arm read mechanically, then main,
  `netsearch_slot.py paired RUN 1 2`, `VERDICT.txt`, and the ledger RELEASE line).
- **Void.** HARNESS_FAIL, > 2% fallbacks on our side, or < 450 pairs on an arm ⇒ replay with the same seeds after the
  fix.
- **Expected time.** ≈ 8 h per 1000 games for arm 1 (Stage 3b's hybrid rate) and ≈ 5-6 h for arm 2 ⇒ ≈ 13-14 h.

## Caveats
- **The battle timer is on vs Foul Play** (`vs_foulplay.py`). lookahead2 needs 0.7-1.1 s per decision at the median,
  and 7-8 s at most in the bench, far below the timer.
- The rule is general strength only: nothing in lookahead2 models Foul Play. The opponent is the equilibrium of the
  one-turn game.
