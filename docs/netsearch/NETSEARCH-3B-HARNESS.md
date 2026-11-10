# Network + search Stage 3b: the Foul Play harness (fastenv, registered 2026-11-10, before any 3b game)
This file fixes the **harness details only** of Stage 3b. The question, arms, n and decision rule are those of
`NETSEARCH-PREREG.md` §Stage 3, registered before Stage 2's first game; nothing here adds a choice to them. 3b runs
only if 3a succeeds (Q1) and the orchestrator approves the submission.

## What 3b plays (from NETSEARCH-PREREG.md, unchanged)
- Arm 1 = H(δ\*) = `netsearch:$C/nets/g9p-cd-104.71M.pt,margin=0.0212,temperature=0.5,ms=1000,samples=4,workers=11`.
  δ\* = 0.0212 is Stage 2's selection (943438, `result.txt`).
- Arm 2 = our pure gen9 search = `search:none,ms=1000,samples=4,eval_config=stock,prior_weight=0,workers=11`. That is
  Stage 2/3's opponent spec, now playing on our side.
- Both arms play Foul Play gen9 for 500 duplicate pairs each, on the same team pairs as 3a (tag 20261111).
- **Q2:** success iff the 95% CI of the paired difference (arm 1 − arm 2, per team pair) excludes 0.
  Computed by `netsearch_slot.py paired RUN 1 2`: mean ± 1.96 sd / √n over the pairs that both arms completed.

## Harness (plan `scripts/fastenv/plans/stage3b.env`, runner `netsearch_slot.sh` with `OPP=foulplay:ms=1000,parallelism=1`)
**Layout, as Stage 3a:**
- One 24-CPU `--partition=training` slot job (claimed in the ledger), fastenv's server copy `$C/.showdown-eval`.
- 5 rounds × 2 arms × 100 pairs, rotated like Stage 2 (round j starts with arm ((j − 1) mod 2) + 1).
- Fixed n: no looks, no futility.
- Block j of each arm runs on a fresh duplicate server seeded `20261111b<j>`.
  - The patch derives pair i's two teams and its battle seed from (seed, i) alone, and our player is the anchor
    (`PORYGON_DUPLICATE_GROUP='^pa'`). So pair (j, i) has the same teams and battle seed as 3a's pair (j, i), whoever
    plays it.
- Code: copied into the run directory at the start, as for Stages 2 and 3; netv2's frozen tree `netv2-fba02fb-20261110`.

**Foul Play** (GPL-3.0, its own clone, run read-only and never copied into our code):
- `/vol/bitbucket/fr1226/foul-play` at commit 8f541f80f91f, with the local-login patch (v1, the 3 s sleep).
  - Venv `.venv-gen9`, CPython 3.12.16.
  - `PYTHONPATH=gloria/.engine-gen9:gloria/src`: its gen9 poke-engine is gloria's .engine-gen9 build (.so sha256
    7965d8b335b716f7). Both the gen9 Foul Play benchmark and netv2's 943190 launch it this way.
- `--search-time-ms 1000 --search-parallelism 1`, i.e. the benchmark's Foul Play compute (2 states × 1 s on one core).
- **3 instances per block, one battle each** = 3 concurrent battles, as in 3a.
  - They are restarted if they exit (`scripts/fastenv/foulplay_instances.sh`; a process still alive 30 s after
    logging a fatal traceback is killed).
  - Fresh instances run per block (each block has a fresh server).
  - Restarts are counted per block in `fp/restarts.txt`.

**Our side** (`scripts/vs_foulplay.py --player spec`; wsp's harness 29ad936 plus a spec mode):
- One player per Foul Play instance, built from the arm's spec with netsearch_h2h's `load_player` (= 3a's), all on
  **one** 11-process engine pool (the first player's).
  - The three players together are 3a's one player at 3 concurrent battles.
  - That pool is guarded like the Foul Play benchmark's (`--search-timeout 30`: a search running > ms + 30 s is
    killed, and the decision falls back).
- We always send the challenge, after wsp's readiness poll (the server lists Foul Play online and out of its last
  battle room).
  - Re-challenges: up to 3, after 45 s each.
  - Battle timeout: 1800 s, then a forfeit.
  - Foul Play turns on the battle timer.
- `--total 200` battles per block over the 3 instances (a shared budget, so whole pairs).
  - A challenge that is never accepted returns its battle to the budget.
  - The client exits non-zero if fewer than 200 battles were played.
- Scoring: pair means from the server's duplicate log (`battles.jsonl`), as in Stages 2 and 3.
  - Timeouts and Foul Play crashes count as the games the server records.

**Duration.**
- The smoke measured ≈ 110 s per game, about 3.6 s per turn (8 CPUs, 3 engines). The gen9 Foul Play benchmark ran
  ≈ 126 s per game per slot.
- So 3b's 2000 games on 3 slots take ≈ 15–19 h.
- Submitted with `--time=30:00:00` on the sbatch command line (training allows 3 days; the script's 14 h is too short).
- BLOCK_TIMEOUT_MIN stays 240 (a block ≈ 1.5–2 h).

**Void** (as Stage 2/3; Foul Play has no fallback counter):
- The run stops itself on a harness failure (HARNESS_FAIL: a block whose client fails with < 90% of its games), or
  is void if > 2% of our decisions fall back.
- A void run is replayed with the same seeds after the fix.
- Reported but not voiding: Foul Play restarts, battle timeouts and re-challenges per block.

**Reported:**
- Each arm's win rate vs Foul Play ± 95% CI (pairs), arm 1's override rate, our fallback rates, hours.
- Q2's paired difference ± 95% CI.
- Restarts, timeouts and no-accepts.
- For context only (not a test): H vs Foul Play next to H vs pure search (3a).

**Checked before 3b's first game: DONE 2026-11-10, 986503, ws/fastenv b9f3d66, PASSED.**
- 4 games per arm, 0 fallbacks, 0 Foul Play restarts; the report and `paired` work.
- Pair (i, game) had the same teams in both arms; the manifest records Foul Play's commit, venv and engine.

The smoke was (`plans/fpsmoke.env`: an old net g9p-67.5M, tag
`fpsmoke20261110`, 2 instances, 2 pairs per arm; a helper job). It checks:
- the instances start;
- the shared pool (one 3-process engine pool per block, not one per player);
- the budget, the logs and the report.

No game of the smoke uses tag 20261111 or the 3b net.
