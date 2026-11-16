# Bet 1 follow-up H: lookahead2 on G9q-cd@800M vs the gen9 benchmark bot: pre-registration (fastenv, 2026-11-16; FINAL, binding and read-only from the ledger claim on)
Main's decision (2026-11-16, Bet 1 GO, item A.2): lookahead2(G9q-cd@800M, δ*) vs the frozen benchmark hybrid,
500 pairs, adopt iff lower 95% > 50%. Main approved the slot claim (after LA2-FP). Bet 1 (`LA2-BET1-PREREG.md`) was
GO. Cluster copy `$C/runs/la2-h/PREREG.txt` (r--, byte-identical), md5 in `PREREG.md5`. Records and deviations after
the claim go in `runs/la2-h/NOTES.txt`, never here.

## Question (fixed n)
Is lookahead2 on G9q-cd@800M stronger head to head than the frozen gen9 benchmark bot H(δ 0.0212) + G9p-cd@104.71M?
- **ADOPTED** iff the lower 95% bound of the pair means is > 50%. lookahead2 then becomes the gen9 benchmark bot
  (HANDOFF item 3), with this run's spec and hashes.
- **NOT ADOPTED** otherwise: H stays the benchmark bot.
- The same rule as re-freeze #1 R.

## Players
- **A** = `lookahead:$C/nets/g9q-cd-805.03M.pt,margin=δ*,temperature=0.5,worlds=8,workers=11,leaf_device=cuda`;
  net sha256 `e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f` (netv2 `runs/g9q-cd800/final/
  00805029873.pt`).
  - **δ\*** = the number in `$C/runs/la2-delta800/DELTA` (`LA2-DELTA800-PREREG.md`: Bet 1's δ rule on this net), read
    by the chain at its start (missing or the δ chain FAILED ⇒ FAILED, nothing played).
- **OPP** = H = `netsearch:$C/nets/g9p-cd-104.71M.pt,margin=0.0212,temperature=0.5,ms=1000,samples=4,workers=11`
  (net sha256 35cd1007…ad1c). The frozen bot exactly as in Stage 3 and re-freeze #1: its own 11-process engine pool,
  gate log `gate-opp.jsonl`.
- Both load through ONE frozen netv2 tree fba02fb. October server and library.

## Harness
- The re-freeze #1 R harness: `netsearch_slot.sh`, 5 blocks of 100 duplicate pairs, block j seeded
  `la2h20261116b<j>`, 3 concurrent battles.
- One 24-CPU `--partition=training` slot job with a GPU, `--dependency=afterany:<LA2-FP's job>`.
- **Chain** `sbatch scripts/fastenv/lookahead2_slot.sbatch h` (code frozen into `runs/la2-h/code` at submission):
  checks (PREREG md5, both nets' sha256, 24 CPUs, GPU, δ\*), smoke of 2 pairs (tag `la2hsmoke20261116`, mechanical:
  4 games, no Traceback / HARNESS_FAIL, A searched and leaves > 0, gate rows, 0 failed worlds, fallbacks
  ≤ max(2, 2%)), main, `VERDICT.txt` (r--, line 1 ADOPTED / NOT ADOPTED / VOID), `RESULT`, ledger RELEASE line.
- **Void.** HARNESS_FAIL, > 2% fallbacks either side, or < 450 pairs ⇒ replay with the same seeds.
- **Reported.** A ± CI, won-both / split / lost-both, A's and H's override rates, fallbacks, hours; H's mean root
  visits per decision from `gate-opp.jsonl` next to Stage 3's (contention check, informational).
- **Expected time.** ≈ 5 h (hybrid-vs-hybrid rate); limit 30 h.

## Caveats
- **A and H share 24 CPUs.** H's search is time-limited (4 × 1000 ms), so CPU contention can cost it iterations. A's
  lookahead runs in bursts of ≈ 1 s on ≤ 11 workers; H runs 4 searches on its own 11 workers. The peak is 22 worker
  processes + clients + server on 24 CPUs, as in re-freeze #1 R (two hybrids).
- **Two changes at once.** The net is upgraded (G9q-cd@800 vs G9p-cd) and search is replaced by the lookahead. No raw
  arm here (main's brief: one arm, 500 pairs); LA2-FP's paired arm and netv2's conf800 separate the net's share
  against other opponents.
