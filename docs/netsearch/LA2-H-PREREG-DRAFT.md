# Bet 1 follow-up H: lookahead2 vs the gen9 benchmark bot: pre-registration DRAFT (fastenv, 2026-11-15; not binding)
Write this only if Bet 1 (`LA2-BET1-PREREG.md`, md5 97c9b4e9…) returns GO. It runs only after main signs off its md5
and the slot is claimed in the ledger. It is drafted before any Bet 1 result is known; on sign-off it is copied
unchanged to `$C/runs/la2-h/PREREG.txt` (r--), with its md5 in `PREREG.md5`.

## Question (fixed n)
Is lookahead2 on G9q-cd@605M stronger head to head than the frozen gen9 benchmark bot H(δ 0.0212) + G9p-cd@104.71M?
- **ADOPTED** iff the lower 95% bound of the pair means is > 50%. lookahead2 then becomes the gen9 benchmark bot
  (HANDOFF item 3), with this run's spec and hashes.
- **NOT ADOPTED** otherwise: H stays the benchmark bot.
- The same rule as re-freeze #1 R.

## Players
- **A** = `lookahead:$C/nets/g9q-cd-605.04M.pt,margin=0.01,temperature=0.5,worlds=8,workers=11,leaf_device=cuda`
  (net sha256 b7957be4…001b).
- **OPP** = H = `netsearch:$C/nets/g9p-cd-104.71M.pt,margin=0.0212,temperature=0.5,ms=1000,samples=4,workers=11`
  (net sha256 35cd1007…ad1c). This is the frozen bot exactly as in Stage 3 and re-freeze #1: its own 11-process engine
  pool and gate log `gate-opp.jsonl`.
- Both load through ONE frozen netv2 tree fba02fb. October server and library.

## Harness
- The re-freeze #1 R harness: `netsearch_slot.sh`, 5 blocks of 100 duplicate pairs, block j seeded
  `la2h20261116b<j>`, 3 concurrent battles.
- One 24-CPU `--partition=training` slot job with a T4.
- Chain: `sbatch scripts/fastenv/lookahead2_slot.sbatch h` (smoke of 2 pairs, main, `VERDICT.txt`, ledger RELEASE).
- **Void.** HARNESS_FAIL, > 2% fallbacks either side, or < 450 pairs ⇒ replay with the same seeds.
- **Reported.** A ± CI, won-both / split / lost-both, A's and H's override rates, fallbacks, leaves and seconds per
  decision, hours.
- **Expected time.** ≈ 5 h (hybrid-vs-hybrid rate).

## Caveats (decide before sign-off)
- **A and H share 24 CPUs.** H's search is time-limited (4 × 1000 ms), so CPU contention can cost it iterations.
  - A's lookahead runs in bursts of ≈ 1 s on ≤ 11 workers. H runs 4 searches on its own 11 workers. The peak is
    22 worker processes + clients + server on 24 CPUs, as in re-freeze #1 R (two hybrids).
  - Record H's mean root visits per decision (`gate-opp.jsonl`) next to Stage 3's.
- **Two changes at once.** The net is upgraded (G9q-cd vs G9p-cd) and search is replaced by the lookahead.
  - Optional arm 2 (main's choice before sign-off; +5 h): the raw G9q-cd@605M vs H on the same team pairs. It
    separates the two changes.
  - The decision of record stays arm 1's rule.
