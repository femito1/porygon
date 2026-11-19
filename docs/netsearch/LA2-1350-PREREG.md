# LA2-1350: lookahead2 on G9q-cd@1350M vs the gen9 benchmark bot: pre-registration (fastenv, 2026-11-19; FINAL, binding and read-only from the ledger claim on)
Main's design (2026-11-19, below verbatim): the LA2-H-style adoption test of the new deployment-net candidate
G9q-cd@1350M, gated on netv2's Foul Play confirmation F1350, as one cluster-side slot chain. Written before any data
of this test. Cluster copy `$F/runs/la2-1350/PREREG.txt` (r--, byte-identical), md5 in `PREREG.md5`. Records and
deviations after the claim go in `$F/runs/la2-1350/NOTES.txt`, never here. `$F` = `/vol/bitbucket/fr1226/porygon-ws-fastenv`.

## Main's design (verbatim)
Context: netv2's R1350 (in-process) showed raw G9q-cd@1350 beats cd@805: MUE 52.05% [50.45, 53.62]
(`/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9q-cd1350/READING.txt`). Its Foul Play confirmation F1350 is slot job
1238699 (pending; expected start ≈ 16:33Z, ≈ 1.3 h) and writes `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/conf1350/RESULTS.txt`.
Attempt 1 was a void (Foul Play handshake timeouts), archived in `runs/conf1350-void1`. Net:
`/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9q-cd1350/final/01355036714.pt` (md5 6cf40c0c). Copy it into `$F/nets/`
read-only, and record its sha256. The current benchmark bot is in your HANDOFF: lookahead2 on cd@805, δ 0.005, K 8,
code frozen in `$F/runs/la2-h/code`.

Design (pre-registered by main, 2026-11-19):
1. Gate: the chain plays only if line 2 of netv2's `runs/conf1350/RESULTS.txt` contains "CONFIRMED (lower 95% bound > 50%)". Otherwise (NOT CONFIRMED, FAILED or missing) it writes `$F/runs/la2-1350/WITHDRAWN` with the reason, plays nothing, and writes a RELEASE line.
2. δ for cd@1350 is re-read by your existing precision rule (`lookahead2_delta.sh`, as for la2-delta800) inside the chain. Write it to `$F/runs/la2-1350/DELTA` before any game.
3. Test: lookahead2(cd@1350, δ*, K 8, T 0.5) vs the current benchmark bot lookahead2(cd@805, δ 0.005, K 8, T 0.5). Use your slot harness, LA2-H pattern (`lookahead2_slot.sbatch h`-style; Showdown, duplicate pairs), 1000 games fixed n (500 pairs), new seed 8135444, frozen code copy.
4. Reading (mechanical): lower 95% > 50 ⇒ "ADOPTED (BETTER)"; else lower 95% > 47 ⇒ "ADOPTED (NON-INFERIOR, margin 3 points)"; else "NOT ADOPTED" (the benchmark bot stays lookahead2 on cd@805). Rationale for the PREREG: the raw net's gain is already shown in-process and lookahead2's gain over its own raw net is shown (Bet 1). This test guards against a regression on Showdown; it is not powered for a 2-point superiority claim (≈ 30% power at +2).
5. One slot job on `--partition=training`, 24 CPUs, `--dependency=afterany:1238699`, `--time` sized from LA2-H's measured runtime plus the δ study plus margin; state your basis. It must not need any agent to finish. Each stage writes STATUS. A failed stage writes FAILED. The end writes `$F/runs/la2-1350/VERDICT.txt` (line 1 = the reading) and one RELEASE line in `$F/LEDGER-OUTBOX.md`. Put the CLAIM line in `$F/LEDGER-OUTBOX.md` under its flock when you submit; main copies it into the ledger.
6. Before submitting, run a laptop mock of the gate logic and the reading on fake RESULTS / VERDICT inputs: CONFIRMED, NOT CONFIRMED, FAILED, missing, plus reading values 52.9/50.1/48.0/46.9 lower bounds. Do NOT run games for this outside the slot, unless a ≤ 8-game smoke fits your pattern and leaves the slot untouched.

## Nets
- **Candidate** `$F/nets/g9q-cd-1355.04M.pt` = netv2 `runs/g9q-cd1350/final/01355036714.pt` (G9q's cooldown branch
  from 1350M, ending at 1,355,036,714 steps), copied 2026-11-19 12:52Z, r--; md5 `6cf40c0cd5526b5f4f5b903ec883d66d`,
  sha256 `23593a3138cd17a5a010e42ca9cca1604b4ee98081e64df1c4ee7f49533886cd`, equal for both files. It loads through the
  frozen netv2 tree `$F/frozen/netv2-fba02fb-20261110` (checked 11-19 on the login node: gen 9, 5,809,977 parameters,
  as cd@805). Sparse ±1 rewards as G9q: P(win) = sigmoid(win_logit).
- **Benchmark bot's net** `$F/nets/g9q-cd-805.03M.pt`, sha256 `e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f`.

## Gate (design 1)
- `scripts/fastenv/lookahead2_adopt.py gate`, read once at the chain's start (the job starts when 1238699 has ended):
  PLAY iff line 2 of `/vol/bitbucket/fr1226/porygon-ws-netv2/runs/conf1350/RESULTS.txt` contains "CONFIRMED (lower 95%
  bound > 50%)" and not "NOT CONFIRMED" (netv2's NOT-confirmed wording is "NOT CONFIRMED (lower 95% bound <= 50%)"), and
  line 1 names the net, "md5 6cf40c0c" (the file is about this net).
- Anything else (NOT CONFIRMED, FAILED, missing, no line 2, another net) ⇒ `WITHDRAWN` (the reason and line 2),
  `RESULT`, one RELEASE line; nothing is played, the δ study does not run. No waiting: a further F1350 replay or void
  needs main (re-arming = a new submission on a fresh stage dir).

## δ* (design 2)
- **Bet 1's δ rule, unchanged** (`LA2-BET1-PREREG.md`, md5 97c9b4e9…, section "δ*"), run exactly as
  `LA2-DELTA800-PREREG.md` (md5 77ae985a…) ran it for cd@805: the 1,403 study positions `$F/netsearch/study-20261109`
  (S1 501, S2 250, S2c 251, S3 401), the same record files (data's searchlabels `part-0000[67]`, human
  `part-0027[0123]`; unchanged since 11-09), seed 11, `lookahead2_study.py bench --worlds 8 --retest --ref 32
  --workers 10 --concurrency 3` with the new net for the root, the leaves and judge R.
- δ\* = the smallest grid margin ≥ 0 (grid 0 … 0.15) from which on (every larger finite grid margin too) at least 90%
  of the overrides, weighted by the net's T 0.5 draw probability, are confirmed by judge R (K 32); pooled over the
  1,403 positions (`analyze-bench` on `bench-all.jsonl`, `margin_rule.delta`). Fewer than 98% of the positions usable,
  or no margin meets the rule ⇒ FAILED, nothing played, main decides.
- Run inside the slot job before any game: `lookahead2_delta.sh` with `LA2D_STAGE=$F/runs/la2-1350`, the candidate,
  `LA2D_OUT=$F/lookahead2/study-cd1355-20261119`, `LA2D_PREFIX=DELTA-`, `LA2D_CANCEL=0`. Outputs `DELTA` (r--, one
  number), `DELTA-VERDICT.txt` (r--), `DELTA-RESULT`, the bench files and their sha256 list.
- The candidate plays at δ\*; the benchmark bot keeps its adopted δ 0.005 (cd@805's δ\*, override 19.9%).

## Test (design 3)
- **A** (candidate) = `lookahead:$F/nets/g9q-cd-1355.04M.pt,margin=δ*,temperature=0.5,worlds=8,workers=11,leaf_device=cuda`.
- **OPP** (the benchmark bot) = `lookahead:$F/nets/g9q-cd-805.03M.pt,margin=0.005,temperature=0.5,worlds=8,workers=11,leaf_device=cuda`
  (the HANDOFF's spec = LA2-H's arm A).
- **Harness** = LA2-H's (re-freeze #1 R pattern): `netsearch_slot.sh`, 5 blocks of 100 duplicate pairs = 1000 games,
  fixed n; block j on a fresh duplicate server seeded `8135444b<j>` (seed tag 8135444, unused under `porygon-ws-*`,
  checked 11-19); 3 concurrent battles; one client process holding both players, each with its own 11 lookahead
  workers, both leaf nets on the job's GPU; October server `$F/.showdown-eval` (smogon 7b02eb3 + the duplicate patch,
  code sha256 95a0414e26cfd185); one frozen netv2 tree fba02fb for both players; encoder library in-tree. Gate logs:
  A `gate.jsonl`, OPP `gate-opp.jsonl`.
- **Smoke** first, in the slot job: 2 pairs (tag `8135444smoke`), read mechanically: 4 games, no Traceback, no
  HARNESS_FAIL, one arm; for A and for OPP: searched and leaves > 0, gate rows > 0, 0 failed worlds, fallbacks
  ≤ max(2, 2%). Not PASS ⇒ FAILED, nothing more is played. No games outside the slot.
- **Code** frozen at submission into `$F/runs/la2-1350/code` (git archive of the ws/fastenv commit that adds this file;
  the hash is in `code/COMMIT` and NOTES.txt). The player modules `src/porygon/lookahead2.py`, `netsearch.py` and
  `v3leaf.py` are byte-identical to the benchmark bot's frozen code `$F/runs/la2-h/code` (36a2006); the chain checks
  this before playing. The rest of the diff since 36a2006 does not change this test: new modules (`deepen.py`,
  `nnue/`), `fastenv.py`'s state dumps, `search.py`'s oracle path (used only with true sets), the harness's `deepen:`
  support, `lookahead2_study.py outcome --student`, and `lookahead2_delta.sh`'s `LA2D_*` parameters (defaults = the
  la2-delta800 run).

## Reading (design 4, mechanical: `lookahead2_adopt.py verdict`)
- Statistic: A's pair-mean interval over the 500 pairs (`netsearch_slot.py report`: a pair scores (s0 + s1) / 2, a tie
  ½; mean ± 1.96 sd / √n), lower bound `lo`, compared unrounded with strict >.
- `lo` > 50% ⇒ **ADOPTED (BETTER)**; else `lo` > 47% ⇒ **ADOPTED (NON-INFERIOR, margin 3 points)**; else **NOT
  ADOPTED** (the benchmark bot stays lookahead2 on cd@805, δ 0.005).
- ADOPTED (either) ⇒ the gen9 benchmark bot becomes lookahead2 on G9q-cd@1355.04M at δ\* (this run's spec and hashes);
  its Foul Play headline comes later at the fixed 200k-iteration yardstick (main's plan).
- **Void** (LA2-H's rule): HARNESS_FAIL, > 2% fallbacks on either side, or < 450 pairs ⇒ line 1 `VOID: …`, no reading,
  FAILED + RELEASE; replay with the same seeds after the fix.
- Operating characteristics (informational, normal approximation with LA2-H's half-width 2.82 points at 500 pairs):
  true 47% ⇒ P(ADOPTED) ≈ 2.5%; true 50% ⇒ P(NON-INFERIOR or better) ≈ 55%, P(BETTER) ≈ 2.5%; true 52% ⇒ ≈ 94% / ≈ 28%;
  true 54% ⇒ P(BETTER) ≈ 79%.

## Slot job (design 5)
- `cd $F && sbatch --dependency=afterany:1238699 runs/la2-1350/code/scripts/fastenv/lookahead2_adopt.sbatch`: job
  `ws-fastenv-la2x1350`, `--partition=training`, 24 CPUs, 1 GPU, 48 GB (LA2-H's), `--time=20:00:00`.
- **Time basis.** LA2-H's chain took 5.9 h (its 1000 games 5.78 h, 1.07–1.23 h per 200-game block); the cd@805 δ study
  0.68 h ⇒ ≈ 6.6 h expected. A lookahead2-vs-lookahead2 turn waits for the slower of two lookahead decisions: the mean
  of the maximum of two draws from LA2-H's 35,165 lookahead decision times (mean 1.29 s) is 1.68 s, about LA2-H's turn
  (the hybrid searched ≈ 1 s per decision), so about LA2-H's rate. Worst plausible: half the rate (scale gate B's two
  searchers) plus a 1–2 h cold-node start ≈ 14 h. Limit 20 h ≈ 3× the expectation; `BLOCK_TIMEOUT_MIN` 300 per block.
- **Stages**, each logged to `STATUS`: prereg (PREREG.txt r-- with md5 = PREREG.md5) → gate → checks (both nets'
  sha256, ≥ 24 CPUs, a GPU, the frozen trees, player modules = `runs/la2-h/code`, a fresh bench dir) → delta → smoke →
  main → verdict. A failed stage (or TERM at the time limit) appends to `FAILED` and writes the RELEASE line; nothing
  later runs. A stage dir that already holds a chain is refused (RELEASE line, nothing run).
- **Outputs** in `$F/runs/la2-1350/`: `STATUS`, `WITHDRAWN` | `FAILED`, `DELTA`, `DELTA-VERDICT.txt`, `DELTA-RESULT`,
  `SMOKE.txt`, `smoke-run/`, `main-run/`, `VERDICT.txt` (r--; line 1 = the reading, line 2 its consequence, then the
  numbers: won-both / split / lost-both, both override rates, fallbacks, hours per block), `RESULT` (= line 1); logs
  `delta.log`, `smoke.log`, `main.log`. One RELEASE line in `$F/LEDGER-OUTBOX.md` at the end (done, WITHDRAWN or FAILED).
- **Laptop mock** (design 6, before submission): `tests/test_lookahead2_adopt.py` runs the gate on RESULTS.txt files in
  netv2's exact format (CONFIRMED / NOT CONFIRMED / FAILED / missing, plus no line 2 and another net), the reading at
  lower bounds 52.9 / 50.1 / 48.0 / 46.9% (+ the edges 50.0 and 47.0), the VERDICT on fake and on real report
  structures, the smoke checks, and this chain end to end (`LA2A_MOCK`, refused inside a Slurm job). Plus an
  `NS_DRY=1` dry run of the harness with this plan (synthetic games).

## Caveats
- **Two things change at once:** the net (cd@805 → cd@1350) and its δ\* (re-derived; the same 0.005 if the rule lands
  there). The adopted unit is the pair (net, δ\*), as in LA2-H.
- **Shared machine:** both players share 24 CPUs, one GPU and one client process (up to 22 lookahead workers busy).
  lookahead2's decisions do not depend on time (a fixed K 8 worlds), so contention changes speed, not play. Compare
  within this harness only (HANDOFF harness note).
- **Power:** not powered for superiority at +2 points (≈ 28–30%); the non-inferiority branch is main's answer. Its
  margin (3 points) is about one LA2-H half-width.
- **Judge:** the δ rule's judge R shares A's value function (Bet 1's caveat): δ\* is set by noise only.
