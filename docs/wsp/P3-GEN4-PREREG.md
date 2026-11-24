P3 PREREG: gen4 re-baseline at the fixed Foul Play yardstick (council session 1, portfolio item P3)
wsp, written 2026-11-24 ~14:10Z, before any game of this design. The orchestrator (main) signed off the design and the
rule on 2026-11-24 (user "Go ahead" on the council's top 3, ~13:45Z); its text is copied VERBATIM in section A.
Section B is wsp's implementation inside that design: it changes none of the arms, bot configurations, n, Foul Play's
budget or the rule. Cluster copy: /vol/bitbucket/fr1226/porygon-ws-wsp/council/p3-gen4/PREREG.txt (r--), same bytes;
code: $W/council/p3-gen4/code/ = ws/wsp scripts/wsp/p3g4/. $W = /vol/bitbucket/fr1226/porygon-ws-wsp,
$C = /vol/bitbucket/fr1226/porygon-ws-netv2, gloria = /vol/bitbucket/fr1226/gloria (read only).

== A. Signed-off design and rule (main's brief, verbatim) ==

**Question:** on identical gen4 Foul Play @200k team pairs, does the raw cooled v2 gen4 net (R1b-F-cd) score within 2 points of the incumbent gen4 bot? The incumbent is fp4-04's search + fitted eval + v1@500M prior, measured 59.19 [56.95, 61.43] but time-based, so not comparable with the 200k yardstick.

1. **Launcher.** Extend the fixed-iteration Foul Play launcher to gen4: the gen4 Foul Play / engine build, poke-engine's `iterations` argument, no Foul Play code copied (GPL). Keep the iteration check: ITERS_OK required, else the run is void.
2. **Arms.** Both play Foul Play gen4 at a FIXED 200,000 iterations per state, 1,000 games each (500 duplicate pairs), on the SAME team pairs (one new pre-registered seed shared by both arms):
   - (i) the incumbent exactly as fp4-04 (its own time budget is part of the bot; only Foul Play's budget is fixed);
   - (ii) raw R1b-F-cd at T 0.5. netv2 is producing this cooldown now. When it is ready, `/vol/bitbucket/fr1226/porygon-ws-netv2/council/p3-gen4/R1BF_CD_READY` holds `<final path> <md5> <step>`. If netv2 writes `/vol/bitbucket/fr1226/porygon-ws-netv2/council/p3-gen4/FAILED` instead, your chain writes FAILED too.
3. **Slot use.** Minimise idle slot time:
   - run arm (i) first (it needs nothing from netv2);
   - start arm (ii) only once READY exists, waiting at most 48 h and then writing FAILED;
   - prefer releasing the slot while waiting if your tooling allows.
   - Claim / release the slot through `$W/LEDGER-OUTBOX.md` (the orchestrator copies the lines into ledger.md).
4. **Analysis.** Paired Δ = (ii) − (i) per team pair, with a 95% CI from duplicate pairs; both absolute scores with 95% CIs (they become the gen4 headlines at fixed 200k); short (< 25 turns) vs long split for (ii), informational.
5. **Decision rule (verbatim): BUILD-LA2-GEN4 iff Δ ≥ −2.0 (point estimate); else STOP-GEN4-NET (stop the gen4 net route for now and record the re-baselined incumbent as the gen4 headline at fixed 200k).**
6. **Files** in `$W/council/p3-gen4/`: PREREG.txt (r--), STATUS, VERDICT.txt (line 1 = BUILD-LA2-GEN4 | STOP-GEN4-NET, then Δ and both scores with CIs), FAILED on any failure (the check-in watches these paths). PREREG in your worktree (e.g. `docs/wsp/P3-GEN4-PREREG.md`) = `$W/council/p3-gen4/PREREG.txt`, with its md5.

== B. Implementation (wsp) ==

Seed. Duplicate-server seed 4031124 (new: in no results.csv row, ledger line or PREREG), shared by both arms. The
server (netv2's .showdown-eval: gen4 sets.json md5 77a82cf6, duplicate patch room-battle.js 8d4b4ceb; gloria's and
fastenv's copies are identical) derives pair p's two teams and its battle seed from sha256(seed:p:...) alone, and
Foul Play (names ^fp) gets team A in game 0 of every pair. So both arms play the same team pairs and battle seeds,
whatever their concurrency.

Foul Play. /vol/bitbucket/fr1226/foul-play (8f541f8; GPL-3.0, not copied) in gen4 from its own .venv (stock
poke-engine 0.0.48, gen4 build), --search-time-ms 1000, --search-parallelism 1 (one core per instance), started
through code/fp_fixed_iters.py (md5 5b9ff869 = the fpcal launcher f9c32cdf + gen4 guards: the binding must take an
`iterations` argument, fp.search.main must call that binding, the engine path is logged). Every search of nominal
D ms runs round(200,000 · D / 1000) iterations (2 states × 200,000; early in a game 4 × 100,000) through poke-engine's
own `iterations` argument. Context only: fp4-04's time-based gen4 budget (job 308845's fp0.log, K 4 on dove01) was a
median 216,000 iterations per 1000 ms state (p10 151k, p90 383k) and 118,000 per 500 ms state: 200k ≈ fp4-04's median.

Iteration check and validity. code/fp_iters_check.py (md5 a1ff4886, = fpcal's) runs every 60 s and once after the
games: every logged "Iterations i: V" must be 200,000 (1000 ms states), 100,000 (500 ms) or 1,000 on a single-option
state (poke-engine's quick exit). A mismatch stops the games and writes ITERS_MISMATCH: the arm is void and the chain
writes FAILED (no replay). An arm is VALID iff ITERS_OK lists N-count states, there is no ITERS_MISMATCH and >= 450 of
its 500 pairs are complete. A void arm without a mismatch (infrastructure, e.g. a Foul Play websocket handshake
timeout at start, 0 games) is replayed ONCE with the same seed (the void run kept as <run>-void1); a second void =>
FAILED.

Harness. code/fp4_iters.sbatch (md5 32433427): a copy of the fpcal fixed-iteration harness fp9_iters.sbatch (82a207a6;
for gen4 that is netv2's netv2_foulplay.sbatch 35e7e30b, the v2g4-10 harness, + fixed iterations + the check), run in
the netv2 tree and sourcing its scripts/netv2_common.sh (8f174c94 now), read only, plus a --player search mode for
arm (i). Run dirs $W/council/p3-gen4/runs/foulplay-<arm>-<date>-<job>/ (manifest.txt, battles.jsonl, results.jsonl,
vs.log, result.txt, ITERS_OK, fp0.log.gz).

Arm (i), the incumbent exactly as fp4-04 (gloria jobs 308845-308848, 2026-11-03, dove01/02, 24 CPUs; their vs.log:
SearchPlayer(ms=1000, samples=4, prior_weight=0.25, top_k=None, c_puct=0.0, root_game=None, rm_gamma=0.0,
value_weight=0.0, value_top_k=None, value_min_prob=0.0, engine_python=None, eval_config=runs/evalfit/r1/fe_x2.json,
oracle=None)):
  - gloria's code, run as gloria's slurm_foulplay.sbatch runs it (cwd and PYTHONPATH gloria): src sha256 6a27438d
    (19 files, unchanged since 11-07; search.py 7d52456b, inference.py 2935b640, engine_worker.py f4498aaa),
    scripts/vs_foulplay.py 4e1f0c21, .venv gen4 engine b2c234ec. That is the deployed gen4 bot, with gloria's fixes
    made after 11-03 (e.g. the 11-07 Rest-duration fix). Stage i refuses to start if any of these changed.
  - vs_foulplay.py --player search --ms 1000 --samples 4 --prior-weight 0.25 --eval-config
    gloria/src/porygon/data/eval_r1.json (md5 178ffcb1, the same bytes as fp4-04's fe_x2.json); prior net v1@500M
    gloria/runs/20261025-195826-301040/final_model.zip (sha256 e90aa0ae...68a3; the benchmark default from 10-26 to
    11-03 per the thesis note on the gloria benchmark networks; those runs did not log it); network on cpu,
    --search-timeout 30 (the default).
  - fp4-04's environment: one 24-CPU job on training (= dove01-04), K = 4 Foul Play instances × 1 core, --workers
    24 − 4 − 3 = 17, 4 concurrent battles, the job's own OMP/MKL thread settings (gloria's harness set none). 1,000
    games = 250 per instance. Not kept: per-decision logs (--log-dir), which do not change a move. The logged
    SearchPlayer config is recorded in STATUS.
Arm (ii), raw R1b-F-cd at T 0.5 (sampling): netv2's raw-network player in the netv2 tree (scripts/vs_foulplay.py
c0077268 now; its src as at the run, sha256 in the manifest), K = 20 Foul Play instances × 1 core (the v2g4-10 / F2030
setting), 1,000 games = 50 per instance. Net = <final path> from R1BF_CD_READY; the file's md5 must start with
READY's <md5> (at least 8 hex digits), else FAILED; <step> is recorded.

Launcher check (code/p3_smoke.sbatch, outside the slot, before arm (i)): the same harness at 200k, 4 games per player
mode (K 2, seed 4031125): search with arm (i)'s configuration, policy with a stand-in v2 gen4 net (R1b-F@300M,
$C/runs/g4r1bf/prior-R1bF-300M.pt, md5 c3ec9433). PASS iff for each mode ITERS_OK lists N-count states, there is no
mismatch, the launcher line names Foul Play's .venv poke-engine 0.0.48, >= 2 pairs are complete, the search mode
logged the expected SearchPlayer config, and the two modes played the same teams on their common pairs. Stage i
requires PASS ($W/council/p3-gen4/smoke/RESULT).

Order, slot and waiting (code/p3_chain.sbatch + code/p3_wait.sbatch; job name ws-wsp-p3g4, --partition=training,
24 CPUs; every stage appends to STATUS; any failure writes FAILED and a RELEASE line):
  1. Slot job 1 (stage i; its CLAIM line is written at submission; afterany the smoke): PREREG md5 and r--, code
     md5s, inputs, smoke PASS, then arm (i) (expected ~7.5 h: fp4-04 played 400 games in ~3.0 h at K 4).
  2. Then: netv2's FAILED => FAILED (arm (i)'s score stays in STATUS), RELEASE. READY (and < 22 h of the job used)
     => arm (ii) in the same allocation (~2 h), the reading, RELEASE. Neither => RELEASE the slot, WAIT_START = now,
     and the poller p3_wait.sbatch (1 CPU, one pass every 30 min) until READY (=> a CLAIM line in LEDGER-OUTBOX.md
     and slot job 2 = stage ii, afterany every slot-sized job of ours already queued or running; job 2 RELEASEs at
     its end), netv2's FAILED (=> FAILED) or 48 h after WAIT_START (=> FAILED: no arm (ii)).
  3. touch $W/council/p3-gen4/STOP stops the poller.

Reading. code/p3_read.py (md5 139478fc) -> VERDICT.txt (r--), from each run's battles.jsonl (the server's log; our side
= the non-"fp" player; a win 1, a loss 0, a tie 0.5; a pair is complete when both of its games are logged):
  line 1   BUILD-LA2-GEN4 iff Δ ≥ −2.0 points (point estimate, decided exactly: 100 · Σd ≥ −2.0 · n), else STOP-GEN4-NET;
  line 2   Δ = (ii) − (i) per team pair, over the pairs complete in both arms with the same teams (species of both
           sides), 95% interval mean ± 1.96 sd / √n of the pair differences;
  line 3-4 each arm: the mean of its complete pairs, 95% mean ± 1.96 sd / √n of the pair means: the gen4 headlines at
           a fixed 200,000 iterations per state;
  line 5   (ii) per game, < 25 turns vs ≥ 25 turns (informational; games treated as independent); line 6 the same for
           (i), as context; then the run dirs, the READY line, both ITERS_OK lines and this PREREG's md5.
What follows (main's table in ORCHESTRATOR.md, not automatic): both scores become thesis rows; BUILD-LA2-GEN4 => port
lookahead2 to gen4 (fastenv: v3leaf gen4, leaf parity, δ*); STOP-GEN4-NET => the gen4 bot stays the incumbent. General
strength only: no Foul Play-specific change follows from either outcome.
