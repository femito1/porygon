# LA2-gen4, stage 1: lookahead2 on R1b-F-cd@640M vs the same raw net, gen4 (fastenv, 2026-11-24)
**Status: BINDING.** Main (the orchestrator) signed off the design and the rule on 2026-11-24. They are copied
verbatim below. This file was sealed after the prep chain (port checks, gen4 leaf parity, the gen4 δ\* study) and
before any slot game. A byte-identical copy is at `$C/council/la2-gen4/PREREG.txt` (r--), with its md5 in
`PREREG.md5` (`C=/vol/bitbucket/fr1226/porygon-ws-fastenv`). Records and deviations after arming go in
`$C/council/la2-gen4/NOTES.txt`, never here. Origin: council session 1, `docs/council/sessions/2026-11-24/PORTFOLIO.md`
P3 ("if GO").

## Context (P3, wsp, `/vol/bitbucket/fr1226/porygon-ws-wsp/council/p3-gen4/VERDICT.txt`: BUILD-LA2-GEN4)
At a FIXED 200k iterations per state, the raw gen4 net R1b-F-cd@640M scored 59.85 [56.90, 62.80] vs Foul Play gen4.
The incumbent gen4 search bot scored 59.50 [56.69, 62.31]. Paired Δ +0.35 [−3.57, +4.27] (PREREG a8e0d1ff, seed
4031124).

## Main's design and rule (signed off 2026-11-24; verbatim)

## The item: LA2-gen4, stage 1 = port + checks + the first adoption test
1. **Port lookahead2 to gen4:** `v3leaf` is gen9-hardcoded (`fresh_opp` Pokémon construction and tensor allocation; members cited `v3leaf.py:15,52,139`). Cover gen4 Hidden Power types, gen4 screen / weather durations, and anything else gen4 needs. gen9 behaviour must stay byte-identical: run the existing gen9 leaf tests.
2. **Leaf parity, gen4:** 300 battles, the ns-10 method (LeafEncoder vs the reference encoder on the genuine next states). Pass at the gen9 standard. Any failure: fix it or STOP and report.
3. **δ\* for gen4:** build a gen4 study set analogous to the gen9 1,403-position K32 set: positions from gen4 self-play with R1b-F-cd, one per game, judged by K 32. Apply the same Bet 1 precision rule to pick δ*. Record the set, its md5 and δ* in the PREREG before any slot games.
4. **Kill test (fixed-n, on the Showdown slot):** arms = lookahead2(R1b-F-cd, δ*, K 8, T 0.5) vs raw R1b-F-cd (T 0.5), gen4, the fastenv slot harness, **1,000 duplicate pairs**, a new pre-registered seed.
   - **Decision rule (verbatim): GO iff the pair-mean lower 95% > 50.0%; else STOP (the gen4 bot stays the incumbent search bot; record the result).**
   - Claim / release the slot via your LEDGER-OUTBOX.
5. **Files** in `$C/council/la2-gen4/`: PREREG.txt (r--), STATUS, VERDICT.txt (line 1 = GO | STOP, then the pair-mean with its 95% CI and the override rate), FAILED on any failure. The check-in watches these paths. PREREG = `docs/netsearch/LA2-GEN4-PREREG.md` in your worktree, with its md5.
6. Not in this item: the follow-up (LA2-gen4 vs the incumbent on fresh identical Foul Play @200k pairs, adopt iff the paired lower 95% > 0) is main's decision after the verdict.

## The net
`$C/nets/g4r1bf-cd-640.04M.pt` = netv2 `runs/g4r1bf-cd/final/00640042548.pt` (R1b-F's cooldown branch, step
640,042,548), copied 2026-11-24 21:16Z, r--. md5 `90bf853d643ade36ad76f6ebe0dbc61f`, sha256
`27453ed5b2911d0ff8b8dd9f58ab24d25cf18aa757316651ad624c7cf5dff3aa`, equal for both files. It loads through the frozen
netv2 tree `$C/frozen/netv2-fba02fb-20261110` (gen 4; 5,702,969 parameters). That tree's `porygon/v2` is byte-identical
to netv2's cluster `src/porygon/v2` (encoder, spec, model, player, evaluate, tables), which P3's arm (ii) played with.

## 1. The port (ws/fastenv 2badbfc; code frozen in `$C/council/la2-gen4/code`)
- **`porygon/v3leaf.py`:**
  - `fresh_opp(species, level, gen)` builds the poke-env mon, the battle stub and the arrays in the battle's
    generation (it was gen 9 for any battle). An opponent mon that comes in therefore gets its own gen's random-battle
    library posterior, so gen4's typed Hidden Powers are among its candidates.
  - `is_status_move` reads PORYGON_GEN's move data (it was gen 9).
  - New side conditions: the age of a screen set during the lookahead turn is (the engine's initial duration − turns
    left). poke-engine (genx, every generation build) sets 5 for Reflect / Light Screen / Safeguard / Mist / Lucky
    Chant and **4 for Tailwind**, with no Light Clay. So gen4 uses [5, 5, 5, 5, 5, 4, 5]. gen9 keeps its table
    (Tailwind 3) for byte-identical gen9 leaves; there a new Tailwind reads age 0 instead of 1/8 (rare, noted).
  - Weather needs no gen4 change. Ability weather is permanent in gen4: poke-engine stores −1 turns left there, and
    `search.weather_turns` returns −1. The leaf then adds the turns passed to the root's age (on since the root) or
    reads `turns` (new). Move weather lasts 5 turns in both generations; there are no Heat / Damp Rocks in the engine.
  - Everything else is shared. gen4 has no terastallization: the gen4 engine's `terastallized` is always False and
    gen4 actions stop at 9, so the tera columns stay 0. It also has no terrain and no Aurora Veil.
- **`porygon/lookahead2.py`:** the player runs gen4 or gen9 with PORYGON_GEN = the net's gen (it asserted gen9).
- **Scripts:**
  - `lookahead2_study.py bench` reads the generation from `--format`.
  - `v3_gates.py p1` gets `--format`, `--records-max` and `--no-roots`. It reads `all_decisions` when a record has
    them.
  - `exit_la2.py gen` gets `--all-decisions`.
  - `netsearch_slot.sh` logs and pre-reads the gen4 engine (gloria's `.venv` poke_engine) for GEN 4.
  - For GEN 9 none of these changes anything.
  - New: `la2_gen4.py`, `la2_gen4_prep.sbatch`, `la2_gen4.sbatch`, `tests/test_la2_gen4.py` (28 tests).
- **gen9 byte-identity (prep job 1554791, all PASS):**
  - **ns-10's P1 rerun with the new code** (gen9, its records `$D/datasets/searchlabels/records/gen9randombattle/part-0000[67]`, source 6771931, seed 7311, G9p-cd@104.71M, 300 pairs from 100 records × 2 determinizations): PASS with the 11-12 numbers exactly. A agreement 1.00000 (71,824 elements), A mean |ΔP| 1.90·10⁻⁴; B agreement 0.9877, B mean |ΔP| 0.0209.
  - **P1 rows, new vs old code** (the benchmark bot's frozen code `$C/runs/la2-2030/code`, 8093ece): `p1-rows.jsonl` byte-identical, 300 rows, md5 bf6cec4f1fea9d3799fab7384ad9517f for both.
  - **lookahead2's leaves, new vs old `v3leaf`** (`la2_gen4.py ident`): on the first 300 gen9 study positions (`$C/netsearch/study-20261109/s1-sl-AB.jsonl`) × 2 worlds, `lookahead2.expand_world` was run once with each module. 600 expansions and 143,039 leaves: rows, cols, cells and every observation array are equal (dtype, shape, bytes). Differing expansions 0; the determinism control (the new module twice, 10 positions) 0; no engine-failed world, no replay failure.
  - The laptop tests pass: lookahead2, deepen, exit_la2, la2_2030, lookahead2_adopt, p2_exit_soft, netsearch,
    netsearch_slot (103) and test_la2_gen4 (28). Full suite: 175 passed, 3 skipped. The 4 errors are test_selfplay's
    setup, which needs a built pokemon-showdown that the worktree lacks; they are unrelated.

## 2. Leaf parity, gen4 (the ns-10 method; prep job 1554791)
- **Battles:** 300 games of R1b-F-cd@640M self-play at T 0.5 (fastenv, gen4randombattle, `ps-sets-pinned` = the
  benchmark server's October sets, gen4 sets.json md5 77a82cf6), seed 4241124. One record (side) per battle, chosen by
  `la2_gen4.py split`. These are the first 300 games in sha1 order, disjoint from the study set's games.
  `parity-records.jsonl.gz` md5 dd208ae1872f63ec4c506eb39fb3c015.
- **Method:** V3-GATES-PREREG P1, unchanged.
  - Per battle, up to 3 consecutive decision pairs (t, t+1) of the record's player, seeded (7311).
  - 2 determinizations (`search.convert`, rng seeds 0 and 1).
  - Test A: root b(t+1), state convert(b(t+1)). Test B: root b(t), state convert(b(t+1)).
  - Both compare against the real encoder on b(t+1): element agreement over D, and |ΔP(win)| of the net (CPU, fp32).
- **Standard (gen9's):** A agreement ≥ 99.0%, A mean |ΔP| ≤ 0.01, B mean |ΔP| ≤ 0.03. Void if more than 10% of the
  300 battles fail to replay.
- **Result:** **PASS** (`$C/council/la2-gen4/prep/parity-g4/p1.json`). 300 of 300 battles replayed (0 failed), 900 pairs × 2 determinizations.
  - Test A: element agreement **0.99974** (208,990 elements); mean |ΔP| **6.3·10⁻⁴** (q50 9·10⁻⁵, q90 5.1·10⁻⁴, q99 2.2·10⁻³). Disagreements: action_mask 30, move_num.available 12, move_num.pp_frac 4, field_num.n_switches 4, field_num.n_moves 4.
  - Test B: mean |ΔP| **0.0186** (q50 0.0010, q90 0.051, q99 0.227); element agreement 0.9891. Revealed between t and t+1 (n 642): |ΔP| 0.0458, agreement 0.975. Not revealed: |ΔP| 0.0036, agreement 0.997. Disagreements: move_num.pp_frac 1,365, hp 427, field_num.weather_elapsed 280, mon_num.status_counter 74, mon_num.first_turn 48, action_mask 30, trapped 18 + 18, others ≤ 12.
  - gen9's P1 for comparison: A 1.000 / 1.9·10⁻⁴, B 0.0209.

## 3. δ\* for gen4 (Bet 1's precision rule, unchanged; prep job 1554791)
- **Study set** (`$C/council/la2-gen4/prep/set/`), the gen4 analogue of the gen9 1,403-position K32 set:
  - 1,403 positions from R1b-F-cd@640M self-play at T 0.5. It is the same generation run as the parity battles
    (1,805 finished games, seed 4241124); ties, turn-cap games and crashes are dropped.
  - **One position per game:** games are taken in sha1("4241124:<game>") order after the 300 parity games. Per game
    one side is drawn (sha1("4241124:<game>:side") % 2), and that side's decision is drawn uniformly among its
    decisions with ≥ 2 legal actions (exit_la2's draw).
  - **md5s:** `study-rows.jsonl` **10c452c3ba5989bade1f14c8030ea81e** (the set: record id, side, decision index), `study-records.jsonl.gz`
    8de7a8f50ae2a62743116547fce30c7a (their streams). Sides p1 723 / p2 680; median turn 15.
- **Bench:** Bet 1's arguments. `lookahead2_study.py bench --format gen4randombattle --worlds 8 --retest --ref 32
  --workers 10 --concurrency 3 --seed 11`: per position A = K 8, a fresh retest B = K 8, a fresh judge R = K 32,
  R1b-F-cd for the root, the leaves and R.
- **Rule (Bet 1's, verbatim from LA2-BET1-PREREG):** δ\* is the smallest grid margin ≥ 0 from which on (every larger
  finite grid margin too) at least 90% of the overrides are confirmed by R. An override is weighted by the net's T 0.5
  draw probability; it is confirmed when R's Q of A's pick exceeds R's Q of the net's draw. The grid runs from 0 to
  0.15. Pooled over the 1,403 positions (`analyze-bench`, `margin_rule.delta`), with ≥ 98% of them usable.
- **Result: δ\* = 0.0025.** DELTA* = 0.0025 for R1b-F-cd@640M (gen4) by Bet 1's δ rule (pooled 1403 study positions): 0.0025 override 0.1902 precision_R 0.9133 judgeR_gain_vs_raw +0.00453 usable 1403 of 1403 (prep job 1554791, `council/la2-gen4/PREP-RESULT.txt`, 2026-11-24 21:47Z).
  - Pooled curve (`$C/council/la2-gen4/prep/bench/bench-all.txt`):

      margin | override | judge B: gain vs raw, vs pure, precision | judge R: gain vs raw, vs pure, precision  (P(win) per decision)
      -inf | 0.456 | +0.00487, +0.00000, 0.718 | +0.00466, +0.00000, 0.732
      0.0 | 0.375 | +0.00479, -0.00008, 0.788 | +0.00464, -0.00002, 0.809
      0.0025 | 0.190 | +0.00465, -0.00023, 0.885 | +0.00453, -0.00013, 0.913
      0.005 | 0.168 | +0.00460, -0.00027, 0.898 | +0.00447, -0.00019, 0.918
      0.0075 | 0.144 | +0.00443, -0.00044, 0.900 | +0.00435, -0.00032, 0.927
      0.01 | 0.125 | +0.00432, -0.00055, 0.912 | +0.00421, -0.00045, 0.942
      0.0125 | 0.113 | +0.00422, -0.00065, 0.917 | +0.00411, -0.00056, 0.948
      0.015 | 0.103 | +0.00407, -0.00080, 0.923 | +0.00398, -0.00068, 0.959
      0.0175 | 0.095 | +0.00398, -0.00089, 0.924 | +0.00391, -0.00075, 0.963
      0.02 | 0.082 | +0.00381, -0.00106, 0.945 | +0.00372, -0.00094, 0.979
      0.025 | 0.068 | +0.00354, -0.00133, 0.954 | +0.00344, -0.00122, 0.975
      0.03 | 0.055 | +0.00330, -0.00157, 0.974 | +0.00320, -0.00146, 0.995
      0.035 | 0.044 | +0.00299, -0.00188, 0.968 | +0.00294, -0.00173, 0.996
      0.04 | 0.036 | +0.00271, -0.00216, 0.980 | +0.00267, -0.00199, 0.995
      0.05 | 0.028 | +0.00238, -0.00249, 1.000 | +0.00238, -0.00228, 1.000
      0.06 | 0.021 | +0.00201, -0.00286, 1.000 | +0.00203, -0.00264, 1.000
      0.075 | 0.012 | +0.00149, -0.00338, 1.000 | +0.00146, -0.00321, 1.000
      0.1 | 0.007 | +0.00103, -0.00384, 1.000 | +0.00101, -0.00366, 1.000
      0.15 | 0.002 | +0.00039, -0.00448, 1.000 | +0.00040, -0.00427, 1.000

  - Also measured: run A (worlds 8): leaves median 924 [q10 40, q90 1822, max 4410], wall s median 0.62 q90 1.01 max 4.60; worker cpu s median 0.58 q90 1.09; score 0.339 s (forward 0.183), solve 0.082 s; rows 6.7 x cols 6.4; expl median 0.0000; failed worlds 0, engine errors 0; run R (worlds 32): leaves median 3705 [q10 160, q90 7295, max 17564], wall s median 1.21 q90 1.84 max 3.62; worker cpu s median 2.37 q90 4.35; score 0.515 s (forward 0.340), solve 0.118 s; rows 6.7 x cols 6.4; expl median 0.0000; failed worlds 0, engine errors 0; lookahead pick = net argmax 0.559; A pick = B pick 0.837; selected-gap inflation vs B: +0.0009 (sd 0.0096, 8064 pairs); selected-gap inflation vs R: +0.0013 (sd 0.0080, 8064 pairs); D over the net's T0.5 draws: share 0 0.625; q50 0.0000, q75 0.0001, q90 0.0163, q95 0.0325, q99 0.0873
  - Replayed vs self-play T-1 probabilities at the 1,403 positions (informational): {"positions": 1403, "compared_actions": 9490, "max_abs_dp": 3.876383841361353e-06, "mean_abs_dp": 3.0400488795359675e-07, "legal_mismatch": 0}
- **Machine line (the slot chain checks it):** `LA2G4 DELTA*=0.0025 STUDY_ROWS_MD5=10c452c3ba5989bade1f14c8030ea81e NET_SHA256=27453ed5b2911d0ff8b8dd9f58ab24d25cf18aa757316651ad624c7cf5dff3aa`

## 4. The kill test (one slot job, fixed n)
- **A** = `lookahead:$C/nets/g4r1bf-cd-640.04M.pt,margin=0.0025,temperature=0.5,worlds=8,workers=11,leaf_device=cuda`.
  This is lookahead2(R1b-F-cd, δ\*, K 8, T 0.5): the root net on the CPU, the leaf net on the job's GPU, 11 lookahead
  workers, min_prob 0, 1,000 solver iterations (the module defaults, as every gen9 lookahead2 run).
- **B** = `$C/nets/g4r1bf-cd-640.04M.pt,temperature=0.5`: the raw net at T 0.5 (netv2's NetPlayer through the harness,
  as Bet 1's B).
- **Harness:** the fastenv slot harness (`netsearch_slot.sh`, Bet 1's plan with GEN 4):
  - gen4randombattle on fastenv's duplicate-battle server `$C/.showdown-eval` (smogon 7b02eb3 + the duplicate patch,
    code sha256 95a0414e26cfd185, gen4 sets.json md5 77a82cf6);
  - one client holding both players, 3 concurrent battles, the frozen netv2 tree fba02fb for both;
  - the engine is gloria's `.venv` poke_engine (the gen4 build the incumbent gen4 bot uses);
  - code: `$C/council/la2-gen4/code` (ws/fastenv 2badbfc), the same tree that computed δ\*.
- **n: 1,000 duplicate pairs = 2,000 games, fixed n.** 10 blocks of 100 pairs; block j plays on a fresh server seeded
  `4241125b<j>`. **Seed tag 4241125** is new: no plan, manifest, PREREG or NOTES under `porygon-ws-*`, results.csv
  row or ledger line has it (checked 11-24 21:05Z).
- **Reading** (`la2_gen4.py verdict`, mechanical): a pair scores (s0 + s1) / 2 (a win 1, a tie 0.5). A's pair-mean
  interval over the pairs is mean ± 1.96 sd / √n; its lower bound `lo` is compared unrounded and strictly.
  - **GO iff `lo` > 50.0%; else STOP** (the gen4 bot stays the incumbent search bot: search + fitted eval + v1@500M
    prior; the result is recorded).
  - VERDICT.txt: line 1 `GO` | `STOP`; line 2 the pair-mean with its 95% CI; line 3 the override rate; then the
    details.
- **Void (the slot harness's rule, scaled to 1,000 pairs):** HARNESS_FAIL, > 2% lookahead fallbacks, or < 900
  complete pairs ⇒ line 1 `VOID: …`, no reading, FAILED + RELEASE. Replaying with the same seeds after the fix needs
  main.
- **Smoke (in the slot, mechanical, before any main game):** 2 pairs A vs B (tag `4241125smoke`) = 4 games. PASS
  needs: no Traceback, no HARNESS_FAIL, one arm; A searched and leaves > 0, gate rows > 0, 0 failed worlds, fallbacks
  ≤ max(2, 2%). Not PASS ⇒ FAILED + RELEASE, nothing more played.
- **Operating characteristics** (Bet 1's pair sd 0.314: half-width ≈ 1.95 points at 1,000 pairs):
  - P(GO) ≈ 2.5% at a true 50%, ≈ 17% at 51%, ≈ 52% at 52%, ≈ 86% at 53%, ≈ 98% at 54%.
  - gen9 Bet 1 measured lookahead2 vs its raw net at 53.2% [50.6, 55.7] (median-unbiased).

## Slot job
- `cd $C && sbatch --hold council/la2-gen4/code/scripts/fastenv/la2_gen4.sbatch`, then the CLAIM line, then
  `scontrol release`. Job `ws-fastenv-la2g4`, `--partition=training`, 24 CPUs, 1 GPU, 48 GB, **`--time=20:00:00`**,
  no dependency (the slot has been free since P3's RELEASE at 20:13Z).
- **Time basis:**
  - Bet 1 (gen9, the same arm structure, CONC 3, 10 workers on a 12-CPU helper) played 1,200 games in 3.73 h, so
    2,000 games ≈ 6.2 h.
  - gen4 prep bench, this net (run A = K 8, 10 workers, concurrency 3): leaves median 924 [q10 40, q90 1822, max 4410], wall s median 0.62 q90 1.01 max 4.60; worker cpu s median 0.58 q90 1.09; score 0.339 s (forward 0.183), solve 0. — the same per-decision cost as gen9's lookahead2 (0.5–0.9 s), so the gen9 basis applies.
  - The limit, 20 h, is ≈ 3× the expectation and covers a 1–2 h cold-node start.
- **Stages,** each logged to `STATUS`:
  - prereg: PREREG.txt r-- with md5 = PREREG.md5;
  - checks: DELTA r-- and equal to the machine line above, the study rows md5, the net md5 + sha256, PREP-RESULT's
    PASS lines, ≥ 24 CPUs, a GPU, the frozen tree;
  - smoke; main; verdict; one RELEASE line in `$C/LEDGER-OUTBOX.md`.
  - A failed stage (or TERM) appends to `FAILED` and writes the RELEASE line; nothing later runs. A stage dir that
    already holds a chain is refused.
- **Outputs** in `$C/council/la2-gen4/`: `STATUS`, `FAILED`, `SMOKE.txt`, `smoke-run/`, `main-run/`, `VERDICT.txt`
  (r--), `RESULT`. From the prep: `DELTA` (r--), `PREP-RESULT.txt` (r--), `prep/`.

## Caveats
- **Judge:** the δ rule's judge R shares A's value function (Bet 1's caveat). R can only remove overrides that are
  world-sampling noise; the games decide whether the lookahead or the policy is right.
- **Engine fidelity, gen4:** poke-engine's gen4 build lasts Tailwind 4 turns (PS gen4: 3) and has no Light Clay /
  weather rocks. The leaves are one turn deep, and the conversion uses PS's remaining turns at the root, so only a new
  side condition's age is affected (handled above).
- **What GO means:** lookahead2 beats its own raw net in gen4. It does not mean it beats the incumbent gen4 bot. That
  is the follow-up on fresh Foul Play @200k pairs, main's decision (item 6).
