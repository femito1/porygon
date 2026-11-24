### critique of A1: Give the net the evidence it can't see: an oracle-ceiling test first, then speed/damage evidence and action history as inputs
- verdict: PROMISING-IF-FIXED
- fatal flaws: none (for the feature lever; the proposed zero-shot oracle kill test is fatally confounded as written)
- fixable flaws and the fix:
  1. **Broken JS oracle assumption:** A1 claims `aux` targets in `encoder_v2.js` already hold the true set including Tera and exact stats. In fact, `encoder_v2.js:30,446` leaves `aux_tera = -1` until the opponent terastallizes, and `spec.py:102-115` (`aux_fields`) has no `stats` tensor (must read `opp.team` directly).
  2. **OOD confound in zero-shot kill test:** Under `encoder.py:254`, opponent `tera[6:12]` is nonzero *only* when `terastallized == 1`. Feeding `(terastallized=0, tera!=0)` and turn-1 `item_known=1, n_moves_known=1` into unmodified `cd@2030` tests out-of-distribution corruption, not an information ceiling—and self-play shows zero length gap (`loss-patterns-report.md` §1).
  3. **False Net2Net invariance on part (a):** Zero-initializing new `MON_NUM` columns (`model.py:204-206`) gives `f_θ'(x, e) = f_θ(x)` only for part (b). Part (a) alters existing `item_p`, `move_p`, and `stats`, immediately changing `DamageFeatures` (`damage.py:62-75`). Expanding `MON_NUM_DIM` also breaks `OppStack` signature grouping (`train.py:1129`), `BC-P4` shards, and `v3leaf.py` unless patched together.
  - **Fix:** Implement part (a) inside existing Spec v2.1 shapes first (plus Knock-Off `item_none` and `v3leaf.py` parity), gated by a warm-start fork; add zero-init history columns only in a separate Spec v2.2 branch.
- evidence check: Verified `encoder.py:9,122`, `fp4-07` (`62.12 [58.88, 65.37]`, "+2.9 +- 3.9"), `fp4-08` (`64.44`, "+5.2 +- 3.7"), `fp4-10` ("+8.8"), and `dg-08` (`53.4` vs `64.6`). Caveat: `lookahead2` already uses `inference.py` via `search.convert` (`dg-05`: "hidden item 82-88%... near ceiling") yet still gets `−3.0 ± 7.7` in short games (`la2-len1`), so set blindness is only part of the short-game gap.
- better kill test: Port `inference.py` speed/damage/lock filtering into `encoder.py` + `encoder_v2.js` with unchanged Spec v2.1 shapes (300-battle parity = 100%), run two `30M + 5M cd` forks from `G9q@2030M` on A40 (~12 GPU-h, masking frozen pool obs), and test `spd_h2h` over 3,000 duplicate pairs on Dojo VM (0 slot hours; GO iff lower 95% > 50.5%).
- my estimate of the effect: +1.5 [0.0, +3.0] raw vs Foul Play @200k (+0.8 [−0.2, +2.0] under `lookahead2`)
- rank among all proposals here: 2

### critique of A2: Purify the deployed policy (play cooler than T 0.5)
- verdict: WEAK
- fatal flaws:
  1. **Invalid Foul Play adoption rule (~45–50% false-positive rate):** A2 screens 3 arms without multiple-testing correction, runs the winner on 500 pairs (`seed 8203222`), and adopts if *"paired difference is > 0 and its lower 95% bound is > −1.5"*. Under a dead null (`Δ = 0`, paired SE ≈ 1.2–1.5 pts), a pure coin flip passes this non-inferiority hurdle nearly half the time, violating `BRIEF.md` statistical discipline.
  2. **Destroys simultaneous-move mixed equilibria for minimal noise gain:** `T = 0.5` (`p²`) already crushes a `p = 0.05` exploration tail to `0.27%` (`0.10 → 1.2%`), while `T = 0.1` (`p¹⁰`) collapses genuine 60/40 simultaneous-move equilibria (Sucker Punch, Protect, Encore, switch reads) to `98.3/1.7`, making `a_net` deterministically exploitable.
- fixable flaws and the fix: Drop `T = 0.1`; test only support-thresholding (`drop p < 0.04`, renormalize at `T = 0.5`) so residual `zero-avoid` mass is removed without collapsing 60/40 mixes, and require superiority (`lower 95% > 50%`, Bonferroni-adjusted) before adopting.
- evidence check: Verified `v1-19` (`52.3 [50.9, 53.7]`) and `methods.md:221` (*"cooler gained nothing for v1; v1-19"*). Literature check: `literature-report.md:338-339` confirms Obscuro scored *"70.2% vs its no-purification variant"*, but Obscuro purified a solved subgame Nash policy in sequential FoW chess—not an already `p²`-cooled raw PPO net in a simultaneous-move game.
- better kill test: Single pre-registered arm (`p < 0.04` masked before `T = 0.5` sampling) vs stock `T = 0.5` on `cd@2030`: 8,000 duplicate pairs on Dojo VM (~0.6 h, 0 slot hours). GO only if lower 95% > 50.5% AND a 50M-step RL exploiter (`ns-30` protocol) stays < 52.0%; confirm vs Foul Play @200k requiring paired lower 95% > 0.
- my estimate of the effect: +0.2 [−1.0, +1.0] vs Foul Play @200k
- rank among all proposals here: 8

### critique of A3: Tactical sparring partners in the PPO league
- verdict: WEAK
- fatal flaws:
  1. **Benchmark-contamination hazard (`BRIEF.md` Hard Rule 1):** Foul Play is `pmariglia`'s wrapper around `poke-engine`. Training PPO directly against `poke-engine` MCTS (`search.py`) rewards exploiting `poke-engine`'s stock `evaluate()` weights and decoupled-UCB1 quirks rather than general Pokémon tactics—and Step 1's GO gate evaluates against that *exact same* `search.py` opponent.
  2. **Broken IPC and budget dilemma in `fastenv`:** A3 proposes piping *"the fastenv protocol stream into a Python poke-env Battle"* inside pool workers. Inspection of `src/porygon/fastenv.py:38-45` shows `host.js` never streams Showdown protocol to Python—it sends binary Spec v2 arrays. Worse, by `fpc-04` (`−3.2 pts per 4×` compute), cutting `g9q-23`'s `4,000 ms` search (which already loses `44.5%` to `cd@805`) down to `20 ms` (`200×` less) weakens it by ~12+ points into a ~32% punching bag while still consuming ~3.5 of A40's 8 worker CPUs (`155 decisions/s × 22 ms`), whereas A3's fallback (*"a single true-state world"*) cheats on our hidden team and unused Tera.
- fixable flaws and the fix: Never train against `poke-engine` MCTS or omniscient 1-world search, and never gate GO on win rate vs the training sparring partner. If tactical pressure is needed in training, use our own V2-valued 1-ply lookahead on public beliefs (`C6`) or sharper V2 pool opponents (`B3`).
- evidence check: Verified `train.py:1374-1376`, `dg-08` (`53.4` vs `64.6`, exact short count `341/1000` is printed in `results.csv`—no need to derive 35% from CI width), `ns-30` (`50.47`), `g9q-23` (`55.5 [52.7, 58.3]`), `g9q-16` (`52.5`), `cal-03` (`4.65` pt offset).
- better kill test: Run A3's free Step 0 length-split check on `g9q-23` logs first; if still pursued, test `S` vs `control` strictly in-process (`spd_h2h`, 4,000 pairs, lower 95% > 50.5%) and against `lookahead2`—not against `pure search`.
- my estimate of the effect: +0.0 [−1.5, +1.5] true general strength (+0.5 to +2.0 vs Foul Play confounded by shared `poke-engine` exploits)
- rank among all proposals here: 10

### critique of A4: lookahead2 against the predicted opponent, not the worst-case one
- verdict: PROMISING-IF-FIXED
- fatal flaws: none
- fixable flaws and the fix:
  1. **Misses root-matrix pre-minimization and broken our-switch leaves:** In `lookahead2.py:140,208`, opponent forced switches already take hard `min` inside `M_k` before `solve_bayes`, and our faints/pivots (`needs_switch(s₁)`) are scored as dead-active leaves (`own_act = 0` in `model.py:382`). Blending `y_k^λ` at the root cannot cure short-game KO trades (`la2-len1`) until `C2` fixes `expand_world`.
  2. **Post-hoc `Q_λ` vs saddle-point solve, plus private-state leakage:** `aux_opp` reads the shared trunk (`model.py:235-238`), which attends to our hidden bench/items/moves, and models our `T = 1.0` self-play net rather than searchers. Also, picking `argmax_a Q_λ(a)` post-hoc instead of solving `solve_bayes` on `(1−λ)M_k + λ(M_k q_k)1ᵀ` lets actions outside `supp(x)` exploit `y_k^RM` ties.
  3. **Severely underpowered slot kill test:** Sequential `50 → 54%` with `n_max = 600` pairs (`SE ≈ 1.4` pts, `ns-14`) has >70% futility probability if the true gain is A4's own predicted `+1.0` pt, wasting ~10 Showdown slot hours.
  - **Fix:** Apply on top of `C2`, solve `solve_bayes` on the blended matrix `M̃_k = (1−λ)M_k + λ(M_k q_k)1ᵀ`, and gate offline first.
- evidence check: Verified `la2-len1` (`short −3.0 ± 7.7; long +2.9 ± 4.7`), `ns-20` (`+1.3 [−2.71, 5.31]`), `fpr-307072` (`40.8`), `fpr-307076` (`53.4`), `ns-25` (`51.72`), and `deepen.py:95,327` (`opp_prior` already implemented; `0.5(y + q0)` was used for reach in `deepen`). Formal linearity bound is exact.
- better kill test: Zero-slot-hour offline gate on the 1,403-position `K = 32` judge benchmark (`ns-13`/`ns-15`): sweep `λ ∈ {0.25, 0.5}` at matched override rate and STOP unless `K = 32` root-value regret improves significantly over `λ = 0`; only then spend 1,000 fixed pairs on the slot.
- my estimate of the effect: +0.4 [−0.8, +1.5] vs Foul Play @200k
- rank among all proposals here: 6

### critique of A5: gen4 — move the gen4 bot onto the v2 net plus lookahead2
- verdict: PROMISING-IF-FIXED
- fatal flaws:
  1. **Downgrade-permitting scale-up rule:** A5 proposes adopting `LA2-gen4` over our `fp4-04` search bot if it is *"non-inferior with a 3-point margin"* (`lower 95% > 47%`). That rule adopts a bot up to **3 points weaker** than our `59.19%` Gen 4 headline (`ns-26`/`ns-31` used 3-pt non-inferiority only to refresh checkpoints *within* `LA2` after raw h2h proved superiority, never across architectures).
- fixable flaws and the fix:
  1. **T4 contention risk:** Running a 5M-step cooldown alongside `R1b-F` on the 10-CPU T4 (`BRIEF.md:80`) halves `R1b-F` sps; run the cooldown inline (`campaign-plan-and-rules.md:441`) or on the free A40 in <1 h.
  2. **`v3leaf.py` is Gen 9-hardcoded (`v3leaf.py:15,52,139`):** Porting requires Gen 4 Hidden Power move-type preservation, Gen 4 screen/weather durations, `ns-10` 300-pair parity, `ns-13` `δ*` calibration, and `C2`'s bilateral forced-switch fix before touching the slot.
  3. **Fix adoption rule:** Re-baseline `fp4-04` at fixed 200k first, and adopt `LA2-gen4` iff paired lower 95% > 50.0%.
- evidence check: Verified `fp4-04` (`59.19 [56.95, 61.43]`, time-based), `v2g4-10` (`55.6 [51.5, 59.7]`, 400 g), `v2g4-11` (`51.1`), `v2g4-12` (`51.9`), `v2g4-15` (`55.2 [53.2, 57.2]`, uncooled `@552.68M` vs `@300.03M`). Caveat: in Gen 9 raw net beat stock search (`g9q-23`: `55.5%`), whereas Gen 4 search has `+4.7` pts of fitted leaf eval (`fp4-04` over `fp4-02`), making `fp4-04` a tougher bar. Zero effect on primary Gen 9 target (`BRIEF.md:7`).
- better kill test: Cool `R1b-F@630M` on A40 (~0.9 h), verify `R1b-F-cd@630M` vs `@300M` in-process (`spd_h2h`, 2,000 pairs on Dojo VM), then run paired 1,000-game fixed-200k Foul Play baselines for both `fp4-04` and `R1b-F-cd` (4 slot h) before building `LA2-gen4`.
- my estimate of the effect: 0.0 in Gen 9; +0.0 [−2.0, +2.0] over `fp4-04` in Gen 4 at fixed 200k (~58–61% abs)
- rank among all proposals here: 5

### critique of B1: Give the policy its own Tera type before it chooses whether to Tera
- verdict: STRONG
- fatal flaws: none
- fixable flaws and the fix:
  1. **Shared JS encoder corrupts frozen pool opponents in `FastEnv`:** B1 specifies *"keep auxiliary labels and frozen opponents' encoders fixed."* However, in `fastenv` (`fastenv_v2.py:194-228`), `host.js` runs one shared `encoder_v2.js` for *both* battle slots before Python splits learner (`s % 2 == 0`) from pool opponents (`s % 2 == 1`). Unconditionally populating `tera[0:6]` in `encoder_v2.js` feeds OOD `(terastallized=0, tera!=0)` inputs into frozen `BC-P3`/`cd` pool checkpoints (46% of games). **Fix:** Zero `obs["tera"][:, :6]` and `obs["tera_p"][:, :6]` for un-terastallized mons inside `FastCollector._opp_obs` (`fastenv_v2.py:220`).
  2. **Understates `lookahead2` upside & omits `aux_tera`:** B1 says *"Engine search already receives the type, so I expect less gain on top of lookahead2."* Actually, `lookahead2` keeps `a_net` on ~85% of turns and scores all non-Tera leaves via `v3leaf.LeafRoot` (`E.encode`), where `DamageFeatures.active_tera_vs` (`damage.py:12`) is also zero today—so `lookahead2` gains nearly as much as raw play. Also populate dead `aux_tera` (`encoder_v2.js:30,446`) in the follow-up arm so trunk hidden-Tera representations train too.
  3. **Redundant pre-check:** Code inspection (`encoder.py:213`, `encoder_v2.js:28,267`) already proves pre-Tera `tera` is 0; replace the 100-pair check with 300-battle Python/JS parity (`ns-10`).
- evidence check: Every citation verified: `encoder.py:213`, `encoder_v2.js:28-29`, `damage.py:12-13`, `dg-02` (`90.3%` Tera rate, `loss-patterns-report.md:135`), `g9q-32` (`60.8`), `ns-32` (`61.60`). Direct check of `gen9randombattle_library.json.gz` confirms B1's exact Volcarona Tera Water (`n=1635`) vs Tera Ground (`n=1571`) sets—and shows **368 of 509 Gen 9 species** share identical 4-move sets with different Tera types.
- better kill test: Keep B1's clean 2-arm A40 design (`25M + 5M cd`, ~11 A40 GPU-h at 1.55k sps, 4,000 `spd_h2h` pairs on Dojo VM in 20 min, 0 slot hours), adding the `_opp_obs` mask and 300-battle JS/Python parity check.
- my estimate of the effect: +1.5 [+0.3, +3.0] raw and +1.2 [+0.2, +2.5] with `lookahead2` vs Foul Play @200k
- rank among all proposals here: 1

### critique of B2: Let a simulated move reveal itself to the leaf evaluator
- verdict: WEAK
- fatal flaws:
  1. **`poke-engine` emits neither move-announcement events nor PP decrements:** B2 cites `SIM-PROTOCOL.md` and requires revealing a move *only* when announced (including misses, excluding full paralysis/flinch/faster KO), halting its kill test *"if any test reveals an unannounced move."* However, `lookahead2` calls Rust `pe.generate_instructions(s, ours, theirs)` (`lookahead2.py:194`), which returns low-level state diffs—not protocol lines—and **does not decrement PP** (`V3-GATES-PREREG.md:138`: `pp_frac` had 625 disagreements across 600 Test B leaves). On 0-damage branches (miss, immunity, failed Protect/Sucker Punch vs full paralysis, sleep, flinch, faster faint), Python `LeafEncoder` *cannot tell* whether `theirs` was announced without patching Rust `poke-engine`.
- fixable flaws and the fix:
  1. **Patch Rust `poke-engine`** to set a 2-bit `move_announced` flag per side in `StateInstructions`, and reveal items/abilities at the same time instead of spending 20 slot hours on moves alone.
  2. **Unrecalibrated `δ*` and underpowered slot test:** Fixed 1,000 slot pairs (`12–20` slot h, `SE ≈ 1.2` pts) has <20% power for B2's `+0.8` pt forecast (`needs > +2.3` pts observed for `lower 95% > 50%`), and changing leaf values without recalibrating `δ*` (`ns-13`) confounds accuracy with override rate. Moreover, within column `c₂`, the same move is used across surviving rows `a₁`, so most value shift cancels.
- evidence check: Verified `ns-10` P1-B breakdown in `V3-GATES-PREREG.md:137` (*"revealed (n = 270) agree 0.975, |ΔP| 0.043; not revealed agree 0.998, |ΔP| 0.003"*), `ns-31`/`ns-32` (`δ = 0.0025`, `61.60`), `ns-20` (`+1.3`), `ns-21` (`49.13`), `ns-25` (`51.72`). As B2 notes, `0.043` mostly reflects newly switched-in species/abilities/items, not single moves alone.
- better kill test: Fold into `C2` after adding a Rust `move_announced` bit; gate in 0 slot hours on the 1,403-position `K = 32` judge (`ns-13`) before running any Showdown slot games.
- my estimate of the effect: +0.3 [−0.3, +1.0] vs Foul Play @200k
- rank among all proposals here: 7

### critique of B3: Train against the sharper versions of our existing opponents
- verdict: PROMISING-IF-FIXED
- fatal flaws: none
- fixable flaws and the fix:
  1. **Underpowered 30M training horizon on a 2.2B-step constant-lr net:** Along `G9q`, ~680M steps yield only `+1.99` in-process points (`g9q-31`: `51.99 [50.40, 53.56]`). Changing sampling temperature in only 46% of games (`36%` frozen + `10%` snapshot; `49%` self-play stays `T = 1.0`) over just 30M steps is unlikely to clear the `~+1.1` pt bar needed for 4,000 fixed pairs (`SE ≈ 0.55` pt, `bn-01`), risking a false negative on a real slow-building gain.
  2. **Frozen `BC-P3` mode collapse:** `frozen/` includes human-BC checkpoints (`BRIEF.md:40`), which have higher entropy than `cd` nets; cooling `BC-P3` to `T = 0.5` narrows early/mid-game human opening diversity.
  - **Fix:** Extend both arms to `50M + 5M cd` (~20 A40 GPU-h total, still within free A40 capacity), apply `T = 0.5` to `cd`/snapshot nets while optionally keeping `BC` frozen diversity, and evaluate 6,000 pairs on Dojo VM (~25 min).
- evidence check: Verified directly in `src/porygon/v2/train.py:1099-1106`: `_opp_sample` applies `torch.softmax(logits, -1)` (`T = 1.0`), compatible with `_opp_act_many` (`--stacked-opp`, `:1116`). Verified `v1-19` (`52.3 [50.9, 53.7]`), `dg-08` (`53.4` vs `64.6`), `g9q-22` (`50.64`), and `ns-30` (`50.47 [49.05, 51.90]` at `T 0.5`; *"at T 1.0 51.33 (info)"* in `results.csv`). Note: `ns-30`'s tiny `51.33% vs 50.47%` (`0.86` pt) gap after 100M dedicated exploiter steps against `T = 1.0` bounds B3's ceiling modestly.
- better kill test: Same clean controlled A40 fork (`train.py:1104` `logits / 0.5`), extended to `50M + 5M cd` and 6,000 `spd_h2h` duplicate pairs on Dojo VM (0 slot hours; GO iff lower 95% > 50.0%).
- my estimate of the effect: +0.4 [−0.5, +1.5] vs Foul Play @200k
- rank among all proposals here: 3

### critique of B4: Learn tactical action differences from the rest of the battle
- verdict: WEAK
- fatal flaws:
  1. **Catastrophic Monte Carlo SNR on 24,000 scalar pairs:** Even with a shared opponent choice at turn `t`, distinct moves `(a₁, a₂)` consume different Showdown PRNG draws immediately on turn `t`, decoupling the rest of the battle. Eight terminal `{0, 1}` rollouts yield `SE(ΔW) ≈ √(0.40 / 8) ≈ 0.22` (22 points!) against true tactical edges of `0.02–0.05` (`SNR < 0.04`). Fitting a neural residual on 24k noisy scalar labels (`literature-report.md:344` notes `>10⁵` samples needed; `ns-27` had 120k *low-variance* lookahead labels and still hurt by `−1.8` pts) will fit rollout noise.
  2. **Pointer-architecture and logit-scale mismatch:** `BattleNetV2`'s `policy_head` (`model.py:225`) is an entity pointer over `[candidate, field, own_act, opp_act]`, not a fixed 26-way head, and adding signed win differences `ΔW ∈ [−1, 1]` directly to logits conflates `Q(s, a)` with `log π(s, a)` (either negligible at scale 1 where `ΔW = 0.05` shifts odds by `1.1×`, or mixed-strategy-destroying at large `β`, duplicating `ns-27`/`ns-28`'s failure mode).
- fixable flaws and the fix: Replace 8-rollout terminal MC with 1-turn V2-leaf completed-Q values (`SE` ~10× lower, `24×` cheaper), apply as an *online* zero-avoiding Boltzmann KL term inside PPO (`C6`) rather than an offline frozen-trunk residual, and run on Dojo VM / A40 rather than saturating the 8-CPU A16 data host (`BRIEF.md:81`). Also note `State.serializeBattle` clones the simulator, not `fastenv/js/tracker.js`'s stateful `SideTracker`.
- evidence check: Verified `dg-02`/`dg-03` (`loss-patterns-report.md` §2–3: P8 `1.6` pts, P3s `0.9` pt association; §164 proposed state-cloning *only as a diagnostic* to measure causal ΔP, not as training labels), `dg-08` (`53.4`), `dg-04` (`82%`), `ns-27` (`48.17 [47.47, 48.87]`), `ns-28` (`48.14`).
- better kill test: Before any model training, run 2,000 cloned P8/P3 positions × 64 rollouts on Dojo VM (~0.5 h) purely to check whether held-out `ΔW` predictions beat `0` in R² and agree in sign with `lookahead2` `ΔQ`.
- my estimate of the effect: −0.5 [−2.0, +1.0] vs Foul Play @200k
- rank among all proposals here: 9

### critique of B5: Make simulated opponents choose from their own information
- verdict: PROMISING-IF-FIXED
- fatal flaws: none
- fixable flaws and the fix:
  1. **`poke-env` `battle.team` does not record what our side has revealed publicly:** To construct the opponent's observation (`_encode_opp` over *our* mons), you need the exact subset of our moves, items, and abilities publicly revealed so far. `poke-env` `battle.team` stores all 4 own moves and own items from `request` (`encoder.py:217-232`), not public reveal flags—so reconstructing the opponent's view requires running a passive public-protocol `SideTracker` alongside `Battle` from turn 1.
  2. **Depends on `B1` and `C2`, and leaves `δ = 0.0025` confounded:** Under today's `_encode_own` (`encoder.py:213`), `p_k` is blind to the opponent's *own* sampled `teraType` unless `B1` is applied! Forced switch-ins (`lookahead2.py:140`) also still minimize against our private observation inside `M_k`, and `needs_switch(s₁)` stays unexpanded (`C2`). Crucially, constraining `y_k ≥ 0.5 p_k` inflates `Q(a_L) − Q(a_net)` gaps, spiking overrides at fixed `δ = 0.0025`—`δ*` *must* be recalibrated (`ns-13`), and `p_k`'s temperature (`T = 0.5` vs `1.0`) must be pre-registered.
  3. **Underpowered 1,000-pair slot test:** `12–20` slot hours at `SE ≈ 1.2` pts has <25% power for `+1.3` pts.
- evidence check: Verified `lookahead2.py:78-110` (`solve_bayes`), `fpr-303472` (`53.2`), `fpr-307076` (`53.4`), `fpr-307072` (`40.8`), `ns-25` (`51.72`), and Jaxcalibur's surprise-cutoff rationale. Formal check (`y = 0.5p + 0.5z ⇔ y_j ≥ 0.5p_j`) is exact—and implies `solve_bayes` can run unmodified on `M̃_k = 0.5(M_k p_k)1ᵀ + 0.5 M_k`.
- better kill test: Build after `B1` and `C2` using a turn-1 passive `SideTracker` for opponent-view parity (verified via B5's hidden-bench invariance test), recalibrate `δ*` at matched override rate on the 1,403-position `K = 32` judge (`ns-13`, ~1 Dojo VM GPU-h, 0 slot hours), and advance to the Showdown slot only if `K = 32` precision improves.
- my estimate of the effect: +0.7 [−1.5, +2.0] vs Foul Play @200k (+1.0 [0.0, +2.2] once `B1`, `C2`, and `δ*` recalibration are applied)
- rank among all proposals here: 4

## synthesis from Member C
- best combination: Exploit hardware parallelism (`BRIEF.md:77-85`) so the **Showdown benchmark slot**, the **free A40**, and the **Dojo VM** never block one another, respecting one-change-at-a-time discipline:
  1. **Immediate on Showdown slot + Dojo VM (0 training GPU-h): `C2` (+ `B2`'s Rust `move_announced` bit).** `C2` repairs the verified bug where our faints/pivots (`needs_switch(s₁)`) leave `own_act = 0` in `BattleNetV2` (`lookahead2.py:208`), directly targeting `la2-len1`'s short-game KO-trade deficit in 1.5 days. If `C2` passes, test **`B5` (fixed)** (passive public `SideTracker` + `M̃_k` transform + `δ*` recalibration) offline on `K = 32` (`ns-13`) before any slot run.
  2. **Immediate in parallel on A40 + Dojo VM (0 slot hours): `B1` → `C1`/`A1` (Spec-compatible input repairs).** Run `B1` first (with `_opp_obs` frozen-opponent masking and 300-battle JS/Python parity) to cleanly isolate own pre-Tera visibility (`368/509` Gen 9 species affected). Immediately fork the winner into `C1` + `A1` part (a) (`aux_tera` fix, Knock-Off `item_none = 1`, and `inference.py` speed/damage/Choice-lock posteriors inside Spec v2.1 shapes, updating `v3leaf.py`).
  3. **Next on A40 (0 slot hours): `C3`, then optional `B3` (50M).** Layer `C3` (Markov Dual-View Omniscient Critic + HL-Gauss, gated by `dg-06`'s 5M coherence probe) onto the Spec v2.2 net to strip hidden-team variance, then test `B3`'s 3-line `T = 0.5` pool-opponent sharpening over 50M steps.
  4. **Scale search & Gen 4 transfer: `C4` + `A5` (fixed).** Gate `C4` (Rust IS-RM-MCTS + root-cached CUDA-graph leaf server, ≥35k leaves/s) on Dojo VM (`Stage A`, 0 slot hours), and run `A5` (cooling `R1b-F` on A40/inline, Gen 4 `v3leaf` parity, paired 200k `fp4-04` re-baseline, requiring superiority `> 50%`).
- what everyone missed:
  1. **`DamageFeatures` (`src/porygon/v2/damage.py:76-87`) has deterministic tactical blind spots even on fully visible state:** `damage.py` explicitly omits **Substitute** (`:86`, so `P(KO now)` reports 100% OHKOs straight through an active Sub!), **Orichalcum Pulse / Hadron Engine** (Koraidon/Miraidon 1.33× Atk/SpA), **Grassy Terrain halving Earthquake**, **Tera Shell / Ice Face**, **Collision Course / Electro Drift**, **Supreme Overlord** stacks, **Fishious Rend / Bolt Beak**, and **Counter / Mirror Coat / Final Gambit** (`0` damage). Fixing these parameter-free tensor formulas helps every checkpoint and search leaf at zero training cost.
  2. **Unweighted `K = 8` Monte Carlo variance in `lookahead2` (`lookahead2.py:6,90`):** Eight independent world draws give `0%` or `12.5%` jumps on Scarf/Tera sets, adding sampling noise larger than `δ = 0.0025`; stratified/antithetic sampling with exact library posterior weights `w_k` is free.
  3. **100%-active global gradient clipping (`dg-06`, `BRIEF.md:70`):** `max_grad_norm = 0.5` clips *100% of minibatches* (`coef ≈ 0.4`), biasing AdamW directions whenever rare high-advantage tactical states occur.

