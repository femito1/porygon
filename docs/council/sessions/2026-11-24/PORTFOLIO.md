# Council session 1 — portfolio (2026-11-24, orchestrator synthesis)

Members A / B / C (mapping in `rounds/MAP.txt`; never shown to members). Worktree pinned at f3f6112. Rounds: R1 blind
proposals (16), R2 cross-critique (32 critiques, 2 per proposal), R3 rebuttal (A: 2 revised, 3 withdrawn, adopted B1;
B: 3 revised, 2 withdrawn; C: 3 revised, 3 withdrawn, adopted A1). **Final top pick unanimous: own pre-Tera visibility (B1).**
Tokens: R1 8.03M (incl. a failed 0.19M Carbon call) + R2 4.39M + R3 1.84M = **14.26M gross / 2.21M fresh of the 30M cap**.
No R4 vote (the top pick is uncontested; the rest are ranked by the rule below, disagreements shown).

**Scoring:** expected gain vs Foul Play @200k = median of the members' last estimates (R3 if revised, else R2 critics +
R1 author); P(pass) = the orchestrator's judgment from the critiques; cost on our hardware (BRIEF "Resources").
Rank = gain × P ÷ cost. Gains are forecasts, never results.

**Orchestrator's adversarial checks (done before ranking):**
- `verify` on R1–R3: no unknown row ids; all NUMBER flags read by hand = window artefacts or correctly quoted
  (e.g. la2-len1 "short −3.0 ± 7.7" is the row's verdict text; its estimate −5.9 is short minus long).
- No member read outside its worktree (one `../../../` in R1 was inside a code file Astra read).
- Code facts re-checked by the orchestrator:
  - `fastenv/js/encoder_v2.js:27-35` documents own pre-Tera zeros, `aux_tera` −1 before Tera, removed items encoded as
    unknown, `weather_elapsed` ≈ 0 ("reference behaviours reproduced as is … not fixed");
  - library count: 368 / 509 gen9 species have an identical (moves, ability, item) set under > 1 Tera type, = 60.1% of
    all observed sets (`src/porygon/data/gen9randombattle_library.json.gz`);
  - `lookahead2.py:13` "one where we have to switch is scored directly (the net makes forced switches in training)"
    (C2's core claim was a design choice, not a bug; C2 withdrawn);
  - `damage.py:76-87` "not modelled": substitute, Orichalcum Pulse / Hadron Engine, Tera Shell, Ice Face, grassy
    terrain halving Earthquake, counter / mirror coat (follow-up F4).

**Rulings:**
- Engine provenance (A3): training against our own poke-engine search is allowed (our bot; no Foul Play code, no
  modelling of its rule), but a GO must also hold in-process against our nets. A3 is withdrawn anyway.
- Lean: no proof attempt. The top-5 formal claims are zero-leakage or solver-equivalence identities (unit tests
  settle them); A4's bound was withdrawn. Nothing to install.

**Design rules for every council kill test (R2 "what everyone missed"):**
- Power:
  - h2h → Foul Play transfer ≈ 0.5× (g9q-21…32), so +1 point vs Foul Play ≈ +2 h2h.
  - 4,000 in-process pairs resolve ≈ ±1.1 h2h.
  - 1,000 Showdown pairs resolve only ≈ ±3 points, so slot runs are confirmations and headlines.
  - Search changes pass an offline gate (1,403-position K32 set) first.
- Observation changes use per-side encoder flags in fastenv (`_opp_obs`) and `spd_h2h`. Frozen / snapshot opponents
  keep the encoder they were trained on.
- Lineage:
  - G9q stays untouched on the A30 until a change passes.
  - Promotion: a cooled fork beats step-matched G9q-cd with lower 95% > 50.5. Then restart G9q in place with the
    flag, by a versioned netv2.env deploy.
- At the next NEW run, also evaluate on a held-out monthly set distribution (B).

---

## P1: Own pre-Tera visibility (spec v2.2a)   [origin: B1; = C1 trimmed; adopted by A]   status: KILL-TEST-READY
- lever / generations: inputs (bug fix) / gen9
- claim (one sentence, falsifiable): a fork whose learner sees its own Tera types from turn 1 beats a matched v2.1 fork
  in-process (lower 95% > 50% over 4,000 duplicate pairs) after 30M + 5M-cooldown steps.
- mechanism:
  - Fill own `tera` / `tera_p` (slots 0–5) from the owner's request `teraType`, matched by `ident`, in `encoder.py:213`
    and `encoder_v2.js:267`.
  - Keep `terastallized` = 0 until `-terastallize`. Leave aux targets, legal actions and shapes (5,809,977 params)
    unchanged.
  - `damage.py`'s hypothetical-Tera features (`active_tera_vs`) then carry Tera-STAB / Tera-defence information exactly
    when the Tera decision is made.
  - Today the net decides Tera blind to the type for ≈ 60% of sets.
- evidence:
  - `encoder_v2.js:28-29`, `encoder.py:213`, the library count above; Volcarona: identical set, Tera Water 1,635 /
    Tera Ground 1,571.
  - Tera used in 90.3% of self-play games (dg-02).
  - Baselines g9q-32 60.8, ns-32 61.60.
- strongest objection (R2) and answer (R3):
  - A, C: the shared JS encoder would also change frozen pool opponents' inputs.
    → Per-side flag; `_opp_obs` restores v2.1 zeros for frozen / snapshot policies; both learner seats get v2.2a.
  - A: the warm start sees "type set" only after Tera so far.
    → KL monitor on the first 2M steps; 30M steps.
- expected gain vs Foul Play @200k: raw A +1.0, B +1.2, C +1.5 → **+1.2**; under LA2 A ≈ +0.9, B +0.8, C +0.8 → **+0.8**.
  P(pass) 0.45: real, decision-relevant information, but a 30M fork may be too short to learn it.
- kill test:
  - Stage 0 integrity:
    - 300 streaming battles, Python / JS parity 100% on the corrected fields;
    - legacy pool inputs and aux targets byte-identical;
    - changing only the opponent's unrevealed Tera leaves our observation byte-identical;
    - Water/Ground damage fixture.
  - Stage 1 arms: E (v2.2a for the learner) vs C (v2.1). Both start from the SAME md5-pinned uncooled G9q checkpoint,
    with the same seeds, pool files and schedule; 30M steps + identical 5M cooldown each (cosine ×0.1, floor 0.35).
  - Harness: in-process `spd_h2h`, T 0.5, per-side encoder routing, **4,000 fresh duplicate pairs**.
  - Rule:
    - GO iff E-cd vs C-cd pair-mean lower 95% > 50.0%;
    - lower bound in (50.0, 50.5] ⇒ replicate from a second seed before any promotion;
    - else PARK (a 30M fork can be too short; not a kill of the idea).
  - Cost: ≈ 1–1.5 engineering days, ≈ 13 A40-h (G9-P4 parks meanwhile and resumes), ≈ 0.3 data-host CPU-h.
  - Owner: netv2 (spec owner), with fastenv's parity harness.
- if GO:
  - E-cd and C-cd on the same 1,000 fresh Foul Play @200k pairs, paired difference (≈ 4 slot-h).
  - LA2 on E-cd (v3leaf v2.2a, re-derived δ*) vs the benchmark bot, fixed 1,000 pairs (≈ 6 slot-h).
  - Continue G9q-v22a on the A40 ≥ +150M, then promote by the lineage rule.
  - Queue F1–F4 as single-variable nested forks.
- formal check: zero leakage: for any reassignment τ of unrevealed opponent Tera types that preserves the public
  protocol, encode_v22a(h) = encode_v22a(τ(h)). A unit test, not Lean.

## P2: ExIt with a soft completed-Q target, offline on the ns-27 data   [origin: C6 revised (R3)]   status: KILL-TEST-READY
- lever / generations: training signal (search → net) / gen9
- claim: ExIt-LA2's harm (ns-27) came from its hard-gated target (overridden actions set to exactly 0). Replacing only
  the target with π_imp ∝ π·exp(β(Q̂ − V)) on the same 120k positions makes E′ beat both the own-policy control and the
  parent.
- mechanism: zero-avoiding Boltzmann reweighting keeps the support (no logits pushed to −∞). Everything else matches
  ns-27: data, parent cd@805, epochs, optimiser, aux-coef. β is pre-registered: the value giving mean KL(π_imp ‖ π)
  ≈ 0.05 nats on a calibration split, fixed before training.
- evidence: ns-27 48.17 [47.47, 48.87] and ns-28 48.14 (HARMFUL); ns-29 control 49.41; `EXIT-LA2-DESIGN.md:38-46`
  (hard zeros); r_E 0.347 (the policy moved).
- strongest objection (R2) and answer (R3):
  - A: the failure may be the teacher (one-ply) rather than the target.
  - B: none of the four failure causes was isolated.
  - → R3 makes it single-variable (target only), so a NO-GO closes the ExIt-LA2 line in both forms.
- expected gain: C +0.6, A +0.3, B 0 → **+0.3** (if ported to cd@2030). P(pass) 0.2. Its value is mostly that it settles
  whether search gains can move into the net at all.
- kill test:
  - Arms: E′ (soft target) vs C (ns-29's saved control) and vs S (cd@805.03M).
  - Harness: in-process `spd_h2h`, T 0.5, 4,000 duplicate pairs each.
  - Rule:
    - GO iff E′ vs C lower 95% > 50.5 AND E′ vs S lower 95% > 50.0;
    - else record "ExIt-LA2, soft target: NO-GO" in BRIEF's tried list and close the line.
  - Cost: 0.5 engineering day, ≈ 1.5 GPU-h (data host A16 via `srun --overlap`, or the VM), ≈ 0.5 CPU-h.
  - Owner: fastenv (ExIt-LA2 owner).
- if GO: fresh lookahead2(cd@2030) labels (stage 2 design, 3× positions) and fine-tune cd@2030; test vs cd@2030 and
  under LA2.
- formal check: supp(π_imp) = supp(π) and E_π_imp[q] ≥ E_π[q] (Chebyshev). A unit test; it says nothing about win
  rate.

## P3: Gen4 re-baseline at the fixed yardstick, then LA2-gen4 only if the raw net is close   [origin: A5 revised]   status: KILL-TEST-READY
- lever / generations: search / deployment stack / gen4
- claim: on identical gen4 Foul Play @200k pairs, raw R1b-F-cd scores within 2 points of the incumbent gen4 bot. If it
  does not, the gen4 net route stops for now.
- mechanism:
  - fp4-04 (59.19) is time-based, so it is incomparable with the 200k yardstick and was never paired with a v2 gen4 net.
  - Stage 1 is a measurement the thesis needs anyway: a gen4 headline at the fixed yardstick.
- evidence: v2g4-10 55.6 [51.5, 59.7] (400 g); v2g4-15 55.2 (vs its 300M ancestor, not vs Foul Play); fp4-04 contains
  +4.7 of fitted leaf eval (fp4-02 → fp4-04).
- strongest objection (R2) and answer (R3):
  - B: no measured gen9 → gen4 transfer.
  - C: the non-inferiority adoption rule could downgrade the bot.
  - → Adoption now needs superiority; slot stages are fixed-n; the cooldown runs off R1b-F's critical path.
- expected gain: raw − incumbent: A −1.5, B −1, C 0 → **−1.0** (the deliverable is the re-baseline). LA2-gen4 −
  incumbent: A −0.5. P(port worth building) 0.35.
- kill test:
  - Setup: cool R1b-F's latest checkpoint, 5M cosine ×0.1, INLINE in its T4 job (≈ 3.5 h; keeps the A40 for P1).
  - Arms: the incumbent (fp4-04 config) and raw R1b-F-cd (T 0.5), each 1,000 games vs gen4 Foul Play @200k on the
    same team pairs and seed. The fixed-iteration launcher needs a gen4 extension (½ day) with ITERS_OK checks.
  - Rule:
    - paired Δ = raw − incumbent; build LA2-gen4 iff Δ ≥ −2.0;
    - else stop the gen4 net route and record the re-baselined incumbent as the gen4 headline.
  - Cost: ≈ 0.5 engineering day, ≈ 3.5 T4-h inline, ≈ 4–6 slot-h.
  - Owner: netv2 (cooldown) + wsp (launcher / slot run).
- if GO:
  - Gen4 leaf parity (300 battles, `v3leaf` is gen9-hardcoded) and δ*.
  - LA2-gen4 vs its raw net, fixed 1,000 pairs, lower 95% > 50.
  - Then vs the incumbent on fresh identical pairs; adopt only if the paired lower 95% > 0.
- formal check: none

## P4: Privileged-team control-variate baseline (offline Stage 0 first)   [origin: C3 revised]   status: NEEDS-DESIGN (Stage 0 ready)
- lever / generations: training signal / both
- claim: a training-only baseline that also sees the opponent's true team cuts mid-game return MSE by ≥ 15% and the
  score-gradient variance by ≥ 10% versus V_pub on held-out games.
- mechanism:
  - V_priv(sg(trunk), z_opp) is built from the aux true-team tensors already in every fastenv transition.
  - GAE λ 0.754, the BCE value loss and the deployment V_pub stay unchanged; HL-Gauss is dropped (no bundling).
- evidence: dg-06 (the value/policy gradient-norm ratio 3.8%; coherence ≈ 0.005 is a timeline note, not a row);
  AlphaStar / PerfectDou privileged critics.
- strongest objection (R2) and answer (R3):
  - A, B: no learner u² for pool games; λ is 0.754, not 0.95; bundled with HL-Gauss.
  - → z_opp from the aux tensors (present in pool games too); λ and the loss kept; HL-Gauss dropped; offline gate first.
- expected gain: C +0.8, B +0.4, A +0.3 → **+0.4** at 55M steps. P(Stage 0) 0.35, P(Stage 1 | Stage 0) 0.4.
- kill test:
  - Stage 0: record ≈ 8M frozen-G9q rollout transitions (full opponent mix, force-switch steps included); fit V_pub
    and V_priv on 6M; evaluate on 2M held-out, game-clustered transitions.
  - Rule: KILL unless mid-game (turns 5–25) MSE drops ≥ 15% AND tr Cov(g·Â) drops ≥ 10%.
  - Cost: 1 engineering day, ≈ 1.5 h of collector time, 1–2 GPU-h.
  - Stage 1 (only after Stage 0): 2 × 55M forks on the A40 (≈ 20 h), 4,000 pairs, GO iff lower 95% > 50.
  - Owner: netv2 + wsp.
- if GO: fold into the main recipe by the lineage rule.
- formal check: an action-independent baseline leaves the policy gradient unbiased (standard). It does not make
  bootstrapped GAE unbiased (B). None for Lean.

## P5: Battle-evidence posterior: observed speed order + "unlocked", in the v2.1 shapes   [origin: A1 revised; adopted by C]   status: NEEDS-DESIGN (Step 0 ready); after P1
- lever / generations: inputs (posterior fidelity) / both
- claim: filtering the opponent-set posterior by observed speed races and `Evidence.unlocked` improves next-race
  prediction offline and, as a fork, beats the control in-process.
- mechanism: port `inference.py`'s `speed_ok` (races) and `unlocked` into `_encode_opp` and `encoder_v2.js` (tracker:
  races / outran / speed). `item_p` and `stats`, and through them `DamageFeatures.speed`, then carry the observed
  order. No shape change, no Choice-lock masking, damage `hits` deferred.
- evidence: `encoder.py:9` / `:122` (no speed / damage evidence); dg-05 (the search's sampler gets hidden items 82–88%
  right with that evidence); dg-08 short-game gap 53.4 vs 64.6.
- strongest objection (R2) and answer (R3):
  - B, C: the oracle test is OOD and confounded; posterior and history were bundled.
  - → Oracle dropped (diagnostic only); history becomes a separate later branch; offline Step 0 decides first.
- expected gain: A +0.7, B +0.7, C +2.0 → **+0.7** raw (smaller under LA2: the search worlds already use the evidence).
  P(pass) 0.3.
- kill test:
  - Step 0, offline, held-out self-play battles only: at each equal-priority race after the first for a pair, score
    the observed order under `DamageFeatures.speed` with the filtered vs unfiltered posterior.
  - Step 0 kills if either holds:
    - the filter changes `item_p` by TV ≥ 0.1 on < 2% of opponent-active decision rows;
    - the paired bootstrap lower 95% of the next-race log-loss improvement is ≤ 0.
  - Step 1:
    - 300-battle parity, 100%.
    - Fork E (evidence) vs C, 30M + 5M; share P1's control only if run from the same parent and seeds.
    - 4,000 pairs; GO iff lower 95% > 50.
  - Cost: 2–3 engineering days (JS port), CPU-h for Step 0, ≈ 6–7 A40-h.
  - Owner: netv2 + wsp (Step 0).
- if GO: second seed; paired Foul Play @200k; then the separate v2.2c branch (zero-initialised history columns), then
  damage `hits`.
- formal check: none (C's support / simplex note is a unit test)

## P6: Opponent-view reply prior in lookahead2 (offline comparison first)   [origin: B5 revised; A4 withdrawn in its favour]   status: NEEDS-DESIGN
- lever / generations: search / gen9
- claim: a reply prior built from the opponent's own first-person view predicts held-out opponents' executed decisions
  better (paired log loss) than the net's opponent-action head. Constraining replies to y ≥ 0.5p then improves
  K32-confirmed override precision at a matched override rate.
- mechanism: `solve_bayes` on transformed columns 0.5 M[:,j] + 0.5 M p (B's equivalence).
- evidence: `lookahead2.solve_bayes` minimises against our actual private team; Jaxcalibur's surprise handling; fpr-307072
  and fpr-307076 as cautions.
- strongest objection (R2): building the view needs a public-protocol tracker from turn 1 (6–10 days), the margin must be
  recalibrated, and 1,000 slot pairs cannot resolve +0.6. Answer (R3): offline stages first, no slot request.
- expected gain: B +0.6, A +0.5, C +0.7 → **+0.6** over LA2. P 0.2.
- kill test: the offline prior comparison (≈ 1 engineering day) on 2,000 held-out fastenv roots. Build the tracker only
  if the paired 95% excludes 0. Then a game-disjoint K32 split at matched override rate. Owner: fastenv.
- if GO: tracker + parity (6–10 days), then a variance-sized fixed-n slot test.
- formal check: y feasible ⇔ yⱼ ≥ 0.5pⱼ (solver equivalence; a unit test)

## P7: Reveal-aware leaves (announced moves only)   [origin: B2 revised; C2(3) merged]   status: PARKED
- Needs a Rust `move_announced` event bit; poke-engine emits state diffs, not protocol events.
- Forecast A +0.3, B +0.3, C +0.3 → +0.3; P 0.15; cost 3–5 engineering days.
- Revisit after P1 / P6, using B's offline action-flip gate (≥ 1% of held-out played actions change and K32 favours
  them).

## Follow-ups queued behind P1 (single-variable nested forks; NEEDS-DESIGN)
- F1: removed-item state (v2.2b). Knocked-off / consumed items are encoded as removed, not unknown
  (`encoder_v2.js:31-32`). Separate "unknown", "empty slot" and "new item via Trick" (B).
- F2: `aux_tera` supervision before Tera, from the opponent's request (`encoder_v2.js:30`). A training-signal change,
  separate from inputs.
- F3: `weather_elapsed` kept across `[upkeep]` (`encoder_v2.js:33`).
- F4: damage-feature fidelity: substitute, Orichalcum Pulse / Hadron Engine, Tera Shell, Ice Face, grassy-terrain
  Earthquake (`damage.py:76-87`).
- F5: stratified / antithetic world sampling for lookahead2's K 8 (C, R2; unverified).

## Withdrawn in R3 (not results: do not re-propose without answering the stated reason)
- A2 purification (T < 0.5). T 0.5 already squares the probabilities; colder settings collapse real 60/40 mixes.
  Forecast +0.2 doesn't justify the slot run.
- A3 search sparring partners in the league. fastenv sends binary arrays, not a protocol stream (4–6 day bridge); a
  20 ms search is a different, weak player.
- A4 root reply-mixture. The bound only covered a fixed action; root blending cannot repair pre-minimised forced
  switches. Superseded by P6.
- B3 colder pool opponents. The in-family h2h gate rewards specialisation; v1-19 is a deployment result, not a training
  result.
- B4 MC action-difference residual. SE ≈ 0.22 per root against 0.02–0.05 edges; no justified Q → logit scale.
- C2 bilateral forced switches. Our forced-switch leaves are in distribution (`lookahead2.py:13`); max over b₁ per
  world is information-inconsistent.
- C4 Rust IS-MCTS + CUDA-graph leaf server. Weeks of work, not 3.5 days; ns-12's throughput gap; deeper search FLAT
  (ns-25).
- C5 damage-feature NNUE. CPU leaves do not touch gen9's net-first headline; gen4-only at best.
