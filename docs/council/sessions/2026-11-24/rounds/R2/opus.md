### critique of B1: Give the policy its own Tera type before it chooses whether to Tera
- verdict: STRONG
- fatal flaws: none
- fixable flaws and the fix: the fork changes what our net sees, and frozen and snapshot opponents load nets that use the same encoder. Either pin the opponents to the v2.1 encoder or switch both arms over, so the opponent pool stays identical across arms. B notes this risk but leaves it open. The warm start shifts `tera[0:6]` from 0 to a value. The shift works against the candidate, so a GO is still believable, but log the KL over the first 1M steps.
- evidence check: confirmed. `fastenv/js/encoder_v2.js:28-29` says "own `tera` / `tera_p` stay 0 until the mon terastallizes … poke-env never fills from the request's teraType". The Python side reads `mon.tera_type` at `v2/encoder.py:213`. g9q-32 is 60.8 and ns-32 is 61.60; both quoted correctly.
- better kill test: theirs is fine. The 100-pair byte-check is a good start. 4,000 pairs gives a CI of about ±1, which is the right size.
- my estimate of the effect: +1.0 [0, +2.5] raw; +0.5 [−0.5, +1.5] under LA2, because the engine search already gets the type.
- rank among all proposals here: 2

### critique of B2: Let a simulated move reveal itself to the leaf evaluator
- verdict: PROMISING-IF-FIXED
- fatal flaws: none
- fixable flaws and the fix: the number that matters is action agreement, not |ΔP|. `V3-GATES-PREREG.md:137` reads "revealed (n = 270) agree 0.975 … not revealed agree 0.998". That is about 2.5% disagreement on a small slice of leaves, so ~1 consequential change per 10 games is too high. Stage 2 (1,000 pairs) cannot detect less than about +3. Keep the offline stage 1 gate and add one more: the change must flip a_L or the gate decision in ≥ 1% of decisions on the 1,403-position set. C2(3) duplicates this idea, so merge them.
- evidence check: 0.043 vs 0.003 confirmed (above). The ns-31/32 δ of 0.0025 is quoted correctly.
- better kill test: offline only. The flip rate above, plus a check on the K32 judge that the flips are confirmed. Skip the slot run unless flips are ≥ 1%.
- my estimate of the effect: +0.3 [−0.3, +1].
- rank among all proposals here: 3

### critique of B3: Sharper frozen opponents
- verdict: WEAK
- fatal flaws: the kill test is biased toward the candidate. The candidate trains against T 0.5 copies of the same family and is then judged in a T 0.5 head-to-head against that family. That rewards fitting a sharper copy of the family, which is what B says it doesn't want. v1-19 shows that playing colder makes a player stronger. It does not show that training against colder opponents does.
- fixable flaws and the fix: gate on fixed outside opponents (pure search, BC-P4, LA2) before any head-to-head.
- evidence check: ns-30's note is correct ("at T 1.0 51.33 (info)"). v1-19 is 52.3 [50.9, 53.7], but it is a deployment-temperature result.
- better kill test: both arms play 1,000 games each against pure gen9 search on the same pairs (the g9q-23 harness).
- my estimate of the effect: 0 [−1, +0.5].
- rank among all proposals here: 9

### critique of B4: MC action-difference residual
- verdict: PROMISING-IF-FIXED
- fatal flaws: none. The labels are honest Monte Carlo Q differences under the real hidden world.
- fixable flaws and the fix: the signal-to-noise ratio is poor. With 8 replicates, the SE of a win difference is about 0.25, while true differences are mostly ≤ 0.05. On top of that, 24k roots is little data for a residual that has to generalise. Check offline first: on held-out roots, the residual must predict the sign of the difference (AUC > 0.55) and change the argmax in ≥ 1% of states. Use 32 replicates on fewer roots.
- evidence check: dg-02/03 are logged-pattern associations. B describes them correctly.
- better kill test: the offline AUC and flip-rate check above, before running 4,000 pairs.
- my estimate of the effect: +0.3 [−0.5, +1].
- rank among all proposals here: 5

### critique of B5: Opponent-view prior in lookahead2
- verdict: PROMISING-IF-FIXED
- fatal flaws: none. The formal check (y ≥ 0.5p) is correct.
- fixable flaws and the fix: building the opponent's first-person poke-env observation from an engine world is the risky part. It has to track what of our side they have seen, and handle Illusion and slot order. Realistically that is 6–10 days. My A4 (mixing in the net's opponent-action head, 1 day) tests the same lever, a non-equilibrium opponent prior. Run A4 first, and build B5 only if A4 shows a signal. 1,000 pairs can only detect about +3.
- evidence check: fpr-307072 at 40.8 and fpr-307076 at 53.4 are quoted correctly. Jaxcalibur hands the opponent the prior after a surprise (lit report L140–143), which supports the idea.
- better kill test: A4's sequential design (ns-14 style, n_max 600) as the cheap proxy.
- my estimate of the effect: +0.5 [−1.5, +2].
- rank among all proposals here: 4

### critique of C1: Spec v2.2 encoder repair
- verdict: STRONG (once trimmed)
- fatal flaws: none
- fixable flaws and the fix: all four claimed defects are confirmed, in `encoder_v2.js:28-33`: own Tera, `aux_tera` −1, "removed … encoded as unknown (posterior candidates, item_known 0)", and "weather_elapsed is ~0". Item (5) is my A1(a). The predicted +3.5–6.5 is inflated: it adds up guessed parts and has no oracle ceiling behind it. Bundling everything rules out attribution, so run B1 as a nested arm, or ablate after a GO. Changing the encoder for frozen opponents confounds the result (see B1). The Choice-lock mask is wrong whenever the item has not been revealed, so make it soft. A GO requires lower bound > 51 on 2,000 pairs, i.e. a true h2h of about +2.4, which is strict for a 30M fork. Use 4,000 pairs.
- evidence check: bc-01 (22.4) changed the whole architecture, not the damage features alone. The fp4-07 oracle applies to search, not to the net.
- better kill test: as theirs, with 4,000 pairs and opponent encoders frozen. Run my A1 oracle head-to-head alongside it (1 CPU-hour) to bound the ceiling.
- my estimate of the effect: +1.5 [0, +3] in gen9; +0.5 [−0.5, +1.5] in gen4.
- rank among all proposals here: 1

### critique of C2: Bilateral forced-switch resolution
- verdict: WEAK
- fatal flaws: the core mechanism is wrong. A leaf where our active has fainted is in distribution, because the net is evaluated on forceSwitch requests in training. The lookahead2 docstring says so: "one where we have to switch is scored directly (the net makes forced switches in training)". Zeroing `own_act` is what training sees too. Taking max over b₁ adds optimism on top.
- fixable flaws and the fix: keep δ fixed, because re-deriving δ* confounds the comparison. Part (3) duplicates B2.
- evidence check: the only support is la2-len1, "short pairs … −3.0 ± 7.7", which is post hoc and not significant. The raw net shows the same short-game gap (dg-08), so the gap is not a search defect.
- better kill test: offline, on the 1,403-position set, how often bilateral expansion flips the gate decision. Kill if that is under 2%. The proposed 500 pairs with lower bound > 50.5 needs a true effect of about +3.5.
- my estimate of the effect: +0.2 [−0.5, +0.8].
- rank among all proposals here: 8

### critique of C3: Omniscient critic + HL-Gauss
- verdict: PROMISING-IF-FIXED
- fatal flaws: none, but the evidence is misquoted. dg-06's 3.8 [3.5, 4.3] is the value-to-policy trunk-gradient ratio, not coherence. The coherence figure (≈ 0.005) appears only in timeline.md, and low per-sample cosine is normal in RL.
- fixable flaws and the fix: about 51% of battles are against frozen nets, snapshots or baselines, so there is no learner u² for them. That needs an extra learner forward on the opponent's observation, which costs well over the claimed < 2%. A new critic is biased early, and g9q-10 showed critic bias hurts, so use `--critic-warmup-steps`. Drop HL-Gauss, because bundling it rules out attribution. Fp4-04→fp4-08 is an oracle given to search, not evidence about return variance.
- better kill test: offline, fit V_omni on saved rollouts. Kill unless it explains ≥ 15% more return variance than V_pub at mid-game.
- my estimate of the effect: +0.3 [−1, +1.5] at 50M steps.
- rank among all proposals here: 7

### critique of C4: IS-MCTS in Rust with a CUDA-graph leaf server
- verdict: WEAK
- fatal flaws: the cost is off by roughly 5–10×. A Rust IS-MCTS plus a CUDA-graph leaf server is weeks of work, not 3.5 days. ns-12 already reached only 7,465 of the 35,300 leaves/s target, and Stage A "replays V3-I1", which will fail again until the new server exists. ns-25 found that 4× deeper search was FLAT.
- fixable flaws and the fix: the truncation is misread. Jaxcalibur gives the *opponent* the network prior after we surprise it (lit report L140–142). It does not stop the rollout when the opponent reveals something. +6–11 points over raw is set against g9q-32; the bar is LA2 at 61.6.
- evidence check: the MNZ figures 61.6→67.5 are correct (lit L204). "+100–150 Elo" is correct.
- better kill test: first a throughput prototype limited to static per-root embeddings in Python, against ns-12's figure.
- my estimate of the effect: +1 [−2, +4], and only if it ships.
- rank among all proposals here: 11

### critique of C5: Damage-feature NNUE
- verdict: WEAK
- fatal flaws: in gen9 the bot is net-first, and CPU MCTS is about 52% (g9s-01), so a better CPU leaf does not touch the headline. In gen4, A5 is moving the bot onto the net.
- fixable flaws and the fix: the diagnosis is correct. `nnue/features.py:13` trains on "convert_full's true State" and has no damage terms. Treat this as a gen4-only option. AUC on Set H also maps weakly to win rate.
- evidence check: ns-16 (0.714 / 0.718 / 0.748) is quoted correctly.
- better kill test: theirs, plus a 500-pair gen4 search-vs-search run before any Foul Play games.
- my estimate of the effect: gen4 +1 [−1, +3]; gen9 ≈ 0.
- rank among all proposals here: 10

### critique of C6: Online Gumbel completed-Q targets
- verdict: PROMISING-IF-FIXED
- fatal flaws: the cost. 5% of about 1.5k decisions/s at about 0.85 s per lookahead (ns-14) is roughly 60 CPU-seconds per second. That needs a fastenv-to-poke-engine bridge (A3 estimates 4–6 days) and far more than 15% extra wall time.
- fixable flaws and the fix: the ns-27 critique is partly confirmed. π_LA gives overridden draws exactly zero mass (`EXIT-LA2-DESIGN.md:38-46`), and the run used `--aux-coef 0` (PREREG:37). But in ns-27 the policy moved (r_E 0.347) while the own-policy control was neutral (49.41). That points to the teacher (1-ply, now K = 4, noisier) as the likely cause. "Monotonic improvement" holds only with respect to the net's own Q, which is circular.
- better kill test: offline. Retrain on ns-27's 120k positions with the soft π_imp target and aux 0.1, then run E′ vs C. That costs hours and tests the stated causes directly.
- my estimate of the effect: +0.3 [−1.5, +1.5].
- rank among all proposals here: 6

## synthesis from Member A
- best combination:
  1. One A40 encoder fork: C1 items 1–4 with B1 as a nested arm, opponent encoders frozen, 4,000 pairs. In parallel on the VM, A1's oracle head-to-head (1 CPU-hour) to bound how much C1(5)/A1 evidence can be worth.
  2. Cheap, CPU or slot only: A2 purification, then A4. Build B5 only if A4 shows a signal.
  3. Offline gates that cost hours: B2 flip rate, C2 flip rate, C6 re-run on the ns-27 dataset, C3 explained variance. Promote only what passes.
  4. Gen4: A5.
- what everyone missed:
  - Statistical power. Head-to-head gains transfer to Foul Play at about 0.5×, so +1 point vs Foul Play needs about +2 h2h. Getting a lower bound above 50 then takes about 2,500 or more pairs, and almost every proposed Showdown kill test (500–1,000 pairs) is blind below about 3 points.
  - Lineage. Any encoder fix forks off the A30 G9q main run, so G9q keeps training on the v2.1 encoder. Nobody plans the re-baseline or the decision on which lineage continues.
