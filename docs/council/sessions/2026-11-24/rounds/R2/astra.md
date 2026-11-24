### critique of A1: Give the net the evidence it can't see

- verdict: RULE-BREAK
- fatal flaws: The proposed training arm changes the set posterior and adds action history together. The oracle test cannot decide whether either is worth training: a frozen policy can ignore unfamiliar truthful inputs, and even perfect set knowledge does not reveal an opponent’s action history.
- fixable flaws and the fix: Start with speed-order evidence alone. Zeroing new columns preserves the old function only if the existing posterior inputs also stay unchanged. Repeating a move does not prove Choice lock.
- evidence check: `consistent_revealed` explicitly omits speed and damage evidence, but `damage.py` already supplies an estimated probability of moving first. The missing signal is **observed order**. `fp4-07` says “inference ceiling +2.9 ± 3.9” for gen4 search; `fp4-10` also changed the evaluator. Neither bounds a gen9 net. `aux_fields` has no exact-stat target, and fastenv leaves hidden `aux_tera` at −1. Copying the aux tensors cannot construct Arm O.
- better kill test: On held-out battles, test whether public, equal-priority races improve prediction of the next comparable race. If they do, train identical candidate/control forks differing only in that posterior update. Use a preregistered duplicate-pair gate, then confirm against independent search. Treat a frozen oracle result as diagnostic only.
- my estimate of the effect: +0.7 points [−0.5, +2] for speed evidence alone, raw gen9 vs FP at 200k.
- rank among all proposals here: 2

### critique of A2: Purify the deployed policy

- verdict: RULE-BREAK
- fatal flaws: The campaign rules explicitly say “Evaluate raw networks at T 0.5.” A colder policy cannot replace the raw headline under that rule. A separately reported deployment-controller experiment is a defensible question.
- fixable flaws and the fix: `spd_h2h` and `v2.evaluate` already accept temperature. Preselect one temperature; selecting the best of three nominal 95% intervals inflates the false-positive rate. If testing thresholding separately, specify a fallback when every legal action falls below 0.1.
- evidence check: `v1-19` is “52.3 [50.9, 53.7]” for T 0.5 against T 1, not for further cooling. The cooldown’s 0.35 is a **stop guard**, not a loss pushing probability onto weaker moves. The [purification paper](https://ifaamas.org/Proceedings/aamas2012/papers/3E_4.pdf) reports nonmonotonic results even in Leduc poker. A fixed opponent can punish the modal move without adapting.
- better kill test: Keep the official raw benchmark. For a separately registered controller, test T 0.25 against T 0.5 and against independent search. Confirm on fresh matched FP pairs, verifying team identities. A positive estimate with a lower bound of −1.5 establishes neither improvement nor generality. The proposed three screens total 30,000 games, roughly six CPU-hours at the brief’s data-host rate.
- my estimate of the effect: +0.2 points [−1.5, +1.5], gen9 vs FP at 200k.
- rank among all proposals here: 6

### critique of A3: Tactical sparring partners in the PPO league

- verdict: PROMISING-IF-FIXED
- fatal flaws: None from engine provenance. The brief explicitly permits poke-engine. Playing against our own search without copying or modelling FP is admissible.
- fixable flaws and the fix: The measured opponent and proposed trainer are different players. Four sampled worlds at one second each are not a 20 ms search; a single **true** world also lets the sparring partner counter hidden moves it could not know. Use a small public-information search and measure its actual strength and collector throughput.
- evidence check: `g9q-23`: “55.5 [52.7, 58.3]” against **4 × 1000 ms** search. `dg-08` gives the short-game count directly: 341/1000. Final game length depends on both players’ decisions, so the 11.2-point split is not a recoverable training budget. `v2.evaluate` does log turns. Unlike `bc-12` and `ns-27`, this proposal supplies an opponent rather than imitation labels.
- better kill test: First benchmark the cheap searcher and measure achieved episodes per opponent, steps/s and CPU time. If viable, fork only the pool substitution. Decide on a held-out search configuration and performance against an unexposed net; keep the training opponent as a diagnostic. Do not kill on a retrospective length cutoff.
- my estimate of the effect: +0.6 points [−1, +2.5], raw gen9 vs FP at 200k.
- rank among all proposals here: 5

### critique of A4: Lookahead2 against the predicted opponent

- verdict: RULE-BREAK
- fatal flaws: The proposed sequential Showdown slot test conflicts with the campaign’s fixed-n slot rule. The claimed safety bound is also false: it bounds the *reported score of a fixed action* as the assumed opponent changes. It does not bound what a real opponent can win after our selected action changes. An arbitrarily small perturbation can break a tie in favour of an exploitable pure move.
- fixable flaws and the fix: Call this a reply-mixture heuristic. [piKL](https://arxiv.org/abs/2112.07544) regularizes the optimization objective; A4 interpolates after solving. `deepen.opp_prior` splits one class’s mass among indistinguishable switches and gives unplaceable options a fallback. Check that the head predicts held-out opponents better than a simple legal prior.
- evidence check: `la2-len1` reports “short −3.0 ± 7.7”; `ns-20` reports “+1.3 [−2.71, 5.31].” Neither establishes that conservative replies cost wins. The gen4 root-game failures are relevant cautions, not duplicates of this experiment.
- better kill test: Identify roots where the mixture changes the played action. Use paired simulator continuations against frozen independent policies to check the direction of those changes. Then run fixed-n Showdown pairs against stock LA2. Matching override frequency alone does not match override quality.
- my estimate of the effect: +0.2 points [−2, +2] over stock LA2 vs FP at 200k.
- rank among all proposals here: 7

### critique of A5: Move the gen4 bot onto the v2 net plus lookahead2

- verdict: PROMISING-IF-FIXED
- fatal flaws: None for the raw-net rebaseline, which is useful even if the port never happens. Replace the proposed sequential slot stage with fixed n to comply with the campaign rule.
- fixable flaws and the fix: “Parity at worst” has no basis. There is no measured 0.5 conversion factor from gen9 head-to-head gains to gen4 FP gains. The T4 cooldown costs about 3.5 hours of occupied capacity; run it inline or elsewhere and account for displaced training. `v3leaf.fresh_opp` hardcodes gen9 in both Pokémon construction and tensor allocation.
- evidence check: `v2g4-15`: “55.2 [53.2, 57.2]” **against the 300M ancestor**. `v2g4-10`: “55.6 [51.5, 59.7]” against FP over 400 games. `fp4-04`: “59.19 [56.95, 61.43]” at a time budget. `v2g4-11/12` tested MCTS priors, so this is not a duplicate.
- better kill test: Before porting, run the cooled raw net and incumbent on the same fresh gen4 FP200 pairs; verify iteration counters and report the paired difference. Decide from the gap and its interval, not whether one noisy estimate clears 55%. If competitive, require gen4 leaf parity and test LA2 against its own raw net before testing adoption.
- my estimate of the effect: −1 point [−5, +4] for the full replacement versus the incumbent, both remeasured against gen4 FP at 200k.
- rank among all proposals here: 4

### critique of C1: Spec v2.2 state-fidelity and battle-evidence encoder repair

- verdict: RULE-BREAK
- fatal flaws: Five independent changes enter one treatment. A win cannot identify the useful repair; a loss could bury one. The owner-visible Tera fix is the strongest individual experiment here.
- fixable flaws and the fix: Start with B1 alone. Match requests by `ident`, not list index; knowing the Tera type must not set the “already Terastallized” flag. Test aux labels and removed items separately. Trick can give a replacement item; distinguish an unknown item, an empty slot and a newly held item. Never hard-mask moves merely because Choice lock is plausible.
- evidence check: `encoder_v2.js:28–33` confirms the own-Tera, online aux-Tera, removed-item and weather behaviours. But `damage.py` clears the hypothetical attacker’s **Tera flag**, not its alive flag: `active_tera_vs` can contain ordinary damage, not identically zero. Supervised `aux_targets_from_team` can already supply hidden Tera. `bc-01` compared whole architectures. [MissingNo.Zero](https://nickgreenquist.com/blog/projects/2026/11/02/missingno-zero-self-play-pokemon-showdown/) attributes its 58.4→72.0 result to retraining “with a focus on sleep and long games,” not an isolated encoder patch.
- better kill test: Verify request truth on both sides and absence of opponent leakage. Train owner-Tera-only versus unchanged forks with frozen-opponent inputs held fixed. At 1.55k steps/s, 70M steps already cost 12.5 A40-hours before overhead. Replicate a win with a second training seed before expanding the spec.
- my estimate of the effect: Owner Tera alone: +1.5 points [−0.5, +4] raw gen9; +0.8 [−0.5, +2] over LA2 vs FP200. No gen4 Tera gain.
- rank among all proposals here: 1

### critique of C2: Bilateral forced-switch resolution and world-consistent leaves

- verdict: RULE-BREAK
- fatal flaws: Replacement search and reveal repair change together. Maximizing our replacement independently in each world lets us choose different switch-ins after **identical observations**. A K=32 judge making the same mistake cannot detect it.
- fixable flaws and the fix: Group branches by what we actually observe, then choose one replacement policy per group. Test one-sided replacements before double KOs. A pivoting Pokémon is generally alive; a fainted one is not. Forced switching is a real decision the net trains on. `own_act_tok` feeds the policy; the value head reads the field and pooled mon tokens.
- evidence check: `expand_world` really does expand opponent-only replacements. `ns-10`’s transition study found mean absolute value error “0.043” with reveals versus “0.003” without: direct evidence for a separate reveal test. `la2-len1`’s short-minus-long contrast was “−5.9 [−15.0, 3.2],” not confirmation of a replacement defect.
- better kill test: Test publicly announced move reveals alone first. For replacements, compare old and new action rankings against paired continuations from fresh forced-switch positions. Record branch counts and decision latency: if 20% of leaves expand fivefold, the increase is 80%, before double replacements. Proceed to fixed-n slot play only if rankings improve within the time budget.
- my estimate of the effect: +0.5 points [−1, +2] for information-correct replacement search alone over LA2 vs gen9 FP200. Do not add B2’s forecast.
- rank among all proposals here: 3

### critique of C3: Omniscient critic and HL-Gauss value loss

- verdict: RULE-BREAK
- fatal flaws: This changes privileged information, value loss **and** GAE λ. G9q uses **0.754**, not 0.95. Two encoded observations are not a Markov state simply because they contain both teams: encoders discard history, timers can be ambiguous, and a cached view can predate a forced switch.
- fixable flaws and the fix: Test privileged information alone, with the existing BCE head, λ and reward scale. The trainer uses signed values for GAE and converts returns to probabilities for BCE; substituting P(win) directly into the signed residual is wrong. Frozen pool opponents and heuristics also do not furnish a current learner-trunk representation for free.
- evidence check: `dg-06`’s “3.8 [3.5, 4.3]” is the **value/policy gradient-norm percentage**. Low gradient coherence was a separate observation; it does not identify hidden-team variance. `g9q-10` found λ 0.5 harmful, without establishing why. [HL-Gauss](https://arxiv.org/abs/2403.03950) principally contrasts classification with squared-error regression; our baseline already uses cross-entropy. An action-independent privileged baseline can be valid without making approximate bootstrapping unbiased.
- better kill test: Fit public and privileged scalar critics on identical frozen trajectories. On held-out games compare return error and variance of the actual score-weighted gradient, including forced switches and pool games. Only then run a critic-only PPO fork. Two 55M arms need about 19.7 A40-hours before extra forwards or storage.
- my estimate of the effect: +0.4 points [−1, +2] for a privileged critic alone, raw gen9 vs FP200.
- rank among all proposals here: 8

### critique of C4: Information-set positive-regret MCTS with a cached leaf server

- verdict: RULE-BREAK
- fatal flaws: The solver, information model, stopping rule and inference implementation all change. The latency arithmetic also fails: 8,192 leaves for each of three concurrent decisions in 0.35 s requires **70,217 aggregate leaves/s**, before overhead. Passing 35,000 cannot validate that deployment target.
- fixable flaws and the fix: Test inference first without writing a new tree. Static ID features are already cached by `model.py`; HP, boosts, weather, items and reveals can invalidate damage features. Sharing our node across worlds does not prevent an opponent response from depending on our private bench. The proposed surprise cutoff also watches the wrong side for that problem: Jaxcalibur truncates opponent search when **its own bot** does something surprising.
- evidence check: `ns-12`: “7,465” versus 35,300 leaves/s; at the proposed B=256, contended FP16 achieved **5,751**. The graph path already disabled trimming but failed on an unpinned copy; no graph speedup was measured. [Lisý et al.](https://proceedings.neurips.cc/paper/2013/hash/1579779b98ce9edb98dd85606f2c119d-Abstract.html) require no-regret selection and exploration in perfect-information simultaneous games. The proposed score is a plausible Jaxcalibur heuristic, not a convergence proof.
- better kill test: Repair graph capture only. Measure full encode/copy/forward latency with three concurrent decisions under slot-equivalent contention; check policy and reply flips as well as value error. A four-vCPU VM is useful for debugging, not certification. If the gate passes, test shared nodes with the solver held fixed.
- my estimate of the effect: Conditional on meeting latency, +0.5 points [−3, +4] over stock LA2 vs gen9 FP200. No support for stacking to 78%.
- rank among all proposals here: 11

### critique of C5: Damage-featured incremental Rust NNUE

- verdict: RULE-BREAK
- fatal flaws: Features, training distribution, loss, capacity and teacher all change; subsequent matches also change the gen4 prior or add C4. The four-world bag loss can fit the correct **mean** with wrong, cancelling values in individual worlds. MCTS consumes those individual values.
- fixable flaws and the fix: Determinized training is a substantive reason to revisit `ns-16/17`. Test it first with the old architecture, features and teacher. Then add damage features alone. A damage cache keyed only by mons, boosts, weather and Tera can reuse incorrect results after item removal, status, screens or HP-dependent power changes.
- evidence check: `ns-16`: “student AUC 0.714” versus “evaluate() 0.718 / teacher 0.748”; `ns-17`: “student AUC 0.796” on true states. That difference does not diagnose overfitting to species. `features.py` already includes types, stats and field features, though not explicit damage matchups. The “exact” old gate required **both AUC and calibrated log loss on S and H**; C5 drops criteria. Stockfish’s incremental speed does not price Pokémon damage calculations.
- better kill test: Use fresh held-out games, game-clustered intervals and within-root action rankings. Benchmark real Rust apply/evaluate/reverse traces, including cache misses, against the incumbent at equal wall time. Keep its prior unchanged. The brief assigns search matches to Showdown; 1,000 engine pairs on the A16 are not a one-hour substitute.
- my estimate of the effect: Gen4 +0.5 points [−2, +3] over fitted search; gen9 +0.5 [−2, +2.5] over stock CPU search, both vs FP200.
- rank among all proposals here: 9

### critique of C6: Online Gumbel completed-Q policy improvement

- verdict: WEAK
- fatal flaws: The proof improves expected **estimated Q**, not win probability. A wrong one-turn ranking becomes more influential. Six actions × four worlds gives 24 **before** opponent replies, chance branches and replacements. The proposed 15% overhead is unsupported.
- fixable flaws and the fix: Define the sample: a fixed uncertainty threshold does not select exactly 5%, and uncertainty about who wins does not imply uncertainty about which move is best. Keeping every move’s probability positive does not keep a useful mixed strategy. Specify whether the auxiliary loss averages over selected states or over the rollout; the effective gradient differs twentyfold at 5% coverage.
- evidence check: `ns-27` found “48.17 [47.47, 48.87]” versus the matched control. New soft targets justify a different test, but none of the four explanations for that loss was isolated. The [Gumbel paper](https://iclr.cc/virtual/2022/poster/6418) does not validate noisy one-turn Bayesian scores. At 5% of 50M steps, 2.5M labels at the prior ExIt throughput of 3.5 decisions/s would take **about 198 hours just for labels**. C6 must measure its cheaper teacher.
- better kill test: Use stored ExIt `p` and `Q` for a cheap target comparison. On fresh, player-visible tactical roots, compare the proposed action ranking with paired terminal continuations against independent policies. Stop if the ranking adds no value or if measured throughput misses the training budget. Only then test the auxiliary loss online.
- my estimate of the effect: 0 points [−2, +1.5], raw gen9 vs FP200.
- rank among all proposals here: 10

## synthesis from Member B

- best combination: The ranks order the next test worth funding from each proposal, after the stated repairs. Start **B1/C1 owner Tera only** on the A40 and **B2/C2 announced-move reveals only** on the search helper. Each has an observable ground truth and a distinct control. Use the first available gen4 slot for **A5’s matched rebaseline**, before porting anything. Next test **A1 speed evidence alone**. Use **B4 paired terminal continuations** to judge disputed tactical rankings before funding A4, C2 replacement search or C6. Combine independently successful changes only by testing the second on top of the first.
- what everyone missed: Fresh seeds from the same October team generator do not test transfer to a new month’s sets. That matters most for the leading ideas: Tera ambiguity, speed inference and opponent priors all depend on which sets are common. Reserve an untouched monthly distribution for evaluation, rebaseline both arms together, and change no running training lineage. This tests general strength without learning anything about Foul Play’s decision rule.