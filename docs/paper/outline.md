# Paper outline (draft)

Status markers: **[have]** result exists, with job IDs / commits · **[running]** · **[need]** not done yet ·
**[verify]** number quoted from notes, re-derive from the logs before it goes in the paper.

## Working title

- *Fitting the leaf evaluation of a determinized-MCTS Pokémon agent to self-play outcomes*
- alt: *Where the strength is: search, evaluation, inference and learned priors in Pokémon random battles*

Target: IEEE Conference on Games (short or full paper), or a NeurIPS/AAAI games-workshop paper; arXiv either way.
4–6 pages + appendix (negative results, benchmark protocol).

## One-paragraph pitch

Determinized MCTS on a fast simulator (poke-engine) is the strongest known approach to Pokémon Showdown random
battles; Foul Play, built on it, won the NeurIPS 2025 PokéAgent challenge (Gen9 OU). Its leaf evaluation is a
hand-written material score. We fit that evaluation to the outcomes of 40k engine self-play games (64 features,
logistic, fit within team pairs) and the same search goes from 54.5% to 59.2% against Foul Play in gen4 random
battles, at [compute-matched result]. We then measure, with sample sizes large enough to see 2–3 points, what
else does and does not matter: hidden-information inference is near its ceiling (perfect knowledge of revealed sets
is worth only +2 points over our inference), while network priors, PUCT, regret matching, root equilibrium solving,
value lookahead, more determinizations and more PPO training all fail to help.

## 1. Introduction

- Pokémon random battles: simultaneous moves, stochastic, imperfect information (opponent sets), ~10 actions.
- State of the art = determinized MCTS (Foul Play, poke-engine); learned agents (Wang 2024 PPO + MCTS, 1693 Elo
  gen4; Metamon, arXiv 2603.15563, offline RL on replays) are behind it or in other formats.
- Contributions:
  1. A fitted leaf evaluation: +4.7 ± 1.9 points vs Foul Play, compute-matched [running]; recipe and data cheap
     (one day of CPU self-play).
  2. Ablations isolating where the gain comes from (weights, not just the scale; raw outcomes suffice).
  3. A decomposition of the remaining gap: oracle information vs inference vs search knobs vs learned components,
     with negative results at adequate power.
  4. Benchmark protocol: why 400-game evaluations (common in this literature) cannot resolve these effects;
     duplicate pairing is worth only 1.1–1.4× the games.
  5. Open code (MIT): agent, engine patches (gen4 duration fix, tunable evaluation, search options), benchmark.

## 2. Background and related work

- poke-engine (MIT) and Foul Play (GPL-3.0; used only as an opponent, no code reused): decoupled-UCB MCTS over
  sampled opponent sets, leaf = sigmoid(0.0125 · (eval − eval_root)), no rollouts.
- Evaluation tuning in perfect-information games: Texel tuning (chess), TD-Leaf, TreeStrap; logistic fits to game
  outcomes. Our setting differs: simultaneous moves + chance + determinization.
- RL for Pokémon: Wang 2024 (thesis recipe we follow for PPO), Metamon / PokéAgent challenge, earlier heuristic bots.
- Imperfect-information search: determinization / PIMC and its known pathologies (strategy fusion), ISMCTS.

## 3. System

- 3.1 Search: N determinizations (sets sampled from the random-battle set library, posterior-weighted by evidence),
  each a 1 s poke-engine MCTS; visit shares summed, optionally mixed with a PPO policy prior (weight 0.25).
- 3.2 Inference: library-based set posterior; offline accuracy on true teams: set 59%, item 86–89%, move ~79%,
  ability ~97% [verify: scripts/score_inference.py]; evidence rules 100% consistent with true teams.
- 3.3 Engine corrections: gen4 taunt/encore/confusion durations (stock engine never ends them); neutral for
  strength (49.1% ± 4.1% vs stock engine) but correct.
- 3.4 Network: PPO self-play + opponent pool, Wang's hyperparameters with our deviations (entropy 0.01, light
  potential-based shaping), transformer over 12 mons + field, 3.5M parameters, 600M steps. Raw policy ~90% vs
  SimpleHeuristics. [have; one paragraph, it is not the contribution]

## 4. Fitting the leaf evaluation (core section)

- Features: 64 antisymmetric side-one-minus-side-two terms (stock score/100 as f₀; alive counts, HP, status,
  hazards, boosts, volatiles, screens, items …); evaluation = w · f, leaf value = sigmoid(w · (f_leaf − f_root)).
- Data: 39,823 engine self-play games (20k iterations/move, stock evaluation, team pairs played both ways),
  1,680,721 positions [have: runs/evalfit/r1/report.json].
- Model: logistic P(win | s) = σ(w · f), squared-probability loss (Texel), L2 on corrections toward the stock
  score, splits by team pair. Within-pair fixed effect (`full_fe`): weights identified only by variation inside a
  team pair, which is what a search compares.
- Held-out log loss: stock at engine scale 0.526 → fitted scale 0.522 → full 0.511 / within-pair 0.517 [have].
- Leaf scale: the stock 0.0125 is too soft (≈ too much UCB exploration); shipped config = within-pair fit at ×2.
  At 300k iterations ×2 / ×4 / ×8 are flat (58 / 56 / 57% vs stock, self-play) [have].
- Table: fitted weights vs stock, the largest corrections, interpreted (what the stock score over/under-values).
  [need: extract from fe_x2.json, plot]

## 5. Results vs Foul Play

Protocol: local Showdown, gen4randombattle, Foul Play 1 s/state, duplicate pairing (same seeds, teams swapped),
Wilson/normal 95% intervals, ≥1600 games per headline arm.

| Configuration (ours) | Foul Play | Games | Win rate | Status |
|---|---|---|---|---|
| stock eval, 4 × 1 s, prior 0.25 | default (2 states × 1 s, 1 core) | 3198 | 54.5% ± 1.5% | [have] |
| stock weights at ×2 scale | default | 1600 | 54.3% ± 1.2% | [have] |
| fitted eval, 4 × 1 s | default | 1600 | 59.2% ± 1.1% | [have] jobs 308845–8 |
| fitted eval, **2 × 1 s** (= Foul Play's 2 core-s) | default | 1600 | — | [running] cm-A, jobs 380872/4/6/8 |
| fitted eval, 4 × 1 s | **4 states × 1 s, 2 cores** (= our 4 core-s) | 1608 | — | [running] cm-B, jobs 380873/5/7/9 |
| stock eval, 2 × 1 s | default | — | — | [need] if cm-A drops: separates eval gain from compute |

- Head-to-head, no Foul Play: fitted vs stock search on Showdown 54.9% ± 1.6% (800) [have] — the gain is not
  Foul-Play-specific.
- Raw outcome labels vs luck-adjusted labels: raw better, 52.1% ± 1.1% in self-play [have] → method is plain.

## 6. Where the remaining strength is (analysis section)

- 6.1 Information: oracle sets of revealed mons 61.3%, whole true team 65.4% (vs 54.5% baseline) [have, jobs
  302020/1; rerun with fitted eval [need]]. Inference is near ceiling: the rest is generator randomness.
- 6.2 Search knobs that did not help (400 games each unless noted; table with ±):
  8 samples 53.6%; root PUCT c 2 → 55.0% (2 runs), 8 → 53.2%; Bayesian root game (max) 52.9%; regret matching in
  the tree 53.4%; sampled root game 40.8% (harmful: mixing in equilibrium play loses vs a non-exploiting opponent).
- 6.3 Learned components: prior weight 0.5 54.2%, 1.0 49.6%; network-valued one-turn lookahead 49.7%; 500M/600M
  vs 400M network no gain; +40M PPO steps (and boosted embedding learning ×30/×300) 50.6% / 48.5% vs 600M
  (4000 games each) [have]; value of the prior at 0.25 vs 0 [verify: 53.8% vs 46.2%, source run?].
- 6.4 Oracle labels from the same engine side with the search 82% of the time: same-engine oracles cannot judge a
  learned component (methodology note).

## 7. Benchmark methodology (short section or appendix)

- Power: a 3-point difference needs ~2000 games per arm; single 400-game runs of one config ranged 50.7–57.8%.
- Duplicate pairing (patched Showdown, same seeds, teams swapped): only 1.1–1.4× effective games — random-battle
  variance is mostly in play, not in team draws.
- Compute accounting in core-seconds per decision; Foul Play's sampling schedule (4 × 0.5 s early, 2 × 1 s later).

## 8. Limitations

- One format (gen4 random battles), one strong opponent; no public ladder rating (bot accounts are flagged;
  PokéAgent is OU-only). [need, optional: gen9randombattle replication; a second opponent]
- Self-play data generated with the stock evaluation; one fitting round (no iteration / TreeStrap yet).
- Determinization pathologies untouched.

## 9. Conclusion / future work

- Iterate the fit on its own self-play (round 2), leaf targets from deeper search (TreeStrap), SPSA on groups;
  port to gen9; network values inside the tree need Rust-side inference.

## Figures / tables to make

1. System diagram (inference → determinizations → MCTS with fitted leaf → move).
2. Fitted vs stock weights (bar chart, grouped by feature family).
3. Headline table (§5) with compute in core-seconds.
4. "Where the strength is" forest plot: every lever as Δ win rate ± CI vs baseline (§6).
5. Leaf-scale sweep (§4).

## To do before writing

- [running] compute-matched arms A and B (~14 h).
- [need] stock eval at 2 × 1 s if arm A loses more than ~2 points.
- [need] oracle runs with the fitted eval (to recompute the information gap at the new standing).
- [need] confirm which network checkpoint each headline run used (500M baseline vs 600M fitted runs?).
- [verify] every number marked above against logs; collect job IDs into an appendix table.
- [optional] gen9randombattle replication.
