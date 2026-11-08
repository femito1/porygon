# Reading notes (full-text reads; screening of index hits is in the litsearch corpus)

## Pokémon battle agents

- **Karten et al. 2026, "The PokeAgent Challenge: Competitive and Long-Context Learning at Scale"**, arXiv
  2603.15563, NeurIPS 2025 Competition Track (31 authors incl. Patrick Mariglia, Foul Play's author). OU formats only
  (Gen1 OU, Gen9 OU); no random battles. Foul Play (Appendix): root-parallelized MCTS with DUCT, damage-roll
  grouping, no rollouts, "search is guided by a custom evaluation function"; likelihood-weighted set sampling with
  inference from damage, move order, weather duration, item evidence. Results quoted: >90% GXE gen9 random battles
  (peak 2341), >80% GXE gen9 OU (peak 1879), won Gen9 OU track 50-14. **No description of how the evaluation was
  made — no tuning mentioned.** Other agents: PokéChamp (minimax + LLM evaluation, Karten et al. ICML 2025),
  Metamon (Grigsby et al., RLJ 2025, offline RL), PA-Agent, 4thLesson, Team Q (extend RL baselines), Porygon2AI
  (independent RL; name clash with our planned Porygon2 — note).
- **Mariglia 2025, Foul Play blog post** (pmariglia.github.io/posts/foul-play/): set inference and sampling in detail;
  **evaluation function not discussed at all**; ladder table (gen9randombattle 88% GXE, peak 2341; gen9ou 80% GXE,
  1879); cites DUCT (Maastricht). Code reading (ours): Foul Play uses poke-engine's stock hand-written evaluate().
- **Sarantinos 2022, "Teamwork under extreme uncertainty: AI for Pokémon ranks 33rd in the world"**, arXiv
  2212.13338. Gen7 random battles, peak #33 on the Showdown ladder. 2-turn look-ahead over all joint moves; terminal
  states scored by one-vs-one matchup look-aheads with a hand-written HP-based score; payoff matrices + an opponent
  model trained by a "regret minimization machine learning algorithm" on the opponent's move history; usage-stats
  set inference; forward pruning of dominated strategies; distributed transposition table. **Hand-written
  evaluation, not fitted.** Argues outcome-based RL is hard because of luck (cf. our raw-vs-luck-adjusted result).
  Related work it lists (mostly grey literature, to find and check):
  - Ho & Ramesh, "Percymon: A Pokémon Showdown Artificial Intelligence", Stanford CS221 report — minimax +
    handcrafted evaluation.
  - **"Khosla et al." (Stanford), expectiminimax with an evaluation learned by TD learning instead of a handcrafted
    one; concluded handcrafted evaluation is a poor choice.** Sarantinos's reference [14] duplicates Kalose et al. —
    the real citation must be found. Closest prior art for "learned evaluation inside Pokémon search".
  - Kalose, Kaya, Kim 2018, "Optimal battle strategy in Pokémon using reinforcement learning", Stanford AA228.
  - Chen & Lin 2018, "Gotta Train 'Em All", Stanford CS230 (deep RL).
  - Sanchez, "Learning complex games through self play — Pokémon battles", UPC thesis.
  - Norström, "Comparison of AI algorithms for Pokémon battles", Chalmers thesis (MCTS, simplified gen1).
  - Huang & Lee 2019, "A self-play policy optimization approach to battling Pokémon", IEEE CoG.
  - Lee & Togelius 2017, "Showdown AI competition", IEEE CIG.
  - Dotson, "Future Sight AI" (website, no method); Technical Machine (David Stone, C++ bot); pmariglia/showdown.
- **Ho & Ramesh, "Percymon: A Pokémon Showdown Artificial Intelligence"**, Stanford CS221 report (c. 2016;
  github.com/rameshvarun/showdownbot). Gen6 random battles, ladder play. Depth-2 minimax with alpha-beta and move
  ordering, Showdown's own simulator (so no chance nodes: "eliminates the possibility for an Expectiminimax tree").
  Evaluation: weighted state factors, **weights "borrowed from Technical Machine ... and manually tweaked after
  observing the bot's behavior"** — hand-set.
- **Technical Machine** (David Stone, C++; gens 1-4 most complete). documentation/battling.md: evaluation considers
  only "remaining HP%, whether the Pokemon has been revealed, and entry hazards"; **"The existing weights were picked
  arbitrarily and then tuned once to make Stealth Rock less valuable, but otherwise have remained the same since
  Technical Machine's inception."** Search: depth-limited minimax extended to simultaneous moves — a probability for
  each opponent selection, best expected response — then 1v1 matchup evaluation of what remains.
- **PokéChamp** (Karten, Nguyen, Jin, ICML 2025, arXiv 2503.04094): minimax with an LLM as the value function, gen9
  OU; heuristic baseline = one-step look-ahead on turns-to-KO. Related work cites only Huang & Lee 2019 and a
  Whidden 2023 blog among Pokémon agents; no comparison with Foul Play or poke-engine.
- **Yu 2026, "PokaiTrainer: Scaling Equilibrium Search to Competitive Pokémon VGC"**, arXiv 2608.29197 (independent
  researcher). VGC doubles (Champions Reg M-B), open team sheets only. Own Rust engine; CFR on a Bayesian matrix game
  over public belief states, budgeted PUCT expansion (Student of Games style). **Leaves: a learned counterfactual
  value network** (256-wide, 6-layer, 5.4M params), initialised by behaviour cloning on 122,804 human replays,
  trained by self-play with targets (1-λ)·solver value + λ·realized outcome. Hidden stat spreads as a mixture inside
  each world, Bayesian-updated with the engine. Results: 59% of 150 Bo3 sets vs ~1320-Elo humans (Wilson 51-67%),
  peak 1492. Decoupled UCT substituted for CFR plays at parity (51.7%, 2048 games) — consistent with our regret
  matching / root-game null results. Greedy material-heuristic search beats every raw policy head (71%); "almost
  every training-side lever lands within noise". **No comparison with Foul Play / poke-engine / random battles.**
  => "first learned evaluation inside a Pokémon search agent" is NOT our claim to make. Ours differs: fitting the
  hand-written evaluation of an existing engine (64 features, logistic, minutes of fitting), singles random battles
  (closed information), against the strongest public bot, drop-in for poke-engine/Foul Play.
- **Ihara, Fujii, ... 2018, "Implementation and Evaluation of ISMCTS for Pokémon"**, IEEE SMC (10.1109/smc.2018.00375):
  own simplified simulator; ISMCTS beats determinized MCTS. Read full text for its evaluation (rollouts?).

## Grey literature: the Foul Play / poke-engine issue trackers (GitHub API search; read in full 2026-11-08)

- **foul-play #89 "Tuning evaluation function"** (2021-12-14/15). Mariglia (Foul Play's author): "I've explored
  evaluation function tuning a bit on a private branch"; tried **Texel's tuning method** and **Stockfish's tuning
  method** (SPSA-style self-play). "Tuning via self-play is extremely slow ... When you account for the randomness of
  a game like Pokemon it's almost certainly going to be much more than [100,000 games]"; Stockfish's method: "Too
  slow, and the variables just took a random walk"; Texel: "might be feasible", but "Running all of those battles
  would probably take several weeks ... Then minimizing the error would probably take even longer"; intuition: "A
  static evaluation function just isn't enough to properly play competitive Pokemon ... my gut tells me that any
  parameter optimization technique for Pokemon just isn't going to have good results ... Again very happy to be
  wrong." CLOP suggested by the issue author, not tried. URL:
  https://github.com/pmariglia/foul-play/issues/89#issuecomment-994189305
- **foul-play #262** (2025-07-17), on boost wars: Mariglia: "The evaluation function in poke-engine is the brains of
  search ... it is mostly based on 'vibes' to begin with so its probably not ideal, but any change made should be
  rigorously tested"; points at the boost multiplier table's drop-off from +4 to +5 as the place to start. (Our fit
  finds boosts overvalued, and boost value dependent on the attacker's category.)
- **foul-play #218** (2025-01): the MCTS bot uses poke-engine's evaluate.rs, "largely the same as" the earlier
  expectiminimax one; mcts_parallelism = the random-battles multi-sample bot.
- GitHub repository search for poke-engine / Showdown evaluation tuning (5 queries): no repository doing it.
  => Texel-style tuning of this evaluation was attempted privately by its author and judged infeasible / unlikely to
  help; nobody has published it. Our engine-only self-play (40k games in under 2 h on 48 cores) removes the cost
  objection, and the result answers the "static evaluation isn't enough" intuition (+4.7 vs Foul Play).
- **Wang 2024, "Winning at Pokémon Random Battles Using Reinforcement Learning"**, MIT MEng thesis (DSpace
  1721.1/153888; local copy ~/Downloads/wang-jett-meng-eecs-2024-thesis.pdf). **gen4randombattle — our format.**
  PPO actor-critic by self-play, then MCTS at inference only: rollouts in Showdown's simulator stop at a leaf scored
  by the **learned value function Vθ** ("Stopping at leaf nodes is made possible because we have Vθ, a trained state
  value estimator"), PUCT on the policy prior, opponent modelled by the policy, unknown opponent info sampled per
  trajectory by rejection sampling from Showdown's own team generator; 1000-2000 rollouts in 10 s on 20 workers.
  Rank 8 / 1693 Elo on the gen4 ladder. **So a learned leaf evaluation inside a random-battle search already exists,
  in our format.** Related work: Sarantinos, Future Sight AI, Huang & Lee only.

## Grey literature via the `websearch` tool (2026-11-08; Exa/Brave/… meta-search, user-provided)

- **Khosla, Lin & Qi** (Stanford CS221/CS229 project, c. 2016-17; original report not located): expectimax with an
  **evaluation function trained from 20,000 replays by TD(λ)**, Elo 1344. Source: Kalose, Kaya & Kim 2018 (AA228
  report, web.stanford.edu/class/aa228/reports/2018/final151.pdf), lit review. (Sarantinos's reference to it is wrong.)
- **van der Heijden 2022**, "Nash equilibrium estimation in competitive Pokémon using search and supervised
  learning", Radboud BSc thesis: search + **evaluation network** (gen8 OU data, 74.0% outcome accuracy) → per-turn
  payoff matrix → Nash. Vs Mariglia's heuristic expectiminimax: same heuristic evaluation 17.3%, learned evaluation
  29.5% (learning helped; both lose).
- **randbot-web** (ayuan153, GitHub; random battles browser assistant): depth-2 expectiminimax with an ONNX MLP leaf
  evaluator (206 features, 65% win-prediction accuracy) trained on 500K+ games; heuristic fallback.
- **AlphaMon** (davidemodolo, GitHub): doubles; GPT-style policy + win-probability head from replays, Monte Carlo
  rollout search.
- **Future Sight AI** (aed3, 2021; aed3.github.io): ranks lines by a learned win-probability model (its "Pokemon
  Battle Predictor", TensorFlow.js, 2M+ battles) at the end of each turn; top 6.8% gen8 OU.
- **Jaxcalibur** (jaxcalibur.github.io; Smogon thread 2026-08-28; code and weights NOT released): **gen9 random
  battles**; 8.5M-param transformer, self-play PPO (~100M games, a week on one H100) on a pure-JAX battle engine;
  value = win-probability head; test-time pUCT adapted to simultaneous moves (regret-like target), rollouts from
  scratch per seed, 32 sampled worlds, depth 4, 20,480 rollouts. **Peak Elo 2557, GXE 95.3, #1 on the ladder ~20% of
  Aug 6-27**; "the strongest previous bot I'm aware of, Foul Play, peaked at around 2341 Elo and 88 GXE". Author:
  "most of Jaxcalibur's strength comes from having a really good neural net rather than from its search ... search
  adds 100-150 Elo over its base policy". => Foul Play is NOT the strongest known random-battles agent; it is the
  strongest *open-source* one.
- **poke-engine commit d0e6b34 "perf: gen1 evaluation tuning"** (v0.0.21, 2024-12-14): hand-changed status penalties
  (frozen -40→-80, asleep -25→-75, paralyzed -25→-40), justified by "57% winrate self-play in 200 games" (±7 points:
  not significant) — the project's evaluation is hand-tuned with small self-play tests.
- **Smogon "(Re-)Introducing Foul Play" thread, 2025-07-05** (page did not render; quote from the search index
  snippet, check before citing): Mariglia: "The evaluation function was made using trial and error just based on my
  understanding of certain formats & metagames. I do not think it is completely optimized. Trying to tune it
  algorithmically is really slow unfortunately and I haven't had very much success with that - though I'm sure it is
  possible."
- Ihara et al. 2018 full text available: eprints.lib.hokudai.ac.jp/repo/huscap/all/72345/ihara-smc2018.pdf.
