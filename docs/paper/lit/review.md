# Literature review: what is and is not new

Scope: 1587 scholarly records (OpenAlex, arXiv, Semantic Scholar) screened, 82 included (`included.md`), plus the
grey literature (student reports, blogs, bots, GitHub issues) found with the `websearch` tool. Quotes, details and
URLs: `notes.md`. Extend with `../tools/litsearch.py` (`run_queries.sh` reruns the query set in `queries.txt`).

## Not new
- **Learned or fitted leaf evaluations inside Pokémon search agents:** Khosla, Lin & Qi (Stanford, TD(λ) on 20k
  replays, expectimax); Future Sight AI (learned win probability ranks lines); van der Heijden 2022 (evaluation
  network → Nash payoff matrices); Wang 2024 (MIT thesis, **gen4 random battles**, PPO value at MCTS leaves);
  randbot-web (random battles, MLP evaluator in expectiminimax); PokaiTrainer 2026 (VGC, value net in CFR);
  **Jaxcalibur 2026** (gen9 random battles, self-play transformer value in pUCT; peak 2557 Elo / 95.3 GXE, #1 on the
  ladder; closed) — the strongest known random-battles agent, ahead of Foul Play (2341 / 88).
- **Fitting an evaluation to game outcomes by logistic regression:** Buro 1995 (Othello), Texel tuning (chess),
  Erickson & Buro 2014 (StarCraft), Buro et al. 2009 (Skat), Stanescu et al. 2016 (CNN evaluation in RTS search);
  evolved evaluation weights: Miernik & Kowalski 2022, Neufeld et al. 2019.

## New, as far as found
1. Re-fitting the hand-set evaluation that the strongest open-source bots search with (poke-engine: Foul Play,
   ours) by plain logistic regression on cheap engine-only self-play, and beating Foul Play with it. Foul Play's
   author tried Texel and Stockfish-style tuning privately and judged it impractical (foul-play #89, 2021: SPSA
   "variables just took a random walk", Texel "several weeks"; "my gut tells me that any parameter optimization
   technique for Pokemon just isn't going to have good results. Again very happy to be wrong"), and calls the
   evaluation "mostly based on 'vibes'" (#262, 2025).
2. Identifying the weights from within-team-pair variation (a fixed effect per pair), matched to what a search
   compares; the fitting literature screened pools outcomes.
3. A powered, compute-matched benchmark against the strongest open bot (1600–3200 games per arm); prior Pokémon
   work reports ladder ratings or 100–500 games.
4. Powered null results for search enhancements in singles random battles (PUCT priors, regret matching, Bayesian
   root solving, value lookahead, more determinizations); consistent with PokaiTrainer's DUCT ≈ CFR (51.7%, 2048
   games) in VGC.

## How to frame the paper
Not "first learned evaluation for Pokémon search" and not "Foul Play is the strongest agent". Instead: the strongest
open bots search with a hand-set evaluation; learned values exist (Wang, Jaxcalibur, PokaiTrainer); the cheapest
learning step, a logistic re-fit of the existing features, which the engine's author expected not to work, gives
+4.7 points against Foul Play in gen4. Cite Foul Play #89 as motivation, Erickson & Buro / Buro (Skat) / Texel as
lineage, AIVAT for the luck-adjusted targets, Wang / Jaxcalibur / PokaiTrainer as learned-value search.

## Side notes
- "Porygon2AI" is taken (an independent PokeAgent Challenge entrant): rethink the planned Porygon2 name.
- Gaps: the Khosla–Lin–Qi report itself was not found (known from Kalose et al. 2018); Semantic Scholar was
  rate-limited throughout, so its coverage is thin.
