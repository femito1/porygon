# Literature review protocol

**Question.** Has anyone (a) fitted or learned the leaf evaluation of a search-based Pokémon battle agent, or
(b) fitted a search evaluation to game outcomes in a closely analogous game (simultaneous moves, chance, hidden
information searched by determinization)? Secondary: the full set of Pokémon battle AI work for related work, and
prior negative results on PUCT / regret matching / root solving / value lookahead in such games.

**Sources.** OpenAlex, Semantic Scholar, arXiv (via `docs/paper/tools/litsearch.py`), plus direct reading of key
pages (arXiv HTML, author blogs, GitHub READMEs, theses). Web search engines were unavailable (Tavily over its
limit, built-in search failing), so grey literature is covered by snowballing and direct fetches only — a stated
limitation.

**Inclusion.** Any work that (1) builds an agent for Pokémon battles (any method, any format), or (2) learns or
fits an evaluation / value function used *inside a search* by regression on outcomes, self-play, TD, or search
targets, in any game, prioritising stochastic, simultaneous-move or imperfect-information ones, or (3) evaluates
search enhancements (PUCT priors, regret matching, ISMCTS / determinization fixes, root equilibrium solving) in
simultaneous-move stochastic games.
**Exclusion.** Pokémon RPG / overworld agents (PokeAgent speedrun track, Pokémon Red RL), unless they also battle;
pure LLM-chat demos without a battle agent; evaluation learning without search unless it is the standard reference
for the method (TD-Gammon).

**Query blocks** (each run on all three sources; title/abstract screening of every hit):
- P (Pokémon): "Pokémon battle AI", "Pokémon Showdown", "Pokémon battle agent", "Pokémon reinforcement learning
  battle", "Pokémon Monte Carlo tree search", "Pokémon minimax", "Pokémon VGC AI competition", "Pokémon random
  battles", "competitive Pokémon language model agent", "Pokémon team building AI", "Pokémon game theory".
- E (evaluation fitting): "evaluation function tuning game outcomes logistic regression", "Texel tuning",
  "statistical feature combination evaluation game positions", "learning evaluation function self-play search",
  "TreeStrap bootstrapping game tree search", "evaluation function learning Monte Carlo tree search", "state
  evaluation function real-time strategy logistic regression", "learning leaf evaluation MCTS without rollouts".
- G (analogous games): "simultaneous move Monte Carlo tree search evaluation", "determinization MCTS learned
  evaluation card game", "Hearthstone MCTS evaluation function learning", "Skat state evaluation inference search",
  "Dou Di Zhu search evaluation", "information set MCTS stochastic game evaluation", "microRTS evaluation function
  learning", "StarCraft combat state evaluation".
- S (search enhancements): "decoupled UCT simultaneous move games", "regret matching simultaneous move MCTS",
  "PUCT policy prior imperfect information search", "strategy fusion determinization".

**Snowballing.** Backward (references) and forward (citations) from every included Pokémon work and from the
evaluation-fitting anchors: Buro 1995 (Logistello), Gomboc–Buro–Marsland 2005, Veness et al. 2009 (TreeStrap),
Erickson & Buro 2014 / Stanescu et al. 2016 (RTS state evaluation), Buro et al. 2009 (Skat), Ihara et al. 2018
(ISMCTS for Pokémon), Karten et al. 2025 (PokéChamp), Grigsby et al. 2025 (Metamon), Karten et al. 2026 (PokeAgent
Challenge, arXiv 2603.15563), Wang 2024 (MIT thesis). Repeat on newly included Pokémon works until no new ones.

**Record.** Every hit is screened include / maybe / exclude with a note in the corpus (`litsearch.py screen`);
the result is `review.md`: the novelty verdict, the closest prior work and how ours differs, and the related-work
map, with counts (hits, screened, included).
