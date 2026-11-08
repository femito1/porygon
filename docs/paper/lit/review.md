# Literature review: novelty of fitting the leaf evaluation (2026-11-08)

Protocol: `protocol.md`. Tool: `../tools/litsearch.py` (OpenAlex, Semantic Scholar, arXiv; cached corpus
`corpus.sqlite`). Reading notes with quotes: `notes.md`. Screened lists: `included.md` (82), `maybe.md` (20).

## Counts

31 queries in 4 blocks (Pokémon agents, evaluation fitting, analogous games, search enhancements) on OpenAlex and
arXiv, Semantic Scholar where it was not rate-limited (it returned 429 on most requests; 94 of 1587 records come
from it), plus 7 targeted gap searches and forward citation chasing from 7 Pokémon anchor papers.
**1587 records, all screened: 82 included, 20 maybe, 1485 excluded** (keyword triage first, then title/abstract;
every included Pokémon battle agent read at abstract level, the closest ones in full text). Grey literature: Foul
Play's blog post, README and issue tracker (GitHub API search), poke-engine's issues, Technical Machine's docs, the
Percymon and Sarantinos reports, Wang's thesis, PokéChamp, PokaiTrainer and the PokeAgent Challenge paper in full.

Update: the user's `websearch` meta-search tool (Exa, Brave, …) then covered the grey literature: 12 more queries,
which found Khosla et al., van der Heijden 2022, Future Sight AI, randbot-web, AlphaMon, Jaxcalibur, the Smogon
Foul Play thread and the poke-engine changelog (notes.md, last section). Original limitation: general web search was unavailable (Tavily over its quota, the built-in search failing, DuckDuckGo
answering with a CAPTCHA), so student reports and blogs are found only through citations and direct fetches.
Backward citation chasing is thin for arXiv preprints (OpenAlex indexes few of their references; PokaiTrainer: 1 of
its references). One cited predecessor could not be located: "Khosla et al." (Stanford), cited by Sarantinos 2022
as expectiminimax with a TD-learned evaluation; its reference entry is a duplicate of another paper's.

## Verdict

**Not novel:** a learned or fitted leaf evaluation inside a Pokémon search agent, as such. At least seven exist:
Khosla, Lin & Qi (Stanford, TD(λ) on 20k replays, expectimax), Future Sight AI (learned win probability ranking
lines), van der Heijden 2022 (evaluation network + Nash payoff matrices), Wang 2024 (gen4 random battles, PPO value
in MCTS), randbot-web (random battles, MLP evaluator in expectiminimax), PokaiTrainer 2026 (VGC, value net in CFR),
**Jaxcalibur 2026 (gen9 random battles, self-play transformer value in pUCT; peak 2557 Elo / 95.3 GXE, #1 on the
ladder; unreleased)** — the strongest known random-battles agent, ahead of Foul Play (2341 / 88).
- Wang 2024 (MIT thesis, **gen4 random battles**, our format): MCTS whose leaves are scored by a PPO-trained value
  network, hidden sets sampled from Showdown's generator; rank 8 on the gen4 ladder.
- Yu 2026 (PokaiTrainer, arXiv, VGC doubles, open team sheets): CFR equilibrium search with a 5.4M-parameter
  value network trained by self-play on solver values mixed with outcomes.
- (Possibly) the unlocated Stanford TD-learning project.
- Outside Pokémon, fitting an evaluation to game outcomes by logistic regression is standard: Buro 1995 (Othello),
  Texel tuning (chess, grey literature), Erickson & Buro 2014 (StarCraft states, logistic regression on replays),
  Buro et al. 2009 (Skat: evaluation learned from game data, used for inference and search), Stanescu et al. 2016
  (CNN state evaluation in RTS search), Miernik & Kowalski 2022 (evolved linear evaluation, card game),
  Neufeld et al. 2019 (evolved weights inside MCTS).

**Novel, as far as this search can tell:**
1. Fitting the hand-written feature evaluation of the engine the strongest public bots use (poke-engine: Foul
   Play, our agent), by plain logistic regression on cheap engine-only self-play, and showing it beats Foul Play.
   No publication or repository does this. Foul Play's author tried Texel and Stockfish-style tuning privately and
   judged it impractical ("several weeks" of battles; SPSA "variables just took a random walk"; "my gut tells me that
   any parameter optimization technique for Pokemon just isn't going to have good results. Again very happy to be
   wrong", foul-play issue #89, 2021), and calls the evaluation "mostly based on 'vibes'" (issue #262, 2025).
2. Identifying the weights by within-team-pair variation (a fixed effect per pair), matched to what a search compares.
   No precedent found in the evaluation-fitting literature screened (they fit pooled outcomes).
3. A head-to-head benchmark against the strongest public bot at matched compute, with ~1600-3200 games per arm and a
   power argument; prior Pokémon agents report ladder ratings or 100-500 games (PokaiTrainer: 2048-game ablations,
   150 human sets).
4. The set of well-powered null results for search enhancements in Pokémon singles random battles (PUCT priors,
   regret matching, Bayesian root solving, value lookahead, more determinizations). Consistent with PokaiTrainer's
   finding that decoupled UCT plays CFR at parity (51.7%, 2048 games) in VGC.

**Framing change for the paper.** Not "first learned evaluation for Pokémon search", and not "Foul Play is the
strongest agent": Jaxcalibur (learned value, pUCT) is stronger but closed; Foul Play is the strongest *open-source*
agent, and it and every other released strong bot search with a hand-set evaluation; we show that the cheapest possible learning step, a logistic re-fit of the existing
features on engine self-play, which the engine's author expected not to work, gives +4.7 points against Foul Play,
and that the neural alternatives we tried (prior, value lookahead, more PPO) did not. Cite Wang 2024 and PokaiTrainer
as learned-value search; Erickson & Buro, Buro (Skat), Stanescu, Texel/Buro 1995 as the method's lineage; AIVAT for
the luck-adjusted targets; Foul Play #89 for the motivation.

**Side findings.**
- The name "Porygon2AI" is taken by an independent PokeAgent Challenge entrant: consider before naming Porygon2.
- Foul Play's own reported strength: >88-90% GXE, peak 2341 in gen9 random battles (blog post; PokeAgent paper),
  i.e. it is far stronger in gen9 random battles than in the gen9 OU challenge it won.
