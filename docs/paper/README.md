# Paper: fitting the leaf evaluation of a determinized-MCTS Pokémon agent

Target: IEEE Conference on Games (short/full) or a NeurIPS/AAAI games workshop, plus arXiv.

| File | |
|---|---|
| `paper.tex` | the draft (IEEEtran; `\todo{}` marks missing results). Not compiled yet: no LaTeX on the laptop or cluster, use Overleaf. |
| `fig_weights.py`, `fig_weights.pdf`, `data/feature_sd.json` | Fig. 1, the fitted corrections × feature SD (needs matplotlib) |
| `lit/review.md` | novelty verdict and how to frame the paper |
| `lit/notes.md` | quotes and details of every closely related work, with URLs, for related work and citations |
| `lit/included.md`, `lit/corpus.sqlite` | screened literature (82 included of 1587); `tools/litsearch.py` queries and extends it |

## Before submission
1. **Reframe introduction and related work** per `lit/review.md`: learned evaluations inside Pokémon search are
   not new (Khosla–Lin–Qi, Future Sight AI, van der Heijden 2022, Wang 2024, randbot-web, PokaiTrainer,
   Jaxcalibur), and Foul Play is the strongest *open-source* bot, not the strongest (Jaxcalibur). The contribution
   is re-fitting the hand-set evaluation the strongest open bots use, cheaply, which its author expected not to
   work (foul-play issue #89), with within-pair identification and a powered, compute-matched benchmark.
2. **Generation 9** is a partial replication (self-play +, Foul Play within noise). Either run gen4 with prior 0
   (stock vs fitted vs Foul Play, ~1600 games each) to see whether the gen4 gain needs the network prior, or
   report gen9 as is with that caveat. The gen9 pure-search number stands: the 2026-11-11 network + search test
   re-measured pure gen9 search (stock evaluation, 4 × 1 s) at 51.6% ± 2.8% vs Foul Play (1000 games), next to the
   draft's 51.7% ± 2.4%. The draft's "our network has no tera actions" now refers to the gen4 (v1) network only.
3. **Network per baseline run:** the 54.5% baseline's network (500M) is known only from notes; runs don't log it.
4. **Verify by hand** the Smogon quote ("The evaluation function was made using trial and error…", thread
   "(Re-)Introducing Foul Play", 2025-07-05): it comes from a search-index snippet.
5. Compile; check the bibliography; add Jaxcalibur, PokaiTrainer, van der Heijden, Erickson & Buro 2014,
   Buro et al. 2009 (Skat), AIVAT; write the conclusion.

## Network results: what they could claim (not in this paper)
The network-first results are a separate story from the fitted evaluation. Literature checks of 2026-11-11 set what
could be claimed as new; every item below needs its own pre-registered tests before a write-up.
1. **Network + search (gen9).** The hybrid (raw net at T 0.5, overridden by our search when its pick is better by
   more than δ = 0.0212 in root Q; ~22% of moves) beat pure search 55.0% ± 3.0% and Foul Play 56.2% ± 2.8%, against
   pure search's 51.6% ± 2.8% on the same team pairs: paired +4.6 ± 3.9 points (1000 games per arm, fixed n,
   `docs/netsearch/NETSEARCH-PREREG.md`). A margin rule between a learned policy and search is not ours
   (MissingNo.Zero); the target is 65–75% vs Foul Play, so this is a step, not a headline.
2. **Human data behind a self-play init.** Human-init RL beating scratch is not novel in general (AlphaStar, VPT,
   VGC-Bench in Pokémon), nor is BC strength rising with data. The narrow claim may be new: with the same PPO recipe
   and equal steps, the run from the init cloned from more human data (G9q from P3, 7.05M decisions) beat the run
   from less (G9p from P1, ≈ 0.9M): 61.1% [55.8, 65.9] at 25M and 58.2% [53.8, 62.3] at 50M steps. It is confounded and unfinished:
   - P3 is also all ≥ 1900-rated, so it is "more and better data" until the size-vs-quality arm (U7: as many
     decisions, rated ≥ 1500) reports (pending);
   - one RL seed per arm; the KL anchor to each init lasts until 50M;
   - it may not persist: G9q@112.5M vs G9p@338.6M is 49.4% ± 2.0%, and later step-matched points are not tested.
   Do not claim that human data is necessary (Jaxcalibur and MissingNo.Zero use none) or that the gap persists.
3. **Inputlog re-simulation.** The technique is not novel (Showdown's owner described it in 2022; Metagross and
   Pokecritic re-simulate random-battle inputlogs at the pinned commit). What may be new: a documented, validated,
   released dataset of exact first-person gen9 random-battle trajectories (170k games, ~9.6M decisions, both sides),
   and using it to measure the error of spectator-view inference. Never claim "first to use inputlogs".
4. **Benchmarks.** Foul Play stays the opponent. Metagross cannot serve: no released weights, no licence, and 44.0%
   [37.3, 50.9] against plain Foul Play in its author's own head-to-head.
