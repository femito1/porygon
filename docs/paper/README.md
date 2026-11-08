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
   report gen9 as is with that caveat.
3. **Network per baseline run:** the 54.5% baseline's network (500M) is known only from notes; runs don't log it.
4. **Verify by hand** the Smogon quote ("The evaluation function was made using trial and error…", thread
   "(Re-)Introducing Foul Play", 2025-07-05): it comes from a search-index snippet.
5. Compile; check the bibliography; add Jaxcalibur, PokaiTrainer, van der Heijden, Erickson & Buro 2014,
   Buro et al. 2009 (Skat), AIVAT; write the conclusion.
