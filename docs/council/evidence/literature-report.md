# Research report — network-first campaign (research stream)

> **Reference (2026-11-08).** The facts about other bots, simulators, data and recipes are still accurate. Two of its inferences turned out wrong: our v1 weakness was not the lr floor (constant lr made it worse) and not value-gradient domination (the policy gradient was 9–26× the value's). Of its §7 recommendations: #2 (aux heads), #3 (inputs), #5 (in-process env) and #6 (exact human data) are done; #1 (recipe) is done differently (lr 7·10⁻⁵, fp32, on-policy); #4 (current-month sets) is planned; #7 (network in search) and #8 (benchmark hygiene) are open. See `ORCHESTRATOR.md`.


Status: COMPLETE (2026-11-08). Sections: 0 headlines, 1 Jaxcalibur, 2 other agents, 3 simulators, 4 replay data, 5 recipes from other games, 6 lr/embeddings, 7 ranked recommendations, 8 things that do not matter, 9 open questions.
Convention: **[S]** = stated by the source (quote + URL);
**[I]** = our inference. Unicode math only.

---

## 0. Headline findings (read this if nothing else)

1. **Two independent self-play PPO bots, without human data and with small networks, are already far beyond
   our network — and one of them plays our format with our model size at our throughput.**
   - **MissingNo.Zero** (Nick Greenquist, Nov 2026, https://github.com/nickgreenquist/missingno-zero, report PDF
     in the repo, blog https://nickgreenquist.com/blog/projects/2026/11/02/missingno-zero-self-play-pokemon-showdown/):
     "a small policy-and-value network (about 3M parameters), trained with proximal policy optimization ... Our
     starting recipe came from Wang (2024)" [S]. Trained on Showdown's own simulator through poke-env [S].
     **gen4: 700M steps, ~6 days, one rented GPU, ~$230; vs Foul Play 61.6% (59.4–63.7) WITHOUT search,
     67.5% with search, 2000 games** [S]. **gen9: 800M steps (README later: 1.8B), 66.7% ± 1.7 vs Foul Play
     without search over 3000 games** [S]. #1 on the gen1/2/4/6/7 randbats ladders, gen9 #1 on 2026-11-07 at
     Elo 2421 [S]. Foul Play was fixed to the iteration count it reaches in its time budget on the author's
     laptop [S] — so not identical to our 4 × 1 s setting, but the same order.
   - **Jaxcalibur** (gen9): network alone "I'd guess around 2400 Elo" (author, Smogon thread page 1) [S] —
     i.e. the raw net ≈ Foul Play's peak (2341); search adds 100–150 Elo on top [S].
   - **[I] Consequence:** our 3.5M gen4 network losing ~75% to our search (which itself is only 59% vs Foul
     Play) is a *recipe* failure, not a capacity, throughput or algorithm (PPO) failure. A Wang-recipe 3M PPO
     net at 700M steps beats Foul Play 61.6% in gen4. The most likely culprits, in order: (a) our lr schedule
     decayed ~27× (to ~2·10⁻⁶; its mean over the run is 1/6 of peak) — large self-play projects keep the lr
     ~constant and drop it only at plateaus (§5); (b) the shared-trunk value gradient swamping the policy
     (MissingNo.Zero: "the value gradient into that trunk was about 50x the policy's. So the network got good at
     predicting who wins and bad at actually playing (it ignored things like status conditions and gen 9's Tera
     types), even though its win rates looked fine") [S]; (c) missing inputs/heads (damage calcs,
     opponent-action head) that Jaxcalibur credits.
2. **A fast env is not on the critical path for the first big win.** MissingNo.Zero ran at ~1.35k steps/s
   (gen4: 700M in 6 days) and ~1.2–2.3k steps/s (gen9) on real Showdown [I from their numbers] — the same order
   as our 2.1k. Its author: "I tried porting Showdown's rules to a faster engine for each generation. It was
   slow to build and never quite exact (one port skipped every battle with a Ditto, so the agent never learned
   to play it). Keeping Showdown's own simulator and just changing how training drove it was nearly as fast,
   and exact from day one." [S]
3. **Concrete targets for netv2 (gen9):** a net-only bot at ≥ 65% vs Foul Play is demonstrated (MissingNo.Zero
   66.7%); Jaxcalibur's net is ~2400 Elo. Our gen9 search is 52–53%. "Network beats our search" is a realistic
   first milestone, not a stretch goal.
4. **New public yardsticks exist:** Showdown opened a bot registry (2026-10-13) and made ladder games vs bots
   opt-in (2026-10-20; smogon/pokemon-showdown de1cf598c) [S, MissingNo.Zero report §4.4]; **pkmn.bot** is a
   bot-only ladder with a Gen 9 Random table (PokeCHAD V3 1572, MissingNo.ZeroBot 1438–1577 provisional,
   Claude.ai, spktrm's "headache", Cocoa) [S, https://pkmn.bot/]. This reopens README item 7 ("a public
   rating") and gives opponents other than Foul Play.

---

## 1. Jaxcalibur (gen9 random battles; closed source)

Primary sources: technical write-up https://jaxcalibur.github.io/ (read in full 2026-11-08); Smogon thread
https://www.smogon.com/forums/threads/jaxcalibur-a-gen-9-randbats-bot-that-reached-1-on-the-ladder.3787537/
(now login-walled; pages 1–2 read via Wayback captures 2026-09-28 / 2026-10-03; pages 3–6 not archived);
annotated replays https://jaxcalibur.github.io/replays/ ; self-play win rates https://jaxcalibur.github.io/win_rates/ .
Author anonymous (`jaxcalibur_author`; GitHub user `jaxcalibur` created 2026-07-31, 0 public repos). No paper.

### 1.1 Architecture [S unless marked]
- "a non-causal transformer with various output heads. Most of the model follows a standard pre-RMSNorm
  architecture. The main exception is a position-based mixture-of-experts setup where (for example) positions
  representing Pokémon use different MLP weights from positions representing moves, events, etc."
  → [I] i.e. per-token-type FFNs (not routed MoE): field / active / move / team / opp / event / matchup tokens
  each have their own MLP weights; attention is shared.
- **8.5M parameters.** "in my experiments, smaller networks trained for longer outperformed larger networks
  trained for fewer steps."
- Tokens (≈ 61 [I: 1+2+5(+1 Struggle)+6+6+16+24, matchup tokens join mid-network]):
  - 1 battlefield token (weather, hazards, etc.)
  - 2 active-Pokémon tokens ("stats, moves, damage calcs, etc.")
  - 5 candidate-move tokens ("Struggle gets its own extra token")
  - 6 own-team tokens; 6 opposing-team tokens ("masked if they aren't known yet")
  - **16 history tokens** "containing a history of the last 16 game events (moves/switch-ins)"
  - **24 matchup tokens**, "each containing information such as damage calculations for a pair of Pokémon. It
    matches each of its Pokémon to 4 of its opponent's (an efficiency trick to save 12 tokens, as usually a
    couple opponent Pokémon are unrevealed or fainted). These are concatenated to the current residual stream
    half-way through the network, partially for efficiency and partially so they benefit from fresh
    move/item/ability predictions."
    → [I] The damage calcs appear to use the network's own predicted opponent sets (from the first half of the
    network / the aux heads), then are injected mid-network. Which 4 opponents are chosen is not stated (asked
    on Smogon by Xuwu; no answer found).
- Action heads: "each team position emits a logit for switching to that slot. Each move position emits two
  logits: one for using the move and one for Terastallizing and using the move." → pointer-style logits from
  entity tokens, not a flat action MLP.

### 1.2 Training [S]
- "trained purely with self-play reinforcement learning. The RL algorithm is PPO with generalized advantage
  estimation. Rewards are undiscounted: 1 for a win and 0 for a loss."
- Regularisation: "entropy regularization and a zero-avoiding term, which is essentially KL divergence toward a
  uniform policy. The latter prevents the model from assigning effectively zero probability to a very
  situational move such as Encore and never reconsidering it." → [I] KL(u‖π) = −mean_a log π(a) + const; it
  punishes log π → −∞ (unlike entropy, which barely penalises tiny probabilities).
- Auxiliary losses:
  - "Its probability of winning, which serves as the value function" → [I] value is a win-probability
    classifier (BCE on 0/1), not MSE on ±1;
  - "The item, ability, moves, and Tera type of each opposing Pokémon. It maintains the item and ability
    predictions across turns so it can remember revealed information." → [I] targets are the opponent's true
    hidden set (free in self-play); "maintains across turns" suggests last turn's predictions are fed back as
    inputs (a recurrent belief), since the model is otherwise memoryless beyond 16 history events;
  - "The opponent's next action. This one made the bot substantially stronger, likely because it can react to
    these predictions." No ablation number (asked on Smogon; no answer in readable pages).
- "Unlike AlphaZero, Jaxcalibur does not use search during self-play. Training only the base policy is much
  faster, and search appears to be less valuable for Pokémon than for Go or chess."
- "The final training run took about a week on one H100, during which Jaxcalibur played almost 100 million
  games against itself." → [I] ≈ 165 games/s, ~4–8k decisions/s including learning.
- Opponent pool / league: not mentioned (plain "self-play").
- Effort (author, Smogon p.1): "I worked on it on-and-off for more than six months, and I was lucky enough to
  have access to an H100 ... it would take someone with AI/ML experience at least a few months and several
  hundred dollars in GPU credits to reimplement something comparable."

### 1.3 Simulator [S]
- "I reimplemented Pokémon battling in pure JAX so games can be vmapped and jitted to run quickly on a GPU."
- "The engine always executes every possible game step (moves, switches, end-of-turn effects, and so on) and
  keeps only the relevant results. This wastes FLOPs, but gives every action the same computational structure."
- "Building the engine would have been a massive undertaking without Claude Code." gen9 randbats only — "fewer
  pokemon/moves/abilities [to] implement" (author, Smogon p.1).
- Fidelity: no validation method published (asked on Smogon; unanswered in readable pages). A later author post
  (page 4, search snippet) mentions "a few fixes such as the stealth r[ock ...]" before the MichaelderBeste2 match.
- Related: **PokeJAX** (Karten et al., arXiv 2603.12145): LLM-translated JAX port of Showdown, "500±9M"
  random-action steps/s, "15.2±0.2M" steps/s in PPO, vs "Pokemon Showdown: 21K SPS native, 681 SPS via
  PokeEnv"; no released code found.

### 1.4 Search (test time only) [S]
- "AlphaZero-style pUCT, but with several modifications":
  - **Simultaneous moves:** "samples edges during search according to a mixture of the positive regret and
    network prior ... it shift[s] towards a regret-matching policy as the number of node visits increases":
    ```
        π(s,a) ∝ max(Q(s,a) − V(s), 0)  +  c_puct · P(s,a) / √N(s)
    ```
  - **Chance:** "rolls out every trajectory from scratch with a different random seed ... instead of
    maintaining an explicit search tree Jaxcalibur stores statistics in a dictionary mapping
    {game_state_hash -> node}. To reduce the number of nodes, Pokémon HP is binned into 10 buckets."
    → [I] open-loop MCTS with a transposition table; cheap because the engine is batched on GPU.
  - **Our hidden info:** "As in Foul Play, Jaxcalibur handles this by sampling possible 'worlds' of the
    opponent's unknown Pokémon/moves/items/etc. using its network's predictions." Our nodes are "hashed using
    only revealed information about the opponent, while opponent nodes are additionally hashed on the world ID".
    → [I] information-set statistics for our decisions (shared across worlds; we cannot pick a different action
    per world — avoids determinization's strategy fusion), per-world nodes for the opponent.
  - **Opponent's hidden info:** "Search for the opponent is truncated whenever Jaxcalibur does something
    'surprising,' such as switching to an unrevealed Pokémon or using a move predicted with <65% likelihood
    from the opponent's perspective. From that point onward, the opponent only uses the network prior. This
    prevents search from leaking hidden information such as Tera types to the simulated opponent."
- Budget: "depth 4 (usually 2 full turns ahead), using 20,480 rollouts over 32 possible worlds. This takes just
  a few seconds on a GPU." Leaves are valued by the net (implied by depth truncation).
- Annotated replays [S]: per turn the bot logs "win chance", "expects opponent holding <item> (p%)", per
  candidate move a searched "win chance" and a final "preference" (usually 80–96% on one move, sometimes split,
  e.g. 32/19/15/13%), and the opponent's "likely reply" distribution. Mid-turn forced switches show "Not using
  search because the opponent's next move is unrevealed" → [I] no search on U-turn/Flip Turn switch-ins.
- "my current estimate is that search adds 100-150 Elo over its base policy."

### 1.5 Results and failure modes [S]
- Ladder Aug 6–27 2026: "top of the ladder about 20% of the time and maintained a 68% win rate against other top
  players. It reached a peak Elo of 2557 and a peak GXE of 95.3." Foul Play "peaked at around 2341 Elo and 88
  GXE, which is roughly top-50".
- Net alone: "With search turned off, Jaxcalibur can run on an ordinary laptop and still be very strong; I'd
  guess around 2400 Elo."
- Challenges: replays page "4473 wins 838 losses 84.2% win rate" (5311 played); exhibition vs MichaelderBeste2
  (the one human above it): "Jaxcalibur 5–3".
- Weaknesses (author): "fairly conservative, preferring safe lines over hard reads. It is especially good at
  endgames because it can calculate several turns ahead. I think its clearest weaknesses are an imperfect
  understanding of PP stalling and a tendency to become predictable in some situations."
- Self-play per-species win rates (net, no search): Alomomola 58.9% top, Luvdisc lowest, most 50–54%.
- Sept 12: "I've decided not to release Jaxcalibur's code or model weights."

### 1.6 Imitators that report details ([S] from their own READMEs/posts; unverified)
- **mikumiku37** (VGC Reg M-C #1; Smogon Jaxcalibur thread p.5): "a small transformer (about 8.7M parameters)
  trained from scratch through self-play reinforcement learning: PPO against a league of its own past versions,
  with no human replays, and only terminal rewards. Part of the design is adapted from the Jaxcalibur technical
  writeup"; r/stunfisk: ~48 h of training.
- **Poke-RL** (github.com/MSalman5230/Poke-RL): "Jaxcalibur informed the entity transformer, history,
  opponent-action auxiliary task, league training, and evaluation approach ... This project uses a CPU Showdown
  simulator with GPU learning".
- **MissingNo.Zero**: "The write-up showed that self-play PPO without search in training could reach #1 in
  generation 9. We then dropped search from training ... and adopted several of its published ideas" (which
  ones: not stated). Blog: search during training was "a dead end I'd been stuck on".

### 1.7 What Jaxcalibur implies for us [I]
- Ingredients we lack, in the author's emphasis order: (1) a strong network (most of the Elo); (2) opponent
  next-action aux head ("substantially stronger"); (3) opponent hidden-set aux heads, which also drive world
  sampling in search and feed the damage-calc inputs; (4) damage-calc matchup tokens; (5) zero-avoiding KL.
- Scale: ~100M games ≈ 2.5–5·10⁹ decisions. MissingNo.Zero shows 0.7–1.8·10⁹ steps on real Showdown already
  beats Foul Play 62–67% net-only, so Jaxcalibur's extra scale buys the last ~150–200 Elo, not the first.
- Search design to copy once the net is in the tree: regret+prior sampling at simultaneous nodes; information-set
  hashing for our nodes (shared across worlds); truncate opponent search after our "surprising" actions; HP
  bucketing; and MissingNo.Zero's rule "only trust search when it beats the network's pick by a clear margin"
  (its search went from −6.8 to +1.6 points; overruled the net on 72.8% → 8.5% of moves).

---

## 2. Other strong random-battles agents and their recipes (2023–2026)

### 2.1 MissingNo.Zero (the most transferable; gen1/2/4/6/7/9 randbats) [S]
Sources: README + technical report PDF (https://github.com/nickgreenquist/missingno-zero, no licence, no code),
blog (https://nickgreenquist.com/blog/projects/2026/11/02/missingno-zero-self-play-pokemon-showdown/). Author
Nick Greenquist (Google SWE, personal project, built "with five Claude Code sessions running in parallel").
- Recipe: "Self-play, no human data. Each agent learns only from games against itself and its past versions";
  "PPO ... (about 3M parameters) ... with settings tuned per generation. Our starting recipe came from Wang
  (2024) ... every run learns from both players' sides of each self-play game"; "Search only at play time".
- Simulator: "Gen 1 trained in pkmn/engine ... Gens 2, 4, 6, 7 and 9 trained on Showdown's own simulator ...
  Every game ... reaches the agent through poke-env".
- Cost table (final runs): gen4 700M steps / ~6 days / ~$230; gen9 800M steps / ~8 days / ~$280 (README later
  says "1.8B (800M, then 1B more)", ~9 days, ~$350). Whole project ~$2.0–2.1k incl. dead ends.
- Vs Foul Play (95% Wilson): gen4 **61.6% (59.4–63.7) no search, 67.5% (65.4–69.5) with search, 2000 games**;
  gen9 64.5% (58.4–70.1) / 66.8% (256 games, report) and **66.7% ± 1.7 no search, 3000 games** (README, #1
  model). Foul Play fixed to the iteration count it reaches "in that time on our laptop".
- Lessons [S, blog]:
  1. "Search made the agent worse (at first)": −6.8 points from the optimizer's curse; "only trust search when it
     beats the network's pick by a clear margin" → overrule rate 72.8% → 8.5%, −6.8 → +1.6 points.
  2. "The value head took over": "the value gradient into that trunk was about 50x the policy's ... The fix was
     to only let a fraction of the value gradient reach the trunk (plus a shorter advantage horizon and an
     entropy schedule), and to auto-pause any run where the ratio started climbing early."
  3. Gen2 sleep: the net keyed on "did I just use Rest?" instead of "am I asleep?"; "retrained it with a focus on
     sleep and long games" → 72.0% vs 58.4% vs Foul Play (~3000 games each).
  4. "Use the real simulator" (ports were slow to build and inexact).
  5. "Train on this month's teams" (randbats sets change monthly, old gens included).
- Caveat [S]: "What mattered is unmeasured. We changed the recipe between generations without ablations".
- [I] Ladder Elo in gen4 (peak 1738) ≈ Foul Play's gen4 peak (1728), yet 61.6–67.5% head-to-head — ladder peaks
  are a poor strength measure in thin ladders; head-to-head is what counts.

### 2.2 Others with numbers
- **ps-ppo / "PokemonTransformer"** (Nebraskinator, **MIT licence**, https://github.com/Nebraskinator/ps-ppo),
  gen9 randbats [S]: field token + 12 Pokémon tokens + actor and critic CLS tokens with an attention mask
  (state tokens cannot see the CLS tokens; actor and critic cannot see each other); per-entity MLP tokenizers;
  numeric "bank" embeddings for HP/stats/power; 14 actions (4 moves, 6 switches, 4 tera-moves); 51-bin
  distributional value on [−1.6, 1.6]; BC warm start on SimpleHeuristicsPlayer, then PPO ">250M states over the
  course of 2 days on a consumer PC (RTX 3090)", reward ±1 win + ±0.1 faint; ">1900 ELO" net-only. Also: "the
  replicated [Wang] MLP agent plateaued at approximately 1100 ELO" without MCTS. → [I] an MIT-licensed gen9
  encoder/model we may legally borrow from (attribution).
- **Metamon** (Grigsby et al., RLC 2025, https://arxiv.org/html/2504.04395, https://github.com/UT-Austin-RPL/metamon)
  [S]: OU formats only, no randbats ("we will focus on formats where players design teams"). 475k human replays
  → first-person trajectories; 15M–200M transformers; offline RL in AMAGO; best = "binary weighted BC update with
  value prediction converted to two-hot classification". README: "Almost all improvement in metamon's
  performance is driven by large and diverse datasets of agent vs. agent battles." Self-play on realistic teams
  overfit ("the model believes it is playing SynRL-V1"); fix = diverse teams + mixed opponents. Ratings: GXE
  77/68/64/66 (G1–G4 OU), later Kakuna 82/70/63/64/71 (G1–G4, G9 OU); vs Foul Play: draw in gens 3–4.
- **PokéAgent Challenge** (arXiv 2603.15563) [S]: "13 [of 16 qualifiers] were secured by teams extending our
  public RL baselines"; "the top participants all used RL or MCTS rather than LLM reasoning". Foul Play won Gen9
  OU 50–14; PA-Agent won Gen1 OU (Metamon fine-tune, 6 rounds of inter-model data, human share 100% → 10%);
  4thLesson (SynRL-V2 + Kron optimizer + AID activation "mitigating plasticity loss", ~1.3–1.4M self-play
  replays); Team Q (50M actor-critic, heuristic "mechanics phase", then coach cycle). No second edition with
  randbats; now a "living benchmark" with OU ladders only.
- **PokeCHAD V3**: #1 on pkmn.bot gen9 randbats (1572) and beats Metamon 39–4 in gen9 OU there; "V3 @ 18M Training
  games"; method unknown (https://pkmn.bot/bots/pokechadv3).
- **Victory-Dance** (gen9 singles, per a third-party summary): BC on ~86k human battles → advantage-weighted
  offline RL → KL-anchored PPO league; top-500, ~1700–1744 Elo [S, unverified]. [I] Weak evidence that a human-BC
  start is not by itself a shortcut to the top.
- **PokéLLMon** (arXiv 2402.01118): 49% ladder win rate (≈ matchmaking parity), irrelevant for strength.
- **Safari / Oak** (Smogon 2026-07, open source, RBY): network + information-set MCTS "as an iteration upon Foul
  Play" — gen1 only.
- **Foul Play** (description only; GPL) [S, https://pmariglia.github.io/posts/foul-play/]: root-parallel DUCT
  MCTS; damage rolls grouped by whether they KO, averaged; "Set prediction is as important as, if not more
  important than, accurate and fast search"; expectiminimax → MCTS let lines be searched "to a depth of 10 or
  more". Gen4 peak 1728 (rank 3), gen9 2341 / 88 GXE.
- Wang (2024) follow-ups: MissingNo.Zero and ps-ppo are the direct ones; nothing new by Wang found.

---

## 3. Fast simulators (fastenv) [S unless marked]
- **Showdown itself is fast; poke-env is the bottleneck.** PokeJAX paper: "Pokemon Showdown: 21K SPS native,
  681 SPS via PokeEnv" — "running multiple instances via PokeEnv yields only 681 SPS due to WebSocket overhead"
  (https://arxiv.org/html/2603.12145v1). poke-env maintainers (issue #296): "most of the time the bottleneck is
  not on poke-env's side, but on pokemon showdown's side" (2022) — [I] contradicted by the PokeJAX numbers once the
  server's battle delays are removed. Our 2.1k steps/s on 30 CPUs ≈ 70 steps/s/CPU [I: websocket + Python
  parse/encode bound].
- **MissingNo.Zero**: kept Showdown and "just chang[ed] how training drove it", "nearly as fast" as a port.
- **pkmn/engine** (Zig): gen1 complete, gen2 in progress, gen3–4 "stage 2", modern gens "soft-blocked";
  "more than 1000× faster" than Showdown (3167× on RBY). Not usable for gen4/gen9 now.
- **poke-engine**: a search engine, not a battle runner (no team generator / game loop; damage-roll grouping;
  open mechanics bugs e.g. "Rest succeeds at full HP", "status damage runs before Leftovers").
- **vjeux/pokemon-showdown-rs**: "1-to-1 port", gen9 random battles, "~50 battles/second single-threaded", still
  fixing parity ("failing-seeds.txt"). **Ninjask** (Rust, Champions VGC only): matches Showdown "exactly, about
  200 times faster" on 1000/1000 seeded games; `Batch` API returns per-player observation buffers.
- **JAX**: Jaxcalibur (closed), PokeJAX (no code found; 45 s JIT, every step pays all 1,370 moves).
- [I] Ranked route for ≥10×: (1) in-process Showdown (`BattleStream`/`Battle` in Node worker threads, per-player
  side streams kept, our own JS/native state tracker + encoder emitting numeric buffers, no websockets, no
  poke-env in the loop) — exact for gen4 and gen9 on day one; (2) later, a Rust port for in-tree inference only;
  (3) JAX only if (1) is exhausted. Fidelity risks: the first-person tracker must equal what poke-env sees live
  (parity test on logged real battles), and the generator's monthly set changes (pin a Showdown commit, refresh
  monthly).

---

## 4. Human replay data (data stream) [S unless marked; from a sub-agent, spot-checked]
- **Key finding: random-battle replays publish the full `inputlog`** (seeds, both team seeds, the Showdown git
  commit, every choice). Verified 2026-11-08 on https://replay.pokemonshowdown.com/gen9randombattle-2692834507.inputlog
  (`>version 9fb3a5b…`, `>start {"formatid":"gen9randombattle","seed":"sodium,…"}`, `>player p1 {…,"seed":"sodium,…"}`,
  `>p1 move diamondstorm`, `>p2 switch 2`, …). Server code (`isSafeInputlog`) allows this only for ids ending in
  `randombattle`. → [I] Re-simulating at the logged commit reproduces the battle exactly, giving both sides'
  true teams and real `|request|`s: **exact first-person views, no Metamon-style guessing**. Needs a Showdown
  checkout per commit (group by commit; validate by regenerated log == stored log).
- Corpora: **HolidayOugi/pokemon-showdown-replays** (HF): gen9randombattle 2,012,698 rows (~4.5 GB parquet, 11
  parts), gen4randombattle 80,365 (186 MB); columns `id, format, players, log, uploadtime, views, formatid,
  rating`; no inputlog; overlapping snapshots → dedupe on `id` (one format was overcounted "nearly two to one").
  **milkkarten/pokemon-showdown-replays-merged**: 29.06M replays, gen9 randbats 703,345 after dedupe/filters.
  Metamon datasets: no randbats. No licences stated on the big mirrors; Metamon parsed = CC-BY-NC-4.0.
- API: `search.json?format=&user=&sort=rating|date&before=<uploadtime>&page=` — 51 per page; pages > 100 return
  `[]` (so `sort=rating` reaches only the top ~5,000); walking back with `before` is unlimited;
  `/<id>.json|.log|.inputlog`; batch `/api/replays/batch?ids=` (≤51, no inputlog). No published rate limit;
  others used ≤5 req/s, ≤4 in flight. Only saved replays exist (~1–2% of randbats games [I estimate]); medal
  holders' battles are forced public in the last 3 days of each month.
- Volumes: gen9 ~1,500 saved/day, ~43% rated ≥1700; **gen4 high-rated human data is tiny** (~500 public replays
  ≥1583 ever, ceiling 1771) → gen4 must rely on self-play.
- **Bot games are data too** [I]: Jaxcalibur's public games (2,945 public of 5,311 challenge games, plus its
  ladder games; listed at https://jaxcalibur.github.io/replays/, still playing in Nov 2026 per `search.json?user=jaxcalibur`)
  have inputlogs, and its replay pages expose per-turn policy "preference", searched win chances, item belief and
  predicted opponent reply — soft targets from a 2400–2557 bot. **Flag for the user:** the author withheld the
  weights on community-integrity grounds; distilling his bot from its public outputs is legal-looking but against
  the spirit of that decision. Do not do it without the user's explicit OK.
- Set priors: Smogon publishes no randbats usage stats since 2016; use the generator
  (`data/random-battles/gen9/sets.json`) and pkmn randbats stats (`data.pkmn.cc/randbats/stats/gen4randombattle.json`,
  100k generated teams, hourly).
- Asking PS staff for data is a known route (Future Sight AI got 2M battles that way).

---

## 5. Transferable recipes from other games [S via sub-agent unless marked; numbers spot-checked against memory]
- **AlphaStar** (Nature 2019): human-data ablation (test Elo): no human data 149 → supervised 936 → human init +
  RL 1,020 → + KL to the supervised policy 1,400 → + z statistic 1,540: "We found our use of human data to be
  critical." Value fed the opponent's observations during training only. PFSP + SP beats SP on worst case vs past
  selves (46% → 71% min win rate); exploiters +284 Elo. [I] AlphaStar's action space is vastly bigger than ours;
  both Pokémon #1 bots used no human data, so the human-init gain should be expected to be much smaller here.
- **OpenAI Five**: 80% latest / 20% past opponents; lr changed only at plateaus ("hyperparameters were modified by
  looking for improvement plateaus"); "high quality data matters even more than compute consumed"; sample reuse
  2–3 ≈ 2× slowdown; staleness hurts.
- **KataGo** (arXiv 1902.10565) ablation factors (compute to reach equal strength without the component):
  ownership + score targets 1.65×, opponent-reply policy target 1.30×, global pooling 1.60×, playout-cap
  randomisation 1.37×, forced playouts 1.25×, game-specific input features 1.55× (product ≈ 9×). Mechanism: aux
  targets give "sharper, more localized feedback". [I] Pokémon analogues: opponent hidden set, opponent next
  action (= KataGo's opponent-reply target; Jaxcalibur's biggest stated gain), per-mon faint/HP-at-end, turns left.
- **Policy-gradient vs game-theoretic RL**: Rudolph et al. (ICLR 2026, arXiv 2502.08938; >7,000 runs, exact
  exploitability): "NFSP, PSRO, ESCHER, and R-NaD fail to outperform the generic PG methods (MMD, PPO, PPG)".
  MMD (arXiv 2206.05825) = PPO-like step control + KL to a "magnet" (uniform ⇒ entropy-like) — Jaxcalibur's
  zero-avoiding KL toward uniform is a magnet term in all but name [I].
- **DeepNash/R-NaD** (Stratego): reward transform with a regularisation policy reset every stage; 84% vs experts,
  no search. Not needed if tuned PPO is as good (above).
- **ReBeL / Student of Games / Obscuro**: exact public-belief-state methods impractical for randbats' hidden-set
  space [I]; Obscuro's transferable bits: sample a few hundred determinised states, solve the root, purify the
  final mixed policy (full Obscuro scores 58.0% vs its no-KLUSS variant and 70.2% vs its no-purification variant).
- **Gumbel AlphaZero**: "Gumbel MuZero learns reliably even with 2 simulations" with completed-Q policy targets;
  [I] visit-count targets from few-sim searches are not improvements — for simultaneous moves use an equilibrium /
  completed-Q target over the joint matrix.
- **Expert iteration** (Anthony et al. 2017, Hex): first dataset 100k positions sampled one per game; tree-policy
  targets +50 ± 13 Elo over chosen-move targets; value net "requires more than 10⁵ independent samples"; online >
  batch. [I] our 9k-battle round was ~10× too small and used the wrong target; and both Pokémon #1 bots skip
  search in training entirely ("a dead end I'd been stuck on" — MissingNo.Zero).
- **piKL / Diplodocus**: RL/search regularised toward a human BC policy → top human-play Elo in Diplomacy. Relevant
  only if we use human data (gen9).

## 6. Learning rate, plateaus, embeddings, sizing [S via sub-agent unless marked]
- Andrychowicz et al. 2021 ("What Matters in On-Policy RL", 250k runs): "Linearly decaying the learning rate may
  slightly improve performance but is of secondary importance"; Adam lr 3·10⁻⁴ safe default; separate wide value
  MLP; observation normalisation. OpenAI Five: lr 5·10⁻⁵ → 1·10⁻⁶ only via manual drops at plateaus. KataGo:
  fixed per-sample lr 6·10⁻⁵, one late drop to 6·10⁻⁶ "to maximize final strength"; gains of final nets "most
  definitely due to learning rate drops". Hägele et al. (NeurIPS 2024): constant lr + short (10–20%) cooldown ≈
  cosine; weight averaging ≈ simulated decay.
- [I] Our schedule `10^-4.23/(8x+1)^1.5`: mean over the run = 1/6 of peak; ≤ peak/11 for x > 0.5; peak/27 at the
  end — we spent the second half of training in a cooldown. The 400M→600M "plateau" says nothing about capacity.
- Plasticity: Dohare et al. (Nature 2024) loss of plasticity in long non-stationary training; Lyle et al. 2024:
  "layer normalization and weight decay is highly effective at maintaining plasticity"; 4thLesson used AID
  activations against it. [I] Self-play is non-stationary; pre-LN + mild decay is cheap insurance.
- Embeddings: Adam moves each coordinate by ≲ lr per step, so an N(0,1) table barely moves at lr ~10⁻⁵ [I]; μP /
  Hayou & Liu 2025: embedding lr should be ~√width–width × the hidden lr; SmallInit (RWKV): small init + LayerNorm;
  minGPT excludes embeddings from weight decay. Our ×30/×300 null + Metamon's "no evidence" that `<unknown>`
  masking matters ⇒ [I] embeddings are not the bottleneck; prefer feature-first entities (base stats, types, move
  power/category/priority/flags, damage calcs) with small-init ID residuals (std 0.02 + LN) and 10–30% ID dropout
  for rare sets.
- Sizing: Neumann & Gros 2023 (AlphaZero scaling): optimal size ∝ C^0.62, published game nets are undersized;
  Jaxcalibur: "smaller networks trained for longer outperformed larger networks trained for fewer steps" (at one
  GPU-week). [I] 3–10M is the right range for ≤ 2·10⁹ steps.
- Diagnostics to log every update: approx-KL (healthy ~0.005–0.02), clip fraction (~0.1–0.3), explained variance,
  entropy, ‖Δθ‖/‖θ‖ per parameter group, **‖∇_trunk L_value‖ / ‖∇_trunk L_policy‖** (MissingNo.Zero's 50× alarm).
  [I] Check the old gen4 run logs: KL ≈ 10⁻⁴ and clipfrac ≈ 0 after 400M would confirm "no learning".

---

## 7. Ranked design recommendations

| # | recommendation | expected impact | evidence | cost | owner |
|---|---|---|---|---|---|
| 1 | **Fix the PPO recipe before anything else**: lr warmup → constant (~3·10⁻⁵–1·10⁻⁴) → drop ×3–10 only when head-to-head Elo stalls, final short cooldown (+EMA weights); stop most of the value gradient reaching the shared trunk (separate value tower, or scale value grad into trunk to ~0.1–0.2) and log the value/policy trunk-gradient ratio; shorter GAE horizon; entropy schedule. **Cheap first test:** resume gen4 600M at constant lr with the value-gradient fix for 50–100M steps, benchmark vs Foul Play and vs our search. | very high (MissingNo.Zero gets 61.6% vs FP net-only in gen4 with a Wang-type 3M PPO; we are far below) | strong (same format/size/throughput), but MNZ has no ablations | low | netv2 (wsp can check old logs for KL/clipfrac/grad ratio) |
| 2 | **Aux heads**: opponent next action (Jaxcalibur: "substantially stronger"; KataGo 1.30×), opponent hidden item/ability/moves/tera from the true set (feeds world sampling + damage inputs), value as win-probability BCE (or two-hot/51-bin); optionally a privileged critic (true opponent team) for training-time advantages only, with a separate public value head for search | high | moderate–strong (Jaxcalibur statement, KataGo/AlphaStar ablations) | low | netv2 |
| 3 | **Inputs**: entity tokens + per-token-type FFNs; damage-calc matchup tokens (own × opp, using predicted sets); 16-event history; pointer action logits incl. tera; feature-first entity encoding, small-init ID residuals; explicit state flags (asleep/turns, not "used Rest") | high | moderate (Jaxcalibur design; MNZ sleep lesson +13.6 points) | medium | netv2 (spec), fastenv/data implement |
| 4 | **Train on the current month's random-battle sets**, pin the Showdown commit, refresh monthly; both sides of each self-play game as data; past-self opponents (80/20-style) and track min win rate vs a fixed panel | medium | moderate (MNZ, OpenAI Five, AlphaStar) | low | netv2 + fastenv |
| 5 | **Fast env = in-process Showdown**, not a port: Node worker threads running `BattleStream`/`Battle`, per-player side logs, native/JS encoder to numeric buffers, no websockets/poke-env in training; parity test vs poke-env-tracked live battles. Target ≥10× (21K vs 681 SPS gap in PokeJAX). Rust/JAX only later, for in-tree inference | medium–high (needed for Jaxcalibur scale, not for MNZ level) | moderate | medium | fastenv |
| 6 | **Exact human data via inputlog re-simulation** (gen9): HolidayOugi index → dedupe → fetch `.inputlog` for rated ≥1700 (≈ hundreds of k) → re-simulate per commit → exact first-person trajectories; use for BC pretraining / value pretraining and as a KL anchor early in RL (decay it). gen4: data too small, skip. | medium (sample efficiency, early training), uncertain at the top (both #1 bots used no human data) | moderate (AlphaStar, Metamon, piKL) | medium | data |
| 7 | **Network in the search** (after #1–3): Jaxcalibur-style regret+prior sampling at simultaneous nodes; our nodes hashed on public info (shared across worlds), opponent nodes per world; worlds sampled from the set head; truncate opponent search after our "surprising" moves; HP buckets; **only override the network by a clear value margin** | +100–150 Elo / +6 points over the net | moderate (Jaxcalibur, MNZ) | high (Rust inference) | fastenv (later) + netv2 |
| 8 | **Benchmark hygiene**: fix Foul Play to an iteration count (MNZ) so parallel benchmarks are machine-independent; add pkmn.bot (bot ladder, gen9 randbats: PokeCHAD V3, MissingNo.ZeroBot) and the Showdown bot registry as external yardsticks | medium (decision quality) | strong | low | orchestrator / benchmark slot |
| 9 | Zero-avoiding KL toward uniform (+ entropy) | low–medium | weak–moderate (Jaxcalibur; MMD theory) | trivial | netv2 |
| 10 | Model size 3–10M; scale only after #1 shows the net responds to data | — | moderate | — | netv2 |

## 8. Looks clever, evidence says it doesn't matter (or not yet)
- **Search during training / expert iteration at small scale** — both #1 bots train without search
  (MissingNo.Zero: "a dead end I'd been stuck on"); ExIt needs ≥10⁵ decorrelated positions and non-visit-count
  targets; our 9k-battle round null is expected.
- **Exotic game-theoretic RL (R-NaD, NFSP, PSRO, ESCHER)** — tuned PPO/MMD/PPG match or beat them (Rudolph et al.).
- **Public-belief-state search (ReBeL / Student of Games)** — impractical hidden-set space; determinized worlds +
  info-set statistics are what the top bot uses.
- **Bigger networks at fixed compute** — Jaxcalibur: smaller-for-longer won.
- **A reimplemented fast engine before the recipe works** — MNZ: ports were slow to build and inexact, and Showdown
  driven differently was "nearly as fast"; MNZ reached 61.6% vs FP at ~1.35k steps/s.
- **Boosting embedding learning rates / fancier ID embeddings** — our ×30/×300 null; Metamon: `<unknown>` masking
  had "no evidence" of effect; features carry the information.
- **More determinizations / root game tricks / regret matching in the tree** — our own powered nulls; PokaiTrainer
  DUCT ≈ CFR.
- **Better labels for unobserved actions in replay reconstruction** — Metamon: "no evidence that the new action
  labels impact the RL policies" (irrelevant anyway with inputlog re-simulation).
- **Naive search on top of a good net** — MissingNo.Zero: −6.8 points until a margin rule was added.
- **Human data as the main lever** — neither #1 bot used any; treat it as a sample-efficiency booster for gen9, not
  as the strategy.

## 9. Open questions we could not answer from public sources
- Jaxcalibur: opponent-action-head ablation size; which 4 opponents per matchup token; engine validation; training
  curve (was it still improving at 100M games?); lr schedule; opponent pool. (Smogon pages 3–6 are login-walled and
  not archived.)
- MissingNo.Zero: lr schedule, exact value-gradient fraction, advantage horizon, observation design, how training
  drove Showdown. No code. Its Smogon thread was not readable.
