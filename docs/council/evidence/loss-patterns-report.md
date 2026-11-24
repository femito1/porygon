# Loss patterns of raw G9q-cd@805M (gen9, T 0.5): wsp report, 2026-11-16 ~23:45Z

Net: `porygon-ws-netv2/runs/g9q-cd800/final/00805029873.pt` (sha256 e93ac3f8…b11f), sampled at T 0.5 everywhere.
Code: ws/wsp 1ea7acc+ `scripts/wsp/losses/` (`loss_patterns.py`, `fp_games.py`, `run_games.sh`, `run_analysis.sh`) and the new
opt-in `spd_h2h.py run --game-log DIR` (spd_host.js writes the omniscient Showdown log, every choice and a true-state
snapshot taken just before each choice). Raw outputs here: `patterns.txt` / `patterns.json`, `fp_games_805.txt`,
`fp_games_pooled.txt`.

## Data
1. **Real games vs Foul Play (evidence, but no per-turn data).** The conf800 FP run (job 1173380, 1000 games,
   57.8% ± 2.8) logged only `battles.jsonl` (pair, game, winner, turns, both true teams), `results.jsonl` / `vs.log`
   (win, turns) and `fp0.log` (Foul Play instance 0's own search log for 50 games, which does not record our moves).
   No protocol and none of our decisions. Pooled check: + cd@605's two runs (1173215, 1172567), 2400 games, 1200 pairs.
2. **Logged in-process games (fastenv simulator, exact Showdown mechanics, ran in tracker pass 1173949):**
   - M1 = cd@805 vs itself, 1500 duplicate pairs (3000 games, seed 9161101): A 49.1% ± 1.5.
   - M2 = cd@805 vs BC-P3 (`runs/pretrain/P3-g9-s2.pt`, the human-imitation init), 1000 pairs (seed 9161102): A 72.8% ± 1.8.
   - Games: `m1/games`, `m2/games` (`*.jsonl.gz`, 40 MB). 107k decisions per side in M1.

Comparisons used: (a) in M1, the pattern's rate per opportunity in the games a side won vs the games it lost (same net
on both sides, so any difference reflects the position or the outcome, not the player); (b) in M2, our net vs BC-P3 on
their own decisions (BC-P3 imitates rated humans but is much weaker overall); (c) for the FP data, our win rate by game
length and by team features. Intervals: Wilson 95% over decisions (they are clustered within games, so a little
narrow) or t-intervals over games / pairs. The "association cost" is the OLS slope of A's result on the per-game
difference in counts (A − B) in M1. That is an association: positions that are already lost produce some patterns. It
is not a causal cost.

## Ranked findings (by estimated win-rate impact)

### 1. Our whole edge over Foul Play is in long games; in fast games we are at par or below (EVIDENCE, real games)
| game length (turns) | games | our win rate vs FP (cd@805, 1173380) |
|---|---|---|
| < 15 | 28 | 35.7% [20.7, 54.2] |
| 15–20 | 118 | 40.7% [32.2, 49.7] |
| 20–25 | 207 | 54.6% [47.8, 61.2] |
| 25–30 | 190 | 54.2% [47.1, 61.1] |
| 30–40 | 239 | 64.4% [58.2, 70.2] |
| 40–60 | 173 | 68.8% [61.5, 75.2] |
| ≥ 60 | 45 | 68.9% [54.3, 80.5] |
- Games under 25 turns: 353 games at 48.4%, which hold 182 of our 422 losses. Games of 25 turns or more: 647 at 62.9%.
- Pooled 2400 games (cd@805 + cd@605): under 20 turns 41.7% (343 games); 40–60 turns 69.4% [64.7, 73.8].
- **Duplicate-pair version** (both sides play both teams, so team strength cancels; only the pair's mean length,
  mostly set by the matchup, remains). Pooled pair score: mean length < 20 turns 47.7% ± 5.6 (107 pairs), 20–30
  ~55%, 30–40 63.0% ± 3.3, ≥ 40 63.9% ± 4.5. cd@805 alone: 44.3% ± 7.8 (53 pairs) vs 64.1% ± 4.8 / 66.8% ± 7.0.
- **This is not mechanical.** In self-play (M1) our win rate is flat in game length (50.7 / 49.7 / 48.9 / 45.8 / 50.8%).
  Against BC-P3 (M2) the slope runs the other way: we are strongest in short games (74.6–76.1% under 30 turns, 63.5–67.2%
  over 40). Only against a search bot do we lose the fast games.
- **Impact:** if games under 25 turns were played at the ≥ 25-turn rate, we would gain about +5.1 points (353/1000 ×
  14.5). That is an upper bound, since some of the length effect is endogenous.
- **Mechanism: SPECULATION.** Fast games are KO races (speed order, damage thresholds, setup or priority timing),
  where a one-ply-calculating searcher beats a raw policy, while our net's long-horizon play wins slow games. The
  in-process loss correlates below agree: the events most over-represented in lost games are all tactical KO events,
  and they concentrate in short games. Loser − winner per game, short (< 25 turns) vs long: KO'd before moving after
  staying in +1.81 vs +1.02; set up and got KO'd that turn +0.37 vs +0.23; forced switch into a revealed SE attack
  +0.36 vs +0.18. This is a general weakness (tactics), not a Foul Play rule. It is exactly what lookahead2-type
  search should repair.
- No team feature explains it. Pair score with vs without each of 16 features in either team (hazard setters and
  removers, setup, status, Rest, Sub, Protect, Trick, Choice items, Tera Blast, pivots, Illusion/Ditto, weather,
  Encore/Taunt, phazers, priority): all differences lie within ±4 points. The only nominal hit is Protect, +4.0
  [+0.3, +7.7], which is expected by chance among 16 tests.

### 2. Setting up at < 50% HP while slower (P8) (in-process; association)
- **Definition:** a voluntary self-boosting status move (positive boosts, or Belly Drum) chosen at HP < 50% while our
  active's effective speed is below the opponent's.
- **Rate:** 3.5% of our setup moves (257/7303 in M1, 0.086 per game). Per opportunity, M1 losses 5.1% [4.6, 5.6] vs
  wins 1.8% [1.5, 2.2]. Association cost −0.19 ± 0.035 game win per occurrence, so ≈ 1.6 points.
- **Related outcome:** set up and got KO'd on the same turn (P8c): 7.3% of setups; losses 11.7% vs wins 1.5%.
- **vs BC-P3:** we are not worse (2.2% vs 2.9%, M2).
- **Examples (M1):**
  - pair 63 g0 T17–18: Meganium 44/290 Swords Dance twice vs Girafarig (faster, Nasty Plot then Dazzling Gleam), KO'd.
  - pair 9 g1 T18: Lapras 62/367 (−1 Spe) Dragon Dance vs Hypno, KO'd by Psychic Noise.

### 3. A resisted attack when a neutral-or-better one was available (P3) (in-process; vs BC and association)
- **Definition:** a voluntary attack whose type is resisted (≤ ½, types only, current incl. Tera) by an opponent at
  ≥ 30% HP, while another usable attack is neutral or better with ≥ 70% of its base power. Excluded: pivots (U-turn,
  Volt Switch…), fixed-damage moves and Scrappy / Mind's Eye users. Headline variant P3s counts only turns where the
  opponent stayed in, so switch predictions do not count.
- **P3s rate:** 3.5% of eligible attacks (1433/41398 in M1, 0.48 per game). M1 losses 3.97% vs wins 3.03%.
  Association −0.018 ± 0.010 per occurrence, so ≈ 0.9 point.
- **vs BC-P3:** 3.5% vs 2.7% (M2), so we do it ~30% more often than human imitation.
- **All-turns P3:** 4.6% (M1); M2 4.4% vs BC 3.3%. Some cases are still deliberate (Knock Off utility, Sucker Punch
  risk), so treat the rate as an upper bound.
- **Examples (M1):**
  - pair 51 g0 T6: Rhyperior Stone Edge (misses) into Bisharp, Earthquake available.
  - pair 66 g0 T20: Komala Body Slam into Iron Thorns, Earthquake / Knock Off available.
  - pair 42 g0 T19: Lokix 121/251 Leech Life into Altaria, Sucker Punch / Knock Off available; Lokix KO'd by Brave Bird.

### 4. Choice-locked into a resisted / immune / status move and stayed in (P4) (in-process; association)
- **Definition:** a voluntary decision while choice-locked, where the locked move is immune, resisted on an opponent at
  ≥ 30% HP, or a status move, and a switch was possible. Hit = stayed and used it.
- **Rate:** stayed in 30.7% of such situations (452/1472, 0.15 per game). M1 losses 33.3% [31.0, 35.7] vs wins 25.9%
  [23.6, 28.3]. Association −0.030 ± 0.018, so ≈ 0.45 point.
- **vs BC-P3:** 27.3% vs 32.9% (M2), so we are not worse than human imitation.
- **Examples (M1):**
  - pair 45 g0 T13: Shaymin-Sky locked into Seed Flare vs full-HP Lugia, which Calm Minds.
  - pair 30 g1 T14: Keldeo 69/273 locked into Hydro Pump vs Azumarill; KO'd by Liquidation.
  - pair 21 g1 T21: Typhlosion locked into Fire Blast vs Palkia-Origin (¼).

### 5. "Failed-move" blunders: about 2× BC-P3's rate, but rare (in-process; EVIDENCE of a defect, small impact)
- **Definitions** (each is a move that fails or does nothing by rule, from information the net can see):
  - P1s: a status move into an already-statused or type-immune target, or into a Substitute, and the opponent stayed in.
  - P11: Substitute at ≤ 25% HP or while already behind one.
  - P12: a self-heal at full HP.
  - P14: consecutive Protect.
  - P10b: Rest at full HP.
- **Rates, our net vs BC-P3 (M2):**
  - P1s 4.7% vs 1.9% of status moves.
  - P11 9.2% vs 5.1% of Substitutes.
  - P12 3.9% vs 2.5% of heals.
  - P14 7.3% vs 9.5% of Protects (not worse).
  - P10b 2.0% vs 1.9%.
- **In M1:** P1s 3.5% of status moves, losses 5.2% vs wins 3.7%. P14 losses 11.0% vs wins 5.8%.
- **Frequency:** together ≈ 0.2 per game per side. Association slopes are small (P1s −0.022 ± 0.020), so the total is
  < 0.5 point. It is a clear "the net doesn't model the rule" signature in the MissingNo.Zero sense, but cheap to
  ignore for now.
- **Examples (M1):**
  - pair 81 g0 T34/35/37: Quagsire uses Toxic into Steel Bronzong three times (immune each time).
  - pair 195 g1 T3–5: Paldean Tauros uses Substitute three turns running while already behind a Sub.
  - pair 51 g1 T16–17: Avalugg Recovers at full HP twice.

### 6. Candidates with NO evidence of a weakness (tested, null or not worse than BC-P3)
- **Switching into a revealed super-effective attack** (P5), when an alternative without one existed:
  - voluntary 4.7% of switches (BC 4.0%);
  - forced (after a faint) 12.6% (BC 12.5%); M1 wins 12.9% vs losses 12.3%.
- **Staying in at < 50% HP, slower, vs a revealed SE attack** (P6): KO'd before moving in 60.8% of these stays. M1
  losses 66.2% vs wins 53.3%; BC 65.8%. Sacks can be intended, so this is not counted as a mistake.
- **Attacking into a type immunity while the opponent stays:** 0.8% of attacks (0.15 per game; BC 0.4% vs ours 0.4%
  in M2). Ability immunities: 0.1%.
- **Hazards:**
  - Stealth Rock available and not up in the first 10 turns, not set: 34.6% vs BC 47.8% (we set rocks more often).
  - Removal available with own hazards up, not used: 54.5% (BC 55.3%).
  - Wasted hazard: 0.8–2.7%.
- **Sleep:** asleep (not waking this turn), switch possible, stayed and clicked a move: 54.4% (BC 56.7%). M1 losses
  60.3% vs wins 48.2% (likely reverse causality).
- **Tera** (M1):
  - Tera'd in 90.3% of games, median turn 15, by turn 3 in 10.0%, at HP < 30% in 5.2%.
  - The Tera'd mon fainted within one turn in 27.6% of games.
  - Lost without ever Tera'ing: 5.3% of losses.
  - BC-P3 Teras less (64%, median turn 20) and its Tera'd mon dies within a turn more often (43.8%).
  - No wasted-Tera signature.
- **Long games / endgames:** our best phase vs Foul Play (≥ 40 turns: 68–69%).
- **Fake Out after the first turn:** 1.8% of Fake Outs (3 cases).
- **Side asymmetry (weak lead, likely chance):** in M1 (the same net on both sides) p1 and p2 differ on three of
  about 28 patterns:
  - type-immunity attacks 1.0% vs 0.7% (A − B +0.052 ± 0.044 per game);
  - P1s 3.5% vs 5.6%;
  - Defog with nothing to clear 25/171 vs 5/164.

  With ~28 tests, 1–2 nominal hits are expected by chance. A cheap check would be fastenv v2 encoder parity for side 2.

## Evidence vs speculation
- **Evidence:**
  - The game-length profile vs Foul Play: 1000 + 1400 real games, the pair-level version, and the self-play and BC
    controls.
  - The in-process pattern rates, counts and BC comparisons: exact simulator, true states.
- **Association, not cause:** every "losses vs wins" contrast and the impact column. Points-per-occurrence slopes
  come from self-play game-level regressions.
- **Speculation:** that the fast-game deficit is tactical calculation (KO races, setup or speed timing). The FP logs
  hold no per-turn data to test it, and the in-process correlates only point that way.
- **Limit:** the in-process patterns are measured against our own net and BC-P3, not against Foul Play's choices.

## Suggested next steps (general strength only)
1. **Read LA2-FP (1173671) by game length.** Descriptive, not a new decision. If search repairs tactics, its gain
   over the raw net should sit in the games under 25 turns.
2. **Causal cost of P8 / P3 / P4 by counterfactual replay in fastenv.** Clone the state at flagged decisions, play the
   flagged move vs the best alternative K times with the net, and read Δ win probability. Cheap and no slot needed;
   needs a battle-clone hook in spd_host.js (PS `State.serializeBattle`).
3. **If per-turn FP data is wanted:** a logged Showdown run vs Foul Play (vs_foulplay with decision logging). Needs a
   slot and a prereg.
4. **Optional v2diag metric:** the failed-move rate (§5) per snapshot. It should fall as the net learns the rules.
