# LA2-FP by game length, and where lookahead2 overrides (fastenv, 2026-11-19; POST HOC, informational)
**Status: post hoc and informational.** LA2-FP's pre-registered decision (`LA2-FP-PREREG.md`, md5 1173284f) is
unchanged: **NOT SHOWN**, paired gain +1.30 [−2.71, +5.31] over 500 team pairs. Nothing below is a decision. It was
asked for after wsp's loss analysis (`$WSP/losses/REPORT.md` md5 336c0f99) found that our raw net wins 48.4% of games
shorter than 25 turns against Foul Play and 62.9% of longer ones. The question: does lookahead2's paired gain come
from short games, as that weakness would predict?

- Data: `$C/runs/la2-fp/main-run`. Arm 1 = lookahead2(G9q-cd@805.03M, δ 0.005, K 8); arm 2 = raw cd@805 at T 0.5.
  Both play Foul Play gen9 on the same 500 team pairs (block j seeded `la2fp20261116b<j>`, pair id within the block).
- Script: `scripts/fastenv/la2_fp_by_length.py` (ws/fastenv). Output: `$C/analysis/la2-fp-by-length/result.{txt,json}`.
- Intervals: the 95% intervals are 1.96·sd/√n over team pairs for pair scores and paired differences, Wilson intervals
  for per-game rates, and Wilson intervals over decisions for override rates (decisions cluster within games, so these
  are a little narrow).

## 1. Paired gain by game length
**How pairs are classed.** A team pair is short if the mean length of all four of its games is under 25 turns. The
four games are both arms and both sides. Because the class uses both arms the same way, the paired difference has mean
0 in every class when the arms are equal. Classing by one arm's own lengths would bias the result: short games are
mostly that arm's losses, so regression to the mean would favour the other arm in that arm's short pairs. Game length
also depends on how the players play, so these splits show associations, not effects.

| pairs by mean length (4 games) | pairs | lookahead2 vs FP | raw vs FP | paired lookahead2 − raw |
|---|---|---|---|---|
| all | 500 | 58.2 ± 2.9 | 56.9 ± 3.0 | **+1.3 ± 4.0** (the verdict's number) |
| < 25 turns | 133 | 51.9 ± 5.6 | 54.9 ± 5.5 | **−3.0 ± 7.7** [−10.7, +4.7] |
| ≥ 25 turns | 367 | 60.5 ± 3.4 | 57.6 ± 3.5 | **+2.9 ± 4.7** [−1.8, +7.6] |
| difference, short − long | | | | **−5.9 ± 9.1** |

Finer bins of the pair's mean length (paired lookahead2 − raw): [0, 20) 31 pairs −3.2 ± 13.6; [20, 25) 102 pairs
−2.9 ± 9.2; [25, 30) 133 pairs +0.0 ± 7.4; [30, 40) 161 pairs +2.8 ± 7.4; ≥ 40 73 pairs +8.2 ± 10.3.

Per game (each arm's own game lengths; descriptive):

| arm | games < 25 turns | win rate | games ≥ 25 | win rate | median / mean turns | losses (in games < 25) |
|---|---|---|---|---|---|---|
| lookahead2 | 346 | 48.6% [43.3, 53.8] | 654 | 63.3% [59.5, 66.9] | 28 / 31.0 | 418 (178) |
| raw | 330 | 47.6% [42.2, 53.0] | 670 | 61.5% [57.8, 65.1] | 28 / 31.1 | 431 (173) |

On the same team pairs, lookahead2 minus raw: share of games under 25 turns +1.6 ± 3.7 points; mean game length
−0.09 ± 0.95 turns.

**Reading.**
- **No.** The gain is not concentrated in short games. The point estimates go the other way: −3.0 in short pairs and
  +2.9 in long ones, rising with length. Every interval includes 0, the short-minus-long difference too (−5.9 ± 9.1),
  so the data cannot place the gain anywhere.
- lookahead2 keeps the raw net's short-game weakness (48.6% vs 47.6% in games under 25 turns), and its games are no
  longer or shorter than the raw net's.
- So one turn of lookahead with leaves scored by our own net does not repair the KO races that wsp's report blames for
  the short-game losses (its stated mechanism is speculation). Either the losing pattern cannot be seen within one turn,
  or the value head that scores the leaves shares the raw policy's blind spot.
- This matters for an expert-iteration pilot that uses lookahead2 as the teacher (`EXIT-LA2-DESIGN.md`). It may make
  the net better against itself, as lookahead2 is, without targeting the Foul Play weakness.

## 2. Where lookahead2 overrides the net (arm 1, 34,856 searched decisions)
Overall, lookahead2 overrode the net on 19.1% of searched decisions (6,651). The verdict's 19.08% is per decision over
all of them. The median gap Q(a_L) − Q(a_net) on an override is 0.022.

| turn | searched | override rate |
|---|---|---|
| 1 | 1,093 | 17.3% [15.2, 19.6] |
| 2–5 | 4,540 | 18.9% [17.8, 20.1] |
| 6–10 | 5,756 | 19.6% [18.6, 20.6] |
| 11–20 | 10,903 | 20.1% [19.3, 20.8] |
| 21–30 | 7,056 | 17.7% [16.8, 18.6] |
| ≥ 31 | 5,508 | 18.9% [17.9, 20.0] |

Each override as (the net's draw → what lookahead2 played), with "tera" = a move with Terastallization (actions 22–25):

| net → played | overrides | share |
|---|---|---|
| move → another move | 2,770 | 41.6% |
| switch → another switch | 1,551 | 23.3% |
| switch → move (stay in) | 917 | 13.8% |
| move → switch | 658 | 9.9% |
| tera → move (Tera held back) | 346 | 5.2% |
| move → tera | 198 | 3.0% |
| tera → switch | 114 | 1.7% |
| switch → tera | 53 | 0.8% |
| tera → tera (another move) | 44 | 0.7% |

- **Turns.** The override rate is flat across the game, between 17% and 20% in every turn band. lookahead2 does not
  act mainly early or late.
- **Kind of decision.** 65% of overrides change a move to another move or a switch target to another. 24% change
  whether to switch: switch → stay in 917 times, stay in → switch 658. Tera: lookahead2 held back a Tera the net had
  drawn 460 times (5.2% + 1.7% of overrides) and added one 251 times (3.0% + 0.8%). That is a mild lean against
  terastallizing, about 1% of searched decisions.
- **Game length and result.** Overrides are slightly rarer in short games (17.7% [16.8, 18.5] vs 19.5% [19.0, 20.0]
  in long ones). They are more frequent in games lookahead2 lost (21.6% [20.9, 22.3] vs 17.4% [16.9, 17.9] in wins).
  Both are associations: lost positions are more contested, and the search does not act more where the losses
  concentrate.
- **Cost (for the ExIt design).** Wall time per searched decision: median 0.48 s, mean 0.57 s, q90 1.07 s, max 6.7 s
  (3 concurrent battles, one 11-worker pool on the slot). Leaves per decision: median 1,542, mean 1,992, q90 4,421.
  World expansion: 1.26 core-s per decision on average (median 1.02). Leaf scoring: 0.22 s per decision (forward
  0.19 s, about 10.7k leaves/s on the slot's GPU). 34.9 searched decisions per game.
