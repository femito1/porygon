# Expert iteration with lookahead2 as the teacher: pilot design (fastenv, 2026-11-19)
The binding version is `EXIT-LA2-PREREG.md` (a DRAFT until main signs it off). This file gives the reasons. Main
asked for it on 11-19 (ORCHESTRATOR "11-19 10:00Z", decision 2): the scale gate came back FLAT, and lookahead2 is the
only search with a measured gain. Can the net learn what lookahead2 adds, so that the raw net improves and so does
lookahead2 on top of it?

## 1. Why now, and what is different from the attempts that failed
- **The teacher beats the student.**
  - lookahead2(δ*, K 8) vs its own raw net: 53.17 [50.59, 55.73] (Bet 1, cd@605).
  - vs the old benchmark hybrid: 54.00 [51.18, 56.82] (LA2-H, cd@805).
  - The past search-label attempts had a teacher weaker than the net:
    - bc-12: poke-engine search at 4×500 ms as BC targets scored 34.9% vs human BC.
    - g9s-06/07: neither a prior nor 2× time made that teacher better.
- **The target is the measured player's policy, not visit counts.** The target is π_LA, the exact distribution of
  the move lookahead2 plays (§2). That is the player that won Bet 1 and LA2-H. The equilibrium x and softmax(Q) were
  never tested in games.
  - Visit counts from short searches are not policy improvements (research REPORT §5, Gumbel).
  - Anthony et al. (Hex) found tree-policy targets worth +50 ± 13 Elo over chosen-move targets.
- **Enough decorrelated positions.** 120k positions from 60k independent self-play games (§3). The 2026-10 ExIt round
  (v1-12, tied its source) had 9k battles. The research report's rule is "ExIt needs ≥ 10⁵ decorrelated positions".
- **A control isolates the teacher.** Arm C is the same fine-tune on the same positions with the start net's own
  policy as the target. Any E − C difference is the teacher's.
- **Caveat from LA2-FP** (`LA2-FP-BY-LENGTH.md`). lookahead2's gain over the raw net did not carry over to Foul Play
  (+1.3 [−2.7, +5.3]). It is also not in the short games where our Foul Play losses concentrate (−3.0 ± 7.7 in short
  pairs, +2.9 ± 4.7 in long ones). Distilling lookahead2 should therefore make the net stronger against itself. It is
  not expected to fix the Foul Play weakness, and stage 2 (§6) only checks that.

## 2. Teacher and target
- **Teacher.** lookahead2 exactly as benchmarked: `porygon.lookahead2` at the frozen LA2-FP/H code (36a2006); net
  G9q-cd@805.03M (sha256 e93ac3f8…b11f); K 8 worlds from `search.convert`; δ* 0.005 (`$C/runs/la2-delta800/DELTA`);
  fp32 leaves on the GPU. It has no time budget, so the same position and worlds give the same decision offline as in
  a game.
- **One run per position** gives: p, the net's T-1 probabilities over legal actions; p₀.₅ ∝ p², which is what the raw
  player draws from; and the rows R, Q(a) and the lookahead's own move a_L.
- **π_LA, exact over the net's T 0.5 draw.**

```
  π_LA(a) = Σ  p₀.₅(a') · 1[ g(a') = a ]
            a'

  g(a') = a_L   if a_L ≠ a', a' ∈ R and Q(a_L) − Q(a') > δ*
          a'    otherwise (Q(a') undefined, or a failure ⇒ the net's draw, as in play)
```

  π_LA is the distribution of what lookahead2 plays at that position, given its worlds. It moves mass only from draws
  the gate overrides to a_L. In LA2-FP that is 19.1% of the mass on average.
- **Why π_LA.** It is the teacher whose strength we measured. The δ* rule was set so that ≥ 90% of the
  draw-weighted overrides are confirmed by a judge with 4× more worlds (K 32). That makes the target's changes
  reliable: they are not world-sampling noise. Overrides below δ* are mostly noise, so x (δ = −∞) and Q-softmax targets
  would teach that noise. A one-hot of the played move would be a noisier sample of π_LA.
- **Temperature.** The net is deployed at T 0.5, but pretrain's CE acts on the T-1 softmax. The T-1 target is
  therefore t_E = normalize(√π_LA): a net whose T-1 policy equals t_E plays exactly π_LA at T 0.5. No trainer change
  is needed.
- **Control target.** t_C = normalize(√p₀.₅) = p, the start net's own T-1 policy. At the start weights the CE to t_C
  has zero policy gradient, so arm C moves only through what E and C share: the value loss, id dropout and the
  optimizer. That is the "same fine-tune on the net's own policy targets".
- **The anchor is in the target.** CE(t_E, π_θ) = KL(t_E ‖ π_θ) + const, and t_E equals the start policy wherever the
  gate keeps the net's draw. Most of each target (about 81% of the mass on average) therefore anchors to the start
  net, so a separate KL term would add a second free weight for little. Drift outside the labelled positions is
  limited by the small lr and 2 epochs, and C measures it.
- **Value.** Both arms keep BCE on the game result z, with pretrain's default weight 0.5. That keeps the value head
  honest under a moving trunk, which matters because the next round's lookahead2 scores leaves with it. Distilling
  lookahead2's root value would be a second change; it is kept for round 2. Aux losses are off (aux_coef 0, aux
  fields absent ⇒ pretrain fills them). Play does not use the aux heads, and the setting is the same in both arms.

## 3. Positions
- **Source.** fastenv self-play of cd@805 against itself at T 0.5 on both sides, i.e. the raw benchmarked player. The
  games run in-process (exact Showdown mechanics) with the set library G9q trains on. Why this source:
  - It is the distribution the deployed net actually meets in self-play.
  - It is the same distribution as stage 1's in-process test.
  - It is cheap: NNUE gate 1 generated 139k cd@805 self-play games in 2.05 h on 5 workers, including heavy converters.
  - Human positions would answer a different question.
  - Foul Play games are excluded: there are few of them, and training on them would edge toward fitting one opponent.
- **Sampling.** Each game gives 2 positions, one per side. Each is drawn uniformly from that side's decisions with
  ≥ 2 legal actions, which is the set lookahead2 searches (turn decisions and forced switches). Seed-hashed.
- **Count and decorrelation.**
  - 60,000 games ⇒ 120,000 positions, each from an independent random team pair.
  - The two positions of a game share its outcome but come from different sides at independent turns. The policy
    targets do not share a label, and the value loss sees at most 2 positions per game, about AlphaGo's
    one-per-game fix.
  - Split by game (`pretrain.is_val`'s hash): 95% train (about 114k) and 5% held out (about 6k, for the manipulation
    and value checks).
- **From game to labelled row.**
  - fastenv records each finished game with host.js `--dump-states --dump-streams` through a FIFO reader. The reader
    keeps the G line (inputlog, true teams, both client streams) and the chosen decisions' message indices. It drops
    the state bodies (about 10 KB per decision).
  - Each chosen (game, side, decision) becomes an F2-style record, as in `nnue_parity.records`.
  - The record is replayed by `netsearch_study.replay_targets` (data's StreamReplayer: bit-identical to live play)
    into the poke-env battle lookahead2 needs.
  - The labeller stores the v2 observation from that battle, which is the input pretrain trains on, together with p,
    R, Q, a_L, π_LA, z and the game id.
  - Integrity check: the replayed battle's p must equal the self-play forward's p to ≤ 1e-5 on every labelled row.
    The labeller logs any miss and drops that row. More than 0.5% misses ⇒ FAILED.

## 4. Hardware and throughput (measured where possible)
| step | where | basis | estimate |
|---|---|---|---|
| self-play 60k games + records | fastenv helper (12 CPUs) | NNUE gate 1: 139k games in 2.05 h with converters | ≤ 1 h |
| label 120k positions | the same helper (12 CPUs + its GPU) | **la2-delta800 (measured):** 1,403 replayed positions × (K 8 + K 8 + K 32) = 8,418 K-8 runs in 40 min on a 12-CPU dove helper, so **3.5 K-8 decisions/s** | **≈ 9.5 h** |
| fine-tunes E and C | the helper's GPU | 2 epochs × 114k rows at batch 512 ≈ 450 steps per arm; BIGNET: v2-base step 0.15 s on a T4 | ≤ 0.5 h |
| manipulation and value checks | the helper | 6k held-out rows, forwards only | minutes |
| in-process h2h, 40k games | the helper (fastenv_eval) | VM: 25.7k games/h on 3 workers; the helper has about 3× the CPUs | ≈ 1–2 h |
| **stage 1 total** | one helper chain | | **≈ 12–14 h wall, ≈ 170 CPU-core-h, ≈ 14 GPU-h allocated (mostly idle)** |
| stage 2 (on GO only) | the benchmark slot (24 CPUs, GPU) | LA2-FP: raw cd@805 vs FP took 7.65 h per 500 pairs | ≈ 15–16 h for 2 arms × 500 pairs |

- LA2-FP's own decisions agree with the labelling estimate: 1.26 core-s of world expansion and 0.22 s of leaf
  scoring (10.7k leaves/s) per decision on average. On 12 CPUs that is about 8 decisions/s for the CPU and about
  4.5 decisions/s for the GPU, so the GPU sets the pace. The la2-delta800 rate already includes the replay cost.
- **Why the cluster helper and not the GPU Dojo VM.**
  - The labelling needs CPUs: 1.26 core-s per decision.
  - The VM has 4 vCPUs (about 3 workers), so it would run at roughly 2 decisions/s: about 17 h for the labels and
    about 20 VM-h in all, while its T4 mostly idles.
  - The helper has 12 CPUs, costs nothing from the 88 free VM-hours, and needs nobody to start it.
- **Fallback:** the VM, only if the training partition cannot start the helper within about 12 h of submission.
- The labelling uses fp32 leaves, as measured. fp16 would double GPU throughput but change the teacher.

## 5. How the targets enter training: a supervised distillation fine-tune
**Choice:** `porygon.v2.pretrain --init cd@805`, on shards in the CONTRACTS "v2 supervised shards" format:
`obs/*`, `target` = t_E or t_C, `z`, `game`. The settings are below; the PREREG binds them.
- lr 1e-5 peak, warm-up 50 steps, pretrain's cosine to 0.1×. That is G9q's trainer lr of 1e-4 × the cooldown's 0.1,
  the rate cd@805 was last trained at; the stage-1 checks verify it from cd@805's saved args.
- Batch 512, 2 epochs, final.pt (no selection), seed 0, fp32.
- value_coef 0.5, aux_coef 0. id dropout at the trainer default.
- E and C are identical except for `target`.
- **Pre-registered dose retry, decided before any game.** Manipulation check:

```
                          mean KL(π_LA ‖ p₀.₅ of E)
    r  =  1  −  ──────────────────────────────────────
                mean KL(π_LA ‖ p₀.₅ of the start net)
```

  Both means are over the held-out positions. If r < 0.25, E has barely absorbed the teacher. Then E and C are both
  re-run once at lr 3e-5, and that pair is the one tested. No other tuning.
- **Why not the PPO `--distill` aux loss first.** It exists in the v2 trainer (CE to fixed targets per minibatch,
  `--distill-reload-every` for a growing set).
  - In a PPO branch the distillation is confounded with RL steps and a moving policy. The targets are also off-policy
    for the moving net.
  - P4 showed that a head start from better data washes out under RL: G9-P4 vs G9q went from 51.8 to 50.5 at 245M.
    A one-off target set inside a long RL run would fade the same way.
  - It needs netv2's GPUs and lineage machinery, and a 5M cosine branch per arm.
  - The fine-tune changes one thing (the target), runs in minutes on the helper's GPU and leaves netv2's runs alone.
- **On GO, PPO becomes the scale-up form** (§7).

## 6. The test and the decision rule
- **Stage 1 (in-process, fastenv_eval duplicate pairs, both sides T 0.5, new seeds).** E, C and the start net S are
  played against each other:
  - T1 E vs C, 8,000 pairs (16,000 games). **This is the decision of record: the teacher's contribution.**
  - T2 E vs S, 8,000 pairs (16,000 games): is it better than the deployment net?
  - T3 C vs S, 4,000 pairs (8,000 games), information only: the fine-tune's own drift.
  - Power. Pair sd ≈ 32 points (G9-P4 vs G9q @105M, 2,000 pairs, ± 1.4), so 8,000 pairs give ± 0.70. If E keeps
    half of lookahead2's +3.2 (51.6% against S), T2 clears 50 with probability > 0.99, and T1 does as well if C ≈ S.
- **Verdict** (line 1 of `VERDICT.txt`):
  - **GO** iff T1's lower 95% > 50% and T2's lower 95% > 50%.
  - **MIXED** iff T1's lower > 50 but T2's is not (the fine-tune itself costs strength and the teacher only offsets
    it) ⇒ main decides (e.g. a different lr or anchor).
  - **HARMFUL** iff T1's upper 95% < 50%.
  - **FLAT** iff T1's upper 95% < 51% and not HARMFUL: the teacher adds < 1 point.
  - **NOT SHOWN** otherwise.
  - **VOID** iff any stage FAILED, labels < 110k usable, replay-parity misses > 0.5%, or lookahead2 fallbacks > 2%
    of labelled positions.
- **Stage 2 (only on GO; claims the slot with main's OK).** Raw E vs Foul Play and raw S vs Foul Play, paired on the
  same team pairs, 500 pairs per arm. It uses LA2-FP's arm-2 harness: Stage 3b, October server and library,
  `foulplay:ms=1000`, 5 rounds × 2 arms × 100 pairs.
  - **CARRIES OVER** iff the paired difference's lower 95% > 0; **REVERSED** iff its upper 95% < 0; else **NOT SHOWN**.
  - At 500 pairs this only detects a carry-over of about +4 points or a reversal. LA2-FP's +1.3 for lookahead2 itself
    says a NOT SHOWN is the likely outcome even after a real stage-1 gain. Stage 2 guards against a net that is better
    against itself but worse against an outside bot.

## 7. What each verdict leads to
- **GO and stage 2 not REVERSED ⇒ iterate.** Each step needs its own short prereg:
  1. *Does the better net make the search better?* lookahead2(E, δ* re-derived by `lookahead2_delta.sh`) vs
     lookahead2(S, 0.005) on Showdown, using the Stage 3 harness on the helper (as scale-gate chain B did), Bet 1's
     sequential design. This is the premise of expert iteration.
  2. *Round 2.* Teacher lookahead2(E), positions from E's self-play, the distilled E₂ against E and its own control.
     Same rule. About 14 h of helper time per round.
  3. *Online form.* A G9q cooldown branch (netv2) with `--distill` fed by a labeller job that keeps labelling the
     latest checkpoint (`--distill-reload-every`), so the teacher keeps feeding RL instead of a one-off set that fades.
     Its arms are coef 0 and coef C on the same branch.
  4. Optional in round 2: lookahead2's root value as a value target (KataGo/ExIt-style), as its own change.
- **GO but REVERSED ⇒ stop.** The distilled gain does not transfer to an outside opponent, so report it and do not
  iterate.
- **FLAT or HARMFUL ⇒ stop the route.** The one-ply improvement is either already in the net or not learnable from
  10⁵ positions by this fine-tune. That is a reportable negative: a teacher stronger than the student still did not
  transfer.
- **NOT SHOWN or MIXED ⇒ main decides.** Options: 3× the positions (about 28 h of labelling), or the dose retry if it
  was not used.

## 8. Risks
- **Train/serve sets.** The positions use G9q's training sets (November). lookahead2's world sampler uses the in-tree
  October library (md5 324d8e97), as in every lookahead2 run, and stage 2 uses the October server. Both arms have this
  deviation, and it is recorded.
- **Self-play bias.** E learns lookahead2's corrections against cd@805's own play. If they are opponent-specific,
  stage 2 shows it. Item 1 already says lookahead2 does not fix the short-game losses against Foul Play.
- **Sharpening.** π_LA is sharper than p₀.₅ on overridden positions. The √ transform keeps the deployed T 0.5 policy
  equal to π_LA rather than sharper. C's T3 against S checks that the procedure itself is neutral.
- **The in-process gain may not carry over** (re-freeze #1: an in-process 59.6% gave no hybrid gain). That is why
  stage 2 exists, and why the iterate step 1 measures with Showdown search.

## 9. Implementation (built 11-19 after the sign-off; ws/fastenv, frozen into `$C/runs/exit-la2/code`)
- `scripts/fastenv/exit_la2.py` holds every step as a subcommand:
  - `gen`: fastenv self-play using nnue_gen's batched-forward loop. One reader process per worker keeps the G lines
    and the S-line prefixes, then picks the two positions and writes gzip'd F2-style records with the self-play p.
  - `label`: replay, then the p-parity check, then the v2 obs and one lookahead2 call, then π_LA. Writes npz parts,
    meta and drops, and resumes from what is already there.
  - `shards`: the E and C shard sets.
  - `check`: r, top-1 on moved positions, value log-loss and AUC.
  - `eval`: porygon.fastenv_eval between two v2 nets.
  - `verdict`: the PREREG rule.
- `scripts/fastenv/exit_pilot.sbatch smoke|main`: the stage-1 chain, one helper job.
  - Order: checks → smoke → gen → label → shards → fine-tunes → load checks → check (+ dose retry) → T1–T3 → VERDICT.
  - The fine-tunes run the frozen fba02fb tree's pretrain.
  - Self-play uses the November sets; the in-process tests use October (`NOTES.txt`).
- Tests: `tests/test_exit_la2.py` covers the exact π_LA (the gate, an undefined Q, margin ∞), the √ transform and
  t_C = p, KL, the verdict rule and the S-line parse. A login-node dry run on 4 games covered every subcommand.
