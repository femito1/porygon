# ExIt-LA2 pilot: distil lookahead2 into G9q-cd@805M (fastenv, 2026-11-19; DRAFT, not binding until main signs it off)
**Status: DRAFT.** It becomes binding and read-only when main signs it off. At that point a byte-identical copy goes
to `$C/runs/exit-la2/PREREG.txt` (r--) with its md5 in `PREREG.md5`. Records and deviations after the sign-off go in
`$C/runs/exit-la2/NOTES.txt`, never here. Reasons: `EXIT-LA2-DESIGN.md`.

## Question
Fine-tune the deployment net cd@805 toward lookahead2's played policy, on 120k self-play positions. Does that make the
raw net stronger than (T1) the same fine-tune toward the net's own policy and (T2) the untouched net?

## Fixed inputs
- **Start net S** = `$C/nets/g9q-cd-805.03M.pt` (sha256 e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f).
  It is loaded through the frozen netv2 tree `$C/frozen/netv2-fba02fb-20261110`.
- **Teacher** = lookahead2 at code 36a2006 (LA2-FP/H's frozen `porygon/lookahead2.py`; its sha256 is recorded at the
  stage-1 checks). Net S, K 8, δ* = 0.005 (`$C/runs/la2-delta800/DELTA`), T 0.5, fp32 leaves on the GPU, 11 workers.
- **Hardware:** one fastenv helper job (`--partition=training`, 12 CPUs, 1 GPU, nice 500). No slot and no VM for
  stage 1.

## Data
1. **Self-play.** S vs S, both sides at T 0.5. fastenv in-process with G9q's training set library; the commit and
   library hash are recorded. 60,000 finished games, seed 7191101. Ties, turn-cap games and crashes are dropped and
   replaced.
2. **Positions.** Per game, one decision per side, each uniform over that side's decisions with ≥ 2 legal actions,
   chosen by sha1(seed, game, side). That gives 120,000 positions.
3. **Labels.** Each position's F2-style record is replayed (netsearch_study.replay_targets) and run through one
   lookahead2 call. Stored per row: v2 obs, mask, p (T-1 probabilities), R, Q, a_L, π_LA, z (result for the side to
   move), game id.
   - π_LA(a) = Σ_a' p₀.₅(a')·1[g(a') = a], with p₀.₅ ∝ p².
   - g(a') = a_L iff a_L ≠ a', a' ∈ R and Q(a_L) − Q(a') > δ*; else a'.
   - A lookahead2 failure or fallback ⇒ the row is dropped from both arms (counted).
   - Parity: the replayed p must match the self-play forward's p to ≤ 1e-5, else the row is dropped (counted).
4. **Split.** pretrain.is_val(game, 0.05): held out (H) vs train.
5. **Targets.** E: t_E = normalize(√π_LA). C: t_C = normalize(√p₀.₅), which equals p. Both use the same rows, the same
   obs and z, and the same shard order.

## Fine-tunes (netv2 `porygon.v2.pretrain` at a netv2 commit pinned in the stage-1 checks)
- Arguments: `--init S --gen 9 --epochs 2 --batch-size 512 --lr 1e-5 --warmup 50 --weight-decay 0.01 --value-coef 0.5
  --aux-coef 0 --val-frac 0.05 --seed 0 --no-bf16`, defaults otherwise. The net used is final.pt, with no selection.
- E and C differ only in `target`.
- The stage-1 checks record S's saved trainer lr and cooldown factor. If S's last lr is not 1e-5 ± 50%, the chain
  stops for main. It does not change lr itself.
- Each output must load through fba02fb and play one fastenv game, else FAILED.
- **Manipulation check (on H, forward passes only, before any game):**

```
                          mean KL(π_LA ‖ p₀.₅ of E)
    r  =  1  −  ──────────────────────────────────────
                     mean KL(π_LA ‖ p₀.₅ of S)
```

  **If r < 0.25, E and C are both re-run once with `--lr 3e-5`**, and that pair is tested. This happens at most once
  and is recorded.
- **Also reported on H (information only):**
  - r for C;
  - for E and C, the top-1 agreement with π_LA's argmax on the positions the gate overrides;
  - the value log-loss and AUC on z for S, E and C.

## Stage 1 test (in-process, `porygon.fastenv_eval`, duplicate pairs, every net `v2:PATH,temperature=0.5`)
| test | games | seed | role |
|---|---|---|---|
| T1: E vs C | 16,000 (8,000 pairs) | 7191111 | **decision of record** |
| T2: E vs S | 16,000 (8,000 pairs) | 7191112 | decision |
| T3: C vs S | 8,000 (4,000 pairs) | 7191113 | information |

- Interval: 1.96·sd/√n over pair means. Fixed n, no interim looks.
- **Verdict**, `$C/runs/exit-la2/VERDICT.txt` (r--), line 1 is one of:
  - **GO**: T1 lower 95% > 50% and T2 lower 95% > 50%.
  - **MIXED**: T1 lower > 50% and T2 lower ≤ 50% ⇒ main decides.
  - **HARMFUL**: T1 upper 95% < 50%.
  - **FLAT**: T1 upper 95% < 51%, not HARMFUL.
  - **NOT SHOWN**: anything else.
  - **VOID**: a stage FAILED; or < 110,000 usable labelled rows; or parity drops > 0.5%; or lookahead2 fallbacks
    > 2% of positions ⇒ fix, then replay with the same seeds (main's OK).
- **Reported:** T1–T3 ± CI, r (and whether the dose retry ran), label counts and drops, the override mass of π_LA
  (mean Σ moved), hours per stage.

## Stage 2 (only on GO; slot claim in the ledger with main's OK; fixed n)
- Raw E (T 0.5) and raw S (T 0.5) each vs `foulplay:ms=1000,parallelism=1`, 500 pairs per arm, the same team pairs.
- Harness: LA2-FP's Stage 3b, October server and library. Tag `exla2fp<date>` (grep at the claim); 5 rounds × 2 arms
  × 100 pairs, rotated.
- Decision: E − S paired (`netsearch_slot.py paired RUN 1 2`):
  - **CARRIES OVER** iff lower 95% > 0;
  - **REVERSED** iff upper 95% < 0;
  - else **NOT SHOWN**.
- Void: HARNESS_FAIL, or < 450 pairs on an arm ⇒ replay, same seeds.

## What follows (main's decision; each step needs its own prereg)
- **GO and not REVERSED** ⇒ iterate:
  1. lookahead2(E, δ* re-derived) vs lookahead2(S) on Showdown;
  2. round 2 with lookahead2(E) as the teacher;
  3. the online PPO `--distill` form.
- **GO and REVERSED** ⇒ stop and report.
- **FLAT or HARMFUL** ⇒ stop the route and report as a negative result.
- **NOT SHOWN or MIXED** ⇒ main chooses: 3× the positions, or a redesign.

## Budget
- Stage 1: one helper chain, about 12–14 h wall (labels about 9.5 h at the measured 3.5 decisions/s). About 170
  CPU-core-h; about 14 GPU-h allocated, mostly idle. Time limit 30 h.
- Stage 2: about 15–16 h of the slot.
- VM: 0 h.
