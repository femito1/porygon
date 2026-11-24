# Council test P2: ExIt with a soft completed-Q target, offline on the ns-27 data (fastenv, 2026-11-24)
**Status: BINDING.** Main (the orchestrator) signed off the design and the rule on 2026-11-24. They are copied
verbatim below. A byte-identical copy is at `$C/council/p2-exit-soft/PREREG.txt` (r--), with its md5 in `PREREG.md5`
(`C=/vol/bitbucket/fr1226/porygon-ws-fastenv`). Records and deviations after arming go in
`$C/council/p2-exit-soft/NOTES.txt`, never here. Origin: council session 1, `docs/council/sessions/2026-11-24/PORTFOLIO.md`
P2. The ExIt-LA2 design and prereg are `EXIT-LA2-DESIGN.md` and `EXIT-LA2-PREREG.md` (a8d0fb5d).

## Main's design and rule (signed off 2026-11-24; verbatim)

## The item: council test P2 — ExIt with a soft completed-Q target, offline on the ns-27 data
**Question:** ExIt-LA2 stage 1 was HARMFUL (ns-27: E vs C 48.17 [47.47, 48.87]; ns-28: E vs S 48.14). Did that harm come from its hard-gated target? That target set gate-overridden actions to exactly 0 and renormalised √π_LA (`EXIT-LA2-DESIGN.md:38-46`). Test: does a zero-avoiding soft target on the SAME data make the student better?

1. **Data check first.** The 120k ns-27 positions must store, per position, the parent's raw policy p (T 1), the legal mask, lookahead2's per-action values Q_LA(a) for our legal actions, and V_LA (or enough to compute it as x·Q_LA). If per-action Q values are NOT stored, STOP and report: re-labelling would be a deviation needing main's OK. Also confirm that ns-29's saved control C and the parent S = G9q-cd@805.03M are on disk (md5s).
2. **Single change = the target.** Use π_imp(a) ∝ p(a)·exp(β(Q_LA(a) − V_LA)) over legal actions. Everything else is identical to ns-27's E: parent cd@805.03M, positions, epochs, optimiser, lr, batch, aux-coef, seeds.
3. **β, pre-registered before training.** On a fixed 10% calibration split of the training positions (no outcomes used), bisect β so that the mean KL(π_imp ‖ p) = 0.05 nats. Record β and the split in the PREREG before the fine-tune starts.
4. **Evaluation.** In-process `spd_h2h`, T 0.5, **4,000 fresh duplicate pairs each** (new pre-registered seeds): E′ vs C (ns-29's control) and E′ vs S (cd@805.03M).
5. **Decision rule (verbatim): GO iff E′ vs C pair-mean lower 95% > 50.5 AND E′ vs S pair-mean lower 95% > 50.0; else NO-GO, which closes the ExIt-LA2 line in both forms (hard and soft target).**
6. **Device.** Your one helper job (training partition, with a GPU), or the data host via `srun --overlap` if that is the stream's practice. ≈ 1.5 GPU-h for the fine-tune + ≈ 0.5 h for the 16,000 games. Cluster-side chain: calibrate β → fine-tune → h2h ×2 → verdict.
   - Files in `$C/council/p2-exit-soft/`: PREREG.txt (r--), STATUS, VERDICT.txt (line 1 GO | NO-GO, then both pair-means with 95% CIs), FAILED on any failure (the check-in watches these paths).
7. **PREREG** `docs/netsearch/P2-EXIT-SOFT-PREREG.md` = `$C/council/p2-exit-soft/PREREG.txt`, with its md5.

## Data check (done 2026-11-24 ~13:48Z, before this file was sealed): PASS, no re-labelling
- **Positions:** ns-27's labels `$C/runs/exit-la2/labels`: 60 parts, 119,967 rows. The drops are ns-27's (14 parity,
  33 lookahead2). Row j of `label-*.npz` is line j of its `.meta.jsonl`.
- **Stored per row:**
  - npz `probs`: p, the parent's T-1 policy on the replayed battle (parity ≤ 1e-5 with the self-play forward).
  - npz `obs/action_mask`: the legal mask.
  - npz `pi` (π_LA), `z`, `game`.
  - meta `rows`: our actions with a lookahead row. meta `q`: Q_LA per row. meta `x`: lookahead2's root strategy.
  - meta `value`: V_LA = x·Q_LA. Also meta `a_l` and the record `id`.
- **Q coverage:** Q_LA is stored for 1,046,294 of the 1,048,379 legal actions (99.80%). 427 rows have a legal action
  without a row.
- **Alignment:** max |x·Q − V| is 4.0e-6 (the meta's 6-decimal rounding). Recomputing the moved mass |π_LA − p₀.₅|/2
  from the npz gives the meta's to 5.4e-7 (rows aligned). p is 0 off the mask, and |Σp − 1| ≤ 5e-8.
- **Split:** pretrain.is_val(game, 0.05) keeps 5,776 rows held out (H). The other 114,191 are the training positions,
  as in ns-27's E log.
- **Nets on disk:**
  - S = G9q-cd@805.03M: `$C/nets/g9q-cd-805.03M.pt`. sha256 e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f,
    md5 d0c5b6cfc1c5c7173e1e05111cd28e14.
  - C = ns-29's saved control: `$C/runs/exit-la2/ft-C/final.pt`, md5 6767993369b66e7031722d8479020806.
  - E = ns-27's arm E (used only in the information check): `$C/runs/exit-la2/ft-E/final.pt`, md5
    8097eafd1e9fed73dbdfe85b4efaa78b.

## Implementation (fastenv; fixed before the fine-tune)
Code: `scripts/fastenv/p2_exit_soft.py` (calibrate / shards / check / verdict), `scripts/fastenv/p2_exit_soft.sbatch`,
`tests/test_p2_exit_soft.py`. They are frozen as a git archive of the ws/fastenv commit that adds this file, in
`$C/council/p2-exit-soft/code` (with a COMMIT file).

### Target (the single change)
- **Completed Q.** Q̂(a) = Q_LA(a) for an action with a lookahead row. Q̂(a) = V_LA for a legal action without one,
  i.e. advantage 0, as Gumbel MuZero completes the Q of unvisited actions.
- **π_imp.** Over the legal actions, in float64:

```
                     p(a) · exp(β (Q̂(a) − V_LA))
    π_imp(a)  =  ─────────────────────────────────
                 Σ_b p(b) · exp(β (Q̂(b) − V_LA))
```

  It is zero only where p is zero. V_LA is constant per position, so it cancels.
- **The target.** Arm E′'s T-1 policy target is t_E′ = π_imp, stored as float32 `target`; pretrain renormalises it over
  the mask. For comparison, ns-27 used t_E = normalize(√π_LA) and t_C = p.
  - A net whose T-1 policy equals π_imp plays ∝ p₀.₅·exp(2β(Q̂ − V_LA)) at T 0.5.
- **Formal check.** A unit test checks supp(π_imp) = supp(p) and E_π_imp[Q̂] ≥ E_p[Q̂]. It says nothing about win rate.

### β (pre-registered here)
- **Calibration split:** the training positions (not is_val(game, 0.05)) with
  int(sha1("p2-exit-soft-cal:" + record id).hexdigest()[:12], 16) % 10 == 0. That is 11,478 rows. Only p, the rows,
  Q_LA and V_LA are read; no outcomes.
- **Bisection:** 100 steps on β ≥ 0, on the exact mean KL(π_imp ‖ p) = β·E_π_imp[A] − log Σ p·exp(βA). It is increasing
  in β, because its derivative in β is β·Var_π_imp(A).
  - Target 0.05 nats gives β = 35.90034. **β = 35.9003** (recorded value, used by the chain). The mean KL at
    the recorded value is 0.0499999.
- **The chain recomputes β** from the frozen code at its first stage and stops (FAILED) unless it reproduces 35.9003
  (relative 1e-4).
- **Information:**
  - On the calibration rows, the mean of E_π_imp[Q̂] − E_p[Q̂] is 0.0034 (min 0).
  - ns-27's hard target has a mean KL(t_E ‖ p) of 0.757 nats on the same rows, about 15× this dose.

### Shards
- **How they are built:** ns-27's arm-E shards `$C/runs/exit-la2/shards/E/shard-NNNNN.npz`, with only `target`
  replaced by π_imp. obs, z, game and the shard order are unchanged.
- **Per-shard check:** each shard is checked against its label part. game and the mask must be equal, and ns-27's
  t_E = normalize(√π_LA) must be bit-identical. The data check above is repeated.
- **Scratch:** `$D/shards/Ep`, deleted after the fine-tune and the information check.

### Fine-tune (identical to ns-27's E)
- **Arguments:** `porygon.v2.pretrain` from the frozen tree `$C/frozen/netv2-fba02fb-20261110`:
  `--shards "$D/shards/Ep/shard-*.npz" --gen 9 --out $D/ft-Ep --init S --epochs 2 --batch-size 512 --lr 1e-5
  --warmup 50 --weight-decay 0.01 --value-coef 0.5 --aux-coef 0 --val-frac 0.05 --seed 0 --no-bf16`. Defaults
  otherwise. Run with OMP_NUM_THREADS 4, fp32.
- **The net:** final.pt, with no selection. best.pt and last.pt are deleted.
- **Identity check:** the chain stops unless the pretrain log's data line (119,967 decisions, 114,191 train, 5,776 val,
  60,007 games, 60 shards, spec 2.1) and every argument except the shard and out paths equal ns-27 E's.
- **No dose retry.** Item 2 fixes lr at ns-27's 1e-5, and ns-27's own retry did not fire (r_E 0.347).
- **Load check:** E′ must load through fba02fb (gen 9, finite logits), else FAILED.

### Information only (not part of the rule)
- **When and on what:** on H, before any game.
- **What is reported:**
  - KL(π_imp ‖ p_X) at T 1, and r′ = 1 − KL_X / KL_S, for X = E′, E, C;
  - top-1 agreement with argmax π_imp on the rows where it differs from S's argmax;
  - value log-loss and AUC on z.
- **On failure:** a failure here is noted in STATUS and does not stop the chain.

### Evaluation
- **Harness:** ns-27…29's in-process duplicate-pair harness. results.csv calls it "in-process fastenv simulator
  (spd_h2h)".
  - It is `exit_la2.py eval` → `porygon.fastenv_eval`, run from ns-27's frozen code `$C/runs/exit-la2/code` (1197016).
  - Every net is `v2:PATH,temperature=0.5`. E′ is side A (p1) in every game; each pair swaps the teams.
  - Sets: PORYGON_PS = `$C/ps-sets-pinned` (October), as in T1–T3. 10 workers × 32 battles, GPU forward.
- **The two tests:**

| test | games asked | pairs | seed |
|---|---|---|---|
| T1: E′ vs C (ns-29's control) | 8,000 | 4,000 | 11240221 |
| T2: E′ vs S (cd@805.03M) | 8,000 | 4,000 | 11240222 |

- **Counting:** fixed n, no interim looks. Complete pairs are counted, as in ns-27 (16,000 games asked gave 7,924
  pairs). The interval is 1.96·sd/√n over pair means.
- **Rule:** main's item 5 exactly, in percent with strict inequalities. **GO** iff the T1 lower 95% > 50.5 AND the T2
  lower 95% > 50.0. Else **NO-GO**.
- **VERDICT.txt** (r--):
  - line 1: `GO` or `NO-GO`;
  - line 2: E′ vs C, and line 3: E′ vs S, each as pair mean ± half-width [95% CI] (pairs, seed);
  - then β, the information check and, on NO-GO, the closing line.

### Chain, device, files
- **Job:** `code/scripts/fastenv/p2_exit_soft.sbatch`, fastenv's one helper job: `--partition=training`, 12 CPUs,
  1 GPU, 16 GB, nice 500, `--time 8:00:00`. It needs no benchmark slot.
- **Stages:** checks (PREREG r-- and md5, β line, S sha256, C and E md5, 60 parts, 60 E shards, CPUs, GPU) → calibrate
  (data check + reproduce β) → shards → fine-tune E′ → identity and load checks → information check → T1 → T2 →
  VERDICT.txt.
  - Each stage resumes from finished outputs. A failed stage writes FAILED and stops.
- **Files in `$C/council/p2-exit-soft/`:**
  - PREREG.txt (r--), PREREG.md5, STATUS, FAILED (on any failure);
  - calibration.json, ft-Ep/final.pt (+ ft-Ep.log), check.json, T1/T2 .json + .log;
  - VERDICT.txt (r--), NOTES.txt, code/.
- **Expected:** about 1 h after the job starts (ns-27: about 17 min per fine-tune; 8,000 pairs in 8–10 min).

## What follows
- **GO:** per PORTFOLIO P2 "if GO": fresh lookahead2(cd@2030) labels (stage-2 design, 3× the positions) and a
  fine-tune of cd@2030, tested against cd@2030 and under LA2. That needs main and its own prereg.
- **NO-GO:** record "ExIt-LA2, soft target: NO-GO" in BRIEF's tried list. The ExIt-LA2 line closes in both forms
  (hard and soft target).
