# G9t: the next pre-registered gen9 PPO recipe change, after G9r (DRAFT, not approved, nothing armed)

Drafted 2026-11-11 ~18:00 UTC by a level-2 agent for netv2 (side task (C), ORCHESTRATOR.md:128). Nothing on the
cluster has been started, submitted, armed or edited; the cluster reads were ssh read-only. Worktree HEAD read:
ws/netv2 `058a33a`. netv2.env read: v10 md5 `f28e166b` (runs/g9/MODE = g9r, runs/r1/MODE = g9q). Templates:
runs/g9r/PREREG.txt md5 `ee69eb24`, runs/g9q/PREREG.txt md5 `a64af3b2`. Numbers I did not verify from a file or log
are marked **(est.)**. The name G9t is unused (grep of .orchestra/*.md).

## 1. Evidence

| source | what it says | where |
|---|---|---|
| Wang 2024 | 7 epochs × minibatch 1024 over 78×512 = 39,936 transitions ⇒ 39,936/1024 = 39 minibatches × 7 = **273 optimizer steps per update**; ent 0.0588; γ 0.9999, λ 0.754; clip 0.0829 | memory wang-thesis-recipe.md |
| ours, per update | rollout +41.6k (A16, G9r) / +43.0k (A30, G9q); ⌈41.6k/4096⌉ = 11 minibatches × 3 epochs = **33 optimizer steps** (the 11th is a ~640–2,000-sample remainder) | train.py:518-521 (loop), netv2.env:328 args; train.logs |
| ratio | per sample: Wang 273/39,936 = 6.8·10⁻³ steps, ours 33/41.6k = 7.9·10⁻⁴ ⇒ **8.6× fewer**. Sample passes 7 vs 3 (2.3×); minibatch 4× smaller | derived |
| ours, entropy | `--ent-coef 0.01` constant (train.py:667 default; netv2.env does not set it) + `--zero-avoid-coef 0.003` (KL to uniform). No schedule exists. Wang's 0.0588 kept our policy near-uniform (≤ 9 legal actions) | train.py:548, 667-668; memory wang-thesis-recipe.md |
| ours, λ | `--gae-lambda 0.754` (default = Wang's), γ 0.9999. The critic's target is the same λ-return (ret = adv + V, train.py:158) | train.py:640, 128-160 |
| MissingNo.Zero | names "a shorter advantage horizon" and "an entropy schedule" among its fixes to Wang's recipe; still **no values**: the technical report (github.com/nickgreenquist/missingno-zero, PDF re-read 11-11) gives no hyperparameters and says its recipe changes were not ablated | research/REPORT.md:211-212; the PDF §5 |
| noise diagnostic | policy minibatch-gradient cosine ≈ 0.02 (sd 0.07–0.085) ⇒ B_noise ≈ 1.0–1.4·10⁵ samples ≫ 4096 and > one rollout | G9R-PROPOSAL §1 |
| trust region now (last 60 updates) | G9q A30 @114M: kl 0.0055, clip frac 0.132, ent 0.60, ptop 0.42. G9r A16 @13.8M (lr 4.5e-5, anchor β 0.072): kl 0.0015, clip 0.067, ent 0.74, ptop 0.24. `kl`/`clip` are means over all 3 epochs' minibatches | mirrored train.logs |
| lr ceiling | lr 2·10⁻⁴ degraded v2 after 7.5M; cooling (G9p-cd) gained +2 points | README.md:41; G9R-PROPOSAL §1 |
| eval temperature | raw nets play better at T 0.5 than T 1, and colder than 0.5 is no better | memory network-plan.md |

## 2. The three candidates, analysed

**Adam in our noise-dominated regime (est., the same model as G9R-PROPOSAL §3(a1)).** With gradient-noise variance
σ²/B per minibatch and μ ≪ σ/√B, Adam's step is ≈ lr · m/√v with √v ≈ σ/√B. Per sample, the signal moves
∝ lr · μ/(σ√B) and the noise random-walks ∝ lr/√B: the two scale alike, so the step count per sample acts like a
**learning-rate (temperature) multiplier**, not like more signal:
- minibatch 4096 → 1024 at a fixed lr ≈ lr × √4 = **×2**;
- epochs 3 → 7 on the same rollout ≈ lr × 7/3 = **×2.3** per rollout (the re-used samples carry the same noise), capped
  by the clip: samples already outside 1 ± 0.1 stop contributing;
- both (≈ Wang's 287 steps) ≈ ×4.7, i.e. an effective ≈ 3.3·10⁻⁴ at our 7·10⁻⁵, above the 2·10⁻⁴ that degraded v2.

### (1) More optimizer steps per sample
- **(1a) `--n-epochs 7`** (flag exists, train.py:638). Update time × 7/3 under `--no-overlap` (est.: `upd_s` scales with
  epochs × minibatches; the per-update fixed part, damage features + anchor pass, is small but unmeasured).
- **(1b) `--batch-size 1024`** (flag exists, train.py:636). Same sample passes, 4× the optimizer steps; one 1024 chunk
  per step instead of 2 × 2048 ⇒ smaller kernels, **+10–30% update time (est., unmeasured)**. Memory ≈ 0.
- Expected effect: on our own evidence, a **heating** change on the same axis G9r is testing in the cooling direction.
  If G9r wins (cooler is better), (1) is predicted harmful unless the lr is cut to compensate, which makes it either two
  changes or ≈ a no-op at 2.3× the update cost. If G9q wins, heating is still bounded by the 2·10⁻⁴ datum.
- Risks: kl and clip frac rise with epochs (clip already 0.13 on G9q at 3 epochs; est. 0.2–0.3 at 7); there is **no
  KL early stop** in the trainer (no `--target-kl`; ~15 lines + a test to add); later epochs are off-policy relative to
  the behaviour policy even with `--no-overlap`; faster sharpening ⇒ the entropy / ptop collapse guards fire sooner;
  on the A16 the sps cost is −45%.
- Wang's 273 steps come with a small MLP trained from scratch; we start from P3 with a KL anchor. The 8.6× gap is real
  but is not by itself evidence that we under-optimise.

### (2) An entropy schedule
- Needs code: `--ent-schedule {const,linear,cosine}` + `--ent-coef-final` + `--ent-steps` on absolute lineage steps
  (resume-safe), mirroring `lr_at`; ~25 lines + 2 tests (est. 1–2 agent-hours). sps and memory 0.
- Argument for: the entropy bonus is a deterministic gradient, while the policy gradient is ≥ 97% noise per minibatch,
  so 0.01 + za 0.003 may steer the policy more than the coefficients suggest; and raw nets play best at T 0.5, i.e. the
  T 1 policy is over-dispersed for play. A decay to 0 would test that.
- Against: the direction and the values are unknown (MissingNo.Zero gives none; Wang's 0.0588 was bad for us); for
  the first 50M the KL anchor to P3 dominates the policy's shape (β 0.1 → 0), so a schedule mostly acts after 50M, past
  the 50M decision point; ent 0 + za 0 is already the pre-registered gen4 follow-up #1 (nv2_cfg_r1b, netv2.env:303,
  :312), unspent. Collapse risk with a decay to 0; guards exist (entropy floor, ptop > 0.6).

### (3) A shorter GAE λ (recommended)
- **λ 0.754 → 0.5**, `--gae-lambda 0.5`: the flag exists (train.py:640), **no code**, sps 0, memory 0.
- Mechanism: the only one of the three that reduces gradient **noise** rather than changing the temperature. With
  roughly independent TD errors δ of variance s², Var(A) ≈ s²/(1 − λ²): 2.32 s² at 0.754, 1.33 s² at 0.5. After the
  per-minibatch advantage normalisation that is ≈ √(2.32/1.33) = 1.32× more signal per sample (est., upper bound),
  ≈ 1.7× the effective batch, at zero sps cost, if the critic's bias stays small. The credit horizon 1/(1 − λ) goes
  from 4.1 to 2.0 decisions (games are ~33 decisions per side, `len`).
- Orthogonal to G9r's lever, so it composes with either winner, and it is the "shorter advantage horizon" that
  MissingNo.Zero's Wang-derived recipe used (direction only).
- Risks:
  - **Critic bias.** Advantages lean more on V (ev 0.82–0.84; the value sends ~1/25 of the policy's gradient into the
    trunk). Long plans (hazards, setup, sacrifices) get their credit through V only.
  - **Bundled critic target.** `--gae-lambda` also sets the critic's target to the TD(0.5) return (standard PPO,
    train.py:158). A policy-only change needs a separate `--vf-lambda` (~10 lines); I propose the standard coupled
    change and record the bundling.
  - Counter-evidence: in dense-reward continuous control, λ ≈ 0.9 is usually best (Andrychowicz et al. 2021); ours is
    sparse-terminal and stochastic, so it does not transfer, but there is no direct Pokémon evidence either way.
  - The gain may not survive cooling: cooling also cuts the noise's effect, so the decision of record is cooled vs
    cooled (§3).
- **Pre-check (cheap gate, no strength claim, §3.0):** measure whether λ 0.5 actually raises the policy-gradient
  cosine on our rollouts before spending a GPU on it.

**Recommendation: arm (3), λ 0.754 → 0.5, alone, behind the noise pre-check.** It keeps the one-change rule, needs
no trainer code, costs no sps on the A16, and attacks the measured bottleneck (noise) through a mechanism G9r does
not test. Runner-up: (2), the entropy schedule, after (3). Not recommended now: (1); if the user wants it tested, the
interpretable form is (1a) with a `--target-kl` stop at the matched temperature, not Wang's 273 steps at our lr.

## 3. Design (pre-registration text to finalise before G9t's first step)

**Control C = the winner of G9r's rule 2** (runs/g9r/PREREG.txt): C = G9r's recipe only on "G9r-cd > G9q-cd" SUCCESS;
otherwise (G9q-cd wins, G9r stopped by rule 1 or 3, or undecided at n_max) C = G9q's recipe, the incumbent.
**G9t = C's recipe with ONE change, `--gae-lambda 0.5`.** Everything else as C: init P3 (md5 3e600174, steps 0, fresh
AdamW); KL anchor P3 0.1 → 0 at 50M; frozen P3 + `--frozen-best`; critic warm-up 1M; p_frozen 0.67, p_baseline 0.08;
3 epochs × 4096; lr 7e-5 (C = G9q: WSD constant; C = G9r: power, H 300M); fp32; `--no-overlap`; fastenv 8 × 48
battles, 16 + 16; November sets; evals vs P1 on the October servers; C's guards and fallbacks (state runs/g9t/), with
C's entropy floor (G9q 0.5 / G9r 0.35). C's nets already exist, so the control needs no new GPU time.

**3.0 Noise pre-check (before G9t is armed; informational gate).** From G9q ckpt_00045004575 (and C's 45M ckpt if C =
G9r), ≥ 40 fastenv rollouts; on each rollout, the same two disjoint 2048-sample chunks, `cos_pg` at λ ∈ {0.754, 0.6,
0.5, 0.3} (advantages recomputed from the same rewards and values; weights unchanged). Paired by rollout. Gate: the
paired mean Δcos(0.5 − 0.754) has a 95% interval with lower bound > 0 ⇒ arm G9t; otherwise report and move to the
runner-up (2). λ stays fixed at 0.5 whatever the curve shows (no forking). A higher cosine can also come from bias;
only the strength tests below decide.

Test procedure (as G9r's): in-process spd_h2h via scripts/netv2_cmp_step.sh (md5 a80c7da2), A = G9t's side; both raw
at T 0.5; duplicate pairs; wsp's design runs/cmp/seq-gain.json (θ₀ 50 → θ₁ 52%, α 0.025 per one-sided test, looks
every 100 pairs, n_max 2400); both directions on the same pairs; nested pair-id order = the decision of record,
round-robin alongside.

| # | look | comparison | rule |
|---|---|---|---|
| 1 | **25M harm guard** (raw vs raw; λ does not touch the lr, so raw is unconfounded, but 25M promotes nothing), seed **9825** | G9t's first pool snapshot ≥ 25M vs C's 25M snapshot (G9q: runs/g9/20261109-021316-741849/G9q/pool/00025038951.pt or its parked copy; G9r: the snapshot of its rule 1) | "C > G9t" SUCCESS ⇒ stop G9t (reject λ 0.5) |
| 2 | **50M decision of record** (deployment nets, cooled alike), seed **9850** | G9t-cd@50M vs C-cd@50M (the branch G9r's rule 2 already built: G9q-cd or G9r-cd) | "G9t-cd > C-cd" SUCCESS ⇒ λ 0.5 adopted ⇒ the vs-search milestone; G9t continues as a lineage candidate (placement = orchestrator / user). "C-cd > G9t-cd" SUCCESS ⇒ stop G9t. Undecided at n_max ⇒ λ not adopted; G9t continues only on a GPU nobody else needs, else parked |
| 3 | 50M drift / harm, seed **9875** | raw G9t@50M vs raw G9t@25M (rule 1's snapshot) | "G9t@25M > G9t@50M" SUCCESS ⇒ stop G9t |
| 4 | 50M informational, seed **9876** | raw G9t@50M vs raw C@50M (G9q: pool/00050038810.pt) | no rule; with rule 2 it shows whether cooling absorbs a λ gain |
| 5 | 100M continuation (only if G9t runs on and a C lineage reaches 95M), seed **9900** | G9t-cd@100M vs C-cd@100M, both from ~95M ckpts | "C-cd > G9t-cd" SUCCESS ⇒ stop G9t |

G9t-cd is built exactly as C's branch: G9t's first ckpt ≥ 45M + a 5M cosine to the same absolute floor 7e-6 (C = G9q:
`--cooldown-start <ckpt steps> --total-steps <+5M> --final-lr-frac 0.1`; C = G9r: `--final-lr-frac 0.35` composed
with the power law), entropy floor 0.35, own run / pool dirs, the branch final = the net (G9p-cd's driver machinery).

**Vs our search (milestone, fixed-n, Showdown slot; on rule-2 SUCCESS only):** G9t-cd@50M, 400 games,
netv2_vs_search_at.sbatch … 400 --gen 9 --seed 738164 (the same 200 team pairs), reported paired against G9p-cd's
845624 (50.5%) and C-cd's run if one exists; standing rule: ≥ 50% ⇒ a 1000-game confirmation.

**Seeds 9825 / 9850 / 9875 / 9876 / 9900:** my grep (11-11 ~18:00) found none in .orchestra/*.md, runs/*/PREREG.txt,
runs/cmp, netv2 logs/ or wsp runs/. That is not exhaustive (fastenv / data logs, run NETS / meta files not searched):
**netv2 must grep-check all streams again** before writing the PREREG.

## 4. Where it runs

G9t needs one GPU for ~50M steps + a 5M branch. **Right now no one has a GPU free:** A30 718701 → 810366 = G9q
(main), A16 741849 → 810367 = G9r, 677193 → 810087 = data (hosts small in-process tests), T4 dove04 829660 = gen4
R1b-F (to ~18:00 11-12), dove01 1026613 = the benchmark slot (netsearch 3b). The multigpu cap is 3 GPUs per user, so
a 4th multigpu GPU is not possible. **Nothing is parked for G9t: it waits for a GPU to free.**

G9r's 50M decision is ~11-12 afternoon at the earliest (est.): G9r was at 13.85M at 17:35 11-11, ~557 sps ⇒ 45M ckpt
≈ 09:00 11-12 plus the 810367 successor gap, then G9r-cd 5M (≈ 2.5 h on the A16) and the h2h.

| case (G9r's rule 2) | C | G9t placement | flag |
|---|---|---|---|
| "G9q-cd > G9r-cd" SUCCESS, or G9r stopped by rule 1 / 3 | G9q | **the A16** (freed by G9r's stop): same hardware as G9q's first 50M ⇒ the cleanest control | none: no one displaced |
| undecided at n_max | G9q | the A16: by G9r's own rule it continues only on a GPU nobody else needs, so it parks (resumable, PARKED.txt + MD5SUMS) | orchestrator call, pre-registered consequence |
| "G9r-cd > G9q-cd" SUCCESS | G9r | **no free GPU.** Options: (i) G9r replaces G9q as main on the A30 (G9q parked) and G9t takes the A16; (ii) the T4 after R1b-F's job ends (gen9 sps on a T4 unmeasured; R1b-F "resumes as is"); (iii) wait | **USER DECISION** for (i) or (ii) |

The pre-check (§3.0) runs in the data host (810087 once 677193 ends at 18:10 11-11) via `srun --overlap`, ~1 h of GPU
(est.), or on any GPU's job as an INLINE_CMD; it needs no lineage paused.

## 5. Costs and code

Measured baselines (last 60 updates; `--no-overlap`, so a cycle = collection + update):

| GPU | lineage | rollout | collect_sps | collection | upd_s | sps |
|---|---|---|---|---|---|---|
| A16 741849 (10 CPUs, 8 workers) | G9r @13.8M | 41.6k | 1,641 | 25.4 s | 49.3 s | 557 |
| A16 (brief: G9q's first 50M) | G9q | 41.6k | ~910 | 45.7 s | 43.8 s | ~464 |
| A30 718701 (28 workers) | G9q @114M | 43.0k | 5,092 | 8.4 s | 10.9 s | 2,222 |

Per arm (est. beyond the measured row: `upd_s` × epochs/3; batch 1024 at +10–30%):

| arm | A16 sps | A16 to 25M / 50M | A30 sps | A30 to 25M / 50M |
|---|---|---|---|---|
| **(3) λ 0.5 (recommended)** | 464–557 (= C) | 12.5–15.0 h / 25–30 h | 2,222 | 3.1 h / 6.3 h |
| (2) entropy schedule | = C | same | = C | same |
| (1a) 7 epochs | 281–296 | 23–25 h / 47–49 h | 1,270 | 5.5 h / 10.9 h |
| (1b) batch 1024 | 465–523 (G9r stack) | 13–15 h / 27–30 h | 1,900–2,100 | 3.3–3.7 h / 6.6–7.3 h |
| (1a)+(1b) ≈ Wang (287 steps) | 238–274 | 25–29 h / 51–58 h | 1,040–1,180 | 5.9–6.7 h / 11.8–13.4 h |

Branch G9t-cd 5M: ≈ 2.5–3 h on the A16, ≈ 38 min on the A30. h2h tests ~70 s per 1000 pairs in the data host. Vs
search: 400 games ≈ 1 h of the slot. A16 GPU time for arm (3) to 50M + branch: ≈ 1.2–1.4 GPU-days.

**Code (arm (3)):** trainer **none** (`--gae-lambda` exists). Needed:
1. netv2.env: a mode `g9t` = C's mode (g9q-recipe or g9r) + `--gae-lambda 0.5`, lineage dir G9t, state runs/g9t/,
   marker runs/g9t/G9T_STARTED, PORYGON_PS November; first start from P3, restart in place its own save, a fresh job its
   newest mirrored G9t file; argsdiff showing every live mode unchanged.
2. Mocks: scripts/netv2_env_mock.sh cases for g9t on A16 / A30 / ≥ 40 GB; scripts/netv2_job_mock.sh first start /
   restart / successor / the INLINE_CMD that parks the displaced lineage (reuse scripts/netv2_g9r/inline-a16-park-g9r.sh).
3. The pre-check: a small script `scripts/netv2_lambda_noise.py` (load a ckpt, collect N fastenv rollouts with the
   trainer's collector, `build_rollout` per λ, `gradient_diagnostics` on fixed chunks, write a JSON + the paired
   interval); ~60–80 lines, reusing train.py functions, run from a new read-only `$C/src-<sha>`.
4. G9t-cd driver: G9p-cd's / G9r-cd's adapted (ckpt, steps, frac).
5. Deploy rules as G9r: a new read-only src dir + wrapper only if the pre-check script is shipped; never touch the
   src dirs or wrappers that 810366 / 810367 / 741849 / 718701 read.

Engineering estimate: ~2–3 agent-hours incl. mocks and the pre-check (est.). For comparison: (2) adds ~1–2 h of
trainer code + tests; (1a) with a KL stop adds ~1 h.

**Open questions for the orchestrator / user**
- Coupled (`--gae-lambda` sets the critic target too, standard) or policy-only (`--vf-lambda`, ~10 lines)? Draft:
  coupled.
- λ 0.5 or 0.6? 0.5 gives the larger, easier-to-detect change; the pre-check's curve is informational only.
- The case "G9r wins": which GPU, if any, G9t may take (USER DECISION).
- Should the gate in §3.0 be binding, or informational only?

Status: DRAFT. Requires (i) orchestrator review of the arm and the design, (ii) **USER DECISION** for any placement
that stops, pauses or demotes a running lineage, (iii) a final runs/g9t/PREREG.txt written after G9r's rule 2
resolves (it fixes C) and before G9t's first step.

## ORCHESTRATOR DECISIONS (2026-11-11 ~18:15 UTC; supersede the draft where they differ)
1. λ applies to BOTH the advantage and the critic's target (standard GAE; one change; no new `--vf-lambda`).
2. The pre-check is BINDING and picks the value: pre-register λ ∈ {0.754, 0.6, 0.5} on the same paired rollouts,
   measuring signal per sample (the gradient-cosine metric) and critic bias; the rule, written before running, picks
   the smallest λ whose bias stays under a stated bound and whose gain is ≥ a stated threshold, else "no G9t".
   Build and run it when convenient (in-process job, outside data refresh rounds), after U7's BC training is under way.
3. The g9t mode + mocks are built only when G9r nears 40M (placement in sight). The placement USER DECISION (if G9r
   wins at 50M) goes to the user then, not now.
