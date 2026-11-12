# Benchmark-bot re-freeze #1: pre-registration (fastenv, 2026-11-12 ~01:25 UTC, before the candidate exists; SIGNED OFF by the orchestrator 2026-11-12 ~01:28 UTC, binding, read-only)
Orchestrator approval (2026-11-12, NEXT-PROPOSAL §5 (2)): the rule and the candidate. This file fixes them in full
**before the branch exists**. It becomes binding once main signs it off. The BLOCKING item below was settled with netv2 on 2026-11-12. Until then, no branch starts on its account and no game is played.

## Question
Does the margin-rule hybrid with a newer deployment net beat the frozen gen9 benchmark bot head to head?
- **Frozen bot:** H(G9p-cd@104.71M, δ 0.0212), NEXT-PROPOSAL §1.
- **Question:** is H(cand, δ_c) stronger than H(G9p-cd, 0.0212) at the same budget?

Only the network changes. δ is re-derived by the old rule; everything else stays: search, budget, harness, server,
engine.

## BLOCKING: the set library (settle with netv2 before the branch)
- **Fact needed:**
  - which random-battle set library the candidate's encoder uses when it plays on fastenv's benchmark server (the
    October build);
  - which library G9p-cd used in Stage 2/3.
- G9q trains on the November sets (PORYGON_PS ps-2026-11).
- **Rule (PLAN):** the benchmark server stays on the October sets unless it is re-baselined. Both arms of every game
  below must use the October library, as G9p-cd did in Stage 3.
  - That holds for the net encoders' posteriors and for the search's determinizations.
  - If the candidate can only be evaluated with the November library, this pre-registration stops, and main decides.
    The options would be a re-baseline, or accepting the mismatch as a recorded deviation.
- **Answer: SETTLED 2026-11-12.** netv2 checked on the cluster at 01:24; fastenv checked the JS tables at 01:2x.
  - **The v2 encoder's posterior library** is a file in the code tree,
    `porygon/data/gen9randombattle_library.json.gz` (md5 324d8e97). It is NOT read through PORYGON_PS. The same file
    sits in `frozen/netv2-fba02fb-20261110/src`, netv2 src, src-24c633f (G9q's), src-ff32b71 (G9r's), src-p2 and
    gloria. It is the October library, and G9p-cd used it in Stage 2/3.
  - **G9q's PORYGON_PS=ps-2026-11** only swaps the collector's Showdown sim, i.e. the TRUE sets in its training
    games. The difference is levels only: ±1 on 4 gen9 species. The encoder posterior was the October library
    throughout.
  - So cand loads through fba02fb with the October library, with no env var or flag. PORYGON_PS stays unset on the
    benchmark side, as in Stage 2/3.
  - **fastenv's JS tables** `fastenv/js/tables/gen9_lib_v2.json` (exported from the netv2 reference library) have
    md5 240b7301 in ws/fastenv (beb746a, 11-08), `$C`, and netv2's fastenv copy (G9q's collector). They were not
    rebuilt for November.
  - **Known deviation, recorded, not a blocker:** cand trained on games whose true sets carry the November levels
    (±1 level on 4 gen9 species). It is served on the October server and library, so there is a small train/serve
    shift for those species. G9p-cd had none.

## 1. The candidate (netv2 makes it)
- **S** = the main G9q lineage's first checkpoint ≥ 300,000,000 steps (810366's or its successor's G9q mirror).
- **Branch:** netv2's standard cooldown recipe, as G9p-cd: `--resume ckpt_S --cooldown-start S --total-steps S+5M
  --final-lr-frac 0.1`, with its own run and pool dirs.
  - Entropy floor 0.35 (as G9r-cd and G9q-cd@45M), ptop 0.6 kept (netv2, 2026-11-12). netv2 writes the branch's
    command line in its branch pre-registration before the branch starts.
- **cand** = the branch's final checkpoint, or its final at an early guard stop (as G9p-cd's DEVIATION at 104.71M).
  - It is copied to `$C/nets/g9q-cd-<steps>.pt`, with its sha256 recorded here.
- **netv2 code:** both hybrids load through ONE frozen netv2 tree.
  - If `frozen/netv2-fba02fb-20261110` cannot load cand, a new frozen tree is made (path, commit and sha256 written
    here).
  - On the new tree, G9p-cd's logits on 1,000 recorded observations must equal those from fba02fb (max abs diff ≤ 1e-5)
    before any game.

## 2. Trigger (in-process, no slot; netv2's harness)
- **Test:** cand vs G9p-cd@104.71M, raw, both at T 0.5, in-process gen9 (fastenv_eval / wsp sharded h2h, validated
  against Showdown).
  - wsp's group-sequential design (`.orchestra/wsp/seqtest-plan.md`: Lan–DeMets, α 0.025 one-sided, power 0.8,
    max 2400 pairs).
  - Harness: `scripts/netv2_cmp_step.sh` (md5 a80c7da2; wsp's sequential-gain design); cand = A; the nested
    order is the record.
  - **Seed 9950** (netv2 grep-checked it as unused at 01:24 11-12; reserved for this test).
  - Nested shard order (`seq_monitor.py --order sequential`).
- **Rule:** proceed to §3 iff "cand > G9p-cd" SUCCESS **and** the median-unbiased estimate is ≥ 55%. Otherwise there
  is no re-freeze at this candidate. A later candidate needs a new pre-registration.

## 3. Margin δ_c (offline, helper job)
- **Stage 1 re-score:**
  - command: `NET=nets/g9q-cd-<steps>.pt OUT=netsearch/stage1-g9qcd-<steps> bash scripts/fastenv/netsearch_stage1.sh`;
  - the same positions and searches as Stage 1 (S1, S2, S2c, S3);
  - the same code, i.e. the (id, side) re-score fix 9ac4785 (or later code if Stage 1 needs none).
- **Rule (design §3.1, unchanged):** δ_c = the margin where the R-judged loss L(δ), pooled over S2 + S2c, reaches
  0.0005 Q per decision.
  - Linear interpolation on the grid (`netsearch_arms.py`, level 0.0005).
  - Judge B if R is missing; cap 0.2.
- No tuning games. Only δ_c is used; the 0.001 and 0.002 levels are reported for information.

## 4. Slot confirmation R (benchmark slot, fixed n)
**Players.**
- **Arm 1** = `netsearch:$C/nets/g9q-cd-<steps>.pt,margin=δ_c,temperature=0.5,ms=1000,samples=4,workers=11`.
- **Opponent** = `netsearch:$C/nets/g9p-cd-104.71M.pt,margin=0.0212,temperature=0.5,ms=1000,samples=4,workers=11`,
  the frozen benchmark bot.
- Each side has its own 11-process engine pool (`own_pool`).

**Harness:** `netsearch_slot.sbatch`, with a plan `plans/refreeze1.env` (one arm, `OPP=` the netsearch spec above).
- One 24-CPU `--partition=training` job, claimed in `ledger.md`.
- fastenv's `.showdown-eval` (95a0414e26cfd185); gen9 engine gloria `.engine-gen9`; 3 concurrent battles.
- Code copied into the run directory.

**n:** 5 blocks × 100 pairs = 500 pairs (1000 games), fixed n. No looks, no futility.
- **Seed tag 20261112:** blocks seeded `20261112b<j>`, drawn now. On 11-12 ~01:2x it appeared in no plan or manifest
  under any `porygon-ws-*` cluster dir. It is used by nothing else.

**Rule:** **adopt iff the lower 95% bound of arm 1 (pair means) > 50%.**
- Adopted ⇒ H(cand, δ_c) is the gen9 benchmark bot from then on, and NEXT-PROPOSAL §1 is updated with its spec and
  hashes.
- Not adopted ⇒ G9p-cd stays the benchmark net. cand remains netv2's raw deployment net.

**Void:** HARNESS_FAIL, or > 2% search fallbacks on either side. The run is replayed with the same seeds after the
fix.

**Reported:** arm 1 ± 95% CI; both sides' override rates and fallbacks; the won-both / split / lost-both counts;
hours (≈ 5–5.5 h: both sides play like 3a's H arm, 4.98 h per 1000 games).

**Not part of R:** no Foul Play games. An H(cand) vs Foul Play run is needed only as the pairing control of a later
Foul Play comparison (e.g. V3-F), with its own pre-registration.

## 5. Checks before R's first slot game
- **Harness smoke** in a helper job (not the slot): the plan with `OPP=netsearch:` is untested so far. Stage 2/3 used
  `search:` or `foulplay` opponents.
  - Setup: smoke tag `rfsmoke20261112`, 2 pairs, an old net on both sides.
  - Checks: two separate 11-process pools, both sides' gate logs and stats, and `paired` / the report working.
  - No game of the smoke uses tag 20261112 or cand.
- The manifest must record: cand's sha256, δ_c, the netv2 tree, the server and engine hashes, and the set library
  (the BLOCKING answer).

## Order (orchestrator-approved slot order)
1. netv2's gen4 prior test holds the slot first.
2. R is submitted after it ends, and only once §2 and §3 have passed.
3. Re-freeze comes before any V3 tuning game. V3 itself waits on the user.
