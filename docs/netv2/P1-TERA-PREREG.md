P1-TERA: council test P1, own pre-Tera visibility (spec v2.2a) — PRE-REGISTRATION, FINAL
(netv2, 2026-11-24 ~15:05 UTC; council session 1 portfolio item P1, docs/council/sessions/2026-11-24/PORTFOLIO.md;
design and decision rule signed off by main (the orchestrator) on 11-24; written before any arm step and before any game;
read-only copy /vol/bitbucket/fr1226/porygon-ws-netv2/council/p1-tera/PREREG.txt, md5 in the HANDOFF and the env.)

QUESTION
  Does a fork whose learner sees its own Tera types from turn 1 (spec v2.2a) beat a matched v2.1 fork in-process after
  30M + 5M-cooldown steps? Bug fixed (verified by main): fastenv/js/encoder_v2.js and src/porygon/v2/encoder.py fill our
  own `tera` / `tera_p` only once the mon terastallizes (poke-env never reads the request's teraType), so the net chooses
  Tera blind to its own type; 368 / 509 gen9 species have identical sets under different Tera types.

THE CHANGE (ws/netv2 09a6378 + 2e03cc6; code tree $C/tree-2e03cc6 = src-8892762 + this change, read-only)
  v2.2a = v2.1 + our own tera / tera_p (slots 0-5, gen 9) from the owner's request teraType, matched by ident (as
  porygon.search.request_tera), from turn 1; terastallized stays 0 until -terastallize; aux targets, legal actions,
  shapes and parameter count unchanged. A per-POLICY flag (checkpoint key own_tera_visible): the learner of arm E and its
  own new snapshots / checkpoints / finals are v2.2a; every other policy (the parent's pool, the frozen BC net P3, the
  eval anchor P1, arm C) keeps v2.1: in a v2.2a env the JS encoder also emits each row's exact v2.1 tera / tera_p
  (tera_v21 / tera_p_v21) and FastCollector._opp_obs / Trainer._opp_view / spd_h2h_tera hand those to v2.1 nets. The
  poke-env NetPlayer (Showdown evals) encodes per net. The KL anchor (coefficient 0 at these steps) and the KL monitor
  see porygon.v2.spec.v21_view (diagnostics only; exact except Illusion quirks, 54 of 42,934 decisions in Stage 0 run 1).

STAGE 0 (integrity, before any arm; all must pass; council/p1-tera/stage0/RESULT.txt, scripts/netv2_p1tera/stage0*)
  300 streaming gen9 battles (seeded random legal players, November sets), at every decision of both sides: Python /
  JS parity on every field, flag off and flag on (ints exact, floats 1e-4); flag off == today's code byte for byte
  (Python encoder.py of src-8892762, JS $C/fastenv/js = the live trainers'), inputs and aux; aux unchanged by the flag;
  the flag-on row's tera_v21 / tera_p_v21 == flag off byte for byte; terastallized 0 for every own mon before
  -terastallize; every pre-Tera own slot filled with the flag on. Leakage: 300 battles each played twice, the second
  time with one side's unrevealed Tera types reassigned (mons that never terastallize in pass 1, not Ogerpon / Terapagos
  / Illusion users): the other side's stream and flag-on observations byte-identical at every decision, while the
  re-typed side's own observations differ. Fixture: our lead Volcarona (its library set: Fiery Dance / Giga Drain /
  Quiver Dance / Tera Blast) Tera Water vs Tera Ground: own tera / tera_p as expected, terastallized 0, flag off
  identical; active_tera_vs of Tera Blast vs Iron Moth Ground > Water > flag off, vs Skarmory Water > flag off > Ground
  (immune), other move slots unchanged. Result: PASS (run 2, 11-24 15:0xZ, council/p1-tera/stage0/RESULT.txt md5 d97fe140, SUMMARY.json
  e5eca1a7; tree 2e03cc6): 300 battles / 42,934 decisions, every check 0 mismatches; 215,986 / 215,986 pre-Tera own slots
  filled; leak 300 battles / 20,816 observer decisions / 1,484 victim mons re-typed: 0 stream divergence, 0 observation
  mismatch, the victim's own observations differ in 300 / 300; fixture 16 / 16 (Iron Moth: Tera Blast Ground 1.348 >
  Water 0.674 > off 0.225; Skarmory: Water 0.494 > off 0.165 > Ground 5e-27). Run 1 (tree 09a6378, kept as
  stage0-run1-FAIL/) failed two checks, both Illusion effects: the zeroing view was not exact for 54 decisions (one
  battle: a disguised own mon carries the illusionist's tera type in v2.1) and the leak test re-typed a Zoroark whose
  tera was revealed under its disguise's name (a test bug); fixed by the exact tera_v21 columns and by never re-typing
  Illusion users, then rerun unchanged otherwise.

ARMS (the A40 = g9 run; netv2.env v23 g9 MODE p1tera; order: E first, then C)
  Parent (both arms): G9q's latest uncooled checkpoint at the fork, ckpt_02210021139.pt (2,210,021,139 steps; job
  1483048, written 14:20Z 11-24), r-- copy council/p1-tera/parent/ckpt_02210021139.pt, md5 6db059de (= the node-local file).
  Pool (both arms): the same read-only copy of G9q's live pool at the fork, council/p1-tera/pool/ = hard links to the
  mirrored files of the 599 live snapshots <= S0 + the live frozen set (P3-g9-s2.pt, best-02052532799.pt); POOL_MANIFEST.txt
  (name + size) md5 ecd74c3e; each arm's pool dir = symlinks to a node-local copy of it + the arm's own new snapshots.
  Recipe (both arms) = G9q's on this GPU (= G9-P4's A40 settings): --no-bf16 --n-epochs 3 --batch-size 4096 --lr 7e-5
  --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5 --guard-ptop-max 0.6 --kl-anchor
  P3 --kl-anchor-coef 0.1 --kl-anchor-decay-steps 50000000 (coefficient 0 at these steps) --critic-warmup-steps 1000000
  --frozen P3 --p-frozen 0.67 --p-baseline 0.08 --frozen-best, fastenv 8 workers x 48 battles, --no-overlap, 16 + 16
  battles, --snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200, eval anchor P1, November sets; optimizer and
  counters from the parent. Same seeds: --seed 9241101 at each arm's first launch, --seed 9241102 at each cooldown's
  first launch (later relaunches unseeded, as G9q2).
  Arm E: + --own-tera-visible (v2.2a learner). Arm C: v2.1 (no flag). Nothing else differs.
  Train: --total-steps S0 + 30,000,000 = 2,240,021,139; --kl-monitor parent --kl-monitor-until S0 + 2,000,000 (logged
  KL(arm || parent) per update for the first 2M steps; no loss term).
  Cooldown (identical): from the arm's final at its cap (branch r-- council/p1-tera/{E,C}/branch-<steps>.pt), 5,000,000
  steps cosine lr -> x0.1 (--cooldown-start S --total-steps S + 5M --final-lr-frac 0.1), entropy floor 0.35
  (--guard-entropy-min 0.35). Finals r-- council/p1-tera/{E,C}/final/<steps, 11 digits>.pt (+ FINAL "path md5 steps");
  both present => council/p1-tera/ARMS_DONE.
  Guard trips: recorded in council/p1-tera/TRIPPED, no fallback, no HOLD, the same config continues (both arms alike); a
  third trip of one lineage => FAILED.
  G9-P4: parked by pointer at the switch (council/p1-tera/G9P4-PARK.txt; state stays in its lineage dir / mirror) and
  resumed AUTOMATICALLY from its own newest state when ARMS_DONE or FAILED exists (runs/g9/MODE = g9p4, restart flag
  re-armed; no HOLD for main). The MODE survives the 11-26 10:27Z hand-over (successor 1552709 sources it).

HARNESS AND n (cluster-side chain scripts/netv2_p1tera/verdict-chain.sh, a 1-CPU step in data's host job; the h2h runs as
an srun --overlap step in the RUNNING ws-data-host job with the most time left, R2030 pattern, one retry in the next host)
  When ARMS_DONE exists: in-process spd_h2h (scripts/netv2_p1tera/spd_h2h_tera.py = wsp's ee76994 + per-side encoder
  routing: a v2.1 net gets its exact v2.1 inputs), A = E-cd vs B = C-cd, both raw at T 0.5, gen9randombattle, November
  sets (fastenv's ps-2026-11), 4,000 fresh duplicate pairs (pairs 0-3999 = 40 shards of 100, teams swapped within a
  pair), pair seed 9241103 (new; unused), 5 workers, cuda.

DECISION RULE (verbatim, main 11-24)
  GO iff E-cd vs C-cd pair-mean lower 95% > 50.0%; lower bound in (50.0, 50.5] => REPLICATE (a second training seed
  before any promotion); else PARK (a 30M fork can be too short: not a kill of the idea).
  Read mechanically (lower = pair-mean - 1.96 x pair SD / sqrt(pairs), in %): lower > 50.5 => GO; 50.0 < lower <= 50.5
  => REPLICATE; lower <= 50.0 => PARK.
  Output: council/p1-tera/VERDICT.txt (r--), line 1 = GO | REPLICATE | PARK, line 2 = the pair-mean and its 95% CI (pairs,
  games, ties, voids); then the nets and md5s, the rule, and (info, no rule) the arms' first / last KL-monitor lines and
  guard trips. Any failure writes council/p1-tera/FAILED (one line why) and stops; nothing is replayed without main.
  If GO / REPLICATE: main's follow-ups (PORTFOLIO P1 "if GO"); none is armed here.
