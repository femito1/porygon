# Timeline

Compiled 2026-11-17 from `git log` (389 commits on `main`; commit times are UTC except 27 commits up to 2026-10-24 made
at +01:00, converted here), memory notes, `.orchestra/ORCHESTRATOR.md`, `PLAN.md`, the stream HANDOFFs and the cluster
result files. Each item: date (UTC) — what — why / evidence (commit or file). Result ids refer to `results.md`.
Provenance: dates from commits or file timestamps are [M]; "≈" marks [E].

## Phase 0 — first prototype (2025-03-02 → 2025-03-03)
- 2025-03-02 22:13 — first commit `25462f2` "first version of project"; 3 more commits to 03-03 (training loop starts,
  battles "spam started"). Then **no commits for 19.5 months** [M git].

## Phase 1 — rebuild as self-play PPO, GlorIA → Porygon (2026-10-19 → 10-25)
- 10-19 15:36 — `59f2d08` "Rebuild GlorIA as self-play PPO on poke-env 0.16". **User decision:** follow Jett Wang's MIT
  thesis recipe (methods, not his work) [M memory `wang-thesis-recipe.md`, modified 10-19]. First run dir
  `gloria/runs/20261019-140413`.
- 10-20 — checkpoint tournament, Foul Play runner, RunPod automation (`8832938`); opponent pool (`a1696ed`);
  **network-guided search player on poke-engine MCTS** (`9481474`); Slurm training job with auto-resume (`205d632`) —
  training moves from RunPod (fallback; balance ~$2 on 10-21) to Imperial DoC Slurm (gpucluster2) [M memory].
- 10-20 — first results: 60M net 83.5% vs SimpleHeuristics, 95% vs MaxBasePower (v1-01/02).
- 10-21 — `7020fa8` **rename GlorIA → Porygon**; Foul Play benchmark; search 49% vs Foul Play with the 91M net, raw net
  16.5% (v1-06/07). Ladder: both accounts (porygon-agent, femito) bot-flagged; datacenter IPs locked ⇒ official-ladder
  games only from the laptop [M memory `project-rename-porygon.md`].
- 10-22 — expert iteration (`3a927e1`); 10-23 asynchronous rollouts (`24547c0`, ~6 s → ~1 s per rollout GAE), overlapped
  update (`7e0bd56`).
- 10-24 — 400M checkpoint (`runs/20261024-135245-300664`); 10-25 500M (`runs/20261025-195826-301040`, later the
  benchmark default); 10-26 600M (`runs/20261026-082356-301120`) [M run dirs].
- 10-25 — search with the 400M net beats Foul Play 55.0% ± 4.9% (job 300964) (`28cb698`); expert iteration ties its
  source (v1-12); **v1 value lookahead** in the search does not help (s4-01…03, `9426935`, `1d4e2d6`).

## Phase 2 — search vs Foul Play, inference, fitted leaf evaluation (2026-10-26 → 11-03)
- 10-26 — search fidelity: gen4 taunt/encore/confusion durations, substitute HP (`f55ae4c`) ⇒ engine patch.
- 10-27 — **duplicate battles** on a patched Showdown, decision logs, analysis (`02c5e57`); all later benchmarks are pairs.
  Pooled default search 52.6% (fp4-01). Oracle runs (`052fbd0`): +8.8 / +12.9 with old inference (fp4-10/11).
- 10-28 — **incident:** node images moved `/usr/bin/python3` to 3.14 and some nodes had no 3.12 ⇒ venvs rebuilt on a
  standalone CPython 3.12.16 [M memory `imperial-slurm-training.md`]. Inference from Showdown's own generator (`42aaf31`).
- 10-29 → 11-01 — root PUCT (`cf76924`), damage evidence (`eeff34c`), Bayesian root game and regret matching
  (`00b7a05`, `d04f94f`): none beat the default; sampled root game −13 points (fpr-307072). Baseline 54.5% ± 1.5% (fp4-02,
  `038fb2e`, 11-01).
- 11-02 — **QOS change:** a16 jobs capped at 4 CPUs in QOS normal ⇒ 24–48-CPU jobs move to `--partition=long` [M memory].
- 11-03 — **fitted leaf evaluation** (`3a40ed7`, `c6b56b8`, `abf5305`): 59.2% vs Foul Play (fp4-04), made the gen4 default.
  The README first quoted ±1.1 (a standard error), later ±2.2 (pair-mean 95%).

## Phase 3 — diagnosing the network, gen9 port, paper (2026-11-03 → 11-08)
- 11-05 — oracle labels for logged decisions (`dbee3a6`; the "methodological trap", dg-04); embeddings found never to
  leave their init ⇒ `--embedding-lr-mult` arms (`2c0845c`; no gain, v1-14/15); **incident:** an arm deadlocked for 8.7 h ⇒
  45-min stall watchdog (`37db029`); raw-net h2h job (`19219c0`).
- 11-06 — `--fp-parallelism` for compute-matched benchmarks (`0226baa`; fp4-05/06); paper outline (`588f3be`).
- 11-07 — **gen9 port** (`532e833`, `4f5e17a`, `a848267`, `2dec70f`): `PORYGON_GEN`, gen9 engine build, tera actions,
  MODEL `none`. **Incident / bug:** Rest/Sleep-Talk sleep-turn count (poke-env adds 3 per Sleep Talk turn) pushed the engine
  past MAX_SLEEP_TURNS ⇒ MCTS panic (NonFinite); fixed in `006ff76`; data logged before 11-07 has the Rest-duration bug
  (also: self-play PP 16 made stall loops infinite, fixed in `532e833`). First paper draft (`be56c30`); every number re-derived
  (`5d51467`).
- 11-07/08 — gen9 pure search vs Foul Play 51.7 / 53.3 / 53.3 (g9s-01…03): the gen9 fit gives no measurable gain.
- 11-08 — literature review, 1,587 records screened (`906ec4d`); README rewritten around "network-first" (`f0e3cf3`).
  Laptop disk < 1 GB free [M memory].

## Phase 4 — the network-first campaign (2026-11-08 → now; orchestrator + 4 streams)
**Why:** the research report (`.orchestra/research/REPORT.md`, 11-08) found MissingNo.Zero's ~3M-param PPO net beats Foul
Play 61.6% (gen4) / 66.7% (gen9) with no search, while our raw v1 net scored 46.2% ⇒ "our net's weakness is a recipe
failure" [M memory `literature-review.md`]. **User decisions 11-08:** goal = the best bot in gen9 and gen4 random battles,
beat Jaxcalibur; never distil Jaxcalibur (challenge games for evaluation only); human crawl gen9 rated ≥ 1500 only; yes to
looking into the bot registry [M memory]. Work split into streams netv2 / data / fastenv / wsp, each with a worktree, branch
and cluster dir; a shared benchmark-slot ledger (`.orchestra/PLAN.md`).
- 11-08 (107 commits that day) — observation spec v2.0 → v2.1 (`b8a1b3e`, `1d2574e`); v2 model and PPO trainer (`5f60a4f`,
  `ceb0875`); fastenv in-process Showdown (`6b9646a`, `b5cfbbe`); polite replay crawler (`d319b8c`); exact first-person
  reconstruction from input logs (`67e223b`); BC (`e3b55fc`, `a530999`); search-label generation (`826eac3`). P1 frozen 22:43.
- 11-09 01:11 — v2 trainer guards (`be71647`); 03:43 **incident:** netv2 pool workers hung (forkserver preload left forked
  workers without poke-env's POKE_LOOP thread; every pool baseline player hung, R1's opponent mix silently missing) ⇒
  `972726f` + pool alarm. 04:14 sequential-testing design (wsp, `3f38ba4`); **user 11-09:** sequential tests for internal
  go/no-go only, headline numbers fixed-n [M memory]. 06:45 P2 frozen, 17:35 P3 frozen. 12:14 **partition change** (`long` =
  dove03–04 only; dove01–02 via `training`). 12:40 GPU swap (G9p → A30 job 718701, G9s → A16 job 741849). G9s (from
  scratch) loses to G9p (P1 init) at 10/25/50M (v2g9-09…12) ⇒ G9s parked. Netsearch prototype (`b3ba041`).
- 11-10 00:04 — `24c633f` (stacked-opp fix), the src snapshot every trainer still runs. 01:06 **user decision: G9q = option
  A** (G9p's recipe from P3) (`3517cc4`). 10:14 netsearch Stage 2/3 pre-registered (`4c879d7`). Actor-stream adopted (speed
  1.38×, non-inferior over 5M; v2g9-19/20), then **reverted** 15:25 after the 25M drift check (46.0 vs control 49.6, v2g9-21;
  `d735897`). Stage 1 re-score bug (records of both sides shared an id) found and fixed before registration (`9ac4785`).
  G9p-cd > G9p ⇒ cooldown branches become deployment nets (v2g9-13).
- 11-11 — Stage 3a/3b: the hybrid beats pure search 55.0% and Foul Play 56.2% ⇒ **gen9 benchmark bot** (ns-05/07). G9q beats
  G9p step-matched at 25M / 50M ⇒ **G9q becomes the main gen9 run** (g9q-01/02). U7 frozen 17:55. 18:10 data-host
  self-submission parse failure (submit-plugin banner on stderr), fixed `029c1c5`.
- 11-12 01:43 — **first merge of all streams into `main`** (`1049e1a`, `3e9fab5`, `0f5a5b7`, `d979250`); orchestrator may push
  `main` after merged tests pass (user OK 11-12) [M PLAN.md]. U7: quality matters at equal size (bc-10). V3 gates P1/P2 pass;
  **user decision ≈ 11:30:** V3 goes to a Rust port (design only until GO); 18:59 V3 I1 FAIL ⇒ V3 parked (ns-10…12).
  G9r (lr anneal) and G9t (λ 0.5) arms start; G9r undecided at 50M (g9q-05).
- 11-13/14 — re-freeze #1 trigger passes (g9q-13) but the slot test does not adopt (ns-09). G9t-cd loses to G9q-cd ⇒ STOP
  (g9q-10). R1b-F passes its 300M watch (v2g4-08/09). **User 11-14:** every approved multi-step pipeline must be a
  cluster-side job chain (the laptop sleeps and agents die with it) [M PLAN.md].
- 11-14/15 — first raw gen9 net above Foul Play: G9q-cd@605M 57.5% (400 g), **58.8% ± 2.9% confirmed** over 1000 games
  (g9q-14/17); vs our search 52.5% not confirmed (g9q-16). 11-15 merge of streams (`c4a82d8`, `1499065`).
- 11-15 — **user decisions:** full delegation of campaign choices to the orchestrator ("make all the choices yourself. I
  trust you to achieve the best result"); stream agents must be short-lived (a netv2 agent was flagged at ~700k tokens);
  G9q2 seed-variance run (`2b95226`). R1b-F's stale follow-up rule (would have reset 360M steps to 7.5M) found and disarmed
  [M memory]. Bet 1 lookahead2 written (`fbe7c87`) and pre-registered (`5efda5c`); entropy-schedule arm (`cb76f75`).
- 11-16 — **Bet 1 GO**: lookahead2 beats its raw net 53.17% (ns-14). bf16 SKIP (g9q-20); G9q2 shows seed variance < 2 points
  (g9q-19); G9qent no gain (g9q-22); NNUE gate 1 **FAIL** (ns-16/17); prior test #2 inconclusive (v2g4-12). P4 frozen
  17:56 (READY), BC-P4 ≈ BC-P3 (bc-15), G9-P4 starts on the A40 21:11. **conf800: Milestone 1 CONFIRMED** — raw cd@805
  beats our pure search 55.5% (g9q-23); vs Foul Play 57.8% (g9q-24). Gap assessment told to the user: ≈ 135–165 Elo to
  Jaxcalibur ⇒ route = a search that scales with thinking time, then expert iteration. **User ≈ 22:50:** goal adds a raw
  network as strong as Jaxcalibur's; "do 1 and 2" (loss-pattern analysis + Foul Play calibration). 23:45 loss patterns
  (dg-01). Merges 17:29–17:34 (tests 189 pass / 38 skip / 0 fail).
- 11-17 00:30 — scale-gate pre-registration signed off (`7d861a7`); 00:33 merges, `main` = fd6fd06 pushed (199 / 38 / 0).
  ≈ 02–05Z G9-P4 beats G9q step-matched at 25M and 50M ⇒ **CANDIDATE**; decision: keep both (G9q main on the A30, G9-P4
  challenger on the A40) (g9q-25/26). **03:36 incident: G9q stopped at its step cap** (netv2.env v19 `TOTAL=1e9`): three
  successors exited at the cap, the watchdog wrote `CHAIN_STOPPED` at 03:42 and the **A30 allocation was lost**; fix
  11:25 netv2.env v20 (md5 9fc5196d, cap 3·10⁹ for G9q only), successor 1174222 verified 12:02 at 1,002.1M steps; the
  check-in now flags STALE / STEPCAP [M ORCHESTRATOR.md, netv2 HANDOFF]. 10:23 scale gate chain B: T2 FUTILITY (ns-21).
  10:24 **LA2-FP NOT SHOWN**: lookahead2 58.20% vs Foul Play, +1.30 ± 4.01 over its raw net (ns-18…20).
- Pending at compilation: LA2-H, fpcal, scale gate T1, later G9-P4 readings.

## Incidents (index)
| date | incident | consequence / fix | source |
|---|---|---|---|
| ≈ 10-2x (undated) | an rsync of `src` into the cluster repo under a running training job crashed it (workers spawned later imported new modules next to old ones) | rule: never change code a running job imports; trainers run from read-only `src-<sha>` snapshots | memory `imperial-slurm-training.md` |
| 10-28 | cluster Python moved to 3.14 / missing 3.12 | standalone CPython 3.12.16 for every venv | memory |
| 11-02, 11-09 | QOS and partition changes | `--partition=long`, then `training` | memory |
| 11-05 | arm deadlocked 8.7 h | 45-min stall watchdog `37db029` | git |
| 11-07 | Rest / sleep-count bug ⇒ engine panic; data before 11-07 affected | `006ff76`; data flagged | git, PIPELINE.md |
| 11-09 | netv2 pool-worker hang (forkserver + POKE_LOOP) | `972726f`, pool alarm guard | git |
| 11-10 | Stage 1 re-score matched rows by id only | fixed `9ac4785` before registration | NETSEARCH-PREREG.md |
| 11-10 | actor-stream drift | reverted `d735897` | v2g9-21 |
| 11-11, 11-14 | data host self-submission parse (submit banner) | `029c1c5` | data HANDOFF |
| 11-12 | V3 I1: CUDA-graph capture error (unpinned host→device copy) | graph configs untested; I1 failed on eager configs | V3-I1-PREREG.md |
| 11-14 | Showdown h2h hang at websocket login (0 games) | hang guard (kill after 12 min, void, replay) | `$C/runs/h2h-20261114-152241-1136569` |
| 11-15 | ~700k-token stream agent | short-lived-agent rule | memory `agent-token-budget.md` |
| 11-15 | stale R1b-F follow-up rule could reset 360M → 7.5M | disarmed (`runs/r1/FU.started`) | memory |
| 11-17 | **G9q step cap**: stop at 1.0B, A30 allocation lost | netv2.env v20 (cap 3e9), STALE/STEPCAP alerts | ORCHESTRATOR.md |
| 11-19 | Foul Play websocket handshake timeouts at harness start (slow shared disk), 0 games: fpcal arm L 1174394 (dove04), F1350 1228320 (dove01) | voided, replayed with the same seed (PLAN rule); not node-specific | `$C/runs/conf1350-void1/NOTES.txt` |
| 11-18 → 11-20 | R1b-F hold chain (a job step) died with its job at a hand-over; a later trip waited 1.7 h for a decision | relaunched in the new job (hold-chain-r3.sh); job-step chains must be re-armed after every hand-over | `$C/runs/g4r1bf-hold-chain/NOTES-r3.txt` |

## Standing user decisions (index)
2026-10-19 Wang recipe · 10-2x maximum speed, accept FairShare cost, no heavy jobs on the laptop · 11-08 goal (best gen9/gen4
bot, beat Jaxcalibur), no Jaxcalibur distillation, crawl ≥ 1500 only, bot-registry yes · 11-09 sequential tests internal only
· 11-10 G9q = option A · 11-12 V3 → Rust design; orchestrator may push `main` · 11-14 cluster-side pipelines · 11-15 full
delegation, short-lived agents, G9q2 · 11-16 raw net as strong as Jaxcalibur's; "do 1 and 2" · monthly random-battle sets only
at new runs; benchmark server only with a re-baseline (undated, PLAN.md) · the user warned against an un-designed "testing
random stuff loop" (undated, memory `benchmark-method.md`). [M memory / PLAN.md / ORCHESTRATOR.md]
- 2026-11-18 16:41Z — A40 job 1173274 reached its 3-day limit; successor 1173283 (gpuvm33) resumed G9-P4 exactly (upd 5853 @ 243,290,742 → upd 5854 @ 243,332,444) [M `runs/g9/20261118-164715-1173283/G9P4/train.log`]; its slurm header names the generic g9 resume file, the g9p4 branch overrides it with the lineage's own checkpoint (netv2.env v20). Restart flag re-armed 16:58Z.
- 2026-11-18 17:40Z — T4 job 1137378 (R1b-F) reached its limit; successor 1173311 (dove01) resumed exactly (upd 11576 @ 482,186,281 → upd 11577 @ 482,227,948) [M `runs/g4r1bf/20261118-174152-1173311/R1bF/train.log`]; restart flag re-armed 18:27Z.
- 2026-11-18 17:28Z — G9-P4 vs G9q @245M 50.5 [49.1, 51.9]: the P4 head start washed out (results g9q-28); G9q stays main.
- 2026-11-18 21:12Z — BIGNET: a 2×-wide BC net (11.6M params) trained on the same P4 data as BC-P4 plays even with it (50.13 [49.09, 51.17], 8000 g; held-out policy CE −0.001); at matched compute it is worse (47.99). Capacity is not the bottleneck of BC at 12M decisions. The bigger-net arm stops at this size (pre-registered). [M `docs/thesis/data/bignet/RESULT.txt`]
- 2026-11-19 00:23Z — scale gate T1 FUTILITY: deepen D4 vs D1 51.72 [49.16, 54.40] (1200 g) ⇒ FLAT, the deeper-search route stops (results ns-25).
- 2026-11-19 06:38Z / 10:33Z — fpcal SENSITIVE: raw cd@805 vs Foul Play 61.3 / 57.8 / 54.9% at 50k / 210k (time-based) / 800k iterations, −3.2 pts per 4× budget (fpc-02…04); arm L was first void (Foul Play websocket handshake timeout, job 1174394) and rerun with the same seed. Decision: new headline Foul Play runs at a fixed 200k iterations/state.
- 2026-11-19 10:34Z — ExIt-LA2 stage 1 starts (fastenv helper job 1229025, dove02): cd@805 fine-tuned toward lookahead2's played policy vs toward its own policy.
- 2026-11-19 10:56–11:39Z — G9q-cd@1350 cooldown branch INLINE in G9q's job (final `runs/g9q-cd1350/final/01355036714.pt`, md5 6cf40c0c); G9q relaunched in place, counter continues (1,350,851,430 → 1,350,894,821) [M `runs/g9q-cd1350/STATUS`].
- 2026-11-19 11:52Z — **R1350 SUCCESS**: cd@1350 beats cd@805, MUE 52.05 [50.45, 53.62] (3400 g, look 17) ⇒ new deployment-net candidate; F1350 released (g9q-29).
- 2026-11-19 12:32Z — F1350 attempt 1 (job 1228320) void: Foul Play never connected (websocket handshake timeouts for 30 min, slow shared disk), 0 games; 12:50Z replayed unchanged with the same seed as job 1238699 [M `$C/runs/conf1350-void1/NOTES.txt`].
- 2026-11-19 14:32Z — **F1350 CONFIRMED**: raw G9q-cd@1350 vs Foul Play gen9 59.1 [56.2, 62.0] (906 g; two Foul Play instances stopped accepting after 4–5 games) (g9q-30) ⇒ cd@1350 = deployment net. LA2-1350 (lookahead2 adoption test, PREREG d2308e2e) starts: δ* 0.0075 for cd@1350, 500 pairs from 15:31Z.
- 2026-11-19 ~16:00Z — CSG asked the top /vol/bitbucket users to free space (filesystem 99% full; ours ≈ 314 GB by CSG's count, 351 GB apparent). User approved deleting 5,264 old opponent-pool snapshots (126.9 GB) and 689 eval-time model copies (16.6 GB) from netv2 mirrors no job reads (live mirrors, pool/frozen, checkpoints, finals, results kept); manifest `/vol/bitbucket/fr1226/porygon-thesis/deleted-20261119-pools-evalsnaps.tsv.gz`.
- 2026-11-19 22:23Z — **LA2-1350 ADOPTED (NON-INFERIOR)**: lookahead2 on cd@1350 (δ* 0.0075) vs lookahead2 on cd@805 51.20 [48.36, 54.04] (1000 g, 6.8 h) ⇒ gen9 benchmark bot = lookahead2 on G9q-cd@1350 (ns-26).
- 2026-11-19 23:32Z — **ExIt-LA2 stage 1 HARMFUL**: cd@805 fine-tuned toward lookahead2's played policy loses to its own-policy control 48.17 [47.47, 48.87] and to cd@805 48.14 [47.45, 48.83] (≈ 15.8k games each); the policy moved (r_E 0.347), the value did not; the own-policy fine-tune alone ≈ neutral (49.41) (ns-27…29). Stage 2 not run; the ExIt-LA2 route as designed stops.
- 2026-11-20 11:23Z — A30 job 1174222 (G9q) reached its limit; successor 1174225 (clapper) resumed exactly (upd 35338 @ 1,521,187,377 → upd 35339 @ 1,521,230,075) [M `runs/r1/20261120-114114-1174225/G9q/train.log`]; restart flag re-armed 14:35Z.
- 2026-11-20 14:56Z — R1b-F second guard trip (eval 74.5% < best 85.5% − 10%) ⇒ HOLD. The pre-registered hold chain had died with job 1137378 at the 11-18 hand-over; the orchestrator relaunched it 16:37Z in 1173311 (two-line copy: pending trip unseen, k=2). 16:43Z HOLD #3 CONTINUE: trip@552.7M vs @300M 55.2 [53.2, 57.2] (v2g4-15); trainer relaunched same config. Third CONTINUE in a row: the eval-score guard is noisy.
- 2026-11-20 15:28Z — VTG probe: under G9q's recipe the value loss's trunk gradient is ≈ 3–4% of the policy loss's per minibatch (signal ratio 0.2–0.45), the reverse of MissingNo.Zero's ≈ 50× ⇒ no value_trunk_grad arm (dg-06). Side finding: global-norm clipping active in 100% of minibatches (coef ≈ 0.4); policy-gradient coherence ≈ 0.005.
- 2026-11-21 00:13Z → (ongoing at 16:29Z) — the cluster's Slurm controller (vm-slurmctld01) is DOWN: running jobs keep training, but job hand-overs (A40 1173283 ends 16:41Z → G9E; T4 1173311 ends ≈ 17:40Z) cannot start their successors until it returns.
- 2026-11-21 — dg-07: cd@1350 vs Foul Play by length 51.3% (< 25 turns) / 63.0% (≥ 25): the short-game gap persists.
- 2026-11-23 ≈ 09:10Z — the Slurm controller is back (down ≈ 11-21 00:13Z → 11-23 morning, ≈ 57 h). Over the outage, the A40 and the T4 sat idle for ≈ 42 h and ≈ 40 h. R1b-F's successor 1175389 (dove04) resumed exactly (594,933,352 → upd 594,975,137) and its restart flag was re-armed at 11:03Z. The A40 successor 1175374 (gpuvm33) parked G9-P4 at 641,887,817 and started **G9E** at 10:28Z from cd@1350 against a frozen cd@1350 only (cap 1,455,036,714).
- 2026-11-23 12:12–12:56Z — G9q-cd@2030 cooldown INLINE in G9q's job 1483048 (branch at ckpt 2,030,019,516; final 02035041391.pt md5 55940b3f); G9q relaunched in place (2,033,454,285 → 2,033,497,593).
- 2026-11-23 ≈ 13:05Z — **R2030 SUCCESS**: cd@2030 beats cd@1350 MUE 51.99 [50.40, 53.56] (3600 g) (g9q-31) ⇒ F2030 released.
- 2026-11-23 14:10Z — **F2030 CONFIRMED**: raw cd@2030 vs Foul Play gen9 at FIXED 200k iterations 60.8 [57.9, 63.7] (1000 g; ITERS_OK) (g9q-32) ⇒ cd@2030 = deployment net; first headline at the fixed yardstick. Short games 53.4% vs long 64.6% (dg-08).
- 2026-11-24 01:02Z — **LA2-2030 ADOPTED (NON-INFERIOR)**: lookahead2 on cd@2030 (δ* 0.0025) vs on cd@1350 50.00 [47.05, 52.95] (ns-31) ⇒ benchmark bot = lookahead2 on cd@2030; its headline vs Foul Play at fixed 200k **61.60 [58.80, 64.40]** (ns-32).
- 2026-11-24 07:17Z — **G9E ROBUST**: an exploiter trained 100M steps only against frozen cd@1350 (from its weights) scores 50.47 [49.05, 51.90] against it, and the same against uncooled G9q@2155M (50.53) and cd@805 (50.45) (ns-30): cd@1350 has no blind spot a dedicated RL agent finds at this scale ⇒ the league / opponent-diversity route is closed. The A40 holds for a decision (before 11-26 10:27Z).
- 2026-11-24 11:44Z — G9-P4 resumed on the A40 as a keep-alive (user). 13:04Z netv2.env v22: step caps G9q 5e9, G9-P4 2e9, R1b-F 2e9 (from the 11-26 hand-overs).
- 2026-11-24 12:32–13:21Z — **Research council session 1** (Claude Opus 5.5, GPT-6.2 Astra, Gemini 4 Carbon; read-only, anonymised; R1 proposals → R2 cross-critique → R3 rebuttal; 14.26M gross / 2.21M fresh tokens of the student's 30M cap): 16 proposals → portfolio `docs/council/sessions/2026-11-24/PORTFOLIO.md`. Unanimous top pick: own pre-Tera visibility (the encoders leave our own Tera type blank until we Terastallize; 60% of gen9 sets are Tera-ambiguous; orchestrator-verified in the code). The student signed off P1–P3 (~13:45Z).
- 2026-11-24 14:02Z — **Council P2 NO-GO**: ExIt with a soft completed-Q target on ns-27's positions: E′ vs control 50.20 [49.25, 51.16], vs parent 50.09 [49.11, 51.07] (ns-33/34) ⇒ the hard target caused ExIt-LA2's harm, but the soft target adds nothing: ExIt-LA2 closed in both forms.
