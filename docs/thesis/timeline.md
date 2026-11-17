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

## Standing user decisions (index)
2026-10-19 Wang recipe · 10-2x maximum speed, accept FairShare cost, no heavy jobs on the laptop · 11-08 goal (best gen9/gen4
bot, beat Jaxcalibur), no Jaxcalibur distillation, crawl ≥ 1500 only, bot-registry yes · 11-09 sequential tests internal only
· 11-10 G9q = option A · 11-12 V3 → Rust design; orchestrator may push `main` · 11-14 cluster-side pipelines · 11-15 full
delegation, short-lived agents, G9q2 · 11-16 raw net as strong as Jaxcalibur's; "do 1 and 2" · monthly random-battle sets only
at new runs; benchmark server only with a re-baseline (undated, PLAN.md) · the user warned against an un-designed "testing
random stuff loop" (undated, memory `benchmark-method.md`). [M memory / PLAN.md / ORCHESTRATOR.md]
