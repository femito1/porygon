# Porygon network-first campaign: plan and rules

Every agent reads this first. The orchestrator owns this file; streams don't edit it.
- **State and pending checks:** `ORCHESTRATOR.md`.
- **Per-stream state:** `<stream>/HANDOFF.md`.
- **Interfaces:** `CONTRACTS.md`.

## Goal
1. The best bot in gen9 random battles and in gen4 random battles.
2. A network that beats our search head to head on its own; then network + search beats everything.

The bar:
- **Jaxcalibur** (gen9, closed source; self-play transformer + pUCT; peak 2557 Elo, #1 on the ladder). Its author
  guesses its network alone ≈ 2400 Elo and search adds 100–150.
- **MissingNo.Zero** (≈3M-param PPO, no search): beats Foul Play 61.6% in gen4 and 66.7% in gen9 (Foul Play at an M4
  Pro core's budget).
- **Foul Play** (open source) is the strongest bot we can benchmark. A Jaxcalibur-level bot would beat it ~75% in
  gen9.

## Strategy (as of 2026-11-19)
- **gen9 is the primary target; gen4 is the validation bed.** All infra is generation-parametric.
- **Where we are:** benchmark bot = lookahead2 on G9q-cd@805M, 58.2% vs Foul Play; the raw net alone 57.8% and it beats
  our pure search (55.5%). Gap to Jaxcalibur ≈ 135–165 Elo. Foul Play numbers depend on Foul Play's budget (below).
- **What has worked:** the v2 net with human-BC init; long PPO training of one main run; lr-cooldown branches as
  deployment nets; lookahead2 (net-valued one-turn lookahead with a margin gate) as the only search with a measured gain.
- **What did not close the gap (don't retry as is):** deeper search — V3 (GPU net in MCTS, ~2k leaves/decision), NNUE
  CPU student (gate 1 failed), `porygon.deepen` (scale gate FLAT: 4× budget 51.7% vs 1×; overrides stay 11–13%); a 2×
  wider BC net (BIGNET 50.1%, data-limited); higher-rated human data (G9-P4's head start washed out by 245M); recipe arms
  (actor-stream overlap, lr annealing G9r, GAE λ 0.5 G9t, bf16, entropy schedule G9qent). Seed variance at matched steps
  < ~2 points (G9q2).
- **The network:**
  - **G9q** (v2 net, P3 human-BC init) is the MAIN run on the A30 (constant lr; cap 3e9). Its cooldown branches (5M
    steps cosine → ×0.1, entropy floor 0.35, INLINE in its job) are the deployment nets: cd@805 now; cd@1350 under test.
  - **Current bet: expert iteration with lookahead2 as the teacher** (ExIt-LA2 pilot; the earlier "search labels"
    failure had a teacher weaker than the net). A GO is iterated (better net ⇒ better lookahead2 ⇒ next round).
  - Known weakness: vs Foul Play we win 48% of games < 25 turns vs 63% of longer ones; lookahead2 does not fix it.
- **Network + search:** the gen9 benchmark bot is lookahead2(net, δ 0.005, K 8); a new deployment net gets an LA2-H-style
  adoption test (lookahead2 on it vs the current benchmark bot, adopt iff lower 95% > 50%).
- **Human data:** BC init quality matters for the start; under long RL the advantage fades. Search labels as targets only
  from a teacher stronger than the net (ExIt).
- **Gen4:** R1b-F (v2) trains with a strength watch (guard trips ⇒ HOLD ⇒ pre-registered h2h vs its 300M ckpt). The
  gen4 bot stays search + fitted eval with the v1@500M prior (59.2% vs Foul Play).
- **Random-battle sets (user):** switch to the current month's sets at the next NEW run, never under a running lineage.
  The benchmark server switches only with a re-baseline vs Foul Play.
- **Foul Play yardstick (orchestrator, 11-19, after fpcal SENSITIVE):** our score depends on Foul Play's budget (raw cd@805: 61.3 /
  57.8 / 54.9% at 50k / 210k (time-based) / 800k iterations per state; slope −3.2 points per 4×). New headline Foul Play
  benchmarks run Foul Play at a FIXED 200,000 iterations per state (≈ the time-based median, so they stay comparable with the
  history); runs already armed keep their pre-registered setting. Report the budget with every Foul Play number.
- **Sequential testing (user):** internal go/no-go tests only. Slot jobs and headline numbers stay fixed-n.

## Workstreams
| stream | worktree / branch | cluster dir | owns |
|---|---|---|---|
| netv2 | `.claude/worktrees/ws-netv2` / `ws/netv2` | `/vol/bitbucket/fr1226/porygon-ws-netv2/` | observation spec v2.1, model v2, PPO trainer, the gen9/gen4 RL runs, pretraining, network evaluation |
| data | `.claude/worktrees/ws-data` / `ws/data` | `/vol/bitbucket/fr1226/porygon-ws-data/` | human replay corpus (inputlog re-simulation), datasets and shards, search labels, label-teacher tests |
| fastenv | `.claude/worktrees/ws-fastenv` / `ws/fastenv` | `/vol/bitbucket/fr1226/porygon-ws-fastenv/` | in-process Showdown env and collector, in-process evaluation, Showdown set refresh |
| wsp | `.claude/worktrees/ws-wsp` / `ws/wsp` | `/vol/bitbucket/fr1226/porygon-ws-wsp/` | network diagnostics (trackers, profiles), sequential-test design, sharded in-process h2h |

- The `ws/*` branches were last merged into `main` on 2026-11-20 ~17:45Z (then 11-20 ~18:15Z ws/netv2 8a10fa6 = G9E armed; tests 228 / 38 / 0), and main is pushed. Stream commits after those are unmerged. Streams keep committing to their branches; merges are the orchestrator's call (user delegation). Merge with `git merge --no-ff ws/<stream>`, then run the tests from the main checkout: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=src timeout 900 .venv/bin/python -u -m pytest -q -p no:cacheprovider -o faulthandler_timeout=120 -rfE tests --deselect tests/test_encoder.py::test_random_battles --deselect tests/test_lookahead.py::test_leaves_match_embed_battle_and_lookahead_plays --deselect tests/test_search.py::test_search_battles` (those 3 need a live Showdown server; 0 failures required).
- Stream agents never push. The orchestrator pushes `main` only after the merged tests pass (user OK, 2026-11-12).

## Hard rules (all agents, all depths)
- **Subagent depth cap = 2.** Level-1 agents may spawn level-2 agents, which must not spawn any. Say so in every
  child prompt.
- **Code isolation.**
  - Work only in your own worktree/branch; never edit the main checkout or `main`.
  - Commit locally and don't push.
  - Commits carry only the user's name: no Co-Authored-By, no "Generated with Claude Code".
- **Cluster isolation.**
  - `/vol/bitbucket/fr1226/gloria` is READ-ONLY: use its venvs, node, pokemon-showdown, .engine-gen9, Foul Play
    and models by absolute path.
  - Each stream uses its own cluster dir. New venvs are built from `/vol/bitbucket/fr1226/python/cpython-3.12.16`.
  - Rust builds run inside jobs.
  - Read memory `imperial-slurm-training.md` before the first cluster action.
- **Quotas** (per-user QOS caps are shared; job names `ws-<stream>-*`).
  - **multigpu** (48 CPUs / 3 GPUs per user):
    - netv2's two RL jobs: the A30 (30 CPUs) and an A16/A40 (10 CPUs; widen pending jobs to a16,a40,a100);
    - data's host job (8 CPUs, self-chaining), which also hosts netv2 pretraining / in-process h2h via `srun --overlap`.
  - **long** (per-user CPU and GPU caps; every long job holds a GPU):
    - one shared benchmark slot (one 24-CPU job at a time, claimed in `ledger.md` under flock);
    - at most one helper job per stream.
    - **Since 2026-11-09 ~12:14 `long` = dove03–04 only.**
      - dove01–02 are reachable through `training`: QoS tempgpu, the same per-user caps as long, MaxTime 3 days.
      - Submit new dove jobs with `--partition=training` alone. It is a superset of long's nodes, with PriorityTier 62
        vs long's 10.
      - Never pass a comma list to sbatch: the submit plugin refuses multiple partitions.
      - Only a pending job can be widened, with `scontrol update jobid=J Partition=long,training`.
      - The slot and helper limits above cover both partitions together.
    - The tempgpu per-user caps are 64 CPUs / 4 GPUs (measured 11-17); the multigpu cap is 3 GPUs. A job over a cap
      pends with `QOSMaxGRESPerUser`; idle nodes elsewhere do not help.
  - **GPU Dojo VM** (the user's Azure T4, 4 vCPU, ≈ 88 of 100 free GPU-hours, counted while it runs): outside the Slurm
    caps; for long GPU-bound, CPU-light jobs and in-process h2h. The user starts it; jobs end with `sudo shutdown -h now`
    (memory `gpudojo-vm`).
  - Never `sbatch --export`. No heavy compute on the login node or the laptop.
- **Watchdog (netv2.env v12+, from 2026-11-12):** every lineage keeps one afterany successor queued. Before ANY deliberate scancel of a lineage job, touch `runs/<r>/CHAIN_STOPPED`; kill switch `runs/WATCHDOG_OFF`.
- **Cluster-side pipelines (user, 2026-11-14).** The laptop sleeps and agents die with it. Every approved,
  pre-registered multi-step pipeline (branch → test → verdict; smoke → slot run; successor checks) is written as a
  cluster-side job chain (sbatch with `afterok`/`afterany` dependencies, or a driver step inside a host job) that
  runs to its verdict file with no agent. Agents only prepare, launch and read the results. Each stage writes its
  own STATUS/RESULT file. A failed stage stops the chain and writes `FAILED` (the check-in alerts on it).
  Decisions that need a human or the orchestrator stay manual.
- **Never release a GPU allocation to change flags.** Restart in place. Slurm runs the script copy spooled at submit
  time. Never re-sync a script a running job reads from NFS.
- **Experiment discipline.**
  - Pre-register the question and the decision rule before results, and follow it.
  - Change one thing at a time. Power it.
  - Report 95% intervals from duplicate pairs.
  - Evaluate raw networks at T 0.5.
  - Every strength claim is confirmed in real games. In-process fastenv games count for net-vs-net and net-vs-bot;
    anything involving our search or Foul Play runs on Showdown.
- **General strength only.** Never model or exploit Foul Play's decision rule; confirm gains against our own bots
  too. Don't distil Jaxcalibur (the user said no).
- **Licences.** Foul Play is GPL-3.0: never copy its code.
- **fp32 only** for PPO until a bf16 A/B on v2 passes.
- **Process hygiene.**
  - Poll the cluster at most once a minute.
  - Never run a broad or unanchored pkill/kill: kill only your own processes, by recorded PID.
  - Anything that must keep running goes in a cluster job, not the laptop.
- **The laptop sleeps and has little RAM/disk** (~6 GB free disk, ~1–4 GB RAM). Keep laptop scratch < 100 MB per stream,
  no long-lived laptop loops; data, builds and checkpoints live on the cluster.
- **Don't touch memory.** The orchestrator consolidates memory.
- **Short-lived agents (user, 11-15):** one or two items per agent; arm the work as cluster chains, update the HANDOFF, report, stop. Never stay alive to monitor.
- **Math in reports:** Unicode, never LaTeX.
