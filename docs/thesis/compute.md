# Compute record

All compute used by Porygon from the first commit (2025-03-02) to the snapshot below: the Imperial DoC Slurm
cluster (`gpucluster2`, user `fr1226`; almost everything), RunPod (2026-10-19/20 only) and the laptop.
Provenance tags follow `README.md`: **[M]** measured, **[D]** derived (arithmetic shown or in the script),
**[E]** estimate (basis and range given). Times UTC. Refresh with `python3 scripts/thesis/compute_report.py`
(it rewrites only the AUTO block below and the CSVs in `data/`).

## Summary (snapshot 2026-11-17 12:46Z)

| platform | GPU-hours | CPU-core-hours | provenance |
|---|---:|---:|---|
| Imperial Slurm (839 job records, 2026-10-07 → now) | 1,427.9 | 25,741.5 | [M] per job (jobcomp + squeue), sums [D] |
| RunPod (3 pods, 2026-10-19 11:30 → 10-20 ~14:50) | ≈ 15.8 (range 14–18) | ≈ 400 vCPU-h (range 300–500) | [D] from balance readings ÷ listed $/h; vCPU [E] |
| Laptop (i5-1235U, 10 cores / 12 threads, no CUDA GPU) | 0 | 20–80 | [E] |
| **total** | **≈ 1,444** | **≈ 26,200** | |

- **Measured share:** 1,427.9 of ≈ 1,444 GPU-hours (≈ 98.9%) and 25,741.5 of ≈ 26,200 CPU-core-hours (≈ 98%) are
  [M]; the rest (RunPod, laptop) is [D]/[E] [D: 1,427.9 / (1,427.9 + 15.8) = 0.989].
- **Slurm GPU-hours by type [D]:** T4 551.3 · A16 503.5 · A30 322.6 · A40 50.3 · A100 0.1. Plus RunPod RTX 3090
  ≈ 13.8 h and one ≈ 2 h pod of unrecorded type.
- **By purpose [D]:** RL training 739.4 GPU-h (51.8%), Showdown benchmarks vs Foul Play / our search 351.3 (24.6%),
  data 218.8 (15.3%), diagnostics 56.2 (3.9%), in-process / engine head-to-heads 43.9 (3.1%), BC pretraining and
  distillation as separate jobs 16.2 (1.1%), builds / probes 2.0.
- **By month [D]:** 2026-10: 216.1 GPU-h, 5,703 CPU-core-h; 2026-11 (to the 17th): 1,211.7 GPU-h, 20,038 CPU-core-h.
  2025-03 (the first prototype) used the laptop only [E].
- **Energy [E]:** GPUs ≤ 138 kWh at TDP (upper bound) + CPUs 77–154 kWh ⇒ ≤ ≈ 290 kWh on the cluster; RunPod
  ≤ ≈ 6 kWh (15.8 h × 350 W); laptop < 1 kWh. See "Energy" below for the basis.

## Method

### Job accounting
- **`sacct` / `sreport` were down for the whole session** (retried 2026-11-17 12:3xZ and 12:46Z:
  `_open_persist_conn: failed to open persistent connection to host:localhost:6819: Connection refused`; slurmdbd is
  configured at `AccountingStorageHost = localhost`, port 6819).
- **Fallback used — the Slurm job-completion store [M].** `scontrol show config` lists
  `JobCompType = jobcomp/elasticsearch`, `JobCompLoc = http://cloud-vm-45-09.doc.ic.ac.uk:9200/slurmdev/_doc`.
  Querying it for `username:fr1226` returned **838 records** (jobs 296121 → 1174237, 2026-10-07 17:34 → 2026-11-17
  12:21): per job the name, partition, node, `tres_alloc` (CPUs, `gres/gpu`), `@start`/`@end`, `elapsed`, state and
  work dir. Times are UTC (checked: run dir `runs/20261020-181327-299545` was stamped by the node in BST, one hour after
  the record's `@start` 17:13). Raw dumps: `/vol/bitbucket/fr1226/porygon-thesis/raw/jobcomp-<stamp>.json`.
- **Running jobs [M]** come from `squeue` at the snapshot (elapsed so far): 1105603 data host (A16, since 11-14 18:11),
  1137378 R1b-F (T4), 1173274 G9-P4 (A40), 1173672 LA2-H slot job (T4), 1174222 G9q (A30). They contribute 157.0 GPU-h;
  they will grow until they end, so re-run the script.
- **GPU type** = the node's gres from `sinfo -N -o "%N %G"` [M]: clapper/dipper/hopper A30; gpuvm33–35 A40; gpuvm36,
  parrot A16; dove01–04 T4; vm-he-a100, linnet, merlin A100 80 GB. `vm-kn-a16` has left sinfo; its type A16 is from
  the job logs [M] (`porygon-ws-fastenv/logs/slurm-687853.out`: "GPU 0: NVIDIA A16"; `porygon-ws-netv2/logs/slurm-
  710000.out`: "cuda ok NVIDIA A16"). Every job allocated exactly one whole GPU (`gres/gpu=1`; no shards).
- **Allocated, not used.** GPU-hours and CPU-core-hours are allocation × elapsed. On the A30s the site's submit plugin
  gives every 1-GPU job 30 CPUs whatever `-c` says (memory `imperial-slurm-training.md`), and every partition requires a
  GPU, so CPU-bound jobs (Foul Play benchmarks, crawling, diagnostics) hold a mostly idle GPU.
- **Completeness check [M].** All 804 job ids that appear in `slurm-<id>.out`-style log names anywhere under
  `/vol/bitbucket/fr1226/` (gloria, gloria-gen9, porygon-ws-*, foul-play; list in `raw/logfiles.txt`) are in the
  jobcomp records or in squeue (0 missing). The 35 records without a log file are all cancelled-before-start or
  sub-minute jobs (0.2 h in total). The first record (296121 `a100-probe`, 2026-10-07, `/homes/fr1226/gpu_probe`) is
  the cluster-access probe; 10 more `probe` jobs (300359–300368, 10-23) total < 0.1 h. No Porygon job predates
  2026-10-20 (299539 `gloria-test`), consistent with the git history (cluster scripts added 2026-10-20, 205d632).
- **Jobs inside jobs.** Pretraining, in-process head-to-heads and chain driver steps that other streams ran inside the
  data-host job or a training job via `srun --jobid=J --overlap` add no allocation and are counted under the host job's
  purpose. Known cases [D, from the logs' own timestamps]: BC P2 (1.5 h), P3 (4.6 h) and the 11-09 BC arms
  (vrefit / vr-1pg / vr-c005 / p2ft / p2mix, ≈ 4 h) inside 677193 (A16, "data"); U7 (4.7 h) inside 810087 (A16,
  "data"); BC-P4 (3.1 h) inside 1173274 (A40, "RL training"). So BC pretraining used ≈ 18 A16-h + 3 A40-h in all,
  not the 16.2 GPU-h of its own jobs. The cooldown branches, in-job Showdown checks and async evals run inline in the
  RL jobs and stay under RL training.

### Purpose classification
By job name (`PURPOSE_RULES` in the script; first match wins):
- **RL training (PPO):** `gloria`, `porygon` (= `slurm_train.sbatch`, v1), `arm-*` (v1 arms),
  `ws-netv2-{r0,r1,r1long,g9,g4,train}` (v2 lineages, with their inline cooldowns, checks and P4 BC).
- **BC pretraining / distillation:** `ws-netv2-pre`, `ws-data-bc`, `porygon-expert` (expert iteration: search
  labels + distillation), `porygon-distill`.
- **Benchmarks on Showdown:** `porygon-foulplay`, `porygon-search`, `porygon-oracle`, `cm-*` (compute-matched),
  `or-*` (oracle), `g9-{fit,stock,sx2}-*` (gen9 eval variants vs Foul Play), `ws-netv2-{fp*,fp9-*,rawfp,conf*,
  vs-search,vs9-*,slottest2}`, `ws-wsp-fpcal`, `ws-fastenv-{la2slot,netsearch,scale,smoke,fpsmoke,refreeze1}`,
  `ws-data-{bceval9*,teacher9}`.
- **Head-to-heads (in-process / engine):** `porygon-h2h*`, `gen9-h2h-*`, `ws-wsp-{h2h,spd}`,
  `ws-fastenv-{eval,cmp,helper}` (the helper hosts in-process chains such as Bet 1 and scale-gate chain B).
- **Data:** `ws-data-{host,searchlabels*,normalise}` (crawler, inputlog re-simulation, shards, search labels),
  `gen9-data`, `porygon-selfplay` (engine self-play for the eval fit), `porygon-fit`, `gen9-fit`.
- **Diagnostics / dev:** `ws-wsp-{v2diag,v2pos,box,selfplay,spdbench,spdfp*}` (v2diag = 572 short tracker jobs),
  `ws-fastenv-{diag*,bench,dbench,dev,v3gates,v3i1,eng-spike,nnueparity,train4}`, `ws-netv2-prof`.
- **Builds / probes:** `probe`, `a100-probe`, `porygon-{build,engine,pycheck}`, `gloria-test`, `ws-netv2-smoke`.

### RunPod [D/E]
Sources: the session transcript of 2026-10-19/20 (Claude Code log `71e9ec3a…jsonl`, RunPod API outputs) and a
read-only RunPod GraphQL query on 2026-11-17 [M]. Balance readings [M]: 10-19 11:22 $7.269 → 12:26 $6.793 → 13:41
$6.161 → 15:03 $5.487 → 16:09 $4.940 → 16:37 $4.686 → 18:59 $4.539 → 20:17 $4.239 → 22:11 $3.808 → 10-20 03:03
$2.721 → 03:16 $2.664 → 11:12 $2.556 → 12:29 $2.531 → 14:50 $2.062 → 15:03 $2.061; 2026-11-17: $0.776, spend now
$0.001/h (one exited pod's disk), spend limit $80.
- **Pod A** (RTX 3090 listed at $0.50/h, 11:38Z): the balance fell at $0.49–0.54/h from ≈ 11:30 to ≈ 16:50 ⇒
  ≈ 5.3 GPU-h, $2.58 [D].
- **Pod B** `gloria-train` (RTX 3090, 28 vCPU, $0.22/h, `lastStartedAt` 2026-10-19 18:51:03Z [M, API]): $0.22–0.23/h
  from 18:51 to ≈ 03:20 10-20, then $0.01/h (stopped) ⇒ ≈ 8.5 GPU-h, ≈ $1.9 [D]. Its run is the laptop copy
  `runs/20261019-185359/best_model.zip`.
- **Pod C** (unrecorded GPU, $0.24/h listed 12:38Z): $0.47 between 12:29 and 14:50 10-20 ⇒ ≈ 2.0 GPU-h [D]
  (checkpoint tournament, `runs/phase0/tournament_full.txt` 15:20Z).
- **Total** ≈ 15.8 GPU-h (range 14–18), $5.21 between the first and last 10-19/20 readings; $6.49 from the first
  reading to now. Spend before 10-19 11:22 is not recorded (the initial deposit is unknown) — **gap**.
- vCPU-hours [E]: pod B 28 vCPU × 8.5 h = 238; pods A/C ≈ 31 vCPU (README of 10-20: "31-vCPU RTX 3090 pod") ×
  ≈ 5.3 h + 2 h ⇒ ≈ 400 vCPU-h in all (range 300–500).
- What ran: the v1 network's first 60M PPO steps (README 2026-10-20: "60M self-play steps", "~1.4k steps/s on a
  31-vCPU RTX 3090 pod"; 60M / 1.4k s⁻¹ ≈ 11.9 h [D], consistent with pods A + B ≈ 13.8 h) and the checkpoint
  tournament. The cluster run resumed at 60.05M (job 299541, 2026-10-20 16:52).

### Laptop [E]
Surface laptop, Intel i5-1235U (10 cores / 12 threads, 15 W base power), no CUDA GPU [M]. Used for development, the
2025-03 prototype (4 commits over 2 days), the test suite at each merge (≈ 15 min on 2 threads per run), a CPU
checkpoint tournament (`runs/phase0/tournament_cpu.txt`, 10-20), 20 ladder games (13–7, README of 10-21) and
`porygon.play` sessions; the user rule since 10-21 is "no heavy compute on the laptop". Estimate 20–80 core-hours,
< 1 kWh. Not measured — **gap** (no process accounting was kept).

### Energy [E]
Board power limits (vendor TDP): A30 165 W, A40 300 W, A16 250 W per board of 4 GPUs ⇒ 62.5 W per GPU, T4 70 W,
A100 80 GB PCIe 300 W, RTX 3090 350 W. GPU energy = GPU-h × TDP is an **upper bound**: the RL trainers drive their
GPU (BC pretraining logged 64–100% utilisation in `mem.log`), but the benchmark, data and diagnostic jobs (≈ 45% of
GPU-hours) are CPU-bound. CPU energy uses 3–6 W per allocated core (server sockets of 150–280 W over 32–64 cores,
with a share of uncore). Cooling / PUE not included.

## Tables (generated)

<!-- AUTO:BEGIN compute -->
_Generated by `scripts/thesis/compute_report.py` at 20261117T1246Z (source: jobcomp + squeue; 839 job records, 798 with non-zero elapsed). Every row is [M] from the job records; the totals and splits are [D] (sums of allocated time; month splits pro rata by wall time); running jobs count their elapsed time up to the snapshot._

**Grand total (Imperial DoC Slurm):** 1,427.9 GPU-hours allocated, 25,741.5 CPU-core-hours allocated [D]; of these 157.0 GPU-h belong to jobs still running at the snapshot.

### By GPU type [D]

| GPU | jobs | GPU-hours | CPU-core-hours | TDP per GPU [E] | GPU energy at TDP, kWh [E, upper bound] |
|---|---:|---:|---:|---:|---:|
| T4 | 682 | 551.3 | 9,221.8 | 70 W | 38.6 |
| A16 | 88 | 503.5 | 6,350.2 | 62.5 W | 31.5 |
| A30 | 29 | 322.6 | 9,678.6 | 165 W | 53.2 |
| A40 | 5 | 50.3 | 490.5 | 300 W | 15.1 |
| A100-80GB | 2 | 0.1 | 0.4 | 300 W | 0.0 |
| (never started) | 33 | 0.0 | 0.0 | — W | 0.0 |
| **total** | 839 | **1,427.9** | **25,741.5** | | **138.4** |

CPU energy [E]: 25,741.5 core-h × 3–6 W per core = 77.2–154.4 kWh. Total [E]: ≤ 292.9 kWh (GPU at TDP is an upper bound: most benchmark, data and diagnostic jobs hold a GPU only because every partition requires one, and leave it near idle).

### By purpose [D]

| purpose | jobs | GPU-hours | share | CPU-core-hours | share | A30 | A40 | A16 | T4 | A100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RL training (PPO) | 43 | 739.4 | 51.8% | 13,730.2 | 53.3% | 311.1 | 43.9 | 144.3 | 240.2 | 0.0 |
| benchmarks on Showdown (vs Foul Play / our search) | 111 | 351.3 | 24.6% | 7,953.5 | 30.9% | 0.0 | 0.0 | 129.8 | 221.5 | 0.0 |
| data (crawl, re-simulation, labels, self-play data, eval fits) | 14 | 218.8 | 15.3% | 1,952.5 | 7.6% | 0.0 | 0.3 | 213.0 | 5.4 | 0.0 |
| diagnostics / dev / microbenchmarks | 608 | 56.2 | 3.9% | 502.6 | 2.0% | 0.0 | 6.1 | 3.6 | 46.5 | 0.0 |
| head-to-heads (in-process / engine) | 33 | 43.9 | 3.1% | 1,217.6 | 4.7% | 0.0 | 0.0 | 6.3 | 37.6 | 0.0 |
| BC pretraining / distillation | 5 | 16.2 | 1.1% | 362.0 | 1.4% | 11.2 | 0.0 | 5.0 | 0.0 | 0.0 |
| builds / tests / probes | 25 | 2.0 | 0.1% | 23.1 | 0.1% | 0.3 | 0.0 | 1.6 | 0.0 | 0.1 |

### By workstream [D]

| stream | jobs | GPU-hours | CPU-core-hours |
|---|---:|---:|---:|
| netv2 | 54 | 641.6 | 11,275.6 |
| pre-campaign (gloria repo) | 134 | 419.2 | 10,299.1 |
| data | 15 | 221.8 | 1,941.5 |
| fastenv | 42 | 104.3 | 1,989.6 |
| wsp | 594 | 40.9 | 235.8 |

### By month (pro rata by wall time) [D]

| month | GPU-hours | A30 | A40 | A16 | T4 | A100-80GB | CPU-core-hours | RL training GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-10 | 216.1 | 126.5 | 0.0 | 89.5 | 0.0 | 0.1 | 5,703.4 | 114.9 |
| 2026-11 | 1,211.7 | 196.2 | 50.3 | 414.0 | 551.3 | 0.0 | 20,038.1 | 624.5 |

Job states [M]: COMPLETED 745, CANCELLED 58, FAILED 17, TIMEOUT 12, RUNNING 5, OUT_OF_MEMORY 2.

Largest jobs [M]: 677193 ws-data-searchlabels (A16, 72.0 h); 810367 ws-netv2-g9 (A16, 72.0 h); 810087 ws-data-host (A16, 72.0 h); 741849 ws-netv2-g9 (A16, 71.8 h); 1136569 ws-netv2-g4 (T4, 71.8 h); 1135184 ws-netv2-r1 (A30, 71.8 h); 718701 ws-netv2-r1 (A30, 71.8 h); 829660 ws-netv2-g4 (T4, 71.8 h).
<!-- AUTO:END compute -->

## Training compute by lineage [D]
Wall hours of the trainer segments in each lineage's train.log (details and steps in `runs.md`, data in
`data/lineages.csv`). The RL jobs' 739.4 GPU-h split as 718.5 h of trainer segments and ≈ 21 h of startup,
pause/HOLD windows, inline BC-P4 and the bf16 microbenchmark (97% accounted for).

| lineage | GPU | wall h |
|---|---|---:|
| R1b-F (gen4 v2, running) | T4 | 184.6 |
| G9q (gen9 main, running) | A16 33.9 + A30 129.3 | 163.2 |
| v1 main 60M → 600M | A30 | 115.0 (+ ≈ 13.8 RunPod 3090 for 0 → 60M) |
| G9p | A30 45.0 + A16 8.4 | 53.4 |
| v1 arms (ctrl, x30, x300, r0lr) | T4 53.8 + A30 1.7 | 55.5 |
| G9r + G9r-cd@50 + G9q-cd@50 control | A16 | 45.7 |
| G9t + G9t-cd@50 | A16 | 33.1 |
| G9s | A16 12.5 + A30 7.4 | 19.8 |
| G9-P4 (running) | A40 | 15.5 |
| G9q2 | A40 | 9.6 |
| G9qent + G9qent-cd@880 + G9q-cd@885 control | A40 | 11.0 |
| R1 + R1b | A30 5.3 + A16 2.1 | 7.4 |
| G9q / G9p cooldowns @104.7, @305, @605, @805; actor-stream A/B | A30 | 5.2 |

## Gaps and caveats
1. **sacct was down**; the jobcomp store is the Slurm controller's own completion record (same fields), so the
   per-job numbers are still [M]. A later run with sacct up should agree; if it differs, keep both (strike through).
2. **RunPod before 2026-10-19 11:22** is unrecorded (deposit unknown); pod hours are [D] from balance deltas, with
   ±10% error from the reading times; pod C's GPU type is unknown.
3. **Laptop** compute is [E] only.
4. **Hosted steps** (`srun --overlap`) are not separable in the job records; the BC split above uses the runs' own
   log timestamps. In-process head-to-heads run inside the data-host job (e.g. the P4 R25/R50 readings, G9q2 readings)
   stay under "data".
5. **Running jobs** are counted to the snapshot; pending jobs (7 at the snapshot) count nothing until they start.
6. **Energy** is an upper bound at TDP, not a measurement (the cluster records `energy` in AccountingStorageTRES, but
   only sacct exposes it).

## GPU Dojo VM (Azure, 1× T4, 4 vCPU) — outside Slurm
| date | use | VM wall time | source |
|---|---|---|---|
| 2026-11-17 | setup (venv, Node 22, Showdown/fastenv/netv2 copies) + 50-pair smoke + 2000-pair cross-machine validation (cal-vm1) | ≈ 0.75 h (VM up 16:44Z → shutdown 17:37Z) [M] | VM `uptime` at 16:50Z (up 6 min); shutdown sent 17:37:14Z |
Budget: 100 free GPU-hours, counted while the VM runs (user, 11-17).
| 2026-11-18 | BIGNET (bigger-network BC, PREREG ws/netv2 4597222b): boot 10:38Z, data push (7.64 GB, cluster → VM, 469 s), FAILED microbench 10:49 (missing package) ⇒ auto-shutdown; restarted 10:57Z, arm A D384L4 trained 11:02–21:00Z (140,880 steps, 9.97 h), E1A + E2A evals, DONE 21:12Z ⇒ auto-shutdown; restarted 22:53Z only to copy the result (shutdown 22:55Z) | ≈ 10.5 h (10:38–10:49, 10:57–21:12, 22:53–22:55) [M] | `docs/thesis/data/bignet/` |

VM total to 11-18: ≈ 11.4 of 100 GPU-hours [D].
