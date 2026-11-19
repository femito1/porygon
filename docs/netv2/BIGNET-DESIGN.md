# BIGNET: the first bigger-network experiment (BC on P4, GPU Dojo VM) — design

netv2, 2026-11-17 ~22:00–23:30 UTC. ORCHESTRATOR.md 11-16 plan item 4: a bigger-net arm, as a step toward a raw net as
strong as Jaxcalibur's. The VM was OFF throughout: every VM number here is an estimate from cluster measurements,
labelled [E] with its basis; [M] = measured (file or command named). Pre-registration: `BIGNET-PREREG.md` (DRAFT
until main signs it off). Scripts: `scripts/netv2_bignet/`.

## 0. The question (refined, not widened)
Does a bigger v2 network, trained by behaviour cloning with BC-P4's exact recipe on BC-P4's exact data (P4, own
rating ≥ 1900, `$D/datasets/frozen/p4-gen9-20261116-1502/v2pre-human-shuf`), beat BC-P4 (5,809,977 params) head to
head at T 0.5? If yes, and the gain survives a same-hardware seed control, the next step is RL from it on the cluster.

Why it is worth asking [M]:
- **BC-P4 underfits.** Its late training CE (1.109–1.125 at steps 120k–140k, measured with id-dropout on) is within
  ≈ 0.02–0.03 of its held-out CE (1.13794 at 140,880); the val curve was still falling slowly at the end of the
  cosine (1.146 @105k → 1.140 @125k → 1.138 @140.9k) [M `runs/pretrain/g9-p4-s2/pretrain.log`]. No overfitting gap ⇒
  capacity (or steps) is binding, not data.
- **1.8× more data did not make BC stronger:** BC-P4 vs BC-P3 49.95 [47.81, 52.23] (bc-15) [M]. One reading
  consistent with underfitting: a 5.8M net cannot use the extra data.
- Jaxcalibur's net is ≈ 8.5M params (≈ 61 tokens; RL only); its author found "smaller networks trained for longer
  outperformed larger networks trained for fewer steps" — an RL compute statement. BC on a fixed data set is the
  cheap place to see whether capacity helps at all [research/REPORT.md §1.1].

## 1. Model size
**What the v2 cfg allows** (`src/porygon/v2/model.py` DEFAULT_CFG): d_model, n_layers, n_heads, ffn_mult (+ embedding
dims, pair_hidden, value_hidden). Production = d 256, 4 layers, 8 heads, ffn 1. The FFNs are per token type (4
groups), the sequence is short (21 tokens: field, 12 mons, 4 moves, 4 tera moves), and ~40% of the forward FLOPs
are outside the trunk (move MLP over the 12 × 24 move slots, damage features, aux heads).

**Checkpoints carry their cfg** [M code read, ws/netv2 HEAD and src-24c633f]: `model.save` writes `{format, spec,
gen, cfg, state_dict}`; `model.load` / `net_from_checkpoint` rebuild `Net(gen, **cfg)`. Every consumer loads that
way: spd_h2h (`v2:` nets), fastenv's lookahead2 / deepen / netsearch / fastenv_eval (`porygon.v2.model.load`), the
RL trainer (`--resume` uses the checkpoint cfg: "ignored with --resume"; `--kl-anchor` and `--frozen` via `load`),
and `stacked.OppStack` groups nets by a cfg signature (a bigger pool net gets its own stack). So the collector,
lookahead2 and the h2h take a bigger net unchanged. Exercised for real only by E1 on the VM (spd_h2h + src-24c633f
load the big net); the RL trainer path is untested with a non-production cfg.

**Measured (A16, idle GPU in data's host job 1172609 on parrot, `srun --overlap`, 2026-11-17 22:01–22:20Z; torch
2.14.1+cu126; src-p2b = BC-P4's code; one real P4 shard; `scripts/netv2_bignet/microbench.py`; raw rows
`$C/bignet/mb/a16.jsonl`, `a16-bs2048.jsonl`, `a16-bs4096.jsonl`; copies in `docs/netv2/bignet-mb/`)** [M]. One BC training step exactly as pretrain.py streams it (slice →
batch_to_tensors → losses → backward → clip → AdamW, with its `.item()` syncs), batch 512; FLOPs = torch
FlopCounterMode forward (matmul/bmm/sdpa only), per decision:

| cfg | params | fwd MFLOP/dec (aux / no aux) | train samples/s tf32 · fp32 · fp16 | step time vs base tf32 · fp32 | peak GiB (bs 512) | fwd bs64 / bs512 dec/s (tf32) |
|---|---:|---:|---:|---:|---:|---:|
| **base** (= BC-P4) | 5,809,977 | 123.8 / 93.0 | 2,526 · 2,076 · 3,024 | 1.00 · 1.00 | 1.67 | 2,470 / 7,716 |
| L4F2 | 7,911,225 | 145.9 / 115.0 | 2,450 · 1,941 · 2,971 | 1.03 · 1.07 | 1.78 | 2,451 / 7,467 |
| L6F1 | 7,390,665 | 157.9 / 127.0 | 2,011 · 1,643 · 2,506 | 1.26 · 1.26 | 2.00 | 2,263 / 6,216 |
| L8F1 | 8,971,353 | 192.0 / 161.1 | 1,670 · 1,365 · 2,119 | 1.51 · 1.52 | 2.32 | 2,106 / 5,217 |
| L6F2 (code default) | 10,542,537 | 191.0 / 160.1 | 1,936 · 1,529 · 2,431 | 1.30 · 1.36 | 2.17 | 2,252 / 5,970 |
| **D384L4** | **11,606,329** | 253.4 / 191.6 | 1,970 · 1,418 · 2,448 | **1.28 · 1.46** | 2.23 | 2,327 / 6,522 |
| D384L6 | 15,156,745 | 329.2 / 267.4 | 1,540 · 1,107 · 1,992 | 1.64 · 1.88 | 2.72 | 2,098 / 5,150 |
| D384L8 | 18,707,161 | 405.0 / 343.2 | 1,277 · 912 · 1,682 | 1.98 · 2.28 | 3.22 | 2,126 / 4,358 |
| D512L4 | 19,499,833 | 431.5 / 328.2 | 1,648 · 1,029 · 2,045 | 1.53 · 2.02 | 2.85 | 2,485 / 5,836 |
| D512L6 | 25,806,409 | 565.5 / 462.3 | 1,282 · 799 · 1,630 | 1.97 · 2.60 | 3.53 | 2,177 / 4,495 |

(tf32 = BC-P4's setting, fp32 + TF32 matmuls; fp32 = no TF32, the path a T4 runs since it has no TF32 units; fp16 =
autocast + GradScaler, not what the BC code does.) Calibration: base tf32 2,526 samples/s step-only vs BC-P3's
end-to-end 2,401 median on an A16 (same code, same streaming loader, GPU 87–100% busy) [M] ⇒ loader + validation
overhead ≈ 5%: the microbench reproduces a real run.

Reading the table: at 21 tokens the step is partly launch/latency-bound, so **depth costs, width is cheap** (L4F2:
+36% params for +3–7% time; D384L4: 2.0× params and 2.05× FLOPs for 1.28× tf32 / 1.46× fp32 time). Memory is no
constraint for BC (≤ 3.5 GiB at batch 512 on a 16 GB T4).

**Choice: one size, arm A = D384L4** (`{"d_model":384,"n_layers":4,"ffn_mult":1}`, n_heads 8 ⇒ head dim 48; everything
else default): 11,606,329 params = 2.0× BC-P4 and 1.37× Jaxcalibur's 8.5M; same depth and token layout as BC-P4, so
the one change is width. It is big enough that a capacity effect should show, and cheap enough to be usable for RL
(below). The second slot of the budget goes to a **same-size seed control S** (BC-P4's size, seed 3, trained on the
same T4 with the same code), run only if arm A wins, because the BC seed-to-seed spread in h2h has never been
measured (RL's is < ~2 points, G9q2) and BC-P4 was trained on another GPU and library build. D512L4 (19.5M) is the
natural next point of a width curve; it is a follow-up VM session, not part of this pre-registration.

**Inference cost for later RL on the cluster (is a win usable?)** G9q on the A30 now [M `runs/r1/20261117-113248-1174222/
G9q/train.log` 21:54–21:55Z]: +43.1–43.6k steps per update, sps 2,069–2,153, collect_sps 4,359–4,534, upd_s
10.4–10.9 ⇒ a cycle ≈ 9.7 s collecting + 10.7 s updating (not overlapped). Bigger-net factors [M A16]: actor-style
forward bs64 is flat across sizes (2,470 vs 2,327 dec/s: launch-bound), so collection should not slow; the learner
step at batch 2048 (closest to PPO's 4096 minibatch that fits the A16) is 1.30× (D384L4) and 1.64× (D512L4) base
(2,720 / 2,091 / 1,654 samples/s; `a16-bs2048.jsonl`). Estimate [E: A16 ratios carried to the A30; collection
unchanged]: D384L4 RL ≈ 43.3k / (9.7 + 1.30 × 10.7) ≈ **1,830 sps (−13%)**, D512L4 ≈ 1,560 sps (−26%). Learner memory
at batch 4096 [M base 12.49 GiB at bs 4096 on the A16; D384L4 OOMed there (14.6 GiB card)]; [E ×2 of the bs-2048
peaks 6.31 / 8.30 / 10.27 GiB]: D384L4 ≈ 16.6 GiB, D512L4 ≈ 20.5 GiB ⇒ D384L4 fits the A30 (24 GB) and the A40;
D512L4 would be tight on the A30 next to the collector's nets. lookahead2's batched leaf forward (bs512): 0.85×
base throughput for D384L4 [M A16] ⇒ ≈ +18% per-decision net time [E].

## 2. Compute
**BC-P4 measured** [M `$C/runs/p4/STATUS`, `runs/p4/bc/pretrain-run.log`, `runs/pretrain/g9-p4-s2/{pretrain,mem}.log`]:
A40 gpuvm33 (step of 1173274, nproc 4), 18:01:31 → 21:05:43 on 11-16 = **3.07 h**; 140,880 steps × batch 512 =
72.13M samples = 6 epochs × 23,480 steps over 12,021,692 training decisions; 150 whole-shard streaming parts of 1 GiB
per epoch; median 6,519 samples/s; GPU 66–78% busy, 5.2 GB; peak RSS 16.0 GB (the 767,203 validation rows alone are
767,203 × 13,396 B = 10.3 GB in RAM); 29 validation passes (every 5,000 steps + final). Final val: policy CE 1.13794,
value BCE 0.58486, top-1 0.57867.

**T4 throughput** [E]. The T4 runs the fp32 path (no TF32). Spec ratio T4 / A16 (per GPU): fp32 8.1 / 4.5 TFLOPS =
1.8×, memory bandwidth 320 / 200 GB/s = 1.6×; the launch-bound share of our step gains less, and the 70 W T4 throttles
under sustained load. Planning factor: **T4 = 1.0× the A16's fp32 rate (conservative), point estimate 1.4×.** The
VM pipeline measures it first (stage `mb`) and a budget guard stops training if the projection is too long.
Hours = 72.13M / samples_per_s + 29 × 767,203 / val-forward rate (bs 2048, aux; A16 fp32 5,897 base / 4,048 D384L4):

| run on the T4 | train + validation, 1.0× A16 fp32 | at 1.4× |
|---|---:|---:|
| base (control S) | 9.65 + 1.05 = **10.7 h** | 7.6 h |
| D384L4 (arm A) | 14.1 + 1.5 = **15.7 h** | 11.2 h |
| (D512L4, not in this prereg) | 21.6 h | 15.4 h |

**fp16 / bf16:** not used. fp16 autocast + GradScaler measured 1.46× (base), 1.73× (D384L4), 1.99× (D512L4) faster
than fp32 on the A16 (no inf/NaN in the short runs), so it would likely be a real win on the T4's tensor cores — but
it is not needed for the budget, it is a second change against BC-P4 (fp32 + TF32), and GradScaler is a code change
to pretrain.py. The T4 has no bf16 fast path. Held in reserve if the guard trips.

**VM-hour budget** (VM-on time is what counts):

| item | conservative | basis |
|---|---:|---|
| boot, staging wait, md5 of 7.6 GB, setup | 1.0 h | push 8–30 min [E, §3] |
| T4 microbench (2 cfgs × 2 modes) + 20-pair smoke | 0.25 h | A16 run took ≈ 2 min per cfg |
| arm A training | 15.7 h | table |
| E1A 4,000 pairs + E2A 2,000 pairs (12,000 games) | 0.5 h | ≈ 25.7k games/h on the VM [M cal-vm1, memory gpudojo-vm] |
| *only if E1A = BIGGER BETTER:* control S training | 10.7 h | table |
| E4 4,000 pairs + E0 2,000 pairs | 0.5 h | as above |
| **total** | **17.5 h** (no win) / **28.7 h** (win) | |
| with a 25% margin | 22 h / 36 h | ≤ 60 h ✓ |

Hard caps (pre-registered): the budget guard HOLDs (shuts down, keeps state) if the projected training time at step
5,200 exceeds 24 h (arm A) or 16 h (S); the data wait is ≤ 4 h; a 45-min stall ⇒ FAILED. Worst case with every cap
hit: 1.25 + 24 + 0.5 + 16 + 0.5 ≈ 42 h + the data wait.

## 3. Data transfer
**What the BC reads** [M `ls -l`, `$D/.../MD5SUMS`]: `v2pre-human-shuf/` = 501 training `shard-NNNNN.npz` + 32
`shard-val-NNNN.npz` = 533 files, 7,615,852,955 bytes (max 14.97 MB each; deflate-compressed npz, ≈ 595 B per
decision), `meta.json` 20,211,838 B (read by the whole-shard loader's layout check), `heldout-games-0.06.txt`
151,798 B, `verify.json`; not needed: `rowmap.npz` (49 MB). Plus BC-P4 (24,145,233 B, md5 693c881e) and src-p2b (833
KB). Total ≈ **7.66 GB**. Generic recompression would gain little (already deflated).

**Measured routes (11-17 ~21:50–22:00Z)** [M]:
- cluster → laptop: 298.6 MB of shards in 73.5 s (4.06 MB/s, uncached) and 66.0 s (4.52 MB/s, source cached);
- **laptop → cluster upload: 80 MB in 417 s = 0.19 MB/s** (the laptop's uplink; the earlier VM setup copy did
  ≈ 0.54 MB/s, 325 MB / 10 min);
- cluster NFS sequential read: 298.6 MB in 18.4 s = 16.2 MB/s;
- the gpucluster2 login node reaches the internet on TCP 22 (github.com:22 open); the VM's name resolves
  (20.215.252.178) but its port 22 could not be tested (VM off).
A laptop pipe is bound by the uplink: 7.66 GB / 0.19–0.54 MB/s ≈ **4–11 h of VM time** — rejected.

**Route: cluster → VM directly** (`push_data.sh`, rsync over ssh from the login node, resumable `--partial`, 6
tries). Key: `~/.ssh/porygon_vm_push` on the cluster (ed25519, created by `stage_vm.sh` if missing), authorised in
the VM's `authorized_keys` with `restrict` (no tty, no forwarding). Expected [E]: NFS-bound 16.2 MB/s ⇒ ≈ 8 min;
≤ 30 min if the Imperial → Azure (Poland Central) path is as slow as the cluster → laptop path. Nothing passes
through or is stored on the laptop (it only sends ~40 KB of scripts and the start commands). Needs main's OK:
it gives the cluster account a key into the VM.

**Where it lives:** `~/bignet/` on `/` (persistent; ~48 GB free − 7.66 GB data − ≤ 1 GB checkpoints/state).
Not `/mnt` (wiped on deallocation; a stop/start mid-run would lose the data and force a second copy). Read rate
needed by training ≈ 7.6 GB per epoch over ≥ 2 h ≈ 1 MB/s: the OS disk is ample, and the 27 GB RAM leaves ~10 GB of
page cache next to the trainer's 16 GB. Staging order: the VM runs only while copying + training: `stage_vm.sh`
starts the push and the pipeline together; the data stage waits for `PUSH_DONE`, checks all md5s, then trains.

## 4. Controls and the test
- **Same as BC-P4:** data (the same 533 shards, md5-checked), split (`is_val` crc32 rule, val-frac 0.06: the same
  767,203 val decisions), recipe and command (`--epochs 6 --seed 2 --value-sample one-per-game --val-frac 0.06
  --no-bf16 --stream-gb 1.0 --stream-parts shards --eval-every 5000`, batch 512, lr 3e-4 cosine to 0.1×, warm-up
  1,000, wd 0.01, value 0.5, aux 0.1, clip 1.0), number of steps (140,880 = 6 epochs), and the code
  (`src-p2b` = ws/netv2 745c479; `pretrain_resumable.py` imports it and adds only checkpoint/resume). ONE change:
  `--model-json`. Seed policy as BC-P4 (seed 2): the loader's generator `[2, 7919]` and the drop-seed generator do not
  depend on the net, so arm A sees **the same minibatch sequence and value rows** as BC-P4; only the init differs.
- **Differences we cannot remove:** T4 fp32 vs A40 TF32 numerics; torch 2.15+cu132 vs 2.14.1+cu126; GPU
  non-determinism. The control S (BC-P4's size, seed 3, same VM/code) measures them together with seed noise.
- **Matched cost (informational):** `keep-<s>.pt` of arm A at s = ⌊140,880 × t_base / t_A / 1000⌋ × 1000 with
  t from the VM's own T4 microbench (e.g. ≈ 96k–110k steps if the A16 ratio 1.28–1.46 holds) = the arm-A net after
  as much T4 step time as BC-P4's whole run would take there. It is mid-cosine (lr not annealed), so it is biased
  against the big net; E2A only.
- **Opponent:** BC-P4 itself (`runs/pretrain/P4-g9-s2.pt`, md5 693c881e, copied). In-process h2h on the VM (wsp
  spd_h2h, `--v2-src src-24c633f`, 3 workers, GPU nets, validated vs the cluster: cal-vm1 52.4 vs 51.8), both raw at
  T 0.5, duplicate pairs, gen9randombattle, **new seed 4416017** for every reading (same team pairs ⇒ common random
  numbers across readings), 500-pair shards, pair-mean 95% intervals.
- **Fixed n, not sequential:** h2h is cheap here (≈ 10 min per 2,000 pairs [E]), so 4,000 pairs: half-width
  ≈ 2.1 × √(1000/4000) ≈ ±1.05 points [E from bc-15's ±2.1 at 1,000 pairs], power ≈ 0.97 for a true +2 points at
  one-sided α 0.025.
- **Held-out losses:** policy CE, value BCE, top-1 on the same val rows from the run's own final val line vs BC-P4's
  (1.13794 / 0.58486 / 0.57867); informational, no gate. Same code path, so directly comparable.
- **Thesis rows produced:** results `bc-16` (E1A), `bc-16b` (E2A, info), `bc-17` (E4, if run), `bc-17b` (E0, info),
  the held-out deltas inside bc-16; runs.md BC table rows `g9-p4-D384L4-s2` (and `g9-p4-base-s3`); compute.md GPU
  Dojo row (VM hours from `~/bignet/logs/uptime.log`); methods §3.2 a sentence on sizes; the microbench table above
  as a [M] methods/appendix item.

## 5. The VM pipeline (written, not run)
- `stage_vm.sh` (laptop, once the user has started the VM; ~5 min): copies `run_vm.sh`, `pretrain_resumable.py`,
  `microbench.py`; authorises the push key; starts `push_data.sh` on the login node (nohup setsid); installs and
  starts the systemd unit `porygon-bignet` (`run_vm.sh run` at every boot).
- `run_vm.sh run`: setup check (T4, CUDA, src commit, h2h env, ≥ 24 GB RAM) → data (wait ≤ 4 h for PUSH_DONE, md5 of
  all 536 files, BC-P4 md5) → T4 microbench (base + D384L4, fp32 + fp16 ⇒ matched step) → 20-pair smoke h2h →
  train arm A → E1A, E2A → if E1A = BIGGER BETTER: train S, E4, E0 → `RESULT.txt` → `DONE` → unit disabled →
  `sudo shutdown -h now`. Logs on `/`: `~/bignet/logs/run.log`, `STATUS` (one line per stage), `logs/uptime.log`
  (VM hours per boot), per-run `pretrain.log`, `mem.log`, `stdout.log`, h2h `h2h.log` / shard jsonl / meta.
- **Resume after a stop/start:** every stage has a marker; the unit restarts the pipeline at boot. Training writes
  `state.pt` (net, AdamW, step, best, RNG states, elapsed; previous kept as `state.prev.pt`) every 1,000 steps
  (≈ 5–8 min of T4 time [E]) and resumes **exactly** (the loader's epoch plans are replayed without decompressing,
  the unfinished part reassembled and its done minibatches skipped). Verified on CPU (`check_resume.sh`, data host
  job, tiny net, real P4 shards incl. a val shard): pretrain.py vs the wrapper vs the wrapper stopped and resumed
  give bit-identical final weights: **PASS** 2026-11-17 ~23:20Z (282 steps, 6 epochs, stopped at 200 inside epoch
  5 and resumed; final.pt md5 65988813 for all three; val lines identical; `$C/bignet/mb/resume-test/RESULT.json`)
  [M]. The test has one whole-shard part per epoch, so the across-part skip is exercised only at epoch boundaries. The h2h resumes by whole 500-pair shards.
- **Failure:** FAILED (stage + reason) ⇒ unit disabled ⇒ VM shut down; HOLD (budget guard) the same but resumable;
  with either file present `run` does nothing, so a restart by the user for inspection cannot burn hours by
  itself. Kill switch `touch ~/bignet/NO_AUTORUN`.
- `mock_vm.sh`: a laptop mock of the control flow (stub python/GPU/sudo, fake data): confirmed / no-difference
  branches, kill + resume, budget HOLD, md5 failure: **7/7 PASS** 2026-11-17 ~23:05Z [M].

## 6. Not measured / risks
- T4 BC throughput (planning factor 1.0×–1.4× of A16 fp32); the VM's 4 vCPUs as loader + main thread (the A40 run
  had 4 CPUs and still reached 6.5k samples/s, ≥ 2× what the T4 needs) — the guard covers both.
- Cluster → VM bandwidth and whether the VM's port 22 accepts the cluster's address (no inbound filter seen
  before: the laptop connects).
- The VM's `~/porygon` h2h layout (node, fastenv js, PS paths) is inferred from memory gpudojo-vm and checked by
  the setup stage and the smoke.
- RL cost figures are A16 ratios carried to the A30 [E]; no RL run with a non-production cfg yet.
- BC improvement may not survive RL (BC-P4 ≈ BC-P3 in BC, but G9-P4 > G9q in RL): a NO DIFFERENCE here does not
  prove RL would not benefit; it means BC does not show it.
