# BIGNET — PRE-REGISTRATION (DRAFT: binding once main signs it off; then copy read-only to the VM and to `$C/bignet/PREREG.md`)

netv2, 2026-11-17, written before any bigger-net training step and before any game. Design and every number's basis:
`docs/netv2/BIGNET-DESIGN.md`. Pipeline: `scripts/netv2_bignet/{stage_vm.sh, push_data.sh, run_vm.sh,
pretrain_resumable.py, microbench.py}` on the user's GPU Dojo VM (Azure, 1× Tesla T4 16 GB, 4 vCPU, 27 GB, persistent
`/`). One VM session started by the user; no agent needed until RESULT.txt (or FAILED / HOLD).

**Question.** Does a 2× wider v2 network, BC-trained exactly as BC-P4 on BC-P4's data, beat BC-P4 head to head?

1. **Arm A** = `{"d_model":384,"n_layers":4,"ffn_mult":1}` (11,606,329 trainable params; BC-P4 = 5,809,977), all other
   Net cfg keys default. Trained by `pretrain_resumable.py` (imports `porygon.v2.pretrain` of `src-p2b` = ws/netv2
   745c479, BC-P4's code; adds only state save / exact resume / keep-step copies; bit-identical to pretrain.py on CPU
   in `check_resume.sh`) with BC-P4's command and ONE change, `--model-json`: `--gen 9 --epochs 6 --seed 2
   --value-sample one-per-game --val-frac 0.06 --no-bf16 --stream-gb 1.0 --stream-parts shards --eval-every 5000`
   (batch 512, lr 3e-4, warm-up 1,000, cosine to 0.1×, wd 0.01, value 0.5, aux 0.1, clip 1.0, select total) on the
   533 shards of `p4-gen9-20261116-1502/v2pre-human-shuf` (md5 of every file = `$D/.../MD5SUMS`). 140,880 steps.
   The net of record is `final.pt` (as for BC-P4). fp32 (the T4 has no TF32; BC-P4 ran fp32 + TF32 on an A40).
2. **Matched-cost copy:** `keep-<s>.pt` at s = ⌊140,880 × t_base / t_A / 1000⌋ × 1000, t = median train-step time of
   base and arm A in the VM's own T4 microbench (stage `mb`, before training).
3. **E1A (decision of record):** arm A `final.pt` vs BC-P4 (`P4-g9-s2.pt`, md5 693c881e), wsp spd_h2h on the VM
   (`--v2-src ~/porygon/src-24c633f`, `--workers 3`, GPU), both `temperature=0.5`, gen9randombattle, duplicate pairs
   0–3,999 (8 shards of 500), **seed 4416017**; estimate = mean pair score of arm A; 95% interval = ±1.96 × sd / √n
   over pairs (spd_h2h merge `ci95`). Void games score 0.5 (as always).
   - **BIGGER BETTER** iff lower 95% > 50; **BIGGER WORSE** iff upper 95% < 50; else **NO DIFFERENCE SHOWN**.
4. **Only if E1A = BIGGER BETTER — control S and E4 (confirmation):** S = BC-P4's size (`{"n_layers":4,"ffn_mult":1}`),
   the same command with `--seed 3`, trained on the same VM. **E4:** arm A final vs S final, pairs 0–3,999, seed
   4416017, as E1A. **CONFIRMED** iff E4's lower 95% > 50, else **NOT CONFIRMED**.
5. **Informational only (no decision):** E2A = keep-<s> vs BC-P4, pairs 0–1,999; E0 = S vs BC-P4, pairs 0–1,999
   (seed + hardware noise); held-out policy CE / value BCE / top-1 of each run's final val line vs BC-P4's
   1.13794 / 0.58486 / 0.57867 (same 767,203 val decisions); T4 throughput of both sizes.
6. **OVERALL** (RESULT.txt): CONFIRMED / NOT CONFIRMED / NO DIFFERENCE SHOWN / BIGGER WORSE.
   - **CONFIRMED** ⇒ netv2 writes a proposal + PREREG (main signs off): RL from arm A on the cluster = G9-P4's recipe
     with arm A in BC-P4's three roles (init, KL anchor, frozen net), read step-matched vs G9-P4 at 25M / 50M with the
     sps cost reported (estimate −13%); a follow-up VM session may add D512L4 (19.5M) as a width-curve point.
   - **NOT CONFIRMED** ⇒ the gain is within seed / hardware noise: no RL arm from it; report.
   - **NO DIFFERENCE SHOWN / BIGGER WORSE** ⇒ stop the bigger-net BC arm: at BC on P4, width is not the binding
     constraint (raw-net route stays RL steps + expert iteration); report, with the held-out deltas.
7. **Stops (pre-registered, none lowers a gate):** budget guard at the first train line ≥ step 5,200: projected
   training time (140,880 × 512 / samples_per_s × 1.07) > 24 h (arm A) or > 16 h (S) ⇒ HOLD: state kept, VM shut
   down, main decides (continue unchanged = `rm HOLD; touch GUARD_OK_<run>`; any change to the recipe, e.g. fp16, is a
   new version of this document before resuming). Data not pushed within 4 h, md5 mismatch, a 45-min training stall,
   a failed smoke or h2h ⇒ FAILED, VM shut down. A VM stop/start resumes by itself (systemd unit; exact training
   resume from `state.pt`; h2h by whole shards); no stage re-runs once its marker exists.
8. **Budget:** ≈ 17.5 VM-h (no win) / 28.7 VM-h (win) at the conservative T4 = 1.0× A16-fp32 estimate; ≤ 42 h with
   every cap hit; within the ≈ 60 h allotted of the ≈ 99 free GPU-h.
9. **Staging (needs main's OK):** data, BC-P4 and src-p2b go cluster → VM directly (rsync from the gpucluster2 login
   node with a new cluster-side key `~/.ssh/porygon_vm_push`, authorised on the VM with `restrict`), because the
   laptop's uplink measured 0.19 MB/s (≈ 11 h of VM time for 7.66 GB). The VM stores everything on `/`.
10. **Thesis rows:** results bc-16 (E1A + held-out deltas), bc-16b (E2A), bc-17 (E4), bc-17b (E0); runs.md BC rows
    `g9-p4-D384L4-s2` (+ `g9-p4-base-s3`); compute.md GPU Dojo VM row from `~/bignet/logs/uptime.log`; methods §3.2.
11. **Pipeline files of record (md5, ws/netv2):** run_vm.sh d457f443, pretrain_resumable.py 3abfc7f1, microbench.py
    a17cad2b, push_data.sh 17ef6eeb, stage_vm.sh 82ba234e; tests check_resume.sh (PASS on the cluster CPU, bit-identical
    weights through a stop/resume) and mock_vm.sh (7/7). Any change after sign-off = a new version of this file.
