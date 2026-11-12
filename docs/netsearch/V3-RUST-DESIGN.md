# V3 Rust port: design (fastenv, 2026-11-12 ~11:50 UTC; DESIGN ONLY, nothing built or submitted)
**User decision (relayed by main, 2026-11-12 ~11:30):** V3 goes to a Rust port. The user chose it over speeding up
inference first, building the full Python V3, or pausing. This file covers scope, parity, gates, effort and order.
Tags: [m] measured, [e] estimated.

## 0. What the Python gates measured (job 1127567, `$C/runs/v3gates-20261112-014733-1127567/`)
- **P1 PASS:** A agree 1.000 (71,824 elements), mean |ΔP| 1.9·10⁻⁴; B mean |ΔP| 0.021 (revealed 0.043, not revealed 0.003).
- **P2 PASS:** median 2,048 leaves per decision at B 64 (q10 1,920, q90 2,112); 90/90 decisions, 0 errors.
- **Wall-time split per leaf at B 64 [m]:**

  | engine | encode | tree | collate | forward |
  |---|---|---|---|---|
  | 369 µs | 440 µs | 385 µs | 136 µs | 980 µs |
  | 18% | 21% | 19% | 7% | 47% |

  - Mean depth 2.2.
  - Python's CPU cost is 1,330 µs per leaf, outside the forward pass.
- **The T4 on its own [m]** (eager fp32, aux heads off), in leaves per second:

  | | B 64 | B 256 | B 1024 |
  |---|---|---|---|
  | 1 process | 3.1k | 9.4k | 14.5k |
  | 12 processes | 7.9k | 12.3k | 13.6k |

  - At B 64 it is launch-bound: about 1,000 small kernels per forward, about 650 of them in the damage layer.
- **The v2 net [m, laptop, synthetic observations]:**
  - 10.5M parameters;
  - 158 MFLOP of matmul per leaf (168 with damage untrimmed);
  - 12.3 KB of observation per leaf.
  - At 14.5k leaves/s the T4 runs at 2.3 TFLOP/s, 28% of its 8.1 fp32 peak.

## 1. Scope: what moves to Rust
**A new crate, `porygon-v3`**: a pyo3 extension built with maturin. It links poke-engine 0.0.48 as a Rust library
(path dependency, `features = ["gen9"]`), with exactly the patch set and features that built gloria's `.engine-gen9`
(.so 7965d8b3; the patch md5s go in the manifest).
- **Not a patch inside poke-engine:** the existing patches stay as they are, and nothing that runs today changes.
- **poke-engine is MIT,** so linking it is free (Foul Play's GPL code is not touched).
- **Rust-level calls** (all `pub` in 0.0.48): `generate_instructions_from_move_pair`, `apply_instructions` /
  `reverse_instructions`, `get_all_options`, `State::deserialize`, and `genx::evaluate::evaluate`.
  - The missing `evaluate` in the gen9 *Python binding* does not affect the crate.

**(a) Tree: moves to Rust.**
- Decoupled pUCT at the simultaneous-move nodes:
  - our side: the net's policy as prior;
  - the opponent: the `opp_action` aux head at the root, and its policy below the root. The joint node keeps both
    sides' statistics.
- Chance outcomes are sampled by the instruction percentages.
- Arena (index) nodes, no `HashMap<Box>`. That is SPEEDUPS row 2's fix, which removes the teardown cost.
- Virtual loss, K leaves in flight per tree (a parameter, see §3).
- One OS thread per tree: 3 battles × 4 determinizations = 12 threads per V3 process. The GIL is released throughout.
- Wall budget 1000 ms on a Rust clock. The result is root visits plus Q per (our option, their option), the same shape
  as `monte_carlo_tree_search_joint`, so the netsearch decision code reads it unchanged.

**(b) Leaf encoder: port `v3leaf.LeafEncoder.encode`** (365 lines, the P1-gated reference) and the option lists
(`side_options`, `needs_switch`, `terminal`, `action_mask`).
- **Precomputed in Python, at the root only, not ported:**
  - the set posterior;
  - species / items / abilities / move ids and candidates, stats, levels;
  - the whole root observation, which Rust copies and patches as v3leaf does.
  - P1-B accepted stale knowledge fields at the leaves (0.021).
- **Fresh opponent slots:** per determinization, Python precomputes a slot template (`fresh_opp`) for each unseen
  opponent mon. The posterior with nothing revealed comes out of the existing code.
- **Static tables** go to Rust as one JSON written by Python at init, never hand-copied:
  - volatile → effect column, the status / weather / terrain maps;
  - screen lengths, the hazard and boost attributes, charging moves;
  - `is_status_move` per move slot, the root observation's column indices (spec 2.1, checked against
    `SPEC_VERSION`).
- **The damage features are part of the net** (`porygon.v2.damage`, a torch module on the GPU). Nothing to port.
- Estimate: 5–15 µs per leaf (memcpy of 12.3 KB plus about 300 writes) [e], against 440 µs in Python.

**(c) Inference.**

| option | CPU µs/leaf | T4 leaves/s (all 3 battles) | leaves/decision |
|---|---|---|---|
| Python V3 today (per-worker CUDA) | 1,330 + inline forward | — | 2,048 [m] |
| Python V3 + one shared server | 1,330 | 14.5k | ≤ 3.0k (CPU-bound) |
| R-cpu: Rust + ONNX/ort (or tract) on CPU | ≈ 4,000 (158 MFLOP at ≈ 40 GFLOP/s per dove core) | — | ≈ 1.0k [e]: worse |
| R-pt32: Rust tree + encode; PyTorch eager fp32 shared server in the same process | ≈ 25 | 14.5k [m] | ≈ 4.1k |
| R-pt32g: + CUDA graphs over fixed batch buckets, damage trim off and compiled | ≈ 25 | 18–22k [e] | ≈ 5–6k |
| R-pt16: + fp16 autocast trunk (damage stays fp32, as the model already does) | ≈ 25 | 35–50k [e] | ≈ 10–14k |
| R-trt: ONNX → TensorRT fp16 (ort TRT EP, or a Python TRT engine) | ≈ 25 | 45–65k [e] | ≈ 13–18k |

How the last column is derived:
- leaves/decision ≈ η · min(4·10⁶/c_cpu, G/3), with η ≈ 0.85 for pipeline bubbles;
- c_cpu is the CPU µs per leaf of one tree thread;
- G is the whole T4's rate, shared by 3 concurrent battles over a 1 s window.

**Reading.**
1. In Python, the CPU caps V3 at ≈ 3k leaves per decision whatever the GPU does. The Rust tree and encoder lift that
   cap by ≈ 50× [e], so from then on the T4 binds.
2. Every gain beyond ≈ 4k then comes from the inference path. CPU inference is a regression.
   - **The most:** TensorRT fp16 (R-trt).
   - **Most of it at a fraction of the risk and effort:** PyTorch fp16 with CUDA graphs (R-pt16). It is the same net
     code, needs no export, and there is no op-coverage risk in the ~650-op damage layer (`bucketize`, gathers,
     data-dependent trim).
3. **Proposal:**
   - Build R-pt16: Rust tree + encode in a pyo3 module, with one Python inference thread per V3 process serving all 12
     trees. It hands over zero-copy pinned numpy batches and returns P(win), policy logits and opponent logits.
   - Keep the batch interface runtime-neutral, so R-trt can be swapped in if I1 below shows ≥ 1.3× over R-pt16.
4. **Every [e] row is checked by I1 before any Rust is written** (§5, step 0).

## 2. Parity plan (R1: gate)
**E0, engine identity.** On 10,000 (state, joint move) pairs from P1's roots (`p1/roots.pkl`) and from random tree
paths of depth 1–4, our crate's `generate_instructions_from_move_pair` must equal `.engine-gen9`'s:
- the same instruction lists, in order;
- percentages within |Δ| ≤ 10⁻⁶;
- `State::serialize` round-trips.
- **Rule:** 100% agreement.

**E1, encoder.** Rust `encode(state)` vs Python `LeafEncoder.encode(state)` on the same states:
- P1's 300 pairs × 2 determinizations, test A and test B;
- plus 100,000 tree leaves (depth 1–4) from the P2 roots (seed 7312).
- **Element rule** over every observation field, not only D: ids, one-hots and flags exact; floats |Δ| ≤ 10⁻⁵.
- **Rule:** ≥ 99.99% of elements, and P1's A and B statistics re-computed through Rust identical to 3 decimals
  (A agree 1.000, B |ΔP| 0.021).
- Option lists and action masks: exact on 100%.

**E2, inference** (each configuration that is used, vs PyTorch eager fp32 on the same 100,000 leaf observations):
- fp32 paths (graphs, compiled damage, export round-trip if R-trt):
  - max |ΔP(win)| ≤ 10⁻⁴;
  - policy and opponent logits max |Δ| ≤ 10⁻³.
- fp16 paths:
  - mean |ΔP(win)| ≤ 0.002 and q99 ≤ 0.01;
  - policy argmax agreement ≥ 99.5%;
  - policy total variation q99 ≤ 0.02.
- **The R-trt export round-trip:** ONNX (opset ≥ 17, untrimmed damage, dynamic batch) and the TRT engine on the same
  leaves, under the same rule. The engine is built per GPU class on a dove T4, and its sha256 goes in the manifest.
- E2 is re-run whenever the deployed net changes (re-freeze), ≈ 10 min.

**E3, tree unit tests** (cargo test plus pytest):
- determinism (one thread, fixed seed ⇒ identical visits);
- virtual loss fully undone;
- visits conserved;
- terminals scored without the net (win 1, loss 0, as P2);
- a forced-KO position picks the KO;
- a 1 s budget overruns by ≤ 20 ms at p99.

## 3. Gates before any slot game (pre-registered in full before they run; P1/P2 rules carried over)
- **R1 parity (gate):** E0, E1 and E2 all pass.
- **R2 throughput (gate):**
  - The protocol is P2's (30 roots, seed 7312, 3 battles × 4 determinizations × 1000 ms, a dove T4) at the
    deployed configuration.
  - Plus the slot's real contention: `-c 24`, with 11 pure-search engine workers running alongside, as H's side does in
    the slot.
  - **PASS iff** median leaves/decision ≥ **10,000** AND mean leaf depth ≥ **3.0** (joint plies from the root).
  - In-flight K ∈ {16, 32, 64} per tree: the gate reads the smallest K that passes, which is the least virtual-loss
    distortion.
  - Also reported: the time split, depth quantiles, root visit entropy, and GPU utilisation.
  - Below the gate: report to main and stop. The only allowed fix is the inference row (§1c); no tree changes.
- **R3 (= P3, information):** AUC of the net value vs the engine root value on Stage 1's positions with outcomes. It was
  not run in 1127567.
- **R4 (= P4, smoke):** 20 games V3 vs H on Showdown in a helper job. Crashes, timeouts and fallbacks only; no
  strength read.
  - **Fallback:** on any Rust error the player falls back to H's own search for that decision. Over 2% fallbacks
    voids, as in Stages 2 and 3.
- **Then the slot stages, unchanged from NEXT-PROPOSAL §3:**
  - V3-T: c_puct ∈ {1, 2, 4} vs H;
  - V3-H: 500 pairs, lower 95% > 50%;
  - V3-F vs Foul Play, paired with 3b.
- **Disclosure:** V3 uses a T4 that H does not. H's root forwards stay on CPU, as in Stage 3.

## 4. Effort (agent-days) and risks
| # | component | days | parallel with |
|---|---|---|---|
| I1 | inference bench (no Rust; one helper job ≈ 1 h; existing code): eager / compiled damage + CUDA graphs / fp16 autocast / ort-CUDA, optionally TRT fp16; B ∈ {256, 512, 1024, 2048}; E2-style parity on P2 leaves | 0.5–1 | first; nothing depends on Rust |
| 1 | crate skeleton, vendored deps, build-in-job script, E0 | 0.5–1 | I1 |
| 2 | leaf encoder + option lists + table export, E1 | 1.5–2.5 | 3 (interface: obs layout + batch API fixed in step 1) |
| 3 | tree: pUCT, chance, virtual loss, arena, threads, batch queue, E3 | 2–3 | 2 |
| 4 | inference thread (R-pt16; buckets, pinned buffers), E2 | 1–1.5 (R-trt + 2–3) | 2, 3 |
| 5 | V3 player in the netsearch harness (one V3 process per side, fallback, manifest, logs) | 1–1.5 | after 2–4 |
| 6 | R1–R4 runs, prereg, report | 1 | — |
| | **total R-pt16** | **≈ 7.5–11** (≈ 5–7 elapsed with two level-2 agents on steps 2 and 3) | |

**Risks.**
1. **The T4 is the ceiling.** fp32 caps at ≈ 4.8k leaves/decision, so 10k needs fp16 or TRT.
   - If I1 finds no configuration that passes parity at ≥ 30k leaves/s, R2 is unreachable on a T4. Main decides then,
     BEFORE steps 1–5: a lower target, a bigger GPU (none is free), or a smaller/distilled leaf net (a new net = a new
     question).
   - **DECIDE (main):** fp16 inference for leaves only, allowed iff E2 passes. PLAN's fp32 rule is for PPO.
2. **Leaves are not strength.** Virtual loss with 12 × K leaves in flight broadens the trees. v1's one-turn lookahead
   was null. The value was trained without search. Opponent priors below the root come from its policy, which was
   trained on self-play. Only V3-T/V3-H answer this.
3. **The gen9 binding has no `evaluate`:** this does not apply (§1). It only matters for a later blended-leaf arm.
4. **Set-posterior complexity:** avoided. It stays root-only in Python, and fresh slots are templated. A posterior that
   updates inside the tree is out of scope (no reveals happen in the tree).
5. **Engine-build drift:** the crate must compile the same source and patches as `.engine-gen9`. E0 guards it.
6. **Toolchain on the cluster:**
   - Builds run inside our one helper job, with gloria's rustup used read-only by absolute path and our own
     `CARGO_HOME=$C/.cargo`.
   - crates.io may be unreachable from the nodes ⇒ `cargo vendor` on the laptop into /tmp, rsync it, delete it
     (pyo3 0.29 + deps ≈ tens of MB; laptop scratch < 100 MB).
   - maturin goes into a new venv built from cpython-3.12.16.
   - TensorRT wheels (≈ 1–2 GB) only if R-trt is chosen: download on the cluster if pip works there, never via the
     laptop's disk.
7. **Two poke-engine copies in one process** (`.engine-gen9` + `porygon-v3`): separate statics, and they exchange
   only state strings. Covered by E0.
8. **GIL and latency:** Rust threads never touch Python objects. The inference thread does ≈ 1 ms of Python per batch.

## 5. Order, with the re-freeze (re-freeze first whenever its trigger is real; HANDOFF steps unchanged)
0. **Now (with GO):** I1 in a helper job this afternoon, ≈ 1 h, done before 23:30. It decides the inference row and
   whether R2's 10k is reachable. Report to main; then steps 1–4 (development; helper jobs only for builds and tests).
1. **≈ 23:30–00:30 tonight:** netv2 branches G9q-cd at ≥ 300M. Re-freeze steps 2–5 take my helper slot first:
   - copy plus sha256;
   - netv2's in-process trigger;
   - δ_c in a ≈ 1 h helper job;
   - `rfsmoke20261112`.
   - V3 build/test jobs are held (one helper job per stream).
2. **The re-freeze slot R (≈ 5.5 h)** runs only after netv2's gen4 prior test RELEASEs. That test was released for
   submission at 11:22 and takes ≈ 12–16 h, so it frees ≈ 23:30–03:30. Then main approves and I CLAIM.
3. **V3 R1–R4 ≈ 11-17…11-20 (helper only).** E2 is re-run on the adopted net if the re-freeze adopts G9q-cd.
4. **V3-T → V3-H → V3-F in the slot**, after R, against the H that ships (≈ 19–21 h of slot time).

**DECIDE (main):**
- GO for I1 now;
- fp16-for-leaves subject to E2;
- R2's target (≥ 10k and depth ≥ 3).
