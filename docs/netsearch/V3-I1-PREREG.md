# V3 I1: inference bench, pre-registration (fastenv, 2026-11-12 ~12:20 UTC, before any I1 code or measurement)
**Main's decisions (2026-11-12 ~12:05):**
1. I1 GO: a ≈ 1 h inference bench in a helper job, with no Rust. The pass rule is written first: "10k leaves/decision
   reachable on the slot's T4 under contention", naming the option that reaches it.
2. fp16 is allowed for LEAF inference only, gated on parity (mean |ΔP| ≤ 0.002, top-1 flips ≤ a stated rate).
   Training stays fp32.
3. R2 is ≥ 10k leaves/decision AND depth ≥ 3 under slot contention.
4. The Rust build starts only if I1 PASSES. Otherwise report and stop. The re-freeze keeps priority.

Context: `V3-RUST-DESIGN.md` §1c. Gates 1127567 measured T4 eager fp32 at 14.5k leaves/s at B 1024 (no contention).

## Question
Can one inference process on a dove T4 sustain the leaf rate that R2's 10,000 leaves per decision needs? The
conditions are those of the slot: 3 concurrent battles share the GPU, and the node's CPUs are busy with H's engine
workers. If so, which configuration does it?

## What is measured (one helper job; nothing a running job imports)
**The net:** G9p-cd@104.71M (sha256 35cd1007…ad1c), netv2's frozen tree fba02fb, in inference mode.
- The forward computes the outputs V3 needs: win_logit, the policy logits and the `opp_action` aux head (`aux=True`;
  the other aux heads are computed too, which is conservative).

**Leaf observations, ≥ 5,400 of them (target 6,000):**
- the 30 roots of P2 (`$C/runs/v3gates-20261112-014733-1127567/p2/p2-roots.pkl`, seed 7312) × 4 determinizations
  × 50 random walks;
- each walk is d plies, d uniform in {1, 2, 3}, with seed **7313**;
- every ply takes a uniformly random legal option for each side (`v3leaf.side_options`), and an instruction branch
  sampled by its percentage;
- the walk's end state is encoded by `v3leaf.LeafEncoder.encode`, the P1-gated encoder;
- terminal states and engine errors are skipped.

**Configurations:**
- **E32:** eager fp32, the model as P2 ran it (move and damage trim on). This is the reference.
- **E16:** E32 under `torch.autocast(cuda, float16)`. The damage layer stays fp32, as the model forces.
- **G32:** fp32 with move and damage trim off (static shapes; cfg `trim_moves=False`, `DamageFeatures(9, trim=False)`),
  the whole forward captured in one CUDA graph per batch size and replayed.
- **G16:** G32 with fp16 autocast inside the capture (`cache_enabled=False`).

**Throughput.** Per configuration and B ∈ {128, 192, 256, 384, 512, 1024}, one Python process on the T4 runs a server
loop:
1. copy a batch of leaf observations from pinned host buffers into the device inputs (int64 / float32, as
   `obs_to_tensors` makes them);
2. run the forward (or replay the graph);
3. copy win_logit, the policy logits and opp_action back to pinned host memory;
4. synchronise.

There is one stream and no overlap, which is conservative. G = leaves/s over an 8 s window, after a 2 s warm-up.

**Two conditions:**
- **(u) uncontended:** information only.
- **(c) contended, the gate condition.** Job `-c 16`; 11 processes loop `pe.monte_carlo_tree_search(state, 1000 ms)`
  on the roots' engine states (H's 11 engine workers). 3 more processes spin a busy loop, standing in for Showdown, the
  clients and the 12 Rust tree threads (≈ 25 µs/leaf at 30k leaves/s ≈ 0.75 core [e]). That leaves 15 of 16 CPUs busy.

**Parity** (each configuration vs E32 on every leaf, at the configuration's own B = 256; a partial last batch is
padded by repetition and the padding dropped):
- **G32 (fp32):**
  - max |ΔP(win)| ≤ 10⁻⁴;
  - max |Δ| over legal policy logits ≤ 10⁻³;
  - max |Δ| over legal opp_action logits ≤ 10⁻³.
- **E16, G16 (fp16):**
  - mean |ΔP(win)| ≤ 0.002 and q99 |ΔP(win)| ≤ 0.01;
  - **policy top-1 flips ≤ 0.5%** of leaves with ≥ 2 legal actions;
  - **opp_action top-1 flips ≤ 0.5%** of leaves with ≥ 2 legal classes;
  - no NaN or inf.
- E32 is the reference and passes by definition.
- **The same fp16 rule becomes R1's E2 rule** for the deployed configuration, on 100k leaves.
- Also reported: |ΔP| quantiles, the policy total variation (q50 / q99), and P(win) at B 256 vs B 1024 for E32 (a
  batch-invariance check, no rule).

## Rule
**The per-decision target:**
- R2 needs 10,000 leaves per decision with 3 battles sharing the GPU over a 1 s window, at an assumed pipeline
  efficiency η = 0.85 (V3-RUST-DESIGN §1c).
- So the T4 must sustain G ≥ 3 × 10,000 / 0.85 = **35,300 leaves/s**.

**B is capped at 384:**
- R2's largest in-flight setting is K = 64 per tree, so 12 trees hold ≤ 768 leaves in flight.
- That keeps double-buffered batches of B ≤ 384.
- B 512 and 1024 are information only.

**I1 PASSES iff** some configuration c and some B ≤ 384 have both:
- c's parity passes;
- G_c(B) ≥ **35,300 leaves/s** under condition (c).

**The named option:**
- prefer an fp32 configuration that passes;
- otherwise the passing fp16 configuration with the highest G;
- report its smallest passing B and its projected 0.85 × G/3 leaves per decision.

**I1 FAILS otherwise.** Report the best parity-passing G, and stop: no Rust build (main's (4)).
- ONNX Runtime and TensorRT are not installed in the cluster venv (`pip list`: torch 2.14.1+cu126, triton 3.8.0, no
  onnxruntime / tensorrt).
- Trying them would be a new decision.

**Void:**
- the job dies;
- E32 fails to run;
- fewer than 5,400 leaves are generated.

A void run is re-run once with the same seeds after the fix. A non-reference configuration that errors (e.g. its
graph capture fails) fails that configuration; that does not void the run.

**Disclosures:**
- η = 0.85 is an assumption that R2 tests for real.
- The host-side batch assembly (Rust's job) is not timed here.
- I1 says nothing about strength.

## Output
- `$C/runs/v3i1-<date>-<job>/`: manifest.txt, i1.json (summary + rule + verdict), i1-rows.jsonl, i1.log.
- The manifest holds: code commit, net sha256, node, GPU (`nvidia-smi -q` clocks), CPUs, and the contention PIDs
  (killed by recorded PID at the end).
- Report to main in ≤ 4 lines.
