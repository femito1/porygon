# V3 gates P1 and P2: pre-registration (fastenv, 2026-11-12 ~01:45 UTC, before any gate measurement)
**User decision (relayed by the orchestrator 2026-11-12):** V3 is "Python, gates only first", bounded to ≈ 1–2
agent-days.
- Build only what gates P1 and P2 need:
  - P1: leaf-encoder parity with the real encoder;
  - P2: throughput ≥ 2,000 leaf values per decision at the 4 × 1 s budget, with the time split between encoding,
    inference and the engine.
- Pre-register both pass rules before running them. Run in ONE helper job; no slot.
- Then STOP and report to main (≤ 6 lines): pass/fail, the throughput and time split, and whether P2's numbers argue
  for a Rust port.
- No V3 work beyond the gates without a new decision.

Development checks run only on DIFFERENT records (searchlabels part-00001, laptop) from the gate's records (below).
They never compute the gate statistics.

## What is built (ws/fastenv, new code only; nothing a running job imports)
- **`porygon/v3leaf.py`: the leaf encoder.** `LeafRoot(battle, obs)` holds one decision's root observation (the real
  v2 encoder's `encode`, netv2's frozen tree) and the static data for patching it. `LeafEncoder(root, conv)` serves
  one determinization (a `search.convert` state).
  - **`encode(state)`:** a v2 observation for an engine state reached from that root. It copies the root observation
    and overwrites only the dynamic fields, the set **D** below. Everything else stays as at the root: species,
    types, items / abilities / tera candidates, move ids and candidates, stats, levels.
  - **An unrevealed opponent mon that is in (active, or damaged)** takes the next free opponent slot, with its
    species, level and posterior-expected stats (`posterior` with nothing revealed).
  - It is a port of v1's `porygon.lookahead.Leaves.encode` (the patched-root design) to the v2 fields.
- **`scripts/fastenv/v3_gates.py p1 | p2`:** the two measurements below. `p2` also contains a minimal batched
  decoupled-UCT tree:
  - joint (our option, their option) edges, chance outcomes sampled by `generate_instructions` percentages;
  - net P(win) at the leaves, UCB1 at both sides (c = 1.4);
  - virtual loss while a batch is pending; backup of P(win) from our side's view.

  It exists to produce a realistic leaf workload for P2. It is not a search to be played, and no game uses it.

**D (the patched, compared fields).**
- Per mon (12 slots):
  - `hp` (both columns);
  - `boosts`;
  - `mon_num` columns: present, revealed, active, fainted, hp_frac, status_{brn,frz,par,psn,slp,tox},
    status_counter, terastallized, first_turn, protect_counter, must_recharge, preparing, trapped, effect_0–48;
  - `move_num[:, :, pp_frac]` for our mons and for the opponent's revealed moves;
  - `move_num[:, :, available]`;
  - `tera` / `tera_p` of a mon that terastallized.
- `field_num`: weather_*, terrain_*, trickroom, turn, opp_unseen, own_alive, opp_alive, force_switch, trapped,
  can_tera, used_tera, opp_used_tera, wait, n_switches, n_moves. The *_elapsed columns are compared too.
- `side_num` (both sides: screens, screen ages, hazards); `action_mask`.

## P1: leaf-encoder parity (gate)
**Positions.**
- gen9 search-label F2 records `$D/datasets/searchlabels/records/gen9randombattle/part-0000[67].jsonl.gz`, source
  6771931, i.e. Stage 1's pool. Records shuffled with **seed 7311**.
- Per record, up to 3 consecutive decision pairs (t, t + 1) of the record's player, chosen at random (seeded). Team
  preview is excluded, and both decisions need at least one legal action.
- The first **300 pairs**. Each pair gets **2 determinizations** (`search.convert` with rng seeds 0 and 1).
- Replayed with data's StreamReplayer (the vendored copy), as `netsearch_study.py`.

**Two tests per pair and determinization**, both against the real encoder on b(t+1):
- **A (mapping):** root = b(t+1) and state = convert(b(t+1)). This is `LeafEncoder` vs `encode(b(t+1))`. It isolates
  the engine → observation mapping: no information was revealed in between.
- **B (one turn):** root = b(t), state = convert(b(t+1)), vs `encode(b(t+1))`. That is what a depth-1 leaf would see.
  Its knowledge fields are a turn stale by design: moves, items and abilities revealed during the turn stay hidden.

**Element agreement** over D, counting only the elements where either side is non-zero or that the encoder writes:
- float columns: |Δ| ≤ 0.02;
- one-hots, flags and ids: exact;
- `hp` points: |Δ| ≤ 0.02 × max HP.

**Value:** P(win) = sigmoid(win_logit) of **G9p-cd@104.71M** (the frozen benchmark net, sha256 35cd1007…) on each
observation, CPU, fp32.

**P1 PASSES iff all three hold:**
- Test A element agreement ≥ 99.0% (pooled over all compared elements);
- Test A mean |ΔP(win)| ≤ 0.01;
- Test B mean |ΔP(win)| ≤ 0.03.

**Also reported (no rule):**
- B's element agreement;
- per-field disagreement counts;
- |ΔP| quantiles (50 / 90 / 99%);
- B split by whether anything was revealed between t and t + 1.

## P2: throughput (gate)
**Hardware:** ONE helper job, `--partition=training` (a dove node, 1 T4), `-c 12`, ≤ 1 h. Twelve worker processes
run concurrently: 3 battles × 4 determinizations, as the slot harness would run V3 (3 concurrent battles × 4 × 1 s).
- Each worker: torch threads 1, its own copy of G9p-cd on the T4, fp32, inference mode.
- Note: in the slot the opponent's 11 engine processes would share the CPUs too. The helper does not model that, so P2
  is optimistic on CPU contention. Its 12 CUDA contexts are pessimistic against a shared inference server.

**Positions:** 30 roots from P1's position set (the b(t) side), drawn with **seed 7312**. Each worker plays its
battle's root with determinization seed = its index within the battle.

**One decision:**
- Each of a battle's 4 workers runs the tree from its determinization for a **1000 ms** wall window (the window
  starts after the root is built; building the root is counted separately).
- **Leaves/decision** = the leaf values its 4 workers computed (net forwards completed) inside the window, summed.
- Terminal leaves (one side has nothing left) are scored without the net and are not counted.
- Batch sizes **B ∈ {16, 32, 64}**, each played on all 30 roots. The decision statistic uses the B with the highest
  median.

**P2 PASSES iff** the median leaves/decision over the 30 roots × 3 battles (90 decisions) at that B is
**≥ 2,000**.

**Time split** (accumulated per worker inside the windows; µs per leaf and the share of wall time):
- engine: `generate_instructions`, apply / reverse along the path, option lists;
- encode: `LeafEncoder.encode`, including reading the engine state through pyo3;
- tree: selection and backup in Python;
- collate: stacking the observations and the host → device copy;
- forward: the batched forward plus `cuda.synchronize`.

Also reported, as reference only: GPU forward throughput alone, at B ∈ {64, 256, 1024}, for 1 process and for 12
concurrent processes.

**Reading for the Rust question** (pre-registered wording; the largest share of worker wall time at the chosen B):
- **encode largest:** "Python leaf encoding is the bottleneck." The indicated port is a native leaf encoder: the v2
  arrays written in Rust inside poke-engine, or vectorised numpy. Report projected leaves/decision with encode 20×
  faster (SPEEDUPS: JS/native encoders ran 12–42× Python).
- **forward largest:** network-FLOP bound. Rust does not help; batching, a shared inference server or a smaller net
  do. Report projected leaves/decision with one shared server at the measured GPU-only rate.
- **engine largest:** the engine is already Rust. Only the pyo3 crossings (state reads, instruction objects) could
  move, which is SPEEDUPS row 4: a Rust batched-leaf tree.
- **tree largest:** a Rust tree (row 4).

## Void and stop
- A gate run is void if its job dies, or if fewer than 90% of P1's pairs or of P2's decisions complete. It is re-run
  once with the same seeds after the fix.
- After P1 and P2: report to main and STOP. No tree tuning, no games, no Rust, without a new decision.
- Code commit, net sha256, node, GPU and the CPU count go into the run manifest.
