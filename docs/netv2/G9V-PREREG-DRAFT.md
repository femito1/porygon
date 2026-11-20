# G9V (value-trunk-gradient arm) — NOT DRAFTED: the probe shows no value swamping; recommendation AGAINST the arm

netv2 / orchestrator, 2026-11-20. Brief (orchestrator): measure ‖∇trunk L_value‖ / ‖∇trunk L_policy‖ for G9q's recipe; if
the ratio is ≤ ~2 (no swamping), say so and recommend against the arm instead of drafting it. **The per-minibatch ratio is
≈ 0.03–0.04, i.e. the policy's trunk gradient is ≈ 25–35× the value's — the opposite of MissingNo.Zero's ≈ 50× value
dominance.** No PREREG is proposed, no chain is written, nothing is armed.

## Measurement (`$C/runs/vtg-probe/RESULT.txt`, script `scripts/netv2_vtg/vtg_probe.py` md5 191a3a8e)
Fixed weights, fresh on-policy fastenv self-play rollouts (≈ 41.4k transitions each), G9q's stored recipe (vf_coef 0.5,
ent 0.01, aux 0.1, max_grad_norm 0.5, clip 0.1, batch 4096; value_trunk_grad 1.0). Medians [IQR] over minibatches.

| net | rollouts / minibatches | TRUNK ‖vf‖/‖POL‖ per minibatch | ‖E vf‖/‖E POL‖ (signal, ± jackknife SE) | clip coefficient |
|---|---|---|---|---|
| G9q-cd@1350 (01355036714.pt, md5 6cf40c0c) | 10 / 100 | 0.0384 [0.0346, 0.0428] | 0.45 ± 0.28 | 0.444 [0.427, 0.464], clipped in 100% |
| G9q @1540M (ckpt_01540011678.pt, md5 98871492) | 24 / 240 | 0.0278 [0.0253, 0.0314] | 0.197 ± 0.017 | 0.403 [0.382, 0.421], clipped in 100% |

Even on the averaged (signal) gradient the value's share is ≈ 0.45× (cd@1350, ± 0.28) / 0.20× (G9q@1540M, ± 0.02) the
policy's: the value gradient is small and consistent (coherence ≈ 0.3–0.5), the policy gradient large and mostly noise
(trunk coherence ≈ 0.004–0.006).
Reducing value_trunk_grad would remove a small, coherent signal and fix no imbalance. **Recommendation: no G9v arm.**

## Side observations (hypotheses, not results)
- Global-norm clipping is active in 100% of minibatches at ≈ 0.4×: max_grad_norm 0.5 effectively sets the update size.
- The policy gradient's trunk coherence is ≈ 0.004–0.006: a larger batch (or fewer, larger minibatches) may reduce noise.
  Either would be a separate pre-registered arm.
