"""edw edit library: the gated ActionHead path, unit usage, held-out metrics.

The edit (a program written into two repurposed ActionHead hidden units k1, k2; nothing else changes):
    G(x)       = γ·(g·x + b_g)                       gate on the head input x = [candidate latent c ; field latent ctx]
    unit k1    = GELU(G(x) + a·(r·(c − μ) + ρ))      W1[k1] = γg + a·[r ; 0],  b1[k1] = γb_g + a(ρ − r·μ)
    unit k2    = GELU(G(x))                          W1[k2] = γg,              b1[k2] = γb_g
    w2[k1] = +1/a, w2[k2] = −1/a
  Where the gate is open (G ≫ 0, both units linear): Δlogit = r·(c − μ) + ρ exactly (the G terms cancel).
  Where it is shut (G ≪ 0): both units ≈ 0, Δlogit ≈ 0.  Cost: the two units' old contributions are removed
  (they are chosen as the least-used units by mean ablation).
  r = Σ_k δ_k w_k with w_k a linear readout of matchup feature k from the final latent; δ_k in logits per unit.
"""
from __future__ import annotations

import numpy as np
import torch

from edw_common import head_logits, softmax


def gate_direction(X_on: np.ndarray, X_off: np.ndarray, margin: float = 8.0):
    """Fisher LDA direction on the 512-d head input separating 'on' from 'off' tokens; scaled so that the
    0.5th-percentile 'on' token and the 99.5th-percentile 'off' token sit at ±margin. Returns g, b, report."""
    mu1, mu0 = X_on.mean(0), X_off.mean(0)
    S = np.cov(X_on.T) + np.cov(X_off.T)
    S += 1e-3 * np.trace(S) / len(S) * np.eye(len(S))
    w = np.linalg.solve(S, mu1 - mu0)
    p1, p0 = X_on @ w, X_off @ w
    lo1, hi0 = np.percentile(p1, 0.5), np.percentile(p0, 99.5)
    if lo1 <= hi0:
        raise ValueError(f"gate not separable at the 0.5/99.5 percentiles: {lo1:.3f} <= {hi0:.3f}")
    scale = 2 * margin / (lo1 - hi0)
    b = -scale * (lo1 + hi0) / 2
    g = w * scale
    q1, q0 = X_on @ g + b, X_off @ g + b
    rep = dict(on_min=float(q1.min()), on_p05=float(np.percentile(q1, 0.5)), on_mean=float(q1.mean()),
               off_max=float(q0.max()), off_p995=float(np.percentile(q0, 99.5)), off_mean=float(q0.mean()),
               on_below0=float(np.mean(q1 < 0)), off_above0=float(np.mean(q0 > 0)))
    return g, float(b), rep


def unit_usage(fin, W, mask, rows):
    """KL(orig || unit k mean-ablated) averaged over rows, for every ActionHead hidden unit."""
    lg, H = head_logits(fin[rows], W, mask[rows], return_hid=True)
    H = H.astype(np.float64)
    m = mask[rows] > 0
    mean_k = (H * m[..., None]).sum((0, 1)) / m.sum()
    p0 = softmax(lg)
    w2 = W["w2"].numpy()
    kl = np.zeros(H.shape[-1])
    for k in range(H.shape[-1]):
        l1 = np.where(m, lg + w2[k] * (mean_k[k] - H[..., k]), -np.inf)
        p1 = softmax(l1)
        with np.errstate(invalid="ignore", divide="ignore"):
            kl[k] = np.nansum(np.where(p0 > 0, p0 * (np.log(p0) - np.log(p1)), 0), 1).mean()
    frac_active = (H > 0.05).mean((0, 1))
    return kl, frac_active


def make_W(W: dict, units, gate_g, gate_b, r, mu, rho, a: float = 1.0) -> dict:
    k1, k2 = units
    W2 = {k: v.clone() for k, v in W.items()}
    g = torch.as_tensor(gate_g, dtype=torch.float64)
    rr = torch.zeros(512, dtype=torch.float64); rr[:256] = torch.as_tensor(r, dtype=torch.float64)
    W2["W1"][k1] = g + a * rr
    W2["b1"][k1] = gate_b + a * (rho - float(np.dot(r, mu)))
    W2["W1"][k2] = g
    W2["b1"][k2] = gate_b
    W2["w2"][k1] = 1.0 / a
    W2["w2"][k2] = -1.0 / a
    return W2


def write_policy(pol, W2: dict):
    s = pol.action_net.score
    with torch.no_grad():
        s[0].weight.copy_(W2["W1"].to(s[0].weight.dtype))
        s[0].bias.copy_(W2["b1"].to(s[0].bias.dtype))
        s[2].weight.copy_(W2["w2"].to(s[2].weight.dtype)[None])
        s[2].bias.copy_(torch.as_tensor([W2["b2"]], dtype=s[2].bias.dtype))
