"""Group-sequential stopping for duplicate-pair head-to-heads: bounds and operating characteristics.

DRAFT (wsp seq plan, .orchestra/wsp/seqtest-plan.md) -- not in use until the orchestrator approves.

The statistic is A's pair score X in {0, .25, .5, .75, 1} (mean of A's two game scores in a duplicate pair, a tie
scoring .5); theta = E[X] = A's win rate. Looks happen every k completed pairs (in pair-id order, see
seq_monitor.py). At look j with n_j pairs, mean m_j and sample SD s_j (plug-in, re-estimated at every look):
    SE_j = s_j / sqrt(n_j),   Z0_j = (m_j - theta0) / SE_j,   Z1_j = (m_j - theta1) / SE_j
    success  if Z0_j >= b_j          (the repeated 95% CI excludes theta0 from below when b = r)
    futility if Z1_j <= -c_j         (the data exclude theta1 from above at the beta-spending level)
    the last look always decides (b_K = theta0-side and theta1-side bounds meet): no inconclusive outcome.
b_j: Lan-DeMets alpha-spending; c_j: beta-spending under theta1 (Jennison & Turnbull 2000, ch. 7), computed by the
usual recursive numerical integration of the canonical joint distribution (ch. 19) with information I_j = n_j/sigma^2.
Two kinds:
    onesided  : H0 theta <= theta0 vs H1 theta = theta1 (alpha one-sided, futility NON-binding: b_j ignores the
                futility bound, so alpha holds even if a futility stop is overruled). Types (a) gain/margin, (c)
                non-inferiority.
    threshold : symmetric decision "theta >= c" with errors alpha = beta at c -/+ d (theta0 = c - d,
                theta1 = c + d; binding). Type (b); the final-look bound sits at ~c, like the fixed rule.
Presets (the plan's designs): gain (50 -> 52%, alpha .025, power .8, k 100), margin10 (50 -> 60%, power .9, k 25),
threshold (c = 40% by default, alpha = beta = .05, d matched to the fixed --n-fixed rule, k 50), noninf (48.5 ->
50.5%, power .8, k 100). Spending: HSD(-4) for alpha, HSD(-2) for beta unless --alpha-sf / --beta-sf;
--boundary sprt gives a truncated group SPRT on the same looks instead (last-look cut tuned to alpha).
Usage:
    python scripts/wsp/seq_design.py --preset gain [--sims 20000]          # bounds + OC (+ fixed-n comparator)
    python scripts/wsp/seq_design.py --preset threshold --c 0.60 --save design.json
    python scripts/wsp/seq_design.py --kind onesided --theta0 .5 --theta1 .55 --alpha .025 --beta .1 --k 50
    python scripts/wsp/seq_design.py --all-presets --sims 20000            # every table of the plan
    python scripts/wsp/seq_design.py --robust                              # alpha/power under other mixes
    python scripts/wsp/seq_design.py --explore gain                        # spending-function / look-interval scan
    python scripts/wsp/seq_design.py --preset gain --mix-counts 1743 13 6071 10 2871   # plan with another mix
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

try:  # scipy if present; the cluster venv (gloria .venv) has none, so numpy + stdlib otherwise (same results)
    if os.environ.get("SEQ_NO_SCIPY"):
        raise ImportError
    from scipy.optimize import brentq
    from scipy.stats import norm
    HAVE_SCIPY = True
except ImportError:
    from statistics import NormalDist
    HAVE_SCIPY = False
    _erfc = np.vectorize(math.erfc, otypes=[float])
    _ND = NormalDist()

    class norm:  # noqa: N801 -- the subset of scipy.stats.norm used here
        @staticmethod
        def sf(x):
            r = 0.5 * _erfc(np.asarray(x, float) / math.sqrt(2))
            return r if np.ndim(r) else float(r)

        @staticmethod
        def cdf(x):
            r = 0.5 * _erfc(-np.asarray(x, float) / math.sqrt(2))
            return r if np.ndim(r) else float(r)

        @staticmethod
        def isf(p):
            return -_ND.inv_cdf(float(p))

    def brentq(f, a, b, xtol=2e-12, rtol=8.881784197001252e-16, maxiter=200):
        """Brent's root finder (as scipy.optimize.brentq; Numerical Recipes zbrent)."""
        fa, fb = f(a), f(b)
        if fa == 0:
            return a
        if fb == 0:
            return b
        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have different signs")
        c, fc, d = a, fa, b - a
        e = d
        for _ in range(maxiter):
            if fb * fc > 0:
                c, fc, d = a, fa, b - a
                e = d
            if abs(fc) < abs(fb):
                a, b, c = b, c, b
                fa, fb, fc = fb, fc, fb
            tol = 2 * rtol * abs(b) + 0.5 * xtol
            m = 0.5 * (c - b)
            if abs(m) <= tol or fb == 0:
                return b
            if abs(e) >= tol and abs(fa) > abs(fb):
                s = fb / fa
                if a == c:
                    p, q = 2 * m * s, 1 - s
                else:
                    q, r = fa / fc, fb / fc
                    p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
                    q = (q - 1) * (r - 1) * (s - 1)
                if p > 0:
                    q = -q
                else:
                    p = -p
                if 2 * p < min(3 * m * q - abs(tol * q), abs(e * q)):
                    e, d = d, p / q
                else:
                    d = e = m
            else:
                d = e = m
            a, fa = b, fb
            b += d if abs(d) > tol else (tol if m > 0 else -tol)
            fb = f(b)
        raise RuntimeError("brentq: no convergence")

# Empirical pair-outcome mix, net vs net (wsp out/h2h jobs 676503, 688986, 698101: 10 runs, 16,400 pairs).
# Categories of A's pair score: 0, .25, .5, .75, 1 (the .25/.75 pairs contain one tie).
MIX_NET = {"counts": [3316, 127, 9570, 82, 3305], "source": "wsp h2h 676503+688986+698101, 16,400 pairs"}
VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
SQ2PI = math.sqrt(2 * math.pi)


# ----------------------------------------------------------------------------------------------- pair-outcome model
def mix_probs(theta: float, split: float | None = None, tie: float | None = None,
              counts=None) -> np.ndarray:
    """P(X = 0, .25, .5, .75, 1) at mean theta: the empirical split (X = .5) and one-tie (X = .25 / .75, split
    evenly) fractions kept, the both-won / both-lost mass shifted to give mean theta (split shrunk if needed)."""
    c = np.asarray(MIX_NET["counts"] if counts is None else counts, float)
    s = c[2] / c.sum() if split is None else split
    t = (c[1] + c[3]) / 2 / c.sum() if tie is None else tie
    lo, hi = 0.5 * s + t, 1 - 0.5 * s - t
    if not lo <= theta <= hi:  # outside the reachable range: shrink the split so the mix stays valid
        s = 2 * min(theta, 1 - theta) - 2 * t - 1e-9
        s = max(s, 0.0)
    w = theta - 0.5 * s - t
    p = np.array([1 - s - 2 * t - w, t, s, t, w])
    assert np.all(p >= -1e-12) and abs(p @ VALUES - theta) < 1e-9, (theta, p)
    return np.clip(p, 0, 1) / np.clip(p, 0, 1).sum()


def mix_latent(theta: float, split0: float | None = None) -> np.ndarray:
    """Alternative (robustness) model: per-pair team luck tau ~ N(0, s_tau^2) on the logit scale, game wins
    independent given tau with P = F(beta +/- tau); s_tau fitted to the empirical split at theta = .5, beta to theta.
    Ties ignored. The split fraction then falls as theta moves away from .5 (as in search-vs-Foul-Play runs)."""
    c = np.asarray(MIX_NET["counts"], float)
    s0 = (c[2] + c[1] + c[3]) / c.sum() if split0 is None else split0
    x, wq = np.polynomial.hermite_e.hermegauss(80)
    wq = wq / wq.sum()
    F = lambda u: 1 / (1 + np.exp(-u))
    def split_at(st, b):
        p0, p1 = F(b + st * x), F(b - st * x)
        return wq @ (p0 * (1 - p1) + (1 - p0) * p1), wq @ (p0 * p1), wq @ ((1 - p0) * (1 - p1))
    st = brentq(lambda v: split_at(v, 0.0)[0] - s0, 1e-6, 20)
    b = brentq(lambda v: (lambda s, w, l: w + s / 2)(*split_at(st, v)) - theta, -10, 10)
    s, w, l = split_at(st, b)
    return np.array([l, 0.0, s, 0.0, w]) / (l + s + w)


def sd_of(p: np.ndarray) -> float:
    m = p @ VALUES
    return float(math.sqrt(p @ VALUES ** 2 - m * m))


# ----------------------------------------------------------------------------------------------- spending functions
def spend(sf: tuple, t: np.ndarray, total: float) -> np.ndarray:
    """Cumulative error spent at information fractions t. sf = ('hsd', gamma) Hwang-Shih-DeCani (gamma -4 ~
    O'Brien-Fleming-like, 1 ~ Pocock-like), ('pow', rho) Kim-DeMets total*t^rho, ('obf',) Lan-DeMets OBF-type,
    ('pocock',) Lan-DeMets Pocock-type."""
    t = np.clip(np.asarray(t, float), 0, 1)
    kind = sf[0]
    if kind == "hsd":
        g = float(sf[1])
        return total * t if abs(g) < 1e-9 else total * (1 - np.exp(-g * t)) / (1 - np.exp(-g))
    if kind == "pow":
        return total * t ** float(sf[1])
    if kind == "obf":
        z = norm.isf(total / 2)
        with np.errstate(divide="ignore"):
            return np.where(t > 0, 2 * norm.sf(z / np.sqrt(np.maximum(t, 1e-300))), 0.0)
    if kind == "pocock":
        return total * np.log1p((math.e - 1) * t)
    raise ValueError(sf)


# ------------------------------------------------------------------------------- canonical joint distribution engine
def _simpson(lo: float, hi: float, h: float):
    n = max(2, int(math.ceil((hi - lo) / h)))
    n += n % 2
    z = np.linspace(lo, hi, n + 1)
    w = np.ones(n + 1)
    w[1:-1:2], w[2:-1:2] = 4, 2
    return z, w * (hi - lo) / n / 3


class Path:
    """Sub-density of Z_k on the continuation region of looks 1..k under drift eta (Z_k ~ N(eta sqrt(I_k), 1),
    independent score increments). Advance look by look; exit probabilities for any bound value at the next look."""

    def __init__(self, I: np.ndarray, eta: float):
        self.I, self.sq, self.eta, self.k = np.asarray(I, float), np.sqrt(I), eta, -1
        self.z = self.gw = None
        self.dead = False

    def _cond(self, k):  # conditional mean / sd of S_k = Z_k sqrt(I_k) given grid z at look k-1
        d = self.I[k] - self.I[k - 1]
        return self.z * self.sq[k - 1] + self.eta * d, math.sqrt(d)

    def p_up(self, k: int, b) -> np.ndarray:
        """P(reach look k (continue at 1..k-1), Z_k >= b) for an array of b."""
        b = np.atleast_1d(np.asarray(b, float))
        if k == 0:
            return norm.sf(b - self.eta * self.sq[0])
        if self.dead:
            return np.zeros_like(b)
        m, sd = self._cond(k)
        return norm.sf((b[:, None] * self.sq[k] - m[None, :]) / sd) @ self.gw

    def p_dn(self, k: int, a) -> np.ndarray:
        a = np.atleast_1d(np.asarray(a, float))
        if k == 0:
            return norm.cdf(a - self.eta * self.sq[0])
        if self.dead:
            return np.zeros_like(a)
        m, sd = self._cond(k)
        return norm.cdf((a[:, None] * self.sq[k] - m[None, :]) / sd) @ self.gw

    def advance(self, k: int, a: float, b: float, h: float = 0.05, per_sd: float = 7.0) -> None:
        """Condition on continuing at look k (a < Z_k < b); k must be self.k + 1. Grid spacing: <= h and <= 1/per_sd
        of the transition-kernel sd (in z units) into and out of look k (Simpson's rule)."""
        assert k == self.k + 1
        self.k = k
        if self.dead:
            return
        mu = self.eta * self.sq[k]
        lo, hi = max(a, mu - 8.5), min(b, mu + 8.5)
        if lo >= hi:
            self.dead = True
            return
        if k > 0:
            h = min(h, math.sqrt((self.I[k] - self.I[k - 1]) / self.I[k]) / per_sd)
        if k + 1 < len(self.I):
            h = min(h, math.sqrt((self.I[k + 1] - self.I[k]) / self.I[k]) / per_sd)
        z, w = _simpson(lo, hi, h)
        if k == 0:
            g = np.exp(-0.5 * (z - mu) ** 2) / SQ2PI
        else:
            m, sd = self._cond(k)
            u = (z[:, None] * self.sq[k] - m[None, :]) / sd
            g = np.exp(-0.5 * u * u) @ self.gw * (self.sq[k] / sd / SQ2PI)
        self.z, self.gw = z, g * w


def exit_probs(I, a, b, eta):
    """Per-look probabilities of crossing the upper bound b_k / lower bound a_k (Z scale) under drift eta."""
    P = Path(I, eta)
    up, dn = np.zeros(len(I)), np.zeros(len(I))
    for k in range(len(I)):
        up[k] = P.p_up(k, b[k])[0] if np.isfinite(b[k]) else 0.0
        dn[k] = P.p_dn(k, a[k])[0] if np.isfinite(a[k]) else 0.0
        if k < len(I) - 1:
            P.advance(k, a[k], b[k])
    return up, dn


# ------------------------------------------------------------------------------------------------------- the design
@dataclass
class Design:
    kind: str                       # 'onesided' | 'threshold'
    theta0: float
    theta1: float
    alpha: float
    beta: float
    k: int                          # pairs per look
    sigma: float                    # planning SD of the pair score
    alpha_sf: list = field(default_factory=lambda: ["hsd", -4.0])
    beta_sf: list = field(default_factory=lambda: ["hsd", -2.0])
    rci_alpha: float = 0.025        # per side: 95% repeated CI
    n_fixed: int = 0                # comparator fixed-n design (pairs)
    boundary: str = "spend"         # 'spend' (error spending) | 'sprt' (truncated group SPRT, last-look cut tuned)
    binding_futility: bool | None = None   # None: binding for kind 'threshold', non-binding for 'onesided'
    name: str = ""
    note: str = ""
    # filled by solve()
    n_max: int = 0
    n: list = field(default_factory=list)       # pairs at each look
    b: list = field(default_factory=list)       # success: Z0 >= b
    c: list = field(default_factory=list)       # futility: Z1 <= -c
    r: list = field(default_factory=list)       # RCI constants (non-binding alpha-spending at rci_alpha)
    created: str = ""

    @property
    def binding(self) -> bool:
        return self.kind == "threshold" if self.binding_futility is None else bool(self.binding_futility)

    def info(self, sigma: float | None = None) -> np.ndarray:
        return np.asarray(self.n, float) / (sigma or self.sigma) ** 2

    def a_z0(self, sigma: float | None = None) -> np.ndarray:
        """Futility bound on the Z0 scale (a_j = eta1 sqrt(I_j) - c_j)."""
        return (self.theta1 - self.theta0) * np.sqrt(self.info(sigma)) - np.asarray(self.c)

    def bounds_wr(self, sigma: float | None = None):
        """Planned bounds on the win-rate scale: success if mean >= hi, futility if mean <= lo."""
        se = (sigma or self.sigma) / np.sqrt(np.asarray(self.n, float))
        return self.theta0 + np.asarray(self.b) * se, self.theta1 - np.asarray(self.c) * se


def _solve_bounds(t, I, eta1, alpha, beta, asf, bsf, binding):
    """Bounds b (Z0 success) and a (Z0 futility) for looks at information I (fractions t). Last look: a_K := b_K."""
    K = len(t)
    fa, fb = spend(asf, t, alpha), spend(bsf, t, beta)
    da, db = np.diff(np.r_[0.0, fa]), np.diff(np.r_[0.0, fb])
    P0, P1 = Path(I, 0.0), Path(I, eta1)
    a, b = np.full(K, -np.inf), np.full(K, np.inf)
    a_raw_last = None
    for k in range(K):
        if da[k] > 1e-15:
            f = lambda x: P0.p_up(k, x)[0] - da[k]
            b[k] = brentq(f, -12, 40, xtol=1e-10) if f(-12) > 0 else -12.0
        if db[k] > 1e-15:
            g = lambda x: P1.p_dn(k, x)[0] - db[k]
            a[k] = brentq(g, -40, 40, xtol=1e-10) if g(40) > 0 else 40.0
        if k == K - 1:
            a_raw_last = a[k]
            a[k] = b[k]
        a[k] = min(a[k], b[k])
        if k < K - 1:
            P0.advance(k, a[k] if binding else -np.inf, b[k])
            P1.advance(k, a[k], b[k])
    return a, b, a_raw_last


def _sprt_bounds(I, eta1, alpha, beta, binding):
    """Truncated group SPRT on the normal approximation: LLR_j = eta1 S_j / sigma^2 - eta1^2 I_j / 2 (S_j = Z0_j
    sqrt(I_j)); success if LLR >= log((1-beta)/alpha), futility if LLR <= log(beta/(1-alpha)); at the last look a
    single cut, tuned so that P(success | theta0) = alpha exactly (with the futility stops if binding, without them
    if not). Returns a, b (Z0 scale) and the power at theta1."""
    A, B = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    sq = np.sqrt(I)
    b = (A + eta1 ** 2 * I / 2) / (eta1 * sq)
    a = (B + eta1 ** 2 * I / 2) / (eta1 * sq)

    def with_cut(cut):
        bb, aa = b.copy(), a.copy()
        bb[-1] = aa[-1] = cut
        return aa, bb

    def alpha_of(cut):
        aa, bb = with_cut(cut)
        if not binding:
            aa = np.r_[np.full(len(I) - 1, -np.inf), cut]
        return exit_probs(I, aa, bb, 0.0)[0].sum()

    if alpha_of(40.0) > alpha:      # the upper boundary alone spends more than alpha: raise A
        raise ValueError("SPRT upper boundary exceeds alpha before the last look; use more looks or n_max")
    cut = brentq(lambda x: alpha_of(x) - alpha, -8, 40, xtol=1e-9)
    a, b = with_cut(cut)
    return a, b, exit_probs(I, a, b, eta1)[0].sum()


def solve(d: Design, n_max: int | None = None) -> Design:
    """Fill n_max (the smallest multiple of k whose beta-spending futility bound at the last look reaches the
    success bound, i.e. power >= 1 - beta at theta1, unless n_max is given), the look schedule n = k, 2k, ...,
    n_max, and the bounds b (Z0 success), c (Z1 futility critical values), r (RCI constants)."""
    eta1 = d.theta1 - d.theta0
    asf, bsf = tuple(d.alpha_sf), tuple(d.beta_sf)

    def bounds_for(K):
        t = np.arange(1, K + 1) / K
        I = np.arange(1, K + 1) * d.k / d.sigma ** 2
        if d.boundary == "sprt":
            a, b, power = _sprt_bounds(I, eta1, d.alpha, d.beta, d.binding)
            return t, I, a, b, power >= 1 - d.beta - 1e-9
        a, b, araw = _solve_bounds(t, I, eta1, d.alpha, d.beta, asf, bsf, d.binding)
        return t, I, a, b, araw >= b[-1] - 1e-9

    if n_max is None:
        z = norm.isf(d.alpha) + norm.isf(d.beta)
        n_fix = (z * d.sigma / eta1) ** 2
        K = max(1, math.floor(0.8 * n_fix / d.k) if d.boundary == "sprt" else math.floor(n_fix / d.k))
        while True:  # smallest K with power >= 1 - beta (spending: a_K(raw) >= b_K; a_K(raw) rises with K)
            try:
                ok = bounds_for(K)[4]
            except ValueError:
                ok = False
            if ok or K * d.k > 4 * n_fix:
                break
            K += 1
        n_max = K * d.k
    K = max(1, round(n_max / d.k))
    n_max = K * d.k
    t, I, a, b, _ = bounds_for(K)
    # RCI constants: one-sided rci_alpha spending with the alpha spending family, no futility
    r = _solve_bounds(t, I, eta1, d.rci_alpha, d.beta, asf, bsf, False)[1]
    d.n_max = int(n_max)
    d.n = [int(x) for x in (np.arange(1, K + 1) * d.k)]
    d.b = [float(x) for x in b]
    d.c = [float(x) for x in (eta1 * np.sqrt(I) - a)]
    d.r = [float(x) for x in r]
    d.created = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    return d


def load(path: str) -> Design:
    return Design(**json.load(open(path)))


def save(d: Design, path: str) -> None:
    json.dump(asdict(d), open(path, "w"), indent=1)


# ----------------------------------------------------------------------------------------------- presets
def sigma_at(theta: float) -> float:
    return sd_of(mix_probs(theta))


def preset(name: str, **kw) -> Design:
    """The plan's designs (see seqtest-plan.md). kw overrides any field."""
    if name == "gain":           # (a) gain claim: 95% (two-sided) interval excludes 50%, powered at +2 points
        p = dict(kind="onesided", theta0=0.50, theta1=0.52, alpha=0.025, beta=0.20, k=100, n_fixed=2000,
                 note="T1-T5 style gain claim; fixed comparator 4000 games")
    elif name == "margin10":     # (a) "A beats B", powered at +10 points (1000-game scale)
        p = dict(kind="onesided", theta0=0.50, theta1=0.60, alpha=0.025, beta=0.10, k=25, n_fixed=500,
                 note="superiority powered at +10; fixed comparator 1000 games")
    elif name == "noninf":       # (c) non-inferiority: lower 95% bound > -1.5 points, powered at +0.5
        p = dict(kind="onesided", theta0=0.485, theta1=0.505, alpha=0.025, beta=0.20, k=100, n_fixed=2000,
                 note="T2 style safe-edit claim; fixed comparator 4000 games")
    elif name == "threshold":    # (b) decide theta >= c; errors as the fixed 1000-game rule's at c -/+ d
        p = dict(kind="threshold", alpha=0.05, beta=0.05, k=50, n_fixed=500, c=0.40, d=None,
                 note="P1/R1 style point-estimate threshold; fixed comparator 1000 games")
    else:
        raise ValueError(name)
    p.update({k: v for k, v in kw.items() if v is not None})
    if p["kind"] == "threshold":
        c, dd = p.pop("c"), p.pop("d")
        if dd is None:  # the half-width at which the fixed n_fixed-pair point-estimate rule errs with prob alpha
            dd = norm.isf(p["alpha"]) * sigma_at(c) / math.sqrt(p["n_fixed"])
        p["theta0"], p["theta1"] = c - dd, c + dd
    else:
        p.pop("c", None), p.pop("d", None)
    p.setdefault("sigma", sigma_at((p["theta0"] + p["theta1"]) / 2))
    return Design(name=name, **p)


# ------------------------------------------------------------------------------------------- operating characteristics
def oc_numeric(d: Design, theta: float, sigma: float | None = None) -> dict:
    """Exact OC under the normal model with known sigma (the design's own approximation)."""
    s = sigma or sd_of(mix_probs(theta))
    I = d.info(s)
    a, b = d.a_z0(s), np.asarray(d.b)
    up, dn = exit_probs(I, a, b, theta - d.theta0)
    stop = up + dn
    reach = 1 - np.r_[0, np.cumsum(stop)[:-1]]
    n = np.asarray(d.n, float)
    return {"p_success": float(up.sum()), "p_futility": float(dn.sum()), "E_pairs": float(stop @ n),
            "p_reach_max": float(reach[-1])}


def fixed_rule(d: Design, mean: np.ndarray, se: np.ndarray) -> np.ndarray:
    """The comparator fixed-n decision (True = success)."""
    if d.kind == "threshold":
        return mean >= (d.theta0 + d.theta1) / 2
    return mean - norm.isf(d.alpha) * se > d.theta0


def simulate(d: Design, theta: float, reps: int = 20000, seed: int = 0, model: str = "empirical",
             variance: str = "plugin", fixed: bool = True, split: float | None = None) -> dict:
    """Monte Carlo OC with pair outcomes drawn from the pair-outcome mix at mean theta, plug-in (or planning)
    variance at every look, exactly the monitor's rule."""
    rng = np.random.default_rng(seed)
    p = mix_probs(theta, split=split) if model == "empirical" else mix_latent(theta)
    K, k = len(d.n), d.k
    cnt = rng.multinomial(k, p, size=(reps, K)).cumsum(axis=1)          # (reps, K, 5)
    n = np.asarray(d.n, float)
    sx, sxx = cnt @ VALUES, cnt @ VALUES ** 2
    mean = sx / n
    var = np.maximum((sxx - n * mean ** 2) / (n - 1), 1e-12) if variance == "plugin" else np.full_like(mean, d.sigma ** 2)
    se = np.sqrt(var / n)
    succ = (mean - d.theta0) / se >= np.asarray(d.b)
    fut = (mean - d.theta1) / se <= -np.asarray(d.c)
    fut[:, -1] = ~succ[:, -1]                                            # the last look always decides
    stop = succ | fut
    first = np.argmax(stop, axis=1)
    rows = np.arange(reps)
    is_succ = succ[rows, first]
    N = n[first]
    est = mean[rows, first]
    out = {"theta": theta, "p_success": float(is_succ.mean()), "p_futility": float((~is_succ).mean()),
           "p_inconclusive": 0.0, "E_pairs": float(N.mean()), "p90_pairs": float(np.percentile(N, 90)),
           "p_reach_max": float((first == K - 1).mean()), "bias_naive": float(est.mean() - theta),
           "mc_se_success": float(math.sqrt(max(is_succ.mean() * (1 - is_succ.mean()), 1e-12) / reps))}
    if fixed and d.n_fixed:
        cf = rng.multinomial(d.n_fixed, p, size=reps)
        m = cf @ VALUES / d.n_fixed
        v = np.maximum((cf @ VALUES ** 2 - d.n_fixed * m ** 2) / (d.n_fixed - 1), 1e-12)
        out["fixed_p_success"] = float(fixed_rule(d, m, np.sqrt(v / d.n_fixed)).mean())
    return out


# --------------------------------------------------------------------------------------- inference at stopping
def stagewise_p(d: Design, j: int, z0: float, eta: float, sigma: float) -> float:
    """P_eta(outcome at least as extreme as stopping at look j with Z0 = z0) in the stage-wise ordering
    (Armitage; Tsiatis, Rosner & Mehta 1984), bounds as specified (futility followed)."""
    I = d.info(sigma)
    a, b = d.a_z0(sigma), np.asarray(d.b)
    P = Path(I, eta)
    tot = 0.0
    for k in range(j):
        tot += P.p_up(k, b[k])[0]
        P.advance(k, a[k], b[k])
    return float(tot + P.p_up(j, z0)[0])


def final_inference(d: Design, j: int, mean: float, sd: float) -> dict:
    """At a stop at look j (0-based) with pair-score mean and SD: naive estimate, 95% repeated CI, and the
    median-unbiased estimate with its stage-wise-ordering 95% CI."""
    n = d.n[j]
    se = sd / math.sqrt(n)
    z0 = (mean - d.theta0) / se
    out = {"look": j + 1, "pairs": n, "naive": mean, "se": se,
           "rci": (mean - d.r[j] * se, mean + d.r[j] * se)}
    def root(q):
        f = lambda eta: stagewise_p(d, j, z0, eta, sd) - q
        lo, hi = -0.5, 0.5
        return d.theta0 + brentq(f, lo, hi, xtol=1e-6) if f(lo) < 0 < f(hi) else float("nan")
    out["mue"], out["ci_lo"], out["ci_hi"] = root(0.5), root(0.025), root(0.975)
    return out


# ------------------------------------------------------------------------------------------------------- printing
def fmt_bounds(d: Design, every: int = 1) -> str:
    hi, lo = d.bounds_wr()
    lines = [f"design {d.name or d.kind}: theta0 {100 * d.theta0:.2f}%, theta1 {100 * d.theta1:.2f}%, "
             f"alpha {d.alpha} ({'binding' if d.binding else 'non-binding futility'}), power {1 - d.beta:.2f}, "
             f"sigma {d.sigma:.4f}, look every {d.k} pairs, n_max {d.n_max} pairs ({2 * d.n_max} games), "
             + ("truncated group SPRT (last-look cut tuned to alpha)" if d.boundary == "sprt" else
                f"spending alpha {tuple(d.alpha_sf)} beta {tuple(d.beta_sf)}"),
             "  look  pairs  games | success: Z0>=b   win rate >= | futility: Z1<=-c  win rate <= | RCI const"]
    K = len(d.n)
    for j in range(K):
        if (j + 1) % every and j != K - 1:
            continue
        lines.append(f"  {j + 1:4d} {d.n[j]:6d} {2 * d.n[j]:6d} | {d.b[j]:13.3f} {100 * hi[j]:12.2f}% |"
                     f" {d.c[j]:15.3f} {100 * lo[j]:12.2f}% | {d.r[j]:8.3f}")
    return "\n".join(lines)


def fmt_oc(d: Design, thetas, reps: int, seed: int = 1, model: str = "empirical") -> str:
    head = (f"  OC ({reps} sims/point, {model} mix, plug-in variance; fixed = {d.n_fixed} pairs) \n"
            "  theta  P(succ) P(fut)  P(inc) E[pairs] p90  P(n_max) | fixed P(succ) pairs | saved  naive bias")
    rows = [head]
    for i, th in enumerate(thetas):
        o = simulate(d, th, reps, seed + i, model=model)
        fx = o.get("fixed_p_success", float("nan"))
        rows.append(f"  {100 * th:5.1f}% {o['p_success']:6.3f} {o['p_futility']:6.3f} {o['p_inconclusive']:6.3f}"
                    f" {o['E_pairs']:7.0f} {o['p90_pairs']:5.0f} {o['p_reach_max']:7.3f} | {fx:12.3f} {d.n_fixed:6d}"
                    f" | {1 - o['E_pairs'] / d.n_fixed:5.0%} {100 * o['bias_naive']:+7.2f}pt")
    return "\n".join(rows)


THETAS = {"gain": [0.45, 0.48, 0.50, 0.51, 0.52, 0.53, 0.55, 0.60],
          "margin10": [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65],
          "noninf": [0.46, 0.47, 0.485, 0.495, 0.50, 0.505, 0.52]}


def thetas_for(d: Design) -> list:
    """True win rates for the OC table: the preset's list, else theta0 / theta1 and points around them (threshold
    designs: c -7, -4, -d, 0, +d, +4, +7 points)."""
    if d.kind == "threshold":
        c, h = (d.theta0 + d.theta1) / 2, (d.theta1 - d.theta0) / 2
        return [c - 0.07, c - 0.04, c - h, c, c + h, c + 0.04, c + 0.07]
    if d.name in THETAS:
        return THETAS[d.name]
    lo, hi = d.theta0, d.theta1
    w = hi - lo
    return [lo - 2 * w, lo - w, lo, (lo + hi) / 2, hi, hi + w, hi + 2 * w]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["gain", "margin10", "noninf", "threshold"])
    ap.add_argument("--all-presets", action="store_true")
    ap.add_argument("--kind", choices=["onesided", "threshold"])
    ap.add_argument("--theta0", type=float)
    ap.add_argument("--theta1", type=float)
    ap.add_argument("--c", type=float, help="threshold: decision point")
    ap.add_argument("--d", type=float, help="threshold: half-width of the indifference zone (default: matched "
                                            "to the fixed n_fixed-pair rule's error alpha)")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--beta", type=float)
    ap.add_argument("--k", type=int, help="pairs per look")
    ap.add_argument("--n-max", type=int, help="pairs; default: smallest multiple of k with power >= 1 - beta")
    ap.add_argument("--n-fixed", type=int, help="comparator fixed design (pairs)")
    ap.add_argument("--sigma", type=float, help="planning SD of the pair score (default: empirical mix)")
    ap.add_argument("--alpha-sf", nargs="+", help="e.g. hsd -4 | obf | pocock | pow 3")
    ap.add_argument("--beta-sf", nargs="+")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--thetas", type=float, nargs="+")
    ap.add_argument("--model", default="empirical", choices=["empirical", "latent"])
    ap.add_argument("--every", type=int, default=1, help="print every n-th look")
    ap.add_argument("--save", help="write the design JSON (the monitor's input)")
    ap.add_argument("--explore", choices=["gain", "noninf", "threshold", "margin10"])
    ap.add_argument("--robust", nargs="*", help="robustness table for these presets (default: all)")
    ap.add_argument("--confseq", choices=["gain", "noninf", "margin10"], help="always-valid comparator")
    ap.add_argument("--boundary", choices=["spend", "sprt"], help="sprt: truncated group SPRT instead of spending")
    ap.add_argument("--binding", choices=["yes", "no"], help="override the futility convention")
    ap.add_argument("--mix-counts", type=float, nargs=5, metavar=("N0", "N25", "N50", "N75", "N100"),
                    help="planning pair-outcome counts (seq_monitor.py --mix), default net-vs-net 16,400 pairs")
    a = ap.parse_args()
    if a.mix_counts:
        MIX_NET["counts"] = list(a.mix_counts)
        MIX_NET["source"] = "--mix-counts"

    def sf(x):
        return None if x is None else [x[0]] + [float(v) for v in x[1:]]

    if a.explore:
        explore(a.explore, a.sims)
        return
    if a.confseq:
        confseq(a.confseq)
        return
    if a.robust is not None:
        robustness(a.robust or ["gain", "margin10", "threshold", "noninf"], a.sims)
        return
    names = ["gain", "margin10", "threshold", "noninf"] if a.all_presets else [a.preset]
    for nm in names:
        if nm is None:
            p = dict(kind=a.kind or "onesided", theta0=a.theta0, theta1=a.theta1, alpha=a.alpha or 0.025,
                     beta=a.beta or 0.2, k=a.k or 100, n_fixed=a.n_fixed or 0)
            if p["kind"] == "threshold" and a.c is not None:
                dd = a.d if a.d is not None else norm.isf(p["alpha"]) * sigma_at(a.c) / math.sqrt(p["n_fixed"] or 500)
                p["theta0"], p["theta1"] = a.c - dd, a.c + dd
            p["sigma"] = a.sigma or sigma_at((p["theta0"] + p["theta1"]) / 2)
            if a.alpha_sf:
                p["alpha_sf"] = sf(a.alpha_sf)
            if a.beta_sf:
                p["beta_sf"] = sf(a.beta_sf)
            p["boundary"] = a.boundary or "spend"
            p["binding_futility"] = None if a.binding is None else a.binding == "yes"
            d = Design(name="custom", **p)
        else:
            d = preset(nm, alpha=a.alpha, beta=a.beta, k=a.k, n_fixed=a.n_fixed, sigma=a.sigma, c=a.c, d=a.d,
                       alpha_sf=sf(a.alpha_sf), beta_sf=sf(a.beta_sf), boundary=a.boundary,
                       binding_futility=None if a.binding is None else a.binding == "yes")
        solve(d, a.n_max)
        print(fmt_bounds(d, a.every))
        if a.sims:
            print(fmt_oc(d, a.thetas or thetas_for(d), a.sims, model=a.model))
        print()
        if a.save:
            save(d, a.save if len(names) == 1 else a.save.replace(".json", f"-{nm}.json"))


def robustness(names, reps: int) -> None:
    """alpha (P(success) at theta0) and power (at theta1) when the pair-outcome mix differs from the planning one:
    split fraction 0.45 / 0.70 instead of 0.58 (pair SD 0.343 / 0.274 vs 0.319), the latent team-luck model, and
    the planning sigma held fixed instead of re-estimated at each look."""
    cases = [("empirical mix, plug-in", dict()), ("latent model, plug-in", dict(model="latent")),
             ("split .45, plug-in", dict(split=0.45)), ("split .70, plug-in", dict(split=0.70)),
             ("split .45, fixed sigma", dict(split=0.45, variance="fixed")),
             ("split .70, fixed sigma", dict(split=0.70, variance="fixed"))]
    print(f"robustness ({reps} sims per cell): alpha = P(success | theta0), power = P(success | theta1), "
          "E[pairs] at theta0")
    for nm in names:
        d = solve(preset(nm))
        print(f"  {nm}: nominal alpha {d.alpha} (non-binding)" if not d.binding else
              f"  {nm}: nominal alpha = beta = {d.alpha}")
        for i, (lab, kw) in enumerate(cases):
            s0 = simulate(d, d.theta0, reps, 100 + i, fixed=False, **kw)
            s1 = simulate(d, d.theta1, reps, 200 + i, fixed=False, **kw)
            print(f"    {lab:24s} alpha {s0['p_success']:.4f}  power {s1['p_success']:.3f}  E[pairs|theta0] "
                  f"{s0['E_pairs']:6.0f}")


def confseq(name: str, alpha: float = 0.05) -> None:
    """Comparator: a two-sided always-valid normal-mixture confidence sequence (Robbins) on the same looks,
    |Z0_n| >= sqrt((1 + rho/n) (log(1 + n/rho) + 2 log(1/alpha))), no futility stop. For each mixing scale rho
    (pairs): the smallest n_max (multiple of k) with power >= 1 - beta at theta1, and E[pairs] (normal model)."""
    d0 = preset(name)
    s, e1 = d0.sigma, d0.theta1 - d0.theta0
    print(f"{name}: always-valid confidence sequence, two-sided alpha {alpha}, power target {1 - d0.beta}")
    for rho in (50, 100, 200, 400, 800, 1600):
        for N in range(d0.k, 20 * d0.n_fixed + 1, d0.k):
            n = np.arange(d0.k, N + 1, d0.k).astype(float)
            I = n / s ** 2
            z = np.sqrt((1 + rho / n) * (np.log1p(n / rho) + 2 * math.log(1 / alpha)))
            up, dn = exit_probs(I, -z, z, e1)
            if up.sum() >= 1 - d0.beta:
                def EN(eta):
                    u, dd = exit_probs(I, -z, z, eta)
                    st = u + dd
                    return float(st @ n + (1 - st.sum()) * N)
                u0, d0n = exit_probs(I, -z, z, 0.0)
                print(f"  rho {rho:5d}: n_max {N:6d}, final |z| {z[-1]:.2f}, power {up.sum():.3f}, alpha "
                      f"{u0.sum() + d0n.sum():.4f}, E[pairs] at theta1 {EN(e1):.0f}, at theta0 {EN(0.0):.0f}")
                break


def explore(name: str, reps: int) -> None:
    """Spending-function / look-interval scan (numerical OC): n_max, E[pairs] at theta0, midpoint, theta1."""
    base = preset(name)
    print(f"{name}: theta0 {base.theta0:.4f} theta1 {base.theta1:.4f} alpha {base.alpha} beta {base.beta}; "
          f"fixed-sample n for (alpha, beta): {((norm.isf(base.alpha) + norm.isf(base.beta)) * base.sigma / (base.theta1 - base.theta0)) ** 2:.0f}")
    print("  alpha_sf      beta_sf        k  n_max  E[n|t0] E[n|mid] E[n|t1] E[n|t0-d] power  alpha")
    for asf in (["hsd", -4.0], ["hsd", -2.0], ["hsd", 0.0], ["obf"], ["pocock"]):
        for bsf in (["hsd", -4.0], ["hsd", -2.0], ["hsd", 0.0], ["hsd", 1.0]):
            for k in (base.k // 2, base.k, base.k * 2):
                d = solve(Design(**{**asdict(preset(name)), "alpha_sf": asf, "beta_sf": bsf, "k": k}))
                dd = d.theta1 - d.theta0
                e = [oc_numeric(d, th) for th in (d.theta0, d.theta0 + dd / 2, d.theta1, d.theta0 - dd)]
                print(f"  {str(asf):13s} {str(bsf):13s} {k:4d} {d.n_max:6d} {e[0]['E_pairs']:8.0f} {e[1]['E_pairs']:8.0f}"
                      f" {e[2]['E_pairs']:8.0f} {e[3]['E_pairs']:9.0f} {e[2]['p_success']:6.3f} {e[0]['p_success']:6.4f}")


if __name__ == "__main__":
    sys.exit(main())
