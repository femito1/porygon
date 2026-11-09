"""Network + search, phase-3 prototype (design and pre-registered test plan: .orchestra/fastenv/NETSEARCH-DESIGN.md).

The margin rule (MissingNo.Zero: "only trust search when it beats the network's pick by a clear margin"):
  1. the raw v2 network samples its move a_net at temperature 0.5 (the benchmarked raw player);
  2. our search runs unchanged (porygon.search: `samples` determinizations x `ms`, pure engine MCTS, stock eval
     in gen9), independently of the network;
  3. the search's pick a_s is what pure search would play: the most visits (mean share over determinizations);
  4. play a_s instead of a_net only if  D = Q(a_s) - Q(a_net) > margin,  where Q(a) is the mean over
     determinizations of the root statistic total_score / visits of action a: the engine's leaf score in [0, 1]
     (sigmoid(0.0125 * (eval(leaf) - eval(root))), so 0.5 = no change from the root's evaluation).
margin = +inf is the raw network, margin = -inf is pure search, so the rule interpolates between our two benchmarked
players. Every searched decision's statistics go to `on_gate` (logging, offline studies).

Variants at the same budget: c_puct > 0 puts the network's T-1 probabilities into the search as PUCT priors at our
root (patched engine); prior_weight > 0 makes the search's pick argmax(share + prior_weight * p_net), as
porygon.search.combine (margin -inf then = netv2's "search:NET,prior_weight=w" player).

Needs PORYGON_GEN set before import (porygon.search reads it) and netv2's `porygon.v2` importable (its cluster copy
appended to `porygon.__path__`, never edited; see scripts/fastenv/netsearch_study.py).
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder

from porygon import search as S

log = logging.getLogger("porygon.netsearch")


# ---------------------------------------------------------------- root statistics and the gate (pure functions)

@dataclass
class RootStats:
    """Our root statistics of one decision's searches, by poke-env action index: n[k, a] visits and w[k, a] total
    score (sum of leaf scores in [0, 1]) in determinization k; ok[k] = that search returned usable options."""
    n: np.ndarray
    w: np.ndarray
    ok: np.ndarray
    errors: list[str] = field(default_factory=list)

    @property
    def q_k(self) -> np.ndarray:
        """[K, A] Q of each action in each determinization (NaN: not an option / not visited / failed search)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            q = self.w / self.n
        return np.where((self.n > 0) & self.ok[:, None], q, np.nan)

    @property
    def q(self) -> np.ndarray:
        """[A] mean Q over the determinizations that visited the action (NaN where none did)."""
        qk = self.q_k
        cnt = np.isfinite(qk).sum(0)
        tot = np.where(np.isfinite(qk), qk, 0.0).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    @property
    def share(self) -> np.ndarray:
        """[A] mean visit share over the successful determinizations (porygon.search's mcts_shares)."""
        n = self.n[self.ok]
        if not len(n):
            return np.zeros(self.n.shape[1])
        return (n / np.maximum(n.sum(1, keepdims=True), 1)).mean(0)

    @property
    def root(self) -> float:
        """Mean root value sum(w) / sum(n) over the successful determinizations (our side, [0, 1])."""
        n, w = self.n[self.ok], self.w[self.ok]
        return float(np.mean(w.sum(1) / np.maximum(n.sum(1), 1))) if len(n) else math.nan

    @property
    def visits(self) -> float:
        """Mean root visits per successful determinization."""
        n = self.n[self.ok]
        return float(n.sum(1).mean()) if len(n) else 0.0


def root_stats(convs: list, results: list, n_actions: int) -> RootStats:
    """RootStats from `EnginePool.search` results ([(engine choice, visits, total score)] or an exception per
    determinization) and the determinizations' choice maps (`porygon.search.Converted.action`). A search whose
    options cannot all be mapped is dropped, like porygon.search's `_sample_shares`."""
    k = len(convs)
    n, w, ok, errors = np.zeros((k, n_actions)), np.zeros((k, n_actions)), np.zeros(k, bool), []
    for i, (conv, options) in enumerate(zip(convs, results)):
        if isinstance(options, BaseException):
            errors.append(f"{type(options).__name__}: {options}")
            continue
        try:
            for choice, visits, total in options:
                if choice == "No Move":  # the engine thinks we cannot act: no information
                    continue
                a = conv.action(choice)
                if a is None:
                    raise ValueError(f"unmappable engine choice {choice!r}")
                n[i, a] += visits
                w[i, a] += total
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            n[i], w[i] = 0, 0
            continue
        ok[i] = n[i].sum() > 0
    return RootStats(n, w, ok, errors)


def sharpen(probs: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(logits / T) from softmax(logits): p^(1/T), renormalised (0 stays 0)."""
    if temperature == 1.0:
        return probs
    with np.errstate(divide="ignore"):
        z = np.where(probs > 0, np.log(np.where(probs > 0, probs, 1.0)) / temperature, -np.inf)
    if not np.isfinite(z).any():
        return probs
    e = np.exp(z - z[np.isfinite(z)].max())
    return e / e.sum()


def search_pick(stats: RootStats, mask: np.ndarray, probs: np.ndarray | None = None,
                prior_weight: float = 0.0) -> int:
    """The search's own move: argmax over legal actions of mean share (+ prior_weight * p_net), first on ties."""
    score = stats.share + (prior_weight * probs if prior_weight and probs is not None else 0.0)
    return int(np.argmax(np.where(mask > 0, score, -np.inf)))


@dataclass
class Gate:
    action: int  # the move to play
    net: int  # the network's move
    search: int  # the search's pick
    gap: float  # Q(search) - Q(net); NaN if either is undefined
    override: bool  # played the search's pick instead of the network's


def gate(a_net: int, stats: RootStats, mask: np.ndarray, margin: float, probs: np.ndarray | None = None,
         prior_weight: float = 0.0) -> Gate:
    """The margin rule. margin = -inf always plays the search's pick (pure search); a finite margin keeps the
    network's move when its Q is undefined (not an engine option in any determinization)."""
    a_s = search_pick(stats, mask, probs, prior_weight)
    q = stats.q
    gap = float(q[a_s] - q[a_net]) if np.isfinite(q[a_s]) and np.isfinite(q[a_net]) else math.nan
    override = a_s != a_net and (margin == -math.inf or (math.isfinite(gap) and gap > margin))
    return Gate(a_s if override else a_net, a_net, a_s, gap, bool(override))


def gap_t(stats: RootStats, a: int, b: int) -> tuple[float, float, int]:
    """(mean, standard error, count) over determinizations of Q_k(a) - Q_k(b), where both are defined: the
    per-position noise of the gap (offline studies; the pre-registered rule uses a fixed margin)."""
    qk = stats.q_k
    d = qk[:, a] - qk[:, b]
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return math.nan, math.nan, 0
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else math.nan
    return float(d.mean()), se, len(d)


# ---------------------------------------------------------------- our search, returning root statistics

class WeatherSeen:
    """porygon.search.SearchPlayer's per-battle weather start turn, for code that is not a SearchPlayer (offline
    studies replaying logged battles): call `first_seen(battle)` at every decision with more than one legal action,
    in order, as the live player does."""

    def __init__(self) -> None:
        self.weather_seen: dict[str, tuple[Any, int]] = {}

    def first_seen(self, battle: Battle) -> int | None:
        return S.SearchPlayer._weather_first_seen(self, battle)  # type: ignore[arg-type]  # uses only weather_seen


async def search_root(pool: Any, battle: Battle, rng: Any, first_seen: int | None, *, samples: int, ms: int,
                      probs: np.ndarray | None = None, mask: np.ndarray | None = None, c_puct: float = 0.0,
                      rm_gamma: float = 0.0, eval_config: dict | None = None) -> tuple[RootStats, list]:
    """Our search on `samples` fresh determinizations exactly as porygon.search runs it (same conversion, engine
    pool and request options; c_puct > 0 needs probs + mask), returning the root statistics and the
    determinizations. eval_config None = the stock evaluation (gen9's DEFAULT_EVAL)."""
    hist = S.History(battle)
    convs = [S.convert(battle, rng, first_seen, hist) for _ in range(samples)]
    kw: dict[str, Any] = {"rm_gamma": rm_gamma} if rm_gamma > 0 else {}
    kw |= {"eval_config": eval_config} if eval_config is not None else {}
    if c_puct > 0:
        assert probs is not None and mask is not None, "PUCT needs the network's probabilities and the mask"
        calls = [pool.search(c.state_str, ms, c.priors(probs, mask), c_puct, **kw) for c in convs]
    elif kw:
        calls = [pool.search(c.state_str, ms, None, 0.0, **kw) for c in convs]
    else:
        calls = [pool.search(c.state_str, ms) for c in convs]
    results = await asyncio.gather(*calls, return_exceptions=True)
    return root_stats(convs, results, S.N_ACTIONS), convs


# ---------------------------------------------------------------- the v2 network

def net_forward(net: Any, battle: AbstractBattle) -> tuple[np.ndarray, np.ndarray, float]:
    """(T-1 probabilities [A] over poke-env actions, legal mask [A], P(win) from the value head) for a v2 Net:
    the same encode / forward / float64 softmax as porygon.v2.player.policy_probs, plus the value."""
    from porygon.v2.encoder import encode
    from porygon.v2.model import obs_to_tensors
    obs = encode(battle)
    with torch.no_grad():
        out = net(obs_to_tensors([obs], next(net.parameters()).device), aux=False)
    mask = np.asarray(obs["action_mask"])
    logits = out["logits"][0].double().cpu().numpy()
    probs = np.zeros(len(mask), dtype=np.float64)
    legal = mask > 0
    if legal.any():
        z = logits[legal] - logits[legal].max()
        e = np.exp(z)
        probs[legal] = e / e.sum()
    win = float(torch.sigmoid(out["win_logit"][0].double()).item())
    return probs, mask, win


# ---------------------------------------------------------------- player

class NetSearchPlayer(S.SearchPlayer):
    """The margin rule as a poke-env player (module docstring). `net`: a v2 Net (porygon.v2.model.load) of
    PORYGON_GEN's generation. margin: +inf = raw network, -inf = pure search. temperature: the network's sampling
    temperature (0.5 = how raw nets are benchmarked); deterministic = its argmax. c_puct / prior_weight: the
    variants. The search options (ms, samples, rm_gamma, eval_config, workers, engine_python, seed) are
    porygon.search's; eval_config defaults to its DEFAULT_EVAL (gen9: stock, as the benchmark search).
    own_pool (default): this player's `workers` engine processes are its own. porygon.search.engine_pool is one
    process-wide pool per (workers, engine_python), so without it a SearchPlayer opponent with the same workers in
    the same harness process would share them (both sides queue on one set of processes)."""

    def __init__(self, net: Any, *args: Any, margin: float = 0.03, temperature: float = 0.5,
                 deterministic: bool = False, c_puct: float = 0.0, prior_weight: float = 0.0,
                 seed: int | None = None, own_pool: bool = True, **kwargs: Any):
        super().__init__(None, *args, prior_weight=0.0, c_puct=0.0, seed=seed, **kwargs)
        if own_pool:  # the registry pool stays for others (its processes start on first use, so none run here)
            self.pool = S.EnginePool(self.pool.n, python=self.pool.python)
        assert int(net.gen) == S.GEN_NUM, f"the v2 net is gen{net.gen} but PORYGON_GEN={S.GEN_NUM}"
        self.net = net.eval()
        self.margin, self.temperature, self.deterministic = float(margin), float(temperature), deterministic
        self.c_puct, self.net_prior_weight = float(c_puct), float(prior_weight)
        self.np_rng = np.random.default_rng(seed)
        self.stats.update(overrides=0, gaps_undefined=0)

    def net_move(self, probs: np.ndarray) -> int:
        p = sharpen(probs, self.temperature)
        if self.deterministic:
            return int(np.argmax(p))
        return int(self.np_rng.choice(len(p), p=p / p.sum()))

    async def root_search(self, battle: Battle, probs: np.ndarray, mask: np.ndarray) -> RootStats:
        """`search_root` with this player's options and engine pool; raises when the search is unusable."""
        stats, convs = await search_root(self.pool, battle, self.rng, self._weather_first_seen(battle),
                                         samples=self.samples, ms=self.ms, probs=probs, mask=mask,
                                         c_puct=self.c_puct, rm_gamma=self.rm_gamma, eval_config=self.eval_config)
        self.last_convs[battle.battle_tag] = convs
        self.stats["failed_samples"] += int((~stats.ok).sum())
        if not stats.ok.any():
            raise RuntimeError(f"every determinization failed: {stats.errors[:1]}")
        if stats.share[mask > 0].sum() < 0.5:  # the engine searched mostly illegal options -> desync
            raise ValueError(f"engine options disagree with the request: {stats.share} vs mask {mask}")
        return stats

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        assert isinstance(battle, Battle)
        probs, mask, win = net_forward(self.net, battle)
        if not (mask > 0).any():
            return self.choose_random_move(battle)
        a_net = action = self.net_move(probs)
        self.stats["decisions"] += 1
        if mask.sum() > 1 and self.margin != math.inf:
            try:
                stats = await self.root_search(battle, probs, mask)
                g = gate(a_net, stats, mask, self.margin, probs, self.net_prior_weight)
                action = self.last_action[battle.battle_tag] = g.action
                self.stats["searched"] += 1
                self.stats["agree"] += g.search == a_net
                self.stats["overrides"] += g.override
                self.stats["gaps_undefined"] += not math.isfinite(g.gap)
                self.on_gate(battle, probs, mask, win, stats, g)
            except Exception as e:
                self.stats["fallbacks"] += 1
                log.warning("search failed in %s turn %d, playing the network's move: %s", battle.battle_tag,
                            battle.turn, e)
        return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)

    def on_gate(self, battle: Battle, probs: np.ndarray, mask: np.ndarray, win: float, stats: RootStats,
                g: Gate) -> None:
        """Called after every searched decision (logging subclasses)."""


class LoggedNetSearchPlayer(NetSearchPlayer):
    """NetSearchPlayer that appends one JSON line per searched decision to `log_path` (the override analysis of a
    test: where the gate helped and hurt): battle tag, turn, legal mask, network probabilities and P(win), both
    moves, the gap, the override, the search's Q / shares over legal actions, root value and visits."""

    def __init__(self, net: Any, *args: Any, log_path: str, **kwargs: Any):
        super().__init__(net, *args, **kwargs)
        self.log_path = log_path

    def on_gate(self, battle: Battle, probs: np.ndarray, mask: np.ndarray, win: float, stats: RootStats,
                g: Gate) -> None:
        import json
        legal = np.flatnonzero(mask > 0)
        q = stats.q
        rec = {"battle": battle.battle_tag, "turn": battle.turn, "legal": legal.tolist(),
               "probs": [round(float(probs[a]), 5) for a in legal], "win": round(win, 5), "net": g.net,
               "search": g.search, "gap": None if not math.isfinite(g.gap) else round(g.gap, 5),
               "override": g.override, "action": g.action,
               "q": [None if not math.isfinite(q[a]) else round(float(q[a]), 5) for a in legal],
               "share": [round(float(stats.share[a]), 5) for a in legal], "root": round(stats.root, 5),
               "visits": round(stats.visits), "ok": int(stats.ok.sum())}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def load_spec(spec: str, net_loader: Any = None, **player_kwargs: Any) -> NetSearchPlayer:
    """A player from `netsearch:NET[,margin=M][,temperature=T][,deterministic=1][,c_puct=C][,prior_weight=W]
    [,ms=MS][,samples=K][,workers=N][,own_pool=0][,seed=S][,device=D][,log=PATH]` (margin: a number, inf or -inf;
    own_pool=0 shares porygon.search's process-wide engine pool, see NetSearchPlayer). For harnesses
    that build players from spec strings (netv2's porygon.v2.evaluate.load_player takes no such spec; wrap it)."""
    base, _, rest = spec.partition(",")
    kind, _, path = base.partition(":")
    assert kind == "netsearch" and path, f"not a netsearch spec: {spec}"
    opts = dict(kv.split("=", 1) for kv in rest.split(",") if kv)
    device = opts.pop("device", "cpu")
    if net_loader is None:
        from porygon.v2.model import load as net_loader
    net = net_loader(path, device=device)
    floats = {"margin", "temperature", "c_puct", "prior_weight"}
    ints = {"ms", "samples", "workers", "seed"}
    kw: dict[str, Any] = {}
    for k, v in opts.items():
        if k in floats:
            kw[k] = float(v)
        elif k in ints:
            kw[k] = int(v)
        elif k in ("deterministic", "own_pool"):
            kw[k] = v not in ("0", "false", "False")
        elif k != "log":
            raise ValueError(f"unknown netsearch option {k!r} in {spec}")
    if "log" in opts:
        return LoggedNetSearchPlayer(net, log_path=opts["log"], **kw, **player_kwargs)
    return NetSearchPlayer(net, **kw, **player_kwargs)
