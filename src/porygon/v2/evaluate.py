"""Evaluation harness for raw networks (v1 SB3 checkpoints, v2 nets), search and baselines: duplicate-pair
head-to-heads with 95% intervals from pair means.

Player specs (`load_player`): BASE[,key=value...]
    runs/x/final_model.zip, pool/123.pt  v1 SB3 checkpoint / opponent-pool snapshot -> PolicyPlayer
    net.pt                               v2 checkpoint (dict with "cfg", "gen", "state_dict") -> v2 NetPlayer
        options: deterministic=1 (argmax; default 0 = sample, as training and slurm_h2h_nets play), temperature=T
        (sample from softmax(logits / T); netv2's evaluation default from 2026-11-08 is T = 0.5), device=cuda
    search:MODEL|none                    SearchPlayer (gen from PORYGON_GEN), MODEL = v1 or v2 checkpoint as the
                                         prior, none = pure engine search (prior_weight 0)
        options: any porygon.search CONFIG key (ms, samples, prior_weight, top_k, c_puct, root_game, rm_gamma,
        eval_config, engine_python, ...), workers, seed, device, log_dir (LoggedSearchPlayer: decisions/replays)
    search                               = search:none
    random | max_power | heuristic       poke-env baselines

CLI: A plays B on one server; we report A's win rate.
    python -m porygon.v2.evaluate A B --n-battles N --port P --format F --out DIR [--concurrency C]

Pairing: on the duplicate-battle server (`scripts/build_eval_server.sh`, started with PORYGON_DUPLICATE=<seed tag>,
PORYGON_DUPLICATE_LOG=DIR/battles.jsonl, PORYGON_DUPLICATE_GROUP='^pa'), consecutive battles k = 2i, 2i+1 on a
server form pair i: same battle seed, the two team seeds swapped. Only A and B play on that server, so every pair
is the same matchup with the teams swapped; A's username starts with "pa" so the group regex anchors team A on A.
A pair scores (s₀ + s₁)/2 (tie = ½), the interval is mean ± 1.96·sd/√pairs. Without the duplicate log (a plain
server) the interval is the per-game binomial one. A challenges in the first half, B in the second (p1/p2 swap).

Outputs in --out: results.jsonl (client side, one line per finished game: battle tag, both names and specs, A's
status, turns, A's side, seed tag), summary.json, and battles.jsonl (written by the server, if pointed there).
`scripts/netv2_pool_ci.py DIR...` pools runs.
"""

import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

BASELINE_NAMES = ("random", "max_power", "heuristic")
NAME_A, NAME_B = "pa", "pb"  # username prefixes of the two sides (the server's PORYGON_DUPLICATE_GROUP is '^pa')


def norm_id(name: str | None) -> str:
    """Showdown user id: lower case, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


# ---------------------------------------------------------------- specs and checkpoints


def parse_spec(spec: str) -> tuple[str, dict[str, str]]:
    """'BASE,k=v,k2=v2' -> (BASE, {k: v}); BASE is everything before the first comma."""
    base, *rest = spec.split(",")
    opts = {}
    for kv in rest:
        k, sep, v = kv.partition("=")
        if not sep:
            raise ValueError(f"bad option {kv!r} in spec {spec!r} (want key=value)")
        opts[k.strip().replace("-", "_")] = v.strip()
    return base.strip(), opts


def checkpoint_kind(path: str | Path) -> str:
    """'v1' (SB3 PPO .zip or MaskedPolicy snapshot .pt) or 'v2' (dict with 'cfg' / 'gen')."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".zip":
        return "v1"
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and ("cfg" in blob or "gen" in blob):
        return "v2"
    return "v1"  # SB3 BasePolicy.save: {"state_dict", "data"}


def is_v2_net(obj: Any) -> bool:
    """A loaded v2 Net (as opposed to a v1 SB3 ActorCriticPolicy or None)."""
    if obj is None:
        return False
    from stable_baselines3.common.policies import ActorCriticPolicy
    return not isinstance(obj, ActorCriticPolicy)


def load_net(path: str | Path, device: str = "cpu") -> Any:
    """v1 policy (eval mode) or v2 Net from a checkpoint path."""
    if checkpoint_kind(path) == "v2":
        from porygon.v2.model import load
        net = load(str(path), device=device)
        return net.eval() if hasattr(net, "eval") else net
    from porygon.model import load_policy
    return load_policy(path, device)


def format_gen(battle_format: str) -> int:
    m = re.match(r"gen(\d+)", battle_format)
    if not m:
        raise ValueError(f"no generation in battle format {battle_format!r}")
    return int(m.group(1))


def check_net_gen(net: Any, battle_format: str, what: str) -> None:
    gen = format_gen(battle_format)
    if is_v2_net(net):
        if int(net.gen) != gen:
            raise ValueError(f"{what}: v2 net is gen{net.gen}, the format is {battle_format}")
    else:
        from poke_env.environment import SinglesEnv
        n = int(net.action_space.n)
        if n != SinglesEnv.get_action_space_size(gen):
            raise ValueError(f"{what}: v1 policy has {n} actions, {battle_format} needs "
                             f"{SinglesEnv.get_action_space_size(gen)} (v1 nets are gen4)")


def _cast(key: str, value: str) -> Any:
    from porygon.search import CONFIG_TYPES
    if value.lower() in ("none", "null", ""):
        return None
    if key in ("workers", "seed", "ms", "samples", "top_k", "value_top_k"):
        return int(value)
    return CONFIG_TYPES.get(key, int)(value)


def _flag(v: str | bool) -> bool:
    return v if isinstance(v, bool) else v.lower() in ("1", "true", "yes", "on")


def load_player(spec: str, **player_kwargs: Any):
    """poke-env Player for a spec (module docstring). player_kwargs go to the Player (battle_format,
    server_configuration, account_configuration, max_concurrent_battles, ...); `device`, `deterministic` and
    `workers` given here are defaults that the spec's options override."""
    base, opts = parse_spec(spec)
    device = opts.pop("device", player_kwargs.pop("device", None)) or "cpu"
    deterministic = _flag(opts.pop("deterministic", opts.pop("det", player_kwargs.pop("deterministic", False))))
    temperature = float(opts.pop("temperature", opts.pop("t", player_kwargs.pop("temperature", 1.0))))
    workers = player_kwargs.pop("workers", None)
    fmt = player_kwargs.get("battle_format", "gen4randombattle")
    if base in BASELINE_NAMES:
        if opts:
            raise ValueError(f"baseline {base} takes no options: {opts}")
        from porygon.evaluate import BASELINES
        return BASELINES[base](**player_kwargs)
    if base == "search" or base.startswith("search:"):
        from porygon.search import CONFIG, SearchPlayer, load_model
        model = base.partition(":")[2] or "none"
        policy = load_model(model, device)
        cfg = {k: _cast(k, opts.pop(k)) for k in list(opts) if k in CONFIG}
        cfg.setdefault("prior_weight", 0.0 if policy is None else 0.25)
        extra = {"workers": _cast("workers", opts.pop("workers")) if "workers" in opts else workers}
        if "seed" in opts:
            extra["seed"] = int(opts.pop("seed"))
        log_dir = opts.pop("log_dir", None)
        if opts:
            raise ValueError(f"unknown search options {sorted(opts)} (known: {CONFIG}, workers, seed, device, log_dir)")
        if policy is not None:
            check_net_gen(policy, fmt, spec)
        if log_dir:
            from porygon.decisions import LoggedSearchPlayer
            return LoggedSearchPlayer(policy, **cfg, **extra, out=log_dir, **player_kwargs)
        return SearchPlayer(policy, **cfg, **extra, **player_kwargs)
    if opts:
        raise ValueError(f"unknown network options {sorted(opts)} (known: deterministic, temperature, device)")
    net = load_net(base, device)
    check_net_gen(net, fmt, spec)
    if is_v2_net(net):
        from porygon.v2.player import NetPlayer
        return NetPlayer(net, deterministic=deterministic, temperature=temperature, **player_kwargs)
    from porygon.env import PolicyPlayer
    return PolicyPlayer(net, deterministic=deterministic, temperature=temperature, **player_kwargs)


# ---------------------------------------------------------------- scoring


def game_score(winner: str | None, ours: set[str]) -> float:
    """1 if one of `ours` (user ids) won, 0 if the other side did, 0.5 for a tie."""
    return 0.5 if not winner else float(norm_id(winner) in ours)


def pair_means(battles: list[dict], ours: set[str], key: Any = None) -> tuple[list[float], list[float], int]:
    """Duplicate-server log rows -> (pair means, every game's score, incomplete pairs). Only rows where one of
    `ours` played count; pairs are keyed by (key, pair id) so pairs from different logs never merge."""
    by_pair: dict[Any, dict[int, dict]] = defaultdict(dict)
    games = []
    for b in battles:
        if not ({norm_id(b.get("p1")), norm_id(b.get("p2"))} & ours):
            continue
        by_pair[(key, b["pair"])][b["game"]] = b
        games.append(game_score(b.get("winner"), ours))
    means = [(game_score(p[0].get("winner"), ours) + game_score(p[1].get("winner"), ours)) / 2
             for p in by_pair.values() if len(p) == 2 and 0 in p and 1 in p]
    return means, games, sum(len(p) != 2 for p in by_pair.values())


def interval(xs: list[float]) -> tuple[float, float]:
    """(mean, 95% half-width = 1.96 sd / √n); half-width nan with fewer than 2 values."""
    n = len(xs)
    if not n:
        return math.nan, math.nan
    m = sum(xs) / n
    if n < 2:
        return m, math.nan
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def summarize(means: list[float], games: list[float]) -> dict[str, Any]:
    """Win rate by pair means (if any) and per game, with 95% half-widths."""
    g = sum(games) / len(games) if games else math.nan
    g_ci = 1.96 * math.sqrt(g * (1 - g) / len(games)) if games else math.nan
    out = {"games": len(games), "win_rate_games": g, "ci95_games": g_ci, "pairs": len(means)}
    if means:
        m, ci = interval(means)
        out |= {"win_rate": m, "ci95": ci, "won_both": sum(x == 1 for x in means),
                "lost_both": sum(x == 0 for x in means)}
        out["split"] = len(means) - out["won_both"] - out["lost_both"]
    else:
        out |= {"win_rate": g, "ci95": g_ci}
    return out


def format_summary(s: dict[str, Any]) -> str:
    line = f"{s['win_rate']:.1%} ± {s['ci95']:.1%}"
    if s["pairs"]:
        line += (f" ({s['pairs']} duplicate pairs, {s['games']} games; won both {s['won_both']}, split {s['split']}, "
                 f"lost both {s['lost_both']}; per-game {s['win_rate_games']:.1%} ± {s['ci95_games']:.1%})")
    else:
        line += f" ({s['games']} games, binomial: no duplicate-server log)"
    return line


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.open():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut by a killed job
                pass
    return rows


# ---------------------------------------------------------------- CLI


async def play(args: argparse.Namespace) -> dict[str, Any]:
    from poke_env import AccountConfiguration
    from porygon.env import local_server

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.name_tag or f"{random.randint(0, 10**7):07d}"
    names = (f"{NAME_A}{tag}", f"{NAME_B}{tag}")
    group = os.environ.get("PORYGON_DUPLICATE_GROUP")
    if group and not (re.search(group, names[0]) and not re.search(group, names[1])):
        raise SystemExit(f"PORYGON_DUPLICATE_GROUP={group!r} must match A ({names[0]}) and not B ({names[1]})")
    common = dict(battle_format=args.format, server_configuration=local_server(args.port),
                  max_concurrent_battles=args.concurrency, log_level=40, device=args.device)
    if args.workers is not None:
        common["workers"] = args.workers
    a = load_player(args.a, account_configuration=AccountConfiguration(names[0], None), **common)
    b = load_player(args.b, account_configuration=AccountConfiguration(names[1], None), **common)
    seed_tag = os.environ.get("PORYGON_DUPLICATE")
    results = open(out / "results.jsonl", "a")
    counts = {"win": 0, "loss": 0, "tie": 0}
    finished = a._battle_finished_callback

    def record(battle) -> None:  # A's view of every finished game
        status = "win" if battle.won else "tie" if battle.won is None else "loss"
        counts[status] += 1
        results.write(json.dumps({"battle": battle.battle_tag, "a": names[0], "b": names[1], "spec_a": args.a,
                                  "spec_b": args.b, "status": status, "turns": battle.turn,
                                  "side": battle.player_role, "seed_tag": seed_tag}) + "\n")
        results.flush()
        n = sum(counts.values())
        if n % args.print_every == 0:
            print(f"[{n}/{args.n_battles}] A {counts['win']}-{counts['loss']}-{counts['tie']}", flush=True)
        finished(battle)

    a._battle_finished_callback = record
    n = args.n_battles + args.n_battles % 2  # whole pairs
    half = n // 2
    start = time.time()
    await a.battle_against(b, n_battles=n - half)  # A is p1 (challenger) ...
    await b.battle_against(a, n_battles=half)  # ... then p2
    results.close()
    elapsed = time.time() - start

    dup_log = Path(args.duplicate_log or os.environ.get("PORYGON_DUPLICATE_LOG") or out / "battles.jsonl")
    ours = {norm_id(names[0])}
    battles = [r for r in load_jsonl(dup_log) if {norm_id(r.get("p1")), norm_id(r.get("p2"))} == set(map(norm_id, names))]
    if battles:
        means, games, incomplete = pair_means(battles, ours, str(dup_log))
    else:
        games = [{"win": 1.0, "loss": 0.0, "tie": 0.5}[r["status"]] for r in load_jsonl(out / "results.jsonl")
                 if r["a"] == names[0]]
        means, incomplete = [], 0
    s = summarize(means, games) | {"a": args.a, "b": args.b, "name_a": names[0], "name_b": names[1],
                                   "format": args.format, "seed_tag": seed_tag, "duplicate_log": str(dup_log),
                                   "incomplete_pairs": incomplete, "seconds": round(elapsed),
                                   "client_counts": counts}
    for name, player in (("a", a), ("b", b)):
        if hasattr(player, "stats"):  # SearchPlayer
            s[f"search_stats_{name}"] = dict(player.stats)
    (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(f"{args.a} vs {args.b} [{args.format}]: {format_summary(s)}, {elapsed:.0f}s", flush=True)
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("a", help="player spec whose win rate is reported")
    p.add_argument("b", help="opponent spec")
    p.add_argument("--n-battles", type=int, default=100, help="rounded up to whole pairs")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--format", default=None, help="default gen{PORYGON_GEN}randombattle")
    p.add_argument("--out", required=True, help="results.jsonl / summary.json go here")
    p.add_argument("--concurrency", type=int, default=8, help="simultaneous battles")
    p.add_argument("--device", default="cpu", help="for networks (spec option device= overrides)")
    p.add_argument("--workers", type=int, default=None, help="engine processes for search specs (default cores-1)")
    p.add_argument("--duplicate-log", help="the server's battles.jsonl (default $PORYGON_DUPLICATE_LOG, else OUT/battles.jsonl)")
    p.add_argument("--name-tag", help="username suffix (default random); A = pa<tag>, B = pb<tag>")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--print-every", type=int, default=50)
    args = p.parse_args()
    if args.format is None:
        args.format = f"gen{os.environ.get('PORYGON_GEN', '4')}randombattle"
    import asyncio
    import logging
    import torch
    torch.set_num_threads(args.torch_threads)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(play(args))


if __name__ == "__main__":
    main()
