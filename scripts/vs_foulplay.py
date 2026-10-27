"""Our network (raw or with search) vs Foul Play (pmariglia's search bot) on a local server.

Foul Play is GPL-3.0 and is NOT vendored here: run it from a separate clone with its own venv.

Setup (once):
    git clone https://github.com/pmariglia/foul-play.git /tmp/foul-play && cd /tmp/foul-play
    python3.12 -m venv .venv && .venv/bin/pip install requests websockets==14.1 python-dateutil==2.8.0
    .venv/bin/pip install -v --no-cache-dir poke-engine==0.0.48 \
        --config-settings="build-args=--features poke-engine/gen4 --no-default-features"
    Foul Play logs in through play.pokemonshowdown.com, whose assertion a local server rejects
    ("|nametaken|NAME|Your authentication token was invalid."). Patch PSWebsocketClient.login in
    fp/websocket_client.py: when there is no password and the URI is localhost, just send
    "/trn NAME,0," (the --no-security local server accepts it) and return the username.

Start one Foul Play instance per opponent name (each plays one battle at a time):
    cd /tmp/foul-play && .venv/bin/python run.py --websocket-uri ws://localhost:8000/showdown/websocket \
        --ps-username fp0 --bot-mode accept_challenge --pokemon-format gen4randombattle \
        --search-time-ms 1000 --run-count 100000 --log-level INFO > /tmp/fp0.log 2>&1 &
    (Foul Play searches 2x--search-parallelism states per decision, --search-parallelism processes at a time;
    default parallelism 1 = one core. It turns the battle timer on, so a stalled player loses on inactivity.)
Then:
    python scripts/vs_foulplay.py MODEL --player policy --opponents fp0 fp1 ... --n 10
    python scripts/vs_foulplay.py MODEL --player search --opponents fp0 fp1 --n 50 --workers 8
"""

import argparse
import asyncio
import json
import logging
import random
import time
from functools import partial

import numpy as np
from poke_env import AccountConfiguration
from poke_env.concurrency import handle_threaded_coroutines

from porygon.env import PolicyPlayer, local_server
from porygon.model import load_policy
from porygon.decisions import LoggedSearchPlayer
from porygon.search import CONFIG, SearchPlayer, add_value_args

log = logging.getLogger("vs_foulplay")


def guard_pool(pool, timeout: float) -> None:
    """poke-engine can spin forever on some states, which stalls the player until the server's inactivity
    timer forfeits the battle. Time out each search; EnginePool.search kills and replaces the worker."""
    if getattr(pool, "_guarded", False):
        return
    search = pool.search

    async def guarded(state_str: str, ms: int):
        try:
            return await asyncio.wait_for(search(state_str, ms), timeout + ms / 1000)
        except asyncio.TimeoutError:
            log.warning("poke-engine search exceeded %.0fs, worker replaced", timeout + ms / 1000)
            raise RuntimeError("poke-engine search timed out")

    pool.search, pool._guarded = guarded, True


async def send(player, message: str, room: str = "") -> None:
    await handle_threaded_coroutines(player.ps_client.send_message(message, room), player.ps_client.loop)


async def one_battle(player, opp: str, args) -> tuple[str, object]:
    """Challenge `opp` once; returns (status, battle). Re-challenges if Foul Play misses the challenge."""
    known = set(player.battles)
    for _ in range(args.retries):
        task = asyncio.ensure_future(player.send_challenges(opp, 1))
        deadline = time.time() + args.accept_timeout
        while time.time() < deadline and not (set(player.battles) - known) and not task.done():
            await asyncio.sleep(0.5)
        if set(player.battles) - known:
            break
        task.cancel()
        await send(player, f"/cancelchallenge {opp}")
        log.warning("%s did not accept the challenge, retrying", opp)
        await asyncio.sleep(3)
    else:
        return "no_accept", None
    battle = player.battles[next(iter(set(player.battles) - known))]
    try:
        await asyncio.wait_for(asyncio.shield(task), args.battle_timeout)
    except asyncio.TimeoutError:
        log.warning("%s vs %s timed out, forfeiting", battle.battle_tag, opp)
        await send(player, "/forfeit", battle.battle_tag)
        try:
            await asyncio.wait_for(task, 60)
        except asyncio.TimeoutError:
            task.cancel()
        return "timeout", battle
    return ("win" if battle.won else "tie" if battle.won is None else "loss"), battle


async def series(player, opp: str, args, out) -> dict[str, int]:
    counts = {"win": 0, "loss": 0, "tie": 0, "timeout": 0, "no_accept": 0}
    for i in range(args.n):
        await asyncio.sleep(3)  # let Foul Play leave the last battle room and go back to accepting
        status, battle = await one_battle(player, opp, args)
        counts[status] += 1
        rec = {"opponent": opp, "player": player.username, "status": status,
               "battle": battle and battle.battle_tag, "turns": battle and battle.turn}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"[{opp} {i + 1}/{args.n}] {status} {rec['battle']} ({rec['turns']} turns)", flush=True)
        if status == "no_accept":
            log.error("%s is not accepting challenges, giving up on it", opp)
            break
    return counts


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--player", choices=["policy", "search"], default="policy")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--opponents", nargs="+", required=True, help="usernames of running Foul Play instances")
    p.add_argument("--n", type=int, default=1, help="battles per opponent")
    p.add_argument("--ms", type=int, default=1000, help="SearchPlayer: MCTS ms per determinization")
    p.add_argument("--samples", type=int, default=4, help="SearchPlayer: determinizations per decision")
    p.add_argument("--prior-weight", type=float, default=0.25)
    p.add_argument("--top-k", type=int, default=None)
    add_value_args(p)
    p.add_argument("--workers", type=int, default=None, help="SearchPlayer engine processes (default cores-1)")
    p.add_argument("--search-timeout", type=float, default=30, help="seconds beyond --ms before a search is killed")
    p.add_argument("--accept-timeout", type=float, default=45)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--battle-timeout", type=float, default=1800)
    p.add_argument("--results", default="foulplay_results.jsonl", help="per-battle JSONL (appended)")
    p.add_argument("--log-dir", help="search: record decisions and replays here (porygon.decisions)")
    p.add_argument("--format", default="gen4randombattle")
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")

    policy = load_policy(a.model, a.device)
    tag = random.randint(0, 10**5)
    common = dict(battle_format=a.format, server_configuration=local_server(a.port),
                  max_concurrent_battles=1, log_level=40)
    cfg = {k: getattr(a, k) for k in CONFIG}
    players = []
    for i, _ in enumerate(a.opponents):
        acc = AccountConfiguration(f"{a.player[:3]}{tag}x{i}", None)
        search = (partial(LoggedSearchPlayer, out=a.log_dir) if a.log_dir else SearchPlayer)
        players.append(search(policy, **cfg, workers=a.workers, account_configuration=acc, **common)
                       if a.player == "search" else PolicyPlayer(policy, account_configuration=acc, **common))
    for pl in players:
        if a.player == "search":
            guard_pool(pl.pool, a.search_timeout)
    desc = (f"SearchPlayer({', '.join(f'{k}={v}' for k, v in cfg.items())})"
            if a.player == "search" else "PolicyPlayer")
    start = time.time()
    with open(a.results, "a") as out:
        results = await asyncio.gather(*(series(pl, o, a, out) for pl, o in zip(players, a.opponents)))
    tot = {k: sum(r[k] for r in results) for k in results[0]}
    n = tot["win"] + tot["loss"] + tot["tie"]
    se = np.sqrt(max(tot["win"], 1) * max(n - tot["win"], 1) / max(n, 1) ** 3)
    print(f"{desc} vs Foul Play: won {tot['win']}/{n} = {tot['win'] / max(n, 1):.1%} (±{1.96 * se:.1%})  "
          f"losses {tot['loss']}, ties {tot['tie']}, timeouts {tot['timeout']}, "
          f"not accepted {tot['no_accept']}, {time.time() - start:.0f}s")
    if a.player == "search":
        s = {k: sum(pl.stats[k] for pl in players) for k in players[0].stats}
        print(f"decisions {s['decisions']}, searched {s['searched']}, fallbacks {s['fallbacks']}, "
              f"failed samples {s['failed_samples']}, same as network argmax {s['agree'] / max(s['searched'], 1):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
