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
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from poke_env import AccountConfiguration
from poke_env.concurrency import handle_threaded_coroutines

from porygon.env import PolicyPlayer, local_server
from porygon import inference
from porygon.decisions import LoggedSearchPlayer
from porygon.search import CONFIG, SearchPlayer, add_oracle_args, add_value_args, load_model

log = logging.getLogger("vs_foulplay")
TIMINGS: list[dict] = []  # per battle: pre-game wait, challenge / start / end times, retries
# per-battle records are written by one thread: a write + flush on the slow shared disk can block for 15-30 s, which
# on the event loop stalled every series (no challenge goes out meanwhile); one thread keeps the file in order
WRITER = ThreadPoolExecutor(max_workers=1)


def write_line(out, line: str) -> None:
    out.write(line)
    out.flush()


def guard_pool(pool, timeout: float) -> None:
    """poke-engine can spin forever on some states, which stalls the player until the server's inactivity
    timer forfeits the battle. Time out each search; EnginePool.search kills and replaces the worker."""
    if getattr(pool, "_guarded", False):
        return
    search = pool.search

    async def guarded(state_str: str, ms: int, *args, **kwargs):
        try:
            return await asyncio.wait_for(search(state_str, ms, *args, **kwargs), timeout + ms / 1000)
        except asyncio.TimeoutError:
            log.warning("poke-engine search exceeded %.0fs, worker replaced", timeout + ms / 1000)
            raise RuntimeError("poke-engine search timed out")

    pool.search, pool._guarded = guarded, True


async def send(player, message: str, room: str = "") -> None:
    await handle_threaded_coroutines(player.ps_client.send_message(message, room), player.ps_client.loop)


def to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


ROOM_AUTH = re.compile(r"^[^a-z0-9]+")  # userdetails room keys carry the user's room-auth symbol (players: ☆)


def battle_rooms(details: dict) -> list[str] | None:
    """Battle rooms the server lists for a userdetails reply; None if the user is offline."""
    rooms = details.get("rooms")
    if not isinstance(rooms, dict):
        return None
    return [ROOM_AUTH.sub("", r) for r in rooms if ROOM_AUTH.sub("", r).startswith("battle-")]


class ServerQueries:
    """The Showdown server's view of other users (`/cmd userdetails NAME`, exempt from the chat throttle) for one
    poke-env player, plus a count of popups (rejected challenges arrive as popups). Hooks the player's websocket
    message handler, which runs on poke-env's own event-loop thread; this script's loop reads `details`."""

    def __init__(self, player):
        self.player = player
        self.details: dict[str, tuple[float, dict]] = {}  # userid -> (monotonic time received, reply)
        self.popups: list[str] = []
        self.checked = False  # the opponent's battle room was looked for once during a battle
        self.blind = False    # ... and the server did not list it: fixed sleeps for this opponent
        client = player.ps_client
        handle = client._handle_message

        async def hooked(message: str):
            if message.startswith("|queryresponse|userdetails|"):
                try:
                    d = json.loads(message.split("|", 3)[3])
                    if isinstance(d, dict):
                        self.details[to_id(str(d.get("userid") or d.get("id") or ""))] = (time.monotonic(), d)
                except ValueError:
                    pass
                return
            if message.startswith("|popup|"):
                self.popups.append(message[7:])
            await handle(message)
        client._handle_message = hooked

    async def userdetails(self, name: str, timeout: float = 5.0) -> dict | None:
        uid, t0 = to_id(name), time.monotonic()
        await send(self.player, f"/cmd userdetails {name}")
        while True:  # (the reply is looked for once more after the deadline: this loop may itself have been stalled)
            got = self.details.get(uid)
            if got and got[0] >= t0:
                return got[1]
            if time.monotonic() - t0 > timeout:
                return None
            await asyncio.sleep(0.02)


async def wait_ready(player, queries: ServerQueries, opp: str, args) -> tuple[str, float]:
    """Before challenging `opp`: wait until the server has it online and in no battle room. Foul Play leaves a finished
    battle with /leave and then reads (and drops) everything until that room's |deinit|; once the server has processed
    the /leave, the |deinit| is already queued on its connection ahead of our challenge, so the challenge reaches its
    accept loop instead of being swallowed. Returns (state, seconds waited): "ready"; "busy" (still in a battle room
    after --ready-timeout s) or "offline" (not connected for --offline-timeout s, e.g. restarting): challenge anyway
    and let the retries handle it, as before; "noreply" (no userdetails answer for --noreply-timeout s) or "blind" (the
    server did not list the opponent's battle room during a battle): the old fixed sleep.
    --pre-game-wait sleep: the old fixed --pre-game-sleep before every challenge."""
    t0 = time.monotonic()
    if args.pre_game_wait == "sleep" or queries.blind:
        await asyncio.sleep(args.pre_game_sleep)
        return "sleep" if args.pre_game_wait == "sleep" else "blind", time.monotonic() - t0
    warned = False
    while True:
        d = await queries.userdetails(opp, timeout=2.0)
        waited = time.monotonic() - t0
        if d is None:  # the server is busy (it froze 15-30 s on synchronous log writes to the shared disk): ask again
            if waited > args.noreply_timeout:  # never answers: the old fixed sleep (for good if it never answered)
                queries.blind = not queries.details
                await asyncio.sleep(args.pre_game_sleep)
                return "noreply", time.monotonic() - t0
            continue
        rooms = battle_rooms(d)
        if rooms is not None:
            if not rooms:
                return "ready", waited
            if waited > args.ready_timeout:
                log.warning("%s still in a battle room after %.0f s: challenging anyway", opp, waited)
                return "busy", waited
        else:
            if not warned:
                log.warning("%s is offline: waiting up to %.0f s for it to come back", opp, args.offline_timeout)
                warned = True
            if waited > args.offline_timeout:
                return "offline", waited
        await asyncio.sleep(0.1)


async def one_battle(player, queries: ServerQueries, opp: str, args, timing: dict) -> tuple[str, object]:
    """Challenge `opp` once; returns (status, battle). Re-challenges if Foul Play misses the challenge."""
    known = set(player.battles)
    for _ in range(args.retries):
        timing["t_challenge"] = time.time()
        task = asyncio.ensure_future(player.send_challenges(opp, 1))
        deadline = time.time() + args.accept_timeout
        while time.time() < deadline and not (set(player.battles) - known) and not task.done():
            await asyncio.sleep(args.start_poll)
        if set(player.battles) - known:
            break
        task.cancel()
        await send(player, f"/cancelchallenge {opp}")
        log.warning("%s did not accept the challenge, retrying", opp)
        timing["retries"] = timing.get("retries", 0) + 1
        state, waited = await wait_ready(player, queries, opp, args)
        timing["retry_wait"] = timing.get("retry_wait", 0.0) + waited
    else:
        return "no_accept", None
    timing["t_start"] = time.time()
    battle = player.battles[next(iter(set(player.battles) - known))]
    if args.pre_game_wait == "ready" and not queries.checked:
        # once per series: the server must list Foul Play's battle room while it plays, or "ready" would mean nothing
        d = await queries.userdetails(opp)
        queries.checked = d is not None  # no reply: look again in the next battle
        timing["fp_room_visible"] = bool(d and battle.battle_tag in (battle_rooms(d) or []))
        if d is not None and not timing["fp_room_visible"]:
            queries.blind = True  # fall back to the fixed sleep for this opponent
            log.warning("%s: userdetails does not list its battle room (%s): fixed --pre-game-sleep from now on", opp, d)
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
    finally:
        timing["t_end"] = time.time()
    return ("win" if battle.won else "tie" if battle.won is None else "loss"), battle


async def series(player, opp: str, args, out) -> dict[str, int]:
    counts = {"win": 0, "loss": 0, "tie": 0, "timeout": 0, "no_accept": 0}
    queries = ServerQueries(player)
    # the old fixed sleep also covered the player's own login; userdetails queries need the websocket up
    await handle_threaded_coroutines(player.ps_client.logged_in.wait(), player.ps_client.loop)
    for i in range(args.n):
        state, waited = await wait_ready(player, queries, opp, args)  # Foul Play has left its last battle room
        timing = {"pre_wait": waited, "pre_state": state}
        status, battle = await one_battle(player, queries, opp, args, timing)
        counts[status] += 1
        TIMINGS.append(timing)
        rec = {"opponent": opp, "player": player.username, "status": status,
               "battle": battle and battle.battle_tag, "turns": battle and battle.turn,
               **{k: round(v, 3) if isinstance(v, float) else v for k, v in timing.items()},
               "popups": len(queries.popups)}
        WRITER.submit(write_line, out, json.dumps(rec) + "\n")
        print(f"[{opp} {i + 1}/{args.n}] {status} {rec['battle']} ({rec['turns']} turns)", flush=True)
        if status == "no_accept":
            log.error("%s is not accepting challenges, giving up on it", opp)
            break
    if queries.popups:
        log.warning("%s: %d popups: %s", opp, len(queries.popups), queries.popups[:5])
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
    p.add_argument("--prior-weight", type=float, default=None, help="default 0.25; 0 with MODEL none")
    p.add_argument("--c-puct", type=float, default=0.0, help="SearchPlayer: PUCT on the network's priors at the root")
    p.add_argument("--root-game", choices=["max", "sample"], help="SearchPlayer: solve the Bayesian root game")
    p.add_argument("--rm-gamma", type=float, default=0.0, help="SearchPlayer: regret matching in the engine's MCTS")
    p.add_argument("--top-k", type=int, default=None)
    add_value_args(p)
    add_oracle_args(p)
    p.add_argument("--workers", type=int, default=None, help="SearchPlayer engine processes (default cores-1)")
    p.add_argument("--search-timeout", type=float, default=30, help="seconds beyond --ms before a search is killed")
    p.add_argument("--accept-timeout", type=float, default=45)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--pre-game-wait", choices=["ready", "sleep"], default="ready",
                   help="before each challenge: ready = until the server shows Foul Play online and out of its last "
                        "battle room (event-based); sleep = the old fixed --pre-game-sleep")
    p.add_argument("--pre-game-sleep", type=float, default=3.0, help="seconds (sleep mode, and the no-reply fallback)")
    p.add_argument("--ready-timeout", type=float, default=20, help="ready: challenge anyway after this long in a battle room")
    p.add_argument("--offline-timeout", type=float, default=600, help="ready: wait this long for an offline Foul Play")
    p.add_argument("--noreply-timeout", type=float, default=60, help="ready: fixed sleep if the server does not answer")
    p.add_argument("--start-poll", type=float, default=0.1, help="seconds between checks for the battle start (was 0.5)")
    p.add_argument("--battle-timeout", type=float, default=1800)
    p.add_argument("--results", default="foulplay_results.jsonl", help="per-battle JSONL (appended)")
    p.add_argument("--log-dir", help="search: record decisions and replays here (porygon.decisions)")
    p.add_argument("--format", default=inference.FORMAT)
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")

    policy = load_model(a.model, a.device)  # "none": pure engine search (gen9)
    if a.prior_weight is None:
        a.prior_weight = 0.0 if policy is None else 0.25
    tag = random.randint(0, 10**5)
    common = dict(battle_format=a.format, server_configuration=local_server(a.port),
                  max_concurrent_battles=1, log_level=40)
    cfg = {k: getattr(a, k) for k in CONFIG}
    players = []
    for i, _ in enumerate(a.opponents):
        acc = AccountConfiguration(f"{a.player[:3]}{tag}x{i}", None)
        search = (partial(LoggedSearchPlayer, out=a.log_dir) if a.log_dir else SearchPlayer)
        players.append(search(policy, **cfg, workers=a.workers, oracle=a.oracle, oracle_dir=a.oracle_dir,
                              account_configuration=acc, **common)
                       if a.player == "search" else PolicyPlayer(policy, account_configuration=acc, **common))
    for pl in players:
        if a.player == "search":
            guard_pool(pl.pool, a.search_timeout)
    desc = (f"SearchPlayer({', '.join(f'{k}={v}' for k, v in cfg.items())}, oracle={a.oracle})"
            if a.player == "search" else "PolicyPlayer")
    start = time.time()
    with open(a.results, "a") as out:
        try:
            results = await asyncio.gather(*(series(pl, o, a, out) for pl, o in zip(players, a.opponents)))
        finally:
            WRITER.shutdown(wait=True)  # every record is on disk before the file closes
    tot = {k: sum(r[k] for r in results) for k in results[0]}
    n = tot["win"] + tot["loss"] + tot["tie"]
    se = np.sqrt(max(tot["win"], 1) * max(n - tot["win"], 1) / max(n, 1) ** 3)
    print(f"{desc} vs Foul Play: won {tot['win']}/{n} = {tot['win'] / max(n, 1):.1%} (±{1.96 * se:.1%})  "
          f"losses {tot['loss']}, ties {tot['tie']}, timeouts {tot['timeout']}, "
          f"not accepted {tot['no_accept']}, {time.time() - start:.0f}s")
    w = [t["pre_wait"] for t in TIMINGS]
    acc = [t["t_start"] - t["t_challenge"] for t in TIMINGS if "t_start" in t]
    states = {k: sum(t["pre_state"] == k for t in TIMINGS) for k in {t["pre_state"] for t in TIMINGS}}
    if w:
        print(f"pre-game wait ({a.pre_game_wait}): mean {np.mean(w):.2f} s, max {np.max(w):.2f} s, {states}; "
              f"challenge to battle start: mean {np.mean(acc) if acc else float('nan'):.2f} s; "
              f"retries {sum(t.get('retries', 0) for t in TIMINGS)}")
    if a.player == "search":
        s = {k: sum(pl.stats[k] for pl in players) for k in players[0].stats}
        print(f"decisions {s['decisions']}, searched {s['searched']}, fallbacks {s['fallbacks']}, "
              f"failed samples {s['failed_samples']}, same as network argmax {s['agree'] / max(s['searched'], 1):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
