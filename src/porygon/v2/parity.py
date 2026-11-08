"""Parity harness for observation spec v2: fixtures of real battles (Showdown protocol chunks as each player
received them) with the reference encoder's observations and aux targets at every decision of player 1.

Another implementation (fastenv, data) passes when, for every fixture decision, its observation from the same
protocol stream matches the reference field by field (`compare`), or when the reference re-encodes the protocol
stream it emitted and matches its observations (`check --candidate`).

    python -m porygon.v2.parity record --format gen9randombattle --n 30 --out fixtures/gen9 --port 8000
    python -m porygon.v2.parity check fixtures/gen9                      # reference vs stored (regression)
    python -m porygon.v2.parity check fixtures/gen9 --candidate obs.npz  # another implementation's arrays

Fixture directory: `battles.jsonl.gz` (one battle per line: format, the two usernames, both players' chunk
lists; each chunk is poke-env's split message list, i.e. the lines of one websocket message split on "|") and
`decisions.npz` (`battle`, `k1`, `k2`: decision d of battle b replays p1's first k1 and p2's first k2 chunks;
`obs/<field>` and `aux/<field>` stacked over decisions; `spec` = SPEC_VERSION).
"""

import argparse
import asyncio
import gzip
import json
import random
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer

from porygon.env import local_server
from porygon.v2.encoder import aux_targets, encode
from porygon.v2.spec import SPEC_VERSION, aux_fields, fields

TOLERANCE = 1e-4


# ---------------------------------------------------------------- replay

async def _silent(*args, **kwargs) -> None:
    return None


class ReplayPlayer(Player):
    """Rebuilds poke-env battles from recorded chunks with exactly the live parsing code."""

    def __init__(self, username: str, battle_format: str):
        super().__init__(account_configuration=AccountConfiguration(username, None), battle_format=battle_format,
                         start_listening=False, log_level=50)
        self.ps_client.send_message = _silent  # never connected: drop /leave etc.

    def choose_move(self, battle):
        return self.choose_default_move()

    async def _handle_battle_request(self, battle, maybe_default_order: bool = False):
        return None

    def _battle_finished_callback(self, battle) -> None:
        pass


async def _replay(chunks: list[list[list[str]]], username: str, battle_format: str) -> AbstractBattle:
    p = ReplayPlayer(username, battle_format)
    for chunk in chunks:
        await p._handle_battle_message(chunk)
    return next(iter(p.battles.values()))


def replay(chunks: list[list[list[str]]], username: str, battle_format: str) -> AbstractBattle:
    """The poke-env battle after `chunks` (the first k recorded chunks of one player)."""
    return asyncio.run(_replay(chunks, username, battle_format))


# ---------------------------------------------------------------- recording

class Recorder:
    """Mixin: keeps every battle chunk this player receives; p1 also snapshots obs / aux at each decision."""

    def setup_recorder(self, store: dict, opponent: "Recorder | None") -> None:
        self.store, self.opponent = store, opponent

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0]
        self.store.setdefault(tag, {"p1": [], "p2": [], "decisions": []})
        self.store[tag][self.side].append(split_messages)
        await super()._handle_battle_message(split_messages)

    def choose_move(self, battle):
        if self.side == "p1" and not battle.wait:
            tag = f">{battle.battle_tag}"
            rec = self.store[tag]
            opp_battle = next((b for t, b in self.opponent.battles.items() if t == battle.battle_tag), None)
            obs = encode(battle)
            aux = aux_targets(battle, opp_battle, obs)
            rec["decisions"].append((len(rec["p1"]), len(rec["p2"]), obs, aux))
        return super().choose_move(battle)


class RecHeuristic(Recorder, SimpleHeuristicsPlayer):
    side = "p1"


class RecRandom(Recorder, RandomPlayer):
    side = "p2"


class RecHeuristic2(Recorder, SimpleHeuristicsPlayer):
    side = "p2"


async def _record(battle_format: str, n: int, port: int) -> tuple[dict, str, str]:
    store: dict = {}
    tag = uuid.uuid4().hex[:6]
    common = dict(battle_format=battle_format, server_configuration=local_server(port), log_level=40,
                  max_concurrent_battles=1)
    p1 = RecHeuristic(account_configuration=AccountConfiguration(f"par1{tag}", None), **common)
    p2s = [RecHeuristic2(account_configuration=AccountConfiguration(f"par2{tag}", None), **common),
           RecRandom(account_configuration=AccountConfiguration(f"par3{tag}", None), **common)]
    for p2 in p2s:
        p1.setup_recorder(store, p2)
        p2.setup_recorder(store, None)
        await p1.battle_against(p2, n_battles=n // 2 if p2 is p2s[0] else n - n // 2)
    return store, p1.username, [p.username for p in p2s]


def record(battle_format: str, n: int, out: Path, port: int = 8000) -> None:
    store, u1, u2s = asyncio.run(_record(battle_format, n, port))
    out.mkdir(parents=True, exist_ok=True)
    gen = int(battle_format[3])
    obs_cols: dict[str, list] = {k: [] for k in fields(gen)}
    aux_cols: dict[str, list] = {k: [] for k in aux_fields(gen)}
    bidx, k1s, k2s = [], [], []
    with gzip.open(out / "battles.jsonl.gz", "wt") as f:
        for b, (tag, rec) in enumerate(store.items()):
            # p2's username: whichever recorded player's name appears in its chunks
            text = json.dumps(rec["p2"][:3])
            u2 = next(u for u in u2s if u in text)
            f.write(json.dumps({"format": battle_format, "p1": u1, "p2": u2, "chunks_p1": rec["p1"],
                                "chunks_p2": rec["p2"]}) + "\n")
            for k1, k2, obs, aux in rec["decisions"]:
                bidx.append(b), k1s.append(k1), k2s.append(k2)
                for k in obs_cols:
                    obs_cols[k].append(obs[k])
                for k in aux_cols:
                    aux_cols[k].append(aux[k])
    np.savez_compressed(out / "decisions.npz", spec=np.array(SPEC_VERSION), battle=np.array(bidx),
                        k1=np.array(k1s), k2=np.array(k2s),
                        **{f"obs/{k}": np.stack(v) for k, v in obs_cols.items()},
                        **{f"aux/{k}": np.stack(v) for k, v in aux_cols.items()})
    print(f"{len(store)} battles, {len(bidx)} decisions -> {out}")


# ---------------------------------------------------------------- comparison

def compare(ref: dict[str, np.ndarray], got: dict[str, np.ndarray], tol: float = TOLERANCE) -> list[str]:
    """Field-by-field differences (empty = parity). Integer fields must match exactly, floats within tol."""
    problems = []
    for k, r in ref.items():
        if k not in got:
            problems.append(f"{k}: missing")
            continue
        g = np.asarray(got[k])
        if g.shape != r.shape:
            problems.append(f"{k}: shape {g.shape} != {r.shape}")
            continue
        if np.issubdtype(r.dtype, np.integer):
            bad = np.argwhere(g != r)
        else:
            bad = np.argwhere(~np.isclose(g, r, atol=tol, rtol=0))
        if len(bad):
            i = tuple(bad[0])
            problems.append(f"{k}: {len(bad)} entries differ, first at {i}: {g[i]} vs reference {r[i]}")
    return problems


def load(fixture: Path) -> tuple[list[dict], dict[str, Any]]:
    with gzip.open(fixture / "battles.jsonl.gz", "rt") as f:
        battles = [json.loads(line) for line in f]
    z = np.load(fixture / "decisions.npz")
    return battles, {k: z[k] for k in z.files}


def check(fixture: Path, candidate: Path | None = None, limit: int | None = None, aux: bool = True) -> int:
    battles, z = load(fixture)
    assert str(z["spec"]) == SPEC_VERSION, f"fixture spec {z['spec']} != reference {SPEC_VERSION}"
    cand = dict(np.load(candidate)) if candidate else None
    n = len(z["battle"]) if limit is None else min(limit, len(z["battle"]))
    order = range(n) if limit is None else sorted(random.Random(0).sample(range(len(z["battle"])), n))
    n_bad = 0
    for d in order:
        b = battles[z["battle"][d]]
        ref = {k[4:]: v[d] for k, v in z.items() if k.startswith("obs/")}
        ref_aux = {k[4:]: v[d] for k, v in z.items() if k.startswith("aux/")}
        if cand is not None:  # another implementation's observations vs the stored reference
            problems = compare(ref, {k[4:]: v[d] for k, v in cand.items() if k.startswith("obs/")})
        else:  # the reference encoder now vs when the fixture was made (regression)
            battle = replay(b["chunks_p1"][: z["k1"][d]], b["p1"], b["format"])
            obs = encode(battle)
            problems = compare(ref, obs)
            if aux:
                opp = replay(b["chunks_p2"][: z["k2"][d]], b["p2"], b["format"])
                problems += [f"aux {p}" for p in compare(ref_aux, aux_targets(battle, opp, obs))]
        if problems:
            n_bad += 1
            if n_bad <= 10:
                print(f"decision {d} (battle {z['battle'][d]}, turn {ref['field_num'][0]}):", *problems, sep="\n  ")
    print(f"{n - n_bad}/{n} decisions match (spec {SPEC_VERSION})")
    return n_bad


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--format", default="gen9randombattle")
    r.add_argument("--n", type=int, default=30)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--port", type=int, default=8000)
    c = sub.add_parser("check")
    c.add_argument("fixture", type=Path)
    c.add_argument("--candidate", type=Path, help="npz with obs/<field> arrays aligned with the fixture decisions")
    c.add_argument("--limit", type=int)
    c.add_argument("--no-aux", action="store_true")
    a = p.parse_args()
    if a.cmd == "record":
        record(a.format, a.n, a.out, a.port)
    else:
        raise SystemExit(1 if check(a.fixture, a.candidate, a.limit, not a.no_aux) else 0)


if __name__ == "__main__":
    main()
