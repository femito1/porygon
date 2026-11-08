"""Fidelity of the first-person views `porygon.replay` builds, against ground truth.

    # 1. live parity: play battles on a local server with players that log every websocket message and, at every
    #    decision, the live observation / mask / action; then replay the logged streams and compare exactly
    python scripts/score_reconstruction.py record-live --format gen9randombattle --battles 4 --port 8650 --out DIR
    python scripts/score_reconstruction.py live-parity DIR [DIR ...]
    #    the same check on search-label runs (<run>/{a,b}/streams + decisions.jsonl: mask and action)
    python scripts/score_reconstruction.py live-parity RUN_DIR

    # 2. re-simulation fidelity of human replays (scripts/resim_inputlog.js output): exact reproduction of the stored
    #    log by commit grouping, failure classes, decisions per game, legality of the mapped human actions
    python scripts/score_reconstruction.py resim RESIM.jsonl[.gz] [...] [--encode]

Numbers go to .orchestra/data/RECONSTRUCTION.md.
"""

import argparse
import asyncio
import base64
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from porygon import replay


def open_text(path: Path, mode: str = "rt"):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def load_encoder(spec: str):
    import importlib
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


# ---------------------------------------------------------------- record live games

def record_live(fmt: str, battles: int, port: int, out: Path, encoder_spec: str) -> None:
    """Two random players on a local server, each logging its raw messages and its live decisions."""
    from poke_env import AccountConfiguration, ServerConfiguration
    from poke_env.environment import SinglesEnv
    from poke_env.player import RandomPlayer

    encoder = load_encoder(encoder_spec)
    server = ServerConfiguration(f"ws://localhost:{port}/showdown/websocket",
                                 "https://play.pokemonshowdown.com/action.php?")
    (out / "streams").mkdir(parents=True, exist_ok=True)
    (out / "decisions").mkdir(parents=True, exist_ok=True)

    class Recorder(RandomPlayer):
        async def _handle_battle_message(self, split_messages):
            tag = split_messages[0][0][1:]
            with gzip.open(out / "streams" / f"{tag}.{self.username}.jsonl.gz", "at") as f:
                f.write(json.dumps(split_messages) + "\n")
            await super()._handle_battle_message(split_messages)

        def choose_move(self, battle):
            order = super().choose_move(battle)
            mask = SinglesEnv.get_action_mask(battle)
            try:
                action = int(SinglesEnv.order_to_action(order, battle, fake=True, strict=False))
            except Exception:
                action = -1
            x = np.asarray(encoder(battle), dtype=np.float32)
            rec = {"battle": battle.battle_tag, "turn": battle.turn, "player": self.username, "action": action,
                   "mask": list(mask), "obs32": base64.b64encode(x.tobytes()).decode()}
            with open(out / "decisions" / f"{battle.battle_tag}.{self.username}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            return order

    stamp = str(int(time.time()))[-5:]
    a = Recorder(AccountConfiguration(f"rla{stamp}", None), battle_format=fmt, server_configuration=server,
                 log_level=40)
    b = Recorder(AccountConfiguration(f"rlb{stamp}", None), battle_format=fmt, server_configuration=server,
                 log_level=40)
    asyncio.run(a.battle_against(b, n_battles=battles))
    print(f"recorded {battles} {fmt} battles into {out}")


# ---------------------------------------------------------------- live parity

def stream_files(d: Path) -> list[tuple[Path, Path, str]]:
    """(stream file, decisions file, player) for a record-live dir or a search-label run ({a,b}/streams)."""
    out = []
    if (d / "streams").is_dir() and (d / "decisions").is_dir():  # record-live layout
        for s in sorted((d / "streams").glob("*.jsonl.gz")):
            tag, player = s.name[: -len(".jsonl.gz")].rsplit(".", 1)
            out.append((s, d / "decisions" / f"{tag}.{player}.jsonl", player))
        return out
    for sub in [d] + [d / x for x in ("a", "b")]:
        if not (sub / "streams").is_dir():
            continue
        dec = sub / "decisions.jsonl"
        for s in sorted((sub / "streams").glob("*.jsonl.gz")):
            out.append((s, dec, ""))
    return out


def live_parity(dirs: list[Path], encoder_spec: str) -> dict:
    encoder = load_encoder(encoder_spec)
    gen_of = lambda tag: int(tag.split("gen")[1][0]) if "gen" in tag else 9
    c = Counter()
    blocks = Counter()
    decisions_cache: dict[Path, dict[tuple[str, str], list[dict]]] = {}
    for d in dirs:
        for s, dec, player in stream_files(d):
            stream = [json.loads(line) for line in gzip.open(s, "rt")]
            tag = stream[0][0][0][1:]
            if not player:  # search-label run: the player is the one whose |request|s the stream holds
                player = next((json.loads(m[2]).get("side", {}).get("name") for msg in stream for m in msg
                               if len(m) > 2 and m[1] == "request" and m[2]), None)
            if dec not in decisions_cache:
                by = defaultdict(list)
                if dec.exists():
                    for line in open(dec):
                        r = json.loads(line)
                        by[(r["battle"], r["player"])].append(r)
                decisions_cache[dec] = by
            logged = decisions_cache[dec].get((tag, player), [])
            try:
                rep = replay.replay_stream(stream, player, gen_of(tag), encoder=encoder, battle_tag=tag)
            except Exception as e:
                c[f"replay error {type(e).__name__}"] += 1
                continue
            c["games"] += 1
            c["decisions"] += len(rep.decisions)
            c["logged decisions"] += len(logged)
            pairs = list(replay.align_logged(rep.decisions, logged))
            c["paired"] += len(pairs)
            for ours, rec in pairs:
                c["mask equal"] += ours.mask.tolist() == list(rec["mask"])
                if rec.get("action", -1) >= 0:
                    c["logged action legal in our mask"] += bool(ours.mask[rec["action"]])
                    c["logged actions"] += 1
                x = None
                if "obs32" in rec:
                    x = np.frombuffer(base64.b64decode(rec["obs32"]), dtype=np.float32)
                    y = np.asarray(ours.obs, dtype=np.float32)
                elif "obs" in rec:
                    x = np.frombuffer(base64.b64decode(rec["obs"]), dtype=np.float16).astype(np.float32)
                    y = np.asarray(ours.obs, dtype=np.float16).astype(np.float32)
                if x is not None:
                    c["obs compared"] += 1
                    same = np.array_equal(x, y, equal_nan=True)
                    c["obs equal"] += bool(same)
                    if not same:
                        for name, sl in obs_blocks(len(x)):
                            if not np.array_equal(x[sl], y[sl]):
                                blocks[name] += 1
    return {"counts": dict(c), "obs mismatch by block": dict(blocks)}


def obs_blocks(dim: int):
    """Named slices of the v1 observation (porygon.encoder) for mismatch reports."""
    from porygon import encoder as E
    if dim != E.OBS_DIM:
        return [("all", slice(0, dim))]
    out = [("field", slice(0, E.FIELD_DIM))]
    for side, base in (("own", 0), ("opp", 6)):
        for i in range(6):
            lo = E.FIELD_DIM + (base + i) * E.MON_DIM
            out += [(f"{side} mon: categorical", slice(lo, lo + E.N_CAT)),
                    (f"{side} mon: movepool", slice(lo + E.N_CAT, lo + E.N_CAT + E.MOVEPOOL)),
                    (f"{side} mon: numerics", slice(lo + E.N_CAT + E.MOVEPOOL, lo + E.N_CAT + E.MOVEPOOL
                                                    + E.MON_NUM_DIM)),
                    (f"{side} mon: moves", slice(lo + E.N_CAT + E.MOVEPOOL + E.MON_NUM_DIM, lo + E.MON_DIM))]
    return out


# ---------------------------------------------------------------- re-simulation fidelity

def resim_report(files: list[Path], encode: bool, encoder_spec: str) -> dict:
    encoder = load_encoder(encoder_spec) if encode else None
    c = Counter()
    errors = Counter()
    by_group = defaultdict(Counter)  # exact sha built vs not -> exact count
    t0, replay_s = time.time(), 0.0
    for f in files:
        for line in open_text(f):
            r = json.loads(line)
            c["battles"] += 1
            if not r.get("ok"):
                errors[(r.get("error") or "?").split(":")[0]] += 1
                continue
            group = "exact commit" if r["sim"] == r["version"] else "other commit"
            by_group[group]["battles"] += 1
            if r.get("exact") is None:
                by_group[group]["no stored log"] += 1
            else:
                by_group[group]["exact"] += bool(r["exact"])
                if not r["exact"]:
                    kind = (r.get("diff") or {}).get("log") or ""
                    by_group[group]["first diff: " + (kind.split("|")[1] if kind.count("|") > 1 else "end")] += 1
            if "streams" not in r:
                continue
            t = time.time()
            gen = int(r["id"].split("gen")[1][0]) if r["id"].startswith("gen") else 9
            try:
                sides = replay.resim_sides(r, gen, encoder)
            except Exception as e:
                errors[f"replay {type(e).__name__}"] += 1
                continue
            replay_s += time.time() - t
            c["games replayed"] += 1
            for sr in sides.values():
                for d in sr.decisions:
                    c["decisions"] += 1
                    c[f"decisions: {d.kind}"] += 1
                    c["n_legal = 1"] += d.n_legal == 1
                    if d.choice is None:
                        c["no choice (game ended / unanswered)"] += 1
                    elif d.action < 0:
                        c["choice not mapped"] += 1
                    else:
                        c["actions"] += 1
                        c["action legal"] += bool(d.legal)
                        c["tera actions"] += d.action >= replay.TERA_OFFSET
                        c["default choices"] += d.default
    c["seconds"] = round(time.time() - t0, 1)
    c["replay+encode seconds"] = round(replay_s, 1)
    return {"counts": dict(c), "by commit": {k: dict(v) for k, v in by_group.items()}, "errors": dict(errors)}


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record-live")
    r.add_argument("--format", default="gen9randombattle")
    r.add_argument("--battles", type=int, default=4)
    r.add_argument("--port", type=int, default=8650)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--encoder", default="porygon.encoder:embed_battle")
    lp = sub.add_parser("live-parity")
    lp.add_argument("dirs", nargs="+", type=Path)
    lp.add_argument("--encoder", default="porygon.encoder:embed_battle")
    rs = sub.add_parser("resim")
    rs.add_argument("files", nargs="+", type=Path)
    rs.add_argument("--encode", action="store_true", help="also encode every decision (throughput)")
    rs.add_argument("--encoder", default="porygon.encoder:embed_battle")
    a = p.parse_args()
    if a.cmd == "record-live":
        record_live(a.format, a.battles, a.port, a.out, a.encoder)
    elif a.cmd == "live-parity":
        print(json.dumps(live_parity(a.dirs, a.encoder), indent=1))
    else:
        print(json.dumps(resim_report(a.files, a.encode, a.encoder), indent=1))


if __name__ == "__main__":
    sys.exit(main())
