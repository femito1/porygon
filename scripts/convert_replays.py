"""Convert battles into first-person training data: game records (F2) and encoded shards (F3).

    # human replays with input logs (corpus F1 parts, or the crawler's replay shards): re-simulated exactly
    python scripts/convert_replays.py corpus RAW/corpus/gen9randombattle/part-*.jsonl.gz --format gen9randombattle \
        --builds SHOWDOWN/builds --out OUT [--workers 3] [--encoder porygon.encoder:embed_battle] [--min-rating 0]
    # our runs that logged every websocket message (search-label runs: <run>/{a,b}/streams + decisions.jsonl)
    python scripts/convert_replays.py runs RUN_DIR [...] --format gen9randombattle --out OUT
    # old gen4 benchmark runs with --log-oracle (live v1 obs in decisions.jsonl, no requests): v1 shards only
    python scripts/convert_replays.py logged-obs gloria/runs/search-20261103-* --format gen4randombattle --out OUT

Outputs (formats: .orchestra/CONTRACTS.md "Data formats"):
  OUT/records/<format>/part-NNNNN.jsonl.gz   one JSON per (battle, side): the player's whole client message stream
      (`stream`, so any encoder can replay it: `porygon.replay.replay_stream`), both true teams, the result and every
      decision {idx (stream message index of the request), turn, kind, action, a (action index), n_legal, legal,
      search?}
  OUT/encoded/<format>/<encoder name>/shard-NNNNN.npz   obs f16 [N, D], mask i8 [N, A], action i16, target f16 [N, A],
      z f32, rating i16, opp_rating i16, game i32, turn i16, source u8, forced u8, n_legal u8, and from search runs
      q f16 [N, A] (NaN: unvisited / absent), root_value f16 (NaN: absent), visits f32, searched u8
  OUT/encoded/<format>/<encoder name>/meta.json, OUT/records/<format>/meta.json   counts, errors by class, sources

Work is split into units of --unit battles (in input order; unit k writes part k / shard k), so the output does not
depend on the number of workers. A battle that fails anywhere is skipped and counted under its error class. Human
battles are kept only when the re-simulation reproduced the stored log exactly (--keep-inexact to keep them
flagged). Games with a player named like jaxcalibur are skipped (we do not distil that bot).
"""

import argparse
import gzip
import importlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from porygon import replay  # noqa: E402
from porygon.encoder import OBS_DIM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESIM = ROOT / "scripts" / "resim_inputlog.js"
SKIP_PLAYERS = ("jaxcalibur",)
V1 = "porygon.encoder:embed_battle"
SET_FIELDS = ("species", "level", "item", "ability", "moves", "tera_type", "stats")
# shard `source` ids (stable across units and runs)
SOURCE_IDS = {"other": 0, "showdown-replays": 1, "holidayougi": 2, "pokechamp": 3, "searchlabels": 10, "benchmark": 11}


def source_id(name: str) -> int:
    if name.startswith("run:"):
        return SOURCE_IDS["searchlabels" if "searchlabels" in name else "benchmark"]
    return SOURCE_IDS.get(name.lower(), next((v for k, v in SOURCE_IDS.items() if k in name.lower()), 0))


def open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def load_encoder(spec: str, extra_src: list[str] | None = None):
    """`module:function`; extra_src: other src trees whose `porygon` subpackages join ours (e.g. netv2's
    porygon.v2 from its worktree), so `--encoder porygon.v2.encoder:encode --extra-src ../ws-netv2/src` works."""
    import porygon
    for d in extra_src or []:
        p = str(Path(d).resolve() / "porygon")
        if p not in porygon.__path__:
            porygon.__path__.append(p)
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


class V2PreEncoder:
    """netv2's v2 observation plus its auxiliary targets (`porygon.v2.encoder.aux_targets_from_team`): the
    opponent's true sets and its actual action at the same turn request (the "v2 supervised shards" format,
    CONTRACTS.md). Needs per-side context: `set_context(opp_team, opp_actions)` before each replay."""

    def __init__(self, extra_src: list[str] | None):
        load_encoder("porygon.v2.encoder:encode", extra_src)
        from porygon.v2 import encoder as v2, spec
        self.encode, self.aux, self.spec = v2.encode, v2.aux_targets_from_team, spec
        self.opp_team: list[dict] = []
        self.opp_actions: dict[int, tuple[str, str, bool]] = {}

    def set_context(self, opp_team: list[dict], opp_actions: dict[int, tuple[str, str, bool]]) -> None:
        self.opp_team, self.opp_actions = opp_team, opp_actions

    def __call__(self, battle) -> dict:
        obs = self.encode(battle)
        act = None if battle.force_switch else self.opp_actions.get(battle.turn)
        return {"obs": obs, "aux": self.aux(battle, self.opp_team, obs, act)}


class MultiEncoder:
    """Several encoders over the same replay: battle -> {spec: output}."""

    def __init__(self, fns: dict):
        self.fns = fns
        self.needs_context = any(hasattr(f, "set_context") for f in fns.values())

    def set_context(self, **kw) -> None:
        for f in self.fns.values():
            if hasattr(f, "set_context"):
                f.set_context(**kw)

    def __call__(self, battle) -> dict:
        return {spec: fn(battle) for spec, fn in self.fns.items()}


def opp_actions_of(decisions: list) -> dict[int, tuple[str, str, bool]]:
    """turn -> (kind, move id / species, tera) of a side's turn decisions (its action at each turn request)."""
    out = {}
    for d in decisions:
        if d.kind == "turn" and not d.retry and d.desc:
            out[d.turn] = (d.desc["type"], d.desc.get("move") or replay.to_id(d.desc.get("species", "")),
                           bool(d.desc.get("tera")))
    return out


def encoder_name(spec: str) -> str:
    return {"porygon.encoder:embed_battle": "v1"}.get(spec, spec.replace(":", ".").replace("/", "_"))


def netv2_commit(extra_src: list[str]) -> str:
    for d in extra_src:
        for f in (Path(d).parent / "COMMIT", Path(d).parent / "GIT_COMMIT"):
            if f.exists():
                return f.read_text().strip()[:12]
        try:
            out = subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                 timeout=10).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    return "unknown"


def code_commit() -> str:
    if os.environ.get("PORYGON_COMMIT"):
        return os.environ["PORYGON_COMMIT"]
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- inputs

def load_inputlogs(files: list[Path]) -> dict[str, str]:
    """id -> input log from the crawler's inputlog shards ({id, status, inputlog}) for corpus records without one
    (mirror datasets such as HolidayOugi have the log only)."""
    out = {}
    for f in files:
        for line in open_text(f):
            if line.strip():
                r = json.loads(line)
                il = r.get("inputlog") or ""
                if str(r.get("status", "200")) == "200" and il.startswith(">"):
                    out[r["id"]] = il
    return out


def corpus_battles(files: list[Path], fmt: str, inputlogs: dict[str, str] | None = None) -> Iterator[dict[str, Any]]:
    """F1 corpus records ({id, format, log, inputlog?, ...}) or crawler shards ({id, status, body: replay JSON});
    `inputlogs`: id -> input log for records that lack one."""
    for f in files:
        for line in open_text(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if "body" in r:  # crawler shard: the replay JSON as served
                if str(r.get("status")) != "200":
                    continue
                b = json.loads(r["body"])
                r = {"id": b.get("id", r["id"]), "format": b.get("formatid") or fmt, "log": b.get("log"),
                     "inputlog": b.get("inputlog"), "players": b.get("players"), "rating": b.get("rating"),
                     "uploadtime": b.get("uploadtime"), "source": "showdown-replays"}
            if replay.to_id(r.get("format") or str(r.get("id", "")).split("-")[0]) != fmt:
                continue
            r["format"] = fmt
            if not r.get("inputlog") and inputlogs:
                r["inputlog"] = inputlogs.get(r["id"])
            yield r


def human_unit(battles: list[dict], builds: Path, gen: int, encoder, keep_inexact: bool,
               min_rating: int) -> Iterator[tuple[dict | None, list[dict], str | None]]:
    """(resim result, side records, error class) per battle, re-simulated by one node process."""
    todo, out = [], {}
    for b in battles:
        players = [str(p).lower() for p in (b.get("players") or [])]
        if any(s in p for p in players for s in SKIP_PLAYERS):
            out[b["id"]] = (None, [], "skipped: jaxcalibur")
        elif not b.get("inputlog") or not str(b["inputlog"]).startswith(">"):
            out[b["id"]] = (None, [], "no inputlog")
        elif min_rating and (b.get("rating") or 0) < min_rating:
            out[b["id"]] = (None, [], "below min rating")
        else:
            todo.append(b)
    results = {}
    if todo:
        proc = subprocess.Popen(["node", str(RESIM), "--builds", str(builds), "-"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1 << 20)

        def feed():
            for b in todo:
                proc.stdin.write(json.dumps({"id": b["id"], "inputlog": b["inputlog"], "log": b.get("log")}) + "\n")
            proc.stdin.close()
        t = threading.Thread(target=feed, daemon=True)
        t.start()
        for line in proc.stdout:
            r = json.loads(line)
            results[r.get("id")] = r
        t.join()
        proc.wait()
    for b in battles:
        if b["id"] in out:
            yield out[b["id"]]
            continue
        r = results.get(b["id"])
        if r is None:
            yield None, [], "resim: no output"
            continue
        if not r.get("ok"):
            yield r, [], "resim: " + (r.get("error") or "?").split(":")[0]
            continue
        if r.get("exact") is False and not keep_inexact:
            yield r, [], "resim: not exact"
            continue
        try:
            if getattr(encoder, "needs_context", False):  # pass 1: both sides' actions; pass 2: encode
                first = replay.resim_sides(r, gen, None)
                sides = {}
                for side in ("p1", "p2"):
                    other = "p2" if side == "p1" else "p1"
                    encoder.set_context(opp_team=true_team(r["teams"][other], gen),
                                        opp_actions=opp_actions_of(first[other].decisions))
                    choices = {c["chunk"]: c["choice"] for c in r["choices"][side]}
                    sides[side] = replay.replay_stream(r["streams"][side], r["players"][side], gen, side, choices,
                                                       encoder, f"battle-{r['id']}")
            else:
                sides = replay.resim_sides(r, gen, encoder)
        except Exception as e:
            yield r, [], f"replay: {type(e).__name__}"
            continue
        recs = [human_side(b, r, s, sr) for s, sr in sides.items()]
        pair_opponent_actions(recs)
        yield r, recs, None


def true_team(sets: list[dict], gen: int) -> list[dict]:
    """Showdown sets -> the records' team format (every field known: revealed=True)."""
    from porygon import inference
    out = []
    for s in sets:
        species = replay.to_id(s.get("species") or s.get("name", ""))
        try:
            st = inference.stats(inference.set_species(species) if species not in inference.GEN.pokedex else species,
                                 s.get("level", 100), {**dict.fromkeys(inference.STATS, 0), **s.get("evs", {})},
                                 {**dict.fromkeys(inference.STATS, 31), **s.get("ivs", {})})
        except Exception:
            st = None
        m = {"species": s.get("species"), "level": s.get("level"), "item": replay.to_id(s.get("item", "")),
             "ability": replay.to_id(s.get("ability", "")), "moves": [replay.to_id(x) for x in s.get("moves", [])],
             "tera_type": s.get("teraType") if gen >= 9 else None, "stats": st, "gender": s.get("gender"),
             "evs": s.get("evs"), "ivs": s.get("ivs"), "role": s.get("role")}
        m["revealed"] = dict.fromkeys(SET_FIELDS, True)
        out.append(m)
    return out


def decision_json(d: replay.Decision) -> dict:
    j = {"idx": d.msg, "turn": d.turn, "kind": d.kind, "action": d.desc, "a": d.action, "n_legal": d.n_legal,
         "legal": d.legal, "mask": d.mask.tolist()}
    if d.default:
        j["default"] = True
    if d.retry:
        j["retry"] = True
    return j


def pair_opponent_actions(sides: list[dict]) -> None:
    """Each turn decision gets the other side's action at the same turn request (`opp_action`, for auxiliary
    targets: what the opponent did while we chose); forced decisions get none."""
    if len(sides) != 2:
        return
    for me, other in ((sides[0], sides[1]), (sides[1], sides[0])):
        theirs = {d.turn: d.desc for d in other["_decisions"] if d.kind == "turn" and not d.retry}
        for d, j in zip(me["_decisions"], me["decisions"]):
            j["opp_action"] = theirs.get(d.turn) if d.kind == "turn" else None


def result_of(winner: str | None, side: str) -> int:
    return 0 if winner in (None, "tie") else (1 if winner == side else -1)


def human_side(b: dict, r: dict, side: str, sr: replay.SideReplay) -> dict:
    other = "p2" if side == "p1" else "p1"
    gen = int(r["id"].split("gen")[1][0]) if "gen" in r["id"] else 9

    def rating(s: str) -> int:
        v = b.get(f"{s}_rating") or (r.get("ratings") or {}).get(s)
        return int(v) if v else -1
    return {"id": r["id"], "format": b.get("format"), "side": side, "player": r["players"][side],
            "rating": rating(side), "opp_rating": rating(other), "result": result_of(r.get("winner"), side),
            "end": r.get("end", "normal"), "source": b.get("source", "showdown-replays"),
            "uploadtime": b.get("uploadtime"), "showdown_commit": r.get("version"), "sim_commit": r.get("sim"),
            "resim_exact": r.get("exact"), "team": true_team(r["teams"][side], gen),
            "true_team": true_team(r["teams"][side], gen), "opp_team": true_team(r["teams"][other], gen),
            "log": None, "stream": r["streams"][side], "seed": replay.seed_of(r["id"]),
            "decisions": [decision_json(d) for d in sr.decisions], "_decisions": sr.decisions}


def run_battles(d: Path) -> Iterator[dict]:
    """A run dir's battles (battles.jsonl) with each logged player's stream and decisions."""
    subs = [s for s in [d] + [d / x for x in ("a", "b")] if (s / "streams").is_dir()]
    logged: dict[tuple[str, str], list[dict]] = {}
    for s in subs:
        if (s / "decisions.jsonl").exists():
            for line in open(s / "decisions.jsonl"):
                r = json.loads(line)
                logged.setdefault((r["battle"], r["player"]), []).append(r)
    for line in open(d / "battles.jsonl"):
        b = json.loads(line)
        b["streams"] = [p for s in subs for p in [s / "streams" / f"{b['roomid']}.jsonl.gz"] if p.exists()]
        b["logged"] = logged
        b["run"] = d.name
        yield b


SEARCH_FIELDS = ("shares", "probs", "value", "q", "root_value", "visits")  # decisions.jsonl fields kept per decision


def run_unit(battles: list[dict], gen: int, encoder, counts: Counter
             ) -> Iterator[tuple[dict | None, list[dict], str | None]]:
    def one_side(b: dict, stream: list, enc, count: bool) -> dict:
        player = next((json.loads(m[2]).get("side", {}).get("name") for msg in stream for m in msg
                       if len(m) > 2 and m[1] == "request" and m[2]), None)
        side = "p1" if player == b["p1"] else "p2"
        sr = replay.replay_stream(stream, player, gen, side, encoder=enc, battle_tag=b["roomid"])
        logged = b["logged"].get((b["roomid"], player), [])
        if count:
            counts["logged decisions"] += len(logged)
        for ours, rec in replay.align_logged(sr.decisions, logged):
            if count:
                counts["paired with logged"] += 1
                counts["mask equal to logged"] += ours.mask.tolist() == list(rec["mask"])
            ours.action = int(rec["action"])
            ours.legal = bool(ours.mask[ours.action])
            ours.desc = ours.describe(ours.action)
            ours.choice = "logged"
            ours.search = {k: rec[k] for k in SEARCH_FIELDS if k in rec}
        for d in sr.decisions:  # unlogged decisions with a single legal action
            if d.action < 0 and d.n_legal == 1:
                d.action, d.legal = int(np.argmax(d.mask)), True
                d.desc = d.describe(d.action)
        winner = None if not b.get("winner") else ("p1" if b["winner"] == b["p1"] else "p2")
        other = "p2" if side == "p1" else "p1"
        rec = {"id": b["roomid"], "format": f"gen{gen}randombattle", "side": side, "player": player,
               "rating": -1, "opp_rating": -1, "result": result_of(winner, side), "end": "normal",
               "source": f"run:{b['run']}", "team": true_team(b[f"{side}team"], gen),
               "true_team": true_team(b[f"{side}team"], gen), "opp_team": true_team(b[f"{other}team"], gen),
               "log": None, "stream": stream, "seed": replay.seed_of(b["roomid"]), "_decisions": sr.decisions}
        rec["decisions"] = []
        for d in sr.decisions:
            j = decision_json(d)
            if d.search:
                j["search"] = d.search
            rec["decisions"].append(j)
        return rec

    for b in battles:
        if not b["streams"]:
            yield None, [], "run: no stream"
            continue
        try:
            streams = [[json.loads(line) for line in gzip.open(path, "rt")] for path in b["streams"]]
            if getattr(encoder, "needs_context", False) and len(streams) == 2:
                first = [one_side(b, st, None, True) for st in streams]
                sides = []
                for k, st in enumerate(streams):
                    encoder.set_context(opp_team=first[k]["opp_team"],
                                        opp_actions=opp_actions_of(first[1 - k]["_decisions"]))
                    sides.append(one_side(b, st, encoder, False))
            else:
                sides = [one_side(b, st, encoder, True) for st in streams]
        except Exception as e:
            yield None, [], f"replay: {type(e).__name__}"
            continue
        pair_opponent_actions(sides)
        yield None, sides, None


def logged_obs_battles(d: Path) -> Iterator[dict]:
    """Old benchmark runs with `--log-oracle` (gloria runs/search-*-309116, -309120): their decisions.jsonl holds the
    live v1 observation (`obs`, float16) next to mask, action and search shares, but no |request| was logged, so
    these battles cannot be replayed into poke-env: v1 encoder only."""
    by: dict[str, list[dict]] = {}
    for line in open(d / "decisions.jsonl"):
        r = json.loads(line)
        if "obs" in r:
            by.setdefault(r["battle"], []).append(r)
    for line in open(d / "battles.jsonl"):
        b = json.loads(line)
        b["logged_obs"] = by.get(b["roomid"], [])
        b["run"] = d.name
        b["replay"] = d / "replays" / f"{b['roomid']}.log"
        yield b


class LoggedDecision:
    """A decision rebuilt from a decisions.jsonl record (no battle object): what `encode_rows` reads."""

    def __init__(self, r: dict):
        import base64
        self.obs = {V1: np.frombuffer(base64.b64decode(r["obs"]), dtype=np.float16)}
        self.mask = np.asarray(r["mask"], dtype=np.int8)
        self.action = int(r["action"])
        self.legal = bool(self.mask[self.action]) if 0 <= self.action < len(self.mask) else False
        self.turn = int(r["turn"])
        self.kind = "forced" if not self.mask[6:].any() and self.mask[:6].any() else "turn"
        self.n_legal = int(self.mask.sum())
        self.search = {k: r[k] for k in SEARCH_FIELDS if k in r}
        self.choice, self.desc, self.retry = "logged", None, False


def logged_obs_unit(battles: list[dict], gen: int) -> Iterator[tuple[dict | None, list[dict], str | None]]:
    for b in battles:
        if not b["logged_obs"]:
            yield None, [], "logged-obs: no observations"
            continue
        player = b["logged_obs"][0]["player"]
        side = "p1" if player == b["p1"] else "p2"
        other = "p2" if side == "p1" else "p1"
        winner = None if not b.get("winner") else ("p1" if b["winner"] == b["p1"] else "p2")
        decs = [LoggedDecision(r) for r in b["logged_obs"]]
        if len(decs[0].obs[V1]) != OBS_DIM:
            yield None, [], "logged-obs: unexpected obs size"
            continue
        rec = {"id": b["roomid"], "format": f"gen{gen}randombattle", "side": side, "player": player, "rating": -1,
               "opp_rating": -1, "result": result_of(winner, side), "end": "normal", "source": f"run:{b['run']}",
               "view": "logged v1 obs, no requests (v1 encoder only; search with the PPO network prior)",
               "team": true_team(b[f"{side}team"], gen), "true_team": true_team(b[f"{side}team"], gen),
               "opp_team": true_team(b[f"{other}team"], gen), "stream": None,
               "log": b["replay"].read_text() if b["replay"].exists() else None, "seed": replay.seed_of(b["roomid"]),
               "decisions": [{"turn": d.turn, "kind": d.kind, "a": d.action, "n_legal": d.n_legal,
                              "legal": d.legal, "mask": d.mask.tolist(), "search": d.search} for d in decs],
               "_decisions": decs}
        yield None, [rec], None


# ---------------------------------------------------------------- units

def encode_rows(rec: dict, game: int, source: int, n_actions: int, spec: str) -> dict[str, list]:
    """Shard rows of one record for encoder `spec` (decision.obs = {encoder spec: its output})."""
    rows: dict[str, list] = {k: [] for k in DTYPES}
    for d in rec["_decisions"]:
        q = np.full(n_actions, np.nan, dtype=np.float32)
        if d.search and d.search.get("q") is not None and len(d.search["q"]) == n_actions:
            q = np.array([np.nan if v is None else v for v in d.search["q"]], dtype=np.float32)
        rv = d.search.get("root_value") if d.search else None
        rows["q"].append(q.astype(np.float16))
        rows["root_value"].append(np.nan if rv is None else rv)
        rows["visits"].append((d.search or {}).get("visits") or 0)
        rows["searched"].append(d.search is not None)
        target = np.zeros(n_actions, dtype=np.float32)
        search = d.search
        shares = np.asarray(search["shares"], dtype=np.float32) if search and search.get("shares") else None
        if shares is not None and len(shares) == n_actions and (shares * d.mask).sum() > 0:
            target = shares * d.mask / (shares * d.mask).sum()
        elif d.action >= 0 and d.legal:
            target[d.action] = 1.0
        x = d.obs[spec]
        rows["obs"].append(x if isinstance(x, dict) else np.asarray(x, dtype=np.float16))
        rows["mask"].append(d.mask.astype(np.int8))
        rows["action"].append(d.action if d.legal else -1)
        rows["target"].append(target.astype(np.float16))
        rows["z"].append(rec["result"])
        rows["rating"].append(rec["rating"])
        rows["opp_rating"].append(rec["opp_rating"])
        rows["game"].append(game)
        rows["turn"].append(d.turn)
        rows["source"].append(source)
        rows["forced"].append(d.kind == "forced")
        rows["n_legal"].append(d.n_legal)
    return rows


DTYPES = {"obs": np.float16, "mask": np.int8, "action": np.int16, "target": np.float16, "z": np.float32,
          "rating": np.int16, "opp_rating": np.int16, "game": np.int32, "turn": np.int16, "source": np.uint8,
          "forced": np.uint8, "n_legal": np.uint8,
          # search-label runs (additive): the search's values, NaN / 0 where absent
          "q": np.float16, "root_value": np.float16, "visits": np.float32, "searched": np.uint8}


def do_unit(job: tuple) -> dict:
    """Convert one unit of battles: one record part, and one shard per encoder (all encoders see the same
    decisions of the same replay pass). Returns counts."""
    (unit, base, battles, mode, a) = job
    fmt, gen = a["format"], a["gen"]
    specs = a["encoders"]
    fns = {spec: V2PreEncoder(a["extra_src"]) if spec == "v2pre" else load_encoder(spec, a["extra_src"])
           for spec in specs}
    encoder = MultiEncoder(fns)
    from poke_env.environment import SinglesEnv
    n_actions = SinglesEnv.get_action_space_size(gen)
    rec_dir = Path(a["rec_dir"])
    rec_dir.mkdir(parents=True, exist_ok=True)
    counts, sources, commits = Counter(), {}, Counter()
    games: list[str | None] = []
    cols = {spec: {k: [] for k in DTYPES} for spec in specs}
    obs_dim: dict[str, Any] = {}
    t0 = time.time()
    try:
        if mode == "corpus":
            it = human_unit(battles, Path(a["builds"]), gen, encoder, a["keep_inexact"], a["min_rating"])
        elif mode == "runs":
            it = run_unit(battles, gen, encoder, counts)
        else:
            assert specs == [V1], "logged-obs runs hold v1 observations only"
            it = logged_obs_unit(battles, gen)
        with gzip.open(rec_dir / f"part-{unit:05d}.jsonl.gz.tmp", "wt") as f:
            for i, (r, sides, err) in enumerate(it):
                counts["battles"] += 1
                if r is not None:
                    commits[r.get("version") or "?"] += 1
                    if r.get("exact") is not None:
                        counts["resim exact"] += bool(r["exact"])
                        counts["resim validated"] += 1
                if err:
                    counts[f"error: {err}"] += 1
                    continue
                counts["games"] += 1
                games.append(sides[0]["id"] if sides else None)
                for rec in sides:
                    sources[rec["source"]] = source_id(rec["source"])
                    for spec in specs:
                        rows = encode_rows(rec, base + i, sources[rec["source"]], n_actions, spec)
                        if spec == "v2pre":  # a stable game id (pretrain splits by crc32 of it)
                            rows["game"] = [replay.seed_of(rec["id"])] * len(rows["game"])
                        for k, v in rows.items():
                            cols[spec][k].extend(v)
                    for d in rec["_decisions"]:
                        counts["decisions"] += 1
                        counts["decisions with action"] += d.action >= 0 and bool(d.legal)
                        counts["illegal actions"] += d.action >= 0 and not d.legal
                        counts["unmapped choices"] += d.action < 0 and d.choice is not None
                        for spec in specs:
                            if spec not in obs_dim and d.obs is not None:
                                x = d.obs[spec]
                                if spec == "v2pre":
                                    x = {**{f"obs/{k}": v for k, v in x["obs"].items()},
                                         **{f"aux/{k}": v for k, v in x["aux"].items()}}
                                obs_dim[spec] = ({k: [list(v.shape), str(v.dtype)] for k, v in x.items()}
                                                 if isinstance(x, dict) else int(np.asarray(x).shape[0]))
                    rec.pop("_decisions")
                    f.write(json.dumps(rec) + "\n")
        os.replace(rec_dir / f"part-{unit:05d}.jsonl.gz.tmp", rec_dir / f"part-{unit:05d}.jsonl.gz")
        for spec in specs:
            c = cols[spec]
            if not c["obs"]:
                continue
            enc_dir = Path(a["enc_dirs"][spec])
            enc_dir.mkdir(parents=True, exist_ok=True)
            obs = c.pop("obs")
            arrays = {k: (np.stack(v) if k in ("mask", "target", "q") else np.asarray(v)).astype(DTYPES[k])
                      for k, v in c.items()}
            if spec == "v2pre":  # CONTRACTS.md "v2 supervised shards": obs/<field>, aux/<field>, f32 target, i64 game
                arrays["target"] = arrays["target"].astype(np.float32)
                arrays["game"] = np.asarray(c["game"], dtype=np.int64)
                arrays.update({f"obs/{k}": np.stack([o["obs"][k] for o in obs]) for k in obs[0]["obs"]})
                arrays.update({f"aux/{k}": np.stack([o["aux"][k] for o in obs]) for k in obs[0]["aux"]})
            elif isinstance(obs[0], dict):  # a structured encoder (netv2's v2): one array per field, own dtypes
                arrays.update({f"obs_{k}": np.stack([o[k] for o in obs]) for k in obs[0]})
            else:
                arrays["obs"] = np.stack(obs).astype(np.float16)
            tmp = enc_dir / f"shard-{unit:05d}.tmp.npz"
            (np.savez if a.get("no_compress") else np.savez_compressed)(tmp, **arrays)
            os.replace(tmp, enc_dir / f"shard-{unit:05d}.npz")
    except Exception as e:
        counts[f"unit error: {type(e).__name__}"] += 1
        traceback.print_exc()
    return {"unit": unit, "counts": dict(counts), "sources": sources, "commits": dict(commits),
            "obs_dim": obs_dim, "n_actions": n_actions, "seconds": time.time() - t0, "games": games}


def units(items: Iterator[dict], size: int, mode: str, a: dict, start: int = 0) -> Iterator[tuple]:
    batch, base, k = [], 0, 0
    for i, b in enumerate(items):
        if not batch:
            base = i
        batch.append(b)
        if len(batch) == size:
            if k >= start:
                yield (k, base, batch, mode, a)
            batch, k = [], k + 1
    if batch and k >= start:
        yield (k, base, batch, mode, a)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["corpus", "runs", "logged-obs"])
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--format", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--builds", type=Path, default=Path("/vol/bitbucket/fr1226/porygon-ws-data/showdown/builds"))
    p.add_argument("--encoder", action="append", default=None,
                   help="module:function; repeat to encode the same decisions with several encoders in one pass")
    p.add_argument("--extra-src", action="append", default=[], help="another src tree for porygon.* encoders")
    p.add_argument("--v2pre-out", type=Path, default=None,
                   help="final path of the `--encoder v2pre` shards (default OUT/encoded/<format>/v2pre)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--unit", type=int, default=500, help="battles per record part / encoded shard")
    p.add_argument("--start-unit", type=int, default=0, help="resume: skip the units before this one")
    p.add_argument("--min-rating", type=int, default=0)
    p.add_argument("--inputlogs", nargs="*", type=Path, default=[],
                   help="crawler inputlog shards to join by id (corpus records without an input log)")
    p.add_argument("--keep-inexact", action="store_true")
    p.add_argument("--no-compress", action="store_true", help="write shards with np.savez (faster to load)")
    a = p.parse_args()
    gen = int(a.format[3])
    if int(os.environ.get("PORYGON_GEN", "4")) != gen:
        print(f"warning: PORYGON_GEN={os.environ.get('PORYGON_GEN', '4 (default)')} but format is gen{gen}",
              file=sys.stderr)
    a.encoder = a.encoder or [V1]
    args = {"format": a.format, "gen": gen, "encoders": a.encoder, "out": str(a.out), "builds": str(a.builds),
            "keep_inexact": a.keep_inexact, "min_rating": a.min_rating, "extra_src": a.extra_src,
            "no_compress": a.no_compress}
    if a.mode == "corpus":
        items = corpus_battles(a.inputs, a.format, load_inputlogs(a.inputlogs) if a.inputlogs else None)
    else:
        items = (b for d in a.inputs for b in (run_battles(d) if a.mode == "runs" else logged_obs_battles(d)))
    # Readers may be using OUT while it is refreshed: each run writes new version dirs next to the final paths and
    # then swaps symlinks onto them (rename is atomic), keeping the previous version for readers mid-read.
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    rec_final = a.out / "records" / a.format
    enc_finals = {spec: (a.v2pre_out if spec == "v2pre" and a.v2pre_out else
                         a.out / "encoded" / a.format / encoder_name(spec)) for spec in a.encoder}
    args["rec_dir"] = str(rec_final) + f".v{stamp}"
    args["enc_dirs"] = {spec: str(d) + f".v{stamp}" for spec, d in enc_finals.items()}
    total, sources, commits = Counter(), {}, Counter()
    game_ids: list[str] = []
    obs_dim: dict[str, Any] = {}
    n_actions = None
    t0 = time.time()
    jobs = units(items, a.unit, a.mode, args, a.start_unit)
    with Pool(a.workers) as pool:
        for res in pool.imap(do_unit, jobs):
            total.update(res["counts"])
            commits.update(res["commits"])
            sources.update(res["sources"])
            for k, v in res["obs_dim"].items():
                obs_dim.setdefault(k, v)
            n_actions = res["n_actions"]
            game_ids += [g for g in res["games"] if g]
            el = time.time() - t0
            print(f"unit {res['unit']}: {res['counts'].get('games', 0)} games, {res['counts'].get('decisions', 0)} "
                  f"decisions in {res['seconds']:.0f} s; total {total['games']} games / {total['battles']} battles, "
                  f"{total['games'] / max(el, 1e-9):.2f} games/s", flush=True)
    meta = {"format": a.format, "PORYGON_GEN": os.environ.get("PORYGON_GEN", "4"), "n_actions": n_actions,
            "encoders": {encoder_name(spec): {"encoder": spec, "obs_dim": obs_dim.get(spec)} for spec in a.encoder},
            "counts": dict(total), "sources": sources, "source_ids": SOURCE_IDS, "showdown_commits": len(commits),
            "code_commit": code_commit(), "inputs": [str(x) for x in a.inputs], "seconds": round(time.time() - t0, 1),
            "workers": a.workers, "version": stamp, "n_games": len(game_ids), "games": game_ids,
            "note": "game index g in a shard = position of the battle in input order; `games` lists the converted ids;"
                    " every encoder dir of one version holds the same decisions"}
    for spec, d in args["enc_dirs"].items():  # per-encoder meta keeps the old single-encoder keys too
        Path(d).mkdir(parents=True, exist_ok=True)
        extra = {}
        if spec == "v2pre":
            load_encoder("porygon.v2.encoder:encode", a.extra_src)
            from porygon.v2 import spec as v2spec
            extra = {"spec_version": v2spec.SPEC_VERSION, "gen": gen, "format_name": "v2 supervised shards",
                     "netv2_commit": netv2_commit(a.extra_src), "data_commit": meta["code_commit"],
                     "game_id": "crc32 of the battle id (`game`), shared by all of a game's decisions",
                     "extras": "q f16[N,A], root_value f16, visits f32, searched u8, action i16, mask i8, turn, "
                               "forced, n_legal (ignored by porygon.v2.pretrain)"}
        (Path(d) / "meta.json").write_text(json.dumps(
            {**meta, **extra, "encoder": spec, "encoder_name": encoder_name(spec), "obs_dim": obs_dim.get(spec)},
            indent=1))
    Path(args["rec_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(args["rec_dir"]) / "meta.json").write_text(json.dumps(meta, indent=1))
    swap_in(rec_final, Path(args["rec_dir"]))
    for spec, final in enc_finals.items():
        swap_in(final, Path(args["enc_dirs"][spec]))
    print(json.dumps(meta["counts"], indent=1))


def swap_in(final: Path, new: Path, keep: int = 2) -> None:
    """Point `final` (a symlink) at version dir `new` atomically; keep the newest `keep` versions."""
    if final.exists() and not final.is_symlink():  # an old plain dir: becomes a version itself (one-time)
        final.rename(final.parent / f"{final.name}.v00000000-000000")
    tmp = final.parent / f".{final.name}.lnk-{os.getpid()}"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(new.name)
    os.replace(tmp, final)
    versions = sorted(p for p in final.parent.glob(f"{final.name}.v*") if p.is_dir())
    for old in versions[:-keep]:
        if old.resolve() != new.resolve():
            import shutil
            shutil.rmtree(old, ignore_errors=True)


if __name__ == "__main__":
    main()
