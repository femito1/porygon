"""Parity check: JS tracker + encoder_v1 (fastenv/js) vs poke-env 0.16.1 + porygon.encoder.embed_battle.

Replays each side's recorded message stream (scripts/fastenv/record_battles.js) into a poke-env `Battle` exactly as
`Player._handle_battle_message` would (server header lines, |request| -> parse_request -> choose_move, |win|,
|error| handling), and at every decision point compares the JS outputs with
  embed_battle(battle)                                   (per-slot mismatch counts, |a-b| <= 1e-6 * max(1, |a|))
  SinglesEnv.get_action_mask(battle)
  SinglesEnv.action_to_order(a, battle, strict=False)    for every action (poke-env's random fallback -> 'default')
  reward_computing_helper's current_value (fainted 0.05, hp 0.05, status 0, victory 1), also after the battle ends.
A poke-env exception must happen at the same event as the JS PokeEnvError.

    python scripts/fastenv/parity_v1.py FILE.jsonl [FILE2.jsonl ...] [--max-report 20]
    python scripts/fastenv/parity_v1.py --record 300 gen9randombattle [--seed 1]   # record (node) + check
"""
import argparse
import base64
import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import orjson  # noqa: E402
from poke_env.battle import Battle  # noqa: E402
from poke_env.data import GenData  # noqa: E402
from poke_env.player import SimpleHeuristicsPlayer  # noqa: E402
from poke_env.environment import SinglesEnv  # noqa: E402
from poke_env.environment.env import PokeEnv  # noqa: E402
from poke_env.player import Player  # noqa: E402
from poke_env.player.battle_order import BattleOrder  # noqa: E402

import porygon.encoder as enc  # noqa: E402

SHAPING = dict(fainted_value=0.05, hp_value=0.05, status_value=0.0, victory_value=1.0)
PLAYER_IGNORE = Player.MESSAGES_TO_IGNORE


class _Random(BattleOrder):
    @property
    def message(self):
        return "/choose default"  # poke-env picks a random valid order here; the JS side returns 'default'


Player.choose_random_singles_move = staticmethod(lambda battle: _Random())


class _RewardSelf:
    def __init__(self):
        self._reward_buffer = {}


def state_value(battle) -> float:
    return PokeEnv.reward_computing_helper(_RewardSelf(), battle, **SHAPING)


def slot_names() -> list[str]:
    names = []
    for w in enc.WEATHERS:
        names.append(f"field.weather.{w.name}")
    for w in enc.WEATHERS:
        names.append(f"field.weather_age.{w.name}")
    names.append("field.no_weather")
    for f in enc.FIELDS:
        names.append(f"field.field.{f.name}")
    for f in enc.FIELDS:
        names.append(f"field.field_age.{f.name}")
    for side in ("own", "opp"):
        names += [f"field.{side}.screen.{c.name}" for c in enc.SCREENS]
        names += [f"field.{side}.screen_age.{c.name}" for c in enc.SCREENS]
        names += [f"field.{side}.hazard.{c.name}" for c in enc.HAZARDS]
    names += [f"field.{x}" for x in ["turn", "opp_unseen", "own_alive", "opp_alive", "force_switch", "trapped",
                                     "maybe_trapped", "can_tera", "used_tera", "opp_used_tera", "wait",
                                     "no_active", "n_switches", "n_moves"]]
    assert len(names) == enc.FIELD_DIM
    mon = ["species", "item", "ability"] + [f"move{i}" for i in range(4)] + [f"pool{i}" for i in range(enc.MOVEPOOL)]
    mon += ["hp", "fainted"] + [f"base.{s}" for s in enc.STATS] + [f"stat.{s}" for s in enc.STATS] + ["level"]
    mon += [f"type.{t.name}" for t in enc.TYPES] + [f"tera.{t.name}" for t in enc.TYPES] + ["is_tera"]
    mon += [f"boost.{b}" for b in enc.BOOSTS] + [f"status.{s.name}" for s in enc.STATUSES] + ["status_counter"]
    mon += [f"effect.{e.name}" for e in enc.EFFECTS]
    mon += ["active", "mine", "first_turn", "protect_counter", "must_recharge", "preparing", "weight", "trapped",
            "is_opp_active", "unknown"]
    mv = ["bp", "acc", "priority", "pp", "available"] + [f"cat.{c.name}" for c in enc.CATEGORIES]
    mv += [f"type.{t.name}" for t in enc.TYPES] + ["stab", "log_mult", "immune", "heal", "drain", "recoil",
                                                   "self_switch", "force_switch", "side_condition", "status",
                                                   "volatile", "boosts", "self_boost", "protect", "breaks_protect",
                                                   "hits", "weather_terrain", "self_destruct"]
    assert len(mv) == enc.MOVE_DIM
    for i in range(4):
        mon += [f"m{i}.{x}" for x in mv]
    assert len(mon) == enc.MON_DIM
    for side in ("own", "opp"):
        names += [f"{side}.{x}" for x in mon] * 6  # tokens aggregated per side
    assert len(names) == enc.OBS_DIM
    return names


SLOTS = slot_names()


def dense(sp) -> np.ndarray:
    ix = np.frombuffer(base64.b64decode(sp[0]), dtype=np.uint16)
    vs = np.frombuffer(base64.b64decode(sp[1]), dtype=np.float32)
    out = np.zeros(enc.OBS_DIM, dtype=np.float32)
    out[ix] = vs
    return out


def handle(battle: Battle, message: str) -> bool:
    """Player._handle_battle_message for one websocket message (after the >battle- header). Returns True when
    poke-env would call choose_move (the agent's decision point)."""
    decision = False
    for split_message in [m.split("|") for m in message.split("\n")]:
        if not split_message or len(split_message) == 1:
            continue
        kind = split_message[1]
        if kind == "":
            battle.parse_message(split_message)
        elif kind in PLAYER_IGNORE:
            pass
        elif kind == "request":
            if split_message[2]:
                battle.parse_request(orjson.loads(split_message[2]))
                if not battle._wait and not battle.teampreview:
                    decision = True
        elif kind in ("win", "tie"):
            if kind == "win":
                battle.won_by(split_message[2])
            else:
                battle.tied()
        elif kind == "error":
            if split_message[2].startswith("[Invalid choice]") and not battle._wait and not battle.teampreview:
                decision = True
        elif kind in ("bigerror", "showteam"):
            pass
        else:
            battle.parse_message(split_message)
    return decision


def choice_of(battle, a: int) -> str:
    try:
        msg = SinglesEnv.action_to_order(np.int64(a), battle, strict=False).message
    except IndexError:
        return "default"
    return msg[len("/choose "):] if msg.startswith("/choose ") else msg


class Stats:
    def __init__(self):
        self.battles = self.decisions = self.obs_bad = self.mask_bad = self.choice_bad = self.value_bad = 0
        self.heur = self.heur_bad = 0
        self.waits = self.wait_bad = self.waits_missing = 0  # wait points (host.js --wait-rows)
        self.final_bad = self.align_bad = self.crash_both = self.crash_bad = 0
        self.slots = Counter()
        self.kinds = Counter()  # protocol coverage: message kinds (and a few effect/ability markers) seen
        self.examples = []

    def note(self, what, limit):
        if len(self.examples) < limit:
            self.examples.append(what)


def check_side(rec, s, side, st: Stats, limit: int):
    fmt = rec["format"]
    gen = GenData.from_format(fmt).gen
    tag = f"battle-{fmt}-{rec['id']}"
    username = f"p{s + 1}"
    battle = Battle(tag, username, logging.getLogger("parity"), gen=gen)
    handle(battle, f"|init|battle\n|title|p1 vs. p2\n|j|☆p1\n|j|☆p2")
    decs = side["dec"]
    waits = {w["e"]: w for w in side.get("wait", [])}
    k = 0
    py_crash = None
    for i, (kind, text) in enumerate(side["ev"]):
        if kind == "L":
            for ln in text.split("\n"):
                p = ln.split("|")
                if len(p) > 1:
                    st.kinds[p[1]] += 1
                    if p[1] in ("-start", "-activate", "-end", "-singleturn", "-singlemove") and len(p) > 3:
                        st.kinds[f"{p[1]} {p[3]}"] += 1
                    if p[1] == "move" and len(p) > 4 and p[-1].startswith("[from]"):
                        st.kinds[f"move {p[-1]}"] += 1
        line = text if kind == "L" else (f"|request|{text}" if kind == "R" else f"|error|{text.removeprefix('|error|')}")
        try:
            decision = handle(battle, line)
        except Exception as e:  # noqa: BLE001
            py_crash = (i, repr(e))
            break
        if side["crash"] and side["crash"]["e"] == i:
            break
        if i in waits:  # host.js --wait-rows: the waiting side's row = poke-env's wait step (obs, mask, reward state)
            w = waits.pop(i)
            st.waits += 1
            if not battle._wait:
                st.wait_bad += 1
                st.note(f"{tag} p{s + 1} ev {i}: JS wait row but poke-env is not waiting", limit)
            else:
                obs = enc.embed_battle(battle)
                js = dense(w["obs"])
                bad = np.flatnonzero(np.abs(obs - js) > 1e-6 * np.maximum(1.0, np.abs(obs)))
                mask = SinglesEnv.get_action_mask(battle)
                v = state_value(battle)
                if len(bad) or mask != w["mask"] or abs(v - w["v"]) > 1e-9:
                    st.wait_bad += 1
                    st.note(f"{tag} p{s + 1} ev {i} wait: obs slots " + ", ".join(
                        f"{j}:{SLOTS[j]} py={obs[j]:.6g} js={js[j]:.6g}" for j in bad[:6])
                        + f" mask {mask == w['mask']} value py={v} js={w['v']}", limit)
        elif kind == "R" and battle._wait and not decision and "wait" in side:
            st.waits_missing += 1
            st.note(f"{tag} p{s + 1} ev {i}: poke-env waits, no JS wait row", limit)
        if not decision:
            continue
        if k >= len(decs) or decs[k]["e"] != i:
            st.align_bad += 1
            st.note(f"{tag} p{s + 1}: decision alignment at event {i} (JS decision {k})", limit)
            return
        d = decs[k]
        k += 1
        st.decisions += 1
        try:
            obs = enc.embed_battle(battle)
        except Exception as e:  # noqa: BLE001
            st.obs_bad += 1
            st.note(f"{tag} p{s + 1} ev {i}: embed_battle raised {e!r}", limit)
            continue
        js = dense(d["obs"])
        bad = np.flatnonzero(np.abs(obs - js) > 1e-6 * np.maximum(1.0, np.abs(obs)))
        if len(bad):
            st.obs_bad += 1
            for j in bad:
                st.slots[SLOTS[j]] += 1
            st.note(f"{tag} p{s + 1} ev {i} turn {battle.turn}: obs slots " +
                    ", ".join(f"{j}:{SLOTS[j]} py={obs[j]:.6g} js={js[j]:.6g}" for j in bad[:6]), limit)
        mask = SinglesEnv.get_action_mask(battle)
        if mask != d["mask"]:
            st.mask_bad += 1
            st.note(f"{tag} p{s + 1} ev {i}: mask py={mask} js={d['mask']}", limit)
        ch = [choice_of(battle, a) for a in range(len(mask))]
        if ch != d["ch"]:
            st.choice_bad += 1
            diff = [(a, x, y) for a, (x, y) in enumerate(zip(ch, d["ch"])) if x != y]
            st.note(f"{tag} p{s + 1} ev {i}: choices (action, py, js) {diff[:4]}", limit)
        if "h" in d and d["h"] is not None:  # JS SimpleHeuristicsPlayer port (bots.js); None = poke-env goes random
            py = SimpleHeuristicsPlayer.choose_singles_move(battle)[0].message.removeprefix("/choose ")
            st.heur += 1
            if py != d["h"]:
                st.heur_bad += 1
                st.note(f"{tag} p{s + 1} ev {i}: heuristic py={py!r} js={d['h']!r}", limit)
        v = state_value(battle)
        if abs(v - d["v"]) > 1e-9:
            st.value_bad += 1
            st.note(f"{tag} p{s + 1} ev {i}: value py={v} js={d['v']}", limit)
    js_crash = side["crash"]
    if py_crash or js_crash:
        if py_crash and js_crash and py_crash[0] == js_crash["e"]:
            st.crash_both += 1
            st.note(f"{tag} p{s + 1}: both raise at ev {py_crash[0]}: py {py_crash[1][:120]} | js {js_crash['msg'][:120]}", limit)
        else:
            st.crash_bad += 1
            st.note(f"{tag} p{s + 1}: crash mismatch py={py_crash} js={js_crash}", limit)
        return
    if k != len(decs):
        st.align_bad += 1
        st.note(f"{tag} p{s + 1}: {len(decs) - k} JS decisions not seen by poke-env", limit)
    v = state_value(battle)
    if side["final"] is None or abs(v - side["final"]) > 1e-9:
        st.final_bad += 1
        st.note(f"{tag} p{s + 1}: final value py={v} js={side['final']}", limit)


def check_lines(lines, st: Stats, limit: int):
    for line in lines:
        rec = json.loads(line)
        st.battles += 1
        for s, side in enumerate(rec["sides"]):
            check_side(rec, s, side, st, limit)


def report(st: Stats, label: str) -> bool:
    print(f"{label}: {st.battles} battles, {st.decisions} decisions compared")
    print(f"  obs mismatching decisions {st.obs_bad}, mask {st.mask_bad}, action->choice {st.choice_bad}, "
          f"state value {st.value_bad}, final value {st.final_bad}, alignment {st.align_bad}, "
          f"crash mismatches {st.crash_bad} (both-raise {st.crash_both}); heuristic bot {st.heur_bad} of {st.heur}")
    print(f"  wait points {st.waits}: mismatching {st.wait_bad}, poke-env waits without a JS wait row "
          f"{st.waits_missing}")
    if st.slots:
        print("  mismatching obs slots (count over decisions x tokens):")
        for k, v in st.slots.most_common(30):
            print(f"    {v:6d}  {k}")
    rare = sorted((v, k) for k, v in st.kinds.items() if not k.startswith(("-start", "-activate", "-end", "-single",
                                                                             "move ")) or v < 50)
    print("  protocol coverage (count of log lines per kind / marker, both sides):")
    print("   ", ", ".join(f"{k}:{v}" for v, k in sorted(rare, key=lambda x: x[1]) if v))
    for e in st.examples:
        print("  -", e)
    return not (st.obs_bad or st.mask_bad or st.choice_bad or st.value_bad or st.final_bad or st.align_bad
                or st.crash_bad or st.heur_bad or st.wait_bad or st.waits_missing)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*")
    p.add_argument("--record", nargs=2, metavar=("N", "FORMAT"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-report", type=int, default=20)
    a = p.parse_args()
    logging.getLogger("poke-env").setLevel(logging.CRITICAL)
    logging.getLogger("parity").setLevel(logging.CRITICAL)
    st = Stats()
    for f in a.files:
        with open(f) as fh:
            check_lines(fh, st, a.max_report)
    label = ", ".join(Path(f).name for f in a.files)
    if a.record:  # stream node's JSONL straight into the checker (no large temp files)
        proc = subprocess.Popen(["node", str(ROOT / "scripts/fastenv/record_battles.js"), a.record[0], a.record[1],
                                 "-", str(a.seed)], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
        check_lines(proc.stdout, st, a.max_report)
        if proc.wait() != 0:
            raise SystemExit("record_battles.js failed")
        label = f"{a.record[1]} x{a.record[0]} (seed {a.seed})"
    ok = report(st, label)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
