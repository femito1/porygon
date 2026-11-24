"""netv2 2026-11-24, council P1 (spec v2.2a) Stage 0, Python side: reads stage0_record.js output and checks, at every
decision of either side of every battle (both sides' streams replayed into poke-env exactly as Player._handle_battle_
message would: ws/fastenv scripts/fastenv/parity_v1.handle, copied below):

  parity_off   porygon.v2.encoder.encode(battle)            == JS flag off      (every field; ints exact, floats 1e-4)
  parity_on    encode(battle, own_tera_visible=True)         == JS flag on       (every field, same tolerance)
  parity_aux   aux_targets(battle, opp_battle, obs)          == JS aux           (both flags)
  py_off_old   encode(battle) == today's encoder (OLD_PY = HEAD's encoder.py) byte for byte, aux too
  py_aux_flag  aux_targets on the flag-on obs == on the flag-off obs, byte for byte
  py_v21_view  porygon.v2.spec.v21_view(flag on) == flag off, byte for byte (what v2.1 pool / frozen nets get)
  py_pre_tera  mon_num terastallized == 0 for every own mon that has not terastallized (flag on)
plus the JS-only checks of the same run (stage0_record.js stream), the leak test (stage0_record.js leak) and the
Volcarona fixture (Tera Water vs Tera Ground: own tera / tera_p and active_tera_vs of Tera Blast vs the Heatran lead).

    python stage0_check.py --stream S.jsonl --old-py OLD_encoder.py --leak L.json --volcarona V.json --out SUMMARY.json
Exit 0 iff every check passes (and the streams cover >= --min-battles battles).
"""
import argparse
import base64
import importlib.util
import json
import logging
import sys
from collections import Counter

import numpy as np
import orjson
import torch
from poke_env.battle import Battle
from poke_env.data import GenData
from poke_env.player import Player

import porygon.v2.encoder as enc
from porygon.v2 import spec
from porygon.v2.spec import MON_NUM, TYPE_ID, v21_view

TOL = 1e-4
PLAYER_IGNORE = Player.MESSAGES_TO_IGNORE
T_COL = MON_NUM.index("terastallized")


def handle(battle: Battle, message: str) -> bool:  # ws/fastenv scripts/fastenv/parity_v1.handle (verbatim)
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


def dense(sp_, shape, dtype) -> np.ndarray:
    out = np.zeros(int(np.prod(shape)), dtype=dtype)
    ix = np.frombuffer(base64.b64decode(sp_[0]), dtype=np.uint16)
    out[ix] = np.frombuffer(base64.b64decode(sp_[1]), dtype=dtype)
    return out.reshape(shape)


def js_dict(rec: dict, fl: dict) -> dict:
    return {k: dense(rec[k], shape, np.dtype(dt)) for k, (shape, dt) in fl.items() if k in rec}


class Stats:
    def __init__(self):
        self.battles = self.decisions = 0
        self.bad = Counter()  # check -> decisions failing it
        self.fields = Counter()  # check/field -> decisions
        self.pre_slots = self.pre_filled = self.on_ne_off = 0
        self.examples = []

    def fail(self, check, msg, limit=30):
        self.bad[check] += 1
        if len(self.examples) < limit:
            self.examples.append(f"{check}: {msg}")


def close(st: Stats, check: str, ref: dict, got: dict, label: str) -> bool:
    ok = True
    for k, r in ref.items():
        g = got[k]
        bad = np.argwhere(g != r) if np.issubdtype(r.dtype, np.integer) else np.argwhere(~np.isclose(g, r, atol=TOL, rtol=0))
        if len(bad):
            ok = False
            st.fields[f"{check}/{k}"] += 1
            i = tuple(bad[0])
            if len(st.examples) < 30:
                st.examples.append(f"{check}: {label}: {k} {len(bad)} entries, first {i}: js {g[i]} ref {r[i]}")
    if not ok:
        st.bad[check] += 1
    return ok


def same(st: Stats, check: str, a: dict, b: dict, label: str) -> bool:
    diff = [k for k in a if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or a[k].tobytes() != b[k].tobytes()]
    if diff:
        for k in diff:
            st.fields[f"{check}/{k}"] += 1
        st.fail(check, f"{label}: {diff}")
    return not diff


def check_side(rec: dict, s: int, st: Stats, old):
    fmt = rec["format"]
    gen = GenData.from_format(fmt).gen
    F, A = spec.fields(gen), spec.aux_fields(gen)
    tag = f"battle-{fmt}-{rec['id']}"
    mk = lambda u: Battle(tag, u, logging.getLogger("stage0"), gen=gen)  # noqa: E731
    me, opp = mk(f"p{s + 1}"), mk(f"p{2 - s}")
    hdr = "|init|battle\n|title|p1 vs. p2\n|j|☆p1\n|j|☆p2"
    handle(me, hdr)
    handle(opp, hdr)
    side, oside = rec["sides"][s], rec["sides"][1 - s]

    def line(kind, text):
        return text if kind == "L" else (f"|request|{text}" if kind == "R" else f"|error|{text.removeprefix('|error|')}")
    decs, k, j_opp = side["dec"], 0, 0
    for i, (kind, text) in enumerate(side["ev"]):
        if side["crash"] and side["crash"]["e"] == i:
            return
        if not handle(me, line(kind, text)):
            continue
        if k >= len(decs) or decs[k]["e"] != i:
            st.fail("alignment", f"{tag} p{s + 1}: decision alignment at event {i}")
            return
        d = decs[k]
        k += 1
        while j_opp < d["k"] and not (oside["crash"] and oside["crash"]["e"] <= j_opp):
            handle(opp, line(*oside["ev"][j_opp]))
            j_opp += 1
        st.decisions += 1
        label = f"{tag} p{s + 1} ev {i} turn {me.turn}"
        ob = None if oside["crash"] else opp
        off, on, o_old = enc.encode(me), enc.encode(me, own_tera_visible=True), old.encode(me)
        aux, aux_on, aux_old = enc.aux_targets(me, ob, off), enc.aux_targets(me, ob, on), old.aux_targets(me, ob, o_old)
        js_off = js_dict(d["off"], F)
        js_on = {**js_off, **js_dict(d["on"], {kk: F[kk] for kk in ("tera", "tera_p")})}
        close(st, "parity_off", off, js_off, label)
        close(st, "parity_on", on, js_on, label)
        close(st, "parity_aux", aux, js_dict(d["aux"], A), label)
        same(st, "py_off_old", off, o_old, label)
        same(st, "py_off_old_aux", aux, aux_old, label)
        same(st, "py_aux_flag", aux_on, aux, label)
        same(st, "py_v21_view", v21_view(on), off, label)
        if any(on[kk].tobytes() != off[kk].tobytes() for kk in on):
            st.on_ne_off += 1
        for j, mon in enumerate(list(me.team.values())[:6]):
            if not mon.is_terastallized:
                st.pre_slots += 1
                if on["mon_num"][j, T_COL] != 0:
                    st.fail("py_pre_tera", f"{label}: own slot {j} terastallized != 0 before -terastallize")
                if gen >= 9 and on["tera"][j] != 0:
                    st.pre_filled += 1


def volcarona_check(v: dict) -> tuple[bool, list[str]]:
    """Own tera / tera_p and active_tera_vs of our lead Volcarona's Tera Blast (move slot 3) vs the opponent's lead
    (opponent slot 0), Tera Water vs Tera Ground: vs Iron Moth (Fire/Poison) Ground x4 > Water x2; vs Skarmory
    (Steel/Flying) Ground immune < Water x1; flag off: no difference at all between Water and Ground."""
    from porygon.v2.damage import DamageFeatures
    gen = int(v["format"][3])
    F = spec.fields(gen)
    lines, ok = [], True
    lay = DamageFeatures(gen, trim=False)
    tw, tg = TYPE_ID["water"], TYPE_ID["ground"]

    def feats(rec):
        o = js_dict(rec, F)
        t = {kk: torch.from_numpy(x[None]) for kk, x in o.items()}
        with torch.no_grad():
            return o, lay(t)["active_tera_vs"][0].numpy()  # [4 move slots, 6 opponents, 4]
    cases = {(c["opp"], c["tera"]): c for c in v["cases"]}
    for opp, want in (("ironmoth", "ground"), ("skarmory", "water")):
        res = {}
        for tt in ("Water", "Ground"):
            c = cases[(opp, tt)]
            if c["active"] != "volcarona" or c["opp_active"] != opp or c["moves"][3] != "terablast":
                return False, [f"fixture setup: {opp} {tt}: active {c['active']} vs {c['opp_active']}, moves {c['moves']}"]
            res[tt] = {"on": feats(c["on"]), "off": feats(c["off"])}
        w, g = res["Water"], res["Ground"]
        tb_w, tb_g, tb_off = w["on"][1][3, 0, 0], g["on"][1][3, 0, 0], w["off"][1][3, 0, 0]
        exp = [("flag on: own tera[0] = water / ground", int(w["on"][0]["tera"][0]) == tw and int(g["on"][0]["tera"][0]) == tg),
               ("flag on: own tera_p[0] one-hot on water / ground",
                w["on"][0]["tera_p"][0, tw] == 1 and w["on"][0]["tera_p"][0].sum() == 1
                and g["on"][0]["tera_p"][0, tg] == 1 and g["on"][0]["tera_p"][0].sum() == 1),
               ("flag on: terastallized[0] = 0 (pre-Tera)",
                w["on"][0]["mon_num"][0, T_COL] == 0 and g["on"][0]["mon_num"][0, T_COL] == 0),
               ("flag off: own tera[0] = 0 and tera_p[0] = 0 for both",
                int(w["off"][0]["tera"][0]) == 0 and int(g["off"][0]["tera"][0]) == 0
                and w["off"][0]["tera_p"][0].sum() == 0 and g["off"][0]["tera_p"][0].sum() == 0),
               ("flag off: the Water and Ground observations are byte-identical",
                all(w["off"][0][kk].tobytes() == g["off"][0][kk].tobytes() for kk in F)),
               ("flag off: active_tera_vs identical for Water / Ground", np.array_equal(w["off"][1], g["off"][1])),
               ("flag on: the other move slots' active_tera_vs equal for Water / Ground (only Tera Blast's type changes)",
                np.allclose(w["on"][1][:3], g["on"][1][:3], atol=1e-6))]
        if want == "ground":
            exp.append((f"flag on: Tera Blast vs Iron Moth: Ground (x4, tera STAB) {tb_g:.4f} > Water (x2) {tb_w:.4f} > "
                        f"flag off (Normal, x1) {tb_off:.4f}", tb_g > tb_w > tb_off > 0))
        else:
            exp.append((f"flag on: Tera Blast vs Skarmory: Water (x1, tera STAB) {tb_w:.4f} > flag off (Normal, x0.5) "
                        f"{tb_off:.4f} > Ground (immune) {tb_g:.2e}", tb_w > tb_off > tb_g and tb_g < 1e-6))
        for name, good in exp:
            lines.append(("PASS " if good else "FAIL ") + f"[{opp}] " + name)
            ok &= bool(good)
    return ok, lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stream", required=True)
    p.add_argument("--old-py", required=True, help="today's porygon/v2/encoder.py (git show HEAD:...)")
    p.add_argument("--leak", required=True)
    p.add_argument("--volcarona", required=True)
    p.add_argument("--min-battles", type=int, default=300)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    logging.getLogger("poke-env").setLevel(logging.CRITICAL)
    logging.getLogger("stage0").setLevel(logging.CRITICAL)
    sp_ = importlib.util.spec_from_file_location("porygon.v2.encoder_old", a.old_py)
    old = importlib.util.module_from_spec(sp_)
    sp_.loader.exec_module(old)
    st, js = Stats(), None
    with open(a.stream) as f:
        for ln in f:
            rec = json.loads(ln)
            if "js_checks" in rec:
                js = rec["js_checks"]
                continue
            st.battles += 1
            for s in (0, 1):
                check_side(rec, s, st, old)
    leak = json.load(open(a.leak))["leak"]
    vok, vlines = volcarona_check(json.load(open(a.volcarona)))
    js_keys = ("off_vs_old_obs", "off_vs_old_aux", "on_vs_off_other", "aux_on_vs_off", "v21_view_vs_off", "tera_pre_nonzero")
    checks = {
        "stream battles >= min": st.battles >= a.min_battles and js is not None and js["battles"] == st.battles,
        **{f"py {c}": st.bad[c] == 0 for c in ("alignment", "parity_off", "parity_on", "parity_aux", "py_off_old",
                                               "py_off_old_aux", "py_aux_flag", "py_v21_view", "py_pre_tera")},
        **{f"js {c}": js is not None and js[c] == 0 for c in js_keys},
        "flag on fills own pre-Tera tera (gen 9: every pre-Tera own slot)": st.pre_slots > 0 and st.pre_filled == st.pre_slots,
        "leak: no observer stream / observation difference": leak["stream_diverged"] == 0 and leak["obs_mismatch"] == 0
        and leak["compared_battles"] > 0,
        "leak: tau visible to the victim itself (test has power)": leak["victim_obs_differs"] == leak["compared_battles"],
        "volcarona fixture": vok,
    }
    summary = {"battles": st.battles, "decisions": st.decisions, "py_failures": dict(st.bad), "py_fields": dict(st.fields),
               "own_pre_tera_slots": st.pre_slots, "own_pre_tera_filled": st.pre_filled, "decisions_on_ne_off": st.on_ne_off,
               "js": js, "leak": leak, "volcarona": vlines, "checks": checks, "examples": st.examples,
               "pass": all(checks.values())}
    json.dump(summary, open(a.out, "w"), indent=1)
    print(f"stage 0: {st.battles} battles, {st.decisions} decisions (Python replay); JS {js and js['decisions']} decisions")
    print(f"  own pre-Tera slots {st.pre_slots}, filled with flag on {st.pre_filled}; decisions where on != off: {st.on_ne_off}")
    print(f"  leak: {leak['compared_battles']} battles compared, {leak['decisions_compared']} observer decisions, "
          f"{leak['mons_changed']} victim mons re-typed; stream diverged {leak['stream_diverged']}, obs mismatch "
          f"{leak['obs_mismatch']}, victim's own obs differs in {leak['victim_obs_differs']}")
    for ln in vlines:
        print("  volcarona:", ln)
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    for e in st.examples[:20] + (js or {}).get("examples", [])[:10] + leak.get("examples", [])[:10]:
        print("  -", e)
    print("STAGE0", "PASS" if summary["pass"] else "FAIL")
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
