"""Export the static tables the JS tracker/encoder (fastenv/js/{tracker,encoder_v1}.js) need, FROM PYTHON, so every
static feature is bit-identical to `porygon.encoder.embed_battle` on poke-env 0.16.1 objects.

    python scripts/fastenv/export_tables_v1.py            # writes fastenv/js/tables/*.json

Outputs (fastenv/js/tables/):
  common_v1.json   encoder layout dims, enum name lists in encoder order, poke-env enum member tables (aliases
                   resolved), effect property sets, vocab indices (porygon/data/vocab.json -> INDEX), move-id sets
  gen{4,9}_v1.json poke-env GenData for that gen, reduced to what the tracker/encoder read:
                   dex[id]   = [baseStats(hp..spe), types, possible abilities (ids, dict order), weightkg,
                                baseSpecies id, forme-change-ability id or null, abilities["0"] id]
                   moves[id] = {s: static MOVE_DIM feature vector from porygon.encoder.encode_move (dynamic slots
                                filled in by JS), pp, type, bp, tgt, mp (mustpressure flag), st (Move.should_be_stored),
                                err (encode_move raised)}
                   chart     = GenData.type_chart (defender -> attacker -> multiplier)
  {format}_sets_v1.json  species -> 24 movepool vocab ids (encoder's opponent prior), for gen4/gen9 randombattle
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from poke_env.battle import (  # noqa: E402
    Effect, Field, Move, PokemonType, SideCondition, Status, Weather,
)
from poke_env.battle import effect as effect_mod  # noqa: E402
from poke_env.battle import move as move_mod  # noqa: E402
from poke_env.battle.side_condition import STACKABLE_CONDITIONS  # noqa: E402
from poke_env.data import GenData, to_id_str  # noqa: E402

import porygon.encoder as enc  # noqa: E402

OUT = ROOT / "fastenv" / "js" / "tables"
FORMATS = ["gen4randombattle", "gen9randombattle"]


class _FakeUser:
    types: list = []


def static_move(mid: str, gen: int) -> dict:
    m = Move(mid, gen=gen)
    row = {"pp": m.entry["pp"], "type": m.entry["type"], "bp": m.entry.get("basePower", 0),
           "tgt": m.entry.get("target"), "mp": int("mustpressure" in m.entry.get("flags", {})),
           "st": int(Move.should_be_stored(mid, gen)), "err": 0}
    try:
        row["s"] = enc.encode_move(m, _FakeUser(), None, gen, set())
    except Exception as e:  # poke-env would raise on this move: JS raises too
        row["s"], row["err"] = None, 1
        row["why"] = repr(e)
    return row


def gen_tables(gen: int) -> dict:
    d = GenData.from_gen(gen)
    dex = {}
    for sid, e in d.pokedex.items():
        forme = e.get("forme", "")
        fca = None
        if forme and (forme.startswith("Mega") or forme in ["Primal", "Stellar", "Terastal"]
                      or forme.endswith("-Tera")):
            fca = to_id_str(e["abilities"]["0"])
        dex[sid] = [[e["baseStats"][s] for s in enc.STATS], e["types"],
                    [to_id_str(a) for a in e["abilities"].values()], e["weightkg"],
                    to_id_str(e["baseSpecies"]), fca, to_id_str(e["abilities"].get("0", ""))]
    moves = {mid: static_move(mid, gen) for mid in d.moves}
    return {"gen": gen, "dex": dex, "moves": moves, "chart": d.type_chart}


def common() -> dict:
    E = effect_mod
    return {
        "dims": {k: getattr(enc, k) for k in ["OBS_DIM", "FIELD_DIM", "MON_DIM", "MON_NUM_DIM", "MOVE_DIM",
                                               "N_CAT", "MOVEPOOL", "N_MOVES", "ACTIVE_FLAG"]},
        "types": [t.name for t in enc.TYPES],
        "allTypes": [t.name for t in PokemonType],
        "statuses": [s.name for s in enc.STATUSES],
        "allStatuses": [s.name for s in Status],
        "boosts": enc.BOOSTS, "stats": enc.STATS,
        "effects": [e.name for e in enc.EFFECTS],
        "allEffects": [e.name for e in Effect],
        "effTurnCountable": sorted(e.name for e in E._TURN_COUNTER_EFFECTS),
        "effActionCountable": sorted(e.name for e in E._ACTION_COUNTER_EFFECTS),
        "effEndsOnTurn": sorted(e.name for e in E._ENDS_ON_TURN_EFFECTS),
        "effBreaksProtect": sorted(e.name for e in E._PROTECT_BREAKING_EFFECTS),
        "weathers": [w.name for w in enc.WEATHERS],
        "weatherMembers": {k: v.name for k, v in Weather.__members__.items()},
        "fields": [f.name for f in enc.FIELDS],
        "fieldMembers": {k: v.name for k, v in Field.__members__.items()},
        "screens": [c.name for c in enc.SCREENS],
        "hazards": [[c.name, n] for c, n in enc.HAZARDS.items()],
        "sideMembers": {k: v.name for k, v in SideCondition.__members__.items()},
        "stackable": {c.name: n for c, n in STACKABLE_CONDITIONS.items()},
        "categories": [c.name for c in enc.CATEGORIES],
        "vocab": enc.INDEX,
        "protectCounterMoves": sorted(move_mod._PROTECT_COUNTER_MOVES),
        "specialMoves": sorted(move_mod.SPECIAL_MOVES),
        "unknownItem": GenData.UNKNOWN_ITEM,
    }


def sets_table(fmt: str) -> dict:
    out = {}
    for species, prior in enc.randbats_sets(fmt).items():
        pool = [enc.idx("moves", m) for m in prior.get("moves", [])[:enc.MOVEPOOL]]
        out[species] = pool + [enc.PAD] * (enc.MOVEPOOL - len(pool))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "common_v1.json").write_text(json.dumps(common(), separators=(",", ":")))
    for gen in (4, 9):
        (OUT / f"gen{gen}_v1.json").write_text(json.dumps(gen_tables(gen), separators=(",", ":")))
    for fmt in FORMATS:
        (OUT / f"{fmt}_sets_v1.json").write_text(json.dumps(sets_table(fmt), separators=(",", ":")))
    for p in sorted(OUT.glob("*.json")):
        print(f"{p.name}: {p.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
