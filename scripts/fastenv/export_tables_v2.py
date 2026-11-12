"""Export the static tables of observation spec v2 for the JS encoder (fastenv/js/encoder_v2.js), FROM the netv2
reference implementation (porygon.v2, branch ws/netv2, read-only), so the set posterior and every id are
bit-identical to `porygon.v2.encoder.encode` / `aux_targets`.

    python scripts/fastenv/export_tables_v2.py [--netv2 PATH/TO/src]   # writes fastenv/js/tables/*_v2.json

Outputs (fastenv/js/tables/):
  spec_v2.json        SPEC_VERSION, fields(gen) / aux_fields(gen) as [name, dtype, shape] (spec order), MON_NUM,
                      FIELD_NUM, TYPES (id = index + 1), WEATHERS, TERRAINS, STATUSES, BOOSTS, KM/KI/KA, TERA_OFFSET
  gen{4,9}_lib_v2.json  reference `library(gen)`: species -> [level, [[moves, ability, item, evs(6), ivs(6), n,
                      tera], ...]]; setSpecies (pokedex ids whose `set_species` differs from the id);
                      moveType (`move_type_of(gen, id)` for every GenData and library move id)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NETV2_SRC = ROOT / "src"
OUT = ROOT / "fastenv" / "js" / "tables"
DT = {"float32": "f32", "int32": "i32", "uint32": "u32", "int16": "i16", "uint16": "u16", "uint8": "u8",
      "int8": "i8"}


def import_reference(netv2_src: Path = NETV2_SRC):
    """Our `porygon` package first; its `v2` subpackage resolved from the netv2 worktree (read-only)."""
    sys.path.insert(0, str(ROOT / "src"))
    import porygon
    p = str(Path(netv2_src) / "porygon")
    if p not in porygon.__path__:
        porygon.__path__.append(p)
    import porygon.v2.encoder as ref
    import porygon.v2.spec as spec
    return ref, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netv2", type=Path, default=NETV2_SRC)
    a = ap.parse_args()
    ref, spec = import_reference(a.netv2)
    import numpy as np

    def flist(d):
        return [[k, DT[np.dtype(dt).name], list(shape)] for k, (shape, dt) in d.items()]
    out = {"version": spec.SPEC_VERSION, "fields": {g: flist(spec.fields(g)) for g in (4, 9)},
           "auxFields": {g: flist(spec.aux_fields(g)) for g in (4, 9)},
           "MON_NUM": spec.MON_NUM, "FIELD_NUM": spec.FIELD_NUM, "TYPES": spec.TYPES, "WEATHERS": spec.WEATHERS,
           "TERRAINS": spec.TERRAINS, "STATUSES": spec.STATUSES, "BOOSTS": spec.BOOSTS, "STATS": spec.STATS,
           "KM": spec.KM, "KI": spec.KI, "KA": spec.KA, "TERA_OFFSET": spec.TERA_OFFSET,
           "N_OPP_CLASSES": spec.N_OPP_CLASSES}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spec_v2.json").write_text(json.dumps(out, separators=(",", ":")))
    for gen in (4, 9):
        lib = ref.library(gen)
        species = {sp: [lvl, [[list(s.moves), s.ability, s.item, [s.evs[k] for k in spec.STATS],
                               [s.ivs[k] for k in spec.STATS], s.n, s.tera] for s in sets]]
                   for sp, (lvl, sets) in lib.items()}
        dex = ref.gendata(gen).pokedex
        set_species = {k: ref.set_species(gen, k) for k in dex if ref.set_species(gen, k) != k}
        ids = set(ref.gendata(gen).moves) | {m for _, sets in lib.values() for s in sets for m in s.moves}
        move_type = {m: ref.move_type_of(gen, m) for m in sorted(ids)}
        (OUT / f"gen{gen}_lib_v2.json").write_text(json.dumps(
            {"gen": gen, "species": species, "setSpecies": set_species, "moveType": move_type},
            separators=(",", ":")))
    for p in sorted(OUT.glob("*_v2.json")):
        print(f"{p.name}: {p.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
