"""Parity: JS encoder_v2 (fastenv/js/encoder_v2.js on the tracker) vs netv2's reference observation spec v2
(`porygon.v2.encoder.encode` / `aux_targets`, branch ws/netv2, imported read-only).

Streaming mode: seeded random battles from scripts/fastenv/record_battles.js --v2 (node -> python pipe, no temp
files); both sides' streams are replayed into poke-env exactly as Player._handle_battle_message would (parity_v1's
`handle`), and at every decision of either side every observation field and every aux target is compared
(ints exact, floats |a - b| <= 1e-4 = porygon.v2.parity.TOLERANCE). aux uses the other side's battle replayed to
the same point as the JS opponent tracker (k events).

Fixture mode: netv2's recorded fixtures (real local server); the JS observations go through the reference's own
`porygon.v2.parity.check(FIXTURE, candidate=obs.npz)`, and the aux targets are compared to the fixture's.

    python scripts/fastenv/parity_v2.py --record 300 gen9randombattle [--seed 1]
    python scripts/fastenv/parity_v2.py --fixture /path/to/fixtures/gen9 [--keep-candidate obs.npz]
"""
import argparse
import base64
import gzip
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export_v2 = _load("export_tables_v2", HERE / "export_tables_v2.py")
ref, spec = export_v2.import_reference()
parity_v1 = _load("parity_v1", HERE / "parity_v1.py")  # poke-env message handling (and its logging setup)
from poke_env.battle import Battle  # noqa: E402
from poke_env.data import GenData  # noqa: E402

TOL = 1e-4
NP = {"f32": np.float32, "i32": np.int32, "u32": np.uint32, "i16": np.int16, "u16": np.uint16, "u8": np.uint8,
      "i8": np.int8}


def dense(sp, shape, dtype) -> np.ndarray:
    out = np.zeros(int(np.prod(shape)), dtype=dtype)
    ix = np.frombuffer(base64.b64decode(sp[0]), dtype=np.uint16)
    out[ix] = np.frombuffer(base64.b64decode(sp[1]), dtype=dtype)
    return out.reshape(shape)


def js_dict(rec: dict, gen: int, aux: bool) -> dict:
    fl = (spec.aux_fields if aux else spec.fields)(gen)
    return {k: dense(rec[k], shape, np.dtype(dt)) for k, (shape, dt) in fl.items()}


class Stats:
    def __init__(self):
        self.battles = self.decisions = self.bad = 0
        self.fields = Counter()  # field -> decisions with a mismatch
        self.entries = Counter()  # field -> mismatching entries
        self.examples = []


def compare(st: Stats, refd: dict, got: dict, label: str, limit: int, prefix="") -> bool:
    ok = True
    for k, r in refd.items():
        g = got[k]
        if np.issubdtype(r.dtype, np.integer):
            bad = np.argwhere(g != r)
        else:
            bad = np.argwhere(~np.isclose(g, r, atol=TOL, rtol=0))
        if len(bad):
            ok = False
            st.fields[prefix + k] += 1
            st.entries[prefix + k] += len(bad)
            if len(st.examples) < limit:
                i = tuple(bad[0])
                st.examples.append(f"{label}: {prefix}{k} {len(bad)} entries, first {i}: js {g[i]} ref {r[i]}")
    return ok


def check_side(rec: dict, s: int, st: Stats, limit: int):
    fmt = rec["format"]
    gen = GenData.from_format(fmt).gen
    tag = f"battle-{fmt}-{rec['id']}"
    mk = lambda u: Battle(tag, u, logging.getLogger("parity"), gen=gen)  # noqa: E731
    me, opp = mk(f"p{s + 1}"), mk(f"p{2 - s}")
    hdr = "|init|battle\n|title|p1 vs. p2\n|j|☆p1\n|j|☆p2"
    parity_v1.handle(me, hdr)
    parity_v1.handle(opp, hdr)
    side, oside = rec["sides"][s], rec["sides"][1 - s]

    def line(kind, text):
        return text if kind == "L" else (f"|request|{text}" if kind == "R" else f"|error|{text.removeprefix('|error|')}")
    decs, k, j_opp = side["dec"], 0, 0
    for i, (kind, text) in enumerate(side["ev"]):
        if side["crash"] and side["crash"]["e"] == i:
            return
        decision = parity_v1.handle(me, line(kind, text))
        if not decision:
            continue
        if k >= len(decs) or decs[k]["e"] != i:
            st.bad += 1
            st.examples.append(f"{tag} p{s + 1}: decision alignment at event {i}")
            return
        d = decs[k]
        k += 1
        while j_opp < d["k"] and not (oside["crash"] and oside["crash"]["e"] <= j_opp):
            parity_v1.handle(opp, line(*oside["ev"][j_opp]))
            j_opp += 1
        st.decisions += 1
        obs = ref.encode(me)
        aux = ref.aux_targets(me, None if oside["crash"] else opp, obs)
        label = f"{tag} p{s + 1} ev {i} turn {me.turn}"
        ok = compare(st, obs, js_dict(d["o2"], gen, False), label, limit)
        ok = compare(st, aux, js_dict(d["a2"], gen, True), label, limit, "aux/") and ok
        st.bad += not ok


def stream(n: int, fmt: str, seed: int, st: Stats, limit: int):
    proc = subprocess.Popen(["node", str(HERE / "record_battles.js"), str(n), fmt, "-", str(seed), "--v2"],
                            stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    for ln in proc.stdout:
        rec = json.loads(ln)
        st.battles += 1
        for s in (0, 1):
            check_side(rec, s, st, limit)
    if proc.wait() != 0:
        raise SystemExit("record_battles.js failed")


def fixture(fx: Path, st: Stats, limit: int, keep: Path | None) -> int:
    from porygon.v2 import parity as refparity
    battles, z = refparity.load(fx)
    gen = int(battles[0]["format"][3])
    order = {}
    for d, b in enumerate(z["battle"]):
        order.setdefault(int(b), []).append(d)
    jobs = "".join(json.dumps({**battles[b], "dec": [[int(z["k1"][d]), int(z["k2"][d])] for d in ds]}) + "\n"
                   for b, ds in order.items())
    out = subprocess.run(["node", str(HERE / "v2_fixture_encode.js")], input=jobs, capture_output=True, text=True,
                         check=True)
    print(out.stderr.strip())
    rows = [json.loads(x) for x in out.stdout.splitlines()]
    flat = [d for ds in order.values() for d in ds]
    n = len(z["battle"])
    obs = {k: np.zeros((n, *shape), dt) for k, (shape, dt) in spec.fields(gen).items()}
    for d, r in zip(flat, rows):
        for k, v in js_dict(r["o2"], gen, False).items():
            obs[k][d] = v
        st.decisions += 1
        refaux = {k[4:]: v[d] for k, v in z.items() if k.startswith("aux/")}
        got = js_dict(r["a2"], gen, True)
        st.bad += not compare(st, refaux, got, f"fixture decision {d}", limit, "aux/")
    cand = keep or Path(tempfile.mkstemp(suffix=".npz")[1])
    np.savez_compressed(cand, **{f"obs/{k}": v for k, v in obs.items()})
    n_bad = refparity.check(fx, cand)  # the reference's own check (prints "x/y decisions match")
    if keep is None:
        cand.unlink()
    return n_bad


def report(st: Stats, label: str):
    print(f"{label}: {st.battles} battles, {st.decisions} decisions, {st.bad} with any mismatch")
    for k, v in sorted(st.fields.items()):
        print(f"    {k}: {v} decisions, {st.entries[k]} entries")
    for e in st.examples:
        print("  -", e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", nargs=2, metavar=("N", "FORMAT"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fixture", type=Path, action="append", default=[])
    p.add_argument("--keep-candidate", type=Path)
    p.add_argument("--max-report", type=int, default=20)
    a = p.parse_args()
    logging.getLogger("poke-env").setLevel(logging.CRITICAL)
    logging.getLogger("parity").setLevel(logging.CRITICAL)
    fail = False
    for fx in a.fixture:
        st = Stats()
        n_bad = fixture(fx, st, a.max_report, a.keep_candidate)
        report(st, f"fixture {fx} (aux vs stored; obs via porygon.v2.parity.check)")
        fail |= bool(n_bad or st.bad)
    if a.record:
        st = Stats()
        stream(int(a.record[0]), a.record[1], a.seed, st, a.max_report)
        report(st, f"{a.record[1]} x{a.record[0]} (seed {a.seed}), spec {spec.SPEC_VERSION}")
        fail |= bool(st.bad)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
