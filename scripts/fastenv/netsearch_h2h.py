"""netv2's duplicate-pair head-to-head harness (`porygon.v2.evaluate`, imported read-only from netv2's tree) with one
more player spec: `netsearch:NET[,margin=M][,temperature=T][,c_puct=C][,prior_weight=W][,ms=MS][,samples=K]
[,workers=N][,seed=S][,device=D][,log=PATH]` = porygon.netsearch's margin-rule hybrid (`load_spec`). Every other spec
(search:none, nets, baselines) goes to the harness's own load_player, so both sides play exactly as in netv2's runs.

    PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$C/src $PY scripts/fastenv/netsearch_h2h.py \
        'netsearch:NET.pt,margin=0.015,ms=1000,samples=4,workers=11,log=OUT/gate.jsonl' \
        'search:none,ms=1000,samples=4,eval_config=stock,workers=11' --n-battles 400 --concurrency 3 --port P --out OUT
on a duplicate-battle server set up as netv2's scripts/netv2_h2h.sbatch does (PORYGON_DUPLICATE / _LOG / _GROUP).
`--check SPEC` builds the player without a server and prints its settings (no games).
Note: `porygon.search` resolves to this tree's (= main's) module, so `search:NET` with a v2 net is not available here;
the v2-prior variant is `netsearch:NET,margin=-inf,prior_weight=W`.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import porygon  # noqa: E402

_NV = Path(os.environ.get("NETV2_SRC", "/vol/bitbucket/fr1226/porygon-ws-netv2/src")) / "porygon"
if str(_NV) not in porygon.__path__:
    porygon.__path__.append(str(_NV))

from porygon.v2 import evaluate as E  # noqa: E402

_load_player = E.load_player


def load_player(spec: str, **kwargs):
    """The harness's load_player, plus `netsearch:` specs. The harness passes device / workers / deterministic /
    temperature defaults as keyword arguments; for a netsearch spec its own options win (as in load_player)."""
    if not spec.startswith("netsearch:"):
        return _load_player(spec, **kwargs)
    from porygon import netsearch as NS
    extra = []
    for key in ("device", "workers"):
        value = kwargs.pop(key, None)
        if value is not None and f",{key}=" not in spec:
            extra.append(f"{key}={value}")
    kwargs.pop("deterministic", None)
    kwargs.pop("temperature", None)
    return NS.load_spec(",".join([spec] + extra), **kwargs)


E.load_player = load_player  # play() looks it up by module global

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        fmt = f"gen{os.environ.get('PORYGON_GEN', '4')}randombattle"
        p = load_player(sys.argv[2], battle_format=fmt, start_listening=False)
        print(type(p).__name__, {k: getattr(p, k, None) for k in ("margin", "temperature", "deterministic", "c_puct",
                                                                   "net_prior_weight", "ms", "samples", "eval_config")})
        print("net gen", getattr(getattr(p, "net", None) or getattr(p, "model", None), "gen", None),
              "pool workers", getattr(getattr(p, "pool", None), "n", None))  # (a raw-net player has no pool)
        sys.exit(0)
    E.main()
