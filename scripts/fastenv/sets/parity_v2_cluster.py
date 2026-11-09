"""Run scripts/fastenv/parity_v2.py on the cluster, where the netv2 reference (porygon.v2) is not at the worktree
sibling path export_tables_v2.py defaults to: netv2's cluster copy is appended to our package path first.

    PORYGON_PS=/path/to/ps PYTHONPATH=src $PY scripts/fastenv/sets/parity_v2_cluster.py --record 100 gen9randombattle --seed 21
    NETV2_SRC overrides the reference (default /vol/bitbucket/fr1226/porygon-ws-netv2/src, read-only).
"""
import os
import runpy
import sys
from pathlib import Path

FASTENV = Path(__file__).resolve().parents[1]  # scripts/fastenv
ROOT = FASTENV.parents[1]
NETV2_SRC = Path(os.environ.get("NETV2_SRC", "/vol/bitbucket/fr1226/porygon-ws-netv2/src"))

sys.path.insert(0, str(ROOT / "src"))
import porygon  # noqa: E402  (ours: porygon.encoder, data/)

porygon.__path__.append(str(NETV2_SRC / "porygon"))  # porygon.v2 resolves here
script = FASTENV / "parity_v2.py"
sys.argv = [str(script)] + sys.argv[1:]
runpy.run_path(str(script), run_name="__main__")
