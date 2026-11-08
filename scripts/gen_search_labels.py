"""Search-distillation data: our search plays itself on a local duplicate-battle Showdown server and BOTH sides
log every searched decision (porygon.decisions.LoggedSearchPlayer: action, legal mask, visit shares, what each
determinization assumed) plus their first-person replay logs, while the server logs both true teams
(PORYGON_DUPLICATE_LOG / PORYGON_DUPLICATE_TEAMS, scripts/build_eval_server.sh). The output has exactly the layout
of a benchmark run (battles.jsonl, a/decisions.jsonl + a/replays/, b/...), so scripts/convert_replays.py turns it
into encoder-agnostic game records and encoded shards (first-person mode, true teams) like any benchmark log;
the policy target is the visit-share distribution, the value target the game result.

gen9 needs PORYGON_GEN=9 and the gen9 engine first on PYTHONPATH; MODEL "none" is pure engine search.

    PORYGON_GEN=9 python scripts/gen_search_labels.py none --out runs/searchlabels-x --port 8000 \
        --n-battles 1000 --concurrency 3 --ms 250 --samples 4 --workers 7
"""

import argparse
import asyncio
import logging
import random
import sys
import time
from pathlib import Path

from poke_env import AccountConfiguration

sys.path.insert(0, str(Path(__file__).parent))
from vs_foulplay import guard_pool  # noqa: E402
from porygon import inference  # noqa: E402
from porygon.decisions import LoggedSearchPlayer  # noqa: E402
from porygon.env import local_server  # noqa: E402
from porygon.search import CONFIG, add_value_args, load_model  # noqa: E402


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", help='checkpoint, or "none" (pure engine search; gen9)')
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--n-battles", type=int, default=100, help="per chunk of games between progress lines")
    p.add_argument("--chunks", type=int, default=1, help="repeat --n-battles this many times (resumable output)")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--ms", type=int, default=250)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--prior-weight", type=float, default=None)
    p.add_argument("--c-puct", type=float, default=0.0)
    p.add_argument("--root-game", default=None)
    p.add_argument("--rm-gamma", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=None)
    add_value_args(p)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--search-timeout", type=float, default=30)
    p.add_argument("--format", default=inference.FORMAT)
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    policy = load_model(a.model, a.device)
    if a.prior_weight is None:
        a.prior_weight = 0.0 if policy is None else 0.25
    cfg = {k: getattr(a, k) for k in CONFIG}
    tag = random.randint(0, 10**5)
    common = dict(battle_format=a.format, server_configuration=local_server(a.port),
                  max_concurrent_battles=a.concurrency, log_level=40)
    # the "sla" player is the duplicate group anchor (PORYGON_DUPLICATE_GROUP='^sla')
    pa = LoggedSearchPlayer(policy, **cfg, workers=a.workers, out=a.out / "a",
                            account_configuration=AccountConfiguration(f"sla{tag}", None), **common)
    pb = LoggedSearchPlayer(policy, **cfg, workers=a.workers, out=a.out / "b",
                            account_configuration=AccountConfiguration(f"slb{tag}", None), **common)
    guard_pool(pa.pool, a.search_timeout)  # one process-wide pool: guarding it once covers both players
    print(f"search labels: {cfg}, {a.chunks} x {a.n_battles} battles -> {a.out}", flush=True)
    start = time.time()
    for c in range(a.chunks):
        await pa.battle_against(pb, n_battles=a.n_battles)
        s = {k: pa.stats[k] + pb.stats[k] for k in pa.stats}
        dt = time.time() - start
        print(f"[{dt / 3600:.2f} h] chunk {c + 1}/{a.chunks}: {pa.n_finished_battles} battles, "
              f"{s['searched']} searched decisions ({s['searched'] / dt:.2f}/s), fallbacks {s['fallbacks']}, "
              f"failed samples {s['failed_samples']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
