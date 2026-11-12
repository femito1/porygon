"""Raw-network head-to-head on a duplicate-battle Showdown server: MODEL A vs MODEL B, sampled actions (as in
training and porygon.tournament), duplicate pairs (same seeds, teams swapped; the server pairs consecutive
battles, A's name matches PORYGON_DUPLICATE_GROUP='^wspa').

Start the server with PS_DIR=$GLORIA/.showdown-eval, PORYGON_DUPLICATE=<seed>, PORYGON_DUPLICATE_LOG=<out>/
battles.jsonl, PORYGON_DUPLICATE_GROUP='^wspa' (scripts/wsp/h2h_dup.sbatch does this), then
    python scripts/wsp/h2h_dup.py A.pt|A.zip B.pt|B.zip --port P --n-battles N --out DIR
    python scripts/wsp/h2h_dup.py --score DIR [DIR ...]       # pooled duplicate-pair win rate of A, 95% CI
"""

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np


def score(dirs: list[str]) -> str:
    means, w, n = [], 0, 0
    for d in map(Path, dirs):
        by = {}
        for line in open(d / "battles.jsonl"):
            b = json.loads(line)
            r = 0.5 if not b["winner"] else float(b["winner"].lower().startswith("wspa"))
            w += r
            n += 1
            by.setdefault(b["pair"], {})[b["game"]] = r
        means += [sum(p.values()) / 2 for p in by.values() if len(p) == 2]
    m = float(np.mean(means))
    sd = float(np.std(means, ddof=1))
    naive = 1.96 * math.sqrt(max(w / n * (1 - w / n), 1e-9) / n)
    return (f"A win rate {w / n:.1%} over {n} games (naive ±{naive:.1%}); duplicate pairs ({len(means)}): "
            f"{m:.1%} ± {1.96 * sd / math.sqrt(len(means)):.1%}")


async def run(a) -> None:
    import torch
    from poke_env import AccountConfiguration
    from porygon.env import PolicyPlayer, local_server
    from porygon.model import load_policy
    torch.set_num_threads(1)
    common = dict(battle_format=a.format, server_configuration=local_server(a.port),
                  max_concurrent_battles=a.concurrency, log_level=40)
    tag = np.random.randint(1e6)
    pa = PolicyPlayer(load_policy(a.a), account_configuration=AccountConfiguration(f"wspa{tag}", None), **common)
    pb = PolicyPlayer(load_policy(a.b), account_configuration=AccountConfiguration(f"wspb{tag}", None), **common)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a)))
    done = 0
    while done < a.n_battles:  # chunks so partial results are usable
        k = min(a.chunk, a.n_battles - done)
        await pa.battle_against(pb, n_battles=k)
        done += k
        print(f"{done} battles: A won {pa.n_won_battles}/{pa.n_finished_battles}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", nargs="?")
    p.add_argument("b", nargs="?")
    p.add_argument("--score", nargs="+")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--n-battles", type=int, default=2000)
    p.add_argument("--chunk", type=int, default=200, help="even, so pairs never straddle chunks")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--format", default="gen4randombattle")
    p.add_argument("--out", default="out/h2h")
    a = p.parse_args()
    if a.score:
        print(score(a.score))
        return
    assert a.chunk % 2 == 0
    asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
