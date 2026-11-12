"""Fixed diagnostic position set for network v2: Showdown games of the v1 500M net against itself (sampling),
both sides recorded; at every decision the v2 observation (porygon.v2.encoder.encode), v1's policy probabilities,
v1's value and the sampled action; the winner per game. Same positions are used for every v2 checkpoint.

    python scripts/wsp/v2diag_record.py ckpt/ref/m500.zip --port P --n-battles 150 --out v2diag/pos/p0
"""

import argparse
import asyncio
import pickle
from pathlib import Path

import numpy as np
import torch
from poke_env import AccountConfiguration
from poke_env.environment import SinglesEnv
from poke_env.player import DefaultBattleOrder

from porygon.encoder import embed_battle
from porygon.env import PolicyPlayer, local_server
from porygon.model import load_policy
from porygon.v2.encoder import encode as encode_v2


class Rec(PolicyPlayer):
    def __init__(self, *a, store: dict, **kw):
        super().__init__(*a, **kw)
        self.store = store

    def choose_move(self, battle):
        if battle.wait:
            return DefaultBattleOrder()
        x = embed_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle))
        obs = {"observation": torch.as_tensor(x).unsqueeze(0), "action_mask": torch.as_tensor(mask).unsqueeze(0)}
        with torch.no_grad():
            dist = self.policy.get_distribution(obs)
            a = int(dist.sample().item())
            probs = dist.distribution.probs[0].numpy().astype(np.float32)
            v = float(self.policy.predict_values(obs))
        try:
            o2 = encode_v2(battle)
        except Exception as e:  # keep going; count failures
            self.store.setdefault("_fail", []).append(repr(e)[:200])
            o2 = None
        if o2 is not None:
            self.store.setdefault("rows", []).append({"gid": (self.username, battle.battle_tag), "obs": o2,
                                                      "v1_probs": probs, "v1_value": v, "action": a,
                                                      "turn": battle.turn})
        return SinglesEnv.action_to_order(np.int64(a), battle, strict=False)

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        self.store.setdefault("won", {})[(self.username, battle.battle_tag)] = bool(battle.won)


async def run(a):
    torch.set_num_threads(1)
    pol = load_policy(a.model)
    store: dict = {}
    common = dict(battle_format=a.format, server_configuration=local_server(a.port),
                  max_concurrent_battles=a.concurrency, log_level=40)
    tag = np.random.randint(1e6)
    p1 = Rec(pol, store=store, account_configuration=AccountConfiguration(f"vda{tag}", None), **common)
    p2 = Rec(pol, store=store, account_configuration=AccountConfiguration(f"vdb{tag}", None), **common)
    await p1.battle_against(p2, n_battles=a.n_battles)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in store.get("rows", []) if r["gid"] in store.get("won", {})]
    for r in rows:
        r["won"] = store["won"][r["gid"]]
    with open(out / "rows.pkl", "wb") as f:
        pickle.dump(rows, f)
    print(f"done: {a.n_battles} battles, {len(rows)} decisions, {len(store.get('_fail', []))} encode failures "
          f"{store.get('_fail', [])[:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n-battles", type=int, default=150)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--format", default="gen4randombattle")
    ap.add_argument("--out", default="v2diag/pos")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
