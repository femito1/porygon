"""On-policy self-play rollouts of one network against itself on Showdown, logged like PPO's rollout buffer:
for every decision of both sides: obs (float16), mask, sampled action, log-prob, entropy, V(s), Φ(s) (poke-env's
shaping state value, porygon.expert.state_value) and, at the end of each battle, Φ_end (incl. ±1).
Output: <out>/rollouts-<k>.npz shards with per-decision arrays and `game`, `side`, `t`, plus `phi_end` per game.

    python scripts/wsp/selfplay_log.py MODEL --port P --n-battles 600 --out out/selfplay/run1
"""

import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch
from poke_env import AccountConfiguration
from poke_env.environment import SinglesEnv
from poke_env.player import DefaultBattleOrder

from porygon.encoder import embed_battle
from porygon.env import PolicyPlayer, local_server
from porygon.expert import state_value
from porygon.model import load_policy


class Logger:
    def __init__(self, out: Path):
        self.out, self.rows, self.games, self.k = out, [], {}, 0
        out.mkdir(parents=True, exist_ok=True)

    def flush(self, final=False):
        done = [g for g, v in self.games.items() if "phi_end" in v]
        if not done or (len(done) < 100 and not final):
            return
        keep = set(done)
        rows = [r for r in self.rows if r["gid"] in keep]
        self.rows = [r for r in self.rows if r["gid"] not in keep]
        gids = sorted(keep)
        gi = {g: i for i, g in enumerate(gids)}
        arr = {k: np.asarray([r[k] for r in rows]) for k in ("obs", "mask", "action", "logp", "ent", "value", "phi",
                                                              "turn")}
        arr["obs"] = arr["obs"].astype(np.float16)
        arr["game"] = np.asarray([gi[r["gid"]] for r in rows], np.int32)
        arr["phi_end"] = np.asarray([self.games[g]["phi_end"] for g in gids], np.float32)
        np.savez_compressed(self.out / f"rollouts-{self.k:03d}.npz", **arr)
        self.k += 1
        for g in gids:
            del self.games[g]


class LoggingPlayer(PolicyPlayer):
    def __init__(self, *args, logger: Logger, **kw):
        super().__init__(*args, **kw)
        self.log = logger

    def choose_move(self, battle):
        if battle.wait:
            return DefaultBattleOrder()
        x = embed_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle))
        obs = {"observation": torch.as_tensor(x).unsqueeze(0), "action_mask": torch.as_tensor(mask).unsqueeze(0)}
        with torch.no_grad():
            dist = self.policy.get_distribution(obs)
            a = dist.sample()
            logp = dist.log_prob(a)
            ent = dist.entropy()
            v = self.policy.predict_values(obs)
        gid = (self.username, battle.battle_tag)
        self.log.games.setdefault(gid, {})
        self.log.rows.append({"gid": gid, "obs": x, "mask": mask, "action": int(a.item()), "logp": float(logp),
                              "ent": float(ent), "value": float(v), "phi": float(state_value(battle)),
                              "turn": battle.turn})
        return SinglesEnv.action_to_order(np.int64(a.item()), battle, strict=False)

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        gid = (self.username, battle.battle_tag)
        if gid in self.log.games:
            self.log.games[gid]["phi_end"] = float(state_value(battle))
        self.log.flush()


async def run(a):
    torch.set_num_threads(1)
    pol = load_policy(a.model)
    log = Logger(Path(a.out))
    common = dict(battle_format=a.format, server_configuration=local_server(a.port),
                  max_concurrent_battles=a.concurrency, log_level=40)
    tag = np.random.randint(1e6)
    p1 = LoggingPlayer(pol, logger=log, account_configuration=AccountConfiguration(f"spa{tag}", None), **common)
    p2 = LoggingPlayer(pol, logger=log, account_configuration=AccountConfiguration(f"spb{tag}", None), **common)
    await p1.battle_against(p2, n_battles=a.n_battles)
    log.flush(final=True)
    print(f"done: {a.n_battles} battles, {log.k} shards")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n-battles", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--format", default="gen4randombattle")
    ap.add_argument("--out", default="out/selfplay")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
