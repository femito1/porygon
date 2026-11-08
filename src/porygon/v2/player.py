"""poke-env player for model v2 (`porygon.v2.model.Net`): encode -> forward -> sample (or argmax) over legal actions.

    net = load("runs/x/final_model.pt")
    player = NetPlayer(net, battle_format="gen9randombattle", server_configuration=local_server(8000))
"""

from typing import Any

import numpy as np
import torch
from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, Player

from porygon.v2.encoder import encode
from porygon.v2.model import Net, obs_to_tensors


def _device(net: Net) -> torch.device:
    return next(net.parameters()).device


@torch.no_grad()
def policy_probs(net: Net, battle: AbstractBattle) -> tuple[np.ndarray, np.ndarray]:
    """(probabilities [A] over poke-env actions, legal mask [A]); probabilities are 0 off the mask."""
    obs = encode(battle)
    out = net(obs_to_tensors([obs], _device(net)), aux=False)
    mask = obs["action_mask"]
    logits = out["logits"][0].double().cpu().numpy()  # float64: rng.choice needs probabilities summing to 1 ± 1e-8
    probs = np.zeros_like(mask, dtype=np.float64)
    legal = mask > 0
    if legal.any():
        z = logits[legal] - logits[legal].max()
        e = np.exp(z)
        probs[legal] = e / e.sum()
    return probs, mask


class NetPlayer(Player):
    """Plays a v2 Net. `deterministic` = argmax instead of sampling."""

    def __init__(self, net: Net, *args: Any, deterministic: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.net = net
        self.deterministic = deterministic
        self.rng = np.random.default_rng()

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        probs, mask = policy_probs(self.net, battle)
        if not (mask > 0).any():
            return self.choose_random_move(battle)
        a = int(np.argmax(probs)) if self.deterministic else int(self.rng.choice(len(probs), p=probs))
        return SinglesEnv.action_to_order(np.int64(a), battle, strict=False)
