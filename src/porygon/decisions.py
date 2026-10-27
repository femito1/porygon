"""Benchmark recording for diagnosis (`scripts/analyze_benchmark.py`).

`LoggedSearchPlayer` writes one JSON line per searched decision to `<out>/decisions.jsonl`: the network's
probabilities, the search shares, the action played, the network's value of the position (V + Φ, as in
`porygon.lookahead`), and what each determinization assumed about the opponent's team next to what was
revealed. Every finished battle's protocol log goes to `<out>/replays/<battle tag>.log`. With the duplicate
server (`scripts/build_eval_server.sh`) the true teams are in its log, so the samples can be scored.
"""

import json
from pathlib import Path

import numpy as np
import poke_engine as pe
import torch
from poke_env.battle import AbstractBattle, Battle
from poke_env.data import GenData

from porygon.encoder import embed_battle
from porygon.lookahead import potential
from porygon.search import SearchPlayer


def opponent_samples(state_str: str) -> list[dict]:
    """The opponent's team as one determinization has it."""
    side = pe.State.from_string(state_str).side_two
    return [{"species": m.id.lower(), "item": m.item.lower(), "ability": m.ability.lower(),
             "moves": [mv.id.lower() for mv in m.moves if mv.id.lower() != "none"]} for m in side.pokemon]


def revealed(battle: Battle) -> list[dict]:
    """What the opponent has shown so far, in the engine's mon order (`search.convert`)."""
    return [{"species": m.species, "item_known": m.item != GenData.UNKNOWN_ITEM, "ability_known": m.ability is not None,
             "moves": list(m.moves), "active": m.active} for m in battle.opponent_team.values()]


class LoggedSearchPlayer(SearchPlayer):
    def __init__(self, *args, out: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = Path(out)
        (self.out / "replays").mkdir(parents=True, exist_ok=True)
        self.file = open(self.out / "decisions.jsonl", "a")

    def on_decision(self, battle: Battle, probs: np.ndarray, mask: np.ndarray, shares: np.ndarray) -> None:
        convs = self.last_convs.pop(battle.battle_tag, [])
        obs = {"observation": torch.as_tensor(embed_battle(battle)).unsqueeze(0).to(self.policy.device),
               "action_mask": torch.as_tensor(mask).unsqueeze(0).to(self.policy.device)}
        with torch.no_grad():
            value = float(self.policy.predict_values(obs)[0, 0])
        phi = potential(pe.State.from_string(convs[0].state_str))[0] if convs else 0.0
        rec = {"battle": battle.battle_tag, "turn": battle.turn, "player": self.username,
               "action": self.combine(shares, probs, mask), "mask": mask.astype(int).tolist(),
               "probs": np.round(probs, 4).tolist(), "shares": np.round(shares, 4).tolist(),
               "value": round(value + phi, 4), "revealed": revealed(battle),
               "samples": [opponent_samples(c.state_str) for c in convs]}
        self.file.write(json.dumps(rec) + "\n")
        self.file.flush()

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super()._battle_finished_callback(battle)
        lines = ["|".join(ev) for ev in battle._replay_data]
        (self.out / "replays" / f"{battle.battle_tag}.log").write_text("\n".join(lines) + "\n")
