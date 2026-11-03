"""Benchmark recording for diagnosis (`scripts/analyze_benchmark.py`).

`LoggedSearchPlayer` writes one JSON line per searched decision to `<out>/decisions.jsonl`: the network's
probabilities, the search shares, the action played, the network's value of the position (V + Φ, as in
`porygon.lookahead`), and what each determinization assumed about the opponent's team next to what was
revealed. Every finished battle's protocol log goes to `<out>/replays/<battle tag>.log`. With the duplicate
server (`scripts/build_eval_server.sh`) the true teams are in its log, so the samples can be scored.

With log_oracle (needs the duplicate server's true teams, `oracle_dir`) each record also carries what offline
network studies need: the network's observation ("obs", float16, base64) and the position converted with the
opponent's whole true team ("oracle": engine state string plus the ids that map its choices to action indices),
so an oracle search can later label which action was right without replaying the battle.
"""

import base64
import json
import random
from pathlib import Path

import numpy as np
import poke_engine as pe
import torch
from poke_env.battle import AbstractBattle, Battle
from poke_env.data import GenData

from porygon.encoder import embed_battle
from porygon.lookahead import potential
from porygon.search import History, SearchPlayer, convert


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
    def __init__(self, *args, out: str | Path, log_oracle: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_oracle = log_oracle
        assert not log_oracle or self.oracle_dir, "log_oracle needs the true teams (oracle_dir)"
        self.out = Path(out)
        (self.out / "replays").mkdir(parents=True, exist_ok=True)
        self.file = open(self.out / "decisions.jsonl", "a")

    def on_decision(self, battle: Battle, probs: np.ndarray, mask: np.ndarray, shares: np.ndarray) -> None:
        convs = self.last_convs.pop(battle.battle_tag, [])
        x = embed_battle(battle)
        obs = {"observation": torch.as_tensor(x).unsqueeze(0).to(self.policy.device),
               "action_mask": torch.as_tensor(mask).unsqueeze(0).to(self.policy.device)}
        with torch.no_grad():
            value = float(self.policy.predict_values(obs)[0, 0])
        phi = potential(pe.State.from_string(convs[0].state_str))[0] if convs else 0.0
        rec = {"battle": battle.battle_tag, "turn": battle.turn, "player": self.username,
               "action": self.last_action[battle.battle_tag], "mask": mask.astype(int).tolist(),
               "probs": np.round(probs, 4).tolist(), "shares": np.round(shares, 4).tolist(),
               "value": round(value + phi, 4), "revealed": revealed(battle),
               "samples": [opponent_samples(c.state_str) for c in convs]}
        if self.log_oracle:
            rec["obs"] = base64.b64encode(np.asarray(x, dtype=np.float16).tobytes()).decode()
            rec["oracle"] = self.oracle_record(battle)
        self.file.write(json.dumps(rec) + "\n")
        self.file.flush()

    def oracle_record(self, battle: Battle) -> dict | None:
        try:
            tag = battle.battle_tag
            teams = json.loads((Path(self.oracle_dir) / f"{tag}.json").read_text())
            truth = teams["p2team"] if teams["p1"] == self.username else teams["p1team"]
            seen = self.weather_seen.get(tag)
            conv = convert(battle, random.Random(0), seen[1] if seen else None, History(battle), truth, True)
            return {"state": conv.state_str, "switch_ids": conv.switch_ids, "move_ids": conv.move_ids}
        except Exception as e:  # a missing team file, a conversion failure: the record just has no oracle
            return {"error": f"{type(e).__name__}: {e}"[:200]}

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super()._battle_finished_callback(battle)
        lines = ["|".join(ev) for ev in battle._replay_data]
        (self.out / "replays" / f"{battle.battle_tag}.log").write_text("\n".join(lines) + "\n")
