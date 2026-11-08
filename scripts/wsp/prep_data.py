"""Decode the logged real-game decisions (search --log-oracle runs) into one array file for network studies.

Output (npz, one row per logged decision that carries an observation):
  obs      float16 [N, OBS_DIM]   the network's observation (embed_battle) at that decision
  mask     int8    [N, 10]        legal actions
  probs    float32 [N, 10]        the 500M network's policy as logged
  shares   float32 [N, 10]        the search's visit shares (what it played from)
  action   int8    [N]            action played (by the search)
  value    float32 [N]            logged network value V + Φ (porygon.lookahead.potential)
  turn     int16   [N]
  game     int32   [N]            index into games.json (battle tag, run, outcome)
  won      float32 [N]            1 win / 0 loss / 0.5 tie for the logged player
  run      int8    [N]            0 = 309116 (prior 0 vs search), 1 = 309120 (prior 0.25 vs 1-sample search)
  src_line int32   [N]            line in the run's decisions.jsonl (oracle-labels key)
Side files: oracle.jsonl (row -> oracle engine state string + switch_ids + move_ids, or null), games.json.

    python scripts/wsp/prep_data.py OUT_DIR RUN_DIR [RUN_DIR ...]
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

from porygon.encoder import OBS_DIM


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cols = {k: [] for k in ("obs", "mask", "probs", "shares", "action", "value", "turn", "game", "won", "run",
                            "src_line")}
    games, gidx = [], {}
    n_skip = 0
    with open(out / "oracle.jsonl", "w") as orc:
        for r, run in enumerate(map(Path, sys.argv[2:])):
            winners = {}
            for line in open(run / "battles.jsonl"):
                b = json.loads(line)
                winners[b["roomid"]] = (b["winner"], b["p1"], b["p2"], b["pair"], b["game"])
            for i, line in enumerate(open(run / "decisions.jsonl")):
                rec = json.loads(line)
                if "obs" not in rec:
                    n_skip += 1
                    continue
                tag = rec["battle"]
                if tag not in gidx:
                    w = winners.get(tag)
                    gidx[tag] = len(games)
                    games.append({"tag": tag, "run": run.name, "player": rec["player"],
                                  "winner": w[0] if w else None, "pair": w[3] if w else None,
                                  "pair_game": w[4] if w else None})
                g = games[gidx[tag]]
                won = np.nan if g["winner"] is None and tag not in winners else (
                    0.5 if not g["winner"] else float(g["winner"] == rec["player"]))
                x = np.frombuffer(base64.b64decode(rec["obs"]), dtype=np.float16)
                assert x.size == OBS_DIM, (x.size, OBS_DIM)
                cols["obs"].append(x)
                cols["mask"].append(rec["mask"])
                cols["probs"].append(rec["probs"])
                cols["shares"].append(rec["shares"])
                cols["action"].append(rec["action"])
                cols["value"].append(np.nan if rec["value"] is None else rec["value"])
                cols["turn"].append(rec["turn"])
                cols["game"].append(gidx[tag])
                cols["won"].append(won)
                cols["run"].append(r)
                cols["src_line"].append(i)
                o = rec.get("oracle")
                orc.write(json.dumps(o if o and "error" not in o else None) + "\n")
    dt = {"obs": np.float16, "mask": np.int8, "probs": np.float32, "shares": np.float32, "action": np.int8,
          "value": np.float32, "turn": np.int16, "game": np.int32, "won": np.float32, "run": np.int8,
          "src_line": np.int32}
    np.savez(out / "decisions.npz", **{k: np.asarray(v, dtype=dt[k]) for k, v in cols.items()})
    (out / "games.json").write_text(json.dumps(games))
    n = len(cols["obs"])
    print(f"{n} decisions, {len(games)} games, {n_skip} skipped (no obs); "
          f"outcome known {np.isfinite(np.asarray(cols['won'], float)).mean():.1%}")


if __name__ == "__main__":
    main()
