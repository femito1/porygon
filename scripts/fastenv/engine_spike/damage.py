"""Damage fidelity: poke-engine's calculate_damage vs Showdown's 16 rolls (damage.js) for the same lead-vs-lead
(attacker, defender, move) samples, no crit, neutral field.

Compared quantities (samples where at least one simulator deals damage):
  immunity    one says 0 / immune, the other deals damage
  max roll    engine max roll vs Showdown's max roll (roll 0): share exact, share within 1 HP, mean |rel. error|
  applied     what the engine actually applies in a game for a non-crit, non-KO hit, floor(0.925 max) (its average
              roll; fixed-damage moves: the fixed value) vs the mean of Showdown's 16 rolls: mean |rel. error|
The engine never samples the 16 rolls: it branches only on "this roll range kills" vs not, and on crits.

    PORYGON_GEN=9 PYTHONPATH=eng9:src python damage.py --pairs pairs.jsonl --showdown sd_damage.jsonl
"""

import argparse
import json
import math
from collections import defaultdict

import poke_engine as pe

from porygon.selfplay import engine_move, initial_state


FIRST = {"suckerpunch", "thunderclap", "fakeout", "firstimpression", "upperhand"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--showdown", required=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--exclude", default="tripleaxel",
                    help="moves left out (Triple Axel: the engine models every hit as 40 BP, Showdown's first is 20)")
    a = ap.parse_args()
    pairs = [json.loads(l) for l in open(a.pairs) if l.strip()]
    sd = [json.loads(l) for l in open(a.showdown) if l.strip()]
    states: dict[int, object] = {}
    rows = []
    for r in sd:
        if r["rolls"] is None or r["move"] in a.exclude.split(","):
            continue
        p = r["pair"]
        if p not in states:
            states[p] = initial_state(*pairs[p])
        m = engine_move(r["move"])
        try:
            # the attacker moves second, like Showdown's getDamage with an empty queue (Payback, Analytic), except
            # for the moves that fail unless they go first
            first = r["move"] in FIRST
            if r["side"] == 1:
                rolls = pe.calculate_damage(states[p], m, "splash", first)[0]
            else:
                rolls = pe.calculate_damage(states[p], "splash", m, not first)[1]
        except BaseException as e:  # pyo3 PanicException
            if isinstance(e, KeyboardInterrupt):
                raise
            rolls = None
        sdr = r["rolls"]
        if -1 in sdr:
            continue
        s_max, s_mean = sdr[0], sum(sdr) / 16
        if rolls is None:
            rows.append((r["move"], s_max, s_mean, None, None))
            continue
        e_max = rolls[0]
        e_applied = rolls[0] if len(rolls) == 1 else math.floor(rolls[0] * 0.925)
        rows.append((r["move"], s_max, s_mean, e_max, e_applied))
    dmg = [x for x in rows if x[1] > 0 or (x[3] or 0) > 0]
    panics = sum(1 for x in dmg if x[3] is None)
    imm = [x for x in dmg if x[3] is not None and ((x[1] == 0) != (x[3] == 0))]
    both = [x for x in dmg if x[3] is not None and x[1] > 0 and x[3] > 0]
    exact = sum(1 for x in both if x[3] == x[1])
    within1 = sum(1 for x in both if abs(x[3] - x[1]) <= 1)
    rel_max = [abs(x[3] - x[1]) / x[1] for x in both]
    rel_app = [abs(x[4] - x[2]) / x[2] for x in both]
    big = sum(1 for e in rel_max if e > 0.05)
    out = {"samples": len(dmg), "engine_panics": panics, "immunity_mismatch": len(imm) / len(dmg),
           "max_exact": exact / len(both), "max_within_1hp": within1 / len(both),
           "max_mean_abs_rel_err": sum(rel_max) / len(both), "max_share_err_over_5pct": big / len(both),
           "applied_vs_sd_mean_mean_abs_rel_err": sum(rel_app) / len(both),
           "applied_vs_sd_mean_signed": sum((x[4] - x[2]) / x[2] for x in both) / len(both)}
    print(json.dumps(out))
    per = defaultdict(lambda: [0, 0, 0.0])
    for x in dmg:
        k = per[x[0]]
        k[0] += 1
        if x[3] is None or (x[1] == 0) != (x[3] == 0):
            k[1] += 1
        elif x[1] > 0:
            k[2] += abs(x[3] - x[1]) / x[1]
    worst = sorted(per.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))[:a.top]
    print("worst moves (n, immunity/panic mismatches, summed |rel err| of max roll):")
    for mv, (n, mis, err) in worst:
        print(f"  {mv:<20} n={n:<4} mismatch={mis:<4} sum_rel_err={err:.2f}")


if __name__ == "__main__":
    main()
