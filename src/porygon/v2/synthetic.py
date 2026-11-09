"""Synthetic v2 observations for tests and benchmarks (tests/test_v2_stacked_opp.py, scripts/bench_stacked_opp.py):
random but structurally valid (ids in range, probabilities that sum to 1, one active per side, consistent masks),
not realistic battles. Real ones: `porygon.v2.parity` fixtures or fastenv."""

import numpy as np

from porygon.v2.model import COL
from porygon.v2.spec import FIELD_NUM, N_TYPES, VOCAB_SIZES, fields


def synthetic_obs(gen: int, n: int, max_moves: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """n observations in the Packer dtypes plus drop_seed 0, as the collector passes them to a pool opponent: seen /
    unseen opponent mons, fainted mons, sometimes a fainted active (force switch: switches only), the last row of a
    batch of > 2 without a legal action; own mons 1-4 move slots, opponent mons 1..max_moves candidate slots (row 0's
    first opponent exactly max_moves, so the batch's move-slot trim is max(4, max_moves))."""
    ob = {k: np.zeros((n, *shape), dt) for k, (shape, dt) in fields(gen).items()}
    for b in range(n):
        n_opp = int(rng.integers(1, 7))
        forced = rng.random() < 0.1
        for i in range(12):
            own = i < 6
            if not own and i - 6 >= n_opp:
                continue  # unseen opponent mon
            mn = ob["mon_num"][b, i]
            mn[COL["present"]], mn[COL["mine"]], mn[COL["level"]] = 1, float(own), rng.uniform(0.7, 1.0)
            fainted = (forced and i == 0) or (rng.random() < 0.2 and i not in (0, 6))
            mn[COL["fainted"]] = float(fainted)
            mn[COL["hp_frac"]] = 0.0 if fainted else rng.uniform(0.05, 1.0)
            mn[COL["log_weight"]] = rng.uniform(0, 1)
            mn[COL["status_brn"]] = float(rng.random() < 0.1)
            ob["species"][b, i] = rng.integers(2, VOCAB_SIZES["species"])
            ob["types"][b, i] = rng.integers(1, N_TYPES, 2) * np.array([1, rng.integers(0, 2)])
            maxhp = rng.uniform(150, 400)
            ob["hp"][b, i] = [maxhp * mn[COL["hp_frac"]], maxhp]
            ob["stats"][b, i] = rng.uniform(50, 400, 6)
            ob["boosts"][b, i] = rng.integers(-2, 3, 7)
            nm = max_moves if (b == 0 and i == 6) else int(rng.integers(1, (4 if own else max_moves) + 1))
            ob["moves"][b, i, :nm] = rng.integers(2, VOCAB_SIZES["moves"], nm)
            ob["move_p"][b, i, :nm] = 1.0 if own else rng.uniform(0.05, 1.0, nm)
            ob["move_type"][b, i, :nm] = rng.integers(1, N_TYPES, nm)
            ob["move_num"][b, i, :nm] = rng.uniform(0, 1, (nm, ob["move_num"].shape[-1]))
            ni, na = int(rng.integers(1, 4)), int(rng.integers(1, 3))
            ob["item"][b, i, :ni] = rng.integers(2, VOCAB_SIZES["items"], ni)
            ob["item_p"][b, i, :ni] = rng.dirichlet(np.ones(ni))
            ob["ability"][b, i, :na] = rng.integers(2, VOCAB_SIZES["abilities"], na)
            ob["ability_p"][b, i, :na] = rng.dirichlet(np.ones(na))
            if gen >= 9:
                ob["tera"][b, i] = rng.integers(1, N_TYPES) if own else 0
                ob["tera_p"][b, i, 1:] = rng.dirichlet(np.ones(N_TYPES - 1))
        ob["mon_num"][b, 0, COL["active"]] = ob["mon_num"][b, 6, COL["active"]] = 1
        ob["field_num"][b] = rng.uniform(0, 1, len(FIELD_NUM)) * (rng.random(len(FIELD_NUM)) < 0.2)
        ob["side_num"][b] = rng.uniform(0, 1, ob["side_num"].shape[1:]) * (rng.random(ob["side_num"].shape[1:]) < 0.2)
        if b == n - 1 and n > 2:
            continue  # no legal action (the opponent then plays uniformly over all actions)
        am = ob["action_mask"][b]
        am[1:6] = ob["mon_num"][b, 1:6, COL["fainted"]] == 0
        if not forced:
            am[6:10] = ob["moves"][b, 0, :4] > 0
            if gen >= 9 and rng.random() < 0.5:
                am[22:26] = am[6:10]
    ob["drop_seed"] = np.zeros(n, np.int64)
    return ob
