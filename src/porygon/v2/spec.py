"""Observation spec v2 (owner: netv2; published in .orchestra/CONTRACTS.md). Constants shared by the reference
encoder (`porygon.v2.encoder`), the model and every other implementation (fastenv, data).

Versioning: SPEC_VERSION changes whenever a field, shape, dtype, ordering or semantics changes. Implementations
check it against the parity fixtures' version.
"""

import json
from pathlib import Path

import numpy as np

SPEC_VERSION = "2.0"

DATA = Path(__file__).resolve().parent.parent / "data"
VOCAB = json.loads((DATA / "vocab.json").read_text())  # same vocabulary as v1
PAD, UNK = 0, 1  # id 0 = padding / none, 1 = unknown; vocab[kind][k] has id k + 2
INDEX = {k: {e: i + 2 for i, e in enumerate(v)} for k, v in VOCAB.items()}
VOCAB_SIZES = {k: len(v) + 2 for k, v in VOCAB.items()}

# Types: id 0 = none, then 1..19 in this order (poke-env PokemonType names, lower case).
TYPES = ["bug", "dark", "dragon", "electric", "fairy", "fighting", "fire", "flying", "ghost", "grass", "ground",
         "ice", "normal", "poison", "psychic", "rock", "steel", "water", "stellar"]
TYPE_ID = {t: i + 1 for i, t in enumerate(TYPES)}
N_TYPES = len(TYPES) + 1  # incl. none

WEATHERS = ["sunnyday", "raindance", "sandstorm", "hail", "snow", "desolateland", "primordialsea", "deltastream"]
TERRAINS = ["electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]
STATUSES = ["brn", "frz", "par", "psn", "slp", "tox"]
STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

N_MONS = 12  # 0-5 ours (battle.team order = switch actions 0-5), 6-11 opponent (order of first appearance)
KM = 24  # move slots per mon: ours 0-3 = known moves (move actions 6-9 / tera 22-25 for the active), opponents:
#          revealed moves (order of revelation) then the posterior's other candidate moves, most likely first
KI = 8  # item candidates per mon
KA = 4  # ability candidates per mon
N_EFFECTS = 49  # porygon.encoder.EFFECTS (v1 list, same order)
SIDE_DIM = 2 * 7 + 4  # per side: 7 screens/conditions present + their elapsed turns, 4 hazards


def n_actions(gen: int) -> int:
    """poke-env's action space: 0-5 switch, 6-9 move, (gen9) 22-25 move + terastallize; 10-21 always masked."""
    return 26 if gen >= 9 else 10


TERA_OFFSET = 22

# mon_num layout (float32, per mon); names give the column order
MON_NUM = (
    ["present", "mine", "active", "fainted", "revealed", "level", "hp_frac"]
    + [f"status_{s}" for s in STATUSES]
    + ["status_counter", "terastallized", "first_turn", "protect_counter", "must_recharge", "preparing",
       "trapped", "item_known", "item_none", "ability_known", "n_moves_known", "log_weight"]
    + [f"effect_{i}" for i in range(N_EFFECTS)]
)
MON_NUM_DIM = len(MON_NUM)
MOVE_NUM = ["known", "pp_frac", "available"]
MOVE_NUM_DIM = len(MOVE_NUM)

FIELD_NUM = (
    [f"weather_{w}" for w in WEATHERS] + ["weather_elapsed"]
    + [f"terrain_{t}" for t in TERRAINS] + ["terrain_elapsed"]
    + ["trickroom", "trickroom_elapsed", "gravity", "turn", "opp_unseen", "own_alive", "opp_alive",
       "force_switch", "trapped", "maybe_trapped", "can_tera", "used_tera", "opp_used_tera", "wait", "no_active",
       "n_switches", "n_moves"]
)
FIELD_NUM_DIM = len(FIELD_NUM)

# Opponent-action classes in our frame (aux target `aux_opp_class`): 0-5 switch to opponent slot (mon 6 + k),
# 6 switch to a mon we have not seen, 7 + j use the move in the opponent active's move slot j, 7 + KM + j the same
# with terastallization.
N_OPP_CLASSES = 7 + 2 * KM


def fields(gen: int) -> dict[str, tuple[tuple[int, ...], type]]:
    """name -> (shape, numpy dtype) of every observation field."""
    return {
        "field_num": ((FIELD_NUM_DIM,), np.float32),
        "side_num": ((2, SIDE_DIM), np.float32),  # 0 = us, 1 = opponent
        "species": ((N_MONS,), np.int16),
        "types": ((N_MONS, 2), np.int16),  # current pre-tera types (type changes included), 0 = none
        "tera": ((N_MONS,), np.int16),  # known tera type (ours always, gen9; theirs once terastallized), 0 = unknown
        "tera_p": ((N_MONS, N_TYPES), np.float32),  # tera type distribution (col 0 unused); gen4: zeros
        "item": ((N_MONS, KI), np.int16),
        "item_p": ((N_MONS, KI), np.float32),
        "ability": ((N_MONS, KA), np.int16),
        "ability_p": ((N_MONS, KA), np.float32),
        "moves": ((N_MONS, KM), np.int16),
        "move_p": ((N_MONS, KM), np.float32),  # P(the mon has this move); known = 1
        "move_type": ((N_MONS, KM), np.int16),  # the move's type id as this mon uses it (hidden power etc.)
        "move_num": ((N_MONS, KM, MOVE_NUM_DIM), np.float32),
        "stats": ((N_MONS, 6), np.float32),  # raw stats hp atk def spa spd spe (ours exact, theirs expected)
        "hp": ((N_MONS, 2), np.float32),  # current HP, max HP in points (theirs: fraction x expected max)
        "boosts": ((N_MONS, 7), np.float32),  # raw stages
        "mon_num": ((N_MONS, MON_NUM_DIM), np.float32),
        "action_mask": ((n_actions(gen),), np.float32),
    }


def aux_fields(gen: int) -> dict[str, tuple[tuple[int, ...], type]]:
    """Training-only targets (`porygon.v2.encoder.aux_targets`); -1 = no target."""
    return {
        "aux_item": ((6,), np.int16),  # true item id of opponent slot k (mon 6 + k); PAD = no item
        "aux_ability": ((6,), np.int16),
        "aux_tera": ((6,), np.int16),  # true tera type id; -1 in gen4
        "aux_moves": ((6, 4), np.int16),  # true move ids
        "aux_move_in_set": ((6, KM), np.float32),  # candidate move slot j is in the true set (1/0), -1 = empty slot
        "aux_unseen": ((6,), np.int16),  # species ids of the opponent's mons not yet seen, -1 padded
        "aux_opp_class": ((n_actions(gen),), np.int16),  # opponent's action a (its frame) -> class in our frame
    }
