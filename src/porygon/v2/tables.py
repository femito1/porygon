"""Static data tables for model v2 (owner: netv2): per-vocabulary-id feature rows for species, moves, items and
abilities, built from the generation's Showdown data (`poke_env.data.GenData`) plus hand-made item / ability
classes (poke-env has no item / ability data). Rows are indexed by vocab id (spec.INDEX: id = position + 2);
rows 0 (pad / none) and 1 (unknown) are zero, as is any id the generation's data does not know.

    feats = static_features(9)          # {"species", "moves", "items", "abilities"} -> float32 [vocab size, F]
    FEATURE_NAMES["moves"]              # column names, same order
    raw = move_raw(9)                   # un-normalised per-move numbers the damage layer uses (porygon.v2.damage)

Values are O(1): base stats / 255, base power / 150, priority / 5, PP / 40, boosts summed / 6, multiplier
classes as (multiplier - 1) or 0/1 flags. Generation differences that change mechanics (immunities that only
exist from gen 5, gen-4 hidden power 70, ...) are applied here, so the gen-4 and gen-9 tables can differ for the
same id.
"""

from functools import cache

import numpy as np

from porygon.v2.spec import INDEX, TYPE_ID, TYPES, VOCAB, VOCAB_SIZES

NT = len(TYPES)  # 19 type columns (spec.TYPES order; type id t -> column t - 1)
TYPE_COLS = [f"type_{t}" for t in TYPES]


@cache
def _gendata(gen: int):
    from poke_env.data import GenData
    return GenData.from_gen(gen)


def _tid(name: str | None) -> int:
    return TYPE_ID.get(str(name).lower(), 0) if name else 0


# ------------------------------------------------------------------------------------------------ species

SPECIES_FEATURES = ["hp", "atk", "def", "spa", "spd", "spe"] + TYPE_COLS + ["log_weight", "nfe", "bst"]


def species_table(gen: int) -> np.ndarray:
    dex = _gendata(gen).pokedex
    out = np.zeros((VOCAB_SIZES["species"], len(SPECIES_FEATURES)), np.float32)
    for name, i in INDEX["species"].items():
        e = dex.get(name)
        if e is None:
            continue
        b = e["baseStats"]
        out[i, :6] = [b[s] / 255 for s in ("hp", "atk", "def", "spa", "spd", "spe")]
        for t in e.get("types", []):
            if _tid(t):
                out[i, 6 + _tid(t) - 1] = 1.0
        out[i, 6 + NT] = np.log10(max(float(e.get("weightkg", 50.0)), 0.1)) / 3
        out[i, 7 + NT] = float(bool(e.get("evos")))
        out[i, 8 + NT] = sum(b.values()) / 720
    return out


@cache
def species_raw(gen: int) -> dict[str, np.ndarray]:
    """Per species id: weight (kg), NFE flag, and the ids the damage layer special-cases."""
    dex = _gendata(gen).pokedex
    n = VOCAB_SIZES["species"]
    weight = np.full(n, 50.0, np.float32)
    nfe = np.zeros(n, np.float32)
    for name, i in INDEX["species"].items():
        e = dex.get(name)
        if e is not None:
            weight[i] = float(e.get("weightkg", 50.0))
            nfe[i] = float(bool(e.get("evos")))
    # intact Disguise / Ice Face forms (the busted / no-ice formes have other species ids)
    disguise = np.zeros(n, np.float32)
    for s in ("mimikyu", "mimikyutotem"):
        if s in INDEX["species"]:
            disguise[INDEX["species"][s]] = 1.0
    iceface = np.zeros(n, np.float32)
    if "eiscue" in INDEX["species"]:
        iceface[INDEX["species"]["eiscue"]] = 1.0
    return {"weight": weight, "nfe": nfe, "disguise": disguise, "iceface": iceface}


# ------------------------------------------------------------------------------------------------ moves

MOVE_FLAGS = ["contact", "sound", "punch", "bite", "slicing", "pulse", "bullet", "wind", "powder"]
STATUS_KINDS = ["brn", "frz", "par", "psn", "slp", "tox"]

# fixed-damage kinds (move_raw "fixed"): 1 = attacker's level, 2 = half the target's current HP, 3 = a constant
# ("fixed_value"), 4 = endeavor (target HP - user HP), 5 = final gambit (user HP), 6 = psywave (level on average),
# 7 = OHKO (the target's current HP)
FIXED = {"seismictoss": 1, "nightshade": 1, "superfang": 2, "ruination": 2, "naturesmadness": 2,
         "sonicboom": 3, "dragonrage": 3, "endeavor": 4, "finalgambit": 5, "psywave": 6}

# variable-power / type / category kinds (move_raw "special"); porygon.v2.damage evaluates most of them (see its
# docstring for the ones it gives a fixed power)
SPECIAL = {
    "eruption": 1, "waterspout": 1, "dragonenergy": 1,
    "flail": 2, "reversal": 2,
    "gyroball": 3, "electroball": 4,
    "lowkick": 5, "grassknot": 5,
    "heavyslam": 6, "heatcrash": 6,
    "storedpower": 7, "powertrip": 7,
    "punishment": 8,
    "weatherball": 9,
    "terablast": 10,
    "acrobatics": 11,
    "facade": 12,
    "hex": 13, "infernalparade": 13, "bittermalice": 13,
    "venoshock": 14, "barbbarrage": 14,
    "brine": 15,
    "knockoff": 16,
    "crushgrip": 17, "wringout": 17, "hardpress": 17,
    "lastrespects": 18,
    "wakeupslap": 19, "smellingsalts": 20,
    "judgment": 21, "revelationdance": 21, "multiattack": 21,
    "ivycudgel": 22, "ragingbull": 22,
    "photongeyser": 23, "shellsidearm": 23,
    "collisioncourse": 24, "electrodrift": 24,
    "psyblade": 25,
    "solarbeam": 26, "solarblade": 26,
    "poltergeist": 27,
}
# base-power overrides (None in data / callbacks with a fixed random-battle value)
BP_OVERRIDE = {"return": 102, "frustration": 102, "ficklebeam": 104, "magnitude": 71, "present": 52,
               "naturalgift": 80, "spitup": 100, "beatup": 50, "trumpcard": 60, "ragefist": 50,
               "lastrespects": 50, "punishment": 60, "flail": 20, "reversal": 20, "lowkick": 20,
               "grassknot": 20, "heavyslam": 40, "heatcrash": 40, "gyroball": 0, "electroball": 0,
               "crushgrip": 120, "wringout": 120, "hardpress": 100, "storedpower": 20, "powertrip": 20}
# multi-hit moves whose later hits are stronger / checked separately: (per-hit base power, expected hits given
# the first hit lands, hits with Skill Link)
MULTI_SPECIAL = {"tripleaxel": (34.87, 3, 3), "triplekick": (17.43, 3, 3),
                 "populationbomb": (20.0, 6.51, 10)}
ACC_KINDS = {"thunder": 1, "hurricane": 1, "bleakwindstorm": 1, "wildboltstorm": 1, "sandsearstorm": 1,
             "blizzard": 2}  # 1: always hits in rain, 50% in sun; 2: always hits in hail / snow

MOVE_FEATURES = (
    ["bp", "variable_power", "physical", "special", "status"] + TYPE_COLS
    + ["accuracy", "always_hits", "priority", "pp", "crit_stage", "will_crit", "multihit"]
    + MOVE_FLAGS
    + ["drain", "recoil", "heal", "self_switch", "force_switch"]
    + [f"status_{s}" for s in STATUS_KINDS]
    + ["volatile_status", "confusion", "flinch_chance", "side_condition", "hazard", "screen", "weather",
       "terrain", "target_boosts", "self_boosts", "boost_abs", "protect", "breaks_protect", "selfdestruct",
       "secondary_chance", "fixed_level", "fixed_half_hp", "fixed_other", "ohko", "special_case", "two_turn",
       "recharge", "sets_trickroom", "phys_vs_def", "uses_def", "uses_target_atk", "multiaccuracy"]
)
HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
SCREENS = {"reflect", "lightscreen", "auroraveil"}


def _num(x, default=0.0) -> float:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else default


def _expected_hits(mh, gen: int) -> tuple[float, float]:
    if mh is None:
        return 1.0, 1.0
    if isinstance(mh, list):
        lo, hi = mh
        if (lo, hi) == (2, 5):
            return (3.0 if gen <= 4 else 3.1), 5.0
        return (lo + hi) / 2, float(hi)
    return float(mh), float(mh)


@cache
def move_raw(gen: int) -> dict[str, np.ndarray]:
    """Un-normalised per-move numbers (vocab id -> value) for porygon.v2.damage."""
    moves = _gendata(gen).moves
    n = VOCAB_SIZES["moves"]
    z = lambda dt=np.float32: np.zeros(n, dt)  # noqa: E731
    out = {k: z() for k in ("bp", "accuracy", "priority", "hits", "max_hits", "fixed_value", "recoil",
                            "has_secondary", "will_crit", "multiaccuracy", "selfdestruct", "uses_def",
                            "uses_target_atk", "phys_vs_def", "variable_hits", "known")}
    out["accuracy"][:] = 1.0
    out["category"] = z(np.int64)  # 0 = unknown / none, 1 physical, 2 special, 3 status
    out["type"] = z(np.int64)
    out["fixed"] = z(np.int64)
    out["special"] = z(np.int64)
    out["acc_kind"] = z(np.int64)
    for f in MOVE_FLAGS:
        out[f] = z()
    for name, i in INDEX["moves"].items():
        e = moves.get(name)
        if e is None:
            continue
        out["known"][i] = 1.0
        cat = {"Physical": 1, "Special": 2, "Status": 3}.get(e.get("category"), 0)
        out["category"][i] = cat
        out["type"][i] = _tid(e.get("type"))
        bp = _num(e.get("basePower"))
        if name == "hiddenpower":
            bp = 70.0 if gen <= 5 else 60.0
        if name in BP_OVERRIDE:
            bp = float(BP_OVERRIDE[name])
        acc = e.get("accuracy", True)
        out["accuracy"][i] = 1.0 if acc is True else _num(acc, 100.0) / 100
        out["priority"][i] = _num(e.get("priority"))
        hits, mx = _expected_hits(e.get("multihit"), gen)
        out["variable_hits"][i] = float(isinstance(e.get("multihit"), list))
        if name in MULTI_SPECIAL:
            bp, hits, mx = MULTI_SPECIAL[name]
        out["bp"][i] = bp
        out["hits"][i], out["max_hits"][i] = hits, mx
        out["multiaccuracy"][i] = float(bool(e.get("multiaccuracy")))
        dmg = e.get("damage")
        if name in FIXED:
            out["fixed"][i] = FIXED[name]
        elif dmg == "level":
            out["fixed"][i] = 1
        elif isinstance(dmg, (int, float)) and not isinstance(dmg, bool):
            out["fixed"][i] = 3
        if e.get("ohko"):
            out["fixed"][i] = 7
        if out["fixed"][i] == 3:
            out["fixed_value"][i] = _num(dmg, {"sonicboom": 20.0, "dragonrage": 40.0}.get(name, 0.0))
        out["special"][i] = SPECIAL.get(name, 0)
        out["acc_kind"][i] = ACC_KINDS.get(name, 0)
        rec = e.get("recoil")
        out["recoil"][i] = rec[0] / rec[1] if rec else (0.5 if e.get("hasCrashDamage") else 0.0)
        out["has_secondary"][i] = float(bool(e.get("secondary") or e.get("secondaries")
                                             or e.get("hasSheerForceBoost")))
        out["will_crit"][i] = float(bool(e.get("willCrit")))
        out["selfdestruct"][i] = float(bool(e.get("selfdestruct")) and cat in (1, 2))
        out["uses_def"][i] = float(e.get("overrideOffensiveStat") == "def")
        out["uses_target_atk"][i] = float(e.get("overrideOffensivePokemon") == "target")
        out["phys_vs_def"][i] = float(e.get("overrideDefensiveStat") == "def")
        flags = e.get("flags", {})
        for f in MOVE_FLAGS:
            out[f][i] = float(bool(flags.get(f)))
    return out


def _secondaries(e: dict) -> list[dict]:
    s = e.get("secondaries") or ([e["secondary"]] if e.get("secondary") else [])
    return [x for x in s if isinstance(x, dict)]


def move_table(gen: int) -> np.ndarray:
    moves = _gendata(gen).moves
    raw = move_raw(gen)
    col = {f: k for k, f in enumerate(MOVE_FEATURES)}
    out = np.zeros((VOCAB_SIZES["moves"], len(MOVE_FEATURES)), np.float32)
    for name, i in INDEX["moves"].items():
        e = moves.get(name)
        if e is None:
            continue
        r = out[i]
        r[col["bp"]] = raw["bp"][i] / 150
        r[col["variable_power"]] = float(bool(e.get("basePowerCallback") or e.get("onBasePower")
                                              or name in SPECIAL or name in BP_OVERRIDE))
        cat = raw["category"][i]
        if cat:
            r[col[["physical", "special", "status"][cat - 1]]] = 1.0
        if raw["type"][i]:
            r[col[TYPE_COLS[raw["type"][i] - 1]]] = 1.0
        r[col["accuracy"]] = raw["accuracy"][i]
        r[col["always_hits"]] = float(e.get("accuracy", True) is True)
        r[col["priority"]] = raw["priority"][i] / 5
        r[col["pp"]] = _num(e.get("pp")) / 40
        r[col["crit_stage"]] = (_num(e.get("critRatio"), 1.0) - 1) / 2
        r[col["will_crit"]] = raw["will_crit"][i]
        r[col["multihit"]] = raw["hits"][i] / 5 if raw["max_hits"][i] > 1 else 0.0
        for f in MOVE_FLAGS:
            r[col[f]] = raw[f][i]
        d = e.get("drain")
        r[col["drain"]] = d[0] / d[1] if d else 0.0
        r[col["recoil"]] = raw["recoil"][i]
        h = e.get("heal")
        r[col["heal"]] = h[0] / h[1] if h else 0.0
        r[col["self_switch"]] = float(bool(e.get("selfSwitch")))
        r[col["force_switch"]] = float(bool(e.get("forceSwitch")))
        secs = _secondaries(e)
        st = e.get("status")
        if st in STATUS_KINDS:
            r[col["status_" + st]] = 1.0
        for s in secs:
            if s.get("status") in STATUS_KINDS:
                r[col["status_" + s["status"]]] = max(r[col["status_" + s["status"]]], _num(s.get("chance"), 100) / 100)
        vs = e.get("volatileStatus")
        r[col["volatile_status"]] = float(bool(vs))
        r[col["confusion"]] = float(vs == "confusion") + sum(
            _num(s.get("chance"), 100) / 100 for s in secs if s.get("volatileStatus") == "confusion")
        r[col["flinch_chance"]] = sum(_num(s.get("chance"), 100) / 100 for s in secs
                                      if s.get("volatileStatus") == "flinch")
        sc = e.get("sideCondition")
        r[col["side_condition"]] = float(bool(sc))
        r[col["hazard"]] = float(sc in HAZARDS)
        r[col["screen"]] = float(sc in SCREENS)
        r[col["weather"]] = float(bool(e.get("weather")))
        r[col["terrain"]] = float(bool(e.get("terrain")))
        target = e.get("target")
        boosts = e.get("boosts") or {}
        tb = sum(boosts.values()) if target != "self" else 0
        sb = sum(boosts.values()) if target in ("self", "adjacentAllyOrSelf", "allies") else 0
        for s in secs:
            ch = _num(s.get("chance"), 100) / 100
            tb += ch * sum((s.get("boosts") or {}).values())
            sb += ch * sum(((s.get("self") or {}).get("boosts") or {}).values())
        sb += sum(((e.get("self") or {}).get("boosts") or {}).values())
        sb += sum(((e.get("selfBoost") or {}).get("boosts") or {}).values())
        r[col["target_boosts"]] = tb / 6
        r[col["self_boosts"]] = sb / 6
        r[col["boost_abs"]] = sum(abs(v) for v in boosts.values()) / 6
        r[col["protect"]] = float(bool(e.get("stallingMove")))
        r[col["breaks_protect"]] = float(bool(e.get("breaksProtect")))
        r[col["selfdestruct"]] = float(bool(e.get("selfdestruct")))
        r[col["secondary_chance"]] = max([_num(s.get("chance"), 100) / 100 for s in secs], default=0.0)
        fx = raw["fixed"][i]
        r[col["fixed_level"]] = float(fx in (1, 6))
        r[col["fixed_half_hp"]] = float(fx == 2)
        r[col["fixed_other"]] = float(fx in (3, 4, 5))
        r[col["ohko"]] = float(fx == 7)
        r[col["special_case"]] = float(raw["special"][i] > 0)
        flags = e.get("flags", {})
        r[col["two_turn"]] = float(bool(flags.get("charge")))
        r[col["recharge"]] = float(bool(flags.get("recharge")))
        r[col["sets_trickroom"]] = float(e.get("pseudoWeather") == "trickroom")
        r[col["phys_vs_def"]] = raw["phys_vs_def"][i]
        r[col["uses_def"]] = raw["uses_def"][i]
        r[col["uses_target_atk"]] = raw["uses_target_atk"][i]
        r[col["multiaccuracy"]] = raw["multiaccuracy"][i]
    return out


# ------------------------------------------------------------------------------------------------ items

TYPE_BOOST_ITEMS = {
    "silverpowder": "bug", "blackglasses": "dark", "dragonfang": "dragon", "magnet": "electric",
    "fairyfeather": "fairy", "blackbelt": "fighting", "charcoal": "fire", "sharpbeak": "flying",
    "spelltag": "ghost", "miracleseed": "grass", "softsand": "ground", "nevermeltice": "ice",
    "silkscarf": "normal", "poisonbarb": "poison", "twistedspoon": "psychic", "hardstone": "rock",
    "metalcoat": "steel", "mysticwater": "water",
    "insectplate": "bug", "dreadplate": "dark", "dracoplate": "dragon", "zapplate": "electric",
    "pixieplate": "fairy", "fistplate": "fighting", "flameplate": "fire", "skyplate": "flying",
    "spookyplate": "ghost", "meadowplate": "grass", "earthplate": "ground", "icicleplate": "ice",
    "toxicplate": "poison", "mindplate": "psychic", "stoneplate": "rock", "ironplate": "steel",
    "splashplate": "water",
    "oddincense": "psychic", "rockincense": "rock", "roseincense": "grass", "seaincense": "water",
    "waveincense": "water",
}
GEMS = {f"{t}gem": t for t in TYPES if t != "stellar"}
RESIST_BERRIES = {
    "tangaberry": "bug", "colburberry": "dark", "habanberry": "dragon", "wacanberry": "electric",
    "roseliberry": "fairy", "chopleberry": "fighting", "occaberry": "fire", "cobaberry": "flying",
    "kasibberry": "ghost", "rindoberry": "grass", "shucaberry": "ground", "yacheberry": "ice",
    "chilanberry": "normal", "kebiaberry": "poison", "payapaberry": "psychic", "chartiberry": "rock",
    "babiriberry": "steel", "passhoberry": "water",
}
# items that boost the holder's own STAB types x1.2 (legendary orbs / cores / gen-7+ Soul Dew) or all its moves
STAB_ITEMS = {"adamantorb", "lustrousorb", "griseousorb", "adamantcrystal", "lustrousglobe", "griseouscore"}
MASKS = {"wellspringmask", "hearthflamemask", "cornerstonemask"}
_ITEM_GROUPS = {
    "choiceband": {"choiceband"}, "choicespecs": {"choicespecs"}, "choicescarf": {"choicescarf"},
    "lifeorb": {"lifeorb"}, "expertbelt": {"expertbelt"}, "muscleband": {"muscleband"},
    "wiseglasses": {"wiseglasses"}, "punchingglove": {"punchingglove"},
    "stab_boost": STAB_ITEMS | MASKS, "souldew": {"souldew"}, "lightball": {"lightball"},
    "thickclub": {"thickclub"}, "metronome": {"metronome"},
    "assaultvest": {"assaultvest"}, "eviolite": {"eviolite"}, "heavydutyboots": {"heavydutyboots"},
    "focussash": {"focussash", "focusband"}, "airballoon": {"airballoon"},
    "rockyhelmet": {"rockyhelmet"}, "leftovers": {"leftovers"}, "blacksludge": {"blacksludge"},
    "shellbell": {"shellbell"}, "bigroot": {"bigroot"},
    "heal_berry": {"sitrusberry", "oranberry", "figyberry", "wikiberry", "magoberry", "aguavberry",
                   "iapapaberry", "berryjuice", "enigmaberry", "goldberry", "berry"},
    "status_berry": {"lumberry", "chestoberry", "cheriberry", "pechaberry", "rawstberry", "aspearberry",
                     "persimberry", "mintberry", "przcureberry", "psncureberry", "iceberry", "burntberry",
                     "bitterberry", "miracleberry"},
    "pinch_berry": {"liechiberry", "petayaberry", "salacberry", "ganlonberry", "apicotberry", "starfberry",
                    "lansatberry", "micleberry", "custapberry", "keeberry", "marangaberry", "jabocaberry",
                    "rowapberry"},
    "leppaberry": {"leppaberry"},
    "weaknesspolicy": {"weaknesspolicy"}, "boosterenergy": {"boosterenergy"}, "whiteherb": {"whiteherb"},
    "powerherb": {"powerherb"}, "mentalherb": {"mentalherb"}, "mirrorherb": {"mirrorherb"},
    "toxicorb": {"toxicorb"}, "flameorb": {"flameorb"}, "lightclay": {"lightclay"},
    "ironball": {"ironball"}, "laggingtail": {"laggingtail", "fullincense"}, "stickybarb": {"stickybarb"},
    "redcard": {"redcard"}, "ejectbutton": {"ejectbutton"}, "ejectpack": {"ejectpack"},
    "throatspray": {"throatspray"}, "loadeddice": {"loadeddice"}, "covertcloak": {"covertcloak"},
    "clearamulet": {"clearamulet"}, "protectivepads": {"protectivepads"},
    "safetygoggles": {"safetygoggles"}, "utilityumbrella": {"utilityumbrella"},
    "weather_rock": {"damprock", "heatrock", "smoothrock", "icyrock"}, "terrainextender": {"terrainextender"},
    "roomservice": {"roomservice"}, "blunderpolicy": {"blunderpolicy"},
    "accuracy_item": {"widelens", "zoomlens"},
    "crit_item": {"scopelens", "razorclaw", "stick", "leek", "luckypunch"},
    "evasion_item": {"brightpowder", "laxincense"},
    "flinch_item": {"kingsrock", "razorfang"}, "quickclaw": {"quickclaw"},
    "binding_item": {"bindingband", "gripclaw"}, "shedshell": {"shedshell"},
    "abilityshield": {"abilityshield"}, "mega_or_z": set(),
}
ITEM_FEATURES = (["type_boost", "gem", "resist_berry"] + TYPE_COLS + list(_ITEM_GROUPS)
                 + ["berry", "known"])


def item_table(gen: int) -> np.ndarray:
    col = {f: k for k, f in enumerate(ITEM_FEATURES)}
    out = np.zeros((VOCAB_SIZES["items"], len(ITEM_FEATURES)), np.float32)
    type_boost = 0.2 if gen >= 4 else 0.1
    gem = 0.5 if gen == 5 else 0.3
    for name, i in INDEX["items"].items():
        r = out[i]
        r[col["known"]] = 1.0
        if name in TYPE_BOOST_ITEMS:
            r[col["type_boost"]] = type_boost
            r[col[f"type_{TYPE_BOOST_ITEMS[name]}"]] = 1.0
        elif name in GEMS:
            r[col["gem"]] = gem
            r[col[f"type_{GEMS[name]}"]] = 1.0
        elif name in RESIST_BERRIES:
            r[col["resist_berry"]] = 1.0
            r[col[f"type_{RESIST_BERRIES[name]}"]] = 1.0
        for g, members in _ITEM_GROUPS.items():
            if name in members:
                r[col[g]] = 1.0
        if name.endswith("ite") and name not in ("eviolite",) or name.endswith("iumz") or name.endswith("itex") \
                or name.endswith("itey"):
            r[col["mega_or_z"]] = 1.0
        r[col["berry"]] = float(name.endswith("berry"))
    return out


# ------------------------------------------------------------------------------------------------ abilities

ABILITY_TYPE_ROLE = {  # ability -> (role, type): one type column per ability, the role says what it does
    "aerilate": ("ate", "flying"), "pixilate": ("ate", "fairy"), "refrigerate": ("ate", "ice"),
    "galvanize": ("ate", "electric"), "normalize": ("ate", "normal"),
    "blaze": ("pinch_boost", "fire"), "torrent": ("pinch_boost", "water"), "overgrow": ("pinch_boost", "grass"),
    "swarm": ("pinch_boost", "bug"),
    "steelworker": ("type_boost", "steel"), "steelyspirit": ("type_boost", "steel"),
    "transistor": ("type_boost", "electric"), "dragonsmaw": ("type_boost", "dragon"),
    "rockypayload": ("type_boost", "rock"), "waterbubble": ("type_boost", "water"),
    "levitate": ("immune", "ground"), "eartheater": ("immune", "ground"), "flashfire": ("immune", "fire"),
    "wellbakedbody": ("immune", "fire"), "waterabsorb": ("immune", "water"), "dryskin": ("immune", "water"),
    "stormdrain": ("immune", "water"), "voltabsorb": ("immune", "electric"),
    "lightningrod": ("immune", "electric"), "motordrive": ("immune", "electric"),
    "sapsipper": ("immune", "grass"),
}
GEN5_IMMUNITIES = {"stormdrain", "lightningrod", "sapsipper"}  # redirect only before gen 5
_ABILITY_GROUPS = {
    # attacking
    "atk_double": {"hugepower", "purepower"}, "hustle": {"hustle"}, "gorillatactics": {"gorillatactics"},
    "guts": {"guts"}, "technician": {"technician"}, "adaptability": {"adaptability"},
    "sheerforce": {"sheerforce"}, "ironfist": {"ironfist"}, "toughclaws": {"toughclaws"},
    "strongjaw": {"strongjaw"}, "sharpness": {"sharpness"}, "megalauncher": {"megalauncher"},
    "reckless": {"reckless"}, "punkrock": {"punkrock"}, "tintedlens": {"tintedlens"},
    "skilllink": {"skilllink"}, "sniper": {"sniper"}, "superluck": {"superluck"},
    "scrappy": {"scrappy", "mindseye"}, "moldbreaker": {"moldbreaker", "teravolt", "turboblaze", "myceliummight"},
    "infiltrator": {"infiltrator"}, "unaware": {"unaware"}, "noguard": {"noguard"},
    "compoundeyes": {"compoundeyes"}, "serenegrace": {"serenegrace"}, "solarpower": {"solarpower"},
    "defeatist": {"defeatist"}, "sandforce": {"sandforce"}, "analytic": {"analytic"},
    "toxicboost": {"toxicboost"}, "flareboost": {"flareboost"}, "stakeout": {"stakeout"},
    "supremeoverlord": {"supremeoverlord"}, "parentalbond": {"parentalbond"},
    "protean": {"protean", "libero"}, "contrary": {"contrary"}, "simple": {"simple"},
    "boost_on_ko": {"moxie", "beastboost", "chillingneigh", "grimneigh", "asoneglastrier", "asonespectrier",
                    "soulheart"},
    "orichalcum_hadron": {"orichalcumpulse", "hadronengine"},
    "ruin": {"swordofruin", "beadsofruin", "tabletsofruin", "vesselofruin"},
    "protosynthesis": {"protosynthesis", "quarkdrive"},
    "intrepid": {"intrepidsword", "dauntlessshield"}, "zerotohero": {"zerotohero"},
    "download": {"download"}, "poison_on_attack": {"poisontouch", "toxicchain", "poisonpuppeteer"},
    # defending
    "immune_absorb": {"waterabsorb", "voltabsorb", "dryskin", "eartheater", "sapsipper", "motordrive",
                      "stormdrain", "lightningrod", "flashfire", "wellbakedbody"},
    "wonderguard": {"wonderguard"}, "bulletproof": {"bulletproof"}, "soundproof": {"soundproof"},
    "windrider": {"windrider", "windpower"}, "overcoat": {"overcoat"}, "goodasgold": {"goodasgold"},
    "magicbounce": {"magicbounce"}, "thickfat": {"thickfat"}, "heatproof": {"heatproof"},
    "filter": {"filter", "solidrock", "prismarmor"}, "multiscale": {"multiscale", "shadowshield"},
    "fluffy": {"fluffy"}, "furcoat": {"furcoat"}, "icescales": {"icescales"},
    "purifyingsalt": {"purifyingsalt"}, "marvelscale": {"marvelscale"}, "sturdy": {"sturdy"},
    "disguise": {"disguise", "iceface"}, "terashell": {"terashell"}, "priority_block":
        {"queenlymajesty", "dazzling", "armortail"},
    "contact_status": {"static", "flamebody", "poisonpoint", "effectspore", "cutecharm"},
    "contact_damage": {"roughskin", "ironbarbs"}, "cursedbody": {"cursedbody"},
    "stat_on_hit": {"justified", "defiant", "competitive", "stamina", "weakarmor", "berserk", "angershell",
                    "rattled", "steamengine", "watercompaction", "thermalexchange", "electromorphosis",
                    "angerpoint", "seedsower", "sandspit", "cottondown", "mirrorarmor"},
    "status_immune": {"limber", "insomnia", "vitalspirit", "immunity", "waterveil", "magmaarmor",
                      "owntempo", "oblivious", "innerfocus", "leafguard", "comatose", "purifyingsalt",
                      "pastelveil", "sweetveil", "flowerveil", "aromaveil", "thermalexchange", "waterbubble"},
    "stat_drop_immune": {"clearbody", "whitesmoke", "fullmetalbody", "hypercutter", "keeneye", "bigpecks",
                         "mirrorarmor", "guarddog"},
    "unnerve": {"unnerve", "asoneglastrier", "asonespectrier"},
    # speed / priority
    "swiftswim": {"swiftswim"}, "chlorophyll": {"chlorophyll"}, "sandrush": {"sandrush"},
    "slushrush": {"slushrush"}, "surgesurfer": {"surgesurfer"}, "unburden": {"unburden"},
    "quickfeet": {"quickfeet"}, "speedboost": {"speedboost"}, "slowstart": {"slowstart"},
    "truant": {"truant"}, "prankster": {"prankster"}, "galewings": {"galewings"}, "triage": {"triage"},
    "quickdraw": {"quickdraw"},
    # field
    "sets_rain": {"drizzle", "primordialsea"}, "sets_sun": {"drought", "desolateland", "orichalcumpulse"},
    "sets_sand": {"sandstream", "sandspit"}, "sets_snow": {"snowwarning"}, "sets_wind": {"deltastream"},
    "sets_electric": {"electricsurge", "hadronengine"}, "sets_grassy": {"grassysurge", "seedsower"},
    "sets_misty": {"mistysurge"}, "sets_psychic": {"psychicsurge"}, "cloudnine": {"cloudnine", "airlock"},
    # misc
    "intimidate": {"intimidate"}, "regenerator": {"regenerator"}, "magicguard": {"magicguard"},
    "naturalcure": {"naturalcure"}, "poisonheal": {"poisonheal"}, "shedskin": {"shedskin", "hydration"},
    "trace": {"trace"}, "pressure": {"pressure"}, "trapping": {"shadowtag", "arenatrap", "magnetpull"},
    "illusion": {"illusion"}, "imposter": {"imposter"}, "harvest": {"harvest", "cheekpouch", "cudchew"},
    "rockhead": {"rockhead"}, "liquidooze": {"liquidooze"}, "frisk": {"frisk"}, "drought_like": set(),
}
_ABILITY_GROUPS.pop("drought_like")
ABILITY_FEATURES = (["ate", "pinch_boost", "type_boost", "immune"] + TYPE_COLS + list(_ABILITY_GROUPS)
                    + ["known"])


def ability_table(gen: int) -> np.ndarray:
    col = {f: k for k, f in enumerate(ABILITY_FEATURES)}
    out = np.zeros((VOCAB_SIZES["abilities"], len(ABILITY_FEATURES)), np.float32)
    for name, i in INDEX["abilities"].items():
        r = out[i]
        r[col["known"]] = 1.0
        if name in ABILITY_TYPE_ROLE and not (gen <= 4 and name in GEN5_IMMUNITIES):
            role, t = ABILITY_TYPE_ROLE[name]
            r[col[role]] = 1.0
            r[col[f"type_{t}"]] = 1.0
        for g, members in _ABILITY_GROUPS.items():
            if name in members:
                if g == "immune_absorb" and gen <= 4 and name in GEN5_IMMUNITIES:
                    continue
                r[col[g]] = 1.0
    return out


# ------------------------------------------------------------------------------------------------ API

FEATURE_NAMES = {"species": SPECIES_FEATURES, "moves": MOVE_FEATURES, "items": ITEM_FEATURES,
                 "abilities": ABILITY_FEATURES}


@cache
def _static(gen: int) -> dict[str, np.ndarray]:
    out = {"species": species_table(gen), "moves": move_table(gen), "items": item_table(gen),
           "abilities": ability_table(gen)}
    for v in out.values():
        v[:2] = 0.0
        v.setflags(write=False)
    return out


def static_features(gen: int) -> dict[str, np.ndarray]:
    """kind -> float32 [VOCAB_SIZES[kind], len(FEATURE_NAMES[kind])] (copies; rows 0 and 1 are zero)."""
    return {k: v.copy() for k, v in _static(gen).items()}


def column(kind: str, name: str) -> int:
    return FEATURE_NAMES[kind].index(name)


assert all(len(VOCAB[k]) + 2 == VOCAB_SIZES[k] for k in FEATURE_NAMES)
