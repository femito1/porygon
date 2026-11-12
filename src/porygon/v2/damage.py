"""Batched damage / speed features for model v2 (owner: netv2). A parameter-free `nn.Module` (buffers only, all
non-persistent) that turns a batch of v2 observations (porygon.v2.spec; dict of tensors with a leading batch dim,
integer fields of any integer dtype, floats float32) into damage and speed-order features. Fully vectorised: a fixed
~650 aten ops per call whatever B (no Python loop over B, mons, moves, items or abilities).

    feats = DamageFeatures(gen)(obs)        # gen 4 or 9; DamageFeatures(gen, trim=False) for static shapes
    feats["move_vs"]         [B, 12, 24, 6, 4]  attacker mon i, its move slot j, vs mon k of the OTHER side
                                                (i < 6: defenders 6..11; i >= 6: defenders 0..5)
                             (0) expected damage / defender max HP (mean of the 16 rolls, no crit, accuracy not
                                 applied; clipped to [0, 4]), (1) expected damage / defender current HP (clip [0, 2]),
                             (2) P(KO now) = share of rolls >= current HP, (3) = (0) x accuracy
    feats["active_tera_vs"]  [B, 4, 6, 4]       our active mon's move slots 0-3 vs the 6 opponent mons as if it had
                                                terastallized (to `tera`) first; zeros in gen 4 or without an active
    feats["speed"]           [B, 12, 6]         P(mon i moves before mon k of the other side at equal priority)
    feats["pair"]            [B, 12, 12, 3]     opposite-side pairs (i, k): (0) i's best move vs k (feature 0, over
                                                move slots with move_p >= 0.5 or known), (1) the same from k to i,
                                                (2) speed[i before k]; zero for same-side pairs / absent mons
Exact zeros for status moves, empty / unknown move slots and absent or fainted mons on either side; immune pairs
(type chart, Levitate, ...) get ~1e-13 rather than an exact 0 (log-space tables, see NEG).
`trim=True` (default) computes only the move slots used anywhere in the batch (real observations use <= 10 of the
24) and pads the rest with zeros; it reads that count with one host sync. Use trim=False for CUDA graphs / compile.

Damage model (gen-correct for gens 4 and 9; float arithmetic, no intermediate floors):
    damage = ((2L/5 + 2) * BP * A / D / 50 + 2) * modifiers * roll, roll uniform over 0.85..1.00
- A, D: raw stats x boost stages (Unaware on the other side mixes in the unboosted stat; Mold Breaker ignores it),
  Huge / Pure Power, Hustle, Gorilla Tactics, Guts (incl. burn), Defeatist, Toxic / Flare Boost, Solar Power,
  Slow Start, the Ruin abilities (as the holder's own x4/3), Protosynthesis / Quark Drive (highest raw stat, active
  when sun / electric terrain or Booster Energy may be active), Choice Band / Specs, Muscle Band, Wise Glasses,
  Light Ball, Thick Club, gen-4 Soul Dew, Eviolite (NFE species), Assault Vest (gen >= 6), Fur Coat, Marvel
  Scale, burn, snow (Ice Def, gen 9), sand (Rock SpD); Body Press (Def), Foul Play (target's Atk), Psyshock /
  Psystrike / Secret Sword (Def); Tera Blast (if terastallized) / Photon Geyser / Shell Side Arm take the
  category of the higher attack; Reflect / Light Screen / Aurora Veil x0.5 (Infiltrator, gen >= 5, ignores them).
- BP: table power, and eruption / water spout / dragon energy (user HP), flail / reversal, stored power / power
  trip, last respects, acrobatics (no item), facade, weather ball (power and type), psyblade, solar beam / blade,
  tera blast (type, stellar 100), heavy slam / heat crash and low kick / grass knot (weights), hex / infernal
  parade / bitter malice, knock off (x1.5 if the target may hold an item, gen >= 6), poltergeist (fails without
  one), judgment / revelation dance / multi-attack (user's primary type, revelation dance: tera type), ivy cudgel
  / raging bull (user's secondary type), return / frustration 102, gen-4 hidden power 70, gen-4 Explosion /
  Self-Destruct x2 (Defence halving), tera-type moves below 60 BP raised to 60; multi-hit expected hits (2-5:
  3.0 in gen 4, 3.1 later; Skill Link, Loaded Dice; triple axel / triple kick / population bomb with per-hit
  accuracy), Technician (on the table BP).
- modifiers: STAB 1.5 / Adaptability 2, gen-9 tera rules (tera-type STAB, original types kept, 2x / 2.25x when
  they coincide, stellar 2x on original types and 1.2x otherwise); the generation's type chart against the
  defender's tera type (not stellar), stellar moves 2x vs terastallized mons, Scrappy / Mind's Eye, Freeze-Dry,
  Magnet Rise; sun / rain (desolate land / primordial sea cancel water / fire); terrains for grounded mons (x1.3
  gen >= 8, x1.5 gen 6-7; misty halves dragon); Life Orb, Expert Belt, type-boosting items / plates / incenses,
  legendary orbs / cores, Ogerpon masks, gen-9 Soul Dew, Punching Glove; -ate abilities (applied when more
  likely than not), pinch abilities (Blaze ...), Steelworker / Transistor / Dragon's Maw / Rocky Payload /
  Water Bubble, Sand Force, Supreme Overlord, Iron Fist, Tough Claws, Strong Jaw, Sharpness, Mega Launcher,
  Reckless, Punk Rock, Sheer Force, Tinted Lens; Flash Fire / Charge volatiles; defender Thick Fat, Heatproof,
  Dry Skin, Fluffy, Water Bubble, Purifying Salt, Ice Scales, Punk Rock, Multiscale / Shadow Shield (full HP),
  Filter / Solid Rock / Prism Armor, type-resist berries, Glaive Rush; will-crit moves (x1.5 gen >= 6).
- immunities (probabilities): type chart, Levitate, Earth Eater, Flash Fire, Well-Baked Body, Water Absorb, Dry
  Skin, Storm Drain / Lightning Rod (gen >= 5), Volt Absorb, Motor Drive, Sap Sipper, Wonder Guard, Bulletproof,
  Soundproof, Wind Rider, an intact Disguise, Air Balloon, harsh sun / heavy rain. Mold Breaker / Teravolt /
  Turboblaze / Mycelium Might ignore defender abilities.
- fixed damage: seismic toss / night shade / psywave (level), super fang / ruination / nature's madness (half
  current HP), sonic boom / dragon rage, endeavor, OHKO moves (current HP; Sturdy blocks).
- P(KO): Focus Sash / Focus Band / Sturdy (gen >= 5) at full HP stop single-hit KOs.
- accuracy: move accuracy x accuracy / evasion stages, thunder / hurricane / blizzard in weather, No Guard (either
  side), Compound Eyes, Hustle, Wide Lens / Zoom Lens (x1.1); OHKO 30%.
- speed: raw Spe x stage x Protosynthesis / Quark Drive / Slow Start x paralysis (x0.25 gen <= 6, x0.5 gen >= 7)
  x tailwind (x2), with 2 ability bins (plain; Swift Swim / Chlorophyll / Sand Rush / Slush Rush / Surge Surfer
  x2 in their weather / terrain, Unburden x2 with no item, or Quick Feet: x1.5 if statused, no paralysis drop)
  x 2 item bins (none; Choice Scarf x1.5 or Iron Ball x0.5); trick room reverses; ties 0.5. The bins' probabilities
  come from the candidate distributions and are integrated exactly over both mons.

Uncertain opponent sets: item / ability effects are expectations over the candidate distributions (`item`/
`item_p`, `ability`/`ability_p`), attacker item, attacker ability, defender item and defender ability taken as
independent; most factors are exact expectations, the few that couple attacker and defender (Mold Breaker vs
defender abilities, Expert Belt / Tinted Lens vs effectiveness, Scrappy, the move-flag ability factors) are
geometric (exact when the probability is 0 or 1). P(KO) uses the expected multiplier given no immunity times
P(no immunity), i.e. ignores the extra spread from the uncertainty. The opponent's stats are the posterior
expectation in the observation, used as if exact.

Not modelled (rare in random battles or needing information the observation lacks): crits other than will-crit
moves, spread moves; Analytic, Stakeout, Parental Bond, Normalize, Liquid Voice, Orichalcum Pulse / Hadron
Engine, Flower Gift, Rivalry, Plus / Minus, Battery / Power Spot, Tera Shell, Ice Face, Cloud Nine / Air Lock,
Protean / Libero (the observed types are used); Metronome (item), gems, Utility Umbrella, Ability Shield;
order-dependent doubling (payback, avalanche, revenge, assurance, pursuit, bolt beak, fishious rend),
collision course / electro drift SE boost, rage fist hits, stomping tantrum / temper flare / lash out, expanding
force, rising voltage, grav apple, beat up, magnitude / present / natural gift / fling / spit up (fixed averages);
gyro ball / electro ball / crush grip / wring out / hard press / punishment / venoshock / brine / wake-up slap /
smelling salts use a fixed power (they do not occur in the gen-4 / gen-9 random-battle sets); counter / mirror
coat / metal burst / comeuppance / bide / final gambit (no damage); Technician on variable-power moves;
substitute, roost type loss, smack down, grassy terrain halving earthquake, illusion, the facade burn exemption,
damage floors between steps (~1% of max HP).
"""

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as Fn

from porygon.v2 import tables as T
from porygon.v2.spec import FIELD_NUM, INDEX, KM, MON_NUM, N_MONS, TYPE_ID, VOCAB_SIZES

N_MOVE_VS = 4
N_PAIR = 3
NTV = 22  # type rows of the per-pair tables: 0 = typeless, 1..19 = spec.TYPES, 20 = freeze-dry, 21 = no damage
FREEZEDRY_T, INVALID_T = 20, 21
NEG = -30.0  # log of "zero" in the tables (exact zeros come from masks; exp underflow is very slow on CPU)
COL = {n: i for i, n in enumerate(MON_NUM)}
FCOL = {n: i for i, n in enumerate(FIELD_NUM)}
EFFECT = {"magnetrise": 13, "slowstart": 24, "flashfire": 25, "charge": 26, "glaiverush": 40}  # EFFECTS index
t_ = TYPE_ID
LN = math.log

# tables.SPECIAL code -> (power function, type function, category function, d-kind)
PF = {1: 1, 2: 2, 7: 3, 18: 4, 11: 5, 12: 6, 9: 7, 25: 8, 26: 9, 10: 10}
PF_ABS = {2, 3, 4}  # absolute power functions (the pack's bp is 1)
TF = {9: 1, 10: 2, 21: 3, 22: 4}
CF = {10: 1, 23: 2}
# defender-dependent power kinds kept (the others never occur in the gen-4 / gen-9 random-battle sets and use a
# fixed power, BP_FALLBACK): 1 heavy slam / heat crash, 2 low kick / grass knot, 3 hex-likes, 4 knock off,
# 5 poltergeist (fails without a target item)
DKINDS = {6: 1, 5: 2, 13: 3, 16: 4, 27: 5}
DK_ABS = {1, 2}
BP_FALLBACK = {"gyroball": 60, "electroball": 60, "crushgrip": 100, "wringout": 100, "hardpress": 75}
# fixed-damage kind (tables.FIXED) -> fixed row: level (seismic toss, night shade, psywave), half current HP,
# constant (sonic boom, dragon rage), endeavor, OHKO; final gambit is not modelled (no damage)
FIXED_ROW = {1: 0, 6: 0, 2: 1, 3: 2, 4: 3, 7: 4}

# ---- ability table: unconditional columns (probabilities / deviations), then context-dependent deviations
AB1 = ["adaptability", "technician", "skilllink", "moldbreaker", "scrappy", "infiltrator", "unaware", "noguard",
       "compoundeyes", "hustle", "tintedlens", "quickfeet", "proto_p", "wonderguard", "filter", "multiscale",
       "disguise", "sturdy", "sturdy_ohko", "icescales", "fluffy", "punkrock", "soundproof", "bulletproof",
       "windrider", "levitate", "sandforce", "supremeoverlord"]
A1 = {n: i for i, n in enumerate(AB1)}
O_ATTRB, O_ATE, O_TB, O_PINCH, O_DT, O_IMM = (len(AB1) + k for k in (0, 10, 30, 50, 70, 90))
N_AB1 = len(AB1) + 110  # + move-flag boosts (10), ate, type boost, pinch, defensive type dev, immunity (20 each)
CTX = ["one", "statused", "brn", "psn", "hph", "sun", "rain", "sand", "hail", "eterr", "noitem"]
CX = {n: i for i, n in enumerate(CTX)}
AB2 = ["atk", "def", "spa", "spd", "dbl", "proto_cond"]  # stat-multiplier deviations, P(speed doubled), P(proto cond)
A2 = {n: i for i, n in enumerate(AB2)}
# ---- item table (probabilities / deviations)
IT = ["atk", "def", "spa", "spd", "eviolite", "allmoves", "expertbelt", "choicescarf", "ironball", "focussash",
      "airballoon", "loadeddice", "noitem", "accuracy", "punchingglove", "boosterenergy", "stab_item"]
I1 = {n: i for i, n in enumerate(IT)}
O_ITB, O_BERRY = len(IT), len(IT) + 20
N_IT = len(IT) + 40


def _log(x: Tensor) -> Tensor:
    return torch.log(x.clamp(min=1e-13))


def _ability_tables(gen: int) -> tuple[np.ndarray, np.ndarray]:
    G = T._ABILITY_GROUPS
    n = VOCAB_SIZES["abilities"]
    t1 = np.zeros((n, N_AB1), np.float32)
    t2 = np.zeros((n, len(CTX), len(AB2)), np.float32)
    simple = {k: G[k] for k in ("adaptability", "technician", "skilllink", "moldbreaker", "scrappy", "infiltrator",
                                "unaware", "noguard", "compoundeyes", "hustle", "tintedlens", "quickfeet",
                                "wonderguard", "filter", "multiscale", "disguise", "icescales", "fluffy", "punkrock",
                                "soundproof", "bulletproof", "windrider", "sandforce")}
    simple["proto_p"] = {"protosynthesis", "quarkdrive"}
    simple["sturdy_ohko"] = {"sturdy"}
    simple["sturdy"] = {"sturdy"} if gen >= 5 else set()
    simple["levitate"] = {"levitate"}
    simple["supremeoverlord"] = {"supremeoverlord"}
    attrb = {"toughclaws": (0, 0.3), "punkrock": (1, 0.3), "ironfist": (2, 0.2), "strongjaw": (3, 0.5),
             "sharpness": (4, 0.5), "megalauncher": (5, 0.5), "reckless": (8, 0.2), "sheerforce": (9, 0.3)}
    tb = {"steelworker": 0.5, "steelyspirit": 0.5, "transistor": 0.3 if gen >= 9 else 0.5, "dragonsmaw": 0.5,
          "rockypayload": 0.5, "waterbubble": 1.0}
    dt = {"thickfat": {"fire": -0.5, "ice": -0.5}, "heatproof": {"fire": -0.5}, "dryskin": {"fire": 0.25},
          "fluffy": {"fire": 1.0}, "waterbubble": {"fire": -0.5}, "purifyingsalt": {"ghost": -0.5}}
    c = CX
    # stat deviations relative to (1 - burn / 2) for atk (Guts removes the burn halving)
    atk = {"hugepower": {"one": 1.0, "brn": -0.5}, "purepower": {"one": 1.0, "brn": -0.5},
           "hustle": {"one": 0.5, "brn": -0.25}, "gorillatactics": {"one": 0.5, "brn": -0.25},
           "guts": {"statused": 0.5, "brn": 0.5}, "defeatist": {"hph": -0.5},
           "toxicboost": {"psn": 0.5}}
    spa = {"solarpower": {"sun": 0.5}, "defeatist": {"hph": -0.5}, "flareboost": {"brn": 0.5}}
    dfn = {"furcoat": {"one": 1.0}, "marvelscale": {"statused": 0.5}}
    # Ruin abilities as the holder's own multipliers: Sword (foes' Def x0.75) / Beads (SpD) help its attacks,
    # Tablets (foes' Atk x0.75) / Vessel (SpA) its defence
    atk["swordofruin"] = {"one": 1 / 3, "brn": -1 / 6}
    spa["beadsofruin"] = {"one": 1 / 3}
    dfn["tabletsofruin"] = {"one": 1 / 3}
    spd = {"vesselofruin": {"one": 1 / 3}}
    dbl = {"swiftswim": "rain", "chlorophyll": "sun", "sandrush": "sand", "slushrush": "hail", "surgesurfer": "eterr",
           "unburden": "noitem"}
    proto = {"protosynthesis": "sun", "quarkdrive": "eterr"}
    for name, i in INDEX["abilities"].items():
        for k, members in simple.items():
            if name in members:
                t1[i, A1[k]] = 1.0
        if name in attrb:
            j, v = attrb[name]
            t1[i, O_ATTRB + j] = v
        if name in T.ABILITY_TYPE_ROLE and not (gen <= 4 and name in T.GEN5_IMMUNITIES):
            role, tn = T.ABILITY_TYPE_ROLE[name]
            if role == "ate" and name != "normalize":
                t1[i, O_ATE + t_[tn]] = 1.0
            elif role == "pinch_boost":
                t1[i, O_PINCH + t_[tn]] = 1.0
            elif role == "type_boost":
                t1[i, O_TB + t_[tn]] = tb.get(name, 0.5)
            elif role == "immune":
                t1[i, O_IMM + t_[tn]] = 1.0
        for tn, v in dt.get(name, {}).items():
            t1[i, O_DT + t_[tn]] = v
        for k, dd in (("atk", atk), ("spa", spa), ("def", dfn), ("spd", spd)):
            for cn, v in dd.get(name, {}).items():
                t2[i, c[cn], A2[k]] = v
        if name in dbl:
            t2[i, c[dbl[name]], A2["dbl"]] = 1.0
        if name in proto:
            t2[i, c[proto[name]], A2["proto_cond"]] = 1.0
    t1[:2] = 0.0
    t2[:2] = 0.0
    return t1, t2.reshape(n, -1)


def _item_table(gen: int) -> np.ndarray:
    G = T._ITEM_GROUPS
    n = VOCAB_SIZES["items"]
    t = np.zeros((n, N_IT), np.float32)
    dev = {"atk": {"choiceband": 0.5, "muscleband": 0.1, "lightball": 1.0, "thickclub": 1.0},
           "spa": {"choicespecs": 0.5, "wiseglasses": 0.1, "lightball": 1.0, **({"souldew": 0.5} if gen <= 6 else {})},
           "spd": {**({"assaultvest": 0.5} if gen >= 6 else {}), **({"souldew": 0.5} if gen <= 6 else {})},
           "allmoves": {"lifeorb": 0.3, **{m: 0.2 for m in T.MASKS}}, "accuracy": {"widelens": 0.1, "zoomlens": 0.1},
           "punchingglove": {"punchingglove": 0.1}}
    flag = {"eviolite": G["eviolite"], "expertbelt": G["expertbelt"], "choicescarf": G["choicescarf"],
            "ironball": G["ironball"], "focussash": G["focussash"], "airballoon": G["airballoon"],
            "loadeddice": G["loadeddice"], "boosterenergy": G["boosterenergy"], "stab_item": T.STAB_ITEMS}
    for name, i in INDEX["items"].items():
        for k, d in dev.items():
            if name in d:
                t[i, I1[k]] = d[name]
        for k, members in flag.items():
            if name in members:
                t[i, I1[k]] = 1.0
        if name in T.TYPE_BOOST_ITEMS:
            t[i, O_ITB + t_[T.TYPE_BOOST_ITEMS[name]]] = 0.2
        if name == "souldew" and gen >= 7:
            t[i, O_ITB + t_["dragon"]] = t[i, O_ITB + t_["psychic"]] = 0.2
        if name in T.RESIST_BERRIES:
            t[i, O_BERRY + t_[T.RESIST_BERRIES[name]]] = 1.0
    t[:2] = 0.0
    t[0, I1["noitem"]] = 1.0  # id 0 = no item (unknown id 1 stays neutral)
    return t


MPACK = ["category", "type", "tfid", "cfid", "pfid", "tcol_ovr", "row", "pkcls", "acc", "acc_kind", "isfx", "nf",
         "valid",
         "roll", "lo", "inv", "tera60",
         # columns of the additive log factor (dotted with per-attacker weights, porygon.v2.damage._calc)
         "lbp", "lconst", "lh", "lsl", "lld", "tech_ok", "ate_ok",
         "contact", "sound", "punch", "bite", "slicing", "pulse", "bullet", "wind", "recoil", "secondary"]
MP = {n: i for i, n in enumerate(MPACK)}
L0 = MP["lbp"]
N_LIN = len(MPACK) - L0  # 7 + 10 move-flag columns
N_ATTR = 10


def _move_pack(gen: int) -> tuple[np.ndarray, list]:
    """Static per-move columns (vocab id -> value), everything decided once per generation."""
    raw = T.move_raw(gen)
    n = VOCAB_SIZES["moves"]
    pk = np.zeros((n, len(MPACK)), np.float32)
    cat = raw["category"].astype(np.float32)
    dmg = (raw["known"] > 0) & np.isin(raw["category"], (1, 2))
    dmg[:2] = False
    bp = raw["bp"].copy()
    if gen <= 4:
        bp[raw["selfdestruct"] > 0] *= 2  # gen <= 4 halves the target's Defence
    for name, v in BP_FALLBACK.items():
        if name in INDEX["moves"]:
            bp[INDEX["moves"][name]] = v
    dmg &= (raw["fixed"] != 5) & ((bp > 0) | (raw["fixed"] > 0))  # final gambit; counter / mirror coat / bide ...
    fixed = raw["fixed"] * dmg
    isfx = (fixed > 0).astype(np.float32)
    pk[:, MP["category"]] = cat
    pk[:, MP["type"]] = raw["type"]
    pk[:, MP["tcol_ovr"]] = np.where(~dmg, INVALID_T, 0)
    if "freezedry" in INDEX["moves"]:
        pk[INDEX["moves"]["freezedry"], MP["tcol_ovr"]] = FREEZEDRY_T
    acc = np.where(raw["accuracy"] >= 0.999, 10.0, raw["accuracy"])  # always hits -> 10 (clamped to 1 later)
    pk[:, MP["acc"]] = np.where(fixed == 7, 0.3, acc)
    pk[:, MP["acc_kind"]] = raw["acc_kind"]
    pk[:, MP["isfx"]] = isfx
    pk[:, MP["valid"]] = dmg
    pk[:, MP["nf"]] = 1 - isfx
    pk[:, MP["roll"]] = np.where(isfx > 0, 1.0, 0.925)
    pk[:, MP["lo"]] = np.where(isfx > 0, 100.0, 85.0)
    pk[:, MP["inv"]] = np.where(isfx > 0, 1.0, 1 / 16)
    pk[:, MP["pkcls"]] = np.where(fixed == 7, 2, np.where((raw["max_hits"] > 1) & (isfx == 0), 1, 0))
    # aux rows: 0 phys, 1 spec, 2 psyshock, 3 body press, 4 foul play, 5.. (d-kind, category), then fixed kinds 1-7
    so = np.where(raw["phys_vs_def"] > 0, 2, np.where(raw["uses_def"] > 0, 3, np.where(raw["uses_target_atk"] > 0,
                                                                                       4, 0)))
    row = np.where(so > 0, so, np.clip(cat - 1, 0, 1))
    combos = sorted({(DKINDS[int(s)], int(c)) for s, c, ok in zip(raw["special"], raw["category"], dmg)
                     if ok and int(s) in DKINDS and not (gen <= 5 and DKINDS[int(s)] == 4)})
    crow = {cb: 5 + j for j, cb in enumerate(combos)}
    fixed_row0 = 5 + len(combos)
    absolute = np.zeros(n, bool)
    for i in range(n):
        s, c = int(raw["special"][i]), int(raw["category"][i])
        if not dmg[i]:
            continue
        if s in PF:
            pk[i, MP["pfid"]] = PF[s]
            absolute[i] |= PF[s] in PF_ABS
        if s in TF:
            pk[i, MP["tfid"]] = TF[s]
        if s in CF:
            pk[i, MP["cfid"]] = CF[s]
        if (DKINDS.get(s), c) in crow:
            row[i] = crow[(DKINDS[s], c)]
            absolute[i] |= DKINDS[s] in DK_ABS
    if "revelationdance" in INDEX["moves"]:
        pk[INDEX["moves"]["revelationdance"], MP["tfid"]] = 5
    frow = np.array([FIXED_ROW.get(int(k), 0) for k in fixed])
    row = np.where(fixed > 0, fixed_row0 + frow, row)
    pk[:, MP["row"]] = np.where(dmg, row, 0)
    bpp = np.where(absolute | (isfx > 0), 1.0, bp)
    nf = 1 - isfx
    pk[:, MP["lbp"]] = np.log(np.maximum(bpp, 1e-6)) * dmg
    lconst = LN(1.5 if gen >= 6 else 2.0) * raw["will_crit"] * nf
    lconst = lconst + np.where(fixed == 3, np.log(np.maximum(raw["fixed_value"], 1e-6)), 0.0)
    pk[:, MP["lconst"]] = lconst * dmg
    hits, mx = np.maximum(raw["hits"], 1e-6), np.maximum(raw["max_hits"], 1)
    var = np.maximum(raw["variable_hits"], raw["multiaccuracy"]) > 0
    hi = np.where(raw["variable_hits"] > 0, 4.5, mx)
    pk[:, MP["lh"]] = np.log(hits) * nf * dmg
    pk[:, MP["lsl"]] = np.where(var, np.log(mx / hits), 0.0) * nf * dmg
    pk[:, MP["lld"]] = np.where(var, np.log(hi / hits), 0.0) * nf * dmg * (gen >= 9)
    pk[:, MP["tech_ok"]] = (bp <= 60) & ~absolute & (raw["special"] == 0) & (nf > 0) & dmg
    pk[:, MP["ate_ok"]] = (raw["type"] == t_["normal"]) & (pk[:, MP["tfid"]] == 0) & (nf > 0) & dmg
    if gen >= 9:  # tera-type moves below 60 BP become 60
        ok = (bp < 60) & (raw["priority"] <= 0) & (raw["max_hits"] <= 1) & ~absolute & (nf > 0) & dmg \
            & (pk[:, MP["pfid"]] == 0)
        pk[:, MP["tera60"]] = np.where(ok, np.log(60 / np.maximum(bp, 1e-6)), 0.0)
    for f in ("contact", "sound", "punch", "bite", "slicing", "pulse", "bullet", "wind"):
        pk[:, MP[f]] = raw[f] * nf * dmg
    pk[:, MP["recoil"]] = (raw["recoil"] > 0) * nf * dmg
    pk[:, MP["secondary"]] = raw["has_secondary"] * nf * dmg
    return pk, combos


# columns of the defender-dependent power table: (attacker, defender) kinds first, then defender-only kinds
DK_COL = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


class DamageFeatures(nn.Module):
    """See the module docstring. `trim`: compute only the move slots used anywhere in the batch (one host sync to
    read the count; outputs are padded back to 24). Buffers only, all non-persistent."""

    def __init__(self, gen: int, trim: bool = True):
        super().__init__()
        self.gen, self.trim = gen, trim
        buf = lambda name, x: self.register_buffer(name, torch.as_tensor(np.asarray(x)), persistent=False)  # noqa: E731
        pk, combos = _move_pack(gen)
        buf("m_pack_t", pk.T.copy())
        J = len(combos)
        self.n_rows = 5 + J + 5
        buf("combo_r", np.array([c - 1 for _, c in combos], np.int64))
        buf("combo_dk", np.array([DK_COL[k] for k, _ in combos], np.int64))
        sp = T.species_raw(gen)
        buf("s_tab", np.stack([sp["nfe"], sp["disguise"]], 1))
        t1, t2 = _ability_tables(gen)
        buf("a_tab", np.concatenate([t1, t2], 1))
        buf("i_tab", _item_table(gen))
        chart = np.ones((20, 20), np.float32)  # [attacking type, defending type]
        for dname, r in T._gendata(gen).type_chart.items():
            dt = t_.get(dname.lower())
            for aname, v in r.items():
                at = t_.get(aname.lower())
                if at and dt:
                    chart[at, dt] = v
        scr = chart.copy()
        scr[[t_["normal"], t_["fighting"]], t_["ghost"]] = 1.0
        both = np.concatenate([chart.T, scr.T], 1)  # [defending type, 40]: normal | scrappy attacking types
        buf("chart2", both)

        fcols = ["sun", "rain", "harsh", "heavy", "sand", "hail", "snow", "eterr", "gterr", "mterr", "pterr", "tr",
                 "gravity", "any", "weak"]
        self.FC = {nme: i for i, nme in enumerate(fcols)}
        W = np.zeros((len(FIELD_NUM), len(fcols)), np.float32)

        def w(fname, cols):
            for cn in cols:
                W[FCOL[fname], self.FC[cn]] = 1.0
        w("weather_sunnyday", ["sun", "any"])
        w("weather_desolateland", ["sun", "harsh", "any"])
        w("weather_raindance", ["rain", "any", "weak"])
        w("weather_primordialsea", ["rain", "heavy", "any", "weak"])
        w("weather_sandstorm", ["sand", "any", "weak"])
        w("weather_hail", ["hail", "any", "weak"])
        w("weather_snow", ["hail", "snow", "any", "weak"])
        for fname, cn in (("terrain_electricterrain", "eterr"), ("terrain_grassyterrain", "gterr"),
                          ("terrain_mistyterrain", "mterr"), ("terrain_psychicterrain", "pterr"), ("trickroom", "tr"),
                          ("gravity", "gravity")):
            w(fname, [cn])
        if gen < 9:
            W[:, self.FC["snow"]] = 0.0
        buf("f_W", W)
        Wv = np.zeros((len(FIELD_NUM), 20), np.float32)  # weather type multiplier - 1
        fi, wa = t_["fire"], t_["water"]
        Wv[FCOL["weather_sunnyday"], [fi, wa]] = [0.5, -0.5]
        Wv[FCOL["weather_desolateland"], [fi, wa]] = [0.5, -1.0]
        Wv[FCOL["weather_raindance"], [fi, wa]] = [-0.5, 0.5]
        Wv[FCOL["weather_primordialsea"], [fi, wa]] = [-1.0, 0.5]
        buf("f_Wv", Wv)
        wb = np.zeros(len(FIELD_NUM), np.float32)  # weather ball type = normal + fn @ wb
        for fname, tn in (("weather_sunnyday", "fire"), ("weather_desolateland", "fire"),
                          ("weather_raindance", "water"), ("weather_primordialsea", "water"),
                          ("weather_sandstorm", "rock"), ("weather_hail", "ice"), ("weather_snow", "ice")):
            wb[FCOL[fname]] = t_[tn] - t_["normal"]
        buf("f_wb", wb)
        # accuracy by kind: acc' = acc * m + a; columns (m0 a0 m1 a1 m2 a2) = const + fn @ Wacc
        Wa = np.zeros((len(FIELD_NUM), 6), np.float32)
        for fname, rr, ss in (("weather_raindance", 1, 0), ("weather_primordialsea", 1, 0),
                              ("weather_sunnyday", 0, 1), ("weather_desolateland", 0, 1)):
            Wa[FCOL[fname], 2] = -1.0
            Wa[FCOL[fname], 3] = 1.0 if rr else 0.5
        for fname in ("weather_hail", "weather_snow"):
            Wa[FCOL[fname], 4] = -1.0
            Wa[FCOL[fname], 5] = 1.0
        buf("f_Wacc", Wa)
        buf("acc_c", np.array([1, 0, 1, 0, 1, 0], np.float32))
        tb_ = 1.3 if gen >= 8 else 1.5
        E = np.zeros((5, 20), np.float32)  # log effects on the attacker's type vector
        E[0, t_["fire"]] = LN(1.5)
        E[1, t_["electric"]] = LN(2.0)
        if gen >= 6:
            E[2, t_["electric"]] = E[3, t_["grass"]] = E[4, t_["psychic"]] = LN(tb_)
        buf("e_logv", E)
        rgs = np.zeros(20, np.float32)
        rgs[[t_["rock"], t_["ground"], t_["steel"]]] = 0.3
        buf("v_sandforce", rgs)
        buf("stage_tab", np.array([2 / (2 - s) if s < 0 else (2 + s) / 2 for s in range(-6, 7)], np.float32))
        buf("acc_stage_tab", np.array([3 / (3 - s) if s < 0 else (3 + s) / 3 for s in range(-6, 7)], np.float32))
        buf("flail_b", np.array([1.5, 4.5, 9.5, 16.5, 32.5], np.float32))
        buf("flail_v", np.array([200, 150, 100, 80, 40, 20], np.float32))
        buf("lk_b", np.array([10, 25, 50, 100, 200], np.float32))
        buf("lk_v", np.array([20, 40, 60, 80, 100, 120], np.float32))
        buf("hs_b", np.array([2, 3, 4, 5], np.float32))
        buf("hs_v", np.array([40, 60, 80, 100, 120], np.float32))
        oh = lambda i, k=NTV: np.eye(k, dtype=np.float32)[i]  # noqa: E731
        buf("e_ground", oh(t_["ground"]))
        buf("e_dragon", oh(t_["dragon"]))
        buf("e_normal", oh(t_["normal"]))
        buf("e_stellar20", oh(t_["stellar"], 20))
        buf("e_ground20", oh(t_["ground"], 20))
        buf("e_ice_fd", np.stack([oh(t_["ice"], 20), oh(t_["ice"], 20)], 1))  # [20, 2]: ice column -> freeze-dry
        no0 = np.ones(20, np.float32)
        no0[0] = 0.0
        buf("no_typeless", no0)
        buf("e_punch", oh(2, N_ATTR))
        buf("proto_v", np.array([0.3, 0.3, 0.3, 0.3, 0.5], np.float32))
        buf("brn_v", np.array([-0.5, 0, 0, 0, 0], np.float32))
        buf("evio_v", np.array([0, 0.5, 0, 0.5, 0], np.float32))
        buf("snow_v", np.array([0, 0.5, 0, 0, 0], np.float32))
        buf("sand_v", np.array([0, 0, 0, 0.5, 0], np.float32))
        buf("ss_v", np.array([0.5, 0, 0, 0, 0.5], np.float32))
        buf("fin_i", np.array([MP[k] - L0 for k in ("lconst", "lh", "lsl", "lld", "ate_ok")], np.int64))
        buf("num_i", np.array([0, 2, 2, 1, 0], np.int64))  # aux rows 0-4: attacking stat (atk def spa ...)
        buf("den_i", np.array([1, 3, 1, 1, 1], np.int64))  # defending stat
        buf("mcat_i", np.array([0, 1, 1, 0, 0], np.int64))  # screen / ice scales factor: phys 0, spec 1
        self.par_mult = 0.5 if gen >= 7 else 0.25
        buf("opp_i", np.array(list(range(6, 12)) + list(range(6)), np.int64))

    # ------------------------------------------------------------------------------------------ forward

    def forward(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        gen = self.gen
        mn = obs["mon_num"].float()
        B = mn.shape[0]
        dev = mn.device
        fn = obs["field_num"].float()
        Fv = (fn @ self.f_W).clamp_(max=1)
        FC = self.FC
        fcol = lambda n: Fv[:, FC[n]]  # noqa: E731

        # ================= per mon [B, 12, ...]
        cs = COL["status_brn"]
        stt = mn[..., cs:cs + 6]  # brn frz par psn slp tox
        brn, par = stt[..., 0], stt[..., 2]
        psn = stt[..., 3] + stt[..., 5]
        statused = stt.sum(-1).clamp_(max=1)
        hpx = obs["hp"].float()
        hp, maxhp = hpx[..., 0].clamp(min=0), hpx[..., 1].clamp(min=1)
        hpr = hp / maxhp
        alive = mn[..., COL["present"]] * (1 - mn[..., COL["fainted"]])
        ki, ka = obs["item"].shape[-1], obs["ability"].shape[-1]
        Yi = Fn.embedding_bag(obs["item"].long().view(-1, ki), self.i_tab,
                              per_sample_weights=obs["item_p"].float().reshape(-1, ki), mode="sum").view(B, N_MONS, -1)
        Ya = Fn.embedding_bag(obs["ability"].long().view(-1, ka), self.a_tab,
                              per_sample_weights=obs["ability_p"].float().reshape(-1, ka),
                              mode="sum").view(B, N_MONS, -1)
        p_noitem = Yi[..., I1["noitem"]]
        fld = Fv[:, [FC["sun"], FC["rain"], FC["sand"], FC["hail"], FC["eterr"]]][:, None].expand(B, N_MONS, 5)
        ctx = torch.cat([torch.stack([torch.ones_like(hp), statused, brn, psn, (hpr <= 0.5).float()], -1), fld,
                         (p_noitem > 0.999).float()[..., None]], -1)
        cond = torch.matmul(ctx[..., None, :], Ya[..., N_AB1:].view(B, N_MONS, len(CTX), len(AB2)))[..., 0, :]
        ab = lambda n: Ya[..., A1[n]]  # noqa: E731
        it = lambda n: Yi[..., I1[n]]  # noqa: E731
        Ys = Fn.embedding(obs["species"].long(), self.s_tab)
        types = obs["types"].long()
        t1h = Fn.one_hot(types, 20).sum(-2).float()  # [B,12,20]
        tera_t = obs["tera"].long()
        tera_flag = mn[..., COL["terastallized"]] if gen >= 9 else torch.zeros_like(hp)
        eff = {k: mn[..., COL["effect_0"] + j] for k, j in EFFECT.items()}
        st5 = obs["stats"].float()[..., 1:6].clamp(min=1)
        pc = cond[..., A2["proto_cond"]]
        active = pc + (ab("proto_p") - pc) * it("boosterenergy")
        best = Fn.one_hot(st5.argmax(-1), 5).float()
        ab5 = Fn.pad(cond[..., :4], (0, 1)).add_(1).add_(brn[..., None] * self.brn_v)  # atk def spa spd (spe = 1)
        it5 = Yi[..., [I1["atk"], I1["def"], I1["spa"], I1["spd"], I1["noitem"]]] * self.no_typeless[:5].flip(0)
        it5 = it5.add_(1).add_((it("eviolite") * Ys[..., 0])[..., None] * self.evio_v)
        fl5 = 1 + (fcol("snow")[:, None] * t1h[..., t_["ice"]])[..., None] * self.snow_v \
            + (fcol("sand")[:, None] * t1h[..., t_["rock"]])[..., None] * self.sand_v
        M5 = ab5 * it5 * fl5 * (1 - eff["slowstart"][..., None] * self.ss_v) \
            * (1 + active[..., None] * best * self.proto_v)
        bo = obs["boosts"].float().clamp(-6, 6)
        bm5 = self.stage_tab[(bo[..., :5] + 6).long()]
        # speed bins: ability (none / doubled / quick feet) x item (none / scarf / iron ball)
        tw = obs["side_num"].float()[:, :, 5].repeat_interleave(6, 1)
        sbase = st5[..., 4] * bm5[..., 4] * M5[..., 4] * (1 + tw)
        # speed: 2 ability bins (plain / doubled-or-Quick-Feet) x 2 item bins (none / scarf-or-iron-ball)
        dbl, qf = cond[..., A2["dbl"]], ab("quickfeet")
        par_f = 1 - (1 - self.par_mult) * par
        p_sa = (dbl + qf).clamp(max=1)
        v_sa = (2 * dbl * par_f + qf * (1 + 0.5 * statused)) / p_sa.clamp(min=1e-6)
        sc_, ib_ = it("choicescarf"), it("ironball")
        p_si = (sc_ + ib_).clamp(max=1)
        v_si = (1.5 * sc_ + 0.5 * ib_) / p_si.clamp(min=1e-6)
        abv = torch.stack([par_f, v_sa], -1)
        itv = torch.stack([torch.ones_like(v_si), v_si], -1)
        spe_v = torch.floor((sbase[..., None] * abv)[..., :, None] * itv[..., None, :]).flatten(-2)  # [B,12,4]
        spe_p = (torch.stack([1 - p_sa, p_sa], -1)[..., :, None] * torch.stack([1 - p_si, p_si], -1)[..., None, :]
                 ).flatten(-2)
        flying = torch.where(tera_flag > 0, (tera_t == t_["flying"]).float(), t1h[..., t_["flying"]])
        grounded = (1 - flying) * (1 - ab("levitate")) * (1 - it("airballoon")) * (1 - eff["magnetrise"])
        grounded = torch.maximum(grounded, torch.maximum(it("ironball"), fcol("gravity")[:, None]))
        hp13 = (hp <= maxhp / 3).float()
        ab_tv = torch.addcmul(Ya[..., O_TB:O_TB + 20], Ya[..., O_PINCH:O_PINCH + 20], 0.5 * hp13[..., None]) \
            + (ab("sandforce") * fcol("sand")[:, None])[..., None] * self.v_sandforce
        escal = torch.stack([eff["flashfire"], eff["charge"], grounded * fcol("eterr")[:, None],
                             grounded * fcol("gterr")[:, None], grounded * fcol("pterr")[:, None]], -1)
        latt = torch.log1p(ab_tv) + torch.log1p(Yi[..., O_ITB:O_ITB + 20]) \
            + escal @ self.e_logv
        attrb = Ya[..., O_ATTRB:O_ATTRB + N_ATTR] + it("punchingglove")[..., None] * self.e_punch
        ate_v = Ya[..., O_ATE:O_ATE + 20]
        nfz = mn[..., COL["fainted"]].view(B, 2, 6).sum(-1, keepdim=True).expand(B, 2, 6).reshape(B, N_MONS)
        weight = torch.pow(10.0, 3 * mn[..., COL["log_weight"]])
        # defender-only power factors (DK_COL 1..4): low kick / grass knot, hex, knock off, poltergeist
        dkd = torch.stack([self.lk_v[torch.bucketize(weight, self.lk_b, right=True)], 1 + statused,
                           1.5 - 0.5 * p_noitem, 1 - p_noitem], -1)
        sc = lambda x: x[..., None]  # noqa: E731
        lvl, S5, ab1, typesf, tf_, tt_ = sc(mn[..., COL["level"]] * 100), st5 * M5, Ya[..., :len(AB1)], types.float(), \
            sc(tera_flag), sc(tera_t.float())
        pa = {  # attacker fields
            "alive": sc(alive), "level": lvl, "hp": sc(hp), "hpr": sc(hpr), "S5": S5, "bm5": bm5, "eff_a": S5 * bm5,
            "acc_st": sc(bo[..., 5]), "brn": sc(brn), "st3": sc((brn + par + psn).clamp(max=1)), "types": typesf,
            "t1h": t1h, "tera_flag": tf_, "tera_t": tt_, "weight": sc(weight), "pos": sc(bo.clamp(min=0).sum(-1)),
            "nf_own": sc(nfz), "noitem": sc(p_noitem), "latt": latt, "attrb": attrb, "ate_p": sc(ate_v.sum(-1)),
            "ate_t": sc(ate_v.argmax(-1).float()), "ab1": ab1,
            "allmoves": sc((1 + it("allmoves")) * (1 + 0.1 * nfz.clamp(max=5) * ab("supremeoverlord"))),
            "expertbelt": sc(it("expertbelt")), "loadeddice": sc(it("loadeddice")),
            "acc_mult": sc((1 + 0.3 * ab("compoundeyes")) * (1 + it("accuracy"))), "stab_item": sc(it("stab_item")),
        }
        pd_ = {  # defender fields
            "alive": sc(alive), "hp": sc(hp), "maxhp": sc(maxhp),
            "full": sc((mn[..., COL["hp_frac"]] >= 0.999).float()),
            "S5": S5, "bm5": bm5, "atk_raw": st5[..., :1], "eva_st": sc(bo[..., 6]), "types": typesf, "tera_flag": tf_,
            "tera_t": tt_, "weight": sc(weight), "ab1": ab1, "grounded": sc(grounded), "sash": sc(it("focussash")),
            "balloon": sc(it("airballoon") * (1 - fcol("gravity")[:, None])), "glaive": sc(eff["glaiverush"]),
            "magnetrise": sc(eff["magnetrise"]), "disg": sc(ab("disguise") * Ys[..., 1]), "dkd": dkd,
            "dt": Ya[..., O_DT:O_DT + 20], "imm": Ya[..., O_IMM:O_IMM + 20], "berry": Yi[..., O_BERRY:O_BERRY + 20],
        }

        def pack(parts):
            off, o = {}, 0
            for k, v in parts.items():
                off[k] = slice(o, o + v.shape[-1])
                o += v.shape[-1]
            return torch.cat(list(parts.values()), -1), off
        MA, offA = pack(pa)
        MD, offD = pack(pd_)
        moves = obs["moves"].long()
        nm = KM
        if self.trim:
            used = (moves > 0).any(0).any(0)
            nm = max(4, int(used.nonzero().max()) + 1) if bool(used.any()) else 4
        # 13th attacker row: our active, terastallized (gen 9); attackers per group = 6 mons + that row (dead in
        # group 1). Moves and move types ride along in the same packed tensor.
        MA = torch.cat([MA, moves[..., :nm].float(), obs["move_type"][..., :nm].float()], -1)
        act = mn[:, :6, COL["active"]]
        aidx = act.argmax(1)
        Mt = torch.gather(MA, 1, aidx.view(B, 1, 1).expand(B, 1, MA.shape[-1])).clone()
        Mt[..., offA["tera_flag"]] = (Mt[..., offA["tera_t"]] > 0).float()
        Mt[..., offA["alive"]] *= act.amax(1).view(B, 1, 1)
        if gen < 9:
            Mt[..., offA["alive"]] = 0.0
        A = torch.cat([MA.view(B, 2, 6, -1), torch.cat([Mt, torch.zeros_like(Mt)], 1)[:, :, None]], 2)  # [B,2,7,K]
        D = MD.view(B, 2, 6, -1).flip(1)
        k0 = A.shape[-1] - 2 * nm
        mvA, mtA = A[..., k0:k0 + nm].long(), A[..., k0 + nm:]
        accw = (self.acc_c + fn @ self.f_Wacc).view(B, 3, 2)
        f4 = self._calc(A, D, mvA, mtA, offA, offD, Fv, fn @ self.f_wb + t_["normal"],
                         obs["side_num"].float()[:, [1, 0], :3], accw, _log(1 + fn @ self.f_Wv))
        # f4: 4 x [B,G,ND,NA,nm] -> move_vs [B,12,24,6,4] (zero-padded slots), tera_vs [B,4,6,4]
        move_vs = torch.zeros(B, 2, 6, KM, 6, N_MOVE_VS, device=dev)
        tera_vs = torch.zeros(B, 4, 6, N_MOVE_VS, device=dev)
        for k, x in enumerate(f4):
            move_vs[:, :, :, :nm, :, k] = x[:, :, :, :6].permute(0, 1, 3, 4, 2)
            tera_vs[..., k] = x[:, 0, :, 6, :4].transpose(1, 2)
        move_vs = move_vs.view(B, N_MONS, KM, 6, N_MOVE_VS)

        # speed [B,12,6]: sum over both mons' bins of p_i p_k (1 + s sign(v_i - v_k)) / 2, s = -1 in trick room
        vk = spe_v[:, self.opp_i].view(B, 2, 1, 6, 4)
        pk = spe_p[:, self.opp_i].view(B, 2, 6, 4)
        sgn = torch.sign(spe_v.view(B, 2, 6, 1, 4, 1) - vk[..., None, :])  # [B,2,6,6,4,4]
        e = ((sgn * spe_p.view(B, 2, 6, 1, 4, 1)).sum(-2) * pk[:, :, None]).sum(-1)
        ao = alive[:, self.opp_i].view(B, 2, 1, 6)
        speed = (0.5 + 0.5 * (1 - 2 * fcol("tr")).view(B, 1, 1, 1) * e) * alive.view(B, 2, 6, 1) * ao
        speed = speed.reshape(B, N_MONS, 6)

        # pair [B,12,12,3]
        usable = ((obs["move_num"][..., 0] > 0) | (obs["move_p"] >= 0.5))[..., :nm].float()
        best_mv = (f4[0][:, :, :, :6] * usable.view(B, 2, 1, 6, nm)).amax(-1).transpose(2, 3).reshape(B, N_MONS, 6)
        z = torch.zeros(B, 6, 6, device=dev)
        bfull = torch.cat([torch.cat([z, best_mv[:, :6]], 2), torch.cat([best_mv[:, 6:], z], 2)], 1)
        sfull = torch.cat([torch.cat([z, speed[:, :6]], 2), torch.cat([speed[:, 6:], z], 2)], 1)
        pair = torch.stack([bfull, bfull.transpose(1, 2), sfull], -1)
        return {"move_vs": move_vs, "active_tera_vs": tera_vs, "speed": speed, "pair": pair}

    # ------------------------------------------------------------------------------------------ core

    def _calc(self, A: Tensor, D: Tensor, mv: Tensor, mt_obs: Tensor, offA: dict, offD: dict, Fv: Tensor,
              wb_type: Tensor, screens: Tensor, accw: Tensor, lweather: Tensor) -> tuple[Tensor, ...]:
        """Group g of attackers (g = 0: our 6 mons + our terastallized active, g = 1: their 6 mons + a dead row) vs
        the other side's 6 mons. A [B,G,NA,K] / D [B,G,ND,K] packed mon features (offA / offD: column slices),
        mv / mt_obs [B,G,NA,NM] move ids / observed move types, Fv [B,15] field flags, wb_type [B] weather ball
        type, screens [B,G,3] on the defenders' side, accw [B,3,2] accuracy (mult, add) by kind, lweather [B,20]
        log weather multiplier by type. -> (f0, f1, P(KO), f3), each [B,G,ND,NA,NM]."""
        gen = self.gen
        B, G, NA, NM = mv.shape
        ND = D.shape[2]
        dev = mv.device
        FC = self.FC
        a = lambda k: A[..., offA[k]]  # noqa: E731
        d = lambda k: D[..., offD[k]]  # noqa: E731
        a1 = lambda k: A[..., offA[k].start]  # noqa: E731
        d1 = lambda k: D[..., offD[k].start]  # noqa: E731
        ab1a, ab1d = a("ab1"), d("ab1")
        aa = lambda n: ab1a[..., A1[n]]  # noqa: E731
        ad_ = lambda n: ab1d[..., A1[n]]  # noqa: E731
        u = lambda x: x[..., None]  # noqa: E731
        dpair = lambda x: x[:, :, None, :]  # defender [B,G,ND] -> [B,G,1,ND]  # noqa: E731

        # ================= a-m [B,G,NA,NM]
        Pt = self.m_pack_t.index_select(1, mv.reshape(-1)).view(-1, B, G, NA, NM)  # one contiguous plane per column
        c = lambda n: Pt[MP[n]]  # noqa: E731
        tera, tt, ty = a1("tera_flag"), a1("tera_t"), a("types")
        t0, t1 = ty[..., 0], ty[..., 1]
        mtype = torch.where(mt_obs > 0, mt_obs, c("type"))
        # type functions: 1 weather ball, 2 tera blast, 3 primary type, 4 secondary-or-primary, 5 revelation dance
        tf_tab = torch.stack([t0, wb_type.view(B, 1, 1).expand_as(t0), torch.where(tera > 0, tt, float(t_["normal"])),
                              t0, torch.where(t1 > 0, t1, t0), torch.where(tera > 0, tt, t0)], -1)
        tfid = c("tfid").long()
        mtype = torch.where(tfid > 0, torch.gather(tf_tab, -1, tfid), mtype)
        ate_on = u((a1("ate_p") >= 0.5).float())
        mtype = torch.where(c("ate_ok") * ate_on > 0, u(a1("ate_t")), mtype)
        tcol = torch.where(c("tcol_ovr") > 0, c("tcol_ovr"), mtype).long()
        eff_a = a("eff_a")
        pb = (eff_a[..., 0] > eff_a[..., 2]).float()  # physical attack higher (tera blast, photon geyser)
        cat0 = c("category")
        flip = torch.gather(torch.stack([torch.zeros_like(tera), tera, torch.ones_like(tera)], -1), -1,
                            c("cfid").long())
        cat = cat0 + flip * (2 - u(pb) - cat0)
        rix = (c("row") + (cat - cat0)).long()
        # additive log factor: move columns . attacker weights, + level / items, power function, type factor
        wlin = torch.cat([torch.stack([torch.ones_like(tera), torch.ones_like(tera), torch.ones_like(tera),
                                       aa("skilllink"), a1("loadeddice"), LN(1.5) * aa("technician"),
                                       LN(1.2 if gen >= 7 else 1.3) * (a1("ate_p") >= 0.5).float()], -1),
                          torch.log1p(a("attrb"))], -1)  # [B,G,NA,17]
        wl = wlin.permute(3, 0, 1, 2)[..., None]
        LA = (Pt[L0:] * wl).sum(0)
        # the "+2" of the base damage takes the final modifiers only (STAB, Life Orb, hits, crit, -ate, and in the
        # type tables effectiveness, Expert Belt, Filter, berries ...), not the power / stat ones
        LF = (Pt[L0 + self.fin_i] * wl[self.fin_i]).sum(0)
        hpr = a1("hpr")
        pf = torch.stack([torch.ones_like(hpr), hpr.clamp(min=1 / 150), self.flail_v[torch.bucketize(
            torch.floor(48 * hpr), self.flail_b)], 20 + 20 * a1("pos"), 50 + 50 * a1("nf_own"), 1 + a1("noitem"),
            1 + a1("st3"), (1 + Fv[:, FC["any"]]).view(B, 1, 1).expand_as(hpr),
            (1 + 0.5 * Fv[:, FC["eterr"]]).view(B, 1, 1).expand_as(hpr),
            (1 - 0.5 * Fv[:, FC["weak"]]).view(B, 1, 1).expand_as(hpr),
            1 + 0.25 * tera * (tt == t_["stellar"]).float()], -1)
        own = a("t1h") * self.no_typeless
        p_ad = u(aa("adaptability"))
        stab = 1 + own * (0.5 + 0.5 * p_ad)
        if gen >= 9:
            tm = (u(tt).long() == torch.arange(20, device=dev)).float() * self.no_typeless
            stel = u((tt == t_["stellar"]).float() * tera)
            st_tera = 1 + 0.5 * own + tm * (0.5 + 0.5 * p_ad - 0.25 * p_ad * own)
            stab = stab + u(tera - stel[..., 0]) * (st_tera - stab) + stel * (1.2 + 0.8 * own - stab)
        # [B,G,NA,2,22]: log STAB (a final modifier) | log type-dependent power modifiers (weather, type items /
        # abilities, orbs, Flash Fire, Charge, terrains); column 20 = freeze-dry (ice), 21 = no damage
        lw = lweather.view(B, 1, 1, 20)  # weather: before the "+2" in gen 4, a final modifier later
        lt = torch.stack([torch.log(stab) + (lw if gen >= 5 else 0.0),
                          a("latt") + torch.log1p(0.2 * u(a1("stab_item")) * own) + (lw if gen <= 4 else 0.0)], -2)
        lt = torch.cat([lt, lt[..., t_["ice"]:t_["ice"] + 1], torch.zeros_like(lt[..., :1])], -1)
        fin = u(torch.log(a1("allmoves"))) + torch.gather(lt[..., 0, :], -1, tcol)
        rest = fin + torch.gather(lt[..., 1, :], -1, tcol) + u(torch.log(torch.floor(2 * a1("level") / 5) + 2)) \
            + torch.gather(torch.log(pf), -1, c("pfid").long())
        if gen >= 9:
            rest = rest + c("tera60") * ((mtype == u(tt)) & u(tera > 0)).float()
        LA = LA + rest * c("nf")
        LF = LF + fin * c("nf")
        # accuracy (always hits = 10, clamped after the stage factor)
        aw = Fn.embedding(c("acc_kind").long() + 3 * torch.arange(B, device=dev).view(B, 1, 1, 1), accw.view(-1, 2))
        acc = torch.addcmul(aw[..., 1], c("acc"), aw[..., 0])
        am2 = torch.stack([a1("acc_mult"), a1("acc_mult") * (1 - 0.2 * aa("hustle"))], -1)
        acc = acc * torch.gather(am2, -1, (cat == 1).long())
        acc = torch.maximum(acc, u(10 * (aa("noguard") >= 0.5).float()))

        # ================= (a, d) tables [B,G,NA,ND]
        rows = self.n_rows
        nmb = 1 - aa("moldbreaker")  # [B,G,NA]
        un_d = (dpair(ad_("unaware")) * u(nmb))[..., None]  # [B,G,NA,ND,1]
        un_a = u(u(aa("unaware")))
        # stats with boosts, Unaware on the other side mixing in the unboosted stat: S bm + un (S - S bm)
        S5a, E5a = a("S5"), a("eff_a")
        P1 = E5a[..., self.num_i]
        num = P1[:, :, :, None, :] + un_d * (S5a[..., self.num_i] - P1)[:, :, :, None, :]  # [B,G,NA,ND,5]
        S5d, E5d = d("S5"), d("S5") * d("bm5")
        Q1 = E5d[..., self.den_i]
        den = (Q1[:, :, None] + un_a * (S5d[..., self.den_i] - Q1)[:, :, None]).clamp_(min=1)
        bmd0 = d("bm5")[..., 0][:, :, None, :]
        num[..., 4] = dpair(d1("atk_raw")) * (bmd0 + un_a[..., 0] * (1 - bmd0)) * u(1 - 0.5 * a1("brn"))  # foul play
        inf = aa("infiltrator") if gen >= 5 else torch.zeros_like(nmb)
        veil = screens[..., 2]
        scr2 = (screens[..., :2] + u(veil)).clamp(max=1)  # [B,G,2] phys / spec screens
        mcat = (1 - 0.5 * scr2[:, :, None, :] * u(1 - inf))[:, :, :, None, :] / 50  # [B,G,NA,1,2]
        ice = 1 - 0.5 * dpair(ad_("icescales")) * u(nmb)
        mcat = torch.stack([mcat[..., 0].expand(B, G, NA, ND), mcat[..., 1] * ice], -1)
        R5 = num / den * mcat[..., self.mcat_i]  # [B,G,NA,ND,5]
        wr = u(a1("weight")) / dpair(d1("weight")).clamp(min=0.1)
        dk = torch.cat([u(self.hs_v[torch.bucketize(wr, self.hs_b, right=True)]),
                        d("dkd")[:, :, None].expand(B, G, NA, ND, 4)], -1)  # [B,G,NA,ND,5]
        cdk = dk[..., self.combo_dk]
        lvl_a = u(a1("level")).expand(B, G, NA, ND)
        hp_d = dpair(d1("hp")).expand(B, G, NA, ND)
        hp_a = u(a1("hp"))
        fx = torch.stack([lvl_a, torch.floor(hp_d / 2).clamp(min=1), torch.ones_like(hp_d), (hp_d - hp_a).clamp(min=0),
                          hp_d], -1)
        X = torch.cat([R5, R5[..., self.combo_r] * cdk, fx], -1).transpose(-1, -2)  # [B,G,NA,rows,ND]
        hpos = dpair((d1("hp") > 0).float())
        single = (1 - (dpair(d1("full") * d1("sash")) + dpair(d1("full") * ad_("sturdy")) * u(nmb)).clamp(max=1)) * hpos
        ohko = (1 - dpair(ad_("sturdy_ohko")) * u(nmb)) * hpos
        PM = torch.stack([single, hpos.expand(B, G, NA, ND), ohko], -2)  # [B,G,NA,3,ND]

        # ================= per-type log tables: sum_c alpha_c(a) beta_c(d, t) -> [B,G,NA,22,ND]
        dty = d("types").long()
        dtera = (d1("tera_flag") > 0) & (d1("tera_t") > 0) & (d1("tera_t") != t_["stellar"])
        d0 = torch.where(dtera, d1("tera_t").long(), dty[..., 0])
        dd1 = torch.where(dtera, torch.zeros_like(d0), dty[..., 1])
        mod20 = (1 + u(d1("tera_flag")) * self.e_stellar20 - u(d1("magnetrise")) * self.e_ground20).repeat(1, 1, 1, 2)
        ee = self.chart2[d0] * self.chart2[dd1] * mod20  # [B,G,ND,40] normal | scrappy
        water = u(((d0 == t_["water"]) | (dd1 == t_["water"])).float())
        fdz = (ee.view(B, G, ND, 2, 20) @ self.e_ice_fd[:, :1])[..., 0] * (1 + 3 * water)  # [B,G,ND,2]
        z1 = torch.ones_like(water)
        en = torch.cat([ee[..., :20], fdz[..., :1], z1], -1)  # [B,G,ND,22]
        es = torch.cat([ee[..., 20:], fdz[..., 1:], z1], -1)
        effc = torch.where(en == 0, es, en)
        se, nve = (effc > 1).float(), (effc < 1).float()
        ext = lambda x: torch.cat([x, x[..., t_["ice"]:t_["ice"] + 1], torch.zeros_like(x[..., :1])], -1)  # noqa: E731
        dv = ext(d("dt")) - 0.25 * u(ad_("filter")) * se - u(0.5 * ad_("multiscale") * d1("full")) \
            - u(0.5 * Fv[:, FC["mterr"]].view(B, 1, 1) * d1("grounded")) * self.e_dragon
        imm_ab = (ext(d("imm")) + u(ad_("wonderguard")) * (1 - se) + d("disg")).clamp(max=1)
        lnab = _log(1 - imm_ab)
        defab = _log((1 + dv - imm_ab).clamp(min=1e-6)) - lnab.clamp(min=-13.8)
        def_it = 1 - 0.5 * ext(d("berry")) * torch.maximum(se, self.e_normal)
        bF = torch.stack([_log(effc * def_it * (1 + d("glaive"))), defab, LN(1.2) * se, LN(2.0) * nve], 2)
        lN0 = NEG * (es == 0).float() + _log(1 - d("balloon") * self.e_ground)
        bN = torch.stack([lN0, lnab, ((en == 0) & (es != 0)).float()], 2)  # [B,G,3,ND,22]
        one = torch.ones_like(nmb)
        aF = torch.stack([one, nmb, a1("expertbelt"), aa("tintedlens")], -1)  # [B,G,NA,4]
        aN = torch.stack([one, nmb, _log(aa("scrappy"))], -1)
        tF = torch.matmul(aF, bF.transpose(-1, -2).reshape(B, G, 4, NTV * ND))
        tN = torch.matmul(aN, bN.transpose(-1, -2).reshape(B, G, 3, NTV * ND))

        # ================= full size, layout [B,G,ND,NA,NM] (long contiguous inner dims for the broadcasts)
        am = lambda x: x[:, :, None]  # a-m [B,G,NA,NM] -> [B,G,1,NA,NM]  # noqa: E731
        dl = lambda x: x[:, :, :, None, None]  # defender [B,G,ND] -> [B,G,ND,1,1]  # noqa: E731
        bga = torch.arange(B * G * NA, device=dev).view(B, G, NA, 1)
        emb = lambda ix, tab, k: Fn.embedding(ix + bga * k, tab.reshape(-1, ND)).permute(0, 1, 4, 2, 3)  # noqa: E731
        tci = tcol
        L4 = torch.stack([emb(rix, X, rows), emb(c("pkcls").long(), PM, 3), emb(tci, tF, NTV), emb(tci, tN, NTV)], 2)
        Xg, PMg, lF = L4[:, :, 0], L4[:, :, 1], L4[:, :, 2]
        lFN = L4[:, :, 2:]  # [B,G,2,ND,NA,NM] log F | log N
        # defender move-flag factors (log): Fluffy (contact), Punk Rock (sound); Soundproof, Bulletproof, Wind Rider
        zd = torch.zeros_like(ad_("fluffy"))
        wFN = torch.stack([torch.stack([_log(1 - 0.5 * ad_("fluffy")), _log(1 - 0.5 * ad_("punkrock")), zd, zd], -1),
                           torch.stack([zd, _log(1 - ad_("soundproof")), _log(1 - ad_("bulletproof")),
                                        _log(1 - ad_("windrider"))], -1)], 2).view(B, G, 2 * ND, 4)
        anm = (Pt[[MP["contact"], MP["sound"], MP["bullet"], MP["wind"]]] * nmb[None, ..., None]
               ).permute(1, 2, 0, 3, 4).reshape(B, G, 4, NA * NM)
        lFN.add_(torch.matmul(wFN, anm).view(B, G, 2, ND, NA, NM))
        nf = c("nf")
        lF.mul_(am(nf))
        Fm, Nm = lFN.clamp_(min=NEG).exp_().unbind(2)
        valid = c("valid") * u(a1("alive"))  # exact zeros: invalid / status moves, absent or fainted attackers
        dmax = torch.addcmul(am(2 * nf * valid * torch.exp(LF)), am(torch.exp(LA) * valid), Xg).mul_(Fm)
        mean = (dmax * Nm).mul_(am(c("roll")))
        hp100 = dl((100 * d1("hp")).clamp(min=1e-3))
        inv = am(c("inv"))
        pko = torch.maximum(torch.div(hp100, dmax).ceil_(), am(c("lo"))).mul_(-inv).add_(101 * inv).clamp_(0, 1)
        pko.mul_(PMg).mul_(Nm)
        alive_d = d1("alive")
        f0 = (mean * dl(alive_d / d1("maxhp"))).clamp_(max=4)
        f1 = mean.mul_(dl(torch.where(d1("hp") > 0, alive_d / d1("hp").clamp(min=1e-3), 0.0))).clamp_(max=2)
        stage = self.acc_stage_tab[(u(a1("acc_st")) - dpair(d1("eva_st"))).clamp(-6, 6).long() + 6]
        ngd = dpair((ad_("noguard") >= 0.5).float())
        stage = torch.addcmul(stage, ngd, 10 - stage)  # [B,G,NA,ND]
        f3 = (am(acc) * stage.transpose(2, 3)[..., None]).clamp_(max=1).mul_(f0)
        return f0, f1, pko, f3  # each [B,G,ND,NA,NM]
