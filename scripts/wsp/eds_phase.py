"""eds step 0: ground truth for "Sleep Talk when Rest is about to wear off", from the Showdown replays.

For every logged decision whose active mon is asleep (status SLP in the observation), parse the game's replay to get
  - the sleep's origin (Rest vs another move / ability), from the last '-status|<mon>|slp' line;
  - `slept`: turns the mon has tried to act while asleep since it fell asleep (one per turn: 'cant slp' or a move
    while asleep), i.e. the engine's notion (porygon.search.History);
  - whether it actually woke this turn ('-curestatus|<mon>|slp' during the decision's turn).
Gen4 Rest: wakes at its 3rd move attempt (slept == 2). Writes out/eds/phase.npz (row-aligned) + phase.txt.
    python scripts/wsp/eds_phase.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from beh_common import GLORIA, NUM_OFF, load_local, ratio_boot
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB

OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds"); OUT.mkdir(parents=True, exist_ok=True)
MOVES = ["<pad>", "<unk>"] + VOCAB["moves"]
d = load_local()
N = len(d["mask"]); r = np.arange(N)
mons = d["obs"][:, FIELD_DIM:].reshape(N, 12, MON_DIM)
nums = mons[..., NUM_OFF:NUM_OFF + 125].astype(np.float32)
own_a = nums[:, :6, 115].argmax(1)
A = nums[r, own_a]
slp = (A[:, 63] > .5) & (nums[:, :6, 115].max(1) > .5)
cnt = np.round(A[:, 65] * 10).astype(int)
mids = mons[r, own_a, 3:7].astype(np.int64)
names = np.array(MOVES)[mids]
st_slot = (names == "sleeptalk")
st_legal = st_slot & (d["mask"][:, 6:] > 0)
has_rest = (names == "rest").any(1)
has_st = st_slot.any(1)
games = d["games"]

origin = np.full(N, "", dtype=object); slept_a = np.full(N, -1); woke = np.full(N, -1); acted = np.full(N, -1)
cache = {}


def parse(g):
    """Per replay: for our side, list of events per turn for each mon key."""
    gm = games[g]
    lines = Path(GLORIA, "runs", gm["run"], "replays", gm["tag"] + ".log").read_text().splitlines()
    side = None
    for ln in lines:
        p = ln.split("|")
        if len(p) > 3 and p[1] == "player" and p[3] == gm["player"]:
            side = p[2]
    ev = [ln.split("|") for ln in lines]
    return side, ev


rows = np.flatnonzero(slp)
for i in rows:
    g = int(d["game"][i]); t = int(d["turn"][i])
    if g not in cache:
        cache[g] = parse(g)
    side, ev = cache[g]
    # walk the log up to the end of turn t
    turn = 0; active = None; state = {}  # key -> [origin, slept, slept_turns set]
    woke_t = acted_t = 0; act_at_t = None
    for p in ev:
        if len(p) < 3:
            continue
        kind = p[1]
        if kind == "turn":
            turn = int(p[2])
            if turn > t:
                break
            if turn == t:
                act_at_t = active
            continue
        key = re.sub(r"^(p\d)[a-c]:", r"\1:", p[2])
        if not key.startswith(side):
            continue
        if kind in ("switch", "drag"):
            active = key
        elif kind == "-status" and len(p) > 3 and p[3] == "slp" and turn < t:  # a re-Rest during turn t is after the decision
            state[key] = ["rest" if p[-1].endswith("move: Rest") else ("other:" + p[-1] if len(p) > 4 else "other"), 0, set()]
        elif kind in ("cant", "move") and key in state and (kind == "move" or (len(p) > 3 and p[3] == "slp")) and not (turn == t and woke_t):
            if turn not in state[key][2]:
                state[key][2].add(turn); state[key][1] += 1
                if turn == t and key == act_at_t:
                    acted_t = 1
        elif kind in ("-curestatus", "faint") and key in state:
            if turn == t and key == act_at_t and kind == "-curestatus":
                woke_t = 1; acted_t = 1
            if turn < t:
                state.pop(key)
            elif kind == "faint":
                state.pop(key, None)
    k = act_at_t
    if k in state:
        o, s, ts = state[k]
        origin[i] = o
        slept_a[i] = s - (1 if t in ts else 0)  # attempts BEFORE this turn
        woke[i] = woke_t; acted[i] = acted_t
    else:
        origin[i] = "?"

np.savez(OUT / "phase.npz", slp=slp, cnt=cnt, st_legal=st_legal, has_rest=has_rest, has_st=has_st,
         origin=origin.astype(str), slept=slept_a, woke=woke, acted=acted)
txt = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


P(f"asleep decisions: {slp.sum()}; with Sleep Talk legal {np.sum(slp & st_legal.any(1))}; with Rest in moveset {np.sum(slp & has_rest)}")
P("origin counts (all asleep):", Counter(o.split(':')[0] for o in origin[slp]))
P("origin counts (asleep, Rest in moveset):", Counter(o.split(':')[0] for o in origin[slp & has_rest]))
P("non-Rest origins with Rest in moveset:", Counter(origin[slp & has_rest & (origin != 'rest')]).most_common(10))
P("Sleep Talk in moveset without Rest (all rows):", int(np.sum(has_st & ~has_rest)))
P("\nRest sleep, mon tried to act this turn: woke by slept-before (attempts before this turn)")
m = slp & (origin == "rest") & (acted == 1)
for s in range(0, 5):
    q = m & (slept_a == s)
    if q.sum():
        P(f"  slept {s}: n {q.sum()}, woke {woke[q].mean():.3f}")
lab = np.load("/vol/bitbucket/fr1226/porygon-ws-wsp/out/beh/derived_r2-300k.npz", allow_pickle=True)["lab_q"]
qmax = np.nanmax(np.where(np.isfinite(lab), lab, -9), 1)
j = st_legal.argmax(1)
qgap = lab[r, 6 + j] - qmax
pn = (d["probs"][:, 6:] * st_legal).sum(1); ps = (d["shares"][:, 6:] * st_legal).sum(1)
H = slp & st_legal.any(1)
P("\nasleep, Sleep Talk legal: by origin x slept-before x poke-env counter")
P("origin | slept | counter | n | games | net P(ST) | search share | oracle Q(ST)-maxQ [95% CI over games] | P(woke | acted)")
for o in ("rest", "other"):
    for s in range(-1, 5):
        for c in range(0, 11):
            q = H & (np.char.startswith(origin.astype(str), o)) & (slept_a == s) & (cnt == c)
            if q.sum() == 0:
                continue
            e = ratio_boot(d["game"][q], qgap[q])
            ac = q & (acted == 1)
            P(f"{o} | {s} | {c} | {q.sum()} | {len(np.unique(d['game'][q]))} | {pn[q].mean():.3f} | {ps[q].mean():.3f} | "
              f"{e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}] | {woke[ac].mean() if ac.sum() else float('nan'):.2f} ({ac.sum()})")
for o in ("rest",):
    for s in range(0, 4):
        q = H & (origin == o) & (slept_a == s)
        if q.sum():
            e = ratio_boot(d["game"][q], qgap[q])
            P(f"rest, slept {s} (all counters): n {q.sum()}, net {pn[q].mean():.3f}, search {ps[q].mean():.3f}, Q gap {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}]")
(OUT / "phase.txt").write_text("\n".join(txt) + "\n")
