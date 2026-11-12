"""Which moves / abilities / items / species of the random-battle teams poke-engine does not know or (probably) does
not implement, weighted by how often they appear in generated teams.

Two kinds of evidence:
  unknown   the name does not survive a round trip through the engine (its enums fall back to NONE silently: an
            unknown move does nothing, an unknown ability / item has no effect). Probed with the installed engine
            build (PORYGON_GEN / PYTHONPATH pick the gen), using porygon.selfplay.engine_mon as engine self-play does.
  no code   the engine source never refers to it (Abilities::X / Items::X anywhere, Choices::X outside the move table)
            while Showdown implements it with handlers (showdown_effects.js); for moves additionally only when the
            engine's move-table entry carries no effect field (secondaries, boosts, status, volatile, side condition,
            heal, drain, recoil...). Heuristic: "no code" items are candidates, read the source before trusting one.
Battle-irrelevant handlers in a full-information two-player sim (Frisk, Run Away, ...) are listed in IRRELEVANT.

    PORYGON_GEN=9 PYTHONPATH=eng9:src python coverage.py --pairs pairs.jsonl --effects effects9.json \
        --engine-src .poke-engine [--json out.json]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import poke_engine as pe

from porygon.inference import norm
from porygon.selfplay import engine_mon, engine_move

# no effect in a full-information singles simulation (information-only, overworld or doubles-only)
IRRELEVANT = {"frisk", "runaway", "pickup", "honeygather", "illuminate", "ballfetch", "anticipation", "forewarn",
              "keeneye", "minus", "plus", "telepathy", "friendguard", "healer", "stench", "symbiosis", "curiousmedicine",
              "battlebond", "receiver", "powerofalchemy", "pickpocket", "none", "noability", "mummy", "cudchew",
              "zerotohero", "commander", "illusion"}
IRRELEVANT_DOC = "illusion / zero to hero / commander are hidden-info or doubles mechanics listed for completeness"

# switch-in handlers that do nothing in a full-information sim (announcements)
START_IRRELEVANT = {"pressure", "frisk", "anticipation", "forewarn", "moldbreaker", "teravolt", "turboblaze",
                    "illusion", "unnerve", "airlock", "cloudnine", "asone", "asoneglastrier", "asonespectrier"}

EFFECT_FIELDS = ("secondaries", "boost", "heal", "status", "volatile_status", "side_condition", "drain", "recoil",
                 "crash", "target", "drag", "pivot", "charge", "recharge")  # incl. Flags {pivot, drag, charge, recharge}
INSERT = re.compile(r"moves\.insert\(\s*Choices::([A-Z0-9_]+),\s*Choice \{(.*?)\n\s*\},\s*\);", re.S)


def engine_tables(src: Path) -> dict:
    files = {p: p.read_text() for p in list((src / "src").glob("*.rs")) + list((src / "src" / "genx").glob("*.rs"))}
    choices = (src / "src" / "choices.rs").read_text()
    # Choices::X inside the move table is data; elsewhere in choices.rs (multi_hit(), guaranteed_crit()...) it is code
    files[src / "src" / "choices.rs"] = INSERT.sub("", choices)
    refs = {"Abilities": Counter(), "Items": Counter(), "Choices": Counter()}
    for p, text in files.items():
        for kind in refs:
            for m in re.finditer(rf"\b{kind}::([A-Z0-9_]+)\b", text):
                refs[kind][m.group(1)] += 1
    # move-table entries: moves.insert(Choices::X, Choice { ... });  (several per move when gen-gated)
    entries: dict[str, set[str]] = {}
    for m in INSERT.finditer(choices):
        body = m.group(2)
        fields = {f for f in EFFECT_FIELDS if re.search(rf"\b{f}: (?!false)", body)}
        entries.setdefault(m.group(1), set()).update(fields)
    assert len(entries) > 500, len(entries)
    return {"refs": refs, "entries": entries}


def probe(mon: dict):
    """(species, ability, item, [moves]) as the engine reads them back (NONE for unknown names)."""
    p = engine_mon(mon)
    s = pe.State(side_one=pe.Side(pokemon=[p] * 6), side_two=pe.Side(pokemon=[p] * 6))
    q = pe.State.from_string(s.to_string()).side_one.pokemon[0]
    return q.id.lower(), q.ability.lower(), q.item.lower(), [m.id.lower() for m in q.moves]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--effects", required=True)
    ap.add_argument("--engine-src", required=True)
    ap.add_argument("--json")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    eff = json.load(open(a.effects))
    eng = engine_tables(Path(a.engine_src))
    pairs = [json.loads(l) for l in open(a.pairs) if l.strip()]
    n_teams, n_games = 2 * len(pairs), len(pairs)

    team_has = {k: Counter() for k in ("moves", "abilities", "items", "species")}
    game_has = {k: Counter() for k in team_has}
    unknown = {k: set() for k in team_has}
    seen_probe: dict[str, tuple] = {}
    for pair in pairs:
        in_game = {k: set() for k in team_has}
        for team in pair:
            in_team = {k: set() for k in team_has}
            for mon in team:
                key = json.dumps(mon, sort_keys=True)
                if key not in seen_probe:
                    seen_probe[key] = probe(mon)
                sp, ab, it, mv = seen_probe[key]
                if sp == "none":
                    unknown["species"].add(mon["speciesId"])
                if ab == "none" and norm(mon["ability"]) not in ("none", "noability"):
                    unknown["abilities"].add(norm(mon["ability"]))
                if it == "none" and norm(mon["item"]) not in ("", "none"):
                    unknown["items"].add(norm(mon["item"]))
                for m, back in zip(mon["moves"], mv):
                    if back == "none":
                        unknown["moves"].add(norm(m))
                in_team["species"].add(mon["speciesId"])
                in_team["abilities"].add(norm(mon["ability"]))
                in_team["items"].add(norm(mon["item"]))
                in_team["moves"].update(norm(m) for m in mon["moves"])
            for k in team_has:
                team_has[k].update(in_team[k])
                in_game[k] |= in_team[k]
        for k in team_has:
            game_has[k].update(in_game[k])

    def row(kind, x, why):
        return {"kind": kind, "id": x, "why": why, "teams": team_has[kind][x] / n_teams,
                "games": game_has[kind][x] / n_games, "showdown": eff[kind].get(x, {}).get("keys", [])}

    rows = []
    for kind in team_has:
        for x in sorted(unknown[kind]):
            rows.append(row(kind, x, "unknown"))
    up = lambda x: x.upper()
    for x in team_has["abilities"]:
        if x in unknown["abilities"] or x in IRRELEVANT:
            continue
        if eff["abilities"].get(x, {}).get("keys") and eng["refs"]["Abilities"][up(x)] == 0:
            rows.append(row("abilities", x, "no code"))
    for x in team_has["items"]:
        if x in unknown["items"] or not x:
            continue
        keys = [k for k in eff["items"].get(x, {}).get("keys", []) if k not in ("fling", "naturalGift", "isBerry")]
        if keys and eng["refs"]["Items"][up(x)] == 0:
            rows.append(row("items", x, "no code"))
    for x in team_has["moves"]:
        if x in unknown["moves"]:
            continue
        e = engine_move(x).upper()
        keys = [k for k in eff["moves"].get(x, {}).get("keys", []) if k not in ("priority", "critRatio")]
        if keys and eng["refs"]["Choices"][e] == 0 and not eng["entries"].get(e):
            rows.append(row("moves", x, "no code"))

    rows.sort(key=lambda r: -r["games"])
    hard = [r for r in rows if r["why"] == "unknown"]
    any_unknown = sum(1 for pair in pairs if any(
        norm(m) in unknown["moves"] or norm(mon["ability"]) in unknown["abilities"]
        or norm(mon["item"]) in unknown["items"] or mon["speciesId"] in unknown["species"]
        for team in pair for mon in team for m in mon["moves"]))
    flagged = {(r["kind"], r["id"]) for r in rows}
    any_flag = sum(1 for pair in pairs if any(
        ("moves", norm(m)) in flagged or ("abilities", norm(mon["ability"])) in flagged
        or ("items", norm(mon["item"])) in flagged or ("species", mon["speciesId"]) in flagged
        for team in pair for mon in team for m in mon["moves"]))
    # engine self-play starts with both leads already in, so lead entry effects (Intimidate, weather / terrain
    # setters, Download, Slow Start, Imposter, Booster Energy...) never happen there
    def entry(mon):
        ab = norm(mon["ability"])
        keys = eff["abilities"].get(ab, {}).get("keys", [])
        return ((("onStart" in keys or "onSwitchIn" in keys) and ab not in START_IRRELEVANT)
                or norm(mon["item"]) == "boosterenergy")
    lead_entry = sum(1 for pair in pairs if entry(pair[0][0]) or entry(pair[1][0])) / n_games
    summary = {"pairs": n_games, "games_with_lead_entry_effect": lead_entry, "distinct": {k: len(v) for k, v in team_has.items()},
               "unknown": {k: len(v) for k, v in unknown.items()},
               "games_with_unknown": any_unknown / n_games, "games_with_any_flag": any_flag / n_games}
    print(json.dumps(summary))
    print(f"{'kind':<10} {'id':<22} {'why':<8} {'teams':>6} {'games':>6}  showdown keys")
    for r in rows[:a.top] + [r for r in hard if r not in rows[:a.top]]:
        print(f"{r['kind']:<10} {r['id']:<22} {r['why']:<8} {r['teams']:6.1%} {r['games']:6.1%}  "
              f"{','.join(r['showdown'])[:60]}")
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
