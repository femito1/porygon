"""Opponent set inference: the library posterior and the evidence read from synthetic battle logs."""

from porygon import inference

START = """|player|p1|me|
|player|p2|you|
|switch|p1a: Blissey|Blissey, L84, F|300/300
|switch|p2a: Heracross|Heracross, L79, M|100/100
|turn|1
|"""


def log_of(text: str) -> list[list[str]]:
    return [line.split("|") for line in (START + "\n" + text).splitlines()]


def ev_of(text: str, ours: dict | None = None) -> inference.Evidence:
    ev = inference.evidence(log_of(text), "p2")
    ev.races = inference.races(log_of(text), "p2", ours or {})
    return ev


def items(ev: inference.Evidence, revealed: tuple[str, ...] = ()) -> set[str]:
    class Mon:  # what posterior reads of a poke-env Pokemon
        moves, ability, item, level = dict.fromkeys(revealed), None, inference.GEN.UNKNOWN_ITEM, 79
    return {s.item for s in inference.posterior("heracross", Mon(), ev, "p2: Heracross")[1]}


def test_library_covers_gen4_and_counts_leads():
    lib = inference.library()
    assert len(lib) == 295 and lib["heracross"][0] == 79
    assert all(s.n >= s.n_lead >= 0 for _, sets in lib.values() for s in sets)
    assert {s.item for s in lib["heracross"][1]} == {"toxicorb", "lifeorb", "choicescarf", "choiceband"}


def test_revealed_moves_and_unlocked():
    assert items(ev_of("|turn|2"), ("facade",)) == {"toxicorb"}
    two_moves = """|move|p2a: Heracross|Close Combat|p1a: Blissey
|-damage|p1a: Blissey|200/300
|-damage|p2a: Heracross|90/100|[from] item: Life Orb
|
|upkeep
|turn|2
|
|move|p2a: Heracross|Megahorn|p1a: Blissey
|-damage|p1a: Blissey|150/300
|-damage|p2a: Heracross|80/100|[from] item: Life Orb
|
|upkeep
|turn|3"""
    ev = ev_of(two_moves)
    assert "p2: Heracross" in ev.unlocked and ev.item["p2: Heracross"] == "lifeorb"
    assert items(ev) == {"lifeorb"}


def test_no_lifeorb_after_a_hit_without_recoil():
    hit = """|move|p2a: Heracross|Close Combat|p1a: Blissey
|-damage|p1a: Blissey|200/300
|
|upkeep
|turn|2"""
    assert "p2: Heracross" in ev_of(hit).no_lifeorb
    assert "lifeorb" not in items(ev_of(hit))
    # gen4 shows the recoil after the faint
    ko = hit.replace("|-damage|p1a: Blissey|200/300", "|-damage|p1a: Blissey|0 fnt\n|faint|p1a: Blissey\n"
                     "|-damage|p2a: Heracross|90/100|[from] item: Life Orb")
    assert "p2: Heracross" not in ev_of(ko).no_lifeorb
    assert "p2: Heracross" not in ev_of(hit.replace("Close Combat", "U-turn")).no_lifeorb


def test_no_leftovers_unless_healed_or_full():
    hurt = """|move|p1a: Blissey|Seismic Toss|p2a: Heracross
|-damage|p2a: Heracross|70/100
|
|upkeep
|turn|2"""
    assert "p2: Heracross" in ev_of(hurt).no_leftovers
    wish = hurt.replace("|\n|upkeep", "|\n|-heal|p2a: Heracross|100/100|[from] move: Wish|[wisher] Heracross\n|upkeep")
    assert "p2: Heracross" not in ev_of(wish).no_leftovers


def test_trick_reveals_the_original_item():
    for order in ((0, 1), (1, 0)):  # Showdown sends the two -item messages in either order
        msgs = ["|-item|p2a: Heracross|Leftovers|[from] move: Trick", "|-item|p1a: Blissey|Choice Scarf|[from] move: Trick"]
        ev = ev_of("|move|p2a: Heracross|Trick|p1a: Blissey\n" + "\n".join(msgs[i] for i in order) + "\n|\n|upkeep\n|turn|2")
        assert ev.item["p2: Heracross"] == "choicescarf" and "p2: Heracross" in ev.changed


def test_speed_order_finds_a_choice_scarf():
    race = """|move|p2a: Heracross|Close Combat|p1a: Blissey
|-damage|p1a: Blissey|200/300
|-damage|p2a: Heracross|90/100|[from] item: Life Orb
|move|p1a: Blissey|Seismic Toss|p2a: Heracross
|-damage|p2a: Heracross|60/100
|
|upkeep
|turn|2"""
    ours = {"p1: Blissey": (200, "naturalcure", "leftovers")}  # Heracross: 180, 270 with a scarf
    race = race.replace("|-damage|p2a: Heracross|90/100|[from] item: Life Orb\n", "")
    ev = ev_of(race, ours)
    assert ev.races["p2: Heracross"][0].first
    assert items(ev) == {"choicescarf"}
    # paralyzed (x0.25) it cannot have been first: no set explains it, so the speed order is dropped
    paralyzed = ev_of("|-status|p2a: Heracross|par\n|\n|upkeep\n|turn|2\n|\n" + race.replace("turn|2", "turn|3"), ours)
    assert paralyzed.races["p2: Heracross"][0].paralyzed
    assert items(paralyzed) == {"toxicorb", "choicescarf", "choiceband"}  # no recoil: no life orb
