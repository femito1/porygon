"""Showdown battles -> first-person training decisions (observation, legal mask, chosen action).

A first-person view of a battle is the message stream that player's client received: battle updates (with its own
private lines, e.g. exact HP) and `|request|` JSON. Two sources give it exactly:
  - re-simulation of a random battle's input log (`scripts/resim_inputlog.js`: Showdown's own simulator at the
    battle's commit; the input log holds the team seeds and every choice), for human replays;
  - the raw websocket messages logged by our own players (search-label runs: `<run>/{a,b}/streams/<tag>.jsonl.gz`).

`StreamReplayer` feeds such a stream into a poke-env `Battle` exactly as poke-env 0.16.1's
`Player._handle_battle_message` does live: battle lines go to `Battle.parse_message`, `|request|` to
`Battle.parse_request`, `|win|` / `|tie|` end it, and a decision is taken wherever the live player's `choose_move`
would run (a request that is neither `wait` nor team preview, and a re-request after `[Invalid choice]`). At each
decision the battle object is the live one, so obs = encoder(battle), mask = `SinglesEnv.get_action_mask(battle)`
are what a network playing live would have seen; the choice that answered the request (`move 2 terastallize`,
`switch 3`, `default`) is mapped to poke-env's action index via that request (`choice_action`).

Records (`scripts/convert_replays.py`) keep the stream itself, so any encoder can re-encode them later.
"""

import json
import logging
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from poke_env.battle import Battle
from poke_env.environment import SinglesEnv
from poke_env.player import Player

PLAYER_IGNORE = Player.MESSAGES_TO_IGNORE  # what Player._handle_battle_message drops before the battle sees it
TERA_OFFSET = 22  # poke-env action index of "move 1 + terastallize"
LOG = logging.getLogger("porygon.replay")
LOG.setLevel(logging.ERROR)  # poke-env warns about illusions etc. on every battle


def to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def seed_of(battle_id: str) -> int:
    """Deterministic per-game seed (anything random in a conversion is seeded by the game, not the worker)."""
    return zlib.crc32(battle_id.encode())


# ---------------------------------------------------------------- streams

Message = list[list[str]]  # one websocket message as poke-env splits it: lines, each split on "|"


def split_chunk(chunk: str) -> Message:
    """A raw message (resim chunk, websocket text) -> poke-env's split form, without the room line."""
    lines = chunk.split("\n")
    if lines and lines[0].startswith(">"):
        lines = lines[1:]
    return [line.split("|") for line in lines]


def normalize_stream(stream: Iterable[str | Message]) -> list[Message]:
    """Accept raw chunks (strings) or already split messages (whose first line may be the `>battle-...` room line)."""
    out = []
    for m in stream:
        if isinstance(m, str):
            out.append(split_chunk(m))
        else:
            out.append(m[1:] if m and m[0] and m[0][0].startswith(">") else m)
    return out


# ---------------------------------------------------------------- decisions

@dataclass
class Decision:
    msg: int  # index of the message (request) after which the decision is taken
    turn: int
    kind: str  # "turn" (a move request) or "forced" (forceSwitch: after a faint, U-turn, Eject Button, ...)
    request: dict[str, Any]
    mask: np.ndarray  # int8 [n_actions]
    obs: np.ndarray | None  # encoder(battle); None if no encoder
    retry: bool = False  # a re-decision after an [Invalid choice] error
    choice: str | None = None  # what the player sent
    action: int = -1  # poke-env action index of `choice`, -1 unknown
    desc: dict[str, Any] | None = None  # {"type": "move"|"switch", "move"|"species", "tera"}
    legal: bool | None = None  # mask[action] == 1 (None: no action)
    default: bool = False  # the choice was Showdown's "default" (timer / auto-choice)
    search: dict[str, Any] | None = None  # {"shares", "value"} when a logged search took the decision

    @property
    def n_legal(self) -> int:
        return int(self.mask.sum())


class StreamReplayer:
    """One player's view of one battle, replayed into a poke-env Battle like `Player._handle_battle_message`.

    `feed(message)` returns the decision that message triggers (or None). `encoder(battle) -> np.ndarray`
    (None: no observation). `gen` must match the battle (`|gen|` raises otherwise, as live)."""

    def __init__(self, username: str, gen: int, battle_tag: str = "battle-replay",
                 encoder: Callable[[Battle], np.ndarray] | None = None):
        self.battle = Battle(battle_tag, username, LOG, gen=gen, save_replays=False)
        self.encoder = encoder
        self.n_actions = SinglesEnv.get_action_space_size(gen)
        self.n_messages = 0
        self.finished = False

    def feed(self, message: Message) -> Decision | None:
        battle, decision = self.battle, None
        idx = self.n_messages
        self.n_messages += 1
        for split in message:
            if not split:
                continue
            if len(split) == 1:  # plain text lines (open team sheets only)
                continue
            kind = split[1]
            if kind == "":
                battle.parse_message(split)
            elif kind in PLAYER_IGNORE:
                pass
            elif kind == "request":
                if split[2]:
                    request = json.loads(split[2])
                    battle.parse_request(request)
                    decision = self._decide(idx, request) or decision
            elif kind in ("win", "tie"):
                if kind == "win":
                    battle.won_by(split[2])
                else:
                    battle.tied()
                self.finished = True
            elif kind == "error":
                if split[2].startswith("[Invalid choice]") and battle._last_request:
                    d = self._decide(idx, battle._last_request)
                    if d is not None:
                        d.retry = True
                        decision = d
            elif kind == "bigerror" or kind == "showteam":
                pass
            else:
                battle.parse_message(split)
        return decision

    def _decide(self, idx: int, request: dict[str, Any]) -> Decision | None:
        battle = self.battle
        if battle._wait or battle.teampreview:
            return None
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.int8)
        obs = self.encoder(battle) if self.encoder is not None else None
        kind = "forced" if request.get("forceSwitch") else "turn"
        return Decision(idx, battle.turn, kind, request, mask, obs)


# ---------------------------------------------------------------- choices -> poke-env actions

EVENTS = (" terastallize", " terastal", " mega", " megax", " megay", " zmove", " ultra", " dynamax", " gigantamax",
          " max")


def _same_move(request_id: str, known_id: str) -> bool:
    """Request ids are bare ('hiddenpower', 'return'); poke-env may keep the typed / powered id."""
    if request_id == known_id:
        return True
    for prefix in ("hiddenpower", "return", "frustration"):
        if request_id == prefix and known_id.startswith(prefix):
            return True
    return known_id.rstrip("0123456789") == request_id.rstrip("0123456789")


def parse_choice(choice: str) -> tuple[str, str, str]:
    """'move 2 terastallize' -> ('move', '2', 'terastallize'); 'switch 3' -> ('switch', '3', '')."""
    choice = choice.strip()
    if "," in choice:  # doubles syntax: singles only has one action
        choice = choice.split(",")[0].strip()
    kind, _, arg = choice.partition(" ")
    event = ""
    arg = arg.strip()
    for e in EVENTS:
        if arg.endswith(e):
            event, arg = e.strip(), arg[: -len(e)].strip()
            break
    if kind == "move" and re.search(r"\s[-+]?[1-3]$", arg) and to_id(arg) != "conversion2":
        arg = arg[:-2].strip()  # a target location (doubles)
    return kind, arg, event


def choice_action(choice: str | None, request: dict[str, Any], battle: Battle) -> tuple[int, dict | None, bool]:
    """(poke-env action index or -1, {"type", "move"/"species", "tera"} or None, was 'default')."""
    if not choice:
        return -1, None, False
    kind, arg, event = parse_choice(choice)
    default = kind == "default"
    side = request.get("side", {}).get("pokemon", [])
    if default:  # Showdown's autoChoose: the first move, or the first healthy bench mon
        if request.get("forceSwitch"):
            n_active = len(request.get("forceSwitch"))
            slot = next((i for i, p in enumerate(side) if i >= n_active and not p["condition"].endswith(" fnt")
                         and p["condition"] != "0 fnt"), None)
            if slot is None:
                return -1, None, True
            kind, arg = "switch", str(slot + 1)
        else:
            kind, arg = "move", "1"
    if kind == "switch":
        try:
            slot = int(arg) - 1
            ident = side[slot]["ident"]
        except (ValueError, IndexError):
            ident = next((p["ident"] for p in side if to_id(p["ident"][4:]) == to_id(arg)
                          or to_id(p["details"].split(",")[0]) == to_id(arg)), None)
            if ident is None:
                return -1, None, default
        key = ident[:2] + ident[3:] if ident[2] != ":" else ident  # 'p1a: X' -> 'p1: X'
        keys = list(battle.team)
        if key not in keys:
            return -1, None, default
        mon = battle.team[key]
        return keys.index(key), {"type": "switch", "species": mon.species}, default
    if kind == "move":
        active = (request.get("active") or [{}])[0]
        moves = active.get("moves", [])
        if arg.isdigit():
            i = int(arg) - 1
            if not 0 <= i < len(moves):
                return -1, None, default
            move_id = moves[i]["id"]
        else:
            move_id = to_id(arg)
            if move_id.startswith("hiddenpower"):
                move_id = "hiddenpower"
            maxed = (active.get("maxMoves") or {}).get("maxMoves", [])
            for i, m in enumerate(maxed):  # a max move named by its max name
                if m.get("move") == move_id and i < len(moves):
                    move_id = moves[i]["id"]
        mon = battle.active_pokemon
        if mon is None:
            return -1, None, default
        avail = [m.id for m in battle.available_moves]
        known = [m.id for m in list(mon.moves.values())[:4]]
        listed = avail if len(avail) == 1 and avail[0] not in known else known  # poke-env's struggle/recharge rule
        pos = next((j for j, k in enumerate(listed) if _same_move(move_id, k)), None)
        if pos is None:
            return -1, None, default
        tera = event == "terastallize"
        action = 6 + pos + (TERA_OFFSET - 6 if tera else 0)
        gimmick = {"mega": 4, "megax": 4, "megay": 4, "zmove": 8, "ultra": 8, "dynamax": 12, "gigantamax": 12,
                   "max": 12}.get(event, 0)
        action += gimmick
        return action, {"type": "move", "move": move_id, "tera": tera}, default
    return -1, None, default


# ---------------------------------------------------------------- whole games

@dataclass
class SideReplay:
    """A replayed side: its decisions and how the game ended for it."""
    side: str
    player: str
    decisions: list[Decision] = field(default_factory=list)
    won: bool | None = None
    finished: bool = False
    errors: list[str] = field(default_factory=list)


def replay_stream(stream: Iterable[str | Message], username: str, gen: int, side: str = "",
                  choices: dict[int, str | None] | None = None,
                  encoder: Callable[[Battle], np.ndarray] | None = None,
                  battle_tag: str = "battle-replay") -> SideReplay:
    """Replay one player's stream; `choices`: message index of a request -> the choice that answered it."""
    rep = StreamReplayer(username, gen, battle_tag, encoder)
    out = SideReplay(side, username)
    for i, msg in enumerate(normalize_stream(stream)):
        d = rep.feed(msg)
        if d is None:
            continue
        if choices is not None and not d.retry:
            d.choice = choices.get(i)
            d.action, d.desc, d.default = choice_action(d.choice, d.request, rep.battle)
            d.legal = None if d.action < 0 else bool(d.action < len(d.mask) and d.mask[d.action])
        out.decisions.append(d)
    out.finished = rep.finished
    out.won = rep.battle.won
    return out


def resim_sides(rec: dict[str, Any], gen: int, encoder: Callable[[Battle], np.ndarray] | None = None
                ) -> dict[str, SideReplay]:
    """Both sides of one `scripts/resim_inputlog.js` output record."""
    out = {}
    for side in ("p1", "p2"):
        stream = rec["streams"][side]
        choices = {c["chunk"]: c["choice"] for c in rec["choices"][side]}
        out[side] = replay_stream(stream, rec["players"][side], gen, side, choices, encoder, f"battle-{rec['id']}")
    return out


def align_logged(decisions: list[Decision], logged: list[dict[str, Any]]) -> Iterator[tuple[Decision, dict]]:
    """Pair our decisions with a live player's logged ones (decisions.jsonl: turn, mask, action, ...), in order:
    each logged record goes to the next decision with the same turn (and the same mask when one matches)."""
    j = 0
    for rec in logged:
        k = j
        while k < len(decisions) and decisions[k].turn < rec["turn"]:
            k += 1
        same = [i for i in range(k, len(decisions)) if decisions[i].turn == rec["turn"]]
        if not same:
            continue
        mask = rec.get("mask")
        hit = next((i for i in same if mask is not None and decisions[i].mask.tolist() == list(mask)), same[0])
        yield decisions[hit], rec
        j = hit + 1
