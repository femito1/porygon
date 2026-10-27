"""poke-engine search worker processes (used by `porygon.search`).

poke-engine holds the GIL while it searches, so parallel searches need processes. Workers are
plain subprocesses (`python -m porygon.engine_worker`) speaking one JSON object per line, so they
import only poke-engine (no torch / poke-env) and never re-run the caller's __main__. `EnginePool(python=...)`
runs them under another interpreter, e.g. a venv with a different poke-engine build (A/B tests).
"""

import asyncio
import json
import sys
from pathlib import Path

import poke_engine as pe


def search(state_str: str, ms: int) -> list[tuple[str, int, float]]:
    """MCTS on one determinization -> [(our engine choice, visits, total score in [0, visits])]."""
    result = pe.monte_carlo_tree_search(pe.State.from_string(state_str), duration_ms=ms)
    return [(o.move_choice, o.visits, o.total_score) for o in result.side_one]


def main() -> None:
    for line in sys.stdin:
        req = json.loads(line)
        try:
            reply = {"ok": search(req["state"], req["ms"])}
        except BaseException as e:  # Rust panics surface as pyo3 PanicException (a BaseException)
            reply = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


class EnginePool:
    """`n` worker subprocesses on the running event loop; `search` awaits a free one.

    A search that runs `grace` seconds past its budget is killed and its worker replaced:
    poke-engine can loop forever on some states (seen on a forced switch after our own U-turn),
    which would otherwise stall the player until the battle timer forfeits it."""

    def __init__(self, n: int, grace: float = 30.0, python: str | None = None):
        self.n, self.grace, self.python = n, grace, python
        self.idle: asyncio.Queue | None = None

    async def _spawn(self) -> asyncio.subprocess.Process:
        cmd = [self.python, str(Path(__file__))] if self.python else [sys.executable, "-m", "porygon.engine_worker"]
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, limit=2**24,
        )

    async def search(self, state_str: str, ms: int) -> list[tuple[str, int, float]]:
        if self.idle is None:
            self.idle = asyncio.Queue()
            for _ in range(self.n):
                self.idle.put_nowait(await self._spawn())
        proc, line = await self.idle.get(), b""
        try:
            proc.stdin.write((json.dumps({"state": state_str, "ms": ms}) + "\n").encode())
            await proc.stdin.drain()
            line = await asyncio.wait_for(proc.stdout.readline(), ms / 1000 + self.grace)
        finally:
            if not line:  # crashed (e.g. a Rust abort) or timed out: replace it
                if proc.returncode is None:
                    proc.kill()
                proc = await self._spawn()
            self.idle.put_nowait(proc)
        if not line:
            raise RuntimeError("poke-engine worker died")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"poke-engine: {reply['error']}")
        return [tuple(o) for o in reply["ok"]]


if __name__ == "__main__":
    main()
