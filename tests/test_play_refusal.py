"""porygon.play --challenge ends cleanly when Showdown refuses the challenge (poke-env only logs it and waits)."""

import asyncio
import logging
import threading

from porygon.play import challenge


class FakePlayer:
    """send_challenges never finishes (no battle starts); the refusal arrives as a log line from poke-env's own
    thread, the way ps_client reports popups."""

    def __init__(self, line: str | None, level: int = logging.WARNING):
        self.logger = logging.getLogger(f"fake-player-{id(self)}")  # NOTSET + a stream handler, as poke-env's
        self.console = logging.StreamHandler()
        self.logger.addHandler(self.console)
        self.line, self.level, self.cancelled = line, level, False

    async def send_challenges(self, opponent, n_challenges):
        if self.line is not None:
            threading.Timer(0.05, self.logger.log, (self.level, self.line)).start()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def run(player, capsys):
    asyncio.run(asyncio.wait_for(challenge(player, "someone", 3), timeout=5))
    return capsys.readouterr().out


def test_user_not_found_stops(capsys):
    p = FakePlayer("Popup message received: |popup|The user 'someone' was not found.")
    out = run(p, capsys)
    assert "refused the challenge to someone: The user 'someone' was not found." in out
    assert p.cancelled and p.logger.handlers == [p.console]  # the watch is gone, levels restored
    assert p.logger.level == logging.NOTSET and p.console.level == logging.NOTSET


def test_rejection_pm_stops(capsys):
    p = FakePlayer("Received pm: |pm| someone| porygon-agent|/log someone rejected the challenge.", logging.INFO)
    assert "rejected the challenge" in run(p, capsys)


def test_unrelated_popup_keeps_waiting(capsys):
    p = FakePlayer("Popup message received: |popup|Your battle is starting.")
    try:
        asyncio.run(asyncio.wait_for(challenge(p, "someone", 1), timeout=0.5))
    except (asyncio.TimeoutError, TimeoutError):
        pass
    else:
        raise AssertionError("challenge() returned on an unrelated popup")
    assert p.cancelled


def test_completed_challenges_return(capsys):
    class Done(FakePlayer):
        async def send_challenges(self, opponent, n_challenges):
            return None

    p = Done(None)
    assert "refused" not in run(p, capsys) and p.logger.handlers == [p.console]
