"""Play a trained model on the official Pokémon Showdown server (needs a registered account).

    python -m porygon.play runs/<run>/best_model.zip --username NAME --password PASS --ladder 50
    python -m porygon.play runs/<run>/best_model.zip --username NAME --ladder 50 --search   # porygon.search
    python -m porygon.play runs/<run>/best_model.zip --username NAME --password PASS --accept 5
    python -m porygon.play runs/<run>/best_model.zip --local --challenge YourName   # local server
    python -m porygon.play runs/play/R1bF-300M.pt --username NAME --accept 3 --from FRIEND   # a v2 net (raw, T 0.5)
"""

import argparse
import asyncio
import os

from poke_env import AccountConfiguration, LocalhostServerConfiguration, ShowdownServerConfiguration

from porygon.env import PolicyPlayer
from porygon.model import load_policy


class Logging:
    """Player mixin: prints every finished battle with the running record and ratings."""

    def choose_move(self, battle):
        if battle.turn <= 1 and not getattr(battle, "_porygon_announced", False):
            battle._porygon_announced = True
            print(f"started vs {battle.opponent_username}  {battle.battle_tag}", flush=True)
        return super().choose_move(battle)

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        result = "won " if battle.won else "lost" if battle.lost else "tie "
        print(f"{result} vs {battle.opponent_username:<18} rating {battle.rating} (opp {battle.opponent_rating})"
              f"  record {self.n_won_battles}-{self.n_lost_battles}  {battle.battle_tag}", flush=True)


class LoggingPlayer(Logging, PolicyPlayer):
    pass


async def run(args: argparse.Namespace) -> None:
    from porygon.v2.evaluate import checkpoint_kind

    if checkpoint_kind(args.model) == "v2":  # v2 Net (raw policy at --temperature)
        if args.search:
            raise SystemExit("--search takes a v1 model; v2 nets play raw here")
        from porygon.v2.model import load
        from porygon.v2.player import NetPlayer

        class LoggingNetPlayer(Logging, NetPlayer):
            pass

        policy = load(args.model, device="cpu").eval()
        if int(policy.gen) != int(args.format[3]):
            raise SystemExit(f"the net is gen{policy.gen}, the format is {args.format}")
        cls, extra = LoggingNetPlayer, dict(temperature=args.temperature)
    elif args.search:
        policy = load_policy(args.model)
        from porygon.search import SearchPlayer  # needs the gen4 poke-engine build

        class LoggingSearchPlayer(Logging, SearchPlayer):
            pass

        cls, extra = LoggingSearchPlayer, dict(ms=args.ms, samples=args.samples, prior_weight=args.prior_weight)
    else:
        policy = load_policy(args.model)
        cls, extra = LoggingPlayer, {}
    player = cls(
        policy,
        account_configuration=AccountConfiguration(args.username, args.password or os.environ.get("PS_PASSWORD")),
        battle_format=args.format,
        server_configuration=LocalhostServerConfiguration if args.local else ShowdownServerConfiguration,
        start_timer_on_battle_start=not args.local,
        max_concurrent_battles=args.concurrency,
        **extra,
    )
    if args.ladder:
        await player.ladder(args.ladder)
    elif args.challenge:
        await player.send_challenges(args.challenge, n_challenges=args.n)
    else:
        await player.accept_challenges(args.from_user, args.accept)
    print(f"final record: {player.n_won_battles}-{player.n_lost_battles}")
    if args.search:
        print(f"search stats: {player.stats}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--username", required=True)
    p.add_argument("--password", help="or set PS_PASSWORD")
    p.add_argument("--format", default="gen4randombattle")
    p.add_argument("--local", action="store_true", help="use localhost:8000 instead of the official server")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ladder", type=int, metavar="N", help="play N ladder games")
    mode.add_argument("--accept", type=int, metavar="N", help="accept N challenges from anyone")
    mode.add_argument("--challenge", metavar="USER", help="challenge USER --n times")
    p.add_argument("--from", dest="from_user", metavar="USER", help="with --accept: only accept challenges from USER")
    p.add_argument("--temperature", type=float, default=0.5, help="v2 nets: sampling temperature (0.5 = how we evaluate)")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1, help="simultaneous battles")
    p.add_argument("--search", action="store_true", help="play with SearchPlayer (network-guided MCTS)")
    p.add_argument("--ms", type=int, default=1000, help="search time per decision")
    p.add_argument("--samples", type=int, default=4, help="sampled opponent sets per decision")
    p.add_argument("--prior-weight", type=float, default=0.25)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
