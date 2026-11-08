"""porygon.fastenv behind the `AsyncBattles` interface of porygon.async_env, so `AsyncPPO` (and
`porygon.train --fastenv`) trains on in-process Showdown battles unchanged.

Every slot (one side of one battle position) is a trajectory: a transition is (observation, action, value,
log-prob) when the action is sent, completed by the slot's next row (its reward, and done if that row is
terminal). Transitions still open when a rollout ends (the opponent has not answered yet; rare) carry over to the
next rollout. A segment that ends mid-episode is bootstrapped from the slot's next observation, as in AsyncBattles.
Both sides of every battle are the learner (self-play).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from porygon.fastenv import Batch, FastEnv


class FastBattles(VecEnv):
    def __init__(self, env: FastEnv):
        self.env = env
        first = dict(env.poll_all())
        d, a = first[0].obs.shape[1], first[0].mask.shape[1]
        obs_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(d,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(a,), dtype=np.int8),
        })
        super().__init__(1, obs_space, spaces.Discrete(a))
        self.ready: dict[int, Batch] = first            # worker -> batch whose decisions are not answered yet
        self.open: dict[int, tuple] = {}                # slot -> (obs dict, action, value, log-prob) awaiting outcome
        self.ep_stats: dict[int, list] = defaultdict(lambda: [0.0, 0])
        self.outcomes = {"games": 0, "wins": 0, "ties": 0}

    def _absorb(self, b: Batch, segments: dict, finished: list) -> None:
        """Completes open transitions with this batch's rows (rewards, terminal flags)."""
        for i in range(len(b)):
            s = int(b.slot[i])
            t = self.open.pop(s, None)
            done = bool(b.flags[i] & 2)
            if t is not None:
                r = float(b.reward[i])
                segments[s].append((*t, r, done))
                st = self.ep_stats[s]
                st[0] += r
                st[1] += 1
            if done:
                st = self.ep_stats.pop(s, [0.0, 0])
                finished.append({"r": st[0], "l": st[1]})
                if s % 2 == 0:
                    self.outcomes["games"] += 1
                    self.outcomes["wins"] += int(b.win[i] > 0)
                    self.outcomes["ties"] += int(b.win[i] == 0)

    def collect(self, policy, n: int, on_transitions: Callable[[int, list[dict]], bool]):
        segments: dict[int, list[tuple]] = defaultdict(list)
        in_flight: set[int] = set()
        done_n, keep_going = 0, True  # completed transitions (fill_buffer needs >= n; extras are trimmed)
        while (done_n < n and keep_going) or in_flight:
            if self.ready and done_n < n and keep_going:
                ws = list(self.ready)
                batches = [self.ready.pop(w) for w in ws]
                rows = [b.needs_action for b in batches]
                obs = np.concatenate([b.obs[r] for b, r in zip(batches, rows)])
                mask = np.concatenate([b.mask[r] for b, r in zip(batches, rows)]).astype(np.int8)
                actions, values, log_probs = self._forward(policy, obs, mask)
                k = 0
                for w, b, r in zip(ws, batches, rows):
                    m = int(r.sum())
                    self.env.send_to(w, b, actions[k:k + m])
                    for j, s in enumerate(b.slot[r]):
                        self.open[int(s)] = ({"observation": obs[k + j], "action_mask": mask[k + j]},
                                             int(actions[k + j]), float(values[k + j]), float(log_probs[k + j]))
                    k += m
                    in_flight.add(w)
            finished: list[dict] = []
            got = 0
            for w, b in self.env.poll():
                in_flight.discard(w)
                before = sum(map(len, segments.values()))
                self._absorb(b, segments, finished)
                got += sum(map(len, segments.values())) - before
                self.ready[w] = b
            done_n += got
            if got or finished:
                keep_going = on_transitions(got, finished) and keep_going
        # bootstrap every segment that ends mid-episode from its slot's next observation
        boot: dict[int, float] = {}
        need: list[int] = []
        for s, seg in segments.items():
            if seg[-1][5]:
                continue
            if s in self.open:  # action already taken there (carried over): its value is known
                boot[s] = self.open[s][2]
            else:
                need.append(s)
        if need:
            where = {}
            for b in self.ready.values():
                for i in np.flatnonzero(b.needs_action):
                    where[int(b.slot[i])] = (b, i)
            have = [s for s in need if s in where]
            if have:
                obs = np.stack([where[s][0].obs[where[s][1]] for s in have])
                mask = np.stack([where[s][0].mask[where[s][1]] for s in have]).astype(np.int8)
                vals = self._forward(policy, obs, mask)[1]
                boot.update(zip(have, map(float, vals)))
            for s in need:
                boot.setdefault(s, 0.0)
        return dict(segments), boot, keep_going

    @staticmethod
    def _forward(policy, obs: np.ndarray, mask: np.ndarray):
        batch = {"observation": torch.as_tensor(obs, device=policy.device),
                 "action_mask": torch.as_tensor(mask, device=policy.device)}
        with torch.no_grad():
            actions, values, log_probs = policy(batch)
        return actions.cpu().numpy(), values.flatten().cpu().numpy(), log_probs.cpu().numpy()

    # --- VecEnv facade: SB3 only resets it once during setup ---
    def reset(self):
        return {k: np.zeros((1, *s.shape), s.dtype) for k, s in self.observation_space.spaces.items()}

    def step_async(self, actions):
        raise NotImplementedError("FastBattles is driven by AsyncPPO.collect_rollouts")

    def step_wait(self):
        raise NotImplementedError("FastBattles is driven by AsyncPPO.collect_rollouts")

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        return [None]

    def set_attr(self, attr_name, value, indices=None) -> None:
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]
