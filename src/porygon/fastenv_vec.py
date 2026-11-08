"""porygon.fastenv as a plain lockstep SB3 VecEnv with poke-env's step semantics: one sub-env per battle side,
every step every sub-env gets an action; a side with nothing to decide this step (the opponent is force-switching)
gets a *wait* observation (its last observation, mask = only action 0, like poke-env's `battle._wait`) and its
action is ignored. Episodes auto-reset (SB3 convention: the returned obs of a done sub-env is the next battle's
first observation; info["terminal_observation"] holds the last one). For plain `stable_baselines3.PPO` and for
checking the async path (`fastenv_sb3.FastBattles` + AsyncPPO) against SB3's own rollout code.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from porygon.fastenv import FastEnv


class FastVecEnv(VecEnv):
    def __init__(self, env: FastEnv):
        self.env = env
        b = env.reset()
        d, a = b.obs.shape[1], b.mask.shape[1]
        n = env.n_workers * env.span
        super().__init__(n, spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(d,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(a,), dtype=np.int8),
        }), spaces.Discrete(a))
        self.obs = np.zeros((n, d), np.float32)
        self.mask = np.zeros((n, a), np.int8)
        self.pending = np.zeros(n, bool)
        self._batch = b
        self._apply(b, np.zeros(n, np.float32), np.zeros(n, bool), [{} for _ in range(n)])
        self._actions = None

    def _apply(self, b, rewards, dones, infos) -> None:
        """Rows in order: accumulate rewards; a terminal row ends that sub-env's episode (later rows of the same
        slot in this batch belong to the next battle)."""
        self.pending[:] = False
        for i in range(len(b)):
            s = int(b.slot[i])
            rewards[s] += float(b.reward[i])
            if b.flags[i] & 2:
                dones[s] = True
                infos[s]["terminal_observation"] = {"observation": self.obs[s].copy(),
                                                    "action_mask": self.mask[s].copy()}
                infos[s]["win"] = int(b.win[i])
            elif b.flags[i] & 1:
                self.obs[s] = b.obs[i]
                self.mask[s] = b.mask[i]
                self.pending[s] = True
        wait = ~self.pending
        self.mask[wait] = 0
        self.mask[wait, 0] = 1

    def reset(self):
        return {"observation": self.obs.copy(), "action_mask": self.mask.copy()}

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions)

    def step_wait(self):
        b = self._batch
        rows = np.flatnonzero(b.needs_action)
        self._batch = b = self.env.step(self._actions[b.slot[rows]])
        n = self.num_envs
        rewards, dones, infos = np.zeros(n, np.float32), np.zeros(n, bool), [{} for _ in range(n)]
        self._apply(b, rewards, dones, infos)
        return self.reset(), rewards, dones, infos

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None) -> None:
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
