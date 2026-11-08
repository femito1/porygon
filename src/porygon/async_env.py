"""Asynchronous rollout collection: battles never wait for each other.

The lockstep SB3 vector env makes every step wait for the slowest of ~150 battles. One Showdown
turn usually takes 20-30 ms, but the slowest of 150 takes ~100 ms, so most cores sit idle. Here
every battle runs in its own worker process and asks for an action as soon as it needs one; the
trainer batches whatever requests are waiting into one network forward pass and answers them.

Each battle side's transitions form a separate trajectory segment. A rollout concatenates the
segments in SB3's rollout buffer (n_envs = 1) with an episode start at every segment boundary and,
where a segment ends mid-episode, gamma * V(next observation) added to its last reward: the
same bootstrapping SB3 uses for time limits, so PPO's GAE and update run unchanged.

By default battles pause while PPO trains (on-policy, as in lockstep). With `overlap_update` a
background thread keeps collecting the next rollout during the update, acting with a copy of
the network that is refreshed before each rollout. Training data is then at most one update old;
PPO's ratio uses the stored log-probs of the policy that acted, so it stays a valid importance
ratio, only the clipping is relative to that slightly older policy.
"""

import copy
import multiprocessing as mp
import queue
import threading
from collections import defaultdict
from multiprocessing.connection import Connection, wait
from typing import Any, Callable

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper

from porygon.env import SelfPlayEnv, local_server
from porygon.pool import PoolOpponent


def selfplay_battle(port: int, battle_format: str, shaping: dict[str, float] | None) -> SelfPlayEnv:
    return SelfPlayEnv(battle_format=battle_format, server_configuration=local_server(port), shaping=shaping,
                       log_level=40, open_timeout=None, challenge_timeout=None, strict=False)


def pool_battle(port: int, battle_format: str, shaping: dict[str, float] | None, pool_dir, p_baseline: float):
    return SingleAgentWrapper(selfplay_battle(port, battle_format, shaping),
                              PoolOpponent(pool_dir, p_baseline=p_baseline, battle_format=battle_format))


def _worker(conn: Connection, make: CloudpickleWrapper) -> None:
    """One battle. Sends ("obs", [obs per side]), then for every list of actions received, the
    next observations (a new battle's after the end), the rewards and whether the battle ended."""
    env = make.var()
    if isinstance(env, SingleAgentWrapper):
        def reset():
            return [env.reset()[0]]

        def step(actions):
            obs, reward, term, trunc, _ = env.step(np.int64(actions[0]))
            return [obs], [float(reward)], bool(term or trunc)
    else:  # PettingZoo self-play: both sides are the learner and finish together
        agents = env.possible_agents

        def reset():
            obs, _ = env.reset()
            return [obs[a] for a in agents]

        def step(actions):
            obs, rewards, terms, truncs, _ = env.step({a: np.int64(x) for a, x in zip(agents, actions)})
            return [obs[a] for a in agents], [float(rewards[a]) for a in agents], \
                any(terms.values()) or any(truncs.values())
    conn.send(reset())
    try:
        while (actions := conn.recv()) is not None:
            obs, rewards, done = step(actions)
            conn.send((reset() if done else obs, rewards, done))
    finally:
        env.close()


class AsyncBattles(VecEnv):
    """Battle worker processes behind a VecEnv facade (num_envs = 1) for SB3's setup code;
    `AsyncPPO.collect_rollouts` drives them through `collect`."""

    def __init__(self, makers: list[tuple[Callable[[], Any], str]]):
        probe = SelfPlayEnv(battle_format="gen4randombattle", start_listening=False)
        agent = probe.possible_agents[0]
        super().__init__(1, probe.observation_space(agent), probe.action_space(agent))
        self.conns: list[Connection] = []
        self.procs = []
        for make, start_method in makers:
            ctx = mp.get_context(start_method)
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, CloudpickleWrapper(make)), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        self.ready = {c: c.recv() for c in self.conns}  # worker -> observations awaiting actions
        self.index = {c: i for i, c in enumerate(self.conns)}
        self.ep_stats: dict[tuple[int, int], list] = defaultdict(lambda: [0.0, 0])  # running return, length

    def collect(self, policy, n: int, on_transitions: Callable[[int, list[dict]], bool]):
        """Run battles until n transitions are taken. Returns the segments {(worker, side): [...]}
        in collection order, a bootstrap value per segment, and False if a callback stopped it."""
        segments: dict[tuple[int, int], list[tuple]] = defaultdict(list)
        in_flight: dict[Connection, tuple] = {}
        taken, keep_going = 0, True
        while (taken < n and keep_going) or in_flight:
            if self.ready and taken < n and keep_going:
                conns = list(self.ready)
                obs = [o for c in conns for o in self.ready[c]]
                actions, values, log_probs = self._forward(policy, obs)
                k = 0
                for c in conns:
                    m = len(self.ready[c])
                    c.send(actions[k:k + m].tolist())
                    in_flight[c] = (self.ready.pop(c), actions[k:k + m], values[k:k + m], log_probs[k:k + m])
                    k += m
                taken += len(obs)
            for c in wait(list(in_flight)):
                next_obs, rewards, done = c.recv()
                obs, actions, values, log_probs = in_flight.pop(c)
                finished = []
                for side, (o, a, v, lp, r) in enumerate(zip(obs, actions, values, log_probs, rewards)):
                    key = (self.index[c], side)
                    segments[key].append((o, a, v, lp, r, done))
                    stats = self.ep_stats[key]
                    stats[0] += r
                    stats[1] += 1
                    if done:
                        finished.append({"r": stats[0], "l": stats[1]})
                        del self.ep_stats[key]
                self.ready[c] = next_obs
                keep_going = on_transitions(len(obs), finished) and keep_going
        # every worker is waiting again: bootstrap each segment from its side's pending observation
        keys = list(segments)
        pending = [self.ready[self.conns[w]][side] for w, side in keys]
        boot = dict(zip(keys, self._forward(policy, pending)[1])) if pending else {}
        return segments, boot, keep_going

    @staticmethod
    def _forward(policy, obs: list[dict]):
        batch = {k: torch.as_tensor(np.stack([o[k] for o in obs]), device=policy.device) for k in obs[0]}
        with torch.no_grad():
            actions, values, log_probs = policy(batch)
        return actions.cpu().numpy(), values.flatten().cpu().numpy(), log_probs.cpu().numpy()

    # --- VecEnv facade: SB3 only resets it once during setup ---
    def reset(self):
        return {k: np.zeros((1, *s.shape), s.dtype) for k, s in self.observation_space.spaces.items()}

    def step_async(self, actions):
        raise NotImplementedError("AsyncBattles is driven by AsyncPPO.collect_rollouts")

    def step_wait(self):
        raise NotImplementedError("AsyncBattles is driven by AsyncPPO.collect_rollouts")

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    def get_attr(self, attr_name, indices=None):
        return [None]

    def set_attr(self, attr_name, value, indices=None) -> None:
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]


def fill_buffer(buffer: DictRolloutBuffer, segments: dict, boot: dict, gamma: float) -> None:
    """Concatenate segments into a (buffer_size, 1) rollout buffer and compute GAE. Extra
    transitions beyond buffer_size are dropped from the ends of the longest segments (their
    value becomes that segment's bootstrap). Writes the buffer arrays directly: per-transition
    `buffer.add` and SB3's 1-env GAE loop took ~6 s per 40k-step rollout."""
    excess = sum(map(len, segments.values())) - buffer.buffer_size
    while excess > 0:
        key = max(segments, key=lambda k: len(segments[k]))
        boot[key] = segments[key].pop()[2]
        if not segments[key]:
            del segments[key]
        excess -= 1
    rows = [t for seg in segments.values() for t in seg]
    n = len(rows)
    assert n == buffer.buffer_size, f"{n} transitions for a {buffer.buffer_size}-step buffer"
    values = np.array([t[2] for t in rows], dtype=np.float32)
    rewards = np.array([t[4] for t in rows], dtype=np.float32)
    dones = np.array([t[5] for t in rows], dtype=bool)
    starts = np.zeros(n, dtype=np.float32)
    next_values = np.zeros(n, dtype=np.float32)  # V(s') where the segment continues, 0 if terminal
    i = 0
    for key, seg in segments.items():
        j = i + len(seg)
        starts[i] = 1.0
        starts[i + 1:j] = dones[i:j - 1]
        next_values[i:j - 1] = values[i + 1:j] * ~dones[i:j - 1]
        if not dones[j - 1]:
            next_values[j - 1] = float(boot[key])  # segment cut mid-episode: bootstrap
        i = j
    deltas = rewards + gamma * next_values - values
    # GAE runs backwards within segments and episodes: it is cut wherever the next row starts one
    carry = gamma * buffer.gae_lambda * np.append(1.0 - starts[1:], 0.0)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in range(n - 1, -1, -1):
        gae = deltas[t] + carry[t] * gae
        advantages[t] = gae
    buffer.reset()
    for k in buffer.observations:
        buffer.observations[k][:n, 0] = np.stack([t[0][k] for t in rows])
    buffer.actions[:n, 0, 0] = [t[1] for t in rows]
    buffer.log_probs[:n, 0] = [t[3] for t in rows]
    buffer.values[:n, 0] = values
    buffer.rewards[:n, 0] = rewards
    buffer.episode_starts[:n, 0] = starts
    buffer.advantages[:n, 0] = advantages
    buffer.returns[:n, 0] = advantages + values
    buffer.pos, buffer.full = n, True


class AsyncPPO(PPO):
    """PPO whose rollouts come from `AsyncBattles` instead of a lockstep VecEnv. Set
    `overlap_update = True` to keep battles running during the PPO update (see the module doc)."""

    overlap_update = False
    autocast: str | None = None  # "bf16": run the PPO update under torch.autocast (porygon.train --autocast)

    def train(self) -> None:
        if self.autocast == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return super().train()
        return super().train()

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_actor", "_actor_lock", "_rollouts", "_collector", "_stop"]

    def collect_rollouts(self, env: AsyncBattles, callback: BaseCallback, rollout_buffer, n_rollout_steps: int) -> bool:
        self.policy.set_training_mode(False)
        callback.on_rollout_start()
        if self.overlap_update:
            segments, boot, finished, n = self._next_background_rollout(env, n_rollout_steps)
            self.ep_info_buffer.extend(finished)
            ok = True
            for _ in range(n):
                self.num_timesteps += 1
                ok = callback.on_step() and ok
        else:
            def on_transitions(n: int, finished: list[dict]) -> bool:
                if finished:
                    self.ep_info_buffer.extend(finished)
                ok = True
                for _ in range(n):
                    self.num_timesteps += 1
                    ok = callback.on_step() and ok
                return ok

            threads = torch.get_num_threads()
            torch.set_num_threads(1)  # many small forwards: extra threads only fight the battle workers for CPU
            try:
                segments, boot, ok = env.collect(self.policy, n_rollout_steps, on_transitions)
            finally:
                torch.set_num_threads(threads)
        if not ok:
            if self.overlap_update:
                self._stop.set()
            return False
        fill_buffer(rollout_buffer, segments, boot, self.gamma)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def _next_background_rollout(self, env: AsyncBattles, n: int):
        """Refresh the acting copy with the freshly trained weights, then take the rollout the
        background thread has been collecting (it starts the next one as soon as this is taken)."""
        if getattr(self, "_collector", None) is None:
            torch.set_num_threads(1)  # collection never stops: keep CPU threads for the battle workers
            self._actor = copy.deepcopy(self.policy)
            self._actor.set_training_mode(False)
            self._actor_lock = threading.Lock()
            self._rollouts: queue.Queue = queue.Queue(maxsize=1)
            self._stop = threading.Event()
            self._collector = threading.Thread(target=self._collect_forever, args=(env, n), daemon=True)
            self._collector.start()
        else:
            with self._actor_lock:
                self._actor.load_state_dict(self.policy.state_dict())
        item = self._rollouts.get()
        self._rollouts.task_done()  # lets the collector start the next rollout
        if isinstance(item, BaseException):
            raise item
        return item

    def _collect_forever(self, env: AsyncBattles, n: int) -> None:
        actor_lock = self._actor_lock

        class LockedActor:  # forward passes never see a half-loaded state dict
            device = self._actor.device

            def __call__(_, batch):
                with actor_lock:
                    return self._actor(batch)

        try:
            while not self._stop.is_set():
                finished: list[dict] = []
                segments, boot, _ = env.collect(LockedActor(), n, lambda k, f: finished.extend(f) or True)
                self._rollouts.put((segments, boot, finished, sum(map(len, segments.values()))))
                self._rollouts.join()  # at most one rollout ahead: data is <= one update old
        except BaseException as e:  # surface worker failures in the training thread
            if not self._stop.is_set():
                self._rollouts.put(e)
