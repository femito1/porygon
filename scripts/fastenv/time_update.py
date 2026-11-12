"""Seconds per PPO update of the current recipe (MaskedPolicy 3.5M, rollout 39936, batch 1024, 7 epochs) on this
GPU, with a synthetic rollout: the learner-side ceiling on env steps/s is rollout / update_seconds.

    python scripts/fastenv/time_update.py [--n-actions 26] [--updates 2]
"""
import argparse
import time

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from porygon.async_env import AsyncPPO
from porygon.encoder import OBS_DIM
from porygon.model import MaskedPolicy
from porygon.train import thesis_lr


class Facade(VecEnv):
    def __init__(self, a):
        super().__init__(1, spaces.Dict({"observation": spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
                                         "action_mask": spaces.Box(0, 1, (a,), np.int8)}), spaces.Discrete(a))

    def reset(self):
        return {k: np.zeros((1, *s.shape), s.dtype) for k, s in self.observation_space.spaces.items()}

    def step_async(self, actions): ...
    def step_wait(self): ...
    def close(self): ...
    def get_attr(self, *a, **k): return [None]
    def set_attr(self, *a, **k): ...
    def env_method(self, *a, **k): return [None]
    def env_is_wrapped(self, *a, **k): return [False]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-actions", type=int, default=26)
    p.add_argument("--rollout", type=int, default=78 * 512)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--autocast", choices=["none", "bf16"], default="none")
    a = p.parse_args()
    torch.set_float32_matmul_precision("high")
    model = AsyncPPO(MaskedPolicy, Facade(a.n_actions), learning_rate=thesis_lr, n_steps=a.rollout, batch_size=a.batch,
                     n_epochs=a.epochs, gamma=0.9999, gae_lambda=0.754, clip_range=0.0829, clip_range_vf=0.0184,
                     ent_coef=0.01, vf_coef=0.4375, max_grad_norm=0.543, device="cuda", verbose=0,
                     policy_kwargs=dict(features_extractor_kwargs=dict(d_model=256, n_layers=3)))
    model._setup_learn(10**9, None)
    buf = model.rollout_buffer
    rng = np.random.default_rng(0)
    obs = rng.random((a.rollout, 1, OBS_DIM), dtype=np.float32)
    obs[..., :] = np.where(rng.random(obs.shape) < 0.5, 0.0, obs)
    buf.observations["observation"][:] = np.floor(obs * 3)  # small ints: valid embedding ids everywhere
    buf.observations["action_mask"][:] = 1
    buf.actions[:] = rng.integers(0, a.n_actions, buf.actions.shape)
    buf.log_probs[:] = -2.0
    buf.values[:] = 0.0
    buf.advantages[:] = rng.standard_normal(buf.advantages.shape)
    buf.returns[:] = rng.standard_normal(buf.returns.shape)
    buf.pos, buf.full = a.rollout, True
    print(torch.cuda.get_device_name(), f"rollout {a.rollout} epochs {a.epochs} batch {a.batch} autocast {a.autocast}")
    import contextlib
    ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if a.autocast == "bf16" else contextlib.nullcontext
    for i in range(a.updates):
        torch.cuda.synchronize()
        t = time.time()
        with ctx():
            model.train()
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f"update {i}: {dt:.1f} s -> learner ceiling {a.rollout / dt:,.0f} steps/s")
    # inference throughput (rollout collection forwards)
    pol = model.policy.eval()
    for bs in (256, 2048):
        x = {"observation": torch.as_tensor(obs[:bs, 0] * 0, device="cuda"),
             "action_mask": torch.ones(bs, a.n_actions, dtype=torch.int8, device="cuda")}
        with torch.no_grad():
            pol(x)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(20):
                pol(x)
            torch.cuda.synchronize()
        print(f"inference batch {bs}: {20 * bs / (time.time() - t):,.0f} obs/s")


if __name__ == "__main__":
    main()
