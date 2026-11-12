from pathlib import Path

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from porygon.encoder import (
    ACTIVE_FLAG, FIELD_DIM, MON_DIM, MON_NUM_DIM, MOVE_DIM, MOVEPOOL, N_CAT, N_MOVES, N_TOKENS,
    VOCAB_SIZES,
)

# field | 6 own mons (switch slots 0-5) | 6 opponent mons | 4 moves of our active mon (actions 6-9)
N_ALL_TOKENS = N_TOKENS + N_MOVES
SWITCH_TOKENS = slice(1, 7)
MOVE_TOKENS = slice(N_TOKENS, N_ALL_TOKENS)


def mlp(d_in: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))


class BattleTransformer(BaseFeaturesExtractor):
    """Embeds field, 12 mons and our active mon's 4 moves as tokens and mixes them with a
    pre-norm transformer encoder. Output is the flattened token sequence; `ActionHead` reads
    the switch tokens (own mons, team order) and move tokens back out of it.
    """

    def __init__(self, observation_space: spaces.Dict, d_model: int = 256, n_layers: int = 3,
                 n_heads: int = 8, species_dim: int = 64, move_dim: int = 64,
                 item_dim: int = 32, ability_dim: int = 32):
        super().__init__(observation_space, features_dim=N_ALL_TOKENS * d_model)
        self.d_model = d_model
        self.species = nn.Embedding(VOCAB_SIZES["species"], species_dim, padding_idx=0)
        self.items = nn.Embedding(VOCAB_SIZES["items"], item_dim, padding_idx=0)
        self.abilities = nn.Embedding(VOCAB_SIZES["abilities"], ability_dim, padding_idx=0)
        self.moves = nn.Embedding(VOCAB_SIZES["moves"], move_dim, padding_idx=0)
        self.move_proj = mlp(move_dim + MOVE_DIM, move_dim)
        mon_in = species_dim + item_dim + ability_dim + (N_MOVES + 1) * move_dim + MON_NUM_DIM
        self.mon_proj = mlp(mon_in, d_model)
        self.field_proj = mlp(FIELD_DIM, d_model)
        self.action_move_proj = mlp(move_dim, d_model)
        self.pos = nn.Parameter(torch.randn(N_ALL_TOKENS, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 2 * d_model, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        x = obs["observation"]
        mons = x[:, FIELD_DIM:].reshape(x.shape[0], 12, MON_DIM)
        cats = mons[..., :N_CAT].long()
        pool = mons[..., N_CAT:N_CAT + MOVEPOOL].long()
        nums = mons[..., N_CAT + MOVEPOOL:N_CAT + MOVEPOOL + MON_NUM_DIM]
        move_nums = mons[..., N_CAT + MOVEPOOL + MON_NUM_DIM:].reshape(*mons.shape[:2], N_MOVES, MOVE_DIM)

        moves = self.move_proj(torch.cat([self.moves(cats[..., 3:]), move_nums], -1))  # B,12,4,m
        pool_mask = (pool > 0).unsqueeze(-1).float()
        pool_mean = (self.moves(pool) * pool_mask).sum(-2) / pool_mask.sum(-2).clamp(min=1)
        mon = torch.cat([
            self.species(cats[..., 0]), self.items(cats[..., 1]), self.abilities(cats[..., 2]),
            moves.flatten(-2), pool_mean, nums,
        ], -1)

        # Moves of our active mon (zeros when none is active; those actions are masked anyway).
        active = nums[:, :6, ACTIVE_FLAG]  # B,6
        active_moves = torch.einsum("bs,bsmd->bmd", active, moves[:, :6])

        tokens = torch.cat([
            self.field_proj(x[:, :FIELD_DIM]).unsqueeze(1),
            self.mon_proj(mon),
            self.action_move_proj(active_moves),
        ], 1)
        return self.norm(self.encoder(tokens + self.pos)).flatten(1)


class ActionHead(nn.Module):
    """Scores every action from its own token plus the field token, which summarises the
    battle: switch i <- own mon token i, move j <- active-move token j. Tera variants
    (gen9 actions 22-25) reuse the move tokens shifted by a learned tera vector."""

    def __init__(self, d_model: int, n_actions: int):
        super().__init__()
        self.d_model, self.n_actions = d_model, n_actions
        self.score = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.tera = nn.Parameter(torch.zeros(d_model))
        nn.init.zeros_(self.score[-1].weight)  # start from a uniform policy over legal actions
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        t = latent.view(latent.shape[0], N_ALL_TOKENS, self.d_model)
        moves = t[:, MOVE_TOKENS]
        cands = [t[:, SWITCH_TOKENS], moves]
        if self.n_actions > 10:
            cands += [torch.zeros_like(moves)] * ((self.n_actions - 14) // 4) + [moves + self.tera]
        cands = torch.cat(cands, 1)
        ctx = t[:, :1].expand_as(cands)
        return self.score(torch.cat([cands, ctx], -1)).squeeze(-1)


class MaskedPolicy(ActorCriticPolicy):
    """Actor-critic with per-action pointer logits, masked by obs["action_mask"]."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", BattleTransformer)
        kwargs.setdefault("net_arch", dict(pi=[], vf=[256, 256]))  # actor reads tokens directly
        kwargs.setdefault("ortho_init", False)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = ActionHead(self.features_extractor.d_model, int(self.action_space.n))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def get_distribution(self, obs):
        self._mask = obs["action_mask"]
        return super().get_distribution(obs)

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi)
        logits = logits.masked_fill(self._mask == 0, torch.finfo(logits.dtype).min)
        return self.action_dist.proba_distribution(action_logits=logits)


class _ScaleGrad(torch.autograd.Function):
    """Identity forward; multiplies the gradient by `scale` on the way back."""

    @staticmethod
    def forward(ctx, x, scale: float):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def scale_grad(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x if scale == 1.0 else _ScaleGrad.apply(x, scale)


def set_value_trunk_grad(policy: "MaskedPolicy", scale: float) -> None:
    """Let only `scale` of the value loss's gradient reach the shared transformer trunk (the critic MLP itself
    still gets all of it). MissingNo.Zero found the value gradient into a shared trunk ~50x the policy's: the net
    learns to predict who wins rather than to play. A forward pre-hook on the critic MLP, so checkpoints keep
    their state-dict layout; call again after loading."""
    critic = policy.mlp_extractor.value_net
    old = getattr(policy, "_value_trunk_hook", None)
    if old is not None:
        old.remove()
    policy.value_trunk_grad = scale
    policy._value_trunk_hook = None if scale == 1.0 else critic.register_forward_pre_hook(
        lambda module, args: (scale_grad(args[0], scale),))


def load_policy(path: str | Path, device: str = "cpu") -> MaskedPolicy:
    """Eval-mode policy from a PPO checkpoint (.zip) or an opponent-pool snapshot (.pt)."""
    if Path(path).suffix == ".pt":
        return MaskedPolicy.load(str(path), device=device).eval()
    from stable_baselines3 import PPO
    return PPO.load(path, device=device).policy.eval()
