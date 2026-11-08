"""Shared tools for the wsp (weight-space programming) analysis of the gen4 PPO network.

- `load(path)`            -> MaskedPolicy (eval mode) from a PPO .zip or pool .pt
- `load_data(dir)`        -> dict of numpy arrays from scripts/wsp/prep_data.py (decisions.npz)
- `Net(policy)`           -> an instrumented re-implementation of the forward pass that exposes every
                             intermediate (token inputs, residual stream per layer, per-head attention
                             patterns, MLP hidden units, policy logits, value) and accepts interventions.
                             `Net.check(obs)` verifies it matches the stock forward to float tolerance.
- `split_obs(x)`          -> the observation cut into named pieces (field numerics, per-mon ids/numerics,
                             move numerics), with index constants for the encoder layout.

Token order (17 tokens): 0 field | 1-6 own mons (switch actions 0-5) | 7-12 opponent mons |
13-16 the four moves of our active mon (move actions 6-9).
Value is V(s) ≈ E[Φ_end] − Φ(s) (potential shaping, porygon.lookahead.potential); logged values are V + Φ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from porygon.encoder import (
    ACTIVE_FLAG, FIELD_DIM, MON_DIM, MON_NUM_DIM, MOVE_DIM, MOVEPOOL, N_CAT, N_MOVES, OBS_DIM,
)
from porygon.model import load_policy

GLORIA = "/vol/bitbucket/fr1226/gloria"
CKPTS = {
    "400M": f"{GLORIA}/runs/20261024-135245-300664/final_model.zip",
    "500M": f"{GLORIA}/runs/20261025-195826-301040/final_model.zip",
    "600M": f"{GLORIA}/runs/20261026-082356-301120/final_model.zip",
}
TOKEN_NAMES = ["field"] + [f"own{i}" for i in range(6)] + [f"opp{i}" for i in range(6)] + [f"move{j}" for j in range(4)]
NUM_OFF = N_CAT + MOVEPOOL                      # start of mon numerics inside a mon block
MOVENUM_OFF = NUM_OFF + MON_NUM_DIM            # start of the 4 x MOVE_DIM move numerics


def load(path: str, device: str = "cpu"):
    return load_policy(CKPTS.get(path, path), device=device)


def load_data(d: str | Path) -> dict[str, np.ndarray]:
    z = np.load(Path(d) / "decisions.npz")
    return {k: z[k] for k in z.files}


def split_obs(x: np.ndarray) -> dict[str, np.ndarray]:
    """x: [N, OBS_DIM] -> field [N,FIELD_DIM], cats [N,12,7] (species,item,ability,move1-4), pool [N,12,24],
    nums [N,12,MON_NUM_DIM], move_nums [N,12,4,MOVE_DIM]."""
    mons = x[:, FIELD_DIM:].reshape(len(x), 12, MON_DIM)
    return {"field": x[:, :FIELD_DIM], "cats": mons[..., :N_CAT].astype(np.int64),
            "pool": mons[..., N_CAT:NUM_OFF].astype(np.int64), "nums": mons[..., NUM_OFF:MOVENUM_OFF],
            "move_nums": mons[..., MOVENUM_OFF:].reshape(len(x), 12, N_MOVES, MOVE_DIM)}


@dataclass
class Intervention:
    """Edits applied during Net.forward. All optional.
    head_mask[layer]       : tensor [n_heads] multiplying each head's output (0 = ablate)
    head_mean[layer]       : dict head -> mean output [d_head] used instead of the head's output (mean ablation)
    resid_add[(layer,pos)] : vector added to the residual stream after `layer` (-1 = token inputs) at token pos
                             (pos None = all tokens)
    mlp_mask[layer]        : tensor [d_ff] multiplying MLP hidden units
    zero_embeddings        : set of {"species","items","abilities","moves"} replaced by their table mean
    """
    head_mask: dict = field(default_factory=dict)
    head_mean: dict = field(default_factory=dict)
    resid_add: dict = field(default_factory=dict)
    mlp_mask: dict = field(default_factory=dict)
    zero_embeddings: set = field(default_factory=set)


class Net:
    def __init__(self, policy):
        self.p = policy
        self.fe = policy.features_extractor
        self.layers = list(self.fe.encoder.layers)
        self.d = self.fe.d_model
        self.h = self.layers[0].self_attn.num_heads
        self.dh = self.d // self.h

    # ---- token inputs -------------------------------------------------------------------------------
    def embed(self, x: torch.Tensor, iv: Intervention | None = None, cache: dict | None = None) -> torch.Tensor:
        fe = self.fe
        mons = x[:, FIELD_DIM:].reshape(x.shape[0], 12, MON_DIM)
        cats = mons[..., :N_CAT].long()
        pool = mons[..., N_CAT:NUM_OFF].long()
        nums = mons[..., NUM_OFF:MOVENUM_OFF]
        move_nums = mons[..., MOVENUM_OFF:].reshape(*mons.shape[:2], N_MOVES, MOVE_DIM)

        def emb(name, ids):
            table = getattr(fe, name)
            if iv and name in iv.zero_embeddings:
                w = table.weight[2:].mean(0)  # mean of real rows (skip pad/unk)
                out = w.expand(*ids.shape, -1).clone()
                out[ids == 0] = 0.0
                return out
            return table(ids)

        mv_e = emb("moves", cats[..., 3:])
        moves = fe.move_proj(torch.cat([mv_e, move_nums], -1))
        pool_mask = (pool > 0).unsqueeze(-1).float()
        pool_mean = (emb("moves", pool) * pool_mask).sum(-2) / pool_mask.sum(-2).clamp(min=1)
        sp, it, ab = emb("species", cats[..., 0]), emb("items", cats[..., 1]), emb("abilities", cats[..., 2])
        mon = torch.cat([sp, it, ab, moves.flatten(-2), pool_mean, nums], -1)
        active = nums[:, :6, ACTIVE_FLAG]
        active_moves = torch.einsum("bs,bsmd->bmd", active, moves[:, :6])
        tokens = torch.cat([fe.field_proj(x[:, :FIELD_DIM]).unsqueeze(1), fe.mon_proj(mon),
                            fe.action_move_proj(active_moves)], 1)
        if cache is not None:
            cache.update(mon_in=mon, move_tok_in=moves, active_moves=active_moves, tokens=tokens)
        return tokens + fe.pos

    # ---- one encoder layer --------------------------------------------------------------------------
    def layer(self, i: int, x: torch.Tensor, iv: Intervention | None, cache: dict | None) -> torch.Tensor:
        L = self.layers[i]
        B, T, D = x.shape
        a = L.norm1(x)
        W, b = L.self_attn.in_proj_weight, L.self_attn.in_proj_bias
        qkv = F.linear(a, W, b).view(B, T, 3, self.h, self.dh)
        q, k, v = qkv.unbind(2)                                   # B,T,H,dh
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.dh)
        pat = scores.softmax(-1)                                  # B,H,Tq,Tk
        z = torch.einsum("bhqk,bkhd->bqhd", pat, v)               # B,T,H,dh
        if iv and i in iv.head_mean:
            for hh, m in iv.head_mean[i].items():
                z[:, :, hh] = m.to(z)
        if iv and i in iv.head_mask:
            z = z * iv.head_mask[i].view(1, 1, -1, 1).to(z)
        Wo, bo = L.self_attn.out_proj.weight, L.self_attn.out_proj.bias
        # per-head write into the residual stream: B,T,H,D
        head_out = torch.einsum("bthd,ehd->bthe", z, Wo.view(D, self.h, self.dh))
        attn = head_out.sum(2) + bo
        x_mid = x + attn
        f = L.norm2(x_mid)
        hid = L.activation(L.linear1(f)) if callable(L.activation) else F.gelu(L.linear1(f))
        if iv and i in iv.mlp_mask:
            hid = hid * iv.mlp_mask[i].to(hid)
        mlp = L.linear2(hid)
        x_out = x_mid + mlp
        if cache is not None:
            cache[f"L{i}.pattern"] = pat
            cache[f"L{i}.z"] = z
            cache[f"L{i}.head_out"] = head_out
            cache[f"L{i}.attn"] = attn
            cache[f"L{i}.mid"] = x_mid
            cache[f"L{i}.mlp_pre"] = L.linear1(f)
            cache[f"L{i}.mlp_hid"] = hid
            cache[f"L{i}.mlp"] = mlp
            cache[f"L{i}.out"] = x_out
        return x_out

    # ---- full forward -------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: torch.Tensor, iv: Intervention | None = None,
                cache: dict | None = None):
        """x [B, OBS_DIM] float, mask [B, 10]. Returns (masked logits [B,10], value V [B])."""
        h = self.embed(x, iv, cache)
        if iv:
            for (li, pos), vec in iv.resid_add.items():
                if li == -1:
                    h = h.clone()
                    if pos is None:
                        h = h + vec
                    else:
                        h[:, pos] = h[:, pos] + vec
        if cache is not None:
            cache["L-1.out"] = h
        for i in range(len(self.layers)):
            h = self.layer(i, h, iv, cache)
            if iv:
                for (li, pos), vec in iv.resid_add.items():
                    if li == i:
                        h = h.clone()
                        if pos is None:
                            h = h + vec
                        else:
                            h[:, pos] = h[:, pos] + vec
        latent = self.fe.norm(h)
        if cache is not None:
            cache["final"] = latent
        flat = latent.flatten(1)
        logits = self.p.action_net(flat)
        logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        value = self.p.value_net(self.p.mlp_extractor.forward_critic(flat)).squeeze(-1)
        return logits, value

    def policy_from_latent(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Policy logits from any [B,17,D] stream already passed through a LayerNorm (logit lens)."""
        logits = self.p.action_net(latent.flatten(1))
        return logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

    @torch.no_grad()
    def check(self, x: torch.Tensor, mask: torch.Tensor) -> float:
        obs = {"observation": x, "action_mask": mask}
        dist = self.p.get_distribution(obs)
        ref_p = dist.distribution.probs
        ref_v = self.p.predict_values(obs).squeeze(-1)
        lg, v = self.forward(x, mask)
        return max(float((lg.softmax(-1) - ref_p).abs().max()), float((v - ref_v).abs().max()))


@torch.no_grad()
def run_batched(net: Net, x: np.ndarray, mask: np.ndarray, iv: Intervention | None = None, bs: int = 2048,
                keep: tuple[str, ...] = (), device: str = "cpu"):
    """Forward over a whole dataset. Returns probs [N,10], value [N], and the requested cache entries (numpy)."""
    P, V, C = [], [], {k: [] for k in keep}
    for s in range(0, len(x), bs):
        xb = torch.as_tensor(x[s:s + bs], dtype=torch.float32, device=device)
        mb = torch.as_tensor(mask[s:s + bs], device=device)
        cache = {} if keep else None
        lg, v = net.forward(xb, mb, iv, cache)
        P.append(lg.softmax(-1).cpu().numpy())
        V.append(v.cpu().numpy())
        for k in keep:
            C[k].append(cache[k].float().cpu().numpy())
    out = {"probs": np.concatenate(P), "value": np.concatenate(V)}
    out.update({k: np.concatenate(v) for k, v in C.items()})
    return out


if __name__ == "__main__":  # self-test: instrumented forward == stock forward
    import sys
    pol = load(sys.argv[1] if len(sys.argv) > 1 else "500M")
    data = load_data(sys.argv[2] if len(sys.argv) > 2 else "/vol/bitbucket/fr1226/porygon-ws-wsp/data")
    n = Net(pol)
    x = torch.as_tensor(data["obs"][:512], dtype=torch.float32)
    m = torch.as_tensor(data["mask"][:512])
    print("max |instrumented - stock| =", n.check(x, m))
    out = run_batched(n, data["obs"][:512], data["mask"][:512])
    print("max |probs - logged probs| =", float(np.abs(out["probs"] - data["probs"][:512]).max()))
