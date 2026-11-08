"""Model v2 (owner: netv2): entity transformer over observation spec v2 (`porygon.v2.spec`).

    net = Net(9)                                   # Net(gen, **cfg); cfg defaults in DEFAULT_CFG
    out = net(obs_to_tensors([encode(battle)], "cpu"))
    out["logits"], out["value"], out["win_logit"], out["aux"]["item"], ...
    save(net, "snap.pt", step=123); net = load("snap.pt")

Tokens: field (1) | our mons 0-5 (= switch actions 0-5) | opponent mons 6-11 | our active's moves 0-3 (= move actions
6-9) | (gen9) the same moves with terastallization (= actions 22-25). Every id embedding = learned id vector (std 0.02)
+ MLP(static features of the id, `porygon.v2.tables`). Move vectors for all 12 x 24 move slots mix the move entity, its
type, dynamic numbers and (`use_damage`) damage summaries vs the other side (`porygon.v2.damage`); mon tokens pool
their move vectors. A pre-RMSNorm transformer with per-token-type FFNs (field / own mons / opponent mons / actions;
Jaxcalibur's "position-based mixture of experts") and shared attention with a per-head additive bias from mon-pair
damage / speed features. Heads: pointer policy, win-probability value (its input gradient into the trunk scaled by
`value_trunk_grad`: MissingNo.Zero's fix for a value gradient ~50x the policy's; default 1.0 because the v1 600M
net measured value/policy ~0.09, the trainer logs the ratio), and aux heads predicting the opponent's hidden set, unseen species and next
action.

ID dropout: when the observation carries a per-sample int64 `drop_seed` (nonzero) and cfg `id_dropout` > 0, each
species / move / item / ability id is replaced by UNK in the *id-embedding* lookup only (the static-feature path keeps
the true id) with probability `id_dropout`, decided by a hash of (seed, position). The trainer stores the seed with
the transition, so the acting forward and every PPO-update forward see the same dropped ids (exact importance
ratios); play / evaluation pass no seed and see every id.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from porygon.v2.spec import (
    FIELD_NUM, FIELD_NUM_DIM, KA, KI, KM, MON_NUM, MON_NUM_DIM, MOVE_NUM_DIM, N_MONS, N_OPP_CLASSES, N_TYPES, PAD,
    SIDE_DIM, SPEC_VERSION, TERA_OFFSET, UNK, VOCAB_SIZES, fields, n_actions,
)

COL = {n: i for i, n in enumerate(MON_NUM)}
FCOL = {n: i for i, n in enumerate(FIELD_NUM)}
ENTITY_KINDS = ("species", "moves", "items", "abilities")

DEFAULT_CFG = dict(
    d_model=256, n_layers=6, n_heads=8, ffn_mult=2,
    d_move=128, d_species=128, d_move_emb=96, d_item=64, d_ability=64, d_type=32,
    pair_hidden=32, value_hidden=256, value_trunk_grad=1.0, id_dropout=0.1,
    use_damage=True, use_tables=True, qk_norm=True, mask_fainted=True, trim_moves=True,
)

# Modules outside the shared trunk (for the value/policy trunk-gradient ratio and nothing else).
HEAD_PREFIXES = ("policy_head.", "value_head.", "aux_")


# ---------------------------------------------------------------- small pieces

class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity forward; backward multiplies the gradient by `scale`."""
    return x if scale == 1.0 else _GradScale.apply(x, scale)


def mlp(d_in: int, d_hidden: int, d_out: int, zero_last: bool = False) -> nn.Sequential:
    m = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))
    if zero_last:
        nn.init.zeros_(m[-1].weight)
        nn.init.zeros_(m[-1].bias)
    return m


def _hash_uniform(seed: torch.Tensor, n: int, salt: int) -> torch.Tensor:
    """[B, n] pseudo-uniform floats in [0, 1) from int64 seeds [B] (integer mixing; int64 wraps around)."""
    pos = torch.arange(n, device=seed.device, dtype=torch.int64) + salt * 1_000_003
    x = seed.view(-1, 1) * 0x9E3779B1 + pos * 0x85EBCA77
    for mult in (0x2C1B3C6D, 0x297A2D39, 0x1B873593):
        x = x ^ (x >> 15)
        x = x * mult
    x = x ^ (x >> 13)
    return ((x >> 8) & 0xFFFFFF).to(torch.float32) / 16777216.0


class Entity(nn.Module):
    """id embedding (std 0.02) + MLP(static feature row of the id); zero for PAD."""

    def __init__(self, vocab: int, dim: int, table: torch.Tensor | None):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=PAD)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()
        if table is not None:
            self.register_buffer("table", torch.as_tensor(table, dtype=torch.float32).clone())
            self.feat = mlp(self.table.shape[1], dim, dim)
        else:
            self.table = None

    def all_features(self) -> torch.Tensor | None:
        """MLP over every row once per forward (cheaper than per occurrence for batches beyond ~10). Without
        autograd (acting, play) the result is cached until a parameter changes (optimizer step, load)."""
        if self.table is None:
            return None
        if torch.is_grad_enabled():
            return self.feat(self.table)
        key = (self.table.device, torch.get_autocast_dtype(self.table.device.type)
               if torch.is_autocast_enabled(self.table.device.type) else None,
               tuple(p._version for p in self.feat.parameters()), tuple(id(p) for p in self.feat.parameters()))
        if getattr(self, "_cache_key", None) != key:
            self._cache, self._cache_key = self.feat(self.table), key
        return self._cache

    def forward(self, ids: torch.Tensor, feats: torch.Tensor | None, drop: torch.Tensor | None) -> torch.Tensor:
        lookup = ids if drop is None else torch.where(drop & (ids > UNK), torch.full_like(ids, UNK), ids)
        x = self.emb(lookup)
        if feats is not None:  # F.embedding, not feats[ids]: advanced-index backward was 25% of the update's GPU time
            x = x + F.embedding(ids, feats)
        return x * (ids != PAD).unsqueeze(-1).to(x.dtype)


class Block(nn.Module):
    """Pre-RMSNorm transformer block: shared attention (SDPA, optional additive mask/bias), one FFN per token group.
    `qk_norm`: RMSNorm (learnable per-dim scale) on each head's queries and keys, which bounds the attention logits
    at init (a v1 head grew into a 29-nat one-hot selector)."""

    def __init__(self, d: int, n_heads: int, ffn_mult: int, groups: list[tuple[int, int]], qk_norm: bool = True):
        super().__init__()
        self.n_heads, self.groups = n_heads, groups
        self.q_norm = nn.RMSNorm(d // n_heads) if qk_norm else None
        self.k_norm = nn.RMSNorm(d // n_heads) if qk_norm else None
        self.norm1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.RMSNorm(d)
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))
                                  for _ in groups])

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=None if bias is None else bias.to(q.dtype))
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        h = self.norm2(x)
        return x + torch.cat([f(h[:, s:e]) for f, (s, e) in zip(self.ffn, self.groups)], 1)


def _damage_module(gen: int):
    from porygon.v2.damage import N_MOVE_VS, N_PAIR, DamageFeatures
    return DamageFeatures(gen), N_MOVE_VS, N_PAIR


def _static_tables(gen: int) -> dict[str, np.ndarray]:
    from porygon.v2.tables import static_features
    return static_features(gen)


# ---------------------------------------------------------------- the network

class Net(nn.Module):
    def __init__(self, gen: int, tables: dict | None = None, **cfg):
        """`tables`: kind -> [vocab, F] static features to use instead of `porygon.v2.tables` (load() passes the
        checkpoint's own copy, so a snapshot keeps the features it was trained with)."""
        super().__init__()
        unknown = set(cfg) - set(DEFAULT_CFG)
        assert not unknown, f"unknown Net config keys {unknown}"
        self.gen = int(gen)
        self.cfg = {**DEFAULT_CFG, **cfg}
        c = self.cfg
        self.n_actions = n_actions(self.gen)
        self.tera = self.gen >= 9
        D, dm, dt = c["d_model"], c["d_move"], c["d_type"]

        if c["use_tables"] and tables is None:
            tables = _static_tables(self.gen)
        if not c["use_tables"]:
            tables = None
        dims = {"species": c["d_species"], "moves": c["d_move_emb"], "items": c["d_item"], "abilities": c["d_ability"]}
        for kind in ENTITY_KINDS:
            setattr(self, f"ent_{kind}", Entity(VOCAB_SIZES[kind], dims[kind],
                                                None if tables is None else tables[kind]))
        self.type_emb = nn.Embedding(N_TYPES, dt, padding_idx=0)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        with torch.no_grad():
            self.type_emb.weight[0].zero_()

        if c["use_damage"]:
            self.damage, n_vs, n_pair = _damage_module(self.gen)
        else:
            self.damage, n_vs, n_pair = None, 0, 0
        self.n_vs = n_vs
        dmg_in = 3 * n_vs  # vs the opposing active, max and mean over the alive opposing mons

        move_in = c["d_move_emb"] + dt + MOVE_NUM_DIM + 1 + dmg_in
        self.move_mlp = mlp(move_in, dm, dm)
        mon_in = (c["d_species"] + 2 * dt + c["d_item"] + c["d_ability"] + 2 * dm + MON_NUM_DIM
                  + 6 + 2 + 7 + (2 if n_vs else 0))
        self.mon_mlp = mlp(mon_in, D, D)
        self.absent = nn.Parameter(torch.randn(2, D) * 0.02)  # 0: own empty slot, 1: opponent mon not seen yet
        self.field_mlp = mlp(FIELD_NUM_DIM + 2 * SIDE_DIM, D, D)
        self.move_token = mlp(dm, D, D)
        if self.tera:
            self.tera_token = mlp(dm + dt, D, D)
        self.n_tokens = 1 + N_MONS + 4 + (4 if self.tera else 0)
        token_type = [0] + [1] * 6 + [2] * 6 + [3] * 4 + ([4] * 4 if self.tera else [])
        self.register_buffer("token_type", torch.tensor(token_type), persistent=False)
        self.token_type_emb = nn.Embedding(5, D)
        nn.init.normal_(self.token_type_emb.weight, std=0.02)
        groups = [(0, 1), (1, 7), (7, 13), (13, self.n_tokens)]
        self.blocks = nn.ModuleList([Block(D, c["n_heads"], c["ffn_mult"], groups, c["qk_norm"])
                                     for _ in range(c["n_layers"])])
        self.norm = nn.RMSNorm(D)
        if n_pair:
            self.pair_mlp = mlp(n_pair, c["pair_hidden"], c["n_layers"] * c["n_heads"], zero_last=True)

        # heads
        self.policy_head = mlp(4 * D, D, 1, zero_last=True)  # [candidate, field, our active, their active]
        vh = c["value_hidden"]
        self.value_head = nn.Sequential(nn.Linear(2 * D, vh), nn.GELU(), nn.Linear(vh, vh), nn.GELU(),
                                        nn.Linear(vh, 1))
        self.aux_item = nn.Linear(D, VOCAB_SIZES["items"])
        self.aux_ability = nn.Linear(D, VOCAB_SIZES["abilities"])
        self.aux_tera = nn.Linear(D, N_TYPES)
        self.aux_moves = mlp(dm + D, D, 1)
        self.aux_unseen = nn.Linear(D, VOCAB_SIZES["species"])
        nn.init.constant_(self.aux_unseen.bias, -5.5)  # ~ 6 of 1520 species positive: start near the base rate
        self.aux_opp_switch = mlp(4 * D, D, 1)  # [candidate, field, their active, our active]
        self.aux_opp_unseen = nn.Parameter(torch.randn(D) * 0.02)
        self.aux_opp_move = mlp(dm + 3 * D, D, 1)
        self.aux_opp_tera = nn.Parameter(torch.randn(dm) * 0.02)

    # ------------------------------------------------------------ helpers

    def trunk_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if not n.startswith(HEAD_PREFIXES)]

    def param_groups(self, weight_decay: float) -> list[dict]:
        """AdamW groups: no decay on embeddings, norms, biases and learned vectors."""
        no_decay = {id(m.weight) for m in self.modules() if isinstance(m, (nn.Embedding, nn.RMSNorm))}
        decay, rest = [], []
        for p in self.parameters():
            (decay if p.ndim >= 2 and id(p) not in no_decay else rest).append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]

    def _drop_masks(self, obs: dict, shapes: dict) -> dict:
        p = self.cfg["id_dropout"]
        seed = obs.get("drop_seed")
        if seed is None or p <= 0:
            return {k: None for k in shapes}
        seed = seed.long()
        on = (seed != 0).view(-1, 1)
        out = {}
        for salt, (k, shape) in enumerate(shapes.items()):
            n = int(np.prod(shape))
            out[k] = ((_hash_uniform(seed, n, salt) < p) & on).view(-1, *shape)
        return out

    @staticmethod
    def _other_side(x: torch.Tensor) -> torch.Tensor:
        """[B, 12] per-mon weights -> [B, 12, 6]: for each mon, the weights of the 6 mons of the other side."""
        b = x.shape[0]
        return torch.cat([x[:, None, 6:].expand(b, 6, 6), x[:, None, :6].expand(b, 6, 6)], 1)

    @staticmethod
    def _summary(vs: torch.Tensor, w_act: torch.Tensor, w_alive: torch.Tensor) -> torch.Tensor:
        """vs [..., 6, F] features against the 6 mons of the other side -> [..., 3F]: vs the active, max and mean
        over the alive ones. w_act / w_alive [..., 6] broadcast over the leading dims."""
        act = (vs * w_act.unsqueeze(-1)).sum(-2)
        alive = w_alive.unsqueeze(-1)
        mx = (vs * alive).amax(-2)  # features are >= 0: dead / absent mons contribute 0
        mean = (vs * alive).sum(-2) / alive.sum(-2).clamp(min=1.0)
        return torch.cat([act, mx, mean], -1)

    # ------------------------------------------------------------ forward

    def damage_features(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        """porygon.v2.damage features in float32 (no autocast, no grad: they are a function of the observation)."""
        if self.damage is None:
            return None
        dev = obs["mon_num"].device.type
        with torch.no_grad(), torch.autocast(dev, enabled=False):
            fn = getattr(self, "_damage_compiled", None)
            if fn is not None:
                try:
                    return {k: v.float() for k, v in fn(obs).items()}
                except Exception as e:  # compiler missing / unsupported op: eager from now on
                    print(f"warning: compiled damage layer failed ({e!r:.300}); using eager", flush=True)
                    object.__setattr__(self, "_damage_compiled", None)
            return {k: v.float() for k, v in self.damage(obs).items()}

    def compile_damage(self, **kwargs) -> None:
        """torch.compile the damage layer (many small ops: launch-bound on GPU). The compiled instance does not trim
        unused move slots (`trim` reads a count on the host, which would break the graph / recompile per call)."""
        if self.damage is None:
            return
        layer = self.damage
        try:
            from porygon.v2.damage import DamageFeatures
            layer = DamageFeatures(self.gen, trim=False).to(next(self.damage.buffers()).device)
        except TypeError:  # an older damage layer without `trim`
            pass
        object.__setattr__(self, "_damage_layer_untrimmed", layer)
        object.__setattr__(self, "_damage_compiled", torch.compile(layer, dynamic=True, **kwargs))

    def forward(self, obs: dict[str, torch.Tensor], dmg: dict[str, torch.Tensor] | None = None,
                aux: bool = True) -> dict:
        """`dmg`: precomputed `damage_features(obs)` (the trainer computes them once per rollout); `aux=False`
        skips the aux heads (out["aux"] is None)."""
        c = self.cfg
        ids = {k: obs[k].long() for k in ("species", "moves", "item", "ability", "types", "move_type")}
        mon_num = obs["mon_num"].float()
        b = mon_num.shape[0]
        present = mon_num[..., COL["present"]]
        active = mon_num[..., COL["active"]] * present
        alive = present * (1.0 - mon_num[..., COL["fainted"]])
        w_act, w_alive = self._other_side(active), self._other_side(alive)  # [B, 12, 6]

        if self.damage is None:
            dmg = None
        elif dmg is None:
            dmg = self.damage_features(obs)

        drop = self._drop_masks(obs, {"species": (N_MONS,), "moves": (N_MONS, KM), "items": (N_MONS, KI),
                                      "abilities": (N_MONS, KA)})
        feats = {k: getattr(self, f"ent_{k}").all_features() for k in ENTITY_KINDS}

        # move vectors [B, 12, 24, d_move]; computed only for the first k slots any mon of the batch occupies (slots
        # are filled in order, real data uses <= 10 of 24) and zero-padded back: identical output, ~2.5x less work
        valid_move = ids["moves"] != PAD
        k = KM
        if c.get("trim_moves", True):
            used = valid_move.flatten(0, 1).any(0)
            k = max(4, int(used.nonzero().max().item()) + 1) if bool(used.any()) else 4
        mdrop = None if drop["moves"] is None else drop["moves"][:, :, :k]
        mv_emb = self.ent_moves(ids["moves"][:, :, :k], feats["moves"], mdrop)
        move_parts = [mv_emb, self.type_emb(ids["move_type"][:, :, :k]), obs["move_num"][:, :, :k].float(),
                      obs["move_p"][:, :, :k].float().unsqueeze(-1)]
        if dmg is not None:
            move_parts.append(self._summary(dmg["move_vs"][:, :, :k].float(), w_act[:, :, None], w_alive[:, :, None]))
        move_in = torch.cat(move_parts, -1)
        mvec = self.move_mlp(move_in) * valid_move[:, :, :k].unsqueeze(-1).to(move_in.dtype)
        if k < KM:
            mvec = F.pad(mvec, (0, 0, 0, KM - k))

        # mon tokens
        move_p = obs["move_p"].float().unsqueeze(-1)
        mean_moves = (move_p * mvec).sum(-2) / move_p.sum(-2).clamp(min=1e-6)
        max_moves = mvec.masked_fill(~valid_move.unsqueeze(-1), -1e4).amax(-2)
        max_moves = torch.where(valid_move.any(-1, keepdim=True), max_moves, torch.zeros_like(max_moves))
        item = (obs["item_p"].float().unsqueeze(-1) * self.ent_items(ids["item"], feats["items"], drop["items"])).sum(-2)
        ability = (obs["ability_p"].float().unsqueeze(-1)
                   * self.ent_abilities(ids["ability"], feats["abilities"], drop["abilities"])).sum(-2)
        hp = obs["hp"].float()
        mon_parts = [
            self.ent_species(ids["species"], feats["species"], drop["species"]),
            self.type_emb(ids["types"]).sum(-2),
            obs["tera_p"].float() @ self.type_emb.weight,
            item, ability, mean_moves, max_moves, mon_num,
            torch.log1p(obs["stats"].float().clamp(min=0)) / 6,
            torch.stack([hp[..., 0] / hp[..., 1].clamp(min=1.0), hp[..., 1] / 500], -1),
            obs["boosts"].float() / 6,
        ]
        if dmg is not None:
            speed = dmg["speed"].float()
            mon_parts.append(torch.stack([(speed * w_act).sum(-1),
                                          (speed * w_alive).sum(-1) / w_alive.sum(-1).clamp(min=1.0)], -1))
        mons = self.mon_mlp(torch.cat(mon_parts, -1))
        absent = torch.cat([self.absent[:1].expand(6, -1), self.absent[1:].expand(6, -1)], 0)
        mons = torch.where(present.unsqueeze(-1) > 0, mons, absent.to(mons.dtype))

        field = self.field_mlp(torch.cat([obs["field_num"].float(), obs["side_num"].float().flatten(1)], -1))

        # action tokens: our active's move slots 0-3 (and their tera versions)
        own_act = (active * alive)[:, :6]  # a fainted active (force switch): empty action tokens
        act_mvec = torch.einsum("bi,bijd->bjd", own_act, mvec[:, :6, :4])
        tokens = [field.unsqueeze(1), mons, self.move_token(act_mvec)]
        if self.tera:
            tin = torch.einsum("bi,bijd->bjd", own_act, move_in[:, :6, :4])
            if dmg is not None:
                tvs = self._summary(dmg["active_tera_vs"].float(), w_act[:, None, 0], w_alive[:, None, 0])
                tin = torch.cat([tin[..., :-tvs.shape[-1]], tvs], -1)
            act_valid = torch.einsum("bi,bij->bj", own_act, valid_move[:, :6, :4].to(own_act.dtype)) > 0
            tvec = self.move_mlp(tin) * act_valid.unsqueeze(-1).to(tin.dtype)
            tera_type = torch.einsum("bi,bit->bt", own_act, obs["tera_p"].float()[:, :6]) @ self.type_emb.weight
            tokens.append(self.tera_token(torch.cat([tvec, tera_type.unsqueeze(1).expand(-1, 4, -1)], -1)))
        x = torch.cat(tokens, 1) + self.token_type_emb(self.token_type)

        bias = None
        if c["mask_fainted"]:  # nobody attends to fainted mons (their own tokens still update); unseen slots stay
            fainted = (present * mon_num[..., COL["fainted"]]) > 0
            keymask = torch.zeros(b, self.n_tokens, device=x.device, dtype=torch.float32)
            keymask[:, 1:1 + N_MONS] = keymask[:, 1:1 + N_MONS].masked_fill(fainted, float("-inf"))
            keymask = keymask[:, None, None, :]  # [B, 1, 1, T]
        else:
            keymask = None
        if dmg is not None:
            pres = present > 0
            pb = self.pair_mlp(dmg["pair"].float())  # [B, 12, 12, L*H]
            pb = pb * (pres[:, :, None] & pres[:, None, :]).unsqueeze(-1).to(pb.dtype)
            pb = pb.view(b, N_MONS, N_MONS, c["n_layers"], c["n_heads"]).permute(3, 0, 4, 1, 2)
            t = self.n_tokens
            bias = F.pad(pb, (1, t - 1 - N_MONS, 1, t - 1 - N_MONS))  # [L, B, H, T, T]
        for i, blk in enumerate(self.blocks):
            m = None if bias is None else bias[i]
            if keymask is not None:
                m = keymask if m is None else m + keymask
            x = blk(x, m)
        x = self.norm(x)

        field_t, own_t, opp_t = x[:, 0], x[:, 1:7], x[:, 7:13]
        mons_t = x[:, 1:13]

        live_act = (active * alive).to(x.dtype)  # a fainted active (force switch) gives no context: zeros
        own_act_tok = torch.einsum("bi,bid->bd", live_act[:, :6], own_t)
        opp_act_tok = torch.einsum("bk,bkd->bd", live_act[:, 6:], opp_t)

        # policy: pointer logits from [candidate, field, our active, their active]
        cand = [own_t, x[:, 13:17]] + ([x[:, 17:21]] if self.tera else [])
        cand = torch.cat(cand, 1)
        ctx = torch.cat([field_t, own_act_tok, opp_act_tok], -1).unsqueeze(1).expand(-1, cand.shape[1], -1)
        scores = self.policy_head(torch.cat([cand, ctx], -1)).squeeze(-1).float()
        logits = torch.zeros(b, self.n_actions, device=scores.device, dtype=torch.float32)
        logits[:, :10] = scores[:, :10]
        if self.tera:
            logits[:, TERA_OFFSET:TERA_OFFSET + 4] = scores[:, 10:14]
        logits = logits.masked_fill(obs["action_mask"] == 0, torch.finfo(torch.float32).min)

        # value: win probability; only value_trunk_grad of its gradient reaches the trunk
        v_in = grad_scale(torch.cat([field_t, mons_t.mean(1)], -1), c["value_trunk_grad"])
        win_logit = self.value_head(v_in).squeeze(-1).float()

        out = {"logits": logits, "value": torch.tanh(win_logit / 2), "win_logit": win_logit, "aux": None}
        if not aux:
            return out

        # aux heads
        # per-move-slot heads on the occupied slots only (k, as for the move vectors), zero-padded to KM: the padded
        # slots are empty for every mon, so their targets are -1 and their opponent-action classes masked
        opp_mvec = mvec[:, 6:, :k]
        aux_moves = self.aux_moves(torch.cat([opp_mvec, opp_t.unsqueeze(2).expand(-1, -1, k, -1)], -1)).squeeze(-1)
        opp_act = active[:, 6:]
        opp_act_mv = torch.einsum("bk,bkjd->bjd", opp_act.to(opp_mvec.dtype), opp_mvec)
        # the opponent's view: [candidate, field, its active, our active]
        octx = torch.cat([field_t, opp_act_tok, own_act_tok], -1).unsqueeze(1)
        sw = self.aux_opp_switch(torch.cat([torch.cat([opp_t, self.aux_opp_unseen.expand(b, 1, -1).to(opp_t.dtype)], 1),
                                            octx.expand(-1, 7, -1)], -1)).squeeze(-1)
        ctx = octx.expand(-1, k, -1)
        mv = self.aux_opp_move(torch.cat([opp_act_mv, ctx], -1)).squeeze(-1)
        tmv = self.aux_opp_move(torch.cat([opp_act_mv + self.aux_opp_tera.to(opp_act_mv.dtype), ctx], -1)).squeeze(-1)
        if k < KM:
            aux_moves, mv, tmv = (F.pad(t_, (0, KM - k)) for t_ in (aux_moves, mv, tmv))
        opp_raw = torch.cat([sw, mv, tmv], 1).float()
        # classes that can happen: switch to a seen, alive, inactive mon; an unseen mon; a move the active may have
        fnum = obs["field_num"].float()
        sw_ok = (alive[:, 6:] * (1.0 - active[:, 6:])) > 0
        unseen_ok = fnum[:, FCOL["opp_unseen"]:FCOL["opp_unseen"] + 1] > 0
        mv_ok = (torch.einsum("bk,bkj->bj", opp_act, valid_move[:, 6:].float()) > 0)
        tera_ok = mv_ok & ((fnum[:, FCOL["opp_used_tera"]:FCOL["opp_used_tera"] + 1] == 0) if self.tera
                           else torch.zeros_like(mv_ok))
        opp_mask = torch.cat([sw_ok, unseen_ok, mv_ok, tera_ok], 1)
        aux = {
            "item": self.aux_item(opp_t).float(),
            "ability": self.aux_ability(opp_t).float(),
            "tera": self.aux_tera(opp_t).float(),
            "moves": aux_moves.float(),
            "unseen": self.aux_unseen(field_t).float(),
            "opp_action": opp_raw.masked_fill(~opp_mask, torch.finfo(torch.float32).min),
            "opp_action_raw": opp_raw,
            "opp_action_mask": opp_mask,
        }
        assert opp_raw.shape[1] == N_OPP_CLASSES
        out["aux"] = aux
        return out


# ---------------------------------------------------------------- io

def obs_to_tensors(obs_list: list[dict[str, np.ndarray]], device) -> dict[str, torch.Tensor]:
    """Stack observations (any extra keys too: aux targets, drop_seed); integer arrays -> int64."""
    out = {}
    for k in obs_list[0]:
        a = np.stack([o[k] for o in obs_list])
        t = torch.from_numpy(a)
        out[k] = (t.long() if a.dtype.kind in "iu" else t).to(device, non_blocking=True)
    return out


def batch_to_tensors(batch: dict[str, np.ndarray], device, keys=None) -> dict[str, torch.Tensor]:
    """Already-stacked arrays -> tensors (integers -> int64)."""
    out = {}
    for k in (keys or batch):
        t = torch.from_numpy(np.ascontiguousarray(batch[k]))
        out[k] = (t.long() if batch[k].dtype.kind in "iu" else t).to(device, non_blocking=True)
    return out


def model_keys(gen: int) -> list[str]:
    return list(fields(gen))


def save(net: Net, path, **extra) -> Path:
    """Atomic write of {format, spec, gen, cfg, state_dict, **extra}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": "porygon.v2", "spec": SPEC_VERSION, "gen": net.gen, "cfg": dict(net.cfg),
                "state_dict": net.state_dict(), **extra}, tmp)
    os.replace(tmp, path)
    return path


def is_v2_checkpoint(path) -> bool:
    if Path(path).suffix != ".pt":
        return False
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return isinstance(ck, dict) and ck.get("format") == "porygon.v2"


def net_from_checkpoint(ck: dict, device="cpu") -> Net:
    sd = ck["state_dict"]
    tables = {k: sd[f"ent_{k}.table"] for k in ENTITY_KINDS if f"ent_{k}.table" in sd} or None
    net = Net(ck["gen"], tables=tables, **ck["cfg"])
    missing, unexpected = net.load_state_dict(sd, strict=False)
    bad = [k for k in missing + unexpected if not k.startswith("damage.")]
    if bad:
        raise RuntimeError(f"checkpoint does not match Net: missing/unexpected {bad[:10]}")
    if missing or unexpected:
        print(f"warning: damage-layer buffers differ from the checkpoint ({len(missing)} missing, "
              f"{len(unexpected)} unexpected); using the current damage layer")
    return net.to(device)


def load(path, device="cpu") -> Net:
    """Eval-mode Net from a `save` file or a trainer checkpoint (both hold gen, cfg and state_dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return net_from_checkpoint(ck, device).eval()
