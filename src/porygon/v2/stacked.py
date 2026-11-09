"""Stacked pool-opponent forward (owner: netv2): the policy logits of several v2 nets with the same architecture and
different weights in one pass, for the trainer's pool opponents (`Trainer._opp_act_many`, `--stacked-opp`).

    stack = OppStack(nets[0])                       # one per architecture (`signature`)
    logits, refills = stack.logits(nets, obs_list)  # obs_list[i]: net i's rows (dict of numpy arrays, Packer layout)

Why: the fastenv collector acts 3-4 distinct pool opponents per step with ~40 rows each, one forward each; at that
size a forward is launch-bound (~17 ms on the A30 vs ~29 ms for the learner's ~850 rows), so N opponents cost about N
times the kernel launches of one forward. Here the N nets' rows are folded into one group-major batch: rows of net g
at [g * bm, (g + 1) * bm), each group padded to bm = the largest group with copies of its last row. Every
parameter-free op of `Net.forward` runs once on the folded batch, unchanged; every op with weights runs once with the
N nets' weights stacked: nn.Linear -> one baddbmm, RMSNorm -> norm then a per-group scale, embeddings -> one lookup
into the N stacked tables with per-group id offsets. So one call launches about as many kernels as one forward.

Not torch.func.vmap over `Net.forward`: it fails there (the move-slot trim is a host read; with the trim off and the
damage features precomputed, the in-place writes into fresh tensors, `keymask[...] = ...` and `logits[:, :10] =
...`, cannot take batched values; `Entity.all_features` caches by parameter version), so vmap would need a rewritten
forward anyway, and this one is written for the folded batch directly.

Exactness (tests/test_v2_stacked_opp.py): the logits equal each net's own `net(obs, aux=False)["logits"]` up to
float rounding of the batched GEMMs (fp32: ~1e-6). Shared across the folded batch and computed once: the damage
features (parameter-free, a function of the observation; `nets[0].damage_features`, i.e. the compiled untrimmed
layer when the trainer compiles) and the move-slot trim k (from the numpy observations, no device sync). A net whose
own trim would be smaller computes some extra empty slots, which are exact zeros, so this changes only the shapes of
its GEMMs. No op mixes rows (attention, norms and masks are per row), so padding rows never touch real ones. No id
dropout (opponents pass drop_seed 0 = no dropout) and no value or aux heads (only the logits are used).

What is stacked (`OppStack.shapes`): the policy-path parameters (Linear weights stored transposed, [in, out]), each
entity's id embedding + static-feature MLP output as one table (`emb.weight + all_features()`, the PAD row zero: the
same sum Net computes per lookup, then masks), the absent-mon vectors and the token-type embeddings; ~17 MB per net
for the production gen9 config (d_model 256, 4 layers). Memory: one buffer of `capacity` slots per architecture; each
slot remembers which net (by identity) it holds, so the same nets acted again copy nothing, and only the slots of
nets new to the stack are refilled (~130 tensor copies each). `release` drops the slots of nets that left the trainer's
snapshot cache.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from porygon.v2.model import COL, ENTITY_KINDS, Net, batch_to_tensors, model_keys
from porygon.v2.spec import KM, N_MONS, PAD, TERA_OFFSET

# Net cfg keys without effect on an opponent's logits (no id-dropout seed, no value head, no gradients; without the
# static tables an entity's table is its id embedding alone)
IGNORED_CFG = ("id_dropout", "value_trunk_grad", "value_hidden", "use_tables")
# parameters of the policy path (everything but the entity embeddings / feature MLPs, folded into one table per
# kind, the token-type and absent vectors, stacked as derived tensors, and the value / aux heads, not used)
PREFIXES = ("type_emb.", "move_mlp.", "mon_mlp.", "field_mlp.", "move_token.", "tera_token.", "blocks.", "norm.",
            "pair_mlp.", "policy_head.")
TABLES = tuple(f"ent_{k}" for k in ENTITY_KINDS) + ("type_emb.weight",)  # looked up with per-group id offsets


def _policy_params(net: Net) -> dict[str, torch.Tensor]:
    return {n: p for n, p in net.named_parameters() if n.startswith(PREFIXES)}


def signature(net: Net) -> tuple:
    """Nets with equal signatures can share an `OppStack` (gen, the cfg keys that matter, every stacked shape, dtype
    and device). Cached on the net (snapshot opponents are frozen)."""
    sig = net.__dict__.get("_stack_signature")
    if sig is None:
        sig = (net.gen, net.n_actions, net.n_tokens,
               tuple(sorted((k, v) for k, v in net.cfg.items() if k not in IGNORED_CFG)),
               tuple((n, tuple(p.shape), p.dtype, str(p.device)) for n, p in _policy_params(net).items()),
               tuple(tuple(getattr(net, f"ent_{k}").emb.weight.shape) for k in ENTITY_KINDS))
        net.__dict__["_stack_signature"] = sig
    return sig


# ---------------------------------------------------------------- grouped ops on the folded batch [n * bm, ...]

def _lin(x: torch.Tensor, wt: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """nn.Linear per group: x [n * bm, ..., in], wt [n, in, out] (transposed weights), b [n, out]."""
    y = torch.baddbmm(b.unsqueeze(1), x.reshape(wt.shape[0], -1, x.shape[-1]), wt)
    return y.view(*x.shape[:-1], wt.shape[-1])


def _gmm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ w per group: x [n * bm, ..., a], w [n, a, c]."""
    return torch.bmm(x.reshape(w.shape[0], -1, x.shape[-1]), w).view(*x.shape[:-1], w.shape[-1])


def _rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """nn.RMSNorm(d) (eps=None) per group over the last dim: x [n * bm, ..., d], w [n, d]."""
    n = w.shape[0]
    y = F.rms_norm(x, (x.shape[-1],))
    return (y.unflatten(0, (n, -1)) * w.view(n, *[1] * (x.dim() - 1), w.shape[-1])).flatten(0, 1)


def _emb(table: torch.Tensor, off: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Embedding per group: table [n, V, d] (contiguous), off [n] = g * V, ids [n * bm, ...] raw ids."""
    n = table.shape[0]
    idx = (ids.unflatten(0, (n, -1)) + off.view(n, *[1] * ids.dim())).flatten(0, 1)
    return F.embedding(idx, table.flatten(0, 1))


class OppStack:
    def __init__(self, template: Net):
        """Stacked weights for nets with `signature(template)`; slots are allocated on demand (`bind`)."""
        self.sig = signature(template)
        c = template.cfg
        self.gen, self.tera, self.n_actions, self.n_tokens = (template.gen, template.tera, template.n_actions,
                                                              template.n_tokens)
        self.n_heads, self.n_layers, self.qk_norm = c["n_heads"], c["n_layers"], c["qk_norm"]
        self.mask_fainted, self.trim = c["mask_fainted"], c.get("trim_moves", True)
        self.use_damage = template.damage is not None
        self.groups = list(template.blocks[0].groups)
        self.keys = model_keys(self.gen)
        self.device = template.absent.device
        with torch.no_grad():
            self.shapes = {k: (tuple(v.shape), v.dtype) for k, v in self._sources(template).items()}
        self.capacity, self.buf, self.slots, self.off = 0, {}, [], {}
        self.slot_bytes = sum(int(np.prod(s)) * torch.empty((), dtype=d).element_size()
                              for s, d in self.shapes.values())  # stacked weights per net

    @staticmethod
    def _sources(net: Net) -> dict[str, torch.Tensor]:
        """One net's tensors in the stacked layout (Linear weights transposed; derived tables)."""
        src = {n: (p.t() if p.dim() == 2 and n != "type_emb.weight" else p) for n, p in _policy_params(net).items()}
        for kind in ENTITY_KINDS:
            ent = getattr(net, f"ent_{kind}")
            feats = ent.all_features()
            src[f"ent_{kind}"] = ent.emb.weight if feats is None else ent.emb.weight + feats
        src["absent12"] = torch.cat([net.absent[:1].expand(6, -1), net.absent[1:].expand(6, -1)], 0)
        src["tok"] = net.token_type_emb(net.token_type)
        return src

    def _grow(self, n: int) -> None:
        if n <= self.capacity:
            return
        new = {k: torch.empty((n, *s), dtype=d, device=self.device) for k, (s, d) in self.shapes.items()}
        for k, b in self.buf.items():
            new[k][:self.capacity].copy_(b)
        self.buf, self.slots = new, self.slots + [None] * (n - self.capacity)
        self.capacity = n
        self.off = {k: torch.arange(n, device=self.device) * self.shapes[k][0][0] for k in TABLES}

    def _fill(self, i: int, net: Net) -> None:
        for k, v in self._sources(net).items():
            self.buf[k][i].copy_(v)
        for kind in ENTITY_KINDS:  # Net masks PAD lookups to zero
            self.buf[f"ent_{kind}"][i, PAD].zero_()
        self.slots[i] = net

    @torch.no_grad()
    def bind(self, nets: list[Net]) -> tuple[list[int], int]:
        """Slots 0..n-1 for these n nets: a net already in one of them keeps it, the others are copied in. Returns
        the slot of each net (a permutation of range(n)) and the number of slots refilled."""
        n = len(nets)
        assert all(signature(net) == self.sig for net in nets), "nets of another architecture"
        self._grow(n)
        slot_of, taken = [None] * n, [False] * n
        for p, net in enumerate(nets):
            for i in range(n):
                if not taken[i] and self.slots[i] is net:
                    slot_of[p], taken[i] = i, True
                    break
        free = (i for i in range(n) if not taken[i])
        refills = 0
        for p, net in enumerate(nets):
            if slot_of[p] is None:
                slot_of[p] = next(free)
                self._fill(slot_of[p], net)
                refills += 1
        return slot_of, refills

    def release(self, live: set[int]) -> bool:
        """Forget slots whose net is not in `live` (ids of the nets still cached), so evicted nets can be freed;
        True if a slot still holds a live net."""
        self.slots = [s if s is not None and id(s) in live else None for s in self.slots]
        return any(s is not None for s in self.slots)

    @torch.no_grad()
    def logits(self, nets: list[Net], obs_list: list[dict[str, np.ndarray]]) -> tuple[list[torch.Tensor], int]:
        """Float32 CPU logits [len(obs_list[i]), n_actions] of nets[i] on obs_list[i] (numpy, Packer dtypes; extra
        keys such as drop_seed are ignored), and the number of slots refilled. Call inside the same autocast
        context as the per-net forward."""
        n = len(nets)
        slot_of, refills = self.bind(nets)
        lens = [len(ob["action_mask"]) for ob in obs_list]
        bm = max(lens)
        folded = {}
        for key in self.keys:
            a0 = obs_list[0][key]
            arr = np.empty((n * bm, *a0.shape[1:]), a0.dtype)
            for s, ob, m in zip(slot_of, obs_list, lens):
                arr[s * bm:s * bm + m] = ob[key]
                arr[s * bm + m:(s + 1) * bm] = ob[key][m - 1]  # padding: copies of the group's last row
            folded[key] = arr
        k = KM
        if self.trim:  # Net.forward's trim on the folded batch (= the union of the groups' rows), read on the host
            used = np.flatnonzero((folded["moves"] != PAD).any((0, 1)))
            k = max(4, int(used.max()) + 1) if len(used) else 4
        t = batch_to_tensors(folded, self.device)
        dmg = nets[0].damage_features(t) if self.use_damage else None
        W = {name: b[:n] for name, b in self.buf.items()}
        out = self._forward(W, t, dmg, k, n, bm).float().cpu()
        return [out[s * bm:s * bm + m] for s, m in zip(slot_of, lens)], refills

    # ------------------------------------------------------------ the forward (Net.forward's policy path)

    def _forward(self, W: dict, obs: dict, dmg: dict | None, k: int, n: int, bm: int) -> torch.Tensor:
        """Mirrors `Net.forward(obs, dmg, aux=False)["logits"]` line by line on the folded batch; the parameter-free
        ops are Net's, the ones with weights are the grouped versions above."""
        lin = lambda x, p: _lin(x, W[p + ".weight"], W[p + ".bias"])  # noqa: E731
        mlp = lambda x, p: lin(F.gelu(lin(x, p + ".0")), p + ".2")  # noqa: E731
        emb = lambda name, ids: _emb(W[name], self.off[name][:n], ids)  # noqa: E731
        ids = {key: obs[key].long() for key in ("species", "moves", "item", "ability", "types", "move_type")}
        mon_num = obs["mon_num"].float()
        b = mon_num.shape[0]
        present = mon_num[..., COL["present"]]
        active = mon_num[..., COL["active"]] * present
        alive = present * (1.0 - mon_num[..., COL["fainted"]])
        w_act, w_alive = Net._other_side(active), Net._other_side(alive)  # [B, 12, 6]

        valid_move = ids["moves"] != PAD
        move_parts = [emb("ent_moves", ids["moves"][:, :, :k]), emb("type_emb.weight", ids["move_type"][:, :, :k]),
                      obs["move_num"][:, :, :k].float(), obs["move_p"][:, :, :k].float().unsqueeze(-1)]
        if dmg is not None:
            move_parts.append(Net._summary(dmg["move_vs"][:, :, :k].float(), w_act[:, :, None], w_alive[:, :, None]))
        move_in = torch.cat(move_parts, -1)
        mvec = mlp(move_in, "move_mlp") * valid_move[:, :, :k].unsqueeze(-1).to(move_in.dtype)
        if k < KM:
            mvec = F.pad(mvec, (0, 0, 0, KM - k))

        move_p = obs["move_p"].float().unsqueeze(-1)
        mean_moves = (move_p * mvec).sum(-2) / move_p.sum(-2).clamp(min=1e-6)
        max_moves = mvec.masked_fill(~valid_move.unsqueeze(-1), -1e4).amax(-2)
        max_moves = torch.where(valid_move.any(-1, keepdim=True), max_moves, torch.zeros_like(max_moves))
        item = (obs["item_p"].float().unsqueeze(-1) * emb("ent_items", ids["item"])).sum(-2)
        ability = (obs["ability_p"].float().unsqueeze(-1) * emb("ent_abilities", ids["ability"])).sum(-2)
        hp = obs["hp"].float()
        mon_parts = [
            emb("ent_species", ids["species"]),
            emb("type_emb.weight", ids["types"]).sum(-2),
            _gmm(obs["tera_p"].float(), W["type_emb.weight"]),
            item, ability, mean_moves, max_moves, mon_num,
            torch.log1p(obs["stats"].float().clamp(min=0)) / 6,
            torch.stack([hp[..., 0] / hp[..., 1].clamp(min=1.0), hp[..., 1] / 500], -1),
            obs["boosts"].float() / 6,
        ]
        if dmg is not None:
            speed = dmg["speed"].float()
            mon_parts.append(torch.stack([(speed * w_act).sum(-1),
                                          (speed * w_alive).sum(-1) / w_alive.sum(-1).clamp(min=1.0)], -1))
        mons = mlp(torch.cat(mon_parts, -1), "mon_mlp")
        mons = torch.where(present.unsqueeze(-1).unflatten(0, (n, bm)) > 0, mons.unflatten(0, (n, bm)),
                           W["absent12"][:, None].to(mons.dtype)).flatten(0, 1)

        field = mlp(torch.cat([obs["field_num"].float(), obs["side_num"].float().flatten(1)], -1), "field_mlp")

        own_act = (active * alive)[:, :6]
        act_mvec = torch.einsum("bi,bijd->bjd", own_act, mvec[:, :6, :4])
        tokens = [field.unsqueeze(1), mons, mlp(act_mvec, "move_token")]
        if self.tera:
            tin = torch.einsum("bi,bijd->bjd", own_act, move_in[:, :6, :4])
            if dmg is not None:
                tvs = Net._summary(dmg["active_tera_vs"].float(), w_act[:, None, 0], w_alive[:, None, 0])
                tin = torch.cat([tin[..., :-tvs.shape[-1]], tvs], -1)
            act_valid = torch.einsum("bi,bij->bj", own_act, valid_move[:, :6, :4].to(own_act.dtype)) > 0
            tvec = mlp(tin, "move_mlp") * act_valid.unsqueeze(-1).to(tin.dtype)
            tera_type = _gmm(torch.einsum("bi,bit->bt", own_act, obs["tera_p"].float()[:, :6]), W["type_emb.weight"])
            tokens.append(mlp(torch.cat([tvec, tera_type.unsqueeze(1).expand(-1, 4, -1)], -1), "tera_token"))
        x = (torch.cat(tokens, 1).unflatten(0, (n, bm)) + W["tok"][:, None]).flatten(0, 1)

        bias = None
        if self.mask_fainted:
            fainted = (present * mon_num[..., COL["fainted"]]) > 0
            keymask = torch.zeros(b, self.n_tokens, device=x.device, dtype=torch.float32)
            keymask[:, 1:1 + N_MONS] = keymask[:, 1:1 + N_MONS].masked_fill(fainted, float("-inf"))
            keymask = keymask[:, None, None, :]  # [B, 1, 1, T]
        else:
            keymask = None
        if dmg is not None:
            pres = present > 0
            pb = mlp(dmg["pair"].float(), "pair_mlp")  # [B, 12, 12, L*H]
            pb = pb * (pres[:, :, None] & pres[:, None, :]).unsqueeze(-1).to(pb.dtype)
            pb = pb.view(b, N_MONS, N_MONS, self.n_layers, self.n_heads).permute(3, 0, 4, 1, 2)
            t = self.n_tokens
            bias = F.pad(pb, (1, t - 1 - N_MONS, 1, t - 1 - N_MONS))  # [L, B, H, T, T]
        for i in range(self.n_layers):
            m = None if bias is None else bias[i]
            if keymask is not None:
                m = keymask if m is None else m + keymask
            x = self._block(W, x, m, f"blocks.{i}.", mlp)
        x = _rms(x, W["norm.weight"])

        field_t, own_t, opp_t = x[:, 0], x[:, 1:7], x[:, 7:13]
        live_act = (active * alive).to(x.dtype)
        own_act_tok = torch.einsum("bi,bid->bd", live_act[:, :6], own_t)
        opp_act_tok = torch.einsum("bk,bkd->bd", live_act[:, 6:], opp_t)
        cand = torch.cat([own_t, x[:, 13:17]] + ([x[:, 17:21]] if self.tera else []), 1)
        ctx = torch.cat([field_t, own_act_tok, opp_act_tok], -1).unsqueeze(1).expand(-1, cand.shape[1], -1)
        scores = mlp(torch.cat([cand, ctx], -1), "policy_head").squeeze(-1).float()
        logits = torch.zeros(b, self.n_actions, device=scores.device, dtype=torch.float32)
        logits[:, :10] = scores[:, :10]
        if self.tera:
            logits[:, TERA_OFFSET:TERA_OFFSET + 4] = scores[:, 10:14]
        return logits.masked_fill(obs["action_mask"] == 0, torch.finfo(torch.float32).min)

    def _block(self, W: dict, x: torch.Tensor, m: torch.Tensor | None, p: str, mlp) -> torch.Tensor:
        """`Block.forward` with the group's weights."""
        b, t, d = x.shape
        h = self.n_heads
        qkv = _lin(_rms(x, W[p + "norm1.weight"]), W[p + "qkv.weight"], W[p + "qkv.bias"])
        q, k, v = qkv.view(b, t, 3, h, d // h).permute(2, 0, 3, 1, 4)
        if self.qk_norm:
            q, k = _rms(q, W[p + "q_norm.weight"]), _rms(k, W[p + "k_norm.weight"])
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=None if m is None else m.to(q.dtype))
        x = x + _lin(a.transpose(1, 2).reshape(b, t, d), W[p + "proj.weight"], W[p + "proj.bias"])
        hn = _rms(x, W[p + "norm2.weight"])
        return x + torch.cat([mlp(hn[:, s:e], f"{p}ffn.{g}") for g, (s, e) in enumerate(self.groups)], 1)
