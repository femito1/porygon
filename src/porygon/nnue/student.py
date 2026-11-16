"""NNUE gate 1 student (docs/netsearch/NNUE-GATE1-PREREG.md, "Student"): a shared per-mon MLP (64) -> per side sum
and max pooling + the active mon's vector + the side vector, both sides + the field -> a 2-layer MLP (128) -> one
logit (P(win) of side one = sigmoid). Inputs = porygon.nnue.features. f32.

    from porygon.nnue.student import Student, NumpyStudent
    net = Student(features.spec()); logit = net(ids, mon, side, field)        # torch, batched
    ns = NumpyStudent.load("student.npz"); p = ns.p_win(features(state))      # numpy, one State (CPU leaf cost)
"""
from __future__ import annotations

import json

import numpy as np

EMB = {"species": 16, "item": 8, "ability": 8, "move": 16, "tera": 8}
MON_H, TRUNK_H = 64, 128


def _torch_model():
    import torch
    from torch import nn

    class Student(nn.Module):
        def __init__(self, spec: dict):
            super().__init__()
            self.spec = spec
            self.e_species = nn.Embedding(spec["n_species"], EMB["species"])
            self.e_item = nn.Embedding(spec["n_items"], EMB["item"])
            self.e_ability = nn.Embedding(spec["n_abilities"], EMB["ability"])
            self.e_move = nn.Embedding(spec["n_moves"], EMB["move"], padding_idx=0)
            self.e_tera = nn.Embedding(spec["n_tera"], EMB["tera"])
            d_in = sum(EMB.values()) + spec["F_MON"]
            self.mon1, self.mon2 = nn.Linear(d_in, MON_H), nn.Linear(MON_H, MON_H)
            d_trunk = 2 * (3 * MON_H + spec["F_SIDE"]) + spec["F_FIELD"]
            self.t1, self.t2, self.out = nn.Linear(d_trunk, TRUNK_H), nn.Linear(TRUNK_H, TRUNK_H), nn.Linear(TRUNK_H, 1)

        def forward(self, ids, mon, side, field):
            """ids long [B, 12, 8], mon [B, 12, F_MON], side [B, 2, F_SIDE], field [B, F_FIELD] -> logit [B]."""
            r = torch.relu
            x = torch.cat([self.e_species(ids[..., 0]), self.e_item(ids[..., 1]), self.e_ability(ids[..., 2]),
                           self.e_move(ids[..., 3:7]).sum(-2), self.e_tera(ids[..., 7]), mon], -1)
            h = r(self.mon2(r(self.mon1(x))))                       # [B, 12, H]
            h = h.view(h.shape[0], 2, 6, MON_H)
            per_side = torch.cat([h.sum(2), h.amax(2), h[:, :, 0], side], -1)  # [B, 2, 3H + F_SIDE]
            z = torch.cat([per_side.flatten(1), field], -1)
            return self.out(r(self.t2(r(self.t1(z))))).squeeze(-1)

        def numpy_weights(self) -> dict[str, np.ndarray]:
            return {k: v.detach().float().cpu().numpy() for k, v in self.state_dict().items()}

    return Student


def Student(spec: dict):  # noqa: N802 (a factory: torch is imported only when a torch model is wanted)
    return _torch_model()(spec)


def n_params(spec: dict) -> int:
    d_in = sum(EMB.values()) + spec["F_MON"]
    d_trunk = 2 * (3 * MON_H + spec["F_SIDE"]) + spec["F_FIELD"]
    emb = (spec["n_species"] * EMB["species"] + spec["n_items"] * EMB["item"] + spec["n_abilities"] * EMB["ability"]
           + spec["n_moves"] * EMB["move"] + spec["n_tera"] * EMB["tera"])
    return emb + (d_in + 1) * MON_H + (MON_H + 1) * MON_H + (d_trunk + 1) * TRUNK_H + (TRUNK_H + 1) * TRUNK_H + TRUNK_H + 1


class NumpyStudent:
    """The same forward in numpy (PyTorch-free), one position or a batch."""

    def __init__(self, w: dict[str, np.ndarray], spec: dict):
        self.spec = spec
        f = lambda k: np.ascontiguousarray(w[k], np.float32)  # noqa: E731
        self.es, self.ei, self.ea, self.em, self.et = (f(f"e_{k}.weight") for k in ("species", "item", "ability", "move", "tera"))
        self.W1, self.b1, self.W2, self.b2 = f("mon1.weight").T.copy(), f("mon1.bias"), f("mon2.weight").T.copy(), f("mon2.bias")
        self.T1, self.c1, self.T2, self.c2 = f("t1.weight").T.copy(), f("t1.bias"), f("t2.weight").T.copy(), f("t2.bias")
        self.O, self.o = f("out.weight").T.copy(), f("out.bias")

    @classmethod
    def load(cls, path: str) -> "NumpyStudent":
        z = np.load(path, allow_pickle=False)
        spec = json.loads(str(z["__spec__"]))
        return cls({k: z[k] for k in z.files if k != "__spec__"}, spec)

    @staticmethod
    def save(path: str, weights: dict[str, np.ndarray], spec: dict) -> None:
        np.savez(path, __spec__=np.array(json.dumps(spec)), **weights)

    def logits(self, ids: np.ndarray, mon: np.ndarray, side: np.ndarray, field: np.ndarray) -> np.ndarray:
        """Batched: ids [B, 12, 8], mon [B, 12, F], side [B, 2, S], field [B, F_FIELD] -> [B]."""
        B = ids.shape[0]
        x = np.concatenate([self.es[ids[..., 0]], self.ei[ids[..., 1]], self.ea[ids[..., 2]],
                            self.em[ids[..., 3:7]].sum(-2),  # row 0 = zeros (padding_idx)
                            self.et[ids[..., 7]], mon.astype(np.float32)], -1)
        h = np.maximum(x @ self.W1 + self.b1, 0)
        h = np.maximum(h @ self.W2 + self.b2, 0).reshape(B, 2, 6, MON_H)
        per_side = np.concatenate([h.sum(2), h.max(2), h[:, :, 0], side.astype(np.float32)], -1)
        z = np.concatenate([per_side.reshape(B, -1), field.astype(np.float32)], -1)
        z = np.maximum(z @ self.T1 + self.c1, 0)
        z = np.maximum(z @ self.T2 + self.c2, 0)
        return (z @ self.O + self.o)[:, 0]

    def p_win(self, f: dict[str, np.ndarray]) -> float:
        """One position (features() dict)."""
        lg = self.logits(f["ids"][None], f["mon"][None], f["side"][None], f["field"][None])[0]
        return float(1 / (1 + np.exp(-lg)))
