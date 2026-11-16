"""NNUE gate 1 student fit (docs/netsearch/NNUE-GATE1-PREREG.md, "Student"): target y = 0.5 teacher P(win) + 0.5
outcome, BCE on the logit, AdamW, early stopping on validation BCE (split 1); train = split 0 (nnue_gen.py shards).
Writes OUT/student.pt (state_dict + spec), OUT/student.npz (NumpyStudent weights) and OUT/fit.json (per-epoch log).
Exit 3 on divergence (non-finite loss): the chain may re-run (PREREG: up to 2 more fits for crashes / divergence).

    PORYGON_GEN=9 PYTHONPATH=$CODE/src:$G/.engine-gen9 $PY scripts/fastenv/nnue_fit.py --data nnue/gate1/data --out nnue/gate1/fit
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

COLS = ("ids", "mon", "side", "field", "teacher", "outcome")


def load(data: Path, splits: set[int]) -> dict[str, np.ndarray]:
    """Rows of the given splits from every committed shard (preallocated: no concatenation copies)."""
    shards = sorted(p.with_suffix(".npz") for p in data.glob("shard-*.json") if p.with_suffix(".npz").exists())
    assert shards, f"no shards in {data}"
    counts = []
    for s in shards:
        with np.load(s) as z:
            counts.append(int(np.isin(z["split"], list(splits)).sum()))
    n = sum(counts)
    with np.load(shards[0]) as z:
        out = {k: np.empty((n,) + z[k].shape[1:], z[k].dtype) for k in COLS + ("split", "game")}
    o = 0
    for s, c in zip(shards, counts):
        with np.load(s) as z:
            keep = np.isin(z["split"], list(splits))
            for k in out:
                out[k][o:o + c] = z[k][keep]
        o += c
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-rows", type=int, default=0, help="tests: use at most this many training rows")
    a = p.parse_args()
    import torch
    from porygon.nnue import features as F
    from porygon.nnue.student import NumpyStudent, Student, n_params
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = load(Path(a.data), {0, 1})
    tr = np.flatnonzero(d["split"] == 0)
    va = np.flatnonzero(d["split"] == 1)
    if a.max_rows:
        tr = tr[:a.max_rows]
    y = (0.5 * d["teacher"].astype(np.float32) + 0.5 * d["outcome"].astype(np.float32))
    spec = F.spec()
    net = Student(spec).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch,
                                                pct_start=0.05, anneal_strategy="cos")
    dev = a.device

    def batch(idx):
        t = lambda k, dt: torch.from_numpy(d[k][idx]).to(dev, non_blocking=True).to(dt)  # noqa: E731
        return (t("ids", torch.long), t("mon", torch.float32), t("side", torch.float32), t("field", torch.float32),
                torch.from_numpy(y[idx]).to(dev))

    def val_loss() -> float:
        net.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(va), 16384):
                ids, mon, side, field, yy = batch(va[i:i + 16384])
                tot += float(torch.nn.functional.binary_cross_entropy_with_logits(
                    net(ids, mon, side, field), yy, reduction="sum"))
        net.train()
        return tot / max(len(va), 1)

    log = {"rows_train": int(len(tr)), "rows_val": int(len(va)), "params": n_params(spec), "spec": spec["hash"],
           "args": vars(a), "load_s": time.time() - t0, "epochs": []}
    print(f"loaded {len(tr)} train / {len(va)} val rows in {log['load_s']:.0f} s; params {log['params']}", flush=True)
    best, best_ep, bad = math.inf, -1, 0
    for ep in range(a.epochs):
        te = time.time()
        perm = tr[rng.permutation(len(tr))]
        tot, n = torch.zeros((), device=dev), 0
        for j, i in enumerate(range(0, len(perm), a.batch)):
            idx = np.sort(perm[i:i + a.batch])  # sorted gathers are faster; order inside a batch is irrelevant
            ids, mon, side, field, yy = batch(idx)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(ids, mon, side, field), yy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.detach() * len(idx)  # no per-step GPU sync
            n += len(idx)
            if (j % 500 == 499 or i + a.batch >= len(perm)) and not math.isfinite(float(tot)):
                log["diverged"] = {"epoch": ep, "step": i // a.batch}
                (out / "fit.json").write_text(json.dumps(log, indent=1))
                print(f"diverged at epoch {ep}", file=sys.stderr)
                return 3
        vl = val_loss()
        tot = float(tot)
        log["epochs"].append({"epoch": ep, "train_bce": tot / n, "val_bce": vl, "seconds": time.time() - te,
                              "lr": sched.get_last_lr()[0]})
        print(f"epoch {ep}: train {tot / n:.5f} val {vl:.5f} ({time.time() - te:.0f} s)", flush=True)
        if vl < best - 1e-6:
            best, best_ep, bad = vl, ep, 0
            torch.save({"state_dict": net.state_dict(), "spec": spec, "epoch": ep, "val_bce": vl}, out / "student.pt")
            NumpyStudent.save(str(out / "student.npz"), net.numpy_weights(), spec)
        else:
            bad += 1
            if bad >= a.patience:
                break
        (out / "fit.json").write_text(json.dumps(log, indent=1))
    log.update({"best_epoch": best_ep, "best_val_bce": best, "seconds": time.time() - t0})
    (out / "fit.json").write_text(json.dumps(log, indent=1))
    print(f"done: best epoch {best_ep}, val BCE {best:.5f}, {time.time() - t0:.0f} s")
    return 0 if best_ep >= 0 else 3


if __name__ == "__main__":
    sys.exit(main())
