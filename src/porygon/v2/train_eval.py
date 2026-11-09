"""Asynchronous in-training evaluation (`porygon.v2.train --eval-async`). At each eval point the trainer saves a
snapshot of the live net and starts this module on it in one subprocess; it plays the games of the synchronous
evaluation (the same `porygon.v2.train.Evaluator`: the --anchor net if given, then SimpleHeuristicsPlayer and
MaxBasePowerPlayer, --battles each) on CPU with one torch thread and writes the win rates atomically to a JSON file,
which the trainer polls for once per update and then handles as an eval of the snapshot's weights.

    python -m porygon.v2.train_eval RUN/eval/snap_00002500000.pt --format gen9randombattle --port 8000 \
        --battles 200 [--anchor NET] --out RUN/eval/result_00002500000.json

Result: {"steps": the snapshot's env steps, "res": {opponent: win rate}, "seconds": time spent playing}. The trainer
starts it with `child_env()` (CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=1) and appends its stdout and stderr to
RUN/eval/eval.log. The trainer-side helpers (`launch`, `stop_process`, `copy_checkpoint`, `tail`) live here too.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch

SRC = str(Path(__file__).resolve().parents[2])  # the directory holding the `porygon` package


# ---------------------------------------------------------------- trainer side

@dataclass
class EvalJob:
    """One running eval subprocess: the snapshot's env `steps` and trainer `counters` at the eval point."""
    steps: int
    snap: Path
    result: Path
    counters: dict
    proc: subprocess.Popen
    log: Path
    started: float = field(default_factory=time.time)


def child_env(base: dict | None = None) -> dict[str, str]:
    """The eval subprocess's environment: the trainer's, CPU only, one OpenMP thread, unbuffered output, and the
    trainer's own source tree first on PYTHONPATH (the child runs the same code)."""
    env = dict(os.environ if base is None else base)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (SRC, env.get("PYTHONPATH")) if p)
    return env


def launch(cmd: list[str], log: Path) -> subprocess.Popen:
    """Start `cmd` with `child_env()`, its stdout and stderr appended to `log` (after a line with the command)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(cmd)}\n".encode())
        f.flush()
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=child_env())


def stop_process(proc: subprocess.Popen, timeout: float = 10.0) -> int | None:
    """SIGTERM, wait up to `timeout` s, then SIGKILL; the exit code (None if it still has not exited)."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                pass
    return proc.returncode


def copy_checkpoint(src: Path, dst: Path, **fields) -> Path:
    """Atomic copy of a `porygon.v2.model.save` file with top-level `fields` set (e.g. counters, eval); the weights
    are the source's, bit for bit."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck.update(fields)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, dst)
    return dst


def tail(path: Path, n: int = 20, max_bytes: int = 16384) -> str:
    """The last `n` lines of a text file, indented (for a log message)."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, f.seek(0, os.SEEK_END) - max_bytes))
            lines = f.read().decode(errors="replace").splitlines()[-n:]
    except OSError as e:
        return f"    ({e!r})"
    return "\n".join("    " + line[:400] for line in lines)


# ---------------------------------------------------------------- eval subprocess

def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj) + "\n")
    os.replace(tmp, path)


def unique_player_names() -> None:
    """poke-env names players "<class> <n>", counting from 1 per class in each process, so every eval process would
    log in as the same users: start the counts at a random offset instead, so a previous eval process's users still
    on the server (crashed or orphaned, maybe mid-battle) neither block our login (name taken) nor hand us their
    unfinished battles when we log in as them."""
    try:
        from poke_env.ps_client.account_configuration import CONFIGURATION_FROM_PLAYER_COUNTER as count
    except ImportError:  # other poke-env layout: default names
        return
    base = random.SystemRandom().randrange(10_000, 90_000)
    for cls in ("NetPlayer", "PolicyPlayer", "SimpleHeuristicsPlayer", "MaxBasePowerPlayer"):
        count[cls] = max(count[cls], base)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("snapshot", type=Path, help="v2 net to evaluate (`porygon.v2.model.save` file or trainer checkpoint)")
    a("--format", default="gen9randombattle")
    a("--port", type=int, default=8000, help="Showdown server port")
    a("--battles", "--eval-battles", type=int, default=200, help="battles per opponent")
    a("--anchor", type=Path, help="also play this net (v2 .pt, or v1 .zip / pool .pt), as the trainer's --anchor")
    a("--concurrency", type=int, default=None, help="concurrent battles (default: Evaluator's)")
    a("--steps", type=int, default=None, help="env steps written to the result (default: the snapshot's counters)")
    a("--out", type=Path, help="result JSON, written atomically (default: printed only)")
    a("--threads", type=int, default=1, help="torch CPU threads")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    from porygon.v2.model import net_from_checkpoint
    from porygon.v2.train import Evaluator  # not at module level: porygon.v2.train imports this module
    ck = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    net = net_from_checkpoint(ck, "cpu").eval()
    steps = args.steps if args.steps is not None else ck.get("counters", {}).get("steps")
    del ck
    unique_player_names()
    kw = {} if args.concurrency is None else {"concurrency": args.concurrency}
    ev = Evaluator(net, args.format, args.port, args.anchor, torch.device("cpu"), **kw)
    log(f"eval {args.snapshot} (steps {steps}): {args.battles} battles each vs {', '.join(ev.opponents)} on port "
        f"{args.port}, {torch.get_num_threads()} torch thread(s)")
    start = time.time()
    res = ev.run(args.battles)
    out = {"steps": steps, "res": res, "seconds": time.time() - start}
    if args.out is not None:
        write_json(args.out, out)
    log(f"result: {json.dumps(out)}")
    return out


if __name__ == "__main__":
    code = 1
    try:
        main()
        code = 0
    except SystemExit as e:  # argparse (--help, bad arguments)
        code = e.code if isinstance(e.code, int) else int(e.code is not None)
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)  # never wait on poke-env's websocket threads (as the battle workers do)
