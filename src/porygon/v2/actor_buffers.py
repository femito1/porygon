"""Double-buffered acting weights for the v2 trainer's collector thread (`porygon.v2.train --actor-stream`).

The collector thread acts with a copy of the learner net. With one copy (the default) a refresh overwrites it under a
lock, so the actor stops while the weights are copied, and on CUDA both threads issue their kernels to the device's
default stream, so under overlap the actor's forwards queue behind the PPO update's. `ActorBuffers` keeps two copies:
the actor reads the *front* one (`acquire()`), the learner copies its weights into the *back* one and then flips
(`refresh()`). On CUDA the collector thread runs on `stream`, a non-blocking pool stream of the highest priority (the
trainer makes it that thread's current stream; the current stream is per thread), and two kinds of events per copy
order the two threads' device work:

- `ready[i]`: the last weight copy into copy i is complete (recorded by `refresh` on the learner's stream after the
  copy; `acquire` makes the actor's stream wait on it before the caller launches a forward on copy i);
- `used[i]`: every actor forward that read copy i was launched before it (recorded by `acquire` on the actor's
  stream once the caller has launched its forward; `refresh` makes the learner's stream wait on it before the copy
  into i). One such event per stream that acted (the trainer has one: the collector thread's).

Correctness on CUDA. `acquire` holds `lock` from the snapshot of `front`, through the caller's launch, to the record
of `used`; `refresh` flips under `lock`. A forward that read copy b = 1 - front snapshotted b while b was the front,
i.e. before the last flip, and recorded its `used[b]` event before that flip could take the lock; after the flip
nobody snapshots b, so no `used[b]` event is recorded until b is the front again. Waiting on the latest `used[b]`
event of every stream therefore covers every forward that read b (stream order covers that stream's earlier ones),
and the copy into b, issued on the learner's stream after that wait (and after the learner's own writes to `src`),
cannot overlap a reader. Conversely a forward on the new front is launched after its stream waited on
`ready[front]`, recorded after the copy into it, so it never sees a partly written copy. The two initial deep copies
(asynchronous device-to-device clones) count as the first copy into each: both `ready` events are recorded after them.

On CPU there are no streams or events: a forward runs synchronously inside `acquire` (under the lock), so once the
flip has happened nobody reads the back copy, and `refresh` copies into it without the lock.

Contract: use the yielded net only inside the `with acquire()` block (its outputs may be used after it: they do not
read the weights); call `refresh` from one thread (the learner's), on the stream that wrote `src`'s weights.
"""

import contextlib
import copy
import threading

import torch
from torch import nn


class ActorBuffers:
    """Two eval-mode deep copies of `net` (`nets`), the index of the one the actor reads (`front`), the learner
    version each holds (`versions`, `version` for both at the start) and the lock of the flip (`lock`). On CUDA also
    the actor `stream` (highest priority) and the `ready` / `used` events (see the module docstring)."""

    def __init__(self, net: nn.Module, device=None, version: int = 0):
        self.device = torch.device(device) if device is not None else next(net.parameters()).device
        self.nets = [copy.deepcopy(net).eval(), copy.deepcopy(net).eval()]
        self.versions = [version, version]
        self.front = 0
        self.lock = threading.Lock()
        self.stream = None
        if self.device.type == "cuda":
            self.priority_range = torch.cuda.Stream.priority_range()  # (lowest, highest): a lower number is higher
            self.stream = torch.cuda.Stream(self.device, priority=self.priority_range[1])
            self.ready = [torch.cuda.Event(), torch.cuda.Event()]
            self.used: list[dict] = [{}, {}]  # per copy: stream -> event after its last forward that read the copy
            cur = torch.cuda.current_stream(self.device)
            for ev in self.ready:  # the deep copies were issued on this stream
                ev.record(cur)

    @contextlib.contextmanager
    def acquire(self):
        """Actor side: `with buffers.acquire() as (net, version): out = net(...)`. Under `lock`: the front copy and the
        learner version it holds; on CUDA the current stream first waits for the last copy into it, and an event
        recorded on that stream after the block marks the forward(s) launched inside it."""
        with self.lock:
            i = self.front
            cur = None if self.stream is None else torch.cuda.current_stream(self.device)
            if cur is not None:
                cur.wait_event(self.ready[i])
            try:
                yield self.nets[i], self.versions[i]
            finally:
                if cur is not None:
                    ev = self.used[i].get(cur)
                    if ev is None:
                        ev = self.used[i][cur] = torch.cuda.Event()
                    ev.record(cur)

    @torch.no_grad()
    def refresh(self, src: nn.Module, version: int) -> None:
        """Learner side: copy `src`'s state into the back copy (on the current stream, after the actor's forwards that
        read it), then make it the front, holding `version`. The actor keeps acting on the front copy meanwhile."""
        back = 1 - self.front  # only this (the learner's) thread changes `front`
        cur = None if self.stream is None else torch.cuda.current_stream(self.device)
        if cur is not None:
            for ev in self.used[back].values():
                cur.wait_event(ev)
        self.nets[back].load_state_dict(src.state_dict())
        if cur is not None:
            self.ready[back].record(cur)
        with self.lock:
            self.front = back
            self.versions[back] = version

    def describe(self) -> str:
        """One line for the trainer's log."""
        if self.stream is None:
            return "actor-stream: CPU: double buffer only (two actor copies, no CUDA streams)"
        lo, hi = self.priority_range
        return (f"actor-stream: collector thread on a CUDA stream of priority {self.stream.priority} (range {lo} = "
                f"lowest .. {hi} = highest; the update stays on the default stream), actor weights double-buffered")
