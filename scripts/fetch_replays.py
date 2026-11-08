#!/usr/bin/env python3
"""Polite, resumable crawler for the Pokemon Showdown replay server, plus the corpus builder.

Subcommands
  crawl      Walk replay.pokemonshowdown.com/search.json (top-rated pages, then newest -> oldest
             by uploadtime) for the given formats and download each replay's .json (log +
             inputlog) in priority order: rating bucket (>=2000, >=1800, >=1600, >=1400, rated,
             unrated), newest first within a bucket. Formats are served round-robin.
  stats      Counts from the crawl state (search rows and fetched replays) by rating and month.
  sync       rsync closed replay shards to the cluster (and delete them locally once copied).
  normalise  Build the raw corpus (one JSONL.gz series per format + index) from the crawl shards
             and the Hugging Face dumps (HolidayOugi parquet, PokeChamp parquet). Run on the cluster.
  corpus-stats  Counts by source x rating threshold x month from a corpus index.

Politeness: one request at a time, >= 1 s between requests (the replay server publishes no rate
limit; WEB-API.md documents the JSON endpoints as public), an identifying User-Agent, every
response cached (search pages in SQLite, replays in append-only gzip shards; nothing is fetched
twice), exponential backoff with Retry-After on 429/5xx/network errors.

Downloads are untrusted data: nothing from them is executed. Run with `python -I`.

Crawl state lives in --root (default ~/.cache/porygon-ws-data/raw/showdown-replays):
  state.sqlite                    search pages (zlib bodies), search rows, fetch log, cursors
  replays/<format>/<YYYYmmddTHH>.jsonl.gz
                                  one line per replay: {id, url, fetched, status, body}; body is the
                                  replay JSON text exactly as served (log, inputlog, rating, ...)
Resume: rerun the same `crawl` command; it continues from the stored cursors.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import io
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from datetime import datetime, timezone

UA = "porygon-research (academic; contact via github femito1)"
BASE = "https://replay.pokemonshowdown.com"
DEFAULT_ROOT = os.path.expanduser("~/.cache/porygon-ws-data/raw/showdown-replays")
CLUSTER = "gpucluster2"
CLUSTER_RAW = "/vol/bitbucket/fr1226/porygon-ws-data/raw/showdown-replays"
THRESHOLDS = (1500, 1700, 1800, 1900, 2000)
# Accounts whose games we never fetch or keep: Jaxcalibur (a bot we must not distil; orchestrator rule)
# and our own bot-flagged ladder accounts. Matched on Showdown user ids (lowercase alphanumerics).
EXCLUDE_PLAYERS = {"jaxcalibur", "porygonagent", "femito"}

STOP = False


def _on_signal(signum, frame):  # finish the current request, flush, exit
    global STOP
    STOP = True


def nap(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end and not STOP:
        time.sleep(1)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"{now_iso()} {msg}", flush=True)


PRIO_VERSION = 2
PRIO_BANDS = ((5, 1900), (4, 1800), (3, 1700), (2, 1500))  # then 1 = rated below 1500, 0 = unrated


def prio_of(rating) -> int:
    if rating is None:
        return 0
    r = int(rating)
    for p, t in PRIO_BANDS:
        if r >= t:
            return p
    return 1


def toid(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def excluded(players) -> bool:
    return any(toid(p) in EXCLUDE_PLAYERS for p in (players or []))


# --------------------------------------------------------------------------- HTTP
class Client:
    def __init__(self, min_interval: float = 1.0):
        self.min_interval = max(0.5, min_interval)  # never more than 2 requests/s, one in flight
        self.last = 0.0
        self.n = 0

    def get(self, url: str) -> tuple[int, bytes]:
        """GET with rate limit and backoff. Returns (status, body); 404/403 are returned, not raised."""
        backoff = 30.0
        while not STOP:
            wait = self.last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.n += 1
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return r.status, body
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500 and e.code not in (408, 429):  # final (404, 410 "inputlog not found", ...)
                    return e.code, e.read() or b""
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                log(f"HTTP {e.code} on {url}; backing off {wait:.0f}s")
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                wait = backoff
                log(f"network error {e!r} on {url}; backing off {wait:.0f}s")
            t_end = time.monotonic() + wait
            while time.monotonic() < t_end and not STOP:
                time.sleep(1)
            backoff = min(backoff * 2, 1800.0)
        return -1, b""


# --------------------------------------------------------------------------- state
SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (key TEXT PRIMARY KEY, url TEXT, fetched TEXT, status INT, body BLOB);
CREATE TABLE IF NOT EXISTS rows (id TEXT PRIMARY KEY, format TEXT, uploadtime INT, rating INT,
    players TEXT, private INT, prio INT, fetched INT DEFAULT 0, need TEXT DEFAULT 'json');
CREATE INDEX IF NOT EXISTS rows_queue ON rows (format, fetched, prio, uploadtime);
CREATE TABLE IF NOT EXISTS replays (id TEXT PRIMARY KEY, format TEXT, fetched TEXT, status INT,
    shard TEXT, nbytes INT, has_inputlog INT);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""


class State:
    def __init__(self, root: str):
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.db = sqlite3.connect(os.path.join(root, "state.sqlite"))
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(rows)")}
        if "need" not in cols:  # migration from the first schema
            self.db.execute("ALTER TABLE rows ADD COLUMN need TEXT DEFAULT 'json'")
        if self.kv_get("prio_version") != PRIO_VERSION:
            case = " ".join(f"WHEN rating>={t} THEN {p}" for p, t in PRIO_BANDS)
            self.db.execute(f"UPDATE rows SET prio = CASE WHEN rating IS NULL THEN 0 {case} ELSE 1 END")
            self.kv_set("prio_version", PRIO_VERSION)
        # fetched: 0 pending, 1 done, -1 failed (non-200), -3 excluded player
        self.db.execute("UPDATE rows SET fetched=-3 WHERE fetched=0 AND (" + " OR ".join(
            "lower(players) LIKE ?" for _ in EXCLUDE_PLAYERS) + ")",
            [f'%"{p}"%' for p in EXCLUDE_PLAYERS])
        self.db.commit()

    def kv_get(self, key, default=None):
        r = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def kv_set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value)))

    def cached_page(self, key):
        r = self.db.execute("SELECT status, body FROM pages WHERE key=?", (key,)).fetchone()
        return (r[0], zlib.decompress(r[1])) if r else None

    def save_page(self, key, url, status, body):
        self.db.execute("INSERT OR REPLACE INTO pages VALUES (?,?,?,?,?)",
                        (key, url, now_iso(), status, zlib.compress(body, 6)))

    def add_rows(self, fmt, rows) -> int:
        new = 0
        for r in rows:
            if r.get("password"):  # private-with-password replays are not public; skip
                continue
            cur = self.db.execute(
                "INSERT OR IGNORE INTO rows (id, format, uploadtime, rating, players, private, prio, fetched) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (r["id"], fmt, r.get("uploadtime"), r.get("rating"), json.dumps(r.get("players")),
                 r.get("private") or 0, prio_of(r.get("rating")), -3 if excluded(r.get("players")) else 0))
            new += cur.rowcount
        return new


# --------------------------------------------------------------------------- crawl
class Crawler:
    def __init__(self, a):
        self.a = a
        self.st = State(a.root)
        self.http = Client(a.min_interval)
        self.formats = a.formats
        self.buf: dict[str, list] = collections.defaultdict(list)  # format -> pending shard lines
        self.queue: dict[str, collections.deque] = {f: collections.deque() for f in self.formats}
        self.rr = self.srr = self.frr = 0
        self.last_sync = time.time()
        self.t0 = time.time()
        self.fetched_this_run = 0
        self.min_prio = {k: int(v) for k, v in (x.split("=") for x in a.format_min_prio)}
        self.qbuf: dict[str, list] = {}
        self.qkey = None
        self.open_queue()

    # ---- search
    def search_page(self, fmt: str, params: dict, time_dependent: bool):
        url = f"{BASE}/search.json?" + urllib.parse.urlencode({"format": fmt, **params})
        key = url if not time_dependent else f"{url}#{int(time.time())}"
        cached = None if time_dependent else self.st.cached_page(key)
        if cached:
            status, body = cached
        else:
            status, body = self.http.get(url)
            if status < 0:
                return None
            self.st.save_page(key, url, status, body)
        if status != 200:
            log(f"search {url} -> {status}")
            return []
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            log(f"search {url}: bad JSON ({body[:100]!r})")
            return []

    def step_top_rated(self, fmt: str) -> bool:
        """sort=rating pages 1..100 (server cap). Returns True if it did work."""
        page = self.st.kv_get(f"top_page:{fmt}", 1)
        if page is None or page > 100:
            return False
        rows = self.search_page(fmt, {"sort": "rating", "page": page}, time_dependent=True)
        if rows is None:
            return True
        new = self.st.add_rows(fmt, rows)
        self.st.kv_set(f"top_page:{fmt}", None if len(rows) < 51 else page + 1)
        self.st.db.commit()
        if page % 10 == 0 or len(rows) < 51:
            log(f"top-rated {fmt} page {page}: {len(rows)} rows, {new} new, min rating "
                f"{min((r.get('rating') or 0) for r in rows) if rows else None}")
        return True

    def step_backfill(self, fmt: str) -> bool:
        if self.st.kv_get(f"backfill_done:{fmt}", False):
            return False
        before = self.st.kv_get(f"backfill_before:{fmt}")
        params = {"before": before} if before else {}
        rows = self.search_page(fmt, params, time_dependent=before is None)
        if rows is None:
            return True
        new = self.st.add_rows(fmt, rows)
        if len(rows) < 51 or (rows and rows[-1]["uploadtime"] < self.a.backfill_until):
            self.st.kv_set(f"backfill_done:{fmt}", True)
            log(f"backfill {fmt} done at before={before}")
        else:
            nxt = rows[-1]["uploadtime"]
            if before is not None and nxt >= before:  # >50 uploads in one second: step past it
                nxt = before - 1
            self.st.kv_set(f"backfill_before:{fmt}", nxt)
        if self.st.kv_get(f"topup_until:{fmt}") is None:
            self.st.kv_set(f"topup_until:{fmt}", rows[0]["uploadtime"] if rows else int(time.time()))
        self.st.db.commit()
        n = self.st.kv_get(f"backfill_pages:{fmt}", 0) + 1
        self.st.kv_set(f"backfill_pages:{fmt}", n)
        if n % 50 == 0:
            d = datetime.fromtimestamp(rows[-1]["uploadtime"], timezone.utc).date() if rows else None
            log(f"backfill {fmt}: {n} pages, reached {d}, last page {new} new rows")
        return True

    def step_topup(self, fmt: str) -> bool:
        """Every --topup-hours, walk newest pages until reaching rows we already have."""
        last = self.st.kv_get(f"topup_last:{fmt}", 0)
        if time.time() - last < self.a.topup_hours * 3600:
            return False
        cursor = self.st.kv_get(f"topup_cursor:{fmt}")
        params = {"before": cursor} if cursor else {}
        rows = self.search_page(fmt, params, time_dependent=True)
        if rows is None:
            return True
        new = self.st.add_rows(fmt, rows)
        until = self.st.kv_get(f"topup_until:{fmt}") or 0
        if len(rows) < 51 or new == 0 or rows[-1]["uploadtime"] <= until:
            self.st.kv_set(f"topup_last:{fmt}", int(time.time()))
            self.st.kv_set(f"topup_cursor:{fmt}", None)
            newest = self.st.db.execute("SELECT max(uploadtime) FROM rows WHERE format=?", (fmt,)).fetchone()[0]
            self.st.kv_set(f"topup_until:{fmt}", newest)
        else:
            self.st.kv_set(f"topup_cursor:{fmt}", rows[-1]["uploadtime"])
        self.st.db.commit()
        return True

    # ---- replays
    def refill(self, fmt: str):
        q = self.st.db.execute(
            "SELECT id, need FROM rows WHERE format=? AND fetched=0 AND private=0 AND prio>=? "
            "ORDER BY prio DESC, uploadtime DESC LIMIT 200", (fmt, self.min_prio.get(fmt, self.a.min_prio))).fetchall()
        self.queue[fmt] = collections.deque(q)

    def shard_path(self, fmt: str, kind: str) -> str:
        hour = datetime.now(timezone.utc).strftime("%Y%m%dT%H")
        d = os.path.join(self.st.root, kind, fmt)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{hour}.jsonl.gz")

    def flush(self, fmt: str | None = None):
        if fmt is None and self.qkey:
            self.flush_queue()
        for f in ([fmt] if fmt else list(self.buf)):
            lines = self.buf.get(f)
            if not lines:
                continue
            paths = {}
            for kind in ("replays", "inputlogs"):
                sel = [x[0] for x in lines if x[2] == kind]
                if not sel:
                    continue
                paths[kind] = path = self.shard_path(f, kind)
                data = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sel).encode()
                with open(path, "ab") as fh:  # one gzip member per flush; concatenated members are valid gzip
                    fh.write(gzip.compress(data, 6))
                    fh.flush()
                    os.fsync(fh.fileno())
            for rec, meta, kind in lines:
                path = paths[kind]
                self.st.db.execute("INSERT OR REPLACE INTO replays VALUES (?,?,?,?,?,?,?)",
                                   (rec["id"], f, rec["fetched"], rec["status"],
                                    os.path.relpath(path, self.st.root), meta[0], meta[1]))
                self.st.db.execute("UPDATE rows SET fetched=? WHERE id=?",
                                   (1 if rec["status"] == 200 else -1, rec["id"]))
            self.st.db.commit()
            self.buf[f] = []

    def step_fetch(self, fmt: str) -> bool:
        if not self.queue[fmt]:
            self.refill(fmt)
            if not self.queue[fmt]:
                return False
        rid, need = self.queue[fmt].popleft()
        if self.st.db.execute("SELECT 1 FROM replays WHERE id=?", (rid,)).fetchone() or \
                any(x[0]["id"] == rid for x in self.buf[fmt]):
            return True
        if need == "inputlog":  # log already in a mirror (HolidayOugi): fetch only the small inputlog
            url = f"{BASE}/{rid}.inputlog"
        else:
            url = f"{BASE}/{rid}.json"
        status, body = self.http.get(url)
        if status < 0:
            return True
        text = body.decode("utf-8", errors="replace")
        has_il = 0
        if need == "inputlog":
            has_il = 1 if (status == 200 and text.startswith(">")) else 0
            rec = {"id": rid, "url": url, "fetched": now_iso(), "status": status, "inputlog": text}
            kind = "inputlogs"
        else:
            if status == 200:
                try:
                    has_il = 1 if json.loads(text).get("inputlog") else 0
                except json.JSONDecodeError:
                    status = -2
            rec = {"id": rid, "url": url, "fetched": now_iso(), "status": status, "body": text}
            kind = "replays"
        self.buf[fmt].append((rec, (len(body), has_il), kind))
        self.fetched_this_run += 1
        if len(self.buf[fmt]) >= self.a.flush_every:
            self.flush(fmt)
        return True

    # ---- inputlog queue file (ids whose log a mirror already has)
    def open_queue(self):
        self.qfile = None
        if not self.a.queue:
            return
        self.qkey = "queue_cursor:" + os.path.basename(self.a.queue)
        self.qpos = self.st.kv_get(self.qkey, 0)
        self.qfile = gzip.open(self.a.queue, "rt")
        for _ in range(self.qpos):
            next(self.qfile, None)
        log(f"inputlog queue {self.a.queue}: resuming at line {self.qpos}")

    def step_queue(self) -> bool:
        """Fetch the next `.inputlog` from the queue file (lines: id<TAB>format<TAB>prio, already
        priority-sorted and with excluded players removed). The cursor is saved at each flush."""
        if not self.qfile:
            return False
        line = next(self.qfile, None)
        if line is None:
            self.qfile = None
            log("inputlog queue exhausted")
            return False
        self.qpos += 1
        rid, fmt, prio = line.rstrip("\n").split("\t")[:3]
        if int(prio) < self.a.queue_min_prio:  # stop at the band limit; a later run can continue from here
            self.qpos -= 1
            self.qfile = None
            log(f"inputlog queue reached band {prio} < --queue-min-prio {self.a.queue_min_prio}; queue paused "
                f"at line {self.qpos}")
            self.flush_queue()
            return False
        if self.st.db.execute("SELECT 1 FROM replays WHERE id=? AND has_inputlog=1", (rid,)).fetchone():
            return True  # already have it from a .json fetch
        url = f"{BASE}/{rid}.inputlog"
        status, body = self.http.get(url)
        if status < 0:
            self.qpos -= 1  # interrupted: retry this one next run
            return True
        text = body.decode("utf-8", errors="replace")
        rec = {"id": rid, "url": url, "fetched": now_iso(), "status": status, "inputlog": text}
        self.qbuf.setdefault(fmt, []).append(rec)
        self.fetched_this_run += 1
        if sum(len(v) for v in self.qbuf.values()) >= self.a.flush_every:
            self.flush_queue()
        return True

    def flush_queue(self):
        for fmt, recs in self.qbuf.items():
            if not recs:
                continue
            path = self.shard_path(fmt, "inputlogs")
            data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode()
            with open(path, "ab") as fh:
                fh.write(gzip.compress(data, 6))
                fh.flush()
                os.fsync(fh.fileno())
            ok = sum(r["status"] == 200 and r["inputlog"].startswith(">") for r in recs)
            self.st.kv_set("queue_ok", self.st.kv_get("queue_ok", 0) + ok)
            self.st.kv_set("queue_miss", self.st.kv_get("queue_miss", 0) + len(recs) - ok)
        self.qbuf = {}
        if self.qkey:
            self.st.kv_set(self.qkey, self.qpos)
        self.st.db.commit()

    # ---- housekeeping
    def disk_ok(self) -> bool:
        free = shutil.disk_usage(self.st.root).free
        return free > self.a.min_free_mb * 1e6

    def maybe_sync(self, force=False):
        if not self.a.sync or (not force and time.time() - self.last_sync < self.a.sync_every):
            return
        self.last_sync = time.time()
        self.flush()
        sync_shards(self.st.root, self.a.cluster_dir)

    def run(self):
        log(f"crawl start formats={self.formats} root={self.st.root}")
        idle_logged = False
        while not STOP:
            self.maybe_sync()
            did = False
            # 1) top-rated pages first (cheap: <=100 pages per format)
            for f in self.formats:
                if self.step_top_rated(f):
                    did = True
                    break
            if did:
                continue
            nf = len(self.formats)
            self.rr += 1
            # 2) alternate search turns (backfill / periodic top-up) and fetch turns, each round-robin
            if self.rr % self.a.search_every == 0:
                for i in range(nf):
                    f = self.formats[(self.srr + i) % nf]
                    if self.step_backfill(f) or self.step_topup(f):
                        self.srr += i + 1
                        did = True
                        break
                if did:
                    continue
            if self.disk_ok():
                # fetch turns cycle over [json rows of each format] + queue_weight x [inputlog queue]
                slots = self.formats + ["<queue>"] * self.a.queue_weight
                for i in range(len(slots)):
                    f = slots[(self.frr + i) % len(slots)]
                    if self.step_queue() if f == "<queue>" else self.step_fetch(f):
                        self.frr += i + 1
                        did = True
                        break
                if did:
                    idle_logged = False
                    if self.fetched_this_run and self.fetched_this_run % 500 == 0:
                        rate = self.http.n / max(1, time.time() - self.t0)
                        log(f"fetched {self.fetched_this_run} replays this run; {rate:.2f} req/s")
                    continue
            else:
                self.maybe_sync(force=True)
                if not self.disk_ok():
                    log("low disk; sleeping 10 min")
                    nap(600)
                    continue
            other = any(self.step_backfill(f) or self.step_topup(f) for f in self.formats)
            if not other:
                if not idle_logged:
                    log("nothing to do; idling until the next top-up")
                    idle_logged = True
                self.flush()
                nap(60)
        self.flush()
        self.maybe_sync(force=True)
        log("crawl stopped cleanly")


def sync_shards(root: str, cluster_dir: str) -> bool:
    """rsync replay shards to the cluster; closed shards (not this hour's) are removed locally."""
    hour = datetime.now(timezone.utc).strftime("%Y%m%dT%H")
    ssh = "ssh -o BatchMode=yes -o ConnectTimeout=30"
    base = ["rsync", "-a", "-e", ssh]
    try:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", CLUSTER,
                        f"mkdir -p {cluster_dir}/replays {cluster_dir}/inputlogs"],
                       check=True, capture_output=True, timeout=120)
        rc = 0
        for kind in ("replays", "inputlogs"):
            src = os.path.join(root, kind)
            if not os.path.isdir(src):
                continue
            # closed shards (not this hour's): copy, then delete locally
            r = subprocess.run(base + ["--remove-source-files", "--exclude", f"{hour}.jsonl.gz",
                                       src + "/", f"{CLUSTER}:{cluster_dir}/{kind}/"],
                               capture_output=True, timeout=3600)
            rc = rc or r.returncode
        r1 = subprocess.CompletedProcess([], rc)
        # export the search index for cluster-side counting
        idx = os.path.join(root, "search-rows.jsonl.gz")
        export_rows(root, idx)
        r2 = subprocess.run(base + [idx, f"{CLUSTER}:{cluster_dir}/"], capture_output=True, timeout=600)
        ok = r1.returncode == 0 and r2.returncode == 0
        log(f"sync {'ok' if ok else 'FAILED'} (rsync rc {r1.returncode}/{r2.returncode})")
        return ok
    except (subprocess.SubprocessError, OSError) as e:
        log(f"sync failed: {e!r}")
        return False


def export_rows(root: str, out: str):
    db = sqlite3.connect(os.path.join(root, "state.sqlite"))
    tmp = out + ".tmp"
    with gzip.open(tmp, "wt") as f:
        for rid, fmt, ut, rating, players, priv, fetched in db.execute(
                "SELECT id, format, uploadtime, rating, players, private, fetched FROM rows"):
            f.write(json.dumps({"id": rid, "format": fmt, "uploadtime": ut, "rating": rating,
                                "players": json.loads(players), "private": priv, "fetched": fetched}) + "\n")
    os.replace(tmp, out)


def cmd_queue(a):
    """Write a priority-sorted inputlog queue from corpus indexes: battles from the given sources
    with no inputlog yet, excluded players removed; order = band (replay rating, else min player
    rating) desc, then newest first."""
    rows = []
    for path in a.index:
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r["source"] not in a.sources or r.get("has_inputlog") or excluded(r.get("players")):
                    continue
                pr = [x for x in (r.get("p1_rating"), r.get("p2_rating")) if x]
                rating = r.get("rating") if r.get("rating") is not None else (min(pr) if len(pr) == 2 else None)
                p = prio_of(rating)
                if p < a.min_prio:
                    continue
                rows.append((-p, -(r.get("uploadtime") or r.get("t0") or 0), r["id"], r["format"]))
    rows.sort()
    with gzip.open(a.out, "wt") as f:
        for p, t, rid, fmt in rows:
            f.write(f"{rid}\t{fmt}\t{-p}\n")
    c = collections.Counter(-p for p, *_ in rows)
    log(f"queue {a.out}: {len(rows)} ids; by band {dict(sorted(c.items(), reverse=True))}")


def cmd_queue_hf(a):
    """Light queue builder straight from HolidayOugi parquet metadata columns (no logs read):
    main-server/smogtours ids of the format, excluded players removed, band from the replay
    rating, newest first. Lets the inputlog crawl start before the full normalise has run."""
    import pyarrow.parquet as pq
    rows, seen = [], set()
    drops = collections.Counter()
    for fmt in a.formats:
        gen = re.match(r"gen(\d+)", fmt).group(1)
        files = sorted(p for p in glob.glob(os.path.join(a.holidayougi_dir, f"[[]Gen {gen}[]] RANDOMBATTLE*.parquet"))
                       if re.fullmatch(rf"\[Gen {gen}\] RANDOMBATTLE(_part\d+)?\.parquet", os.path.basename(p)))
        for path in files:
            t = pq.read_table(path, columns=["id", "players", "uploadtime", "rating", "formatid"]).to_pylist()
            for r in t:
                rid = r["id"]
                if rid in seen:
                    drops["duplicate"] += 1
                    continue
                seen.add(rid)
                if (r["formatid"] or "").lower() != fmt or not re.fullmatch(rf"(smogtours-)?{fmt}-\d+", rid):
                    drops["other_server_or_format"] += 1
                    continue
                if excluded(r["players"]):
                    drops["excluded_player"] += 1
                    continue
                rating = r["rating"] if r["rating"] is not None and r["rating"] == r["rating"] else None
                p = prio_of(rating)
                if p < a.min_prio:
                    continue
                rows.append((-p, -(r["uploadtime"] or 0), rid, fmt))
    rows.sort()
    with gzip.open(a.out, "wt") as f:
        for p, t, rid, fmt in rows:
            f.write(f"{rid}\t{fmt}\t{-p}\n")
    c = collections.Counter((fmt, -p) for p, _, _, fmt in rows)
    log(f"queue {a.out}: {len(rows)} ids; drops {dict(drops)}; by (format, band) {dict(sorted(c.items()))}")


def cmd_import_index(a):
    """Queue `.inputlog` fetches for battles whose log we already have from a mirror.

    Priority uses the replay-level rating when present, else min(p1_rating, p2_rating)."""
    st = State(a.root)
    n = new = upd = exc = 0
    with gzip.open(a.index, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] not in a.sources or r.get("has_inputlog"):
                continue
            n += 1
            pr = [x for x in (r.get("p1_rating"), r.get("p2_rating")) if x]
            rating = r.get("rating") if r.get("rating") is not None else (min(pr) if len(pr) == 2 else None)
            ex = excluded(r.get("players"))
            exc += ex
            cur = st.db.execute(
                "INSERT OR IGNORE INTO rows (id, format, uploadtime, rating, players, private, prio, fetched, need) "
                "VALUES (?,?,?,?,?,0,?,?,'inputlog')",
                (r["id"], r["format"], r.get("uploadtime") or r.get("t0"), rating, json.dumps(r.get("players")),
                 prio_of(rating), -3 if ex else 0))
            if cur.rowcount:
                new += 1
            else:
                upd += st.db.execute("UPDATE rows SET need='inputlog' WHERE id=? AND fetched=0", (r["id"],)).rowcount
    st.db.commit()
    log(f"import-index: {n} candidate ids, {new} new rows, {upd} existing rows switched to inputlog-only, "
        f"{exc} excluded (players in {sorted(EXCLUDE_PLAYERS)})")


# --------------------------------------------------------------------------- stats
def month_of(ts) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m") if ts else "unknown"


def cmd_stats(a):
    db = sqlite3.connect(os.path.join(a.root, "state.sqlite"))
    for fmt, in db.execute("SELECT DISTINCT format FROM rows").fetchall():
        n, rated, lo, hi = db.execute(
            "SELECT count(*), count(rating), min(uploadtime), max(uploadtime) FROM rows WHERE format=?",
            (fmt,)).fetchone()
        print(f"\n## {fmt}: {n} search rows ({rated} rated), {month_of(lo)} .. {month_of(hi)}")
        for k, v in db.execute("SELECT key, value FROM kv WHERE key LIKE ? ORDER BY key", (f"%:{fmt}",)):
            print(f"   {k} = {json.loads(v)}")
        print("   replay-rating >= " + "  ".join(
            f"{t}: {db.execute('SELECT count(*) FROM rows WHERE format=? AND rating>=?', (fmt, t)).fetchone()[0]}"
            for t in THRESHOLDS))
        print("   queue by priority band (prio: pending / done / failed / excluded; need):")
        for prio, need, pend, done, fail, exc in db.execute(
                "SELECT prio, need, sum(fetched=0), sum(fetched=1), sum(fetched=-1), sum(fetched=-3) FROM rows "
                "WHERE format=? GROUP BY prio, need ORDER BY prio DESC, need", (fmt,)):
            print(f"     prio {prio} {need:8}: {pend} / {done} / {fail} / {exc}")
        f_ok, f_il = db.execute("SELECT count(*), sum(has_inputlog) FROM replays WHERE format=? AND status=200",
                                (fmt,)).fetchone()
        f_bad = db.execute("SELECT count(*) FROM replays WHERE format=? AND status!=200", (fmt,)).fetchone()[0]
        print(f"   fetched: {f_ok} ok ({f_il or 0} with inputlog), {f_bad} failed")
        print("   inputlog coverage of fetched (month: fetched / with inputlog):  " + "  ".join(
            f"{m}: {nf}/{ni}" for m, nf, ni in db.execute(
                "SELECT strftime('%Y-%m', r.uploadtime, 'unixepoch') m, count(*), sum(p.has_inputlog) "
                "FROM replays p JOIN rows r USING (id) WHERE p.format=? GROUP BY m ORDER BY m", (fmt,))))
        print("   month     rows   rated  >=1500 >=1700 >=1800 >=1900 >=2000")
        for row in db.execute(
                "SELECT strftime('%Y-%m', uploadtime, 'unixepoch') m, count(*), count(rating), "
                + ", ".join(f"sum(rating>={t})" for t in THRESHOLDS)
                + " FROM rows WHERE format=? GROUP BY m ORDER BY m", (fmt,)):
            print("   " + "  ".join(str(x if x is not None else 0).rjust(6) for x in row))


# --------------------------------------------------------------------------- normalise
PLAYER_RE = re.compile(r"^\|player\|(p[12])\|([^|]*)\|[^|]*\|?(\d*)", re.M)
RAW_RATING_RE = re.compile(r"^\|raw\|(.+?)'s rating: (\d+) &rarr; <strong>(\d+)</strong>", re.M)
TURN_RE = re.compile(r"^\|turn\|(\d+)", re.M)
WIN_RE = re.compile(r"^\|win\|(.*)$", re.M)
T_RE = re.compile(r"^\|t:\|(\d+)", re.M)
GAMETYPE_RE = re.compile(r"^\|gametype\|(\w+)", re.M)
TIER_RE = re.compile(r"^\|tier\|(.*)$", re.M)


def tier_to_format(tier: str) -> str:
    """'[Gen 9] Random Battle' -> 'gen9randombattle'."""
    return toid(tier)


def parse_log(log_text: str) -> dict:
    players = {}
    ratings = {}
    for m in PLAYER_RE.finditer(log_text):
        side, name, rating = m.group(1), m.group(2), m.group(3)
        if name and side not in players:
            players[side] = name
        if rating and side not in ratings:
            ratings[side] = int(rating)
    raw = {toid(m.group(1)): (int(m.group(2)), int(m.group(3))) for m in RAW_RATING_RE.finditer(log_text)}
    for side, name in players.items():
        if side not in ratings and toid(name) in raw:
            ratings[side] = raw[toid(name)][0]
    turns = [int(x) for x in TURN_RE.findall(log_text)]
    wins = WIN_RE.findall(log_text)
    winner = None
    if wins:
        w = toid(wins[-1])
        winner = next((s for s, n in players.items() if toid(n) == w), None)
    elif re.search(r"^\|tie\|?", log_text, re.M):
        winner = "tie"
    t = T_RE.search(log_text)
    gt = GAMETYPE_RE.search(log_text)
    tier = TIER_RE.search(log_text)
    return {"players": [players.get("p1"), players.get("p2")], "p1_rating": ratings.get("p1"),
            "p2_rating": ratings.get("p2"), "winner": winner, "has_result": bool(wins) or winner == "tie",
            "n_turns": max(turns) if turns else 0, "t0": int(t.group(1)) if t else None,
            "gametype": gt.group(1) if gt else None,
            "tier_format": tier_to_format(tier.group(1)) if tier else None,
            "raw_ratings_after": {s: raw[toid(n)][1] for s, n in players.items() if toid(n) in raw}}


class PartWriter:
    def __init__(self, outdir: str, fmt: str, target_mb: float):
        self.dir = os.path.join(outdir, fmt)
        os.makedirs(self.dir, exist_ok=True)
        for old in glob.glob(os.path.join(self.dir, "part-*.jsonl.gz")) + \
                glob.glob(os.path.join(self.dir, "index.jsonl.gz")):
            os.remove(old)
        self.target = target_mb * 1e6
        self.n = -1
        self.raw = self.gz = None
        self.index = gzip.open(os.path.join(self.dir, "index.jsonl.gz"), "wt", compresslevel=6)
        self.count = 0
        self.inputlog_ids: set[str] = set()
        self._next()

    def _next(self):
        if self.gz:
            self.gz.close()
            self.raw.close()
        self.n += 1
        self.name = f"part-{self.n:04d}.jsonl.gz"
        self.raw = open(os.path.join(self.dir, self.name), "wb")
        self.gz = gzip.GzipFile(fileobj=self.raw, mode="wb", compresslevel=6)

    def write(self, rec: dict, extra_index: dict):
        if self.raw.tell() >= self.target:
            self._next()
        self.gz.write((json.dumps(rec, ensure_ascii=False) + "\n").encode())
        idx = {k: rec[k] for k in ("id", "format", "source", "uploadtime", "rating",
                                   "p1_rating", "p2_rating", "winner", "players")}
        idx.update(extra_index)
        idx["has_inputlog"] = bool(rec.get("inputlog")) or rec["id"] in self.inputlog_ids
        idx["part"] = self.name
        self.index.write(json.dumps(idx) + "\n")
        self.count += 1

    def close(self):
        self.gz.close()
        self.raw.close()
        self.index.close()


def iter_showdown(shard_dir: str, fmt: str):
    for path in sorted(glob.glob(os.path.join(shard_dir, "replays", fmt, "*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            try:
                for line in f:
                    rec = json.loads(line)
                    if rec["status"] != 200:
                        continue
                    d = json.loads(rec["body"])
                    yield {"id": d["id"], "uploadtime": d.get("uploadtime"), "rating": d.get("rating"),
                           "players": d.get("players"), "log": d.get("log") or "", "formatid": d.get("formatid"),
                           "inputlog": d.get("inputlog") or None, "private": d.get("private"),
                           "provenance": {"url": rec["url"], "fetched": rec["fetched"],
                                          "dataset_revision": None, "shard": os.path.basename(path)}}
            except EOFError:  # truncated final member (crash mid-write): keep what was read
                pass


def iter_parquet(files, columns, batch=256):
    import pyarrow.parquet as pq  # only needed for the HF sources
    for path in files:
        pf = pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=batch, columns=columns):
            for r in b.to_pylist():
                yield path, r


def iter_holidayougi(hdir: str, fmt: str):
    man = json.load(open(os.path.join(hdir, "manifest.json")))
    gen = re.match(r"gen(\d+)", fmt).group(1)
    pat = f"[[]Gen {gen}[]] RANDOMBATTLE"
    files = sorted(p for p in glob.glob(os.path.join(hdir, f"{pat}*.parquet"))
                   if re.fullmatch(rf"\[Gen {gen}\] RANDOMBATTLE(_part\d+)?\.parquet", os.path.basename(p)))
    for path, r in iter_parquet(files, ["id", "players", "log", "uploadtime", "formatid", "rating"]):
        rel = os.path.basename(path)
        yield {"id": r["id"], "uploadtime": r["uploadtime"],
               "rating": int(r["rating"]) if r.get("rating") is not None and r["rating"] == r["rating"] else None,
               "players": r["players"], "log": r["log"] or "", "formatid": r["formatid"], "inputlog": None,
               "provenance": {"url": man["files"].get(rel, {}).get("url"), "fetched": man["files"].get(rel, {}).get("fetched"),
                              "dataset_revision": f"{man['repo']}@{man['revision']}", "file": rel}}


def iter_pokechamp(pdir: str, fmt: str):
    man = json.load(open(os.path.join(pdir, "manifest.json")))
    files = sorted(glob.glob(os.path.join(pdir, "data", "*.parquet")))
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    for path in files:
        pf = pq.ParquetFile(path)
        rel = os.path.relpath(path, pdir)
        for b in pf.iter_batches(batch_size=1024, columns=["gamemode", "battle_id", "elo", "month_year", "text"]):
            b = b.filter(pc.equal(b.column("gamemode"), fmt))
            for r in b.to_pylist():
                num = r["battle_id"].split("-")[0]
                yield {"id": f"{fmt}-{num}", "uploadtime": None, "rating": None, "players": None,
                       "log": r["text"] or "", "formatid": r["gamemode"], "inputlog": None,
                       "provenance": {"url": man["files"].get(rel, {}).get("url"),
                                      "fetched": man["files"].get(rel, {}).get("fetched"),
                                      "dataset_revision": f"{man['repo']}@{man['revision']}", "file": rel,
                                      "elo_bucket": r["elo"], "month_year": r["month_year"],
                                      "battle_id": r["battle_id"], "note": "log is lowercased by PokeChamp"}}


def consolidate_inputlogs(showdown_dir: str, fmt: str, out_dir: str, part_mb: float):
    """Gather every fetched inputlog (from .inputlog fetches and .json bodies) into
    <out_dir>/<fmt>/part-NNNN.jsonl.gz lines {id, inputlog, fetched} plus index.jsonl.gz {id, version, part}."""
    d = os.path.join(out_dir, fmt)
    os.makedirs(d, exist_ok=True)
    for old in glob.glob(os.path.join(d, "*.jsonl.gz")):
        os.remove(old)
    ids: set[str] = set()
    versions = collections.Counter()
    seed_kind = collections.Counter()
    n_part, raw, gz = -1, None, None
    index = gzip.open(os.path.join(d, "index.jsonl.gz"), "wt")

    def records():
        for path in sorted(glob.glob(os.path.join(showdown_dir, "inputlogs", fmt, "*.jsonl.gz"))):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        if r["status"] == 200 and r["inputlog"].startswith(">"):
                            yield r["id"], r["inputlog"], r["fetched"]
            except EOFError:
                pass
        for r in iter_showdown(showdown_dir, fmt):
            if r["inputlog"]:
                yield r["id"], r["inputlog"], r["provenance"]["fetched"]

    for rid, il, fetched in records():
        if rid in ids:
            continue
        ids.add(rid)
        if raw is None or raw.tell() >= part_mb * 1e6:
            if gz:
                gz.close()
                raw.close()
            n_part += 1
            name = f"part-{n_part:04d}.jsonl.gz"
            raw = open(os.path.join(d, name), "wb")
            gz = gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6)
        gz.write((json.dumps({"id": rid, "inputlog": il, "fetched": fetched}) + "\n").encode())
        first = il.split("\n", 2)
        ver = first[0][len(">version "):].strip() if first[0].startswith(">version") else None
        versions[ver] += 1
        start = next((x for x in first[1:] if x.startswith(">start")), "")
        seed_kind["sodium" if '"seed":"sodium' in start else ("array" if '"seed":[' in start else "other")] += 1
        index.write(json.dumps({"id": rid, "version": ver, "part": name}) + "\n")
    if gz:
        gz.close()
        raw.close()
    index.close()
    return ids, versions, seed_kind


def cmd_normalise(a):
    out = a.out
    os.makedirs(out, exist_ok=True)
    report = {}
    for fmt in a.formats:
        il_ids, versions, seed_kind = set(), collections.Counter(), collections.Counter()
        if a.showdown_dir:
            il_ids, versions, seed_kind = consolidate_inputlogs(
                a.showdown_dir, fmt, a.inputlog_out or os.path.join(os.path.dirname(out.rstrip("/")), "inputlogs"),
                a.part_mb)
            log(f"{fmt}: {len(il_ids)} inputlogs, {len(versions)} Showdown versions, seeds {dict(seed_kind)}")
        w = PartWriter(out, fmt, a.part_mb)
        w.inputlog_ids = il_ids
        seen: set[str] = set()
        drops = collections.Counter()
        kept = collections.Counter()
        sources = []
        if a.showdown_dir:
            sources.append(("showdown-replays", iter_showdown(a.showdown_dir, fmt)))
        if a.holidayougi_dir:
            sources.append(("holidayougi", iter_holidayougi(a.holidayougi_dir, fmt)))
        if a.pokechamp_dir:
            sources.append(("pokechamp", iter_pokechamp(a.pokechamp_dir, fmt)))
        for src, it in sources:
            t0 = time.time()
            for r in it:
                rid = r["id"]
                if rid in seen:
                    drops[(src, "duplicate")] += 1
                    continue
                if (r.get("formatid") or "").lower() != fmt:
                    drops[(src, "wrong_format")] += 1
                    continue
                if not re.fullmatch(rf"(smogtours-)?{fmt}-\d+", rid):  # main server + Smogon tour server only
                    drops[(src, "other_server:" + rid.split("-")[0])] += 1
                    continue
                p = parse_log(r["log"])
                if p["gametype"] != "singles" or not all(p["players"]):
                    drops[(src, "not_2p_singles")] += 1
                    continue
                if p["tier_format"] and p["tier_format"] != fmt:
                    drops[(src, "wrong_format")] += 1
                    continue
                if not p["has_result"]:
                    drops[(src, "no_win_or_tie")] += 1
                    continue
                if p["n_turns"] < 2:
                    drops[(src, "lt_2_turns")] += 1
                    continue
                if excluded(r["players"] or p["players"]):
                    drops[(src, "excluded_player")] += 1
                    continue
                seen.add(rid)
                rec = {"id": rid, "format": fmt, "source": src, "uploadtime": r["uploadtime"],
                       "players": r["players"] or p["players"], "rating": r["rating"],
                       "p1_rating": p["p1_rating"], "p2_rating": p["p2_rating"], "winner": p["winner"],
                       "log": r["log"], "inputlog": r["inputlog"], "t0": p["t0"],
                       "provenance": r["provenance"]}
                extra = {"n_turns": p["n_turns"], "t0": p["t0"], "ratings_after": p["raw_ratings_after"]}
                if src == "pokechamp":
                    extra["elo_bucket"] = r["provenance"]["elo_bucket"]
                w.write(rec, extra)
                kept[src] += 1
            log(f"{fmt} {src}: kept {kept[src]}, {time.time() - t0:.0f}s")
        w.close()
        report[fmt] = {"kept": dict(kept), "dropped": {f"{s}:{k}": v for (s, k), v in sorted(drops.items())},
                       "parts": w.n + 1, "built": now_iso(), "inputlogs": len(il_ids),
                       "inputlog_seed_kind": dict(seed_kind),
                       "inputlog_versions": dict(versions.most_common()),
                       "excluded_players": sorted(EXCLUDE_PLAYERS)}
        log(f"{fmt}: {json.dumps(report[fmt])}")
    with open(os.path.join(out, "build-report.json"), "w") as f:
        json.dump(report, f, indent=1)


def cmd_corpus_stats(a):
    for fmt in a.formats:
        path = os.path.join(a.out, fmt, "index.jsonl.gz")
        if not os.path.exists(path):
            continue
        by_src = collections.defaultdict(collections.Counter)
        by_month = collections.defaultdict(collections.Counter)
        rating_rel = collections.Counter()
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                s = r["source"]
                c = by_src[s]
                c["games"] += 1
                c["inputlog"] += r["has_inputlog"]
                pr = [x for x in (r["p1_rating"], r["p2_rating"]) if x]
                if r["rating"] is not None:
                    c["has_rating"] += 1
                if len(pr) == 2:
                    c["has_both_player_ratings"] += 1
                for t in THRESHOLDS:
                    if r["rating"] is not None and r["rating"] >= t:
                        c[f"rating>={t}"] += 1
                    if pr and max(pr) >= t:
                        c[f"max>={t}"] += 1
                    if len(pr) == 2 and min(pr) >= t:
                        c[f"min>={t}"] += 1
                        c[f"il_min>={t}"] += r["has_inputlog"]
                ts = r["uploadtime"] or r.get("t0")
                m = month_of(ts)
                mc = by_month[m]
                mc["games"] += 1
                mc[s] += 1
                mc["inputlog"] += r["has_inputlog"]
                for t in (1500, 1800, 2000):
                    if len(pr) == 2 and min(pr) >= t:
                        mc[f"min>={t}"] += 1
                c["first"] = min(c["first"] or 10**12, ts or 10**12)
                c["last"] = max(c["last"], ts or 0)
                # what is the replay-level rating? compare with pre/post player ratings
                if r["rating"] and len(pr) == 2:
                    after = list((r.get("ratings_after") or {}).values())
                    if r["rating"] == min(pr):
                        rating_rel["=min(pre)"] += 1
                    elif after and r["rating"] == min(after):
                        rating_rel["=min(post)"] += 1
                    elif abs(r["rating"] - sum(pr) / 2) <= 1:
                        rating_rel["=mean(pre)"] += 1
                    else:
                        rating_rel["other"] += 1
        print(f"\n## {fmt}")
        for s, c in by_src.items():
            print(f"  {s}: games {c['games']}, {month_of(c['first'])}..{month_of(c['last'])}, "
                  f"replay rating {c['has_rating']}, both player ratings {c['has_both_player_ratings']}, "
                  f"inputlog {c['inputlog']}")
            for kind in ("rating", "max", "min", "il_min"):
                print(f"     {kind:6}>= " + "  ".join(f"{t}: {c[f'{kind}>={t}']}" for t in THRESHOLDS))
        print(f"  replay rating vs player ratings: {dict(rating_rel)}")
        print("  month    games  " + "  ".join(sorted({k for mc in by_month.values() for k in mc if k != 'games'})))
        keys = sorted({k for mc in by_month.values() for k in mc if k != "games"})
        for m in sorted(by_month):
            mc = by_month[m]
            print(f"  {m}  {mc['games']:7d}  " + "  ".join(f"{mc[k]:7d}" for k in keys))


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crawl")
    c.add_argument("--formats", nargs="+", default=["gen9randombattle", "gen4randombattle"])
    c.add_argument("--root", default=DEFAULT_ROOT)
    c.add_argument("--min-interval", type=float, default=1.0, help="seconds between requests (floor 1.0)")
    c.add_argument("--min-prio", type=int, default=0, help="only fetch replays with priority >= this "
                   "(0 all, 1 rated, 2 >=1500, 3 >=1700, 4 >=1800, 5 >=1900; replay-level rating)")
    c.add_argument("--format-min-prio", nargs="*", default=[], metavar="FORMAT=P",
                   help="per-format override of --min-prio, e.g. gen4randombattle=2")
    c.add_argument("--queue", help="gzip text file of id<TAB>format<TAB>prio (priority-sorted) whose "
                   ".inputlog to fetch; see the `queue` subcommand")
    c.add_argument("--queue-min-prio", type=int, default=0,
                   help="stop the queue at the first id below this band (2 = replay rating >= 1500)")
    c.add_argument("--queue-weight", type=int, default=2,
                   help="queue fetches per round of one json fetch per format")
    c.add_argument("--backfill-until", type=int, default=0,
                   help="stop the search backfill at this unix time (0 = walk all history)")
    c.add_argument("--search-every", type=int, default=4,
                   help="every Nth request is a search page (backfill/top-up); the rest fetch replays")
    c.add_argument("--flush-every", type=int, default=100)
    c.add_argument("--topup-hours", type=float, default=6.0)
    c.add_argument("--sync", action="store_true", help="periodically rsync shards to the cluster")
    c.add_argument("--sync-every", type=float, default=3600.0)
    c.add_argument("--cluster-dir", default=CLUSTER_RAW)
    c.add_argument("--min-free-mb", type=float, default=150.0)
    s = sub.add_parser("stats")
    s.add_argument("--root", default=DEFAULT_ROOT)
    q = sub.add_parser("queue", help="build a priority-sorted .inputlog queue file from corpus indexes")
    q.add_argument("--index", nargs="+", required=True)
    q.add_argument("--sources", nargs="+", default=["holidayougi", "pokechamp"])
    q.add_argument("--min-prio", type=int, default=0)
    q.add_argument("--out", required=True)
    qh = sub.add_parser("queue-hf", help="build the inputlog queue from HolidayOugi parquet metadata only")
    qh.add_argument("--formats", nargs="+", default=["gen9randombattle", "gen4randombattle"])
    qh.add_argument("--holidayougi-dir", required=True)
    qh.add_argument("--min-prio", type=int, default=0)
    qh.add_argument("--out", required=True)
    im = sub.add_parser("import-index", help="queue inputlog fetches for ids in a corpus index")
    im.add_argument("--root", default=DEFAULT_ROOT)
    im.add_argument("--index", required=True, help="corpus index.jsonl.gz (from normalise)")
    im.add_argument("--sources", nargs="+", default=["holidayougi", "pokechamp"])
    y = sub.add_parser("sync")
    y.add_argument("--root", default=DEFAULT_ROOT)
    y.add_argument("--cluster-dir", default=CLUSTER_RAW)
    n = sub.add_parser("normalise")
    n.add_argument("--formats", nargs="+", default=["gen9randombattle", "gen4randombattle"])
    n.add_argument("--showdown-dir", help="dir containing replays/<format>/*.jsonl.gz")
    n.add_argument("--holidayougi-dir")
    n.add_argument("--pokechamp-dir")
    n.add_argument("--out", required=True)
    n.add_argument("--part-mb", type=float, default=80.0)
    n.add_argument("--inputlog-out", help="default: <out>/../inputlogs")
    cs = sub.add_parser("corpus-stats")
    cs.add_argument("--formats", nargs="+", default=["gen9randombattle", "gen4randombattle"])
    cs.add_argument("--out", required=True)
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    if a.cmd == "crawl":
        Crawler(a).run()
    elif a.cmd == "stats":
        cmd_stats(a)
    elif a.cmd == "queue":
        cmd_queue(a)
    elif a.cmd == "queue-hf":
        cmd_queue_hf(a)
    elif a.cmd == "import-index":
        cmd_import_index(a)
    elif a.cmd == "sync":
        sync_shards(a.root, a.cluster_dir)
    elif a.cmd == "normalise":
        cmd_normalise(a)
    elif a.cmd == "corpus-stats":
        cmd_corpus_stats(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
