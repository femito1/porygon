#!/usr/bin/env python3
"""litsearch: cached, polite literature search for a systematic review.

Sources: OpenAlex, Semantic Scholar Graph API (s2), arXiv. Standard library only.

  search "QUERY" [--sources openalex,s2,arxiv] [--limit N] [--year-from Y]
  refs ID / cites ID [--sources openalex,s2] [--limit N] [--new-only]
  show ID
  screen ID[,ID...] include|exclude|maybe|unscreened ["note"]
  list [--status S[,S]] [--grep REGEX] [--source SRC] [--prov TEXT] [--sort id|year|cites|title]
  export [--status include,maybe] [--format md|csv|jsonl] [--out FILE|-]
  stats

ID is a corpus id (p0012), DOI, arXiv id (2603.15563), OpenAlex id (W123...),
S2 paper id (40 hex chars) or a title fragment.
Common options (before or after the subcommand): --json --refresh --offline
--width N --max-retries N.

Files: <lit>/corpus.sqlite (the corpus and screening decisions) and
<lit>/cache/http.sqlite (every HTTP response, zlib-compressed, keyed by URL hash).
<lit> defaults to ../lit next to this script; override with LITSEARCH_DIR.
Optional env: OPENALEX_API_KEY, S2_API_KEY, LITSEARCH_MAILTO (OpenAlex polite pool).
Diagnostics (retries, cache hits) go to stderr, prefixed with '# '.
"""

import argparse
import concurrent.futures
import csv
import difflib
import email.utils
import hashlib
import html
import http.client
import io
import json
import os
import random
import re
import shlex
import shutil
import socket
import sqlite3
import sys
import textwrap
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LIT_DIR = os.environ.get("LITSEARCH_DIR") or os.path.normpath(os.path.join(HERE, "..", "lit"))
USER_AGENT = "porygon-litsearch/1.0 (academic literature review)"

SOURCES = ("openalex", "s2", "arxiv")
SRC_LETTER = {"openalex": "o", "s2": "s", "arxiv": "a"}
HOST_SOURCE = {"api.openalex.org": "openalex", "api.semanticscholar.org": "s2",
               "export.arxiv.org": "arxiv"}
# Minimum seconds between requests to a host, shared across processes via the cache db.
MIN_INTERVAL = {"export.arxiv.org": 3.1, "api.semanticscholar.org": 1.1, "api.openalex.org": 0.15}
BACKOFF_BASE = {"api.semanticscholar.org": 2.0, "export.arxiv.org": 5.0}
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER = 120.0     # give up rather than wait longer than this for one request
STATUSES = ("unscreened", "include", "maybe", "exclude")
STATUS_LETTER = {"unscreened": "-", "include": "I", "maybe": "M", "exclude": "X"}

S2_FIELDS = "title,authors,year,venue,abstract,externalIds,citationCount,url"
OA_SELECT = ("id,doi,display_name,authorships,publication_year,primary_location,locations,"
             "abstract_inverted_index,cited_by_count,ids,type")
ARXIV_NEW = r"\d{4}\.\d{4,5}"
ARXIV_OLD = r"[a-z][a-z\-]*(?:\.[A-Za-z]{2})?/\d{7}"
ARXIV_RE = re.compile(rf"^(?:arxiv:\s*)?({ARXIV_NEW}|{ARXIV_OLD})(?:v\d+)?$", re.I)
ARXIV_URL_RE = re.compile(rf"arxiv\.org/(?:abs|pdf)/({ARXIV_NEW}|{ARXIV_OLD})", re.I)
DOI_RE = re.compile(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/\S+)$", re.I)
STOPWORDS = {"a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with", "by", "via",
             "from", "at", "is", "are", "as"}


class SourceError(Exception):
    """A source failed (network, HTTP error, bad payload). Reported, never fatal."""


class UserError(Exception):
    """Bad input or an unresolvable identifier. Fatal for the command."""


class SourceFailure(Exception):
    """Every source failed for a command (exit status 1)."""


def log(msg):
    print("# " + msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- HTTP

class Http:
    """GETs with a sqlite response cache, per-host throttling, backoff and retries."""

    def __init__(self, path, refresh=False, offline=False, max_retries=5):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path, self.refresh, self.offline, self.max_retries = path, refresh, offline, max_retries
        self.lock = threading.Lock()
        self.hs = defaultdict(Counter)      # host -> counters
        self.budget = {}                    # host -> remaining-budget header
        self.dead = set()                   # hosts that exhausted retries this run
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS http(key TEXT PRIMARY KEY, url TEXT, status INTEGER,
                                                fetched_at REAL, body BLOB);
                CREATE TABLE IF NOT EXISTS hosts(host TEXT PRIMARY KEY, next_at REAL);""")

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        return c

    @staticmethod
    def key(url):
        return hashlib.sha256(url.encode()).hexdigest()

    def _cached(self, url):
        c = self._conn()
        try:
            row = c.execute("SELECT status, body FROM http WHERE key=?", (self.key(url),)).fetchone()
        finally:
            c.close()
        return (row[0], zlib.decompress(row[1])) if row else None

    def _store(self, url, status, body):
        c = self._conn()
        try:
            c.execute("INSERT OR REPLACE INTO http VALUES(?,?,?,?,?)",
                      (self.key(url), url, status, time.time(), zlib.compress(body, 6)))
        finally:
            c.close()

    def is_cached(self, url):
        return not self.refresh and self._cached(url) is not None

    def invalidate(self, url):
        c = self._conn()
        try:
            c.execute("DELETE FROM http WHERE key=?", (self.key(url),))
        finally:
            c.close()

    def _reserve(self, host, interval):
        """Block until `host` may be hit again; cross-process via the hosts table."""
        while True:
            c = self._conn()
            try:
                c.execute("BEGIN IMMEDIATE")
                row = c.execute("SELECT next_at FROM hosts WHERE host=?", (host,)).fetchone()
                now = time.time()
                if row is None or row[0] <= now:
                    c.execute("INSERT OR REPLACE INTO hosts VALUES(?,?)", (host, now + interval))
                    c.execute("COMMIT")
                    return
                c.execute("COMMIT")
                wait = row[0] - now
            finally:
                c.close()
            time.sleep(min(wait, 5.0))

    def _push_back(self, host, delay):
        c = self._conn()
        try:
            c.execute("INSERT OR REPLACE INTO hosts VALUES(?,?)", (host, time.time() + delay))
        finally:
            c.close()

    def _count(self, host, what, n=1):
        with self.lock:
            self.hs[host][what] += n

    def get(self, url, headers=None, secret_query="", retries=None):
        """Return (status, body) with status 200 or 404. Raises SourceError otherwise.

        `secret_query` (API keys) is appended to the request but not to the cache key."""
        host = urllib.parse.urlsplit(url).hostname
        src = HOST_SOURCE.get(host, host)
        if not self.refresh:
            hit = self._cached(url)
            if hit:
                self._count(host, "hit")
                return hit
        if self.offline:
            raise SourceError(f"{src}: not cached and --offline given")
        if host in self.dead:
            self._count(host, "skip")
            raise SourceError(f"{src}: skipped (it already failed after retries in this run)")
        full = url + (("&" if "?" in url else "?") + secret_query if secret_query else "")
        hdrs = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        hdrs.update(headers or {})
        interval = MIN_INTERVAL.get(host, 1.0)
        base = BACKOFF_BASE.get(host, 2.0)
        max_retries = self.max_retries if retries is None else retries
        last = "?"
        for attempt in range(max_retries + 1):
            self._reserve(host, interval)
            self._count(host, "net")
            delay = None
            try:
                req = urllib.request.Request(full, headers=hdrs)
                with urllib.request.urlopen(req, timeout=45) as resp:
                    body = resp.read()
                    rem = resp.headers.get("X-RateLimit-Remaining-USD")
                    if rem is not None:
                        self.budget[host] = rem
                self._store(url, 200, body)
                return 200, body
            except urllib.error.HTTPError as e:
                code = e.code
                try:
                    detail = e.read()[:300].decode("utf-8", "replace")
                except Exception:
                    detail = ""
                if code == 404:
                    self._store(url, 404, detail.encode())
                    return 404, detail.encode()
                if code not in RETRY_STATUS:
                    self._count(host, "fail")
                    raise SourceError(f"{src}: HTTP {code}: {_squash(detail, 160)}")
                if code == 429:
                    self._count(host, "429")
                last = f"HTTP {code}"
                ra = _retry_after(e.headers.get("Retry-After"))
                if ra is not None:
                    if ra > MAX_RETRY_AFTER:
                        self._count(host, "fail")
                        self._push_back(host, ra)
                        raise SourceError(f"{src}: HTTP {code}, server asks to wait {ra:.0f}s "
                                          f"(> {MAX_RETRY_AFTER:.0f}s); giving up")
                    delay = ra + random.uniform(0.5, 2.0)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
                    http.client.HTTPException, OSError) as e:
                last = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            if attempt == max_retries:
                break
            if delay is None:   # exponential backoff with equal jitter
                cap = min(60.0, base * 2 ** attempt)
                delay = random.uniform(cap / 2, cap)
            self._push_back(host, delay)
            self._count(host, "retry")
            log(f"{src}: {last}; retry {attempt + 1}/{max_retries} in {delay:.1f}s")
            time.sleep(delay)
        self._count(host, "fail")
        with self.lock:
            self.dead.add(host)
        raise SourceError(f"{src}: gave up after {max_retries + 1} attempt(s) (last: {last})")

    def json(self, url, **kw):
        """Parsed JSON, or None on 404."""
        status, body = self.get(url, **kw)
        if status == 404:
            return None
        try:
            return json.loads(body)
        except ValueError:
            self.invalidate(url)
            src = HOST_SOURCE.get(urllib.parse.urlsplit(url).hostname, "?")
            raise SourceError(f"{src}: response was not JSON: {_squash(body[:120].decode('utf-8', 'replace'), 120)}")

    def summary(self):
        if not self.hs:
            return None
        parts = []
        for host, c in self.hs.items():
            s = f"{HOST_SOURCE.get(host, host)} net={c['net']} cache={c['hit']}"
            for k in ("429", "retry", "fail", "skip"):
                if c[k]:
                    s += f" {k}={c[k]}"
            if host in self.budget:
                s += f" budget_left=${self.budget[host]}"
            parts.append(s)
        tot_net = sum(c["net"] for c in self.hs.values())
        tot_hit = sum(c["hit"] for c in self.hs.values())
        return f"http: {' | '.join(parts)} || network requests={tot_net} cache hits={tot_hit}"


def _retry_after(v):
    if not v:
        return None
    v = v.strip()
    try:
        return max(0.0, float(v))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(v)
        return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- normalisation

def _squash(s, n=None):
    s = re.sub(r"\s+", " ", s or "").strip()
    return trunc(s, n) if n else s


def trunc(s, n):
    s = s or ""
    if n <= 0:
        return ""
    if len(s) <= n:
        return s
    cut = s[: n - 1]
    sp = cut.rfind(" ")
    if sp >= n - 12 and sp > 0:
        cut = cut[:sp]
    return cut.rstrip(" ,;:-") + "…"


LATEX_ACCENT = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303"}


def clean_text(s):
    if not s:
        return None
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    if "\\" in s:     # arXiv metadata: Pok\'emon, Pok\'{e}mon -> Pokémon
        s = re.sub(r"\{?\\([`'^\"~])\{?([A-Za-z])\}?\}?", lambda m: m.group(2) + LATEX_ACCENT[m.group(1)], s)
        s = unicodedata.normalize("NFC", s)
    s = _squash(s)
    return s or None


def norm_title(t):
    if not t:
        return None
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return t or None


def as_doi(s):
    if not s:
        return None
    m = DOI_RE.match(s.strip())
    return m.group(1).lower().rstrip(".") if m else None


def as_arxiv(s):
    if not s:
        return None
    s = s.strip()
    m = ARXIV_URL_RE.search(s)
    if m:
        return m.group(1).lower()
    m = ARXIV_RE.match(s)
    return m.group(1).lower() if m else None


def split_arxiv_doi(doi, arxiv):
    """arXiv DOIs (10.48550/arXiv.X) are turned into arXiv ids so they dedup with arXiv hits."""
    if doi and doi.startswith("10.48550/arxiv."):
        arxiv = arxiv or as_arxiv(doi.split("arxiv.", 1)[1])
        doi = None
    return doi, arxiv


def make_rec(source, **kw):
    rec = {k: kw.get(k) for k in ("title", "authors", "year", "venue", "abstract", "doi", "arxiv",
                                  "openalex", "s2", "citations", "url")}
    rec["title"] = clean_text(rec["title"])
    rec["abstract"] = clean_text(rec["abstract"])
    rec["venue"] = clean_text(rec["venue"])
    rec["authors"] = [a for a in (clean_text(x) for x in (rec["authors"] or [])) if a]
    rec["doi"], rec["arxiv"] = split_arxiv_doi(as_doi(rec["doi"]) if rec["doi"] else None,
                                               as_arxiv(rec["arxiv"]) if rec["arxiv"] else None)
    try:
        rec["year"] = int(rec["year"]) if rec["year"] else None
    except (TypeError, ValueError):
        rec["year"] = None
    rec["source"] = source
    return rec if rec["title"] else None


# --------------------------------------------------------------------------- OpenAlex

def oa_url(path, params):
    p = dict(params)
    if os.environ.get("LITSEARCH_MAILTO"):
        p["mailto"] = os.environ["LITSEARCH_MAILTO"]
    return "https://api.openalex.org" + path + ("?" + urllib.parse.urlencode(p) if p else "")


def oa_get(http, path, params, **kw):
    key = os.environ.get("OPENALEX_API_KEY")
    sq = urllib.parse.urlencode({"api_key": key}) if key else ""
    return http.json(oa_url(path, params), secret_query=sq, **kw)


def oa_short(x):
    return x.rsplit("/", 1)[-1] if x else None


def oa_abstract(inv):
    if not inv:
        return None
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    pos.sort()
    return " ".join(w for _, w in pos)


def oa_rec(w):
    if not w:
        return None
    arxiv = None
    for loc in (w.get("locations") or []):
        m = ARXIV_URL_RE.search((loc or {}).get("landing_page_url") or "")
        if m:
            arxiv = m.group(1)
            break
    pl = w.get("primary_location") or {}
    venue = (pl.get("source") or {}).get("display_name")
    doi = w.get("doi")
    url = doi or pl.get("landing_page_url") or w.get("id")
    authors = [((a or {}).get("author") or {}).get("display_name") for a in (w.get("authorships") or [])]
    return make_rec("openalex", title=w.get("display_name") or w.get("title"), authors=authors,
                    year=w.get("publication_year"), venue=venue,
                    abstract=oa_abstract(w.get("abstract_inverted_index")), doi=doi, arxiv=arxiv,
                    openalex=oa_short(w.get("id")), citations=w.get("cited_by_count"), url=url)


def oa_search(http, query, limit, year_from):
    per = min(max(limit, 1), 200)
    out, page = [], 1
    flt = [f"from_publication_date:{year_from}-01-01"] if year_from else []
    def params_for(mode, page):
        if mode == "search":
            p = {"search": query, "per-page": per, "page": page, "select": OA_SELECT}
            if flt:
                p["filter"] = ",".join(flt)
            return p
        # fallback: the title/abstract search filter, used when `search=` is throttled
        q = re.sub(r"[,|:]", " ", query)
        return {"filter": ",".join(flt + [f"title_and_abstract.search:{q}"]),
                "per-page": per, "page": page, "select": OA_SELECT}

    mode = "search"
    # A previous run fell back and cached the fallback: reuse it rather than re-hitting search=.
    if (not http.is_cached(oa_url("/works", params_for("search", 1)))
            and http.is_cached(oa_url("/works", params_for("filter", 1)))):
        mode = "filter"
    while len(out) < limit:
        params = params_for(mode, page)
        try:
            # Anonymous `search=` is throttled separately (Retry-After ~30s): on failure, do not
            # sit through retries; fall back to the filter search, which has its own budget.
            d = oa_get(http, "/works", params, retries=0 if mode == "search" else None)
        except SourceError as e:
            if mode == "search" and page == 1:
                log(f"openalex: search= failed ({e}); retrying with the title_and_abstract.search filter")
                mode = "filter"
                http.dead.discard("api.openalex.org")
                continue
            raise
        res = (d or {}).get("results") or []
        out += [r for r in map(oa_rec, res) if r]
        if len(res) < per:
            break
        page += 1
    return out[:limit]


def oa_find_seed(http, rec):
    """Locate the seed record in OpenAlex; returns the raw work (with referenced_works) or None."""
    sel = OA_SELECT + ",referenced_works"
    if rec.get("openalex"):
        w = oa_get(http, f"/works/{rec['openalex']}", {"select": sel})
        if w:
            return w
    for doi in filter(None, [rec.get("doi"),
                             f"10.48550/arxiv.{rec['arxiv']}" if rec.get("arxiv") else None]):
        w = oa_get(http, f"/works/doi:{doi}", {"select": sel})
        if w:
            return w
    if rec.get("title"):
        q = re.sub(r"[,|:]", " ", rec["title"])
        d = oa_get(http, "/works", {"filter": f"title.search:{q}", "per-page": 10, "select": sel})
        return best_title_match(rec, (d or {}).get("results") or [], lambda w: w.get("display_name"),
                                lambda w: w.get("publication_year"))
    return None


def oa_refs(http, rec, limit):
    w = oa_find_seed(http, rec)
    if not w:
        raise SourceError("openalex: seed paper not found")
    ids = [oa_short(x) for x in (w.get("referenced_works") or [])][:limit]
    out = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        d = oa_get(http, "/works", {"filter": "openalex:" + "|".join(chunk), "per-page": 50,
                                    "select": OA_SELECT})
        got = {oa_short(x.get("id")): x for x in (d or {}).get("results") or []}
        out += [r for r in (oa_rec(got.get(c)) for c in chunk) if r]
    return oa_rec(w), out


def oa_cites(http, rec, limit):
    w = oa_find_seed(http, rec)
    if not w:
        raise SourceError("openalex: seed paper not found")
    oid = oa_short(w.get("id"))
    per = min(max(limit, 1), 200)
    out, page = [], 1
    while len(out) < limit:
        d = oa_get(http, "/works", {"filter": f"cites:{oid}", "per-page": per, "page": page,
                                    "sort": "cited_by_count:desc", "select": OA_SELECT})
        res = (d or {}).get("results") or []
        out += [r for r in map(oa_rec, res) if r]
        if len(res) < per:
            break
        page += 1
    return oa_rec(w), out[:limit]


def oa_lookup(http, kind, val):
    if kind == "openalex":
        return oa_rec(oa_get(http, f"/works/{val}", {"select": OA_SELECT}))
    if kind in ("doi", "arxiv"):
        doi = val if kind == "doi" else f"10.48550/arxiv.{val}"
        return oa_rec(oa_get(http, f"/works/doi:{doi}", {"select": OA_SELECT}))
    if kind == "title":
        return oa_rec(oa_find_seed(http, {"title": val}))
    return None


# --------------------------------------------------------------------------- Semantic Scholar

def s2_get(http, path, params):
    url = "https://api.semanticscholar.org/graph/v1" + path + "?" + urllib.parse.urlencode(params)
    key = os.environ.get("S2_API_KEY")
    return http.json(url, headers={"x-api-key": key} if key else None)


def s2_rec(p):
    if not p:
        return None
    ext = p.get("externalIds") or {}
    return make_rec("s2", title=p.get("title"), authors=[a.get("name") for a in (p.get("authors") or [])],
                    year=p.get("year"), venue=p.get("venue"), abstract=p.get("abstract"),
                    doi=ext.get("DOI"), arxiv=ext.get("ArXiv"), s2=p.get("paperId"),
                    citations=p.get("citationCount"), url=p.get("url"))


def s2_search(http, query, limit, year_from):
    q = re.sub(r"(?<=\w)-(?=\w)", " ", query)   # S2 search returns nothing for hyphenated terms
    out, off = [], 0
    while len(out) < limit and off < 1000:
        params = {"query": q, "fields": S2_FIELDS, "limit": min(100, limit - len(out)), "offset": off}
        if year_from:
            params["year"] = f"{year_from}-"
        d = s2_get(http, "/paper/search", params) or {}
        data = d.get("data") or []
        out += [r for r in map(s2_rec, data) if r]
        if not data or "next" not in d:
            break
        off = d["next"]
    return out[:limit]


def s2_key(rec):
    if rec.get("s2"):
        return rec["s2"]
    if rec.get("arxiv"):
        return "arXiv:" + rec["arxiv"]
    if rec.get("doi"):
        return "DOI:" + rec["doi"]
    return None


def s2_find_seed(http, rec):
    key = s2_key(rec)
    if key:
        p = s2_get(http, f"/paper/{urllib.parse.quote(key, safe=':/')}", {"fields": S2_FIELDS})
        if p:
            return p
    if rec.get("title"):
        d = s2_get(http, "/paper/search/match", {"query": rec["title"], "fields": S2_FIELDS})
        return best_title_match(rec, (d or {}).get("data") or [], lambda p: p.get("title"),
                                lambda p: p.get("year"))
    return None


def _s2_graph(http, rec, limit, edge, inner):
    p = s2_find_seed(http, rec)
    if not p or not p.get("paperId"):
        raise SourceError("s2: seed paper not found")
    out, off = [], 0
    while len(out) < limit:
        n = min(1000, limit - len(out))
        d = s2_get(http, f"/paper/{p['paperId']}/{edge}",
                   {"fields": S2_FIELDS, "limit": n, "offset": off}) or {}
        data = d.get("data") or []
        out += [r for r in (s2_rec(x.get(inner)) for x in data) if r]
        if not data or "next" not in d:
            break
        off = d["next"]
    return s2_rec(p), out[:limit]


def s2_refs(http, rec, limit):
    return _s2_graph(http, rec, limit, "references", "citedPaper")


def s2_cites(http, rec, limit):
    return _s2_graph(http, rec, limit, "citations", "citingPaper")


def s2_lookup(http, kind, val):
    key = {"s2": val, "doi": f"DOI:{val}", "arxiv": f"arXiv:{val}"}.get(kind)
    if key:
        return s2_rec(s2_get(http, f"/paper/{urllib.parse.quote(key, safe=':/')}", {"fields": S2_FIELDS}))
    if kind == "title":
        return s2_rec(s2_find_seed(http, {"title": val}))
    return None


# --------------------------------------------------------------------------- arXiv

AX_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def ax_is_raw(query):
    return bool(re.search(r"\b(ti|au|abs|co|jr|cat|rn|id|all):", query)
                or re.search(r"\b(AND|OR|ANDNOT)\b", query))


def ax_query(query, year_from, joiner="AND"):
    """arXiv search_query. Raw arXiv syntax (ti:, abs:, AND, ...) passes through unchanged.

    arXiv indexes accented and unaccented spellings as different tokens ("Pokémon" vs
    "Pokemon"), so a non-ASCII term is searched as (term OR folded-term)."""
    if ax_is_raw(query):
        q = query
    else:
        try:
            toks = shlex.split(query)
        except ValueError:
            toks = query.split()
        terms = []
        for t in toks:
            t = _squash(re.sub(r"[^\w\s]", " ", t))
            if not t or t.lower() in STOPWORDS:
                continue
            variants = list(dict.fromkeys(
                [t, unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()]))
            forms = [f'all:"{v}"' if " " in v else f"all:{v}" for v in variants if v]
            terms.append(forms[0] if len(forms) == 1 else "(" + " OR ".join(forms) + ")")
        if not terms:
            raise SourceError("arxiv: query has no searchable terms")
        q = f" {joiner} ".join(terms)
    if year_from:
        q = f"({q}) AND submittedDate:[{year_from}01010000 TO 299912312359]"
    return q


def ax_parse(body):
    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        raise SourceError(f"arxiv: bad XML: {e}")
    out = []
    for e in root.findall("a:entry", AX_NS):
        eid = e.findtext("a:id", "", AX_NS)
        if "api/errors" in eid:
            raise SourceError("arxiv: query error: " + _squash(e.findtext("a:summary", "", AX_NS), 160))
        jref = e.findtext("arxiv:journal_ref", None, AX_NS)
        cat = e.find("arxiv:primary_category", AX_NS)
        venue = jref or ("arXiv" + (f" ({cat.get('term')})" if cat is not None else ""))
        aid = as_arxiv(eid)
        r = make_rec("arxiv", title=e.findtext("a:title", "", AX_NS),
                     authors=[a.findtext("a:name", "", AX_NS) for a in e.findall("a:author", AX_NS)],
                     year=(e.findtext("a:published", "", AX_NS) or "")[:4], venue=venue,
                     abstract=e.findtext("a:summary", "", AX_NS),
                     doi=e.findtext("arxiv:doi", None, AX_NS), arxiv=aid,
                     url=f"https://arxiv.org/abs/{aid}" if aid else eid)
        if r:
            out.append(r)
    return out


def ax_search(http, query, limit, year_from):
    def run(joiner):
        params = {"search_query": ax_query(query, year_from, joiner), "start": 0,
                  "max_results": min(limit, 500), "sortBy": "relevance"}
        status, body = http.get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params))
        return [] if status == 404 else ax_parse(body)
    out = run("AND")
    if not out and len(query.split()) > 1 and not ax_is_raw(query):
        # Every-term AND found nothing; arXiv's OR query is relevance-ranked, so its top hits
        # are the papers matching the most (and the rarest) terms.
        log("arxiv: no paper matches all terms; using the relevance-ranked OR query")
        out = run("OR")
    return out[:limit]


def ax_lookup(http, kind, val):
    if kind != "arxiv":
        return None
    status, body = http.get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": val}))
    recs = [] if status == 404 else ax_parse(body)
    return recs[0] if recs else None


def best_title_match(rec, items, get_title, get_year, threshold=0.9):
    nt = norm_title(rec.get("title"))
    best, score = None, 0.0
    for it in items:
        s = difflib.SequenceMatcher(None, nt or "", norm_title(get_title(it)) or "").ratio()
        y = get_year(it)
        if rec.get("year") and y and abs(int(y) - int(rec["year"])) > 1:
            s -= 0.2
        if s > score:
            best, score = it, s
    return best if score >= threshold else None


SEARCHERS = {"openalex": oa_search, "s2": s2_search, "arxiv": ax_search}
GRAPH = {"refs": {"openalex": oa_refs, "s2": s2_refs}, "cites": {"openalex": oa_cites, "s2": s2_cites}}
LOOKUPS = (("arxiv", ax_lookup), ("s2", s2_lookup), ("openalex", oa_lookup))


# --------------------------------------------------------------------------- corpus

CORPUS_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers(
  id TEXT PRIMARY KEY, title TEXT, norm_title TEXT, authors TEXT, year INTEGER, venue TEXT,
  abstract TEXT, doi TEXT, arxiv TEXT, openalex TEXT, s2 TEXT, citations INTEGER, url TEXT,
  sources TEXT, provenance TEXT, status TEXT NOT NULL DEFAULT 'unscreened', note TEXT,
  screened_at TEXT, first_seen TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS i_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS i_arxiv ON papers(arxiv);
CREATE INDEX IF NOT EXISTS i_oa ON papers(openalex);
CREATE INDEX IF NOT EXISTS i_s2 ON papers(s2);
CREATE INDEX IF NOT EXISTS i_nt ON papers(norm_title);
CREATE TABLE IF NOT EXISTS aliases(alias TEXT PRIMARY KEY, target TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS screening_log(ts TEXT, id TEXT, status TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""
ID_FIELDS = ("doi", "arxiv", "openalex", "s2")
COLS = ("id", "title", "norm_title", "authors", "year", "venue", "abstract", "doi", "arxiv",
        "openalex", "s2", "citations", "url", "sources", "provenance", "status", "note",
        "screened_at", "first_seen", "updated_at")
MIN_TITLE_MATCH = 20      # shorter normalised titles ("introduction") never dedup by title


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def pid_num(pid):
    return int(pid[1:])


def compatible(a, b):
    for f in ("doi", "arxiv"):
        if a.get(f) and b.get(f) and a[f] != b[f]:
            return False
    ya, yb = a.get("year"), b.get("year")
    return not (ya and yb and abs(ya - yb) > 1)


def is_arxiv_venue(v):
    return not v or v.lower().startswith("arxiv") or v.lower() in ("unknown", "n/a")


def best_url(d):
    if d.get("doi"):
        return "https://doi.org/" + d["doi"]
    if d.get("arxiv"):
        return "https://arxiv.org/abs/" + d["arxiv"]
    return d.get("url")


def merge_fields(old, new):
    """Fill gaps in `old` from `new`; returns the merged dict (does not touch provenance)."""
    m = dict(old)
    for f in ("title", "year") + ID_FIELDS:
        if not m.get(f) and new.get(f):
            m[f] = new[f]
    if new.get("authors") and len(new["authors"]) > len(m.get("authors") or []):
        m["authors"] = new["authors"]
    if new.get("abstract") and len(new["abstract"]) > len(m.get("abstract") or ""):
        m["abstract"] = new["abstract"]
    if new.get("venue") and is_arxiv_venue(m.get("venue")) and not is_arxiv_venue(new["venue"]):
        m["venue"] = new["venue"]
    elif not m.get("venue"):
        m["venue"] = new.get("venue")
    cs = [c for c in (m.get("citations"), new.get("citations")) if c is not None]
    m["citations"] = max(cs) if cs else None
    if not m.get("url"):
        m["url"] = new.get("url")
    m["url"] = best_url(m) or m.get("url")
    return m


class Corpus:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(CORPUS_SCHEMA)
        self.created = set()

    def commit(self):
        self.db.commit()

    def _next_id(self):
        row = self.db.execute("SELECT v FROM meta WHERE k='next_id'").fetchone()
        n = int(row[0]) if row else 1
        self.db.execute("INSERT OR REPLACE INTO meta VALUES('next_id', ?)", (str(n + 1),))
        return f"p{n:04d}"

    @staticmethod
    def _decode(row):
        d = dict(row)
        for f in ("authors", "sources", "provenance"):
            d[f] = json.loads(d[f]) if d[f] else []
        return d

    def get(self, pid):
        row = self.db.execute("SELECT * FROM papers WHERE id=?", (self.canonical(pid),)).fetchone()
        return self._decode(row) if row else None

    def canonical(self, pid):
        seen = set()
        while pid not in seen:
            seen.add(pid)
            row = self.db.execute("SELECT target FROM aliases WHERE alias=?", (pid,)).fetchone()
            if not row:
                return pid
            pid = row[0]
        return pid

    def all(self):
        return [self._decode(r) for r in self.db.execute("SELECT * FROM papers")]

    def _write(self, d):
        d = dict(d)
        d["norm_title"] = norm_title(d.get("title"))
        d["updated_at"] = now_iso()
        vals = [json.dumps(d.get(c) or [], ensure_ascii=False) if c in ("authors", "sources", "provenance")
                else d.get(c) for c in COLS]
        self.db.execute(f"INSERT OR REPLACE INTO papers({','.join(COLS)}) VALUES({','.join('?' * len(COLS))})",
                        vals)

    def _match(self, rec):
        ids = []
        for f in ID_FIELDS:
            if rec.get(f):
                ids += [r[0] for r in self.db.execute(f"SELECT id FROM papers WHERE {f}=?", (rec[f],))]
        nt = norm_title(rec.get("title"))
        if nt and len(nt) >= MIN_TITLE_MATCH:
            for row in self.db.execute("SELECT * FROM papers WHERE norm_title=?", (nt,)):
                if row["id"] not in ids and compatible(rec, dict(row)):
                    ids.append(row["id"])
        return list(dict.fromkeys(ids))

    def upsert(self, rec, prov=None):
        """Add or merge a source record. Returns (canonical id, created-in-this-run)."""
        ids = self._match(rec)
        if not ids:
            pid = self._next_id()
            d = {k: rec.get(k) for k in COLS if k in rec}
            d.update(id=pid, sources=[rec["source"]], provenance=[prov] if prov else [],
                     status="unscreened", first_seen=now_iso())
            d["url"] = best_url(d) or rec.get("url")
            self._write(d)
            self.created.add(pid)
            return pid, True
        ids.sort(key=pid_num)
        target = self.get(ids[0])
        for other in ids[1:]:
            o = self.get(other)
            if o and o["id"] != target["id"] and compatible(target, o):
                target = self._absorb(target, o)
        m = merge_fields(target, rec)
        if rec["source"] not in m["sources"]:
            m["sources"] = m["sources"] + [rec["source"]]
        if prov and prov not in m["provenance"]:
            m["provenance"] = m["provenance"] + [prov]
        self._write(m)
        return m["id"], m["id"] in self.created

    def _absorb(self, target, other):
        m = merge_fields(target, other)
        m["sources"] = list(dict.fromkeys(target["sources"] + other["sources"]))
        m["provenance"] = list(dict.fromkeys(target["provenance"] + other["provenance"]))
        if target["status"] == "unscreened" and other["status"] != "unscreened":
            m["status"], m["note"], m["screened_at"] = other["status"], other["note"], other["screened_at"]
        m["first_seen"] = min(filter(None, [target.get("first_seen"), other.get("first_seen")]), default=None)
        self.db.execute("DELETE FROM papers WHERE id=?", (other["id"],))
        self.db.execute("UPDATE aliases SET target=? WHERE target=?", (target["id"], other["id"]))
        self.db.execute("INSERT OR REPLACE INTO aliases VALUES(?,?)", (other["id"], target["id"]))
        if other["id"] in self.created and target["id"] not in self.created:
            self.created.discard(other["id"])
        self._write(m)
        return m

    def find(self, ident):
        """Local lookup. Returns a list of matching corpus ids (possibly empty)."""
        s = ident.strip()
        if re.fullmatch(r"p\d+", s, re.I):
            pid = self.canonical(f"p{int(s[1:]):04d}")
            return [pid] if self.get(pid) else []
        kind, val = classify(s)
        if kind != "title":
            return [r[0] for r in self.db.execute(f"SELECT id FROM papers WHERE {kind}=?", (val,))]
        nt = norm_title(s)
        if not nt:
            return []
        rows = self.db.execute("SELECT id, norm_title FROM papers WHERE norm_title LIKE ?",
                               (f"%{nt}%",)).fetchall()
        exact = [r[0] for r in rows if r[1] == nt]
        return exact or [r[0] for r in rows]

    def screen(self, pid, status, note):
        d = self.get(pid)
        d["status"] = status
        if note is not None:
            d["note"] = note
        d["screened_at"] = now_iso()
        self._write(d)
        self.db.execute("INSERT INTO screening_log VALUES(?,?,?,?)", (d["screened_at"], d["id"], status, note))
        return d


def classify(s):
    s = s.strip()
    if as_doi(s):
        d, a = split_arxiv_doi(as_doi(s), None)
        return ("doi", d) if d else ("arxiv", a)
    if as_arxiv(s):
        return "arxiv", as_arxiv(s)
    if re.fullmatch(r"W\d{4,}", s, re.I) or re.search(r"openalex\.org/W\d+", s, re.I):
        return "openalex", "W" + re.search(r"\d+", s.split("/")[-1]).group(0)
    if re.fullmatch(r"[0-9a-f]{40}", s, re.I):
        return "s2", s.lower()
    return "title", s


# --------------------------------------------------------------------------- output

def term_width(args):
    if getattr(args, "width", None):
        return args.width
    if os.environ.get("LITSEARCH_WIDTH"):
        return int(os.environ["LITSEARCH_WIDTH"])
    return shutil.get_terminal_size((120, 40)).columns if sys.stdout.isatty() else 120


def src_letters(d):
    return "".join(SRC_LETTER[s] if s in d.get("sources", []) else "." for s in SOURCES)


def print_table(rows, width, new=None, hits=None):
    """rows: corpus dicts. new: set of ids to mark '+'. hits: id -> sources found this time."""
    vw = 22
    fixed = 1 + 6 + 1 + 2 + 5 + 7 + 4 + 2     # mark id st year cites src gaps
    tw = max(20, width - fixed - vw - 2)
    print(f"{' ':1}{'ID':<6} {'S':1} {'YEAR':>4} {'CITES':>6} {'SRC':3} {'TITLE':<{tw}}  VENUE")
    for d in rows:
        mark = "+" if new is not None and d["id"] in new else " "
        src = src_letters({"sources": hits[d["id"]]}) if hits else src_letters(d)
        cites = "-" if d.get("citations") is None else str(d["citations"])
        venue = trunc(d.get("venue") or "", vw)
        print(f"{mark}{d['id']:<6} {STATUS_LETTER.get(d['status'], '?')} {d.get('year') or '':>4} "
              f"{cites:>6} {src:3} {trunc(d['title'], tw):<{tw}}  {venue}".rstrip())


def table_legend(new_marks=True):
    s = "S=status (-=unscreened I=include M=maybe X=exclude); SRC=o:openalex s:s2 a:arxiv"
    return s + ("; '+'=new to the corpus" if new_marks else "")


def public(d):
    return {k: d.get(k) for k in COLS if k not in ("norm_title",)}


# --------------------------------------------------------------------------- commands

def parse_sources(spec, allowed):
    srcs = [s.strip().lower() for s in spec.split(",") if s.strip()]
    srcs = ["s2" if s in ("semanticscholar", "semantic_scholar") else s for s in srcs]
    bad = [s for s in srcs if s not in SOURCES]
    if bad or not srcs:
        raise UserError(f"unknown source(s) {bad or spec!r}; choose from {','.join(SOURCES)}")
    return [s for s in srcs if s in allowed] , [s for s in srcs if s not in allowed]


def run_parallel(jobs):
    """jobs: {source: callable}. Returns {source: (result or None, error string or None)}."""
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
        futs = {src: ex.submit(fn) for src, fn in jobs.items()}
        for src, fut in futs.items():
            try:
                out[src] = (fut.result(), None)
            except SourceError as e:
                out[src] = (None, str(e))
            except Exception as e:      # a bug or odd payload in one source must not kill the rest
                out[src] = (None, f"{src}: internal error {type(e).__name__}: {e}")
    return out


def ingest(ctx, results, prov, order_key=None):
    """Upsert each source's records; returns ordered canonical ids and per-id source hits."""
    c = ctx.corpus
    rank, hits = {}, defaultdict(set)
    for src, recs in results.items():
        for i, rec in enumerate(recs or []):
            pid, _ = c.upsert(rec, prov)
            rank[pid] = min(rank.get(pid, i), i)
            hits[pid].add(src)
    c.commit()
    crank, chits = {}, defaultdict(set)
    for pid, r in rank.items():
        cp = c.canonical(pid)
        crank[cp] = min(crank.get(cp, r), r)
        chits[cp] |= hits[pid]
    ids = sorted(crank, key=lambda p: (crank[p], -len(chits[p]), pid_num(p)))
    return ids, {p: sorted(s, key=SOURCES.index) for p, s in chits.items()}


def report_sources(results, unit="hits"):
    for src, (recs, err) in results.items():
        log(f"{src}: {len(recs)} {unit}" if err is None else f"{src}: FAILED - {err}")


def cmd_search(ctx, a):
    srcs, _ = parse_sources(a.sources, SOURCES)
    if a.limit < 1:
        raise UserError("--limit must be >= 1")
    res = run_parallel({s: (lambda s=s: SEARCHERS[s](ctx.http, a.query, a.limit, a.year_from)) for s in srcs})
    report_sources(res)
    if all(e for r, e in res.values()):
        raise SourceFailure("every source failed; nothing added (see messages above)")
    ids, hits = ingest(ctx, {s: r for s, (r, e) in res.items() if r}, f"search:{a.query}")
    rows = [ctx.corpus.get(p) for p in ids]
    if a.year_from:
        rows = [d for d in rows if not d.get("year") or d["year"] >= a.year_from]
    new = {d["id"] for d in rows if d["id"] in ctx.corpus.created}
    if a.json:
        print(json.dumps([dict(public(d), new=d["id"] in new, found_by=hits[d["id"]]) for d in rows],
                         ensure_ascii=False, indent=1))
        return
    ok = [s for s, (r, e) in res.items() if e is None]
    print(f"search {a.query!r}: {len(rows)} unique papers, {len(new)} new "
          f"(sources ok: {','.join(ok) or 'none'})")
    if rows:
        print_table(rows, term_width(a), new=new, hits=hits)
        print(table_legend())


def lookup_external(ctx, ident):
    kind, val = classify(ident)
    found, tried = [], []
    for src, fn in LOOKUPS:
        if kind == "s2" and src != "s2" or kind == "openalex" and src != "openalex":
            continue
        if src == "arxiv" and kind != "arxiv":
            continue
        try:
            r = fn(ctx.http, kind, val)
            tried.append(f"{src}: {'found' if r else 'not found'}")
            if r:
                found.append(r)
        except SourceError as e:
            tried.append(f"{src}: error ({e})")
    pid = None
    for r in found:
        pid, _ = ctx.corpus.upsert(r, f"lookup:{ident}")
    ctx.corpus.commit()
    if pid is None:
        raise UserError(f"could not resolve {ident!r} (as {kind}); tried: " + "; ".join(tried))
    log(f"resolved {ident!r} ({kind}) via " + "; ".join(tried))
    return ctx.corpus.canonical(pid)


def resolve(ctx, ident, fetch=True):
    ids = ctx.corpus.find(ident)
    if len(ids) == 1:
        return ids[0]
    if len(ids) > 1:
        lines = [f"  {d['id']} {d.get('year') or ''} {trunc(d['title'], 90)}"
                 for d in map(ctx.corpus.get, ids[:15])]
        raise UserError(f"{ident!r} matches {len(ids)} corpus entries; use an id:\n" + "\n".join(lines))
    if not fetch:
        raise UserError(f"{ident!r} is not in the corpus (use `show {ident}` or `search` to add it)")
    return lookup_external(ctx, ident)


def cmd_graph(ctx, a, kind):
    srcs, skipped = parse_sources(a.sources, ("openalex", "s2"))
    for s in skipped:
        log(f"{s}: has no citation graph; skipped")
    seed = resolve(ctx, a.id)
    rec = ctx.corpus.get(seed)
    log(f"seed {seed}: {trunc(rec['title'], 90)} ({rec.get('year') or '?'})")
    res = run_parallel({s: (lambda s=s: GRAPH[kind][s](ctx.http, rec, a.limit)) for s in srcs})
    for s, (r, e) in res.items():
        if r and r[0]:                       # fold the source's view of the seed into the seed record
            ctx.corpus.upsert(r[0])
    ctx.corpus.commit()
    seed = ctx.corpus.canonical(seed)
    report_sources({s: ((r[1] if r else None), e) for s, (r, e) in res.items()},
                   unit="references" if kind == "refs" else "citing works")
    if all(e for r, e in res.values()):
        raise SourceFailure("every source failed; nothing added (see messages above)")
    prov = f"{kind}:{seed}"
    ids, hits = ingest(ctx, {s: r[1] for s, (r, e) in res.items() if r}, prov)
    ids = [p for p in ids if p != ctx.corpus.canonical(seed)]
    rows = [ctx.corpus.get(p) for p in ids]
    new = {d["id"] for d in rows if d["id"] in ctx.corpus.created}
    shown = [d for d in rows if d["id"] in new] if a.new_only else rows
    if a.json:
        print(json.dumps({"seed": public(ctx.corpus.get(seed)),
                          "results": [dict(public(d), new=d["id"] in new, found_by=hits[d["id"]])
                                      for d in shown]}, ensure_ascii=False, indent=1))
        return
    verb = "references of" if kind == "refs" else "works citing"
    print(f"{verb} {seed} {trunc(rec['title'], 70)!r}: {len(rows)} unique, {len(new)} new"
          + (" (showing new only)" if a.new_only else ""))
    if shown:
        print_table(shown, term_width(a), new=new, hits=hits)
        print(table_legend())


def cmd_show(ctx, a):
    pid = resolve(ctx, a.id)
    d = ctx.corpus.get(pid)
    if a.json:
        print(json.dumps(public(d), ensure_ascii=False, indent=1))
        return
    w = min(term_width(a), 110)
    st = d["status"] + (f" ({d['screened_at']})" if d.get("screened_at") else "")
    print(f"{d['id']}  [{st}]")
    print(textwrap.fill(d["title"], w, initial_indent="Title:    ", subsequent_indent=" " * 10))
    au = d["authors"]
    print(textwrap.fill("; ".join(au) if au else "-", w, initial_indent="Authors:  ", subsequent_indent=" " * 10))
    print(f"Year:     {d.get('year') or '-'}   Venue: {d.get('venue') or '-'}   "
          f"Citations: {'-' if d.get('citations') is None else d['citations']}")
    ids = "   ".join(f"{k}: {d[k]}" for k in ("doi", "arxiv", "openalex", "s2") if d.get(k))
    print(f"IDs:      {ids or '-'}")
    print(f"URL:      {d.get('url') or '-'}")
    print(f"Sources:  {', '.join(d['sources'])}   first seen {d.get('first_seen')}")
    prov = d["provenance"]
    print(textwrap.fill(" | ".join(prov) if prov else "-", w, initial_indent="Found by: ",
                        subsequent_indent=" " * 10))
    if d.get("note"):
        print(textwrap.fill(d["note"], w, initial_indent="Note:     ", subsequent_indent=" " * 10))
    print("Abstract:")
    print(textwrap.fill(d.get("abstract") or "(none in any source)", w, initial_indent="  ",
                        subsequent_indent="  "))


def cmd_screen(ctx, a):
    pids = list(dict.fromkeys(resolve(ctx, x, fetch=False) for x in a.id.split(",") if x.strip()))
    out = [ctx.corpus.screen(pid, a.status, a.note) for pid in pids]
    ctx.corpus.commit()
    if a.json:
        print(json.dumps([public(d) for d in out], ensure_ascii=False, indent=1))
        return
    for d in out:
        print(f"{d['id']} -> {d['status']}: {trunc(d['title'], 80)}" + (f"  [{trunc(d['note'], 60)}]" if d.get("note") else ""))


def select_rows(ctx, a):
    rows = ctx.corpus.all()
    if getattr(a, "status", None):
        sts = [s.strip() for s in a.status.split(",")]
        bad = [s for s in sts if s not in STATUSES]
        if bad:
            raise UserError(f"unknown status {bad}; choose from {','.join(STATUSES)}")
        rows = [d for d in rows if d["status"] in sts]
    if getattr(a, "grep", None):
        try:
            rx = re.compile(a.grep, re.I)
        except re.error as e:
            raise UserError(f"bad --grep regex: {e}")
        rows = [d for d in rows if rx.search((d["title"] or "") + "\n" + (d.get("abstract") or ""))]
    if getattr(a, "source", None):
        rows = [d for d in rows if a.source in d["sources"]]
    if getattr(a, "prov", None):
        rows = [d for d in rows if any(a.prov.lower() in p.lower() for p in d["provenance"])]
    if getattr(a, "year_from", None):
        rows = [d for d in rows if d.get("year") and d["year"] >= a.year_from]
    key = {"id": lambda d: pid_num(d["id"]), "year": lambda d: -(d.get("year") or 0),
           "cites": lambda d: -(d.get("citations") or 0), "title": lambda d: d.get("norm_title") or ""}
    rows.sort(key=key[getattr(a, "sort", "id") or "id"])
    return rows


def cmd_list(ctx, a):
    rows = select_rows(ctx, a)
    total = len(rows)
    if a.limit:
        rows = rows[: a.limit]
    if a.json:
        print(json.dumps([public(d) for d in rows], ensure_ascii=False, indent=1))
        return
    print(f"{total} papers match" + (f" (showing {len(rows)})" if len(rows) < total else ""))
    if rows:
        print_table(rows, term_width(a))
        print(table_legend(new_marks=False))


def md_cell(s):
    return (s or "").replace("|", "\\|").replace("\n", " ")


def fmt_authors(au, n=3):
    if not au:
        return ""
    return "; ".join(au[:n]) + (" et al." if len(au) > n else "")


def cmd_export(ctx, a):
    rows = select_rows(ctx, a)
    ext = {"md": "md", "csv": "csv", "jsonl": "jsonl"}[a.format]
    out_path = a.out or os.path.join(LIT_DIR, f"export-{a.status.replace(',', '-')}.{ext}")
    buf = io.StringIO()
    if a.format == "md":
        buf.write(f"<!-- litsearch export: status={a.status}, {len(rows)} papers, {now_iso()} -->\n\n")
        buf.write("| ID | Title | Authors | Year | Venue | Link | Status | Note |\n")
        buf.write("|---|---|---|---|---|---|---|---|\n")
        for d in rows:
            link = f"[link]({d['url']})" if d.get("url") else ""
            buf.write("| " + " | ".join(md_cell(x) for x in (
                d["id"], d["title"], fmt_authors(d["authors"]), str(d.get("year") or ""), d.get("venue"),
                link, d["status"], d.get("note"))) + " |\n")
    elif a.format == "csv":
        w = csv.writer(buf)
        w.writerow(["id", "title", "authors", "year", "venue", "doi", "arxiv", "url", "citations", "status", "note"])
        for d in rows:
            w.writerow([d["id"], d["title"], "; ".join(d["authors"]), d.get("year"), d.get("venue"),
                        d.get("doi"), d.get("arxiv"), d.get("url"), d.get("citations"), d["status"], d.get("note")])
    else:
        for d in rows:
            buf.write(json.dumps(public(d), ensure_ascii=False) + "\n")
    if out_path == "-":
        sys.stdout.write(buf.getvalue())
        return
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"wrote {len(rows)} papers ({a.format}) to {out_path}")


def cmd_stats(ctx, a):
    rows = ctx.corpus.all()
    by_src, by_status, by_ptype, by_step, nsrc = Counter(), Counter(), Counter(), Counter(), Counter()
    for d in rows:
        by_src.update(d["sources"])
        nsrc[len(d["sources"])] += 1
        by_status[d["status"]] += 1
        types = set()
        for p in d["provenance"]:
            types.add(p.split(":", 1)[0])
            by_step[p] += 1
        by_ptype.update(types)
    c = sqlite3.connect(os.path.join(LIT_DIR, "cache", "http.sqlite"))
    cache_n, cache_bytes = c.execute("SELECT count(*), coalesce(sum(length(body)),0) FROM http").fetchone()
    cache_hosts = c.execute("SELECT substr(url, 9, instr(substr(url, 9), '/') - 1), count(*) "
                            "FROM http GROUP BY 1").fetchall()
    c.close()
    aliases = ctx.corpus.db.execute("SELECT count(*) FROM aliases").fetchone()[0]
    if a.json:
        print(json.dumps({"papers": len(rows), "by_source": by_src, "by_n_sources": nsrc,
                          "by_status": by_status, "by_provenance_type": by_ptype, "by_step": by_step,
                          "merged_aliases": aliases, "cache_entries": cache_n, "cache_bytes": cache_bytes},
                         indent=1))
        return
    fmt = lambda cn: ", ".join(f"{k}={v}" for k, v in sorted(cn.items(), key=lambda kv: (-kv[1], str(kv[0]))))
    print(f"papers: {len(rows)}   (merged duplicates: {aliases})")
    print(f"by source:      {fmt(by_src)}")
    print(f"in n sources:   {fmt(nsrc)}")
    print(f"by status:      {', '.join(f'{s}={by_status.get(s, 0)}' for s in STATUSES)}")
    print(f"by provenance:  {fmt(by_ptype)}")
    print("by step (papers found):")
    for step, n in sorted(by_step.items(), key=lambda kv: -kv[1])[:30]:
        print(f"  {n:5d}  {trunc(step, term_width(a) - 10)}")
    if len(by_step) > 30:
        print(f"  ... {len(by_step) - 30} more steps")
    print(f"http cache: {cache_n} responses, {cache_bytes / 1024:.0f} KiB compressed ("
          + ", ".join(f"{HOST_SOURCE.get(h, h)}={n}" for h, n in cache_hosts) + ")")


# --------------------------------------------------------------------------- main

class Ctx:
    def __init__(self, a):
        os.makedirs(os.path.join(LIT_DIR, "cache"), exist_ok=True)
        gi = os.path.join(LIT_DIR, ".gitignore")
        if not os.path.exists(gi):
            with open(gi, "w") as f:
                f.write("cache/\n")
        self.http = Http(os.path.join(LIT_DIR, "cache", "http.sqlite"), refresh=a.refresh,
                         offline=a.offline, max_retries=a.max_retries)
        self.corpus = Corpus(os.path.join(LIT_DIR, "corpus.sqlite"))


def add_common(p, suppress):
    d = (lambda v: argparse.SUPPRESS) if suppress else (lambda v: v)
    p.add_argument("--json", action="store_true", default=d(False), help="machine-readable JSON output")
    p.add_argument("--refresh", action="store_true", default=d(False), help="bypass the HTTP cache (still writes it)")
    p.add_argument("--offline", action="store_true", default=d(False), help="use the cache only; no network")
    p.add_argument("--width", type=int, default=d(None), help="table width (default: terminal or 120)")
    p.add_argument("--max-retries", type=int, default=d(5), help="retries per request on 429/5xx/network errors")


def build_parser():
    ap = argparse.ArgumentParser(prog="litsearch", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    add_common(ap, suppress=False)
    common = argparse.ArgumentParser(add_help=False)
    add_common(common, suppress=True)
    sp = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    p = sp.add_parser("search", parents=[common], help="query sources, add hits to the corpus")
    p.add_argument("query")
    p.add_argument("--sources", default="openalex,s2,arxiv")
    p.add_argument("--limit", type=int, default=25, help="max hits per source (default 25)")
    p.add_argument("--year-from", type=int)

    for name, hlp in (("refs", "backward snowballing: the paper's references"),
                      ("cites", "forward snowballing: works citing the paper")):
        p = sp.add_parser(name, parents=[common], help=hlp)
        p.add_argument("id", help="corpus id, DOI, arXiv id, OpenAlex id, S2 id or title")
        p.add_argument("--sources", default="openalex,s2")
        p.add_argument("--limit", type=int, default=200, help="max results per source (default 200)")
        p.add_argument("--new-only", action="store_true", help="print only papers new to the corpus")

    p = sp.add_parser("show", parents=[common], help="full record including abstract")
    p.add_argument("id")

    p = sp.add_parser("screen", parents=[common], help="record a screening decision")
    p.add_argument("id", help="corpus id(s), comma-separated, or DOI/arXiv id/title already in the corpus")
    p.add_argument("status", choices=STATUSES)
    p.add_argument("note", nargs="?", default=None)

    p = sp.add_parser("list", parents=[common], help="list corpus entries")
    p.add_argument("--status", help="comma-separated: " + ",".join(STATUSES))
    p.add_argument("--grep", help="case-insensitive regex over title+abstract")
    p.add_argument("--source", choices=SOURCES)
    p.add_argument("--prov", help="substring of a provenance step, e.g. 'refs:p0003' or 'search:'")
    p.add_argument("--year-from", type=int)
    p.add_argument("--sort", choices=("id", "year", "cites", "title"), default="id")
    p.add_argument("--limit", type=int, default=0, help="max rows (0 = all)")

    p = sp.add_parser("export", parents=[common], help="write a table of screened papers")
    p.add_argument("--status", default="include,maybe")
    p.add_argument("--format", choices=("md", "csv", "jsonl"), default="md")
    p.add_argument("--out", help="output file, '-' for stdout (default lit/export-<status>.<ext>)")
    p.add_argument("--grep")
    p.add_argument("--sort", choices=("id", "year", "cites", "title"), default="year")

    sp.add_parser("stats", parents=[common], help="counts by source, status and provenance")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    ctx = None
    try:
        ctx = Ctx(a)
        {"search": cmd_search, "refs": lambda c, x: cmd_graph(c, x, "refs"),
         "cites": lambda c, x: cmd_graph(c, x, "cites"), "show": cmd_show, "screen": cmd_screen,
         "list": cmd_list, "export": cmd_export, "stats": cmd_stats}[a.cmd](ctx, a)
        return 0
    except UserError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except SourceFailure as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except sqlite3.Error as e:
        print(f"error: database problem in {LIT_DIR}: {e}", file=sys.stderr)
        return 1
    finally:
        if ctx is not None:
            try:
                ctx.corpus.commit()
            except sqlite3.Error:
                pass
            s = ctx.http.summary()
            if s:
                log(s)


if __name__ == "__main__":
    sys.exit(main())
