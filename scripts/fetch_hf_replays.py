#!/usr/bin/env python3
"""Download Pokemon Showdown replay dumps from Hugging Face, resumably and verified.

Stdlib only, so it runs under the cluster's standalone CPython (the login node can
reach huggingface.co and its xet/CDN redirects; downloads are plain sequential I/O).

Every file is fetched from a pinned dataset revision (commit sha), resumed with HTTP
Range if a partial file exists, and checked against the sha256 that the Hub API
reports for LFS/xet files. A manifest.json next to the files records the revision,
URLs, sizes, hashes and fetch times (provenance for the corpus).

Downloads are untrusted data: this script never executes or imports anything from
them. Read them with `python -I` from a different directory.

Examples
  # list matching files and sizes without downloading
  python -I scripts/fetch_hf_replays.py --repo HolidayOugi/pokemon-showdown-replays \
      --include '[Gen 9] RANDOMBATTLE_part*.parquet' --dest /tmp/x --dry-run
  # download (resumable; rerun the same command after an interruption)
  python -I scripts/fetch_hf_replays.py --repo milkkarten/pokechamp --include 'data/*.parquet' \
      --dest /vol/bitbucket/fr1226/porygon-ws-data/raw/pokechamp
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

UA = "porygon-research (academic; contact via github femito1)"
API = "https://huggingface.co/api/datasets/{repo}"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def repo_info(repo: str, revision: str | None) -> dict:
    url = API.format(repo=repo)
    if revision:
        url += "/revision/" + urllib.parse.quote(revision, safe="")
    return get_json(url + "?blobs=true")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str, size: int | None, retries: int = 8) -> None:
    part = dest + ".part"
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size is not None and have == size:
            break
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            if size is None or os.path.getsize(part) == size:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = min(300, 5 * 2 ** attempt)
            print(f"  retry {attempt + 1}/{retries} after {e!r}; sleeping {wait}s", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f"failed to download {url}")
    os.replace(part, dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default=None, help="commit sha (default: current main, then pinned)")
    ap.add_argument("--include", action="append", required=True, help="fnmatch pattern(s) on repo paths")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip sha256 check")
    a = ap.parse_args()

    info = repo_info(a.repo, a.revision)
    rev = info["sha"]
    files = [s for s in info.get("siblings", [])
             if any(fnmatch.fnmatchcase(s["rfilename"], p) for p in a.include)]
    total = sum(s.get("size") or 0 for s in files)
    print(f"{a.repo}@{rev}: {len(files)} files, {total / 1e9:.2f} GB", flush=True)
    if a.dry_run:
        for s in files:
            print(f"  {s['rfilename']}  {(s.get('size') or 0) / 1e6:.1f} MB")
        return 0

    os.makedirs(a.dest, exist_ok=True)
    man_path = os.path.join(a.dest, "manifest.json")
    manifest = {"repo": a.repo, "revision": rev, "lastModified": info.get("lastModified"),
                "license": (info.get("cardData") or {}).get("license"), "files": {}}
    if os.path.exists(man_path):
        with open(man_path) as f:
            old = json.load(f)
        if old.get("revision") == rev:
            manifest["files"] = old.get("files", {})
    for s in files:
        path = s["rfilename"]
        size = s.get("size")
        want = (s.get("lfs") or {}).get("sha256")
        out = os.path.join(a.dest, path)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        rec = manifest["files"].get(path)
        if rec and os.path.exists(out) and os.path.getsize(out) == size and rec.get("verified"):
            continue
        url = RESOLVE.format(repo=a.repo, rev=rev, path=urllib.parse.quote(path))
        t0 = time.time()
        print(f"get {path} ({(size or 0) / 1e6:.1f} MB)", flush=True)
        if not (os.path.exists(out) and os.path.getsize(out) == size):
            download(url, out, size)
        got = None if a.no_verify else sha256_file(out)
        ok = a.no_verify or want is None or got == want
        if not ok:
            os.remove(out)
            print(f"  SHA256 MISMATCH for {path}: {got} != {want}; removed", flush=True)
        manifest["files"][path] = {"url": url, "size": size, "sha256": got or want,
                                   "verified": bool(ok and want and got), "fetched": now_iso()}
        with open(man_path + ".tmp", "w") as f:
            json.dump(manifest, f, indent=1)
        os.replace(man_path + ".tmp", man_path)
        print(f"  done in {time.time() - t0:.0f}s, verified={manifest['files'][path]['verified']}", flush=True)
    print("all done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
