#!/usr/bin/env python3
"""Summarise Claude Code usage for the Porygon project from the local session transcripts (for docs/thesis/ai-use.md).

Reads only metadata (timestamps, model ids, token usage, tool names), never message text. Usage:
    python3 scripts/thesis/ai_usage.py [--base OLD.json] [PROJECT_TRANSCRIPT_DIR ...] > NEW.json
Default dir: ~/.claude/projects/-home-nand-Documents-personal-projects-ShowdownAgent
Assistant messages are streamed as several JSONL lines sharing one message id; usage is counted once per id.

Claude Code deletes transcripts older than 30 days (its default cleanupPeriodDays), so a fresh scan loses the oldest
days. Always refresh incrementally: `--base docs/thesis/data/ai_usage.json` keeps the previous snapshot and adds only
requests / tool calls after its `last` timestamp, and sessions / subagents whose first event is after it (a session
spanning the boundary stays counted once, in the base). The output records every base window in `increments`.
"""
import argparse, collections, glob, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--base", help="previous snapshot JSON; count only events after its 'last'")
ap.add_argument("dirs", nargs="*")
a = ap.parse_args()
dirs = a.dirs or [os.path.expanduser("~/.claude/projects/-home-nand-Documents-personal-projects-ShowdownAgent")]
base = json.load(open(a.base)) if a.base else None
cut = base["last"] if base else None
seen = set()
tot = collections.Counter()
by_model = collections.defaultdict(collections.Counter)
by_month = collections.defaultdict(collections.Counter)
tools = collections.Counter()
sess_first, sub_first, first, last = {}, {}, None, None
files = main_files = sub_files = 0
for d in dirs:
    for f in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
        files += 1
        is_sub = "subagents" in f.split(os.sep)
        sub_files += is_sub; main_files += not is_sub
        for line in open(f, errors="replace"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            ts = e.get("timestamp")
            if ts:
                first = ts if first is None or ts < first else first
                last = ts if last is None or ts > last else last
            for ids, key, ok in ((sess_first, e.get("sessionId"), True), (sub_first, e.get("agentId"), is_sub)):
                if ok and key:
                    ids[key] = min(ids.get(key) or ts or "", ts or "") if key in ids else (ts or "")
            if cut and (not ts or ts <= cut):
                continue
            m = e.get("message")
            if not isinstance(m, dict) or e.get("type") != "assistant":
                continue
            for c in m.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    key = (m.get("id"), c.get("id"))
                    if key not in seen:
                        seen.add(key); tools[c.get("name", "?")] += 1
            u = m.get("usage")
            mid = m.get("id")
            if not isinstance(u, dict) or not mid or ("u", mid) in seen:
                continue
            seen.add(("u", mid))
            row = {k: int(u.get(k) or 0) for k in
                   ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
            row["requests"] = 1
            tot.update(row); by_model[m.get("model", "?")].update(row)
            if ts:
                by_month[ts[:7]].update(row)

new_sess = sum(1 for t in sess_first.values() if not cut or t > cut)
new_sub = sum(1 for t in sub_first.values() if not cut or t > cut)
if base:
    tot.update(base["totals"])
    for k, v in base["by_model"].items():
        by_model[k].update(v)
    for k, v in base["by_month"].items():
        by_month[k].update(v)
    tools.update(base["tool_calls"])
out = {
    "source_dirs": dirs, "transcript_files": files, "main_session_files": main_files, "subagent_files": sub_files,
    "sessions": new_sess + (base["sessions"] if base else 0),
    "subagents": new_sub + (base["subagents"] if base else 0),
    "first": base["first"] if base else first, "last": last if not base or (last and last > base["last"]) else base["last"],
    "increments": (base.get("increments") or [{"first": base["first"], "last": base["last"], "scan": "full"}]) if base else [],
    "totals": dict(tot), "by_model": {k: dict(v) for k, v in by_model.items()},
    "by_month": {k: dict(v) for k, v in sorted(by_month.items())},
    "tool_calls": dict(tools.most_common()), "tool_calls_total": sum(tools.values()),
}
if base:
    out["increments"] = out["increments"] + [{"after": cut, "last": out["last"], "scan": "incremental",
                                              "transcript_first_on_disk": first, "new_sessions": new_sess,
                                              "new_subagents": new_sub}]
json.dump(out, sys.stdout, indent=1)
print()
