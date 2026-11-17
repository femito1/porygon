#!/usr/bin/env python3
"""Summarise Claude Code usage for the Porygon project from the local session transcripts (for docs/thesis/ai-use.md).

Reads only metadata (timestamps, model ids, token usage, tool names), never message text. Usage:
    python3 scripts/thesis/ai_usage.py [PROJECT_TRANSCRIPT_DIR ...] > docs/thesis/data/ai_usage.json
Default dir: ~/.claude/projects/-home-nand-Documents-personal-projects-ShowdownAgent
Assistant messages are streamed as several JSONL lines sharing one message id; usage is counted once per id.
"""
import collections, glob, json, os, sys

dirs = sys.argv[1:] or [os.path.expanduser("~/.claude/projects/-home-nand-Documents-personal-projects-ShowdownAgent")]
seen = set()
tot = collections.Counter()
by_model = collections.defaultdict(collections.Counter)
by_month = collections.defaultdict(collections.Counter)
tools = collections.Counter()
sessions, subagents, first, last = set(), set(), None, None
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
            if e.get("sessionId"):
                sessions.add(e["sessionId"])
            if is_sub and e.get("agentId"):
                subagents.add(e["agentId"])
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

out = {
    "source_dirs": dirs, "transcript_files": files, "main_session_files": main_files, "subagent_files": sub_files,
    "sessions": len(sessions), "subagents": len(subagents), "first": first, "last": last,
    "totals": dict(tot), "by_model": {k: dict(v) for k, v in by_model.items()},
    "by_month": {k: dict(v) for k, v in sorted(by_month.items())},
    "tool_calls": dict(tools.most_common()), "tool_calls_total": sum(tools.values()),
}
json.dump(out, sys.stdout, indent=1)
print()
