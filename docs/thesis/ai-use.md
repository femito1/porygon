# AI assistance record (for the academic-integrity declaration)

Imperial requires students to use generative AI only where their programme allows it and to acknowledge how they used
it. Agree the permitted use with the supervisor before the project period (May 2027) and declare it in the thesis.
This file records the facts needed for that declaration. Refresh the numbers incrementally (old transcripts are deleted after 30 days, so a fresh full scan undercounts):
`python3 scripts/thesis/ai_usage.py --base docs/thesis/data/ai_usage.json > /tmp/u.json && mv /tmp/u.json docs/thesis/data/ai_usage.json`.

## How the work was done
- **Before AI assistance [M, git]:** 4 commits on 2025-03-02/03 (`25462f2` "first version of project" … training-loop
  prototype; +22,514 / −72 lines). No commits between 2025-03-03 and 2026-10-19.
- **With Claude Code (2026-10-19 onward) [M, git, 11-17]:** 385 commits, +88,110 / −26,403 lines; by 11-25, 458 commits, +113,217 / −27,062 lines (`git rev-list --count --since=2026-10-19 HEAD`; `git log --since=2026-10-19 --numstat`). The system as it now
  stands (Porygon: v1/v2 networks, PPO trainer, fastenv, search + fitted evaluation, lookahead2, deepen, human-data
  pipeline, benchmarks, sequential-test tooling) was written in this period by Claude Code agents directed by the
  student.
- **Working pattern (from 2026-11-08):** one *orchestrator* session (Claude) plans experiments, pre-registers decision
  rules, reviews and signs off pre-registrations, fact-checks results and reports to the student; short-lived *stream
  agents* (netv2, fastenv, data, wsp) implement code, write pre-registrations, and arm cluster-side job chains. The
  student set the goals and constraints and made the standing decisions (examples, recorded in `.orchestra/PLAN.md`
  and memory): the target (beat Jaxcalibur, gen9 primary), no distillation of Jaxcalibur, human data rated ≥ 1500 only,
  sequential tests for internal go/no-go only with fixed-n headline numbers, no opponent-specific exploitation of Foul
  Play, monthly-set policy, the delegation of campaign choices to the orchestrator (2026-11-15).
- **Literature work:** the literature review (`docs/paper/lit/`) and the campaign research report
  (`.orchestra/research/REPORT.md`) were compiled by agents with web search; every citation must be re-checked by the
  student before use (agents can misattribute).
- **Research council (from 2026-11-24) [M, `docs/council/sessions/`]:** at the student's request, three models from
  different labs (Claude Opus 5.5, GPT-6.2 Astra, Gemini 4 Carbon) ran read-only and anonymised (A/B/C) through blind
  proposals, cross-critique and rebuttal, on a facts-only brief with cited result ids. The orchestrator then wrote a
  ranked portfolio of pre-registerable kill tests (`PORTFOLIO.md`), and the student signs off before anything runs.
  Session 1 (2026-11-24): 16 proposals → 7 ranked items + follow-ups. Tokens: 14.26M gross / 2.21M fresh (cap 30M gross,
  set by the student; `BUDGET.tsv`). The council's top finding was a documented encoder gap: own Tera types are
  invisible to the net before Terastallizing; the orchestrator re-checked it in the code. Its proposals are
  hypotheses, not results.
- **Commit attribution:** by the student's choice, commits carry only the student's name (no co-author trailers), so
  git history alone does not show AI authorship; this file and the transcripts are the record.

## Usage volume [M, local transcripts, `data/ai_usage.json`; full scan to 2026-11-17 12:33Z + incremental scan to 2026-11-25 12:04Z]
| quantity | value |
|---|---|
| transcript window | 2026-10-19 10:59Z → 2026-11-25 12:04Z |
| main sessions | 13 (10 to 11-17, +3 started after it) |
| subagents (stream / helper agents) | 135 (118 to 11-17, +17) |
| model requests | 18,162 (Claude Opus 5.5 ≈ 99.8%; 14 requests Opus 4.8; 24 `<synthetic>` client-side records) |
| tool calls | 18,111 (Bash 14,993; SendMessage 565; Write 528; Monitor 397; Edit 369; TaskStop 322; Read 309; web fetch 207; …) |
| output tokens | 21.0 M |
| input tokens (uncached) | 1.55 M |
| cache-write input tokens | 307 M |
| cache-read input tokens | 4.79 B |
| by month (requests / output tokens) | 2026-10: 1,864 / 1.36 M; 2026-11 (to the 25th): 16,298 / 19.6 M |
| not included | the research council's members (headless runs in their own worktree; logged in `docs/council/**/BUDGET.tsv`) |

Caveats:
- **Coverage:** Claude Code deletes transcripts older than its `cleanupPeriodDays` setting (30 days by default). On
  11-25 the oldest transcript left on disk was from 10-25, so 10-19 → 10-25 survives only in the 11-17 full scan, which
  the incremental refresh keeps as its base (`increments` in the JSON lists each window). Refresh at least every
  30 days, always with `--base`. A session that spans a snapshot boundary is counted once, in the earlier window.
- Token counts are from the API usage fields, de-duplicated per message id. They exclude other projects, e.g. the
  separate gpu-exploration project that set up cluster access.
- The model id prefix `databricks-` is the serving route, not a different model.

## What the student should be able to show (suggested, for the viva / presentation)
- Every design decision in the thesis explained in their own words (why duplicate pairs, why pre-registration, why
  human-BC init, why the margin gate, why best-first deepening instead of MCTS).
- Re-run one headline result end to end from the repo (e.g. a raw-net vs Foul Play benchmark) and one analysis
  script, on their own.
- Their own writing in the thesis text, with AI use declared as agreed with the supervisor.
