# AI assistance record (for the academic-integrity declaration)

Imperial requires students to use generative AI only where their programme allows it and to acknowledge how they used
it. Agree the permitted use with the supervisor before the project period (May 2027) and declare it in the thesis.
This file records the facts needed for that declaration. Refresh the numbers with
`python3 scripts/thesis/ai_usage.py > docs/thesis/data/ai_usage.json`.

## How the work was done
- **Before AI assistance [M, git]:** 4 commits on 2025-03-02/03 (`25462f2` "first version of project" … training-loop
  prototype; +22,514 / −72 lines). No commits between 2025-03-03 and 2026-10-19.
- **With Claude Code (2026-10-19 onward) [M, git, 11-17]:** 385 commits, +88,110 / −26,403 lines. The system as it now
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
- **Commit attribution:** by the student's choice, commits carry only the student's name (no co-author trailers), so
  git history alone does not show AI authorship; this file and the transcripts are the record.

## Usage volume [M, local transcripts, `data/ai_usage.json`, snapshot 2026-11-17 12:33Z]
| quantity | value |
|---|---|
| transcript window | 2026-10-19 10:59Z → 2026-11-17 12:33Z |
| main sessions | 10 |
| subagents (stream / helper agents) | 118 (121 subagent transcript files) |
| model requests | 15,773 (Claude Opus 5.5 ≈ 99.8%; 14 requests Opus 4.8) |
| tool calls | 15,704 (Bash 12,900; SendMessage 554; Write 435; Monitor 360; Edit 303; Read 274; web fetch 205; …) |
| output tokens | 18.2 M |
| input tokens (uncached) | 1.38 M |
| cache-write input tokens | 269 M |
| cache-read input tokens | 4.16 B |
| by month (requests / output tokens) | 2026-10: 1,864 / 1.36 M; 2026-11 (to the 17th): 13,909 / 16.8 M |

Caveats:
- **Coverage gap risk:** Claude Code deletes transcripts older than its `cleanupPeriodDays` setting (30 days by
  default). The earliest kept transcript is from 2026-10-19, the first day of AI-assisted commits, so the window
  looks complete. Older transcripts will disappear unless the setting is raised, so refresh the numbers before then.
- Token counts are from the API usage fields, de-duplicated per message id. They exclude other projects, e.g. the
  separate gpu-exploration project that set up cluster access.
- The model id prefix `databricks-` is the serving route, not a different model.

## What the student should be able to show (suggested, for the viva / presentation)
- Every design decision in the thesis explained in their own words (why duplicate pairs, why pre-registration, why
  human-BC init, why the margin gate, why best-first deepening instead of MCTS).
- Re-run one headline result end to end from the repo (e.g. a raw-net vs Foul Play benchmark) and one analysis
  script, on their own.
- Their own writing in the thesis text, with AI use declared as agreed with the supervisor.
