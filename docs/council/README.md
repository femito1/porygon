# Porygon research council — runbook (START HERE)

For the fresh agent who runs the council. Prepared 2026-11-24 by the previous orchestrator session, with the user.

## The mandate (user, 2026-11-24)
> "I think we need to be very very smart here about how to truly improve. Bigger is better sounds appealing but usually
> the best breakthroughs come from cleverness. … spawn 1 Opus 5.5 agent, 1 Codex Astra 6.2 agent, and 1 Gemini Carbon
> agent and have them create a council of discussion that argue about how to definitely improve our performance and
> have us be the best. The changes proposed can be as large or small as possible. All changes should be adversarially
> checked (maybe the use of Lean here could help...?). The goal is to create a self-improvement loop for this project in
> the cleverest way possible."

User decisions:
- both generations are in scope (gen9 primary);
- token budget **5M to start**; the user can raise it;
- outputs in the format best for agents (the user will not read the debate, only your short summary);
- Lean only after checking disk space (see below);
- the user starts you fresh; nothing ran before you except a CLI smoke test (`test-run/`).

## You are also the orchestrator
Before the council, do the normal start-up in `.orchestra/ORCHESTRATOR.md`:
- check-in brief, ledger, crons;
- live campaign duties: G9q main run, A40, R1b-F, deadlines listed there.

The council is your main project; the campaign keeps running alongside it.

## Members (all verified working 2026-11-24, `test-run/`)
| member | model | CLI (headless, read-only) | notes |
|---|---|---|---|
| `opus` | Claude Opus 5.5 | `claude -p --model opus`, Edit/Write disallowed, read tools + websearch MCP allowed | ≈ 115k tokens of fixed overhead per call |
| `astra` | GPT-6.2 Astra (`databricks-gpt-6-2-astra`) | `codex exec -s read-only` | model comes from `~/.codex/config.toml` |
| `carbon` | Gemini 4 Carbon | `agy -p --model "Gemini 4 Carbon" --sandbox --effort high` | the id `gemini-4-carbon` is retired (403): use the label |

- Members run inside a **detached worktree** `.claude/worktrees/council`, pinned at one commit for the whole session.
- They never touch the cluster or the main checkout; the runner captures their final answers. A member's edits in the
  worktree are detected, logged (`*.raw.edits`) and reverted.
- Members are anonymous to each other (letters A/B/C, random, in `rounds/MAP.txt`). Don't reveal the mapping in
  anything you pass them.

## Files
- `BRIEF.md`: the shared, facts-only evidence map with row ids. **Update it before R1** if new results landed since
  2026-11-24 (check `docs/thesis/results.md`), then commit, then `setup` (it re-pins the worktree only before R1).
- `prompts/R1-propose.md`, `R2-critique.md`, `R3-revise.md`: round prompts with a strict output template (one block per
  proposal: lever, mechanism, evidence ids, predicted effect, cost, **kill test with n and decision rule**, failure
  modes, formal check).
- `council.sh`: the runner, with `setup | test | round R1|R2|R3 [member…] | budget | verify`.
  - The three members run in parallel; finished outputs are skipped (resume-safe; `FORCE=1` redoes one).
  - Token accounting goes to `BUDGET.tsv`; no new call starts once the total reaches `COUNCIL_TOKEN_CAP` (5,000,000).
  - The per-call timeout is `COUNCIL_TIMEOUT_S` (5400).
- `verify_citations.py`: flags cited row ids that do not exist, and numbers next to an id that do not match the row.
- `evidence/`: copies of the git-excluded literature report, campaign rules and the loss-pattern report.
- `rounds/`, `BUDGET.tsv`: per-session working files (git-ignored while running). Archive them at the end (step 7).

## The session
1. **Preflight.**
   - Run `bash docs/council/council.sh setup` and `budget` (expect 0).
   - Update BRIEF.md if needed (then commit + setup again).
   - Run it from a background shell: a round can take 30–90 min.
2. **R1, blind proposals:** `bash docs/council/council.sh round R1`.
   - Read the three outputs: is the format right? Is every proposal complete? Are kill tests present?
   - A failed or malformed member: rerun that member alone (`FORCE=1 … round R1 astra`), once.
   - Run `verify`.
   - Note the tokens per call and project the session total. If R1 used > 1.7M, tell the user the projection before R2.
3. **R2, cross-examination:** `round R2`. Each member critiques the other two's R1 (anonymised), plus a synthesis section.
4. **R3, rebuttal and revision:** `round R3`. Each member answers the critiques of its own ids: KEEP / REVISE / WITHDRAW.
5. **Synthesis (you).** Write `rounds/PORTFOLIO.md` in the format below. Rules:
   - Merge duplicates across members. Drop WITHDRAWN and RULE-BREAK items. Any DUPLICATE of a tried row needs a stated
     reason.
   - Score each surviving item: expected gain vs Foul Play (median of the members' estimates), the probability the kill
     test passes (your judgment from the critiques), and the kill-test cost on our hardware (BRIEF "Resources").
   - Rank by expected gain × probability ÷ cost, and keep the members' disagreements visible.
   - **Adversarial check:** re-run `verify`. For every top-5 item, check its cited ids and its kill-test design yourself
     against the harness rules (one change at a time; duplicate pairs; the fixed 200k Foul Play yardstick for Foul Play
     numbers; sequential designs only for internal go/no-go). A design that cannot falsify the claim is redesigned or
     dropped.
   - Optional **R4 vote**, if the top picks are contested: one short call per member with the draft portfolio, asking
     each to rank it and name the single biggest remaining risk, ≤ ~100k tokens each. Do it by hand (`claude -p …` /
     `codex exec …` / `agy -p …` with the same flags as `council.sh`), and log the tokens in BUDGET.tsv.
6. **Report to the user, ≤ 10 lines:**
   - the top 3 kill tests: what each tests, cost, device, how long;
   - what you recommend starting now;
   - tokens used / cap;
   - anything that needs the user (VM, Showdown account, Lean install).

   Wait for sign-off.
7. **Archive and launch.**
   - Copy `rounds/` (with MAP.txt), `BUDGET.tsv` and `PORTFOLIO.md` to `docs/council/sessions/<YYYY-MM-DD>/` and commit.
   - Hand each approved kill test to the owning stream agent: netv2 for training/network, fastenv for search, harness and
     slot runs, wsp for diagnostics and stats, data for datasets. Use the `.orchestra` brief pattern: pre-registration
     with the portfolio's decision rule, a cluster-side chain, then report and stop.
   - Add a row per test to `docs/council/LEDGER.md`: proposal id → PREREG md5 → job → verdict.

## PORTFOLIO.md format (agent-readable; one block per item, ranked)
```
## P<rank>: <title>   [origin: <member letters + ids>]   status: <KILL-TEST-READY | NEEDS-DESIGN | PARKED>
- lever / generations:
- claim (one sentence, falsifiable):
- mechanism:
- evidence (row ids, sources):
- strongest objection (from R2) and the answer (R3):
- expected gain vs Foul Play @200k (members' estimates → median):
- kill test: arms | harness | n pairs | decision rule | cost (device, hours) | owner stream
- if GO: next step and its cost
- formal check: <statement, or none>
```

## The self-improvement loop (after this session)
- **Trigger:** reconvene when a council kill test returns a verdict, or once a week while tests run, whichever is first.
  Never more often than the evidence changes.
- **Next sessions:**
  - Start from the new results: update BRIEF.md (new rows, a "council history" section linking `sessions/*/PORTFOLIO.md`
    and `LEDGER.md`).
  - R1's prompt also asks members to update or retire the previous portfolio's items, given the verdicts.
- **Each GO** scales up through the normal pre-registered pipeline. **Each NO-GO** goes into BRIEF's "tried and did not
  help" list with its row id, so the council never re-proposes it blindly.
- **What the loop must never do:** run untested ideas at scale; change the main run (G9q) without a passed kill test;
  break the hard rules in BRIEF.md; exceed the token cap without telling the user.

## Lean (only if earned)
- A proposal whose "formal check" field states a precise mathematical claim, and that ranks in the top 5, may get a Lean
  proof attempt. Typical claims: a search method's convergence or exploitability bound, or a sequential test's error rate.
  Empirical strength claims are never "checked" by Lean; games settle them.
- **Disk preflight first:** elan + a Lean toolchain + Mathlib's prebuilt cache need roughly 10 GB [estimate; Mathlib's
  cache alone ≈ 5 GB].
  - The laptop had 5.2 GB free on 2026-11-24: not there.
  - Candidates: the GPU Dojo VM (48 GB free on /, but its free GPU-hours tick while it runs, and the user starts it), or
    the cluster login-node home (check `quota -s` / `df -h ~` first; nothing heavy on the login node).
  - Ask the user before installing anything.

## Known facts about the tooling
- The worktree must be at a commit that contains `docs/council/BRIEF.md` (`setup` checks this).
- `.council/` (prompt files) is in `.git/info/exclude`, so it never shows as an edit.
- Opus calls cost ≈ 115k tokens before any work; Astra ≈ 42k; Carbon ≈ 35k (TEST round).
- `agy` print mode returns `status`, `response` and `usage.total_tokens` in its JSON; `codex exec --json` reports usage on
  `turn.completed`.
