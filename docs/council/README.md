# Porygon research council — runbook (START HERE)

The orchestrator runs a council session with this file. Sessions so far: `sessions/2026-11-24/` (session 1). Test
ledger: `LEDGER.md`.

## The mandate (user, 2026-11-24)
> "I think we need to be very very smart here about how to truly improve. Bigger is better sounds appealing but usually
> the best breakthroughs come from cleverness. … spawn 1 Opus 5.5 agent, 1 Codex Astra 6.2 agent, and 1 Gemini Carbon
> agent and have them create a council of discussion that argue about how to definitely improve our performance and
> have us be the best. The changes proposed can be as large or small as possible. All changes should be adversarially
> checked (maybe the use of Lean here could help...?). The goal is to create a self-improvement loop for this project in
> the cleverest way possible."

User decisions:
- both generations are in scope (gen9 primary);
- token cap **30M GROSS per session** (user 11-24). Gross = every token each model request processes, cached context
  re-read on every tool call included (≈ 90% of a long call). `BUDGET.tsv` logs `total` (gross) and `fresh`. Session 1
  used 14.26M gross / 2.21M fresh. Raising the cap needs the user;
- outputs in the format best for agents: the user reads only the ≤ 10-line report and signs off before anything runs;
- Lean only after checking disk space, and with the user's OK to install (see below).

The orchestrator does its normal start-up first (`.orchestra/ORCHESTRATOR.md`). The campaign keeps running alongside.

## Members
| member | model | CLI (headless, read-only; exact flags in `council.sh`) |
|---|---|---|
| `opus` | Claude Opus 5.5 | `claude -p --model opus --tools Read,Grep,Glob,Bash,WebFetch --permission-mode dontAsk`, Bash allowlist ls / cat / grep / head / tail / wc / sed -n, websearch MCP |
| `astra` | GPT-6.2 Astra (`databricks-gpt-6-2-astra`, from `~/.codex/config.toml`) | `codex exec -s read-only` |
| `carbon` | Gemini 4 Carbon (use the label; the id `gemini-4-carbon` returns 403) | `agy -p --model "Gemini 4 Carbon" --sandbox --effort high --settings '{"toolPermission":"always-proceed"}'` |

- Members run inside the **detached worktree** `.claude/worktrees/council`, pinned at one commit for the whole session.
- They never touch the cluster or the main checkout; the runner captures their final answers. A member's edits in the
  worktree are detected, logged (`*.raw.edits`) and reverted.
- Members are anonymous to each other (letters A/B/C drawn per session into `rounds/MAP.txt`). Never reveal the mapping in
  anything passed to them.
- Why these flags:
  - the user's Claude default mode is `bypassPermissions`, under which `--allowedTools` restricts nothing, so opus needs
    `dontAsk` and `--tools`;
  - agy's default review mode denies every shell command headless and then returns an EMPTY answer; always-proceed
    inside `--sandbox` runs commands, confines reads to the worktree (+ /tmp) and allows the web.

## Files
- `BRIEF.md`: the shared, facts-only evidence map with row ids, a "tried and did not help" list and the council history.
- `prompts/R1-propose.md`, `R2-critique.md`, `R3-revise.md`: round prompts with a strict output template. One block per
  proposal: lever, mechanism, evidence ids, predicted effect, cost, **kill test with n and decision rule**, failure modes,
  formal check.
  - R2 / R3 prompts get an appendix with the `results.csv` rows cited in their material, plus a binding tool-call budget.
  - R3 also shows the other members' R1 proposals, for the optional adoption.
- `council.sh`: the runner, `setup | test | round R1|R2|R3 [member…] | budget | verify`.
  - Members run in parallel; finished outputs are skipped (resume-safe; `FORCE=1` redoes one).
  - No new call starts once the gross total reaches `COUNCIL_TOKEN_CAP` (default 30,000,000; `COUNCIL_CAP_BASIS=fresh`
    switches the basis).
  - Per-call timeout `COUNCIL_TIMEOUT_S` (7200).
  - An answer under 500 bytes, or an agy run with status ≠ SUCCESS / denied actions, is FAILED.
- `verify_citations.py`: flags cited row ids that do not exist, and numbers next to an id that do not match the row. It is
  noisy (a 60-character window): read the flags by hand.
- `evidence/`: copies of the git-excluded literature report, campaign rules and the loss-pattern report.
- `rounds/`, `BUDGET.tsv`: per-session working files (git-ignored). Archive them at the end (step 7), then delete them so
  the next `setup` re-pins the worktree and draws new letters.

## The session
1. **Preflight.**
   - Update BRIEF.md with every result since the last session (`docs/thesis/results.md`): new standing rows, NO-GOs into
     the tried list, the council history (each previous portfolio item → verdict). Commit.
   - Run `bash docs/council/council.sh setup` (re-pins the worktree to HEAD) and `budget` (expect 0).
   - Run every round from a background shell under `systemd-inhibit --what=idle:sleep --mode=block` (the laptop sleeps).
   - **Never edit `council.sh` while a round runs:** bash reads scripts incrementally, so write a copy and `mv` it over.
2. **R1, blind proposals:** `round R1`.
   - Read the three outputs: is the format right, is every proposal complete, are kill tests present?
   - Rerun a failed or malformed member alone, once (`FORCE=1 … round R1 carbon`).
   - Run `verify`.
   - Check the raw logs for reads outside the worktree (e.g. grep `rounds/`, `MAP.txt`, `../../../`).
   - Session 1 cost: R1 8.0M gross (1.4–4.8M per member; Astra re-read 100 KB files on every call), R2 4.4M, R3 1.8M;
     each round took 2–30 min. Project the session from R1 and tell the user if it will exceed the cap.
3. **R2, cross-examination:** `round R2`. Each member critiques the other two members' R1 (anonymised), plus a synthesis
   section ("best combination", "what everyone missed").
   - Check the critique headers parse (`### critique of <id>`), so R3 can route them.
4. **R3, rebuttal and revision:** `round R3`. Each member answers the critiques of its own ids (KEEP / REVISE /
   WITHDRAW) and may adopt one other idea.
5. **Synthesis (the orchestrator).** Write `rounds/PORTFOLIO.md` in the format below.
   - Merge duplicates across members. Drop WITHDRAWN and RULE-BREAK items. Any DUPLICATE of a tried row needs a stated
     reason.
   - Score each surviving item: expected gain vs Foul Play (median of the members' latest estimates), the probability the
     kill test passes (your judgment from the critiques), and the kill-test cost on our hardware.
   - Rank by gain × probability ÷ cost, and keep the disagreements visible.
   - **Adversarial check:**
     - re-run `verify`;
     - re-check every top-5 item's code and number claims yourself (session 1: the top claim was confirmed in
       `fastenv/js/encoder_v2.js:28-35`, and another was refuted by `lookahead2.py:13`);
     - check each kill-test design against the harness rules: one change at a time; duplicate pairs; the fixed-200k Foul
       Play yardstick; sequential tests only for internal go/no-go; per-side encoder flags for observation changes;
       power (4,000 in-process pairs ≈ ±1.1 h2h; 1,000 Showdown pairs ≈ ±3 points).
     - A design that cannot falsify its claim is redesigned or dropped.
   - Optional **R4 vote** if the top picks are contested: one short call per member with the draft portfolio (rank it,
     name the biggest risk), by hand with the same flags, tokens logged in BUDGET.tsv.
6. **Report to the user, ≤ 10 lines:** the top 3 kill tests (what each tests, cost, device, how long), what you recommend
   starting now, tokens used / cap, anything that needs the user (VM, Showdown account, Lean install). Wait for sign-off.
7. **Archive and launch.**
   - Copy `rounds/` (with MAP.txt; gzip raw logs > 20 KB), `BUDGET.tsv` and `PORTFOLIO.md` to
     `sessions/<YYYY-MM-DD>/`, commit, then delete `rounds/` and `BUDGET.tsv`.
   - Hand each approved kill test to the owning stream: netv2 for training / network / spec, fastenv for search and the
     ExIt data, wsp for Foul Play runs, diagnostics and stats, data for datasets.
   - Each brief: the portfolio's design and rule pasted to be copied VERBATIM into the PREREG, a cluster-side chain to
     `<stream dir>/council/<test>/VERDICT.txt` (+ STATUS, FAILED), then report and stop.
   - Add a row per test to `LEDGER.md`: proposal → PREREG md5 → job → verdict → row ids. Add per-outcome actions to
     ORCHESTRATOR.md.

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

## The self-improvement loop
- **Trigger:** reconvene when the open council tests have verdicts, or a week after the last session, whichever is first.
  Never more often than the evidence changes.
- **Each session starts from the verdicts.**
  - BRIEF's council history lists every previous item → verdict.
  - R1 builds on GOs and retires NO-GOs.
  - Withdrawn items come back only with an answer to their stated reason.
- **Each GO** scales up through the normal pre-registered pipeline. **Each NO-GO** goes into BRIEF's tried list with its
  row id.
- **Never:**
  - run untested ideas at scale;
  - change the main run (G9q) without a passed kill test;
  - break the hard rules in BRIEF.md;
  - exceed the token cap without the user.

## Lean (only if earned)
- A top-5 proposal whose "formal check" states a precise mathematical claim (a convergence or exploitability bound, a
  sequential test's error rate) may get a Lean proof attempt. Empirical strength claims are settled by games, never by Lean;
  identities a unit test settles do not qualify (session 1 had none that qualified).
- Disk first: elan + a toolchain + Mathlib's cache need ≈ 10 GB [estimate]. The laptop has ≈ 4 GB free, so it can't host
  it. Candidates are the GPU Dojo VM (48 GB free, but its GPU-hours tick while it runs) or the cluster home (check
  `quota -s`). Ask the user before installing anything.

## Tooling facts
- The worktree must be at a commit that contains `docs/council/BRIEF.md` (`setup` checks this). `.council/` (built
  prompts) is in `.git/info/exclude`.
- Token accounting:
  - claude `-p` JSON `usage` is cumulative over the call's requests;
  - `codex exec --json` reports usage on `turn.completed`, and `output_tokens` already includes reasoning;
  - agy's `input_tokens` excludes `cache_read_tokens`, and `output_tokens` excludes `thinking_tokens`.
