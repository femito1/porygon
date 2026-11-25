# Council round 1 — independent proposals (you are Member {{LETTER}})

You are one of three independent AI researchers on a council: different labs, deliberately different perspectives. You
cannot see the others in this round. Your job: propose the changes most likely to make Porygon the strongest
random-battle bot (gen9 first, gen4 too), and to close the gap from ≈ 61% to ≈ 75–78% vs Foul Play. Be ambitious and
precise. **Cleverness beats scale:** a change that buys a big gain cheaply ranks above "train a bigger thing longer".
Radical redesigns are welcome. So are small sharp fixes.

**Read first (read-only; do not modify any file):** `docs/council/BRIEF.md`, then whatever sources it maps that you
need: `docs/thesis/results.md` and `methods.md` at least; the code in `src/porygon/` if your idea depends on it. Web
research is allowed and encouraged (papers, other bots, game-AI methods); cite URLs.

**Previous sessions:** read BRIEF's "Council history". Build on items that passed. Retire or redesign items that
failed, citing their verdict rows. Re-propose a withdrawn item only with an answer to the reason it was withdrawn.

**Rules:**
- Respect the BRIEF's hard rules.
- Do not re-propose something in its "tried and did not help" list unless you name the specific reason the earlier test
  does not cover your version, citing its row id.
- Every factual claim about our results cites a row id. Every outside claim cites a source.
- Separate measured facts from your inferences ("I infer…").

**Output: ONLY the following markdown, nothing else.**
- 3 to 7 proposals, ranked by your expected value per unit of cost.
- Use ids `{{LETTER}}1`, `{{LETTER}}2`, …
- Keep each proposal ≤ ~450 words; the whole answer ≤ ~3,500 words.

```
## {{LETTER}}<n>: <title>
- lever: <search | network/architecture | training signal | data | opponent/league | evaluation/stat | infra/throughput | other>
- generations: <gen9 | gen4 | both>
- mechanism: <what changes, concretely, and WHY it should raise strength vs Foul Play AND vs other strong opponents>
- evidence for: <our row ids + outside sources>
- evidence against / prior attempts: <row ids it might duplicate, and why yours differs>
- predicted effect: <points vs Foul Play at fixed 200k, with a range, and how you got it>
- cost: <GPU-hours on which device, CPU-hours, engineering days, risk to the running main run>
- kill test: <the CHEAPEST pre-registerable experiment that could show it does not work: arms, opponent, harness
  (in-process h2h or Showdown slot), n pairs, decision rule with a threshold, cost>
- scale-up if GO: <what follows a GO>
- failure modes / confounds: <what could make the kill test lie>
- formal check: <only if a core claim is mathematical (e.g. a convergence or error-rate guarantee): state it precisely
  enough to prove (e.g. in Lean); else "none">
```

End with one line: `MEMBER {{LETTER}} TOP PICK: <id> — <one sentence why>`.
