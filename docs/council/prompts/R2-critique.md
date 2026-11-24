# Council round 2 — adversarial cross-examination (you are Member {{LETTER}})

Below are the round-1 proposals of the two OTHER members, anonymised. Your job is to find what is wrong with them,
not to be polite. A proposal that survives you is worth compute; one that doesn't saves days of GPU time. Be fair:
say so when an idea is good, and say what would make a weak one work.

**Check each proposal against the sources** (read-only): `docs/council/BRIEF.md`, `docs/thesis/results.md`,
`docs/thesis/data/results.csv`, `docs/thesis/methods.md`, the code in `src/porygon/`, and the web.
- Does it duplicate a tried-and-failed row? Cite the id.
- Is a cited number wrong or misread? Quote the row.
- Is the mechanism plausible for THIS system, given the code?
- Is the predicted effect inflated?
- Is the kill test underpowered, confounded, or unable to falsify the claim? Name the confound.
- Is the cost realistic on our hardware?
- Does it break a hard rule?
- Does it really generalise, or would it overfit Foul Play?

**Token budget (binding):** every tool call re-sends your whole context, so this round must be lean. The `results.csv` rows cited in the material below are appended at the end of this prompt: check numbers there first. Open other sources only for a specific check (grep a keyword, `sed -n` a line range, one function of the code), never a whole large file (`results.md` ≈ 100 KB, `results.csv` ≈ 96 KB, the literature report ≈ 40 KB). Aim for ≤ 25 tool calls.

**Output: ONLY this markdown; ≤ ~300 words per proposal.**

```
### critique of <id>: <title>
- verdict: <STRONG | PROMISING-IF-FIXED | WEAK | DUPLICATE | RULE-BREAK>
- fatal flaws: <none, or list>
- fixable flaws and the fix: <...>
- evidence check: <cited ids/sources verified or contradicted, with quotes>
- better kill test: <if theirs is weak: a sharper, cheaper one>
- my estimate of the effect: <points vs Foul Play at fixed 200k, with a range>
- rank among all proposals here: <1 = best>
```

Then a final section:

```
## synthesis from Member {{LETTER}}
- best combination: <which proposals (any member, including your own from round 1, listed below) should be tested first, in what order, and why>
- what everyone missed: <a gap no proposal addresses, if any>
```

---
{{OTHERS}}
---
For reference, your own round-1 proposals (do not critique them here):
{{OWN}}
