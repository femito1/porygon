# Council round 3 — rebuttal and revision (you are Member {{LETTER}})

Below are your round-1 proposals and every critique the two other members wrote of them. For each of your proposals,
decide honestly: **KEEP** (the critiques are wrong; rebut with evidence), **REVISE** (fix the flaws), or **WITHDRAW**
(they are right). Defending a dead idea wastes compute. You may also adopt and improve ONE idea from another member if
you now think it beats yours. Credit its id.

Sources as before (read-only): `docs/council/BRIEF.md`, `docs/thesis/*`, `src/porygon/`, the web.

**Token budget (binding):** every tool call re-sends your whole context, so this round must be lean. The `results.csv` rows cited in the material below are appended at the end of this prompt: check numbers there first. Open other sources only for a specific check (grep a keyword, `sed -n` a line range, one function of the code), never a whole large file (`results.md` ≈ 100 KB, `results.csv` ≈ 96 KB, the literature report ≈ 40 KB). Aim for ≤ 25 tool calls.

**Output: ONLY this markdown.**

```
## {{LETTER}}<n>: <title>  —  KEEP | REVISE | WITHDRAW
- response to critiques: <point by point, with evidence>
- revised proposal: <full template fields as in round 1 (lever, generations, mechanism, evidence, predicted effect,
  cost, kill test with n and decision rule, scale-up, failure modes, formal check). Omit if WITHDRAW>
```

Optionally:

```
## adopted: <other member's id> (improved)
<full template fields>
```

End with: `MEMBER {{LETTER}} FINAL TOP PICK: <id> — <one sentence>`.

---
Your round-1 proposals:
{{OWN}}
---
Critiques of your proposals:
{{CRITIQUES}}
---
For the optional adoption only (do not critique them again), the other members' round-1 proposals:
{{OTHERS}}
