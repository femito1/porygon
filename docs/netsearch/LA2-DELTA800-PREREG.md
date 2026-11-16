# lookahead2 margin δ* for G9q-cd@800M: pre-registration (fastenv, 2026-11-16; binding once written, read-only)
Main's decision (2026-11-16, Bet 1 GO follow-ups, item A.1): the follow-ups LA2-FP and LA2-H run lookahead2 on
G9q-cd@800M, our best net (it beat G9q-cd@605M 52.46% [50.67, 54.22] in netv2's R800). δ* is re-derived for this net
by **Bet 1's pre-registered δ rule, unchanged** (`LA2-BET1-PREREG.md`, md5 97c9b4e9…, section "δ*"). This file
only names the new net and the output files; it changes nothing in the rule. Written before any run on this net.
Cluster copy `$C/runs/la2-delta800/PREREG.txt` (r--, byte-identical), md5 in `PREREG.md5`.

## Net
`$C/nets/g9q-cd-805.03M.pt` = netv2 `runs/g9q-cd800/final/00805029873.pt` (copied 11-16, md5 d0c5b6cf…, sha256
`e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f`, equal for both files). Trained from netv2's
`src-24c633f` (as G9q-cd@605M), sparse ±1 rewards (trainer argv has no `--shaping`): P(win) = sigmoid(win_logit),
no V + Φ. It loads through the frozen tree `$C/frozen/netv2-fba02fb-20261110`; the chain's first stage checks this
(a failing load stops the chain with FAILED; nothing else is tried without a new decision).

## Rule (identical to Bet 1's)
- **Data.** The study positions `$C/netsearch/study-20261109` (S1 501, S2 250, S2c 251, S3 401 = 1,403), the same
  record files and rows as Bet 1's bench (`jobq/done/la2-bench2.sh`), the same seed (11).
- **Runs per position.** A with K 8 worlds, a fresh retest B with K 8, a fresh judge R with K 32
  (`lookahead2_study.py bench --worlds 8 --retest --ref 32 --workers 10 --concurrency 3`), the new net for the root,
  the leaves and R.
- **δ\*** = the smallest grid margin ≥ 0 from which on (every larger finite grid margin too) at least 90% of the
  overrides, weighted by the net's T 0.5 draw probability, are confirmed by R (R's Q of A's pick > R's Q of the net's
  draw). Grid as in Bet 1 (0 … 0.15). Pooled over the 1,403 positions (`analyze-bench` on `bench-all.jsonl`, its
  `margin_rule.delta`); per-subset values are informational.
- **No margin meets the rule** ⇒ the chain writes FAILED, no follow-up starts, and main decides.

## Outputs (one cluster-side chain on fastenv's helper job, `scripts/fastenv/lookahead2_delta.sh`)
- Bench files `$C/lookahead2/study-cd805-20261116/bench-{s1,s2,s2c,s3,all,s2all}.{jsonl,json,txt}` + sha256 list.
- Stage dir `$C/runs/la2-delta800/`: STATUS, FAILED, `DELTA` (one line, the number δ\*), `RESULT`, `VERDICT.txt`
  (r--; line 1 `DELTA* = x (override r, R precision p)`, then the pooled table).
- LA2-FP and LA2-H read δ\* from `DELTA` (their PREREGs say so); they start only after it exists.

## Reported (informational)
Override rate and R precision at δ\*, the pooled curve, per-subset δ, leaves and seconds per decision, and the same
figures for cd@605 (Bet 1's bench) next to them.
