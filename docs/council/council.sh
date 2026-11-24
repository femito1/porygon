#!/bin/bash
# Porygon council runner (docs/council/README.md). Members: opus (Claude Opus 5.5, `claude -p`), astra (GPT-6.2 Astra,
# `codex exec`), carbon (Gemini 4 Carbon, `agy -p`). All run read-only inside a detached worktree of main; their final
# answers are captured by this script into docs/council/rounds/<ROUND>/<member>.md; tokens go to docs/council/BUDGET.tsv.
#   bash docs/council/council.sh setup            # worktree + anonymous letters + budget ledger (idempotent)
#   bash docs/council/council.sh test             # 1 tiny call per member: checks CLIs, parsing, read-only (≈ 60k tokens)
#   bash docs/council/council.sh round R1|R2|R3 [member...]   # all three in parallel by default; skips finished outputs
#   bash docs/council/council.sh budget           # token totals per round / member
#   bash docs/council/council.sh verify           # citation check of every round output against results.csv
# Env: COUNCIL_TOKEN_CAP (default 5000000; no new call starts once the ledger total reaches it), COUNCIL_TIMEOUT_S
# (default 5400 per call), FORCE=1 (redo an existing output).
set -uo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
D=$REPO/docs/council; WT=$REPO/.claude/worktrees/council; MAP=$D/rounds/MAP.txt; LEDGER=$D/BUDGET.tsv
CAP=${COUNCIL_TOKEN_CAP:-5000000}; TO=${COUNCIL_TIMEOUT_S:-5400}
MEMBERS="opus astra carbon"
die() { echo "council: $*" >&2; exit 1; }
letter() { awk -v m="$1" '$1==m {print $2}' "$MAP"; }
total() { awk -F'\t' 'NR>1 {s+=$6} END {print s+0}' "$LEDGER" 2>/dev/null; }

setup() {
  mkdir -p "$D/rounds"
  if [ ! -d "$WT" ]; then git -C "$REPO" worktree add --detach "$WT" HEAD > /dev/null || die "worktree add failed"; fi
  # one snapshot per council session: move the worktree to main's HEAD only before round 1 has started
  if [ ! -d "$D/rounds/R1" ] && [ "$(git -C "$WT" rev-parse HEAD)" != "$(git -C "$REPO" rev-parse HEAD)" ]; then
    git -C "$WT" checkout -q --detach "$(git -C "$REPO" rev-parse HEAD)" || die "worktree checkout failed"
  fi
  [ -f "$WT/docs/council/BRIEF.md" ] || die "the worktree's commit has no docs/council/BRIEF.md: commit the council package first"
  grep -qxF '.council/' "$REPO/.git/info/exclude" 2>/dev/null || echo '.council/' >> "$REPO/.git/info/exclude"
  mkdir -p "$WT/.council"
  if [ ! -s "$MAP" ]; then  # random anonymous letters, hidden from the members
    paste -d' ' <(printf '%s\n' $MEMBERS) <(printf '%s\n' A B C | shuf) > "$MAP"
  fi
  [ -s "$LEDGER" ] || printf 'utc\tround\tmember\tinput\toutput\ttotal\tseconds\tstatus\n' > "$LEDGER"
  echo "worktree $WT at $(git -C "$WT" rev-parse --short HEAD); letters hidden in $MAP; ledger $LEDGER (total $(total) / cap $CAP)"
}

# build_prompt ROUND MEMBER -> prompt file path (inside the worktree's ignored .council/)
build_prompt() {
  local r=$1 m=$2 L f; L=$(letter "$m"); f=$WT/.council/prompt-$r-$m.md
  python3 - "$D" "$r" "$m" "$L" "$MAP" "$f" <<'EOF' || return 1
import sys, re, pathlib
D, r, m, L, mapf, out = sys.argv[1:]
D = pathlib.Path(D)
letters = dict(l.split() for l in open(mapf) if l.strip())
tmpl = {'R1': 'R1-propose.md', 'R2': 'R2-critique.md', 'R3': 'R3-revise.md', 'TEST': None}[r]
if r == 'TEST':
    text = 'Reply with exactly one line: OK, then the first line of docs/council/BRIEF.md. Do not modify any file.'
else:
    text = (D / 'prompts' / tmpl).read_text()
own = lambda rr, mm: (D / 'rounds' / rr / f'{mm}.md').read_text()
others = [o for o in letters if o != m]
if r == 'R2':
    blocks = [f'## Proposals of Member {letters[o]}\n\n' + own('R1', o) for o in sorted(others, key=lambda o: letters[o])]
    text = text.replace('{{OTHERS}}', '\n\n'.join(blocks)).replace('{{OWN}}', own('R1', m))
if r == 'R3':
    crit = []
    for o in others:
        parts = re.split(r'(?m)^(?=### critique of )', own('R2', o))
        mine = [p for p in parts if re.match(rf'### critique of {L}\d', p)]
        crit.append(f'## From Member {letters[o]}\n\n' + ('\n'.join(mine) if mine else '(no critiques of your ids found; full text follows)\n' + own('R2', o)))
    text = text.replace('{{OWN}}', own('R1', m)).replace('{{CRITIQUES}}', '\n\n'.join(crit))
text = text.replace('{{LETTER}}', L)
pathlib.Path(out).write_text(text)
EOF
  echo "$f"
}

# call ROUND MEMBER: one headless call, read-only, output + usage captured
call() {
  local r=$1 m=$2 out=$D/rounds/$1/$2.md raw=$D/rounds/$1/$2.raw p rel t0 rc=0 inp=0 outp=0 tot=0 st=ok
  mkdir -p "$D/rounds/$r"
  if [ -s "$out" ] && [ "${FORCE:-0}" != 1 ]; then echo "skip $r/$m (exists)"; return 0; fi
  [ "$(total)" -lt "$CAP" ] || { echo "REFUSED $r/$m: ledger total $(total) >= cap $CAP"; return 1; }
  p=$(build_prompt "$r" "$m") || { echo "FAILED $r/$m: prompt build"; return 1; }
  rel=${p#"$WT/"}
  local ask="Read the file $rel (in the current working directory) and follow its instructions exactly. Your final message must be ONLY the output it asks for. Do not create, modify or delete any file."
  t0=$(date +%s)
  case $m in
    opus)
      ( cd "$WT" && timeout "$TO" claude -p "$ask" --model opus --output-format json \
          --allowedTools "Read Grep Glob WebFetch mcp__websearch__web_search mcp__websearch__web_fetch Bash(ls:*) Bash(cat:*) Bash(grep:*) Bash(head:*) Bash(tail:*) Bash(wc:*) Bash(sed -n:*)" \
          --disallowedTools "Edit Write NotebookEdit" ) > "$raw.json" 2> "$raw.err" || rc=$?
      python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('result',''))" "$raw.json" > "$out" 2>/dev/null
      read -r inp outp < <(python3 -c "import json,sys;u=json.load(open(sys.argv[1])).get('usage',{});print(u.get('input_tokens',0)+u.get('cache_creation_input_tokens',0)+u.get('cache_read_input_tokens',0), u.get('output_tokens',0))" "$raw.json" 2>/dev/null || echo "0 0") ;;
    astra)
      ( cd "$WT" && timeout "$TO" codex exec -s read-only --skip-git-repo-check -C "$WT" --json -o "$out" "$ask" ) > "$raw.jsonl" 2> "$raw.err" || rc=$?
      read -r inp outp < <(python3 -c "
import json,sys
i=o=0
for l in open(sys.argv[1]):
    try: e=json.loads(l)
    except Exception: continue
    if e.get('type')=='turn.completed':
        u=e.get('usage',{}); i+=u.get('input_tokens',0); o+=u.get('output_tokens',0)+u.get('reasoning_output_tokens',0)
print(i,o)" "$raw.jsonl" 2>/dev/null || echo "0 0") ;;
    carbon)
      ( cd "$WT" && timeout "$TO" agy -p "$ask" --model "Gemini 4 Carbon" --sandbox --effort high --output-format json \
          --print-timeout "${TO}s" ) > "$raw.json" 2> "$raw.err" || rc=$?
      python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('response',''))" "$raw.json" > "$out" 2>/dev/null
      read -r inp outp < <(python3 -c "import json,sys;u=json.load(open(sys.argv[1])).get('usage',{});print(u.get('input_tokens',0)+u.get('cache_read_tokens',0), u.get('output_tokens',0)+u.get('thinking_tokens',0))" "$raw.json" 2>/dev/null || echo "0 0") ;;
  esac
  tot=$((inp + outp))
  [ "$rc" = 0 ] && [ -s "$out" ] || st="FAILED(rc=$rc)"
  if [ -n "$(git -C "$WT" status --porcelain)" ]; then  # .council/ is excluded, so any line = a member edit
    st="$st,EDITED-reset"; git -C "$WT" status --porcelain > "$raw.edits"; git -C "$WT" checkout -- . > /dev/null 2>&1; git -C "$WT" clean -fdq > /dev/null 2>&1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$r" "$m" "$inp" "$outp" "$tot" "$(( $(date +%s) - t0 ))" "$st" >> "$LEDGER"
  echo "$r/$m: $st, $tot tokens, $(( $(date +%s) - t0 )) s -> $out"
  [ "$st" = ok ]
}

round() {
  local r=$1; shift; local ms=${*:-$MEMBERS} m pids=() rc=0
  [ -s "$MAP" ] || die "run setup first"
  case $r in R2) for m in $MEMBERS; do [ -s "$D/rounds/R1/$m.md" ] || die "R1/$m.md missing"; done ;;
             R3) for m in $MEMBERS; do [ -s "$D/rounds/R2/$m.md" ] || die "R2/$m.md missing"; done ;; esac
  for m in $ms; do call "$r" "$m" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "round $r done (rc $rc); ledger total $(total) / cap $CAP"; return $rc
}

case ${1:-} in
  setup) setup ;;
  test) setup > /dev/null; FORCE=1; for m in $MEMBERS; do call TEST "$m" & done; wait; head -n 3 "$D"/rounds/TEST/*.md ;;
  round) shift; round "$@" ;;
  budget) column -t -s $'\t' "$LEDGER"; echo "total $(total) / cap $CAP" ;;
  verify) python3 "$D/verify_citations.py" "$REPO/docs/thesis/data/results.csv" "$D"/rounds/R*/*.md ;;
  *) sed -n 2,13p "$0"; exit 2 ;;
esac
