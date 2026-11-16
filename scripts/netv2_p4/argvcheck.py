#!/usr/bin/env python3
"""netv2 2026-11-16 (runs/p4/PREREG.txt): G9-P4's first-start argv vs the pre-registered references.
  python -I argvcheck.py P4_CMDLINE(NUL-separated /proc/PID/cmdline) G9Q2_FIRST_ARGV.txt G9Q_LIVE_ARGV.txt P4_PATH P3_PATH
(1) vs G9q2's first argv (same GPU, same job, same netv2.env branch): after mapping P4_PATH -> P3_PATH and /G9P4 ->
    /G9q2, blanking the --ports value and the job-level (first) --resume value, and dropping G9q2's "--seed 7272027",
    the token lists must be IDENTICAL.
(2) vs G9q's live argv (the A30, recorded at the PREREG): after the same P4 -> P3 mapping and dropping speed / infra
    tokens (--ports, --n-battles, --pool-battles, --pool-dir, --run-dir, --resume, --fastenv-workers, --snapshot-cache,
    --snapshot-cache-cpu with their values; --stacked-opp; --seed with its value), the recipe token lists must be IDENTICAL.
Also: the last --resume is P4_PATH; P3_PATH appears nowhere; no --seed (unseeded, as G9q). Prints a report; exit 0 = both pass, 1 = a mismatch."""
import sys

def load_nul(p):
    return [t for t in open(p, 'rb').read().decode().split('\0') if t]

def load_txt(p):
    return open(p).read().split()

def norm1(a, p4, p3, lin_from):
    out, i, first_resume = [], 0, True
    while i < len(a):
        t = a[i]
        if t == '--seed':
            i += 2; continue
        if t in ('--ports',) and i + 1 < len(a):
            out += [t, 'X']; i += 2; continue
        if t == '--resume' and first_resume and i + 1 < len(a):
            out += [t, 'X']; first_resume = False; i += 2; continue
        t = t.replace(p4, p3).replace('/' + lin_from + '/', '/G9q2/')
        if t.endswith('/' + lin_from):
            t = t[: -len(lin_from)] + 'G9q2'
        out.append(t); i += 1
    return out

DROP_V = {'--ports', '--n-battles', '--pool-battles', '--pool-dir', '--run-dir', '--resume', '--fastenv-workers',
          '--snapshot-cache', '--snapshot-cache-cpu', '--seed'}
DROP_F = {'--stacked-opp'}

def norm2(a, p4, p3):
    out, i = [], 0
    while i < len(a):
        t = a[i]
        if t in DROP_V:
            i += 2; continue
        if t in DROP_F:
            i += 1; continue
        out.append(t.replace(p4, p3)); i += 1
    return out

def diff(x, y):
    import difflib
    return [l for l in difflib.unified_diff(x, y, 'G9-P4', 'ref', lineterm='', n=1)]

def main():
    p4argv = load_nul(sys.argv[1]); q2 = load_txt(sys.argv[2]); q = load_txt(sys.argv[3]); p4, p3 = sys.argv[4], sys.argv[5]
    ok = True
    res = [i for i, t in enumerate(p4argv) if t == '--resume']
    last = p4argv[res[-1] + 1] if res else None
    print(f'last --resume: {last}'); ok &= last == p4
    bad = [t for t in p4argv if p3 in t]; print(f'tokens naming P3: {len(bad)}'); ok &= not bad
    seeds = p4argv.count('--seed'); print(f'--seed tokens: {seeds} (G9-P4 is unseeded, as G9q)'); ok &= seeds == 0
    a1, b1 = norm1(p4argv, p4, p3, 'G9P4'), norm1(q2, p4, p3, 'G9P4')
    d1 = diff(a1, b1); print(f'(1) vs G9q2 first argv: {"IDENTICAL" if not d1 else "DIFFERENT"} ({len(a1)} tokens)')
    for l in d1: print('   ', l)
    a2, b2 = norm2(p4argv, p4, p3), norm2(q, p4, p3)
    d2 = diff(a2, b2); print(f'(2) vs G9q live argv (recipe tokens): {"IDENTICAL" if not d2 else "DIFFERENT"} ({len(a2)} tokens)')
    for l in d2: print('   ', l)
    ok &= not d1 and not d2
    print('ARGV CHECK', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
