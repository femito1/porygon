# Codebase

Measured 2026-11-17 on `main` = fd6fd06 (all four stream branches merged 2026-11-17 00:33Z, pushed) with
`git ls-files` + `wc -l` (cloc/tokei/scc are not installed). Binary files (`.gz .npz .pt .zip .png .pdf .sqlite`) are
excluded; lines include comments and blank lines. All numbers [M] unless marked.

## 1. Size
**551 tracked files, 80,115 lines of text** [M]. By language:
| language | files | lines |
|---|---|---|
| Python | 256 | 52,961 |
| shell (`.sh` 105, `.sbatch` 59, `.env` 7, archived `netv2.env.v15–v19` 5) | 176 | 16,270 |
| JavaScript (fastenv's Node side, Showdown helpers) | 29 | 5,007 |
| patches (poke-engine ×3, Showdown duplicate battles, netv2 ×3) | 7 | 2,245 |
| Markdown | 19 | 2,197 |
| LaTeX (paper draft) | 1 | 205 |
| Rust (fastenv engine spike bindings) | 1 | 132 |
| other (txt, json, conf, in, toml, gitignore, test stubs) | 62 | ~1,100 [D: remainder] |

By area:
| area | path | files | lines | what |
|---|---|---|---|---|
| core package | `src/porygon/` (excluding `v2/`, `nnue/`) | 36 | 9,565 [D 16,510 − 6,392 − 553] | v1 encoder/model/PPO, search, inference, engine workers, netsearch, lookahead(2), deepen, v3leaf, fastenv Python side, replay, bc, play, evaluate |
| v2 network | `src/porygon/v2/` | 16 | 6,392 | spec, encoder, model, damage, PPO trainer, pretraining, evaluation, workers |
| NNUE student | `src/porygon/nnue/` | 4 | 553 | State conversion, features, student |
| compatibility alias | `src/gloria/` | 1 | 9 | old pickles (`gloria.model.*`) |
| fastenv JS | `fastenv/` | 18 | 2,981 | Node host, tracker (poke-env 0.16.1 port), encoders, bot ports, dump |
| scripts, all | `scripts/` | 400 | 48,311 | cluster jobs, analyses, tools |
| — wsp | `scripts/wsp/` | 145 | 16,784 | diagnostics trackers, sequential tests, in-process h2h, loss patterns, fpcal |
| — fastenv | `scripts/fastenv/` | 109 | 11,431 | slot harnesses, netsearch/lookahead2/deepen/NNUE chains, set refresh, parity |
| — netv2 | `scripts/netv2*` (top-level files + `netv2_*/` dirs) | 78 | 12,262 | chains, env mocks, watchdog stubs, comparisons |
| — ops (data stream) | `scripts/ops/` | 14 | 1,218 | freezes, shuffles, rating bands |
| tests | `tests/` | 35 (33 `test_*.py`) | 5,892 | see §2 |
| patches | `patches/` | 7 | 2,245 | see §3 |
| docs | `docs/` (paper 2,063 incl. lit notes; netsearch preregs/designs 1,682) | 28 | 3,745 | |
| README | `README.md` | 1 | 198 | |
Not in git: `.orchestra/` (orchestration state: PLAN, ORCHESTRATOR, CONTRACTS, ledger, 4 stream HANDOFFs, research
reports; 14 Markdown files, 1,979 lines on 2026-11-17) and `docs/thesis/` (this record, untracked at compilation).

## 2. Tests
- **33 test modules, 208 test functions** (`grep -c 'def test_'`; 20 `parametrize` decorators expand them).
- Last full run on the merged `main` (2026-11-17 ~00:45Z, as recorded in PLAN.md / memory): **199 passed, 38 skipped,
  0 failed** (237 collected [D 199 + 38]); the 11-16 merge gave 189 / 38 / 0. Command (PLAN.md): `CUDA_VISIBLE_DEVICES=
  OMP_NUM_THREADS=2 PYTHONPATH=src timeout 900 .venv/bin/python -u -m pytest -q -p no:cacheprovider -o
  faulthandler_timeout=120 -rfE tests --deselect …` (3 tests need a live Showdown server and are deselected; gen9 search
  tests need `PORYGON_GEN=9`).
- Not re-run for this record (laptop constraints).
- Other verification harnesses outside pytest: observation parity (`python -m porygon.v2.parity`, fixtures 944 / 663
  decisions), fastenv parity (`scripts/fastenv/parity_v1.py`, `parity_v2.py`), replay reconstruction
  (`scripts/score_reconstruction.py`), netv2.env mocks (`scripts/netv2_env_mock.sh`: v20 757/757 cases;
  `netv2_job_mock.sh` 184/184) [M] netv2 HANDOFF, NNUE State/feature parity (`nnue_parity.py`, feature parity 98.55%).

## 3. External components and licences
Porygon itself: **MIT**, © 2026 Fernando Rimoli [M] `LICENSE`, `pyproject.toml` (package `porygon` 0.2.0).
| component | version / commit | how it is used | licence | source of the licence |
|---|---|---|---|---|
| Pokémon Showdown (smogon) | git submodule `pokemon-showdown` @ 7b02eb3 (2026-10-18); ~100 per-commit builds for replay re-simulation; `ps-2026-11`, `ps-current` copies | battle simulator, benchmark server (+ `patches/showdown-duplicate-battles.patch`), fastenv | MIT | `pokemon-showdown/LICENSE` [M] |
| poke-engine (pmariglia) | 0.0.48 (+ 3 patches: gen4 durations, search options, tunable eval) built by `scripts/build_engine.sh` | MCTS simulator for the search, lookahead2, deepen | MIT | `.poke-engine/LICENSE` [M] |
| Foul Play (pmariglia) | cluster clone `/vol/bitbucket/fr1226/foul-play`, commit 8f541f8 | benchmark opponent only; **no code copied** (copyleft) | GPL-3.0 | memory `project-rename-porygon.md`, README Credits |
| poke-env (hsahovic) | 0.16.1 | player/battle framework; first-person state tracking (ported to JS in fastenv) | MIT | dist-info METADATA [M] |
| Stable-Baselines3 | 2.9.0 | v1 PPO | MIT | METADATA [M] |
| SuperSuit | 3.11.0 | v1 vector envs | MIT | METADATA [M] |
| PyTorch | ≥ 2.4 (laptop 2.14.1+cpu) | networks | Apache-2.0 / BSD-3-Clause / MIT (mixed) | METADATA [M] |
| TensorBoard | ≥ 2.17 (2.21.0) | logging | Apache-2.0 | METADATA [M] |
| CPython | 3.12.16 standalone on the cluster | runtime | PSF | memory |
| Node.js | cluster `gloria/.node` | Showdown, fastenv | MIT | [E] |
| Data: HolidayOugi/pokemon-showdown-replays (HF rev 15a0a45), PokéChamp (rev b5820ff) | metadata / logs | crawl queue, validation | not recorded here (unknown) | — |
| Method sources (no code): Jett Wang MIT MEng thesis 2024 (PPO recipe); MissingNo.Zero (recipe notes); Jaxcalibur (design notes) | — | cited | — | `.orchestra/research/REPORT.md` |

## 4. History and branches
- **389 commits on `main`** (390 across all refs) from 2025-03-02 to 2026-11-17: 4 in 2025-03, 42 in 2026-10,
  343 in 2026-11 (317 since 2026-11-08, the start of the network-first campaign) [M] `git log`.
- 14 merge commits (stream merges on 2026-11-09 (ws/netv2-fastenv), 11-12, 11-15, 11-16, 11-17).
- Authors: 386 as "nand", 3 as "Fernando Rimoli" (same person, same address) [M] `git shortlog`.
- **Branches:** `main` + 4 stream branches `ws/netv2` (171 commits reachable; 1 ahead of main), `ws/fastenv` (182),
  `ws/data` (134), `ws/wsp` (101), each in its own worktree `.claude/worktrees/ws-<stream>`; remote `origin/main`
  (github femito1/porygon). Results are mirrored to a private repo (femito1/porygon-runs) [M] memory.
- Repo names: GlorIA until 2026-10-21 (commit 7020fa8 renamed it Porygon); local folder still `ShowdownAgent`.
