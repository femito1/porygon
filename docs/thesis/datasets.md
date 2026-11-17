# Datasets

Compiled 2026-11-17. Provenance marks as in `README.md`: `[M]` read from the named file, `[D]` derived (arithmetic
shown), `[E]` estimate. `$D` = `/vol/bitbucket/fr1226/porygon-ws-data`. Pipeline details: `.orchestra/data/PIPELINE.md`;
formats: `.orchestra/CONTRACTS.md` ("Data formats", "Observation spec v2").

## 1. Human replay corpus (gen9 and gen4 random battles)

### 1.1 Sources and policy
| source | what | used for | provenance |
|---|---|---|---|
| Showdown replay server `replay.pokemonshowdown.com` | public uploaded replays; `/<id>.json` has `log` + `inputlog`; ≈2.5k gen9 randbats uploads/day, ≈30 gen4/day | the only source of input logs, hence of every training decision | [M] PIPELINE.md §1 |
| HolidayOugi/pokemon-showdown-replays (Hugging Face rev `15a0a45`, 2026-10-05) | spectator logs, real names, no input log; gen9 randbats 2.01M (2022-12 →), gen4 80k | replay ids and ratings for the crawl queue; stored logs for validation | [M] PIPELINE.md §1 |
| PokéChamp (`milkkarten/pokechamp`, rev `b5820ff`) | lowercased logs, Elo buckets only, no input log | deduplicated into the corpus; never trained on | [M] |
| Metamon, PokéAgent | OU formats only | not used | [M] |

- **Crawl policy (user decisions 2026-11-08):** gen9 rated ≥ 1500 only (nothing unrated or below 1500 is fetched); never
  any Jaxcalibur game; ≤ 2 requests/s, one in flight, exponential backoff 30 s → 30 min honouring Retry-After, every
  response cached once; contact User-Agent; one IP, no evasion. [M] PIPELINE.md §2.
- **Exclusions:** never fetched or kept: `jaxcalibur`, our own `porygonagent` / `femito`, `foulplay|playingfoul|jaxcalibur|
  bot\d*$`. Suspected bots (manual list plus any account with > 150 uploads in one UTC day or activity in ≥ 22 hours of a
  day) are kept but flagged; the own sides of **44 accounts** (`$D/raw/corpus/gen9randombattle/bc-exclude.json`) are
  dropped from BC (opponents kept). The corpus build report lists 53 suspect names for gen9
  (`$D/raw/corpus/build-report.json`). [M]
- **Replay rating semantics:** the replay's `rating` = the lower of the two players' post-battle Elo in 96% of rated
  replays ([M] corpus-stats: `=min(post)` 1,351,242 of 1,413,869 = 95.6% [D]); per-player pre-battle Elo is stored as
  `p1_rating` / `p2_rating`; dataset row filters use the deciding side's own rating.
- **Jaxcalibur check on P4:** grep over all 1,241 record files: 0 hits (`$D/logs/p4/jaxcheck.log`). [M] data HANDOFF.

### 1.2 Corpus (built 2026-11-08 19:21 UTC; `$D/raw/corpus/corpus-stats.txt`, `build-report.json`) [M]
| format | source | games | date range | with replay rating | rating ≥ 1500 / 1700 / 1800 / 1900 / 2000 | with input log |
|---|---|---|---|---|---|---|
| gen9randombattle | holidayougi | 1,959,826 | 2022-12 … 2026-10 | 1,409,498 | 551,177 / 334,942 / 264,709 / 198,089 / 123,806 | 5,767 |
| gen9randombattle | showdown-replays (our crawl, by 11-08) | 4,372 | 2026-10 … 2026-11 | 4,372 | 4,372 / 4,372 / 4,372 / 4,372 / 2,983 | 4,372 |
| gen9randombattle | pokechamp | 35,481 | 2024-06 … 2025-03 | 0 | — | 0 |
| gen4randombattle | holidayougi | 64,875 | 2017-06 … 2026-10 | 37,974 | 0 ≥ 1500 by replay rating | 0 |
| gen4randombattle | showdown-replays | 2,304 | 2019-05 … 2026-11 | 2,304 | 2,304 / 13 / 0 / 0 / 0 | 2,304 |
| gen4randombattle | pokechamp | 3,018 | 2024-03 … 2025-03 | 0 | — | 0 |

- Gen9 kept after deduplication and drops: showdown-replays 4,372 + holidayougi 1,959,826 + pokechamp 35,481 [M]
  build-report; dropped e.g. `holidayougi:excluded_player` 24,958, `lt_2_turns` 24,770, side servers. The memory figure
  "2.0M games indexed (≥1500: 556k, ≥1900: 202k)" is the same corpus [M] memory `gen9-port.md`.
- Monthly volume (gen9, games with `min ≥ 1500`): ~13–17k per month through 2026-04, then 68.8k (2026-05), 86.3k
  (06), 62.6k (07), 24.7k (08), 37.6k (09) [M] corpus-stats; "from 2026-05 the ≥1900 volume jumps 4–7× (a few
  high-volume accounts)" [M] PIPELINE.md. Few public games exist before 2023-12 (replay-database outage).
- Gen4: ratings never exceed 1771 (≥ 1500: ~2.3k games ever), so high-rating gen4 BC is impossible [M] PIPELINE.md.
- **Input-log crawl:** queue `$D/raw/inputlog-queue-hf-v2-ge1500.txt.gz` = 555,685 replay ids rated ≥ 1500, bots removed,
  ordered by band (≥1900, 1800s, 1700s, 1500s) then newest; exhausted 2026-11-14 18:12; the crawler now walks new uploads
  only [M] data HANDOFF. Not queued: 169,920 gen9 games with null replay rating but both pre-battle Elo ≥ 1500 (an open
  user decision) [M].
- **Exact first-person reconstruction:** each input log is re-simulated at its own Showdown commit (~100 builds under
  `$D/showdown/builds/`); a game is kept only if the regenerated log equals the stored log line for line. Fidelity: crawled
  gen9 replays at their own commit 14,933 / 14,934 exact; on the newest older build 98.5% (commit gap 1–5), 100% (6–20),
  93.4% (21–100), ~0% (> 100); refresh rounds 0.12–0.34% "not exact" (skipped), 0 illegal actions [M] PIPELINE.md §6.

### 1.3 The live human set
`$D/datasets/v2pre/gen9-human/` (symlink to the newest versioned dir; grows with a 4-hourly refresh inside the data host job).
| when | games | decisions | own-rating bands (decisions) | source |
|---|---|---|---|---|
| 2026-11-15 16:28 | 554,556 | 32.35M | — | [M] data HANDOFF |
| 2026-11-16 (P4 freeze) | 553,439 | 32,350,338 | ≥1900 12,789,519 · 1700–1899 8,095,727 · 1500–1699 11,272,695 · <1500/unknown 192,397 | [M] `p4-…/summary.json` `live_at_freeze` |
- Live-set rating 5/50/95% (replay rating) 1538 / 1807 / 2247 [M]. Mean decisions per game: P3 55.12, U7 55.94 [M]
  `u7…/summary.json`.

## 2. Frozen datasets (`$D/datasets/frozen/`; hard links, never refreshed, read-only)
Held-out rule everywhere: netv2's `is_val(game, 0.06)` = crc32(int64 game id, 8 bytes LE) % 10000 < 600, so a game held
out once stays held out in every later set [M] data HANDOFF; P3's summary checks it against P1 and P2 (`ok: true`).

| set | path | frozen | games | decisions (train / val) | rating filter | rating 5/50/95% (own) | shards | checks | md5 of summary.json |
|---|---|---|---|---|---|---|---|---|---|
| **P1** | `frozen/p1-gen9-human/{v1,v2pre}` | 2026-11-08 22:43 | 18,554 | 923,322 total | all ≥ 1900 (first bulk pass) | — | v2pre shard-NNNNN (unshuffled) | held-out 1,192 games (P3 check) | no summary.json [M] HANDOFF |
| **P2** | `frozen/p2-gen9-20261109-0645/` | 2026-11-09 06:45 | 45,267 | 2,292,617 (2,156,941 / 135,676; val games 2,711) | ≥ 1900 replays | 1925 / 2070 / 2307 | `v2pre-human-shuf/` 90 train + 6 val (96), seed 20261109 | verify ok 13:00Z, 0 hash mismatches | 1908b4fc |
| **P3** (G9q's init) | `frozen/p3-gen9-20261109-1735/` | 2026-11-09 17:35 | 127,888 | 7,046,891 (6,620,882 / 426,009; val games 7,762) | ≥ 1900 replays; 98.88% of rows own rating ≥ 1900 | 1926 / 2066 / 2322 (own mean 2087.3) | 276 train + 18 val (294), seed 20261109 | verify ok 18:59Z; P1/P2 held-out preserved | de1c215d |
| **U7** (size-vs-quality control) | `frozen/u7-gen9-20261111-1755/v2pre-human-shuf/` | 2026-11-11 17:55 | 126,122 (118,360 train) | 7,046,912 (6,620,903 train; val = P3's 426,009, same inodes) | uniform whole-game draw from ≥ 1500 inside P3's battle-date window; 53.67% of rows own ≥ 1900 | 1552 / 1932 / 2272 (train 1550 / 1918 / 2265; mean 1895.9) | 276 train, val = P3's 18 | verify ok 19:45Z | d07ff664 (MD5SUMS 31fdee57) |
| **P4** | `frozen/p4-gen9-20261116-1502/` | 2026-11-16 17:56:57 (READY) | 234,980 (220,859 train / 14,121 val) | 12,788,895 (12,021,692 / 767,203) = 1.82× P3's training rows [D 12,021,692 / 6,620,882] | own rating ≥ 1900 (100%); ≥ 2100 38.07%; minus set H | 1913 / 2050 / 2329 | 501 train + 32 val (533), ~24k rows each, seed 20261116 | verify ok 17:56Z, selection equalities true, READY written last | fb6a336a (MD5SUMS 6f576fb0) |
| **Set H** (NNUE gate 1 human test set) | P4 `setH-records/`, `select/excluded-games.txt` | 2026-11-16 | 1,968 | 118,133 (only 624 decisions ≥ 1900) | every game of records part-0070[0-3] | — | 4 record files (md5 3b4d26e0, 14edfd2d, b72bb971, aa3841f9) | excluded from P4 entirely | in P4 summary.json |

All values [M] from each set's `summary.json`, `v2pre-human-shuf/meta.json` and `verify.json` (read 2026-11-17),
except P1 (data HANDOFF) and the ratio [D].
- **Battle dates:** P3 training rows 0% 2025-09-23, 5% 2025-12-09, 50% 2026-07-05, 100% 2026-11-08; U7 0% 2025-10-04,
  50% 2026-06-11, 100% 2026-11-08 [M] U7 summary — the U7 confound (median 06-11 vs 07-05) is recorded in its READING.
- **Disk:** `du` 261 MB (P1), 1.7 GB (P2), 5.3 GB (P3), 3.9 GB (U7), 11 GB (P4) [M 2026-11-17]; hard links make these
  overlap, so the sum overstates the footprint.
- **Encoding:** v2 supervised shards (`np.savez`: `obs/<field>` per observation spec v2.1, `aux/<field>`, `target` f32
  [N, A] = one-hot of the human action, `z` = final result for the decider, `game` int64, `rating`/`opp_rating`); a v2
  decision is ~12 KB in RAM, ~140 B compressed [M] `src/porygon/v2/pretrain.py` docstring.
- **What each set was used for:** P1 → BC init of G9p/G9s-anchor; P2, P3 → data-size BC arms (results bc-05…07); P3 → G9q's
  init, KL anchor and frozen opponent; U7 → quality control (bc-10); P4 → BC-P4 and G9-P4 (bc-15, g9q-25/26).

## 3. Search-label and oracle datasets (engine search as teacher)
| set | path | what | size | outcome |
|---|---|---|---|---|
| gen9 search labels | `$D/datasets/v2pre/gen9-searchlabels/` | our gen9 search vs itself on the duplicate server, 4 × 500 ms; `target` = visit shares, `target_argmax`, `target_t05`, `q`, `root_value`, `visits` | 1,997 games [M] HANDOFF; frozen copies P2 116,810 decisions / 1,587 games, P3 166,995 / 2,258 [M] summaries | search labels hurt BC (results bc-12…14) |
| gen4 search labels | `v2pre/gen4-searchlabels/` | same, gen4 | 1,461 games [M] | — |
| gen4 oracle v1 | `$D/datasets/gen4-oracle-v1/` | 1,200 old gen4 benchmark games, v1 observations only (`--log-oracle` runs, e.g. gloria `search-*-309116/309120`), pre-Rest-fix | 1,200 games [M] | oracle labels cannot judge the net (dg-04) |
Labelling is paused (no `$D/host/labels.args`). Data logged before 2026-11-07 has a Rest-duration bug (fixed in
`porygon.search` 006ff76): don't use it for Rest / Sleep Talk [M] wsp HANDOFF, PIPELINE.md §7.

## 4. Engine self-play data (leaf-evaluation fits, NNUE)
| set | path | size | generator | use |
|---|---|---|---|---|
| gen4 fit data | gloria `runs/evalfit/r1` (fits) | 39,823 games, 1,680,721 positions [M] paper.tex §Fit | both players search the full state with the stock eval, 20,000 iterations/move, 10% decisions drawn from visit shares, each team pair played twice with sides swapped | `eval_r1.json` (gen4 default) |
| gen9 fit data | gloria(-gen9) `runs/selfplay-data-20261106-165311-380885` (`games-*.jsonl.gz`), fit `runs/evalfit/gen9-r1` | 39,970 games [M] paper.tex (memory: "40k games") | same recipe | `eval_gen9_r1.json` (log loss stock 0.536 → 0.518, not the gen9 default) |
| NNUE gate 1 | `$F/nnue/gate1/chain/data/` (417 MB sharded npz) | 10,000,602 positions from 139,280 games (71.8 per game), 2.05 h at 1,359 decisions/s [M] `$F/nnue/gate1/VERDICT.txt` | G9q-cd@805M self-play at T 1.0 on fastenv; teacher P(win) + outcome; 1% of State strings | 155k-param student (failed gate 1) |
| loss-pattern logs | `/vol/bitbucket/fr1226/porygon-ws-wsp/losses/{m1,m2}/games/*.jsonl.gz` (40 MB) | M1 1,500 pairs (cd@805 vs itself), M2 1,000 pairs (cd@805 vs BC-P3) [M] losses REPORT.md | spd_h2h `--game-log` (omniscient log + true state per choice) | dg-01…03 |
| v2diag positions | `/vol/bitbucket/fr1226/porygon-ws-wsp/v2diag/pos9/` (8,000 frozen gen9 rows), `pos/` (gen4) | 8,000 rows [M] wsp HANDOFF | from data shards | tracker diagnostics |
| parity fixtures | `$C/fixtures/gen4` (30 battles, 944 decisions), `…/gen9` (30 battles, 663 decisions) | [M] CONTRACTS.md | recorded Showdown protocol + reference obs | encoder parity |
`$C` = `/vol/bitbucket/fr1226/porygon-ws-netv2`, `$F` = `/vol/bitbucket/fr1226/porygon-ws-fastenv`.

## 5. RL self-play volumes (environment steps = decisions of the learner)
Training data in PPO is generated on line and not stored; the volumes are the step counters (details per run in `runs.md`).
| lineage | steps | provenance |
|---|---|---|
| v1 PPO (gen4, SB3) | 600M main run (`runs/20261026-082356-301120`), + 40M arms (`arm-{ctrl,x30,x300}`) | [M] memory `imperial-slurm-training.md` |
| G9q (gen9, P3 init) | 805,029,873 steps / 18,720 updates / **15,549,773 games** at the cd@805 fork [M] checkpoint `counters` of `$C/runs/g9q-cd800/final/00805029873.pt`; 1,000,029,674 at the step-cap stop (2026-11-17 03:36Z); 1,002.1M at 12:02Z [M] ORCHESTRATOR.md | ⇒ ≈ 51.8 decisions per learner game [D 805,029,873 / 15,549,773] |
| G9p (gen9, P1 init) | ≥ 338.6M (G9p@338,554,007 evaluated 11-11; parked) | [M] g9q-vs-g9p READING |
| G9s (gen9, scratch) | ≥ 75M (vs9-75M evaluated); tracker last row 67.5M | [M] vs-search 843442, wsp HANDOFF |
| R1b-F (gen4 v2) | ≈ 398M (11-16); hold trip at 390,214,040 | [M] HANDOFF, hold1 VERDICT |
| G9q2, G9r, G9t | ~50M each (step-matched readings at 25M/50M) | [M] READINGs |
| G9qent | fork + 50M ramp; cooled at 880,064,251 | [M] g9qent VERDICT |
| G9-P4 | ≥ 50.04M by 11-17 (50M reading) | [M] p4 OVERALL |

## 6. Random-battle set versions
- **Showdown submodule** `pokemon-showdown` = smogon `7b02eb39878f…` (committed 2026-10-18) [M] `git ls-files -s`; this is
  the "October sets" pinned for the benchmark server (`$F/.showdown-eval` = 7b02eb3 + the duplicate-battle patch, code sha256
  95a0414e26cfd185; `$F/ps-sets-pinned`) [M] fastenv HANDOFF.
- **Encoder set library:** in-tree `src/porygon/data/gen9randombattle_library.json.gz` (October, md5 324d8e97) used by every
  consumer whatever `PORYGON_PS` is; built from Showdown's own generator (`scripts/build_set_library.js 2000000 4`, 2M teams)
  [M] CONTRACTS.md, README Setup. Gen4 library likewise.
- **November sets** `$F/ps-2026-11` = the collector's Showdown for runs started after the switch (netv2 trainers'
  `PORYGON_PS`; R1b-F resumed on November gen4 sets at 20.88M, 2026-11-09, wsp commit 2a976a1) [M]; `$F/ps-current` = smogon
  master. Rule (user): switch to a month's sets only at a new run, never under a running lineage; the benchmark server
  switches only with a re-baseline (≥ 1000 games per arm). vocab.json is append-only. [M] PLAN.md, fastenv HANDOFF.
- The commits of `ps-2026-11` / `ps-current` were not read for this record (unknown; `git -C $F/ps-2026-11 log -1`).

## 7. Gaps
- P1 has no summary.json; its rating percentiles and shard count are not recorded here.
- Search-label decision counts outside the P2/P3 copies were not re-counted.
- The gloria self-play fit sets' sizes are quoted from the paper, not re-counted from the files.
