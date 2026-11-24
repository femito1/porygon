# Council ledger: proposal id -> PREREG md5 -> job -> verdict

| proposal | session | PREREG md5 | job / chain | verdict | row id |
|---|---|---|---|---|---|
| P1 own pre-Tera visibility (B1; C1 trimmed) | 2026-11-24 | 088a5650 (ws/netv2 55de18d; netv2.env v23 5fc31c6c; parent ckpt_02210021139 md5 6db059de; Stage 0 PASS 300 battles, 0 mismatches) | A40 job 1175374: arm E from 15:42Z, then arm C (30M + 5M cd each); verdict chain = data-host step 1545813.5 → porygon-ws-netv2/council/p1-tera/VERDICT.txt (ETA ≈ 11-25 06–08Z). Chain reads the rule as LB > 50.5 GO, (50.0, 50.5] REPLICATE, ≤ 50.0 PARK (same next steps as the portfolio: replicate before promotion) | pending | — |
| P2 ExIt soft completed-Q target, offline on ns-27 data (C6 revised) | 2026-11-24 | 52be887d (ws/fastenv 2613084; β 35.9003) | job 1554469 (training, dove01, from 13:53Z) → porygon-ws-fastenv/council/p2-exit-soft/VERDICT.txt | **NO-GO** 14:02Z: E′ vs C 50.20 [49.25, 51.16], E′ vs S 50.09 [49.11, 51.07] | ns-33, ns-34 |
| P3 gen4 re-baseline at fixed 200k (A5 revised) | 2026-11-24 | a8e0d1ff (ws/wsp 4a535e5; seed 4031124) | slot 1554479 (dove02, from 14:08Z; arm (i) ≈ 5.3 h, arm (ii) after netv2's R1BF_CD_READY) → porygon-ws-wsp/council/p3-gen4/VERDICT.txt (ETA ≈ 21:30Z if READY by ≈ 19:30Z) | pending | — |
