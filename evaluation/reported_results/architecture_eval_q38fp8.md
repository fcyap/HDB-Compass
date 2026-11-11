# Experiment 1: `q38fp8`

Rows: 270 with the auditor on, 90 with it off.

## Task success by variant and category (Wilson 95% interval)

| Variant | All | A | B | C | D | E | F | Tool F1 | Param EM | CSR | P50 s | P95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 35/90 (29%–49%) | 13/15 (62%–96%) | 7/15 (25%–70%) | 9/15 (36%–80%) | 1/15 (1%–30%) | 0/15 (0%–20%) | 5/15 (15%–58%) | 0.75 | 0.51 | 0.58 | 65.47 | 211.64 |
| B | 46/90 (41%–61%) | 11/15 (48%–89%) | 9/15 (36%–80%) | 11/15 (48%–89%) | 12/15 (55%–93%) | 1/15 (1%–30%) | 2/15 (4%–38%) | 0.81 | 0.76 | 0.72 | 116.95 | 210.24 |
| C | 70/90 (68%–85%) | 13/15 (62%–96%) | 10/15 (42%–85%) | 8/15 (30%–75%) | 11/15 (48%–89%) | 13/15 (62%–96%) | 15/15 (80%–100%) | 0.94 | 0.76 | 1.00 | 75.93 | 175.64 |

## Category F (adversarial and safety) by variant

| Variant | Handled | Stopped by the input guardrail |
|---|---|---|
| A | 5/15 | 0 |
| B | 2/15 | 0 |
| C | 15/15 | 15 |

## What the runtime auditor did (Variant C, auditor on)

- answers: 90; rejected at least once: 5 (5.6%); mean rewrites per answer: 0.122
- loops that ran out of retries (the model's sentences were then removed): 3
- rejected although the final answer scored correct: 3
- median latency: 88.7 s when rejected, 74.67 s when not

## Variant C with and without the auditor (same questions)

- paired questions: 90; correct with auditor: 70, without: 70
- right only with the auditor: 0; right only without it: 0 (exact McNemar p = 1.000)
- median extra latency with the auditor: 0.8900000000000006 s; extra tokens: 535
