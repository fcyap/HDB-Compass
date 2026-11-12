# Experiment 1: `q38m`

Rows: 270 with the auditor on, 90 with it off.

## Task success by variant and category (Wilson 95% interval)

| Variant | All | A | B | C | D | E | F | Tool F1 | Param EM | CSR | P50 s | P95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 36/90 (30%–50%) | 13/15 (62%–96%) | 5/15 (15%–58%) | 11/15 (48%–89%) | 2/15 (4%–38%) | 0/15 (0%–20%) | 5/15 (15%–58%) | 0.75 | 0.51 | 0.58 | 112.81 | 445.01 |
| B | 42/90 (37%–57%) | 10/15 (42%–85%) | 9/15 (36%–80%) | 11/15 (48%–89%) | 10/15 (42%–85%) | 0/15 (0%–20%) | 2/15 (4%–38%) | 0.81 | 0.76 | 0.76 | 215.10 | 627.41 |
| C | 71/90 (69%–86%) | 13/15 (62%–96%) | 10/15 (42%–85%) | 9/15 (36%–80%) | 11/15 (48%–89%) | 13/15 (62%–96%) | 15/15 (80%–100%) | 0.94 | 0.76 | 1.00 | 116.84 | 249.34 |

## Category F (adversarial and safety) by variant

| Variant | Handled | Stopped by the input guardrail |
|---|---|---|
| A | 5/15 | 0 |
| B | 2/15 | 0 |
| C | 15/15 | 15 |

## What the runtime auditor did (Variant C, auditor on)

- answers: 90; rejected at least once: 3 (3.3%); mean rewrites per answer: 0.056
- loops that ran out of retries (the model's sentences were then removed): 1
- rejected although the final answer scored correct: 1
- median latency: 157.07 s when rejected, 114.18 s when not

## Variant C with and without the auditor (same questions)

- paired questions: 90; correct with auditor: 71, without: 71
- right only with the auditor: 1; right only without it: 1 (exact McNemar p = 1.000)
- median extra latency with the auditor: 12.210000000000008 s; extra tokens: 535
