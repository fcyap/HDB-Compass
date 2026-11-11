# Experiment 1: `haiku55`

Rows: 270 with the auditor on, 0 with it off.

## Task success by variant and category (Wilson 95% interval)

| Variant | All | A | B | C | D | E | F | Tool F1 | Param EM | CSR | P50 s | P95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 35/90 (29%–49%) | 12/15 (55%–93%) | 6/15 (20%–64%) | 10/15 (42%–85%) | 2/15 (4%–38%) | 1/15 (1%–30%) | 4/15 (11%–52%) | 0.72 | 0.51 | 0.73 | 9.65 | 26.79 |
| B | 42/90 (37%–57%) | 10/15 (42%–85%) | 4/15 (11%–52%) | 10/15 (42%–85%) | 13/15 (62%–96%) | 2/15 (4%–38%) | 3/15 (7%–45%) | 0.77 | 0.73 | 0.81 | 13.86 | 25.32 |
| C | 76/90 (76%–91%) | 11/15 (48%–89%) | 14/15 (70%–99%) | 9/15 (36%–80%) | 12/15 (55%–93%) | 15/15 (80%–100%) | 15/15 (80%–100%) | 0.94 | 0.73 | 1.00 | 10.32 | 37.01 |

## Category F (adversarial and safety) by variant

| Variant | Handled | Stopped by the input guardrail |
|---|---|---|
| A | 4/15 | 0 |
| B | 3/15 | 0 |
| C | 15/15 | 15 |

## What the runtime auditor did (Variant C, auditor on)

- answers: 90; rejected at least once: 11 (12.2%); mean rewrites per answer: 0.2
- loops that ran out of retries (the model's sentences were then removed): 2
- rejected although the final answer scored correct: 8
- median latency: 23.36 s when rejected, 9.22 s when not
