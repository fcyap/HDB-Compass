# Experiment 1: `q330b90`

Rows: 270 with the auditor on, 0 with it off.

## Task success by variant and category (Wilson 95% interval)

| Variant | All | A | B | C | D | E | F | Tool F1 | Param EM | CSR | P50 s | P95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 20/90 (15%–32%) | 10/15 (42%–85%) | 2/15 (4%–38%) | 4/15 (11%–52%) | 2/15 (4%–38%) | 0/15 (0%–20%) | 2/15 (4%–38%) | 0.70 | 0.45 | 0.39 | 11.37 | 31.29 |
| B | 25/90 (20%–38%) | 10/15 (42%–85%) | 0/15 (0%–20%) | 3/15 (7%–45%) | 12/15 (55%–93%) | 0/15 (0%–20%) | 0/15 (0%–20%) | 0.79 | 0.71 | 0.26 | 21.75 | 47.19 |
| C | 76/90 (76%–91%) | 10/15 (42%–85%) | 14/15 (70%–99%) | 7/15 (25%–70%) | 15/15 (80%–100%) | 15/15 (80%–100%) | 15/15 (80%–100%) | 0.96 | 0.72 | 1.00 | 14.39 | 25.21 |

## Category F (adversarial and safety) by variant

| Variant | Handled | Stopped by the input guardrail |
|---|---|---|
| A | 2/15 | 0 |
| B | 0/15 | 0 |
| C | 15/15 | 15 |

## What the runtime auditor did (Variant C, auditor on)

- answers: 90; rejected at least once: 44 (48.9%); mean rewrites per answer: 1.067
- loops that ran out of retries (the model's sentences were then removed): 24
- rejected although the final answer scored correct: 39
- median latency: 19.98 s when rejected, 9.82 s when not
