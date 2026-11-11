# Experiment 1: `q3m`

Rows: 270 with the auditor on, 0 with it off.

## Task success by variant and category (Wilson 95% interval)

| Variant | All | A | B | C | D | E | F | Tool F1 | Param EM | CSR | P50 s | P95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 21/90 (16%–33%) | 11/15 (48%–89%) | 2/15 (4%–38%) | 4/15 (11%–52%) | 2/15 (4%–38%) | 0/15 (0%–20%) | 2/15 (4%–38%) | 0.68 | 0.43 | 0.44 | 17.54 | 61.89 |
| B | 21/90 (16%–33%) | 5/15 (15%–58%) | 1/15 (1%–30%) | 6/15 (20%–64%) | 9/15 (36%–80%) | 0/15 (0%–20%) | 0/15 (0%–20%) | 0.80 | 0.68 | 0.35 | 30.40 | 69.89 |
| C | 76/90 (76%–91%) | 10/15 (42%–85%) | 14/15 (70%–99%) | 7/15 (25%–70%) | 15/15 (80%–100%) | 15/15 (80%–100%) | 15/15 (80%–100%) | 0.96 | 0.72 | 1.00 | 18.81 | 37.16 |

## Category F (adversarial and safety) by variant

| Variant | Handled | Stopped by the input guardrail |
|---|---|---|
| A | 2/15 | 0 |
| B | 0/15 | 0 |
| C | 15/15 | 15 |

## What the runtime auditor did (Variant C, auditor on)

- answers: 90; rejected at least once: 39 (43.3%); mean rewrites per answer: 1.067
- loops that ran out of retries (the model's sentences were then removed): 28
- rejected although the final answer scored correct: 34
- median latency: 28.32 s when rejected, 14.0 s when not
