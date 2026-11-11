# Experiment 1: `gptoss90`

Rows: 270 with the auditor on, 0 with it off.

## Task success by variant and category (Wilson 95% interval)

| Variant | All | A | B | C | D | E | F | Tool F1 | Param EM | CSR | P50 s | P95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 14/90 (9%–24%) | 5/15 (15%–58%) | 4/15 (11%–52%) | 3/15 (7%–45%) | 2/15 (4%–38%) | 0/15 (0%–20%) | 0/15 (0%–20%) | 0.61 | 0.47 | 0.72 | 19.48 | 84.26 |
| B | 23/90 (18%–35%) | 3/15 (7%–45%) | 4/15 (11%–52%) | 5/15 (15%–58%) | 9/15 (36%–80%) | 0/15 (0%–20%) | 2/15 (4%–38%) | 0.77 | 0.77 | 0.71 | 56.00 | 153.36 |
| C | 73/90 (72%–88%) | 12/15 (55%–93%) | 13/15 (62%–96%) | 6/15 (20%–64%) | 12/15 (55%–93%) | 15/15 (80%–100%) | 15/15 (80%–100%) | 0.94 | 0.75 | 1.00 | 71.63 | 161.59 |

## Category F (adversarial and safety) by variant

| Variant | Handled | Stopped by the input guardrail |
|---|---|---|
| A | 0/15 | 0 |
| B | 2/15 | 0 |
| C | 15/15 | 15 |

## What the runtime auditor did (Variant C, auditor on)

- answers: 90; rejected at least once: 3 (3.3%); mean rewrites per answer: 0.044
- loops that ran out of retries (the model's sentences were then removed): 0
- rejected although the final answer scored correct: 0
- median latency: 49.84 s when rejected, 71.63 s when not
