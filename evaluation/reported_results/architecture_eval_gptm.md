# Experiment 1: `gptm`

Rows: 270 with the auditor on, 0 with it off.

## Task success by variant and category (Wilson 95% interval)

| Variant | All | A | B | C | D | E | F | Tool F1 | Param EM | CSR | P50 s | P95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 12/90 (8%–22%) | 5/15 (15%–58%) | 4/15 (11%–52%) | 2/15 (4%–38%) | 1/15 (1%–30%) | 0/15 (0%–20%) | 0/15 (0%–20%) | 0.62 | 0.47 | 0.77 | 37.62 | 131.02 |
| B | 23/90 (18%–35%) | 1/15 (1%–30%) | 5/15 (15%–58%) | 3/15 (7%–45%) | 11/15 (48%–89%) | 0/15 (0%–20%) | 3/15 (7%–45%) | 0.76 | 0.71 | 0.73 | 92.49 | 270.70 |
| C | 75/90 (74%–90%) | 11/15 (48%–89%) | 14/15 (70%–99%) | 7/15 (25%–70%) | 13/15 (62%–96%) | 15/15 (80%–100%) | 15/15 (80%–100%) | 0.94 | 0.75 | 1.00 | 105.86 | 254.36 |

## Category F (adversarial and safety) by variant

| Variant | Handled | Stopped by the input guardrail |
|---|---|---|
| A | 0/15 | 0 |
| B | 3/15 | 0 |
| C | 15/15 | 15 |

## What the runtime auditor did (Variant C, auditor on)

- answers: 90; rejected at least once: 10 (11.1%); mean rewrites per answer: 0.211
- loops that ran out of retries (the model's sentences were then removed): 4
- rejected although the final answer scored correct: 7
- median latency: 154.09 s when rejected, 96.27 s when not
