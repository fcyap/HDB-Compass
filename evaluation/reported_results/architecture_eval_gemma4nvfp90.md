# Experiment 1: `gemma4nvfp90`

Rows: 270 with the auditor on, 0 with it off.

## Task success by variant and category (Wilson 95% interval)

| Variant | All | A | B | C | D | E | F | Tool F1 | Param EM | CSR | P50 s | P95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 27/90 (22%–40%) | 10/15 (42%–85%) | 2/15 (4%–38%) | 6/15 (20%–64%) | 2/15 (4%–38%) | 0/15 (0%–20%) | 7/15 (25%–70%) | 0.70 | 0.41 | 0.40 | 6.88 | 17.28 |
| B | 31/90 (25%–45%) | 7/15 (25%–70%) | 3/15 (7%–45%) | 6/15 (20%–64%) | 11/15 (48%–89%) | 2/15 (4%–38%) | 2/15 (4%–38%) | 0.76 | 0.65 | 0.64 | 19.65 | 40.97 |
| C | 73/90 (72%–88%) | 10/15 (42%–85%) | 14/15 (70%–99%) | 7/15 (25%–70%) | 12/15 (55%–93%) | 15/15 (80%–100%) | 15/15 (80%–100%) | 0.95 | 0.71 | 1.00 | 15.00 | 37.50 |

## Category F (adversarial and safety) by variant

| Variant | Handled | Stopped by the input guardrail |
|---|---|---|
| A | 7/15 | 0 |
| B | 2/15 | 0 |
| C | 15/15 | 15 |

## What the runtime auditor did (Variant C, auditor on)

- answers: 90; rejected at least once: 0 (0.0%); mean rewrites per answer: 0.0
- loops that ran out of retries (the model's sentences were then removed): 0
- rejected although the final answer scored correct: 0
- median latency: None s when rejected, 15.0 s when not
