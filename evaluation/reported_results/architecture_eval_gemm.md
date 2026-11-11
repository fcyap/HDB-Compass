# Experiment 1: `gemm`

Rows: 270 with the auditor on, 0 with it off.

## Task success by variant and category (Wilson 95% interval)

| Variant | All | A | B | C | D | E | F | Tool F1 | Param EM | CSR | P50 s | P95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 26/90 (21%–39%) | 9/15 (36%–80%) | 1/15 (1%–30%) | 6/15 (20%–64%) | 3/15 (7%–45%) | 0/15 (0%–20%) | 7/15 (25%–70%) | 0.69 | 0.41 | 0.46 | 10.44 | 81.00 |
| B | 33/90 (27%–47%) | 8/15 (30%–75%) | 5/15 (15%–58%) | 7/15 (25%–70%) | 10/15 (42%–85%) | 2/15 (4%–38%) | 1/15 (1%–30%) | 0.78 | 0.66 | 0.64 | 32.77 | 57.62 |
| C | 72/90 (71%–87%) | 9/15 (36%–80%) | 14/15 (70%–99%) | 7/15 (25%–70%) | 12/15 (55%–93%) | 15/15 (80%–100%) | 15/15 (80%–100%) | 0.95 | 0.69 | 1.00 | 22.56 | 43.82 |

## Category F (adversarial and safety) by variant

| Variant | Handled | Stopped by the input guardrail |
|---|---|---|
| A | 7/15 | 0 |
| B | 1/15 | 0 |
| C | 15/15 | 15 |

## What the runtime auditor did (Variant C, auditor on)

- answers: 90; rejected at least once: 2 (2.2%); mean rewrites per answer: 0.022
- loops that ran out of retries (the model's sentences were then removed): 0
- rejected although the final answer scored correct: 1
- median latency: 21.27 s when rejected, 22.57 s when not
