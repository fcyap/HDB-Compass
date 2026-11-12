# Report map: where each part of the written report comes from

The course asks for a concise report (about 10 pages) covering: problem and goals, architecture, data and models,
experimental setup and evaluation, results and discussion, reflection and future work, and risks. This page maps each of
those to the place in this repository that holds the evidence, so that each author can write and verify their own
section against a source. It is an index, not a draft: every sentence of the report is written and checked by the
student who owns that section (course AI-usage policy).

Numbers marked ✔ are copied from the README tables as of `main` after PR #34 and can be re-checked with
`python evaluation/reported_results/verify_results.py` (no model, GPU or database needed). Anything else, read it from
the cited table rather than from this page.

## 1. Problem, motivation, goals
| Report part | Source |
|---|---|
| Who buys, what decisions are hard (price fairness, grants, loans, location, rules) | [README intro](../README.md#hdb-compass-multi-agent--fastmcp-decision-support-system-for-singapore-public-housing), [System overview](../README.md#1-system-overview) |
| Goals and what is out of scope | [Known limitations and next steps](../README.md#5-known-limitations-and-next-steps), [Threats to validity](../README.md#threats-to-validity-and-limitations) |
| Related and baseline approaches | A (monolith) and B (coarse split) are the baselines for C: [Architecture comparison](../README.md#architecture-comparison). The README does not survey the literature; add a short related-work paragraph (agentic RAG, tool-using agents, guardrails) with real references in the report. |

## 2. System architecture, components, data flow
| Report part | Source |
|---|---|
| Architecture diagram | [Agent graph, Variant C](../README.md#2-agent-graph-variant-c-supervisor--4-specialists--auditor), [A/B/C side by side](../README.md#2b-the-three-variants-side-by-side-a-b-and-c), [message flow](../README.md#1b-how-one-message-is-handled-end-to-end) |
| Agents, tools and who may call what | [Agents and tools](../README.md#3-agents-and-the-tools-they-can-use), `agents/schemas.py` (`TOOL_OWNER`) |
| Design choices and what each costs | [Design choices and trade-offs](../README.md#design-choices-and-trade-offs) |
| Context size per architecture | [Why context size matters](../README.md#why-context-size-matters-per-architecture) |
| Interfaces between members' components | `docs/INTERFACES.md`, `docs/FINANCIAL_TOOLS.md`, `docs/MARKET_DATA.md` |

## 3. Data and models
| Report part | Source |
|---|---|
| Datasets, sources, pre-processing | [Official data sources](../README.md#official-government-data-sources), `docs/MARKET_DATA.md`, `data/policy_sources/raw/manifest.json` (URLs, capture dates, hashes of the 55 policy pages) |
| Models, serving flags, context windows | [Models, context window and token budget](../README.md#models-context-window-and-token-budget), [Setup used for these runs](../README.md#setup-used-for-these-runs) |
| Benchmark (90 questions, 6 categories of 15, how truth was built) | [Evaluation benchmark](../README.md#evaluation-benchmark-and-member-5--guardrail-auditor-and-categories-d-and-f), `evaluation/build_*.py` |

## 4. Experimental setup and evaluation method
| Report part | Source |
|---|---|
| Metrics: task success, Tool F1, latency, tokens; Wilson intervals; exact McNemar | [Results and findings](../README.md#results-and-findings-as-of-11-nov-2026) (method paragraph), `evaluation/architecture_metrics.py` |
| Fairness between variants (same tools, same model, seed 42, temperature 0) | [Architecture comparison](../README.md#architecture-comparison) last rows, `tests/test_experiment_fairness.py` |
| Reproduce a run | [Setup used for these runs](../README.md#setup-used-for-these-runs), [REPRODUCE.md](REPRODUCE.md) |

## 5. Results and discussion
Headline, Qwen3.8-27B-FP8 re-run on `4ff2767`, Category F excluded (75 questions; F measures the guardrail, which only C has):

| Variant | Passed ✔ | Tool F1 ✔ |
|---|---|---|
| A · Monolith | 31/75 (41.3%) | 0.749 |
| B · Coarse | 40/75 (53.3%) | 0.814 |
| C · Specialist | 56/75 (74.7%) | 0.936 |

C vs B without F: 23 vs 7 discordant pairs, p = 0.0052 (initial run on `23c9076`: 19 vs 8, p = 0.052) ✔. C passes
71 to 76 of 90 on every model (open-weight re-runs and the hosted Haiku row), while A and B vary widely by model:
[Cross-model comparison](../README.md#cross-model-comparison-task-success-and-tool-f1-per-variant).

| Report part | Source |
|---|---|
| Headline table, paired tests | [Headline](../README.md#headline-qwen38-27b-fp8-all-three-variants-90-questions) |
| Same architecture across models | [Cross-model comparison](../README.md#cross-model-comparison-task-success-and-tool-f1-per-variant) |
| Did the later code change anything? (no detectable change) | [Re-run: before and after](../README.md#re-run-on-the-updated-code-4ff2767-before-and-after) |
| Does the auditor help? (no measured benefit) | [Auditor on vs off](../README.md#auditor-on-vs-off-variant-c-qwen38-27b-fp8-90-paired-questions) |
| Safety category and the scorer caveat | [Category F](../README.md#category-f-and-the-scorer-read-before-quoting-f) |
| Hosted reference row (Claude Haiku 5.5): what is and is not comparable | [Hosted reference row](../README.md#hosted-reference-row-what-is-and-is-not-comparable) |
| Qualitative analysis: why answers fail | [Qualitative checks](../README.md#qualitative-checks-why-qwen38-scores-slightly-lower-in-c), `evaluation/reported_results/qualitative_checks.json` |
| Code help and fallbacks, infrastructure effects | [Infrastructure and code-help effects](../README.md#infrastructure-and-code-help-effects) |
| Policy RAG experiment (retrieval variants, embedding comparison) | [Experiment 2: policy RAG](../README.md#experiment-2-policy-rag), [Member 3 section](../README.md#member-3--policy-rag--regulatory-knowledge-base-category-c) |
| Guardrail and auditor without a model | [Guardrail and auditor](../README.md#guardrail-and-auditor-no-model-needed), `evaluation/auditor_fault_injection.py` |

## 6. How the design and the testing evolved (for the reflection)
| Stage | What happened | Source |
|---|---|---|
| First live runs on Qwen3-30B-A3B (31 Oct to 4 Nov) | what broke and what was fixed | [Earlier live runs](../README.md#earlier-live-runs-on-qwen3-30b-a3b-31-oct---4-nov-what-we-found-and-fixed) |
| Initial four-model runs, code `23c9076` | first complete Experiment 1 | [Re-run: before and after](../README.md#re-run-on-the-updated-code-4ff2767-before-and-after) |
| Supervisor, policy prompt and corpus changes on `main` | what changed in the code between the runs | same section, and `git log 23c9076..4ff2767 -- agents` |
| Re-run on `4ff2767`, Haiku reference on `a6e8557` | whether the changes mattered | [Hosted reference row](../README.md#hosted-reference-row-what-is-and-is-not-comparable) |
| What went wrong during the runs (GPU limits, timeouts, a context window mix, scorer flaws) | evidence of debugging | [What changed or went wrong](../README.md#what-changed-or-went-wrong-during-the-runs) |
| Testing: about 2,000 tests, the result audit, fault injection | validation evidence | [Verify tools and agents](../README.md#4-verify-tools-and-agents), `evaluation/reported_results/validation.md` |

## 7. Risks, guardrails, limitations, future work
| Report part | Source |
|---|---|
| Risks and mitigations (injection, identifiers, speculative advice, neighbourhood bias, ungrounded figures) | [Guardrail and auditor](../README.md#guardrail-and-auditor-no-model-needed), `agents/guardrails.py`, `agents/sanitizer.py` |
| Limits of the evidence (one benchmark written by the team, one seed, single run per code version, parallel streams) | [Threats to validity and limitations](../README.md#threats-to-validity-and-limitations) |
| Open question not yet explained: C2-05 (policy) passed in Variant C in the initial runs of Qwen3-30B-A3B, gpt-oss-20b and Gemma 4 and failed in all three re-runs (it passed in both Qwen3.8 runs) | question-level CSVs `q330b90.csv`/`q3m.csv`, `gptoss90.csv`/`gptm.csv`, `gemma4nvfp90.csv`/`gemm.csv`; stated in the before/after section |
| Future work | [Next steps](../README.md#next-steps) |

## 8. Teamwork (each student documents the others' work)
The course requires each student to document the other members' contributions. The README holds one working section per
member: [Member 1](../README.md#member-1--market-and-financial-tools-categories-a--d-status-and-remaining-work),
[Member 2](../README.md#member-2-ownership--deliverables-categories-b--e),
[Member 3](../README.md#member-3--policy-rag--regulatory-knowledge-base-category-c),
[Member 4](../README.md#member-4--multi-agent-orchestration--architecture-study-category-e) and
[Member 5](../README.md#evaluation-benchmark-and-member-5--guardrail-auditor-and-categories-d-and-f). Per-person
evidence (commits, files) comes from `git shortlog -sn` and `git log --author=... -- <path>`; the section for each
teammate's work should be checked with that teammate before it is submitted.

## Caveats to carry into the report
- Each code version was run once: do not claim that the code changes caused any difference.
- Haiku is a hosted reference row on `a6e8557` at API-default sampling (no temperature or seed sent): not a controlled comparison.
- Category F scores are affected by a scorer flaw on some correct refusals: use the 75-question figures for architecture claims.
- Test counts and the snapshot caveat: see [REPRODUCE.md](REPRODUCE.md).
