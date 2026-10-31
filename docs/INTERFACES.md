# HDB-Compass — Integration Contract (Member 4)

This page is the agreement between the orchestration layer (Member 4) and every
other workstream. If you need to change anything here, open a PR that touches
`agents/schemas.py` and this file together; Member 4 reviews it.

Status: **v1, frozen for Phase 1 (22 Oct 2026).** Member 2's spatial tools were
already implemented and are described as-is.

`query_resale_blocks_by_criteria` is configured as real. It accepts four inputs
and returns shared block records. `QueryResaleBlocksOutput` includes optional
`metadata` with snapshot, window, lease reference and truncation information.
Numeric inputs reject negative or nonfinite prices and lease limits outside
0–99 years. See [setup](../README.md#setup) and
[market semantics](MARKET_DATA.md). The BTO comparison remains mocked. The
unified MCP server registers the real query; the agent registry continues to
execute Python directly.

`compute_town_price_timeseries` is also configured as real. Its three input
names are unchanged; town/type filters are normalized and `lookback_years` is
an integer from 1 to 30. Series points add optional `is_complete`. Summaries
add optional quarter, count and comparison fields; `yoy_appreciation_pct` is
nullable when no complete same-quarter prior-year comparison exists, with
`yoy_unavailable_reason` explaining why. Optional output `metadata` records
source, coverage, requested window and towns with insufficient observations.
Existing mock outputs remain valid. See [market semantics](MARKET_DATA.md#town-price-trends).

The three financial tools are real and registered on the unified server.
Their expanded contracts add explicit eligibility, age, lease, valuation,
funding and fee inputs. Responses add `status`, missing inputs, notes, versioned
rules and sources; numeric outputs are nullable for incomplete or unsupported
scenarios. See [financial contracts](FINANCIAL_TOOLS.md) for the supported scope.
These extensions need Member 4's integration review against the Phase 1 contract.

`ExtractedConstraints.financial_inputs` optionally groups typed `grants`,
`affordability` and `upfront` inputs. Confirmations must come from supplied facts.
The deterministic real-tool sequence passes successful grant results forward;
`grants_included=true` prevents adding grants again in the supervisor. Affordability
excludes purchase costs explicitly. Upfront funding requires a positive price,
valuation, actual candidate loan and quoted fees. Mock runners filter expanded
arguments to the legacy fixture signatures; no real-tool validation is bypassed.

`query_comparable_resale_transactions` adds transaction-level comparison around
verified coordinates, with optional area/lease/storey/date bounds. It is real,
available to the market specialist and registered on both market and unified
MCP servers. Its full Pydantic contract is advertised by the unified server.
`ExtractedConstraints.comparable_resale` carries explicit matching criteria;
the deterministic fallback routes and executes it without requiring a town.
Location resolution remains the spatial workstream's responsibility. Source
coverage, missing geocodes, low samples and truncated examples are explicit.
See [comparable semantics](MARKET_DATA.md#comparable-resale-transactions).

## 1. The 12 tools and who owns them

| # | Tool | Owner | Agent | Input model | Output model |
|---|---|---|---|---|---|
| 1 | `query_resale_blocks_by_criteria` | Member 1 | market_sql | `QueryResaleBlocksInput` | `QueryResaleBlocksOutput` |
| 2 | `compute_town_price_timeseries` | Member 1 | market_sql | `ComputeTownPriceTimeseriesInput` | `ComputeTownPriceTimeseriesOutput` |
| 3 | `compare_bto_launch_vs_resale` | Member 1 | market_sql | `CompareBtoLaunchVsResaleInput` | `CompareBtoLaunchVsResaleOutput` |
| 4 | `filter_blocks_by_amenity_proximity` | Member 2 ✅ | spatial | `FilterBlocksByAmenityProximityInput` | `List[SpatialBlockRecord]` |
| 5 | `verify_primary_school_distance` | Member 2 ✅ | spatial | `VerifyPrimarySchoolDistanceInput` | `List[str]` ("Blk 123 Bishan St 13 (420m)") |
| 6 | `compute_dual_transit_commute` | Member 2 ✅ | spatial | `ComputeDualTransitCommuteInput` | `List[Dict]` |
| 7 | `sync_interactive_map_ui` | Member 2 ✅ | spatial | `SyncInteractiveMapUiInput` | `Dict` (see §4) |
| 8 | `calculate_cpf_housing_grants` | Member 1 | financial | `CalculateCpfHousingGrantsInput` | `CalculateCpfHousingGrantsOutput` |
| 9 | `calculate_maximum_affordability` | Member 1 | financial | `CalculateMaximumAffordabilityInput` | `CalculateMaximumAffordabilityOutput` |
| 10 | `calculate_bsd_and_upfront_outlay` | Member 1 | financial | `CalculateBsdAndUpfrontOutlayInput` | `CalculateBsdAndUpfrontOutlayOutput` |
| 11 | `retrieve_hdb_policy_knowledge` | Member 3 | policy_rag | `RetrieveHdbPolicyKnowledgeInput` | `RetrieveHdbPolicyKnowledgeOutput` |
| 12 | `query_comparable_resale_transactions` | Member 1 | market_sql | `QueryComparableResaleInput` | `QueryComparableResaleOutput` |

All models live in `agents/schemas.py`. The original Phase 1 inventory had 11
tools; comparable-sale analytics adds a twelfth. Variant A gets all 12;
Variant B's combined agent gets the 11 non-policy tools and its Policy agent
gets tool 11. The contract extension needs Member 4's integration review.
Earlier evaluation runs retain their original inventory and mock/real settings;
new evaluations must record the expanded inventory.
`relax_unsatisfied_constraints` is a deterministic supervisor **node**, not an
LLM-callable tool, so it is not counted.

### How to deliver a real tool (Members 1 and 3)

1. Put it in the module named in `agents/tool_registry.py::REAL_IMPL`
   (`mcp_server/market_sql_tools.py`, `mcp_server/comparable_resale_tools.py`, `mcp_server/financial_calc_tools.py`,
   `mcp_server/policy_rag_tools.py`). Keep a plain callable for registry access;
   registration can happen centrally in `mcp_server/server.py` or on a domain
   `FastMCP` instance as in `mcp_server/spatial_onemap_tools.py`.
2. Accept the **exact** parameter names and types of the input model. Return a
   plain `dict` / `list` that validates against the output model
   (`OutputModel(**result)` must not raise). The tests in
   `tests/test_contract.py` show the pattern; add a parametrised case for your tool.
3. Flip your tool from `mock` to `real` in `config/experiment_config.yaml → tools.use_mock`.
4. Register it on the unified server in `mcp_server/server.py`, preserving the
   Pydantic input/output schemas in the advertised MCP contract.

Until then the orchestration layer runs against `mcp_server/mock_tools.py`,
which returns deterministic fixtures with the same shapes.

If a tool module fails to import (a missing package, a syntax error), only its own tools are affected: the registry
prints one loud warning and every call to those tools returns `{"status": "unavailable", "error": "<import error>"}`
(logged as a failed call), while the rest of the graph keeps answering. The answer says what could not be checked
(`supervisor.NOT_CHECKED`, e.g. "Prices, remaining leases and flat types could not be checked"), and when a block
search could not run, the blocks are said to match only "the constraints that could be checked".

**Tools that run only when asked.** `calculate_bsd_and_upfront_outlay` and `query_comparable_resale_transactions`
run only when the user's own words ask for them (`supervisor.requested_tools`, read sentence by sentence: upfront or
initial costs, stamp duty or BSD, cash or CPF needed, a down payment or a price they are buying for; comparables,
recent or nearby transactions, similar flats, or what flats sold or went for). Coordinates the user typed, read into
`comparable_resale` and checked against their words, are an ask for comparable sales too. A refusal ("we don't care
about stamp duty") or a rules question ("the ABSD rules") is not an ask. Code records the asks in
`user_constraints["requested_tools"]`; the no-model plans, the financial sequence and the evidence gate follow it,
and Variant C's specialists are offered those two tools only when asked. `requested_tools` is never shown to a model
(`supervisor.constraints_for_model`), and Variants A and B keep the full tool list, so their models' unasked calls
still count against Tool F1.

### Block identity

Every tool that returns blocks includes `block`, `street`, `town` (plus
`postal`, `lat`, `lng` when known). The supervisor intersects candidates on
`schemas.block_key(block, street)` = `"123 BISHAN ST 13"`. Member 1's SQL
output must use the same street spelling as the data.gov.sg resale dataset
(upper-case, abbreviated, e.g. `BT MERAH VIEW`), which matches Member 2's index.

## 2. Shared state (`agents/state.py`)

Unchanged from proposal §4.2. Member 4 added four optional keys:

| Key | Type | Written by | Purpose |
|---|---|---|---|
| `tool_call_log` | `List[ToolCallRecord]` (reducer: append) | every tool call via `ToolRunner` | Tool Selection F1, Parameter Exact-Match |
| `relaxation` | `RelaxUnsatisfiedConstraintsOutput` | supervisor | use case E2 |
| `variant` | `"A" \| "B" \| "C"` | entry point | which architecture ran |
| `errors` | `List[str]` (reducer: append) | any node | non-fatal failures surfaced to UI and eval |

Sub-agents write **only their own slot**: `sql_market_results`,
`spatial_block_candidates`, `financial_breakdown`, `rag_policy_passages`.
Only the supervisor writes `candidate_blocks_intersection`, `draft_response`
and `map_ui_payload`. Only the auditor writes `auditor_verdict`.

## 3. Guardrail and auditor nodes (Member 5)

Put both functions in **`agents/guardrails.py`**. `agents/supervisor.py` imports that
module once, at import time, and wires whichever of the two names it finds; a missing
name, or a module that fails to import, leaves that node as a pass-through, and the app's
trace then says "Not installed yet (pass-through)". Check the wiring with
`python -c "from agents import supervisor as sv; print(sv._INPUT_GUARDRAIL, sv._AUDITOR)"`.
Tests can inject either node without the module: `sv.set_hooks(input_guardrail=..., auditor=...)`.
Both nodes run in **Variant C and the app only**; Variants A and B have neither (by design).

**`input_guardrail_node(state) -> dict`** runs first, before constraint extraction.
- Read the user's new message with `sv.last_user_query(state)`. `state["messages"]` is a list
  of `{"role", "content"}` dicts and also holds the earlier turns of the same conversation.
- To pass, return `{}`. Do not return `messages`: that key appends, so returning it adds copies
  of the conversation instead of replacing it. The node therefore cannot rewrite or redact the
  user's text; agree a new state key with Member 4 first if redaction is needed.
- To block, return `{"draft_response": "<refusal text>", "auditor_verdict": {"passed": True, "critique": "blocked_by_input_guardrail", "retry_count": 0}}`.
  A non-empty `draft_response` sends the graph straight to `finalize`, which shows that text as the answer.
- Optionally add `"errors": ["input_guardrail: <reason>"]` (also appends) so the evaluation sees why.

**`runtime_auditor_node(state) -> {"auditor_verdict": dict}`** runs after `synthesize`.
- Return `{"auditor_verdict": AuditorVerdict(...).model_dump()}` (`agents/schemas.py`), a plain
  dict, not the model object: `passed: bool`, `critique: str`, `retry_count: int`,
  `ungrounded_claims: List[str]`.
- Evidence to judge against: `candidate_blocks_intersection`, `relaxation`, `sql_market_results`,
  `spatial_block_candidates`, `financial_breakdown`, `rag_policy_passages`, `user_constraints`.
- **What to judge in Variant C.** The answer's facts section (blocks, prices, leases, distances,
  relaxations, grants, budget) is written by code from the tool results, and code has already
  removed every opening sentence that states a figure. Only the opening before it is written by
  the model, so judge only that part (claims, contradictions with the facts, invented rules),
  never the facts section:

  ```python
  facts = sv.template_response(state) if sv.writes_facts_in_code(state) else ""
  draft = state["draft_response"]
  opening = draft.split(facts, 1)[0].strip() if facts and facts in draft else draft
  ```

  `template_response(state)` is the exact text the synthesis appended, and it is the same on
  every retry. Text after the facts (the HFE disclaimer, a follow-up question) is code-written
  too. An empty opening means the model wrote nothing (it failed or was off): pass. A rejection
  only rewrites the opening, so rejecting because of the facts section cannot change anything.
  When `architecture.code_written_facts` is off, `writes_facts_in_code` is false and the whole
  draft is the model's.
- On rejection the graph loops back to `synthesize` with `critique` in the prompt. Set
  `retry_count` = number of rejections so far (previous value + 1 on each rejection; keep the
  previous value on a pass). With `architecture.auditor_max_retries` = 2 the draft is rewritten
  after rejections 1 and 2 and the third rejection ends the loop (3 auditor calls at most).

**Both nodes:**
- Get the model only from `agents.llm.get_llm()`, so the benchmark uses one model. It returns
  `None` under `LLM_PROFILE=none`; the node must then fall back to rules or pass, so the
  no-model tests and evaluation still run.
- Never raise: an exception stops the whole answer. Catch it, pass, and add
  `"errors": ["<node>: <ExceptionType>"]` so the failure is visible.
- Put them in `agents/guardrails.py` as `input_guardrail_node` and `runtime_auditor_node`. While that file does not
  exist both run as pass-throughs, quietly. If it exists but fails to import or lacks a node, the Supervisor prints a
  loud warning, keeps the problem in `supervisor.HOOK_LOAD_PROBLEMS`, adds a note to every answer's `errors`, and the
  evaluation records it in its run settings (`hook_load_problems`).

## 4. Frontend ↔ backend (Member 2) — WebSocket `/ws`

The React app already speaks this protocol; the 5-agent graph will keep it.

Client → server: `{"type": "RUN_AGENT", "query": str, "radius_meters": float, "lat"?: float, "lng"?: float}`

Server → client, in order:
- `{"type": "AGENT_STARTED"}`
- one or more `{"type": "STEP_UPDATE", "node": str, "logs": [{"step","message","status"}], "geojson_buffer"?, "geojson_amenities"?, "viewport"?: {"lat","lng","zoom"}, "summary"?: str, "qualifying_blocks"?: [str], "nearby_amenities"?: [...], "market_summary"?: str, "financial_summary"?: str, "policy_notes"?: str}`
- `{"type": "AGENT_FINISHED"}` or `{"type": "ERROR", "error": str}`

Member 4's change in Phase 2: `backend/main.py` will stream the **Variant C graph**
instead of `geo_agent` directly, emitting one `STEP_UPDATE` per LangGraph node and
filling `market_summary` / `financial_summary` / `policy_notes` from the
sub-agent slots. Member 2's `geo_agent` becomes the spatial sub-agent inside it
(adapter already in `agents/arch_3_specialist_5agent.py`). No frontend change needed.

## 5. LLM

Everyone calls `agents.llm.get_llm()`. Never instantiate a chat model directly.
Profiles (`config → llm.profiles`): `ollama` for local dev, `vllm` for the SCIS
GPU cluster, `none` for deterministic no-model runs. No paid provider. Select with `LLM_PROFILE` in `.env`.
The model is the same for A/B/C and for the auditor — that is a condition of
Experiment 1.

## 6. Evaluation hooks (Member 5)

- Each variant exposes `run(query: str, map_state: dict | None = None, previous: dict | None = None) -> HDBCompassState`
  from `agents/arch_1_monolith.py`, `agents/arch_2_coarse.py`, `agents/arch_3_specialist_5agent.py`.
  `previous` is the final state of the last turn in the same conversation (see §7).
- Category E (`evaluation/category_e_queries.json`, built by `python -m evaluation.ground_category_e`) is grounded on
  the real database: each item has `stated_constraints`, `financial_inputs` (the facts the question states),
  `ground_truth_facts` (matching block keys, grant total, maximum purchase price, relaxations, snapshot and rule
  version) and, for the two deliberately incomplete questions, a `follow_up` turn with its own ground truth.
  `evaluation/architecture_metrics.ground_truth_scores(item, state)` scores an answer against it.
- Category C (Member 3's `evaluation/category_c_queries.json`) goes through the same runner with
  `--dataset category_c`. `ground_truth_scores` returns `{"kind": "policy", "clauses_found", "clauses_total",
  "clause_coverage", "clauses_missing", "known_wrong_clauses", "success"}`: fully right when every `key_clauses` entry
  appears in the answer, matched by Member 3's rule from `run_rag_ablation.is_clause_in_passage` (copied as
  `clause_in_answer`; a test keeps the two in step). Clauses known to be wrong are listed in
  `architecture_metrics.KNOWN_WRONG_CLAUSES` and shown as a caveat in the analysis; correcting them is Member 3's call.
- Every task-success rate in the analysis carries a Wilson 95% interval (`architecture_metrics.wilson_interval`), and
  A and B are each compared with C question by question (exact McNemar test). Tokens and latency are reported as a
  cost proxy: no paid API is used and GPU time is not billed per query.
- Predicted tool calls = `state["tool_call_log"]` (tool name + arguments). Compare against
  `expected_mcp_calls` in `evaluation/golden_dataset_90.json`.
- Routing trajectory = `state["active_subagents"]`.
- Latency/tokens come from LangSmith; the runner should checkpoint per query to JSONL.

## 7. Follow-up questions for missing financial facts (Member 4)

When the grant or affordability calculator returns `needs_information`, the Supervisor sets
`state["clarification"] = {"needed": [calculator fields], "question": "To finish your grant and budget estimate, please
tell me ..."}`, states no budget, and ends the answer with that question. Upfront costs are not asked for during a
search: they need one chosen flat's valuation, loan and quoted fees.

To continue the conversation, pass the previous final state back: `run(reply, previous=state)`. The earlier messages,
constraints and financial facts carry over (`prior_constraints`, `pending_clarification`); the reply's facts are merged
field by field and the request is re-run. For the web app (Member 2), keep the last state per chat session and pass it
on each message. Variants B and C share this memory through the Supervisor; Variant A receives the conversation text.


**Where calculator facts come from.** The financial facts in `user_constraints["financial_inputs"]` are read by code
from the user's own messages (`stated_financial_inputs` in `agents/supervisor.py`), never taken from the model's
extraction: in the 27 Oct live run the model turned a married citizen couple into `SC_Single`, invented ages and an
interest rate, and passed zero grants. Nothing is inferred ("couple" alone gives no citizenship; the youngest core
member is the youngest buyer only when the user says there are no children). A section exists only when the user
stated one of its facts, and the finance agent is routed only then. At the tool boundary the same stated facts
override the model's calculator arguments, and a confirmed grant total is passed to the affordability and upfront
calculators (`guard_financial_args`). Calculator inputs and outputs (Member 1) are unchanged.
