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

Both are plain LangGraph nodes `(HDBCompassState) -> dict` that Member 4 wires in.

- **Input guardrail** runs first. Return `{"messages": [...]}` unchanged to pass,
  or `{"draft_response": "<refusal text>", "auditor_verdict": {"passed": True, "critique": "blocked_by_input_guardrail", "retry_count": 0}}`
  to short-circuit to the end. PII redaction rewrites the last user message in place.
- **Auditor** runs after synthesis. It reads `draft_response`, `sql_market_results`,
  `spatial_block_candidates`, `financial_breakdown`, `rag_policy_passages` and returns
  `{"auditor_verdict": AuditorVerdict(...).model_dump()}`. The graph loops back to
  synthesis while `passed is False and retry_count < architecture.auditor_max_retries` (2),
  passing `critique` into the synthesis prompt. Set `retry_count` = number of rejections so far (previous value + 1 on each rejection); the draft is rewritten after rejections 1 and 2 and the third rejection ends the loop (3 auditor calls max).
- Both nodes must get their LLM from `agents.llm.get_llm()` so the benchmark uses one model.

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
