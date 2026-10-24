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
[market semantics](MARKET_DATA.md). Other Member 1 tools remain mocked. The
unified MCP server registers the real query; the agent registry continues to
execute Python directly.

## 1. The canonical 11 tools and who owns them

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

All models live in `agents/schemas.py`. This count resolves the proposal's
11-vs-12 ambiguity: Variant A (monolith) gets all 11; Variant B's combined
Data/Spatial/Finance agent gets tools 1–10 and its Policy agent gets tool 11.
`relax_unsatisfied_constraints` is a deterministic supervisor **node**, not an
LLM-callable tool, so it is not counted.

### How to deliver a real tool (Members 1 and 3)

1. Put it in the module named in `agents/tool_registry.py::REAL_IMPL`
   (`mcp_server/market_sql_tools.py`, `mcp_server/financial_calc_tools.py`,
   `mcp_server/policy_rag_tools.py`). Decorate with `@mcp.tool()` on your own
   `FastMCP` instance, exactly like `mcp_server/spatial_onemap_tools.py`.
2. Accept the **exact** parameter names and types of the input model. Return a
   plain `dict` / `list` that validates against the output model
   (`OutputModel(**result)` must not raise). The tests in
   `tests/test_contract.py` show the pattern; add a parametrised case for your tool.
3. Flip your tool from `mock` to `real` in `config/experiment_config.yaml → tools.use_mock`.
4. Register it on the unified server in `mcp_server/server.py` with `mcp.add_tool(...)`.

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

- Each variant exposes `run(query: str, map_state: dict | None = None) -> HDBCompassState`
  from `agents/arch_1_monolith.py`, `agents/arch_2_coarse.py`, `agents/arch_3_specialist_5agent.py`.
- Predicted tool calls = `state["tool_call_log"]` (tool name + arguments). Compare against
  `expected_mcp_calls` in `evaluation/golden_dataset_90.json`.
- Routing trajectory = `state["active_subagents"]`.
- Latency/tokens come from LangSmith; the runner should checkpoint per query to JSONL.
