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
[market semantics](MARKET_DATA.md). The BTO comparison is real (official HDB launch annexes; see [BTO launch data](MARKET_DATA.md#bto-launch-data)). The
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

## 1. The 25 tools and who owns them (12 core tools + 13 add-on tools, see §8 and §11)

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
| 13 | `find_primary_schools_near_address` | Member 2 | spatial | `FindSchoolsNearAddressInput` | `FindSchoolsNearAddressOutput` |
| 14 | `get_block_transaction_history` | Member 1 | market_sql | `GetBlockTransactionHistoryInput` | `GetBlockTransactionHistoryOutput` |
| 15 | `get_market_snapshot` | Member 1 | market_sql | `GetMarketSnapshotInput` | `GetMarketSnapshotOutput` |
| 16 | `rank_towns_by_value` | Member 1 | market_sql | `RankTownsByValueInput` | `RankTownsByValueOutput` |
| 17 | `compare_resale_options` | Member 1 | market_sql | `CompareResaleOptionsInput` | `CompareResaleOptionsOutput` |
| 18 | `assess_resale_price` | Member 1 | market_sql | `AssessResalePriceInput` | `AssessResalePriceOutput` |
| 19 | `resolve_location` | Members 1+2 | spatial | `ResolveLocationInput` | `ResolveLocationOutput` |
| 20 | `get_nearby_amenities` | Members 1+2 | spatial | `GetNearbyAmenitiesInput` | `GetNearbyAmenitiesOutput` |
| 21 | `find_resale_blocks_near` | Member 1 | market_sql | `FindResaleBlocksNearInput` | `FindResaleBlocksNearOutput` |
| 22 | `get_area_resale_statistics` | Member 1 | market_sql | `GetAreaResaleStatisticsInput` | `GetAreaResaleStatisticsOutput` |
| 23 | `get_area_price_trend` | Member 1 | market_sql | `GetAreaPriceTrendInput` | `GetAreaPriceTrendOutput` |

Models for tools 1-13 live in `agents/schemas.py`; tools 14-23 live in
`agents/research_schemas.py` and are registered in the same `TOOL_OWNER` /
`TOOL_INPUT_MODEL`. The original Phase 1 inventory had 11 tools; comparable-sale
analytics, address-centred school lookup and the ten research tools add twelve.
Variant A gets all 23; Variant B's combined agent gets the 22 non-policy tools and
its Policy agent gets tool 11. The contract extension needs Member 4's integration review.
Earlier evaluation runs retain their original inventory and mock/real settings;
new evaluations must record the expanded inventory.
`relax_unsatisfied_constraints` is a deterministic supervisor **node**, not an
LLM-callable tool, so it is not counted.

Research tools take the user's address text and resolve it themselves; they never
need model-supplied coordinates. `ExtractedConstraints` gains `addresses`,
`asking_price_sgd`, `bto_project_name`, `trend_years`, six `wants_*` intent flags,
`assumptions` (disclosed defaults, e.g. "near an MRT" = 500 m) and `clarifications`
(questions to ask instead of guessing). The model is given `LLMExtractedConstraints`,
whose typed `amenity_radii_meters` is accepted by strict JSON-schema providers; the
supervisor converts it back and records `user_constraints.extraction_source`
(`llm:<method>` or `regex`). `agents/research_intents.py` holds the deterministic
plans, the coverage check (needed research tools the model skipped are run and
logged as `supervisor`, so Tool F1 still measures only the model), coordinate
provenance checks and answer/map rendering. Spatial `location_results` is a new
state slot; `spatial_block_candidates` is set only by block-filtering tools.

Address-centred school lookup uses `ExtractedConstraints.nearby_schools` and
`HDBCompassState.nearby_school_results`, separate from candidate resale blocks.
Explicit “schools near [address]” requests resolve through this tool before
any market search. Status is `ok`, `needs_information` or `unavailable`;
`schools=[]` means no matches only when status is `ok`. The tool rejects
placeholder street names and ambiguous addresses, and discloses the default
2km straight-line radius and unverified snapshot freshness. Distances are
not official MOE registration boundaries or walking routes.
The WebSocket carries the result as `school_lookup`; map markers and the
answer use that same result, with no `qualifying_blocks` resale list.

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
Profiles: `ollama` for local dev, `vllm` for the SCIS GPU cluster and `none` for
deterministic no-model runs (`config/experiment_config.yaml → llm.profiles`, local
models only, so the study runs on open weights); `groq`/`openai` for hosted
inference and `compatible` for another OpenAI-compatible endpoint
(`config/hosted_models.yaml`, for the app). `agents/llm.py` reads both files. Select
with `LLM_PROFILE` in `.env`; credentials remain in the backend environment.
Opt-in `auto` chooses the first available configured provider and retains it until
restart. Select an explicit profile for benchmarks. Configuration and tool-call
diagnostics are documented in the root README.
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


## 8. Add-on tool packs (Member 1)

The research tools and the address-centred school lookup are **add-on packs** (`agents/addons/`), on top of the 12
core tools (`agents.schemas.CORE_TOOLS`). Each add-on tool is declared once, as a `ToolAddon` (name, implementation,
owning specialist, input and output model, when Variant C offers it, a one-line prompt hint, what the answer says if
its module cannot load, and an example question). The core reads that declaration everywhere it needs it:
`TOOL_OWNER`/`TOOL_INPUT_MODEL`, `tool_registry.REAL_IMPL`, `supervisor.TASK_TOOLS`, `deterministic_plan` (so results
survive `requested_outputs`), `AGENT_SYSTEM` (the owner's prompt) and `NOT_CHECKED`, plus the MCP server.
`tests/test_addons.py` checks every one of these for every add-on, and that its example question reaches it.

To add a tool: write the implementation in `mcp_server/`, its input/output models, then one `ToolAddon` entry in a
pack (`agents/addons/research.py`, `schools.py` or a new pack listed in `agents/addons/__init__._packs`).

**Which runs see them** — `addons.offer_in` in the config (`HDB_ADDONS` overrides it for one run):

| `offer_in` | App (fast path) | Variant C | Variants A, B |
|---|---|---|---|
| `app` (default) | yes | yes, gated by intent | no: original tool lists, so Tool F1 stays comparable |
| `all` | yes | yes | yes |
| `off` | no | no | no: exactly the pre-add-on behaviour |

The Category E runner (`evaluation/run_category_e_prelim.py`) runs with `off` unless `HDB_ADDONS` is set, and
records the mode in each row's run settings.

## 9. Which place a question is about; what carries between turns (Member 1)

`research_intents.resolve_place` decides between the user's map pin, a place named in the message and a place named
in an earlier turn: "here"/"this block" is the pin; "there"/"that block"/"the same area" may be the pin or the earlier
place, and when both are possible (and differ) the app asks — `clarification = {"kind": "location", "options": [...],
"question": ...}` — and nothing runs until the reply ("the pin", "the earlier one", or a new address). A place
question naming no place uses the pin, else the earlier place, and says so in `assumptions`.

All follow-up questions share `state["clarification"]`, with a `kind`: `financial` (§7), `location`, `place` ("what
would you like to know about Blk …?") or `school` ("which primary school?"). The next turn receives it as
`pending_clarification`. One-off asks (`research_intents.TURN_ONLY_KEYS`: addresses, block history, a snapshot...)
carry into a **reply** to an open question but not into a new question; `ALWAYS_FRESH_KEYS` (assumptions, questions
asked) never carry. The app's fast path (no model call for a fully recognised question) reads one message only and
is never used for a reply.

## 10. Scope check, menus and unsupported asks (Member 1)

**Scope check (`agents/scope.py`, code only).** Stand-in for Member 5's input guardrail until it lands: `classify` sorts a
message into in-scope, off-topic (weather, laptops), other-property (condo, EC, landed), prompt-injection, a price
forecast, or small talk (hi, thanks). Anything but in-scope gets a fixed reply that lists what the app can do; no agent
or tool runs. Housing words are matched on whole words, and a bare town name counts as in scope. Mid-conversation, a
short reply to an open question is never judged on its own. Member 5's guardrail runs first when it is on; this only
fills the gap.

**Menu clarification.** An in-scope message that names nothing the tools can act on ("What flat should I buy?",
"I want to buy a bto but im not sure what to buy") no longer falls through to the policy search. It gets a numbered
menu: `clarification = {"kind": "menu", "question", "options": [{"key", "label", "text"}]}`. The user can reply with the
option number, its exact label or a short keyword; `scope.resolve_menu_reply` turns that into `menu_followup`, the
option's own follow-up question (`bto_launch`, `financial`, `rules`, `towns`, `search` kinds), or None when the reply is a
new question (then it is handled as one). The websocket `STEP_UPDATE` carries `clarification` as
`{"kind", "question", "options": [{"label", "text"}]}`; the page shows the options as chips and sends `text` on click.
Location clarifications (`kind: "location"`) have only a label, which is used as the text.

**Policy relevance floor.** `RetrieveHdbPolicyKnowledgeOutput.best_keyword_score` (new, optional float) is the best raw
BM25 score of the retrieved passages. The supervisor says "no policy passage matches" when it is below the floor instead
of printing the nearest weak passages. Member 3/4: the field is additive; the policy tool's other outputs are unchanged.

**Town price questions.** "How much is a 4-room in Tampines?", "median price of 5-room flats in Punggol" and "Is Woodlands
cheaper than Yishun" go to `get_market_snapshot` (the town snapshot) instead of a block search. With two towns the answer
adds a line saying which is cheaper and by how much.

**Unsupported asks.** `scope.unsupported_reply`: a commute question naming one workplace gets "I can't estimate a single
commute yet..." with no search, because `compute_dual_transit_commute` uses town-level averages and ignores destinations.
Two-workplace questions and searches with a commute limit are unchanged. "stamp duty on a $600k flat" (also "a flat costing
$600k") is read as a purchase price for `calculate_bsd_and_upfront_outlay` when the user asks for upfront costs; "find a $600k
flat" stays a budget. A block list says "meet every limit you set" only if the user set a limit; otherwise it reads
"Blocks with recent 4-room sales in Tampines, cheapest first."

**Routing lock.** `tests/test_scope_routing.py` holds a table of questions with the route and tools each must take. It
locks Category C (15 policy questions all go to `['policy_rag']` by regex) and the Category E routes, so a routing change
that moves either fails the test. It runs with `HDB_ADDONS=off` set inside the test module only.


## 11. Broad market questions: grouped statistics and drill-down (Member 1)

**Block features** (`mcp_server/block_features.py`). Built once per market-database snapshot and cached: for every
block with a geocode in `block_locations`, its town, nearest MRT station and distance (LRT excluded), nearest LRT,
nearest hawker centre and supermarket distances, and primary schools within 1 km. Distances are straight-line SVY21
metres from the station point in the spatial index (not station exits); interchange duplicates are merged by name.

**Tools** (research add-on pack, owner `market_sql`):
- `query_resale_statistics` - group sales by one or two of `distance_band`, `town`, `station`, `street`, `flat_type`,
  `year`, `storey_band`, `lease_band`; filters `towns`, `flat_types`, `station`, `near` (`mrt`, `mrt_lrt`, `lrt`,
  `hawker`, `supermarket`, `school`) with `max_distance_m` or `bands` (default 0-200 / 200-500 / 500-1,000 / >1,000 m);
  period `lookback_years` (default 1) or `start_month`/`end_month`; `sort_by`, `order`, `top_n` (10), `include_trend`.
  Returns per group: count, median, p25/p75, median $/sqm, 1-year and period change, optional yearly medians, and
  `drill` parameters for the next level; plus an island-wide (or filtered) baseline, the period, snapshot id and the
  distance definition. Groups with fewer than 10 sales are left out. Invalid values return the allowed lists.
- `describe_resale_data` - the catalog: towns, flat types, MRT and LRT station names, amenity categories, period
  covered, metrics and definitions.

**How a question reaches them.**
- *Rule shapes, answered by code with no model call:* statistics/history/prices/listings near a generic amenity
  (distance bands); "by town" / "which towns"; "which MRT stations are most/least expensive"; near vs far; "<town> by
  station"; floors / lease left / by year (grouped by that dimension, a "near X" becomes a filter); "over the last N
  years" / "since YYYY" (trend); "across Singapore"; region words ("the east", "north-east", "west", "north",
  "central") become that URA region's HDB towns, stated in the answer. A stated distance ("within 300m of an MRT") is
  used, clamped to 50-3,000 m with a note. Named places ("around Bishan MRT") still go to the area tools.
- *Analysis mode, the model decides:* any other data question (price, sales, median, value, premium, compare...) goes
  to the market agent with the analysis tool set (catalog, grouped statistics, area tools, block history, snapshot,
  town ranking and trends) and up to 6 tool steps; with no model it falls back to a distance-band or by-town overview.
  Off when `HDB_ADDONS=off` (the evaluation default).

**The answer.** Code writes it: a title, the headline (the biggest gap or the top group, with figures), a table of
the top groups ("...and N more"), yearly trends when asked, and notes (period, distance definition, LRT handling,
"past sales, not live listings" for listings questions, a like-for-like warning for storey/lease groupings). The model
may only add a checked opening. The map shows station or town markers.

**Drill-down.** Chips (and clicks on station/town markers) send complete questions: distance bands → by town / by
station; a town → its stations; a station → the existing area statistics within the radius; a block → its sales
history; each level has a "back" chip. They travel in `state["suggestions"]`, not as a pending clarification, so a
click is answered as a fresh question on the fast path.

**Tests.** `tests/test_analysis_questions.py` (rule shapes, analysis mode with a fake model, negatives, the chip walk);
the tool's figures are checked against independent SQL. `evaluation/category_a_queries.json` seeds Category A
questions whose ground truth comes from independent SQL (`evaluation/build_category_a.py`).

## 12. Map pin and radius (Member 1)

The page sends `radius_meters` (default 500, set by the radius control on the map, 100-3000 m, remembered in the
browser) with each chat message, next to the pin's `lat`/`lng`. The backend puts it in `map_state["radius_meters"]`.
Precedence for a question about the pin ("what's near here?", "prices around this block"): a distance stated in the
text ("within 300m") > the map control's radius > the default (500 m for "around", 1 km for nearby amenities when no
pin radius is sent). Values outside 50-3000 m (area) are clamped and the answer says so. Answers state the radius and,
when it came from the map, say "within N m of your pin (set on the map)". The pin's border is drawn by the page on its
own `pin-radius-*` layers (the agent's `agent-buffer-*` layers are unchanged).

Radius chips: an area-statistics or nearby-amenities answer about one place carries `state["suggestions"]`
(`[{"label": "Within 200 m", "text": "What are resale prices within 200m of <place>?"}]`, the current radius left out).
The websocket sends them as `clarification = {"kind": "drilldown", "question": null, "options": [...]}` (after any drill-down chips, §11), but they are
not a pending question: `state["clarification"]` stays empty, so a click is answered as a fresh question on the fast path.

## 13. Tracing and evals (LangSmith)

**Tracing** (`agents/tracing.py`). Off unless `LANGSMITH_TRACING=true`. `configure_tracing()` runs once per process
(backend start-up, the evaluation runner, the CLI): it asks LangSmith whether the key is accepted at
`LANGSMITH_ENDPOINT` (a 401/403 is retried against the other regions to say which one accepts it), and checks the
destination. On failure tracing is switched off for the process with one log line. `python -m agents.tracing --check`
prints the verdict and the fix. Tests never upload (`LANGSMITH_TRACING` is forced off).

- Destination: `LANGSMITH_PROJECT`, or an agent address (`LANGSMITH_AGENT_ID` + `LANGSMITH_AGENT_ENVIRONMENT`, runs land
  in the environment's project, e.g. `hdb-compass-local`). Half an address, or an address plus a project, is untraced.
- `run_config(variant, session_id, question_id)` names the root run `hdb-compass-<variant>[:<question id>]`, tags it
  `variant:<v>` and `app`/`eval`, and records `model_profile`, `addons`, `commit`, and `session_id`/`thread_id`
  (LangSmith groups a conversation by `thread_id`). Variants A, B, C and the backend all pass it to the graph.
- `ToolRunner` runs each tool through `traced_tool_call`, a child run of type `tool` (arguments, and the result cut to
  `MAX_TRACED_CHARS`). With tracing off it is a plain call.

**Evals** (`evaluation/langsmith_evals.py`). Datasets `hdb-compass-category-a|c|e` are built from
`evaluation/category_{a,c,e}_queries.json` (example id = uuid5 of dataset name and question id, so uploads are
idempotent; the reference outputs are the file's expected route, tools and ground truth). Evaluators reuse our own
scoring (`route_correct`, `tool_precision/recall/f1`, `ground_truth_success`, `figures_present`, `answerable`).
Experiments are named `hdb-eval-<dataset>-variant-<v>-<suffix>`.

```
python -m evaluation.langsmith_evals --dataset category_a --upload-only
LLM_PROFILE=none python -m evaluation.langsmith_evals --dataset category_a --variant C --limit 3
```
