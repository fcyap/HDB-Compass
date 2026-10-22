# HDB-Compass orchestration graphs (Member 4)

Generated from the compiled LangGraph objects (`graph.get_graph().draw_mermaid()`). Regenerate with `LLM_PROFILE=none .venv/bin/python docs/build_graph_diagrams.py`.

These are the drawings for the Lesson 12 whiteboard: Variant C is the one to reproduce from memory.


## Variant C — 5-agent specialist graph + runtime auditor (proposed)

Parallel `Send` fan-out from `route` to the four specialists; `intersect_candidates` joins SQL ∩ spatial on block key; `relax_constraints` runs only on an empty intersection (E2); `runtime_auditor` → `synthesize` is the self-correction cycle (≤2 retries).

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	input_guardrail(input_guardrail)
	extract_constraints(extract_constraints)
	route(route)
	market_sql_agent(market_sql_agent)
	spatial_agent(spatial_agent)
	financial_agent(financial_agent)
	policy_rag_agent(policy_rag_agent)
	intersect_candidates(intersect_candidates)
	relax_constraints(relax_constraints)
	synthesize(synthesize)
	runtime_auditor(runtime_auditor)
	finalize(finalize)
	__end__([<p>__end__</p>]):::last
	__start__ --> input_guardrail;
	extract_constraints --> route;
	financial_agent --> intersect_candidates;
	input_guardrail -.-> extract_constraints;
	input_guardrail -.-> finalize;
	intersect_candidates -.-> relax_constraints;
	intersect_candidates -.-> synthesize;
	market_sql_agent --> intersect_candidates;
	policy_rag_agent --> intersect_candidates;
	relax_constraints --> synthesize;
	route -.-> financial_agent;
	route -.-> market_sql_agent;
	route -.-> policy_rag_agent;
	route -.-> spatial_agent;
	runtime_auditor -.-> finalize;
	runtime_auditor -.-> synthesize;
	spatial_agent --> intersect_candidates;
	synthesize --> runtime_auditor;
	finalize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Variant B — coarse 2-agent split (baseline)

One combined Data/Spatial/Finance agent (10 tools) and one Policy agent (1 tool); no intersection node, no auditor.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	extract_constraints(extract_constraints)
	route_coarse(route_coarse)
	data_agent(data_agent)
	policy_agent(policy_agent)
	synthesize(synthesize)
	finalize(finalize)
	__end__([<p>__end__</p>]):::last
	__start__ --> extract_constraints;
	data_agent --> synthesize;
	extract_constraints --> route_coarse;
	policy_agent --> synthesize;
	route_coarse -.-> data_agent;
	route_coarse -.-> policy_agent;
	synthesize --> finalize;
	finalize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Variant A — monolithic ReAct agent (baseline)

One agent, all 11 tools, no supervisor, no auditor.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	monolith_agent(monolith_agent)
	finalize(finalize)
	__end__([<p>__end__</p>]):::last
	__start__ --> monolith_agent;
	monolith_agent --> finalize;
	finalize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
