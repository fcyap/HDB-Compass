"""Regenerate docs/architecture_graphs.md from the compiled graphs (Member 4).

Run: LLM_PROFILE=none .venv/bin/python docs/build_graph_diagrams.py
The drawings come from the compiled graphs; the prose around them is in this file, so a regeneration keeps it."""
import os, sys
from pathlib import Path
os.environ.setdefault("LLM_PROFILE", "none")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents import arch_1_monolith as A, arch_2_coarse as B, arch_3_specialist_5agent as C

HEADER = [
    "# HDB-Compass orchestration graphs (Member 4)\n",
    "Generated from the compiled LangGraph objects (`graph.get_graph().draw_mermaid()`). Regenerate with "
    "`LLM_PROFILE=none .venv/bin/python docs/build_graph_diagrams.py`.\n",
    "These are the drawings for the Lesson 12 whiteboard: Variant C is the one to reproduce from memory.\n",
]
VARIANTS = [
    ("Variant C — 5-agent specialist graph + runtime auditor (proposed)", C,
     "Parallel `Send` fan-out from `route` to the four specialists; `intersect_candidates` joins SQL ∩ spatial on block "
     "key; `relax_constraints` runs only on an empty intersection (E2); `runtime_auditor` → `synthesize` is the "
     "self-correction cycle (≤2 retries).\n\n"
     "`input_guardrail` and `runtime_auditor` are Member 5's delivered nodes from `agents/guardrails.py` "
     "(`docs/INTERFACES.md` section 3). The input guardrail is rule-based with no model: it refuses SQL and prompt "
     "injection, identifiers, requests for guaranteed or speculative profit, and out-of-scope asks, and routes "
     "straight to `finalize` with a fixed reply. The runtime auditor checks that the model's own sentences "
     "(`model_opening`) only state figures found in the tool results and say nothing speculative or stigmatising, "
     "then asks an LLM judge when a model is configured; a rejection loops back to `synthesize`."),
    ("Variant B — coarse 2-agent split (baseline)", B,
     "One combined Data/Spatial/Finance agent (11 core tools) and one Policy agent (1 tool); no intersection node, "
     "no auditor."),
    ("Variant A — monolithic ReAct agent (baseline)", A,
     "One agent, the 12 core tools, no supervisor, no auditor."),
]
out = HEADER[:]
for title, mod, note in VARIANTS:
    out += [f"\n## {title}\n", note + "\n", "```mermaid", mod.graph.get_graph().draw_mermaid().strip(), "```"]
Path(__file__).resolve().parent.joinpath("architecture_graphs.md").write_text("\n".join(out) + "\n")
print("ok")
