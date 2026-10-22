"""Regenerate docs/architecture_graphs.md from the compiled graphs (Member 4)."""
import os, sys
from pathlib import Path
os.environ.setdefault("LLM_PROFILE", "none")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents import arch_1_monolith as A, arch_2_coarse as B, arch_3_specialist_5agent as C
out = ["# HDB-Compass orchestration graphs (Member 4)\n", "Generated with `graph.get_graph().draw_mermaid()`.\n"]
for title, mod in [("Variant C — 5-agent specialist graph + runtime auditor", C), ("Variant B — coarse 2-agent split", B), ("Variant A — monolithic ReAct agent", A)]:
    out += [f"\n## {title}\n", "```mermaid", mod.graph.get_graph().draw_mermaid().strip(), "```"]
Path(__file__).resolve().parent.joinpath("architecture_graphs.md").write_text("\n".join(out) + "\n")
print("ok")
