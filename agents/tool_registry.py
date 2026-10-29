"""
Tool registry (agents/tool_registry.py). Owned by Member 4.

Resolves each registered tool to a Python callable — real module or
mock — according to config/experiment_config.yaml -> tools.use_mock, and wraps
every call so that:
  * arguments are validated against agents/schemas.TOOL_INPUT_MODEL (strict
    boundary: a malformed LLM tool call never reaches a real tool);
  * a ToolCallRecord is produced for the eval metrics (Tool F1, Param EM);
  * exceptions become structured errors instead of crashing the graph.

Variants A/B/C and the supervisor import from here only — never from
mcp_server/* directly — so flipping mock->real is a config change.
"""
from __future__ import annotations

import importlib
import inspect
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.tools import StructuredTool
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm import load_config
from agents.schemas import TOOL_INPUT_MODEL, TOOL_OWNER, ToolCallRecord

# Where the REAL implementation of each tool lives (module, attribute).
REAL_IMPL: Dict[str, Tuple[str, str]] = {
    "query_resale_blocks_by_criteria": ("mcp_server.market_sql_tools", "query_resale_blocks_by_criteria"),
    "compute_town_price_timeseries": ("mcp_server.market_sql_tools", "compute_town_price_timeseries"),
    "query_comparable_resale_transactions": ("mcp_server.comparable_resale_tools", "query_comparable_resale_transactions"),
    "compare_bto_launch_vs_resale": ("mcp_server.market_sql_tools", "compare_bto_launch_vs_resale"),
    "filter_blocks_by_amenity_proximity": ("mcp_server.spatial_onemap_tools", "filter_blocks_by_amenity_proximity"),
    "verify_primary_school_distance": ("mcp_server.spatial_onemap_tools", "verify_primary_school_distance"),
    "compute_dual_transit_commute": ("mcp_server.spatial_onemap_tools", "compute_dual_transit_commute"),
    "sync_interactive_map_ui": ("mcp_server.spatial_onemap_tools", "sync_interactive_map_ui"),
    "calculate_cpf_housing_grants": ("mcp_server.financial_calc_tools", "calculate_cpf_housing_grants"),
    "calculate_maximum_affordability": ("mcp_server.financial_calc_tools", "calculate_maximum_affordability"),
    "calculate_bsd_and_upfront_outlay": ("mcp_server.financial_calc_tools", "calculate_bsd_and_upfront_outlay"),
    "retrieve_hdb_policy_knowledge": ("mcp_server.policy_rag_tools", "retrieve_hdb_policy_knowledge"),
}


def _unwrap(fn: Any) -> Callable[..., Any]:
    """FastMCP's @mcp.tool() may wrap the function; get the plain callable back."""
    for attr in ("fn", "func", "__wrapped__"):
        inner = getattr(fn, attr, None)
        if callable(inner):
            return inner
    return fn


def resolve_callable(tool_name: str) -> Callable[..., Any]:
    mode = load_config()["tools"]["use_mock"].get(tool_name, "mock")
    if mode == "real":
        module_name, attr = REAL_IMPL[tool_name]
        return _unwrap(getattr(importlib.import_module(module_name), attr))
    from mcp_server.mock_tools import MOCK_TOOLS
    return _unwrap(MOCK_TOOLS[tool_name])


class ToolRunner:
    """Callable wrapper that validates, times and logs one tool."""

    def __init__(self, tool_name: str):
        self.name = tool_name
        self.agent = TOOL_OWNER[tool_name]
        self.input_model = TOOL_INPUT_MODEL[tool_name]
        self.fn = resolve_callable(tool_name)
        self.is_mock = load_config()["tools"]["use_mock"].get(tool_name, "mock") != "real"
        self.log: List[ToolCallRecord] = []
        self.source = "code"   # set to "model" by the tool loop for a call the model chose (ToolCallRecord.source)

    def __call__(self, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            validated = self.input_model(**kwargs)
        except ValidationError as e:
            rec = ToolCallRecord(agent=self.agent, tool=self.name, arguments=kwargs, ok=False,
                                 error=f"invalid arguments: {e.errors()[:3]}", latency_ms=0.0, source=self.source)
            self.log.append(rec)
            return {"error": rec.error, "tool": self.name}
        try:
            arguments = validated.model_dump()
            if self.is_mock:
                # Legacy benchmark fixtures predate the expanded real-tool contracts.
                parameters = inspect.signature(self.fn).parameters
                arguments = {key: value for key, value in arguments.items()
                             if key in parameters and value is not None}
            result = self.fn(**arguments)
            ok, err = True, None
        except Exception as e:  # noqa: BLE001 — surface tool failures to the LLM, don't crash the graph
            result, ok, err = {"error": str(e), "tool": self.name}, False, str(e)
        self.log.append(ToolCallRecord(agent=self.agent, tool=self.name, arguments=validated.model_dump(),
                                       ok=ok, error=err, latency_ms=(time.perf_counter() - t0) * 1000,
                                       source=self.source))
        return result

    def as_langchain_tool(self) -> StructuredTool:
        """Expose to LangGraph/LangChain agents with the Pydantic schema as the tool's args schema."""
        doc = (getattr(self.fn, "__doc__", None) or self.name).strip()
        return StructuredTool.from_function(
            func=self.__call__, name=self.name, description=doc, args_schema=self.input_model)


def build_runners(tool_names: List[str] | None = None) -> Dict[str, ToolRunner]:
    names = tool_names or list(TOOL_OWNER)
    return {n: ToolRunner(n) for n in names}


def tools_for_agent(agent: str) -> List[str]:
    return [t for t, a in TOOL_OWNER.items() if a == agent]


if __name__ == "__main__":
    runners = build_runners()
    for n, r in runners.items():
        mode = load_config()["tools"]["use_mock"][n]
        print(f"{n:40s} agent={r.agent:11s} {mode}")
