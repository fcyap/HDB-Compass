"""One tool-calling request through the active LLM profile; asserts a well-formed tool call.
    LLM_PROFILE=vllm .venv/bin/python cluster/smoke_test.py     (or LLM_PROFILE=ollama)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.llm import get_llm, active_profile
from agents.tool_registry import ToolRunner
from agents.supervisor import sys_msg
from langchain_core.messages import HumanMessage

p = active_profile(); print(f"profile={p['name']} model={p['model']} base_url={p['base_url']}")
llm = get_llm()
if llm is None:
    sys.exit("LLM_PROFILE=none has no model to test")
tool = ToolRunner("calculate_maximum_affordability").as_langchain_tool()
t0 = time.time()
ai = llm.bind_tools([tool]).invoke([sys_msg("You are a housing finance assistant. Use the tool; never compute yourself."),
    HumanMessage(content="Our household income is $8,500 a month, we have $120,000 in CPF OA and $20,000 cash. What is our maximum budget with an HDB loan?")])
dt = time.time() - t0
calls = getattr(ai, "tool_calls", []) or []
print(f"latency={dt:.1f}s tool_calls={[(c['name'], c['args']) for c in calls]}")
assert calls and calls[0]["name"] == "calculate_maximum_affordability", "model did not call the tool"
assert float(calls[0]["args"]["monthly_income"]) == 8500, calls[0]["args"]
print("OK: well-formed tool call")
