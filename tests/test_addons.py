"""Add-on tool packs (agents/addons): every add-on tool is discoverable by the agent that owns it.

For each declared tool: it imports, its owner is a real specialist and lists it, the MCP server registers it, its
"when to use" line is in the owner's prompt, and its example question makes Variant C offer it, plan it and keep its
result as evidence. A tool missing from any of these fails here rather than silently in the app.
"""
from __future__ import annotations

import asyncio
import os

os.environ["LLM_PROFILE"] = "none"

import pytest  # noqa: E402

from agents import addons  # noqa: E402
from agents import arch_1_monolith as A  # noqa: E402
from agents import arch_2_coarse as B  # noqa: E402
from agents import supervisor as sv  # noqa: E402
from agents import tool_registry as tr  # noqa: E402
from agents.schemas import CORE_TOOLS, TOOL_INPUT_MODEL, TOOL_OWNER  # noqa: E402

ADDONS = addons.all_addons()
IDS = [t.name for t in ADDONS]


def test_packs_add_tools_beside_the_twelve_core_tools():
    assert len(CORE_TOOLS) == 12 and not set(CORE_TOOLS) & set(IDS)
    assert set(TOOL_OWNER) == set(CORE_TOOLS) | set(IDS) == set(TOOL_INPUT_MODEL)
    assert len(IDS) == len(set(IDS))


@pytest.mark.parametrize("tool", ADDONS, ids=IDS)
def test_addon_is_registered_where_the_core_looks(tool):
    assert tool.owner in sv.SPECIALISTS and tool.name in tr.tools_for_agent(tool.owner)
    assert tr.REAL_IMPL[tool.name] == tool.impl and tr.tool_mode(tool.name) == "real"
    assert callable(tr.resolve_callable(tool.name))
    assert TOOL_INPUT_MODEL[tool.name] is addons._load(tool.input_model)
    assert addons.output_model(tool.name).model_json_schema()
    assert sv.TASK_TOOLS[tool.name] is tool.offer_when
    assert sv.NOT_CHECKED[tool.name] == tool.not_checked
    assert tool.prompt_hint in sv.AGENT_SYSTEM[tool.owner]
    assert tool.prompt_hint not in sv.CORE_AGENT_SYSTEM[tool.owner]


def test_mcp_server_lists_every_tool():
    from mcp_server.server import mcp
    assert set(asyncio.run(mcp.get_tools())) == set(TOOL_OWNER)


@pytest.mark.parametrize("tool", ADDONS, ids=IDS)
def test_example_question_reaches_the_tool(tool):
    q = tool.example_query
    c = sv.rule_based_extract(q).model_dump()
    assert tool.owner in sv.decide_routing(c), q
    assert tool.offer_when(c) and sv.needed_for_task(tool.name, c, q), q
    planned = [name for name, _ in sv.deterministic_plan(tool.owner, c, q)]
    # resolve_location is offered for any place question; the plans reach places through the other tools
    if tool.name != "resolve_location":
        assert tool.name in planned, (q, planned)
        kept = sv.requested_outputs(tool.owner, {tool.name: {"status": "ok"}}, c, q)
        assert tool.name in kept


def test_a_plain_block_search_is_offered_no_addon_tool():
    q = "Find 4-room flats in Tampines under $600k within 400m of an MRT"
    c = {**sv.rule_based_extract(q).model_dump(), "requested_tools": []}
    for agent in ("market_sql", "spatial"):
        offered = [n for n in tr.tools_for_agent(agent) if sv.needed_for_task(n, c, q)]
        assert not set(offered) & set(IDS), offered


def test_named_bto_launch_is_offered_the_bto_comparison():
    c = sv.rule_based_extract("How does Sembawang Portico compare with resale prices?").model_dump()
    assert c["bto_project_name"] and not c["wants_bto_comparison"]
    assert sv.needed_for_task("compare_bto_launch_vs_resale", c)
    assert "compare_bto_launch_vs_resale" in dict(sv.deterministic_plan("market_sql", c, "Sembawang Portico"))


# --- the offer_in switch ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode, a, b, c, app", [
    ("app", False, False, True, True),
    ("all", True, True, True, True),
    ("off", False, False, False, False),
])
def test_offer_in_decides_which_runs_see_addons(monkeypatch, mode, a, b, c, app):
    monkeypatch.setenv("HDB_ADDONS", mode)
    assert addons.offered_in("A") is a and addons.offered_in("B") is b and addons.offered_in("C") is c
    assert addons.offered_in("C", fast_path=True) is app
    names = addons.visible_tools(list(TOOL_OWNER), "A")
    assert (set(IDS) <= set(names)) is a and set(CORE_TOOLS) <= set(names)


def test_off_reproduces_the_core_plans(monkeypatch):
    monkeypatch.setenv("HDB_ADDONS", "off")
    c = sv.rule_based_extract("How is the HDB resale market doing now?").model_dump()
    assert sv.deterministic_plan("market_sql", c, "How is the HDB resale market doing now?") == []


def test_a_bad_switch_value_is_refused(monkeypatch):
    monkeypatch.setenv("HDB_ADDONS", "maybe")
    with pytest.raises(ValueError, match="HDB_ADDONS"):
        addons.offer_mode()


def test_variants_a_and_b_keep_the_core_tool_list_by_default():
    assert "get_market_snapshot" not in addons.visible_tools(list(TOOL_OWNER), A.VARIANT)
    assert "get_market_snapshot" not in addons.visible_tools(B.DATA_TOOLS, B.VARIANT)


def test_named_bto_launch_runs_no_town_block_search():
    # found by the app smoke test: the launch's town alone started a 100-block Sembawang search
    q = "How does Sembawang Portico compare with resale prices?"
    c = sv.rule_based_extract(q).model_dump()
    assert [n for n, _ in sv.deterministic_plan("market_sql", c, q)] == ["compare_bto_launch_vs_resale"]
