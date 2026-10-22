import json, os, sys
from pathlib import Path
os.environ["LLM_PROFILE"] = "none"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import architecture_metrics as M
from agents import arch_3_specialist_5agent as C, arch_1_monolith as A

ITEMS = json.loads((Path(__file__).resolve().parent.parent / "evaluation" / "category_e_queries.json").read_text())


def test_f1_and_em_are_perfect_for_variant_c_in_deterministic_mode():
    item = next(i for i in ITEMS if i["id"] == "E1-01")
    row = M.score_item(item, C.run(item["query"]))
    assert row["tool_selection"]["f1"] == 1.0 and row["parameter_em"]["rate"] == 1.0
    assert row["routing"]["exact"] and row["csr"]["rate"] == 1.0


def test_e2_item_scores_relaxation():
    item = next(i for i in ITEMS if i["id"] == "E2-01")
    row = M.score_item(item, C.run(item["query"]))
    assert row["relaxation_attempted"] and row["csr"]["rate"] is None  # a single step need not unlock anything


def test_aggregate_groups_by_variant():
    item = ITEMS[0]
    rows = [M.score_item(item, C.run(item["query"])), M.score_item(item, A.run(item["query"]))]
    agg = M.aggregate(rows)
    assert set(agg) == {"A", "C"} and agg["C"]["tool_selection_f1"] == 1.0
    assert agg["A"]["routing_exact_rate"] is None  # not applicable to the monolith


def test_csr_reads_blocks_from_draft_for_variants_without_intersection():
    sql_rows = [{"block": "924", "street": "TAMPINES ST 91", "town": "TAMPINES", "median_resale_price": 598000, "median_remaining_lease_years": 70.1},
                {"block": "201", "street": "TAMPINES ST 21", "town": "TAMPINES", "median_resale_price": 585000, "median_remaining_lease_years": 66.2}]
    spatial = [{"block": "924", "street": "TAMPINES ST 91", "town": "TAMPINES", "distances": {"mrt_dist_m": 298.2, "hawker_dist_m": 443.7}}]
    state = {"variant": "A", "user_constraints": {"towns": ["TAMPINES"], "max_budget_sgd": 680000, "min_remaining_lease_years": 65,
                                                  "amenity_radii_meters": {"mrt": 400, "hawker": 500}},
             "sql_market_results": {"query_resale_blocks_by_criteria": {"blocks": sql_rows}},
             "spatial_block_candidates": spatial, "candidate_blocks_intersection": [],
             "draft_response": "| Block 924 | ... |\n| Block 201 | ... |\n| Block 777 | ... |", "tool_call_log": [], "active_subagents": ["monolith"]}
    row = M.score_item({"id": "x", "expected_mcp_calls": [], "expected_agents": []}, state)
    # 924 satisfies all; 201 has no spatial evidence (fails radius); 777 is fabricated
    assert row["csr"]["recommended"] == 3 and row["csr"]["satisfying"] == 1 and row["fabricated_blocks"] == 1


def test_csr_fails_blocks_recommended_without_price_evidence():
    state = {"variant": "A", "user_constraints": {"towns": ["BUKIT MERAH"], "max_budget_sgd": 600000, "amenity_radii_meters": {"mrt": 200}},
             "spatial_block_candidates": [{"block": "9B", "street": "BOON TIONG RD", "town": "BUKIT MERAH", "distances": {"mrt_dist_m": 150.0}}],
             "candidate_blocks_intersection": [], "draft_response": "Blk 9B Boon Tiong Rd is within 150m of an MRT.",
             "tool_call_log": [], "active_subagents": ["monolith"]}
    row = M.score_item({"id": "x", "expected_mcp_calls": [], "expected_agents": []}, state)
    assert row["csr"]["recommended"] == 1 and row["csr"]["satisfying"] == 0


def test_recommendations_skip_near_miss_sections_and_read_tables():
    draft_c = ("### Matches\n\n#### Blocks Meeting All Criteria:\n1. **Block 924, Tampines St 91**\n2. **Block 927, Tampines St 91**\n\n"
               "### Blocks That Meet Most Criteria (Except Price or Lease)\n- **Block 201, Tampines St 21**\n\n"
               "### Trade-Offs\n- Block 924 is cheaper than Block 927.")
    assert M.recommended_block_numbers(draft_c) == ["924", "927"]
    draft_b = "### Matches\n\n#### Block 201 - Tampines St 21\n- Price: $585,000\n\n#### Block 924 - Tampines St 91\n"
    assert M.recommended_block_numbers(draft_b) == ["201", "924"]
    table = "| Block | Postal | Price |\n|---|---|---|\n| 924 | 520924 | 598,000 |\n| 201 | 520201 | 585,000 |"
    assert M.recommended_block_numbers(table) == ["924", "201"]
    assert M.recommended_block_numbers("Block 924 fits.\nBlock 201 does not meet the MRT limit.") == ["924"]


def test_tool_f1_ignores_the_map_tool_on_both_sides():
    exp = [{"tool": "query_resale_blocks_by_criteria"}, {"tool": "sync_interactive_map_ui"}]
    log = [{"agent": "spatial", "tool": "sync_interactive_map_ui", "ok": True},
           {"agent": "market_sql", "tool": "query_resale_blocks_by_criteria", "ok": True}]
    assert M.tool_selection_prf(exp, log)["f1"] == 1.0


def test_golden_params_hold_only_what_the_user_stated():
    e101 = next(i for i in ITEMS if i["id"] == "E1-01")
    grants = next(c for c in e101["expected_mcp_calls"] if c["tool"] == "calculate_cpf_housing_grants")
    assert grants["required_params"] == {"monthly_income": 9000.0}


def test_free_text_citizenship_is_rejected_at_the_tool_boundary():
    from agents.tool_registry import ToolRunner
    r = ToolRunner("calculate_cpf_housing_grants")
    out = r(flat_category="resale", citizenship_pair="citizen", first_timer_status=True, monthly_income=9000)
    assert "error" in out and r.log[-1].ok is False


def test_monolith_map_comes_from_its_own_map_call_and_invented_ids_count_as_fabricated():
    state = {"variant": "A", "user_constraints": {"towns": ["TAMPINES"], "amenity_radii_meters": {"mrt": 400}},
             "spatial_block_candidates": [{"block": "924", "street": "TAMPINES ST 91", "town": "TAMPINES", "distances": {"mrt_dist_m": 298.2}}],
             "tool_call_log": [{"agent": "monolith", "tool": "sync_interactive_map_ui", "ok": True,
                                "arguments": {"candidate_blocks": ["924", "BLOCK_1", "BISHAN-01"]}}],
             "map_ui_payload": {}, "draft_response": "", "active_subagents": ["monolith"]}
    blocks = M.map_blocks(state)
    assert [b["_fabricated"] for b in blocks] == [False, True, True]
    assert M._block_number("924 TAMPINES ST 91") == "924" and M._block_number("Blk 9B") == "9B"


def test_task_success_rules():
    ok = {"recommended": 2, "satisfying": 2, "rate": 1.0}
    assert M.task_success("E1", ok, 0, False)
    assert not M.task_success("E1", {"recommended": 2, "satisfying": 1, "rate": 0.5}, 0, False)
    assert not M.task_success("E1", {"recommended": 0, "satisfying": 0, "rate": None}, 0, False)
    assert M.task_success("E2", {"recommended": 0, "satisfying": 0, "rate": None}, 0, True)
    assert not M.task_success("E2", {"recommended": 0, "satisfying": 0, "rate": None}, 0, False)


def test_trade_off_advice_is_recognised_in_plain_text():
    assert M.relaxation_offered({"draft_response": "Consider increasing your budget to $660k to unlock two blocks."})
    assert not M.relaxation_offered({"draft_response": "No properties met the lease requirement."})
