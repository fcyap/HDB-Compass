import json
import os
import sys
from pathlib import Path

import pytest

os.environ["LLM_PROFILE"] = "none"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import architecture_metrics as M
from agents import arch_3_specialist_5agent as C, arch_1_monolith as A

pytestmark = pytest.mark.usefixtures("mock_domain_tools")

ITEMS = json.loads((Path(__file__).resolve().parent.parent / "evaluation" / "category_e_queries.json").read_text())

# The scoring tests below run on the stand-in tools (mock_domain_tools), so they use their own questions written
# for the stand-in fixture rather than the grounded dataset, whose answers come from the real database.
from agents import supervisor as sv  # noqa: E402

MOCK_E1 = ("We earn $9,000/month, have $90k CPF, want a 4-room flat in Tampines under $680k, "
           "within 400m of an MRT and 500m of a hawker centre, with 65+ years lease. Map the matches.")
MOCK_E2 = "Find a 5-room flat in Tiong Bahru under $600k within 200m of an MRT with 70+ years lease."


def _mock_item(qid, sub, query):
    c = sv.rule_based_extract(query).model_dump()
    agents = sv.decide_routing(c)
    search = {"query_resale_blocks_by_criteria", "filter_blocks_by_amenity_proximity"}
    calls = [{"agent": a, "tool": t, "required_params": args if t in search else {}}
             for a in agents for t, args in sv.deterministic_plan(a, c, query)]
    return {"id": qid, "sub_use_case": sub, "query": query, "expected_agents": agents, "expected_mcp_calls": calls}


def test_f1_and_em_are_perfect_for_variant_c_in_deterministic_mode():
    item = _mock_item("M-E1", "E1", MOCK_E1)
    row = M.score_item(item, C.run(item["query"]))
    assert row["tool_selection"]["f1"] == 1.0 and row["parameter_em"]["rate"] == 1.0
    assert row["routing"]["exact"] and row["csr"]["rate"] == 1.0


def test_e2_item_scores_relaxation():
    item = _mock_item("M-E2", "E2", MOCK_E2)
    row = M.score_item(item, C.run(item["query"]))
    assert row["relaxation_attempted"] and row["csr"]["rate"] is None  # a single step need not unlock anything


def test_aggregate_groups_by_variant():
    item = _mock_item("M-E1", "E1", MOCK_E1)
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


def test_grounded_dataset_has_the_promised_shape():
    e1 = [i for i in ITEMS if i["sub_use_case"] == "E1"]
    e2 = [i for i in ITEMS if i["sub_use_case"] == "E2"]
    assert len(e1) == 8 and len(e2) == 7
    assert all(i["ground_truth_facts"]["matching_block_keys"] for i in e1)
    assert all(not i["ground_truth_facts"]["matching_block_keys"] and i["ground_truth_facts"]["best_single_change"] for i in e2)
    assert all(i["ground_truth_facts"]["data"]["snapshot_id"] for i in ITEMS)


def test_grounded_questions_state_what_is_scored():
    for item in ITEMS:
        c = sv.rule_based_extract(item["query"])
        stated = item["stated_constraints"]
        assert (c.towns, c.flat_types, c.max_budget_sgd, c.min_remaining_lease_years, c.amenity_radii_meters) == (
            stated["towns"], stated["flat_types"], stated["max_budget_sgd"], stated["min_remaining_lease_years"],
            stated["amenity_radii_meters"]), item["id"]
        fin = {c_["tool"]: c_["required_params"] for c_ in item["expected_mcp_calls"] if c_["agent"] == "financial"}
        if item["financial_inputs"]:
            assert fin["calculate_cpf_housing_grants"] == item["financial_inputs"]["grants"]
            assert fin["calculate_maximum_affordability"] == item["financial_inputs"]["affordability"]
        else:
            assert not fin


def test_omitted_facts_are_neither_stated_nor_expected():
    e107 = next(i for i in ITEMS if i["id"] == "E1-07")
    assert "cash" not in e107["query"].lower() and "cash_savings" not in e107["financial_inputs"]["affordability"]
    assert "cash_savings" in e107["ground_truth_facts"]["missing_financial_inputs"]
    e108 = next(i for i in ITEMS if i["id"] == "E1-08")
    assert " aged " not in e108["query"] and "youngest_buyer_age" not in e108["financial_inputs"]["grants"]


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


# ---- ground-truth scoring -------------------------------------------------------------------------
_TRUTH_E1 = {"matching_block_keys": ["228 JURONG EAST ST 21", "266 TOH GUAN RD"],
             "financial": {"calculate_cpf_housing_grants": {"total_grants": 90000.0},
                           "calculate_maximum_affordability": {"max_purchase_price": 777734.95}}}


def _gt(item_extra, draft):
    item = {"id": "t", "sub_use_case": item_extra.pop("sub"), **item_extra}
    return M.ground_truth_scores(item, {"draft_response": draft})


def test_e1_complete_needs_exact_blocks_and_both_numbers():
    good = ("Blocks that meet everything:\n- Block 228 Jurong East St 21\n- Block 266 Toh Guan Rd\n"
            "Your grants total $90,000 and your maximum purchase price is $777,735.")
    assert _gt({"sub": "E1", "ground_truth_facts": _TRUTH_E1}, good)["success"]
    extra_block = good.replace("- Block 266 Toh Guan Rd", "- Block 266 Toh Guan Rd\n- Block 301 Jurong East St 32")
    assert not _gt({"sub": "E1", "ground_truth_facts": _TRUTH_E1}, extra_block)["success"]
    wrong_grant = good.replace("$90,000", "$85,000")
    r = _gt({"sub": "E1", "ground_truth_facts": _TRUTH_E1}, wrong_grant)
    assert r["blocks_exact"] and not r["grants_correct"] and not r["success"]


def test_e1_missing_fact_needs_a_question_and_no_invented_budget():
    truth = {**_TRUTH_E1, "missing_financial_inputs": ["cash_savings"]}
    item = {"sub": "E1", "ground_truth_facts": truth, "omitted_financial_facts": ["cash"]}
    asks = "Block 228 Jurong East St 21 and Block 266 Toh Guan Rd match. How much cash savings do you have?"
    assert _gt(dict(item), asks)["success"]
    guesses = asks + " Your maximum budget is $700,000."
    r = _gt(dict(item), guesses)
    assert r["invented_budget"] and not r["success"]
    silent = "Block 228 Jurong East St 21 and Block 266 Toh Guan Rd match."
    assert not _gt(dict(item), silent)["asked_for_missing"]


def test_e2_needs_no_match_and_the_unlocking_number():
    truth = {"matching_block_keys": [], "best_single_change": {"constraint": "amenity_radii_meters.mrt", "relaxed": 550}}
    ok = "No block meets every constraint. Widening the MRT distance to 550m unlocks 1 block."
    assert _gt({"sub": "E2", "ground_truth_facts": truth}, ok)["success"]
    assert not _gt({"sub": "E2", "ground_truth_facts": truth}, "No block meets every constraint.")["success"]
    assert not _gt({"sub": "E2", "ground_truth_facts": truth}, ok + "\nBlock 12 Kim Tian Rd is close.")["success"]


def test_numbers_are_read_with_commas_decimals_and_k():
    assert M.numbers_in("$777,734.95, $90k and 550m") == [777734.95, 90000.0, 550.0]


def test_bold_block_label_is_read():
    assert M.recommended_block_numbers("### Blocks Meeting All Constraints\n- **Block**: 62\n  - **Price**: 590000") == ["62"]


def test_rows_that_lost_the_model_are_not_results():
    from evaluation.run_category_e_prelim import lost_model
    assert lost_model({"errors": ["extract_constraints[json_schema] failed: OpenAIConnectionError: Connection error."]})
    assert not lost_model({"errors": ["spatial: no valid tool call from model; deterministic plan used"]})
    assert not lost_model({})


def test_any_unanswered_model_call_makes_a_row_not_a_result():
    from evaluation.run_category_e_prelim import lost_model, model_fallbacks
    for err in ("synthesize: LLM failed, template used: APITimeoutError",
                "market_sql: LLM loop failed: InternalServerError: 500",
                "extract_constraints[json_schema] failed: RateLimitError: slow down"):
        assert lost_model({"errors": [err]}), err
    both = {"errors": ["extract_constraints[json_schema] failed: ValueError: x",
                       "extract_constraints[json_mode] failed: ValueError: y",
                       "spatial: no valid tool call from model; deterministic plan used"]}
    assert not lost_model(both)
    fb = model_fallbacks(both)
    assert len(fb) == 2 and any("code-only reading" in f for f in fb)
    assert model_fallbacks({"errors": []}) == []


def test_tool_calls_record_who_chose_them():
    from unittest.mock import patch
    from langchain_core.messages import AIMessage
    from agents import supervisor as sv

    class Model:
        def bind_tools(self, tools):
            return self

        def invoke(self, msgs):
            if any(getattr(m, "type", "") == "tool" for m in msgs):
                return AIMessage(content="done")
            return AIMessage(content="", tool_calls=[{"name": "query_resale_blocks_by_criteria", "id": "1", "args": {
                "towns": ["TOA PAYOH"], "flat_types": ["4 ROOM"], "max_price": 750000, "min_lease_years": 65}}])

    with patch.object(sv, "get_llm", return_value=Model()):
        outputs, log, errors, final = sv.run_react("market_sql", sv.tools_for_agent("market_sql"), "x", "q",
                                                   {"towns": ["TOA PAYOH"]})
    assert [r["source"] for r in log] == ["model"] and not errors
    _, log2, _, _ = sv.run_react("market_sql", sv.tools_for_agent("market_sql"), "x", "q",
                                 {"towns": ["TOA PAYOH"], "flat_types": ["4 ROOM"]})   # no model: the plan runs
    assert log2 and all(r["source"] == "code" for r in log2)
    s = M.score_item({"id": "x", "expected_mcp_calls": [{"tool": "query_resale_blocks_by_criteria"}]},
                     {"variant": "C", "tool_call_log": log})
    assert s["tool_selection_model_chosen"]["f1"] == 1.0 and s["tool_calls_by_source"]["model"] == 1


# --- regressions found by the 30 Oct adversarial review -------------------------------------------------------------

def test_refused_calls_or_a_model_that_did_nothing_are_not_results():
    from evaluation.run_category_e_prelim import lost_model, model_fallbacks
    assert lost_model({"errors": ["synthesize: LLM failed, template used: NotFoundError"]})
    nothing = {"errors": ["extract_constraints[json_schema] failed: ValueError: x",
                          "extract_constraints[json_mode] failed: ValueError: y",
                          "market_sql: no valid tool call from model; deterministic plan used",
                          "synthesize: LLM failed, template used: SomethingOdd"],
               "tool_call_log": [{"tool": "query_resale_blocks_by_criteria", "source": "code"}]}
    assert lost_model(nothing)
    repaired = {"errors": ["extract_constraints[json_schema] failed after one repair: ValueError: z"]}
    assert not lost_model(repaired) and model_fallbacks(repaired) == []      # json_mode then read it: not a fallback


def test_resume_only_reuses_rows_from_the_same_code_and_settings():
    from evaluation.run_category_e_prelim import same_settings
    now = {"commit": "abc", "code_fingerprint": "f1", "dataset": "category_e_queries.json", "code_written_facts": True}
    assert same_settings(dict(now), now)
    assert not same_settings({**now, "code_fingerprint": "f2"}, now)          # different uncommitted edits
    assert not same_settings(None, now)                                       # rows that recorded nothing


def test_boilerplate_please_does_not_count_as_asking_for_the_missing_fact():
    draft = "- Affordability: more information needed — cash savings\n- Please confirm with HDB, for example with an HFE letter."
    assert not M.asked_for_missing(draft, ["cash_savings"])
    assert M.asked_for_missing(draft + "\nTo finish, please tell me your cash savings.", ["cash_savings"])


def test_model_reading_is_scored_before_code_fills_it():
    item = {"stated_constraints": {"towns": ["TOA PAYOH"], "flat_types": ["4 ROOM"], "max_budget_sgd": 750000,
                                   "min_remaining_lease_years": 65, "amenity_radii_meters": {"mrt": 400}},
            "financial_inputs": {"grants": {"citizenship_pair": "SC_PR"}, "affordability": {"cpf_oa_balance": 80000.0}}}
    reading = {"towns": ["TOA PAYOH"], "flat_types": [], "max_budget_sgd": 750000, "min_remaining_lease_years": 65,
               "amenity_radii_meters": {"mrt": 400, "hawker": 0},
               "financial_inputs": {"grants": {"citizenship_pair": "SC_SC"}, "affordability": {"cpf_oa_balance": 80000.0,
                                                                                            "cash_savings": 80000.0}}}
    acc = M.model_reading_accuracy(item, {"model_constraints": reading})
    assert acc == {"search_correct": 4, "search_total": 5, "facts_correct": 1, "facts_total": 2,
                   "facts_wrong": 1, "facts_invented": 1}
    assert M.model_reading_accuracy(item, {}) is None                        # no model, nothing to score


def test_logs_without_a_source_tag_get_no_model_chosen_score():
    s = M.score_item({"id": "x", "expected_mcp_calls": [{"tool": "query_resale_blocks_by_criteria"}]},
                     {"variant": "C", "tool_call_log": [{"tool": "query_resale_blocks_by_criteria", "agent": "market_sql"}]})
    assert "tool_selection_model_chosen" not in s
