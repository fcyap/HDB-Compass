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


def test_a_recovered_schema_error_is_not_a_lost_model_but_a_silent_model_is():
    from evaluation.run_category_e_prelim import lost_model
    assert not lost_model({"variant": "C", "errors": ["extract_constraints[json_schema] failed: BadRequestError: 400"],
                           "model_constraints": {"towns": ["X"]}})
    assert lost_model({"variant": "A", "errors": ["monolith: LLM loop failed: ConflictError: 409"],
                       "tool_call_log": [{"source": "code"}]})


def test_a_failed_model_reading_scores_zero_not_missing():
    acc = M.model_reading_accuracy({"stated_constraints": {"towns": ["X"]}, "financial_inputs": {"grants": {"a": 1}}},
                                   {"errors": ["extract_constraints[json_mode] failed: x"]})
    assert acc["search_correct"] == 0 and acc["reading_failed"]
    assert not M.asked_for_missing("More information needed — cash savings Please confirm with HDB", ["cash_savings"])


def test_advice_to_check_with_hdb_is_not_asking_but_a_real_question_is():
    assert not M.asked_for_missing("- **Confirm your loan eligibility** with HDB or a bank for final approval",
                                   ["loan_eligibility_confirmed"])
    assert not M.asked_for_missing("Cash savings not provided; please check with HDB for the latest rules.", ["cash_savings"])
    assert M.asked_for_missing("Could you confirm your loan eligibility? Please confirm with HDB too.",
                               ["loan_eligibility_confirmed"])


# --- Category C policy scoring, uncertainty and cost (31 Oct) ---------------------------------------------------------

CAT_C = json.loads((Path(__file__).resolve().parent.parent / "evaluation" / "category_c_queries.json").read_text())


def test_clause_scorer_follows_member_3s_rule():
    assert M.clause_in_answer(["clawback", "6%", "9%"], "Subsidy **Clawback**: 6% to 9%")      # every word, any order
    assert not M.clause_in_answer(["clawback", "6%", "9%"], "Clawback: 6% to 8%")
    assert M.clause_in_answer("aged 35 and above", "Singles AGED 35 AND ABOVE may apply")     # whole, case ignored
    # a sentence: 3 of its 4 words longer than two letters is enough (75%), 2 of 4 is not
    assert M.clause_in_answer("applies only to the first resale", "it applies to the first sale only")
    assert not M.clause_in_answer("applies only to the first resale", "it applies to every sale")
    assert not M.clause_in_answer(["whole-flat", "rental"], "the whole flat cannot be rented")  # wording, not meaning


def test_clause_scorer_agrees_with_member_3s_function():
    ablation = pytest.importorskip("evaluation.run_rag_ablation")
    answers = ["", "Plus flats have a 10-year MOP and a **3% to 6% subsidy recovery** on the first resale.",
               "Whole-flat rental is prohibited. Clawback of 6% to 9% applies.",
               "Singles aged 35 and above may buy 2-room Flexi BTO flats in any estate; income ceiling $7,000/month."]
    for item in CAT_C:
        for clause in item["ground_truth_facts"]["key_clauses"]:
            answers.append(clause if isinstance(clause, str) else " ".join(clause))
    for item in CAT_C:
        for clause in item["ground_truth_facts"]["key_clauses"]:
            for a in answers:
                assert M.clause_in_answer(clause, a) == ablation.is_clause_in_passage(clause, ablation.clean_text(a)), \
                    (item["id"], clause, a)


def test_policy_answer_is_fully_right_only_with_every_clause():
    item = next(i for i in CAT_C if i["id"] == "C1-03")
    all_in = {"variant": "C", "draft_response": "Plus flats carry a 3% to 6% subsidy recovery, which applies only to the "
                                                "first resale."}
    s = M.score_item(item, all_in)
    gt = s["ground_truth"]
    assert gt["kind"] == "policy" and gt["success"] and gt["clauses_found"] == gt["clauses_total"] == 2
    assert s["task_success"] and s["category"] == "C"
    assert gt["known_wrong_clauses"] == 1                      # the pass rests on the out-of-date Plus figure
    half = M.score_item(item, {"variant": "C", "draft_response": "Plus flats carry a 3% to 6% subsidy recovery."})
    assert not half["task_success"] and half["ground_truth"]["clause_coverage"] == 0.5
    assert half["ground_truth"]["clauses_missing"] == ["applies only to the first resale"]


def test_known_wrong_clauses_are_listed_from_the_dataset_as_it_stands():
    c101 = next(i for i in CAT_C if i["id"] == "C1-01")
    wrong = [w["clause"] for w in M.known_wrong_clauses(c101)]
    assert ["clawback", "3%", "6%"] in wrong and ["clawback", "6%", "9%"] in wrong
    assert M.known_wrong_clauses(next(i for i in CAT_C if i["id"] == "C2-02")) == []
    # once Member 3 corrects a clause it no longer matches the list, so it is no longer flagged
    fixed = {"id": "C1-03", "ground_truth_facts": {"key_clauses": ["6% to 8% subsidy recovery"]}}
    assert M.known_wrong_clauses(fixed) == []


def test_wilson_interval_and_its_label():
    lo, hi = M.wilson_interval(8, 10)
    assert round(lo, 3) == 0.490 and round(hi, 3) == 0.943                # the textbook example
    assert M.wilson_interval(0, 15)[0] == 0.0 and round(M.wilson_interval(0, 15)[1], 3) == 0.204
    assert round(M.wilson_interval(15, 15)[0], 3) == 0.796 and M.wilson_interval(15, 15)[1] == 1.0
    assert M.wilson_interval(0, 0) is None and M.rate_with_ci(0, 0) == "–"
    assert M.rate_with_ci(15, 15) == "15/15 (100%, 95% CI 80%–100%)"
    assert M.rate_with_ci(9, 15, short=True) == "9/15 (36%–80%)"


def test_exact_mcnemar():
    assert M.mcnemar_exact(0, 0) == 1.0
    assert M.mcnemar_exact(2, 0) == 0.5 == M.mcnemar_exact(0, 2)
    assert M.mcnemar_exact(15, 0) == pytest.approx(2 / 2 ** 15)
    assert M.mcnemar_exact(3, 3) == 1.0


def _row(variant, qid, right, sub=None):
    return {"variant": variant, "id": qid, "sub_use_case": sub or qid[:2], "status": "ok", "latency_s": 1.0,
            "scores": {"task_success": right, "routing": {"applicable": variant != "A", "exact": True},
                       "tool_selection": {"f1": 1.0, "expected": [], "predicted": []}, "parameter_em": {"rate": None},
                       "csr": {"rate": None}}}


def test_paired_table_counts_each_question_once_per_pair():
    from evaluation import analyze_category_e as AN
    rows = [_row("A", "E1-01", True), _row("C", "E1-01", True),
            _row("A", "E1-02", False), _row("C", "E1-02", True),
            _row("A", "E2-01", True), _row("C", "E2-01", False),
            _row("A", "E2-02", False), _row("C", "E2-02", False),
            _row("B", "E1-01", False), _row("C", "C1-01", True), _row("B", "C1-01", True)]
    md = AN.paired_table(rows)
    assert "| E | A vs C | 4 | 1 | 1 | 1 | 1 | 1.000 |" in md
    assert "| E | B vs C | 1 | 0 | 1 | 0 | 0 | 1.000 |" in md
    assert "| C | B vs C | 1 | 1 | 0 | 0 | 0 | 1.000 |" in md
    assert "| E1-02 | ✗ | – | ✓ | C only | – |" in md and "| E2-01 | ✓ | – | ✗ | A only | – |" in md
    assert "Needs Variant C" in AN.paired_table([_row("A", "E1-01", True)])


def test_category_c_caveat_names_the_wrong_clauses_and_who_passed():
    from evaluation import analyze_category_e as AN
    passed = _row("C", "C1-03", True)
    passed["scores"]["ground_truth"] = {"kind": "policy", "success": True}
    text = AN.known_wrong_caveat([passed, _row("A", "C1-03", False)])
    assert "Caveat" in text and "3% to 6% subsidy recovery" in text and "6% to 8%" in text
    assert "Variants that pass this question: C." in text
    assert AN.known_wrong_caveat([_row("C", "E1-01", True)]) == ""


def test_aggregates_split_by_category_and_keep_block_rates_to_block_questions():
    block = {"id": "E1-01", "variant": "C", "task_success": True, "ground_truth": {"blocks_exact": True, "success": True},
             "tool_selection": {"f1": 1.0}, "parameter_em": {"rate": 1.0}, "csr": {"rate": 1.0},
             "routing": {"applicable": True, "exact": True}, "n_tool_calls": 3}
    policy = {"id": "C1-01", "variant": "C", "category": "C", "task_success": False,
              "ground_truth": {"kind": "policy", "success": False, "clause_coverage": 0.5},
              "tool_selection": {"f1": 1.0}, "parameter_em": {"rate": 0.0}, "csr": {"rate": None},
              "routing": {"applicable": True, "exact": True}, "n_tool_calls": 1}
    agg = M.aggregate([block, policy])["C"]
    assert agg["blocks_exact_rate"] == 1.0 and agg["policy_clause_coverage"] == 0.5     # the policy row is no block miss
    assert agg["task_success"] == 1 and agg["task_success_ci95"] == M.wilson_interval(1, 2)
    by = M.aggregate_by_category([block, policy])["C"]                   # the block row has no category: read the id
    assert set(by) == {"C", "E"} and by["E"]["task_success"] == 1 and by["C"]["n"] == 1


def test_compare_runs_breaks_a_mixed_file_down_by_category():
    from evaluation import compare_runs as CR
    e = _row("C", "E1-01", True)
    e["scores"]["ground_truth"] = {"success": True, "blocks_exact": True}
    c = _row("C", "C1-01", False)
    c["scores"]["ground_truth"] = {"kind": "policy", "success": False}
    lines = CR.table({"run1": [e, c]})
    assert "| C | C | fully right | 0/1 (0%–79%) | 0% | 0%–0% |" in lines
    assert "| C | E | fully right | 1/1 (21%–100%) | 100% | 100%–100% |" in lines
    assert not any("| C | C | exact blocks" in line for line in lines)      # blocks do not apply to policy questions


def test_cost_proxy_adds_the_follow_up_turn_and_says_it_is_a_proxy():
    from evaluation.run_category_e_prelim import cost_proxy, dataset_path, CATEGORY_C
    rows = [{"latency_s": 2.0, "tokens": {"total_tokens": 100},
             "follow_up": {"status": "ok", "latency_s": 1.0, "tokens": {"total_tokens": 50}}},
            {"latency_s": 4.0, "tokens": {"total_tokens": 300}},
            {"latency_s": 1.0}]                                             # recorded before tokens were stored
    cp = cost_proxy(rows)
    assert cp["total_tokens"] == 450 and cp["median_tokens"] == 225
    assert cp["total_latency_s"] == 8.0 and cp["median_latency_s"] == 3.0
    assert "proxy" in cp["basis"] and "no paid API" in cp["basis"]
    assert cost_proxy([])["total_tokens"] is None
    assert dataset_path("category_c") == CATEGORY_C and CATEGORY_C.exists()


def test_variant_c_policy_answers_are_scored_end_to_end():
    from evaluation.run_category_e_prelim import summarise
    item = next(i for i in CAT_C if i["id"] == "C2-02")
    state = C.run(item["query"])
    row = {"variant": "C", "id": item["id"], "sub_use_case": "C2", "status": "ok", "latency_s": 0.1,
           "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "scores": M.score_item(item, state),
           "state_excerpt": {"n_candidates": 0}}
    assert row["scores"]["tool_selection"]["expected"] == ["retrieve_hdb_policy_knowledge"]
    assert row["scores"]["ground_truth"]["kind"] == "policy"
    agg = summarise([row])["C"]
    assert set(agg["by_category"]) == {"C"} and agg["cost_proxy"]["questions"] == 1
