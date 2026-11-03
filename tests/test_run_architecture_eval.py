"""evaluation/run_architecture_eval.py: the plan it hands to the Experiment 1 runner, and the aggregation of saved rows."""
import os

from evaluation import run_architecture_eval as ev


def _row(variant, qid, ok, latency=1.0, retries=0, passed=True, tokens=100):
    return {"variant": variant, "id": qid, "status": "ok", "latency_s": latency, "tokens": {"total_tokens": tokens},
            "scores": {"task_success": ok}, "state": {"auditor_verdict": {"passed": passed, "retry_count": retries,
                                                                           "critique": ""}, "errors": []}}


def test_the_plan_runs_the_variants_then_c_without_the_auditor():
    steps = ev.plan("t")
    assert len(steps) == 2
    (cmd1, env1), (cmd2, env2) = steps
    assert "A,B,C" in cmd1 and cmd1[cmd1.index("--tag") + 1] == "t" and env1 == {}
    assert cmd2[cmd2.index("--variants") + 1] == "C" and cmd2[cmd2.index("--tag") + 1] == "t-noauditor"
    assert env2 == {"HDB_AUDITOR": "off"}
    assert all(c[c.index("--dataset") + 1].endswith("benchmark_hdb_90.json") for c, _ in steps)


def test_the_ablation_can_be_skipped_and_needs_variant_c():
    assert len(ev.plan("t", ablation=False)) == 1
    assert len(ev.plan("t", variants="A,B")) == 1


def test_auditor_stats_count_rejections_and_the_latency_they_cost():
    rows = [_row("C", "E1-01", True, 2.0, retries=0), _row("C", "E1-02", True, 9.0, retries=1),
            _row("C", "E1-03", False, 20.0, retries=3, passed=False), _row("A", "E1-01", False)]
    a = ev.auditor_stats(rows)
    assert a["answers"] == 3 and a["rejected_at_least_once"] == 2 and a["ran_out_of_retries"] == 1
    assert a["rejected_but_correct"] == 1 and a["mean_rewrites"] == round(4 / 3, 3)
    assert a["latency_p50_s_not_rejected"] == 2.0 and a["latency_p50_s_rejected"] == 20.0


def test_paired_ablation_counts_who_is_right_only_with_the_auditor():
    on = [_row("C", "q1", True, 5.0, tokens=300), _row("C", "q2", True, 4.0), _row("C", "q3", False, 3.0)]
    off = [_row("C", "q1", False, 2.0, tokens=100), _row("C", "q2", True, 4.0), _row("C", "q3", False, 3.0)]
    ab = ev.paired_ablation(on, off)
    assert ab["paired_questions"] == 3 and ab["auditor_on_success"] == 2 and ab["auditor_off_success"] == 1
    assert ab["right_only_with_auditor"] == 1 and ab["right_only_without_auditor"] == 0
    assert ab["median_extra_latency_s"] == 0.0 and 0 <= ab["mcnemar_exact_p"] <= 1


def test_safety_by_variant_counts_guardrail_blocks():
    blocked = _row("C", "F1-01", True)
    blocked["state"]["errors"] = ["input_guardrail: speculative_advice"]
    rows = [blocked, _row("A", "F1-01", False), _row("C", "E1-01", True)]
    s = ev.safety_by_variant(rows)
    assert s["C"] == {"questions": 1, "handled": 1, "blocked_by_guardrail": 1}
    assert s["A"] == {"questions": 1, "handled": 0, "blocked_by_guardrail": 0}


def test_the_auditor_switch_turns_the_whole_node_into_a_pass_through(monkeypatch):
    from agents import guardrails as g
    state = {"messages": [], "variant": "C", "draft_response": "x", "model_opening": "This is guaranteed to appreciate."}
    monkeypatch.setenv("HDB_AUDITOR_LLM", "off")
    monkeypatch.delenv("HDB_AUDITOR", raising=False)
    assert g.runtime_auditor_node(state)["auditor_verdict"]["passed"] is False
    monkeypatch.setenv("HDB_AUDITOR", "off")
    v = g.runtime_auditor_node(state)["auditor_verdict"]
    assert v["passed"] is True and "switched off" in v["critique"]
    assert g.auditor_enabled() is False


def test_the_report_has_every_section_for_a_small_set_of_rows():
    rep = {"tag": "t", "rows_auditor_on": 2, "rows_auditor_off": 1, "experiment1": {},
           "safety_category_f": {"C": {"questions": 1, "handled": 1, "blocked_by_guardrail": 1}},
           "auditor": ev.auditor_stats([_row("C", "q1", True, retries=1)]),
           "ablation": ev.paired_ablation([_row("C", "q1", True)], [_row("C", "q1", False)])}
    md = ev.to_markdown(rep)
    for heading in ("Category F", "What the runtime auditor did", "with and without the auditor"):
        assert heading in md
