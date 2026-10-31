"""Dataset building and evaluators of evaluation/langsmith_evals.py, offline (a fake client, fake runs)."""
from types import SimpleNamespace as NS

import pytest

from evaluation import langsmith_evals as L


class FakeClient:
    def __init__(self):
        self.datasets, self.examples, self.updated, self.evaluated = {}, {}, [], None

    def has_dataset(self, dataset_name):
        return dataset_name in self.datasets

    def create_dataset(self, name, description=None):
        self.datasets[name] = NS(id="ds-" + name, name=name)
        return self.datasets[name]

    def read_dataset(self, dataset_name):
        return self.datasets[dataset_name]

    def list_examples(self, dataset_id=None, dataset_name=None, example_ids=None):
        out = [NS(id=i, **{"inputs": e["inputs"], "outputs": e["outputs"]}) for i, e in self.examples.items()]
        return [e for e in out if not example_ids or e.id in example_ids]

    def create_examples(self, dataset_id, examples):
        for e in examples:
            self.examples[e["id"]] = e

    def update_example(self, example_id, **kw):
        self.updated.append(example_id)
        self.examples[example_id] = {**self.examples[example_id], **kw}

    def evaluate(self, target, data, evaluators, **kw):
        self.evaluated = (target, data, evaluators, kw)
        return NS(experiment_name="x")


def test_examples_have_stable_ids_and_references():
    a, b = L.build_examples("category_c"), L.build_examples("category_c")
    assert [e["id"] for e in a] == [e["id"] for e in b] and len({e["id"] for e in a}) == len(a) == 15
    assert a[0]["inputs"]["query"] and "query" not in a[0]["outputs"] and a[0]["outputs"]["ground_truth_facts"]


def test_upload_is_idempotent():
    c = FakeClient()
    first = L.ensure_dataset(c, "category_a")
    second = L.ensure_dataset(c, "category_a")
    assert first["created"] and first["added"] == 8 and second["added"] == 0 and second["updated"] == 8
    assert len(c.examples) == 8 and len(c.datasets) == 1


def test_run_experiment_limit_and_addons(monkeypatch):
    c = FakeClient()
    L.ensure_dataset(c, "category_e")
    seen = {}
    L.run_experiment(c, "category_e", "C", limit=3, target=lambda i: seen.setdefault("addons", __import__("os").environ["HDB_ADDONS"]) and {})
    target, data, evs, kw = c.evaluated
    assert len(data) == 3 and kw["experiment_prefix"] == "hdb-eval-category-e-variant-c" and kw["metadata"]["addons"] == "off"
    assert len(evs) == 4
    assert L.DATASETS["category_a"]["addons"] == "app"


def fake(example_item, **out):
    ex = NS(inputs={"query": example_item["query"], "question_id": example_item["id"]},
            outputs={k: v for k, v in example_item.items() if k != "query"})
    return NS(outputs=out), ex


def test_category_c_clauses_scored():
    item = L.load_items("category_c")[0]
    clauses = " ".join(str(c) for c in item["ground_truth_facts"]["key_clauses"])
    run, ex = fake(item, answer="nothing relevant", variant="C", active_subagents=["policy_rag"], tool_calls=[
        {"agent": "policy_rag", "tool": "retrieve_hdb_policy_knowledge", "ok": True}])
    res = {r["key"]: r["score"] for r in L.ground_truth(run, ex)["results"]}
    assert res["ground_truth_success"] == 0.0 and res["clause_coverage"] < 1.0
    assert L.route_correct(run, ex)["score"] == 1.0
    assert {r["key"]: r["score"] for r in L.tool_f1(run, ex)["results"]}["tool_f1"] == 1.0


def test_category_a_figures():
    item = L.load_items("category_a")[3]   # two groups
    g = item["ground_truth_facts"]["groups"]
    text = " ".join(f"${x['median_price']:,.0f} from {x['count']:,} sales" for x in g)
    run, ex = fake(item, answer=text, variant="C", active_subagents=["market_sql"], tool_calls=[])
    res = {r["key"]: r["score"] for r in L.ground_truth(run, ex)["results"]}
    assert res["figures_present"] == 1.0 and res["ground_truth_success"] == 1.0
    run2, _ = fake(item, answer="prices are high", variant="C", active_subagents=[], tool_calls=[])
    assert {r["key"]: r["score"] for r in L.ground_truth(run2, ex)["results"]}["figures_present"] == 0.0


def test_answerable_flags_menu_error_and_empty():
    item = L.load_items("category_a")[0]
    ok, ex = fake(item, answer="Median is $500,000", errors=[], tool_calls=[{"ok": True}], clarification_kind=None)
    assert L.answerable(ok, ex)["score"] == 1.0
    for bad in (dict(answer="", errors=[]), dict(answer="x", clarification_kind="menu"), dict(answer="x", errors=["boom"]),
                dict(answer="x", tool_calls=[{"ok": False}])):
        run, ex = fake(item, **bad)
        assert L.answerable(run, ex)["score"] == 0.0


def test_check_credentials_refuses(monkeypatch, capsys):
    from agents import tracing
    monkeypatch.setattr(tracing, "preflight", lambda *a, **k: {"ok": False, **{"x": 1}})
    monkeypatch.setattr(tracing, "format_check", lambda pf: "DIAGNOSIS")
    with pytest.raises(SystemExit):
        L.check_credentials()
    assert "DIAGNOSIS" in capsys.readouterr().out
