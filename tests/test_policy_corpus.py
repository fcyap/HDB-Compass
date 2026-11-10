"""
The HDB policy corpus is built from captured official pages (Member 3; data/policy_sources/).

Checks:
  * every source in sources.yaml has a unique id, a known policy category and a capture method;
  * every verified fact in evaluation/policy_fact_sheet.yaml quotes its captured page word for word;
  * once the rebuilt corpus is installed (data/policy_documents written by extract_sources.py): every document
    names its page, publisher and retrieval date, and none repeats a figure known to be out of date.

Run:
    .venv/bin/pytest tests/test_policy_corpus.py -v
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.policy_sources.fetch_sources import load_sources  # noqa: E402
from mcp_server.policy_rag_tools import POLICY_DOMAINS  # noqa: E402

FACT_SHEET = PROJECT_ROOT / "evaluation" / "policy_fact_sheet.yaml"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "policy_sources" / "extracted"
CORPUS_DIR = PROJECT_ROOT / "data" / "policy_documents"

# figures the legacy corpus stated that the team found out of date (evaluation/architecture_metrics.KNOWN_WRONG_CLAUSES)
STALE_FIGURES = [r"Plus\b[^.\n]{0,80}\b3% to 6%", r"Plus\b[^.\n]{0,80}\b3-6%"]


def normalise(text: str) -> str:
    """Compare words, not Markdown: drop emphasis, escapes, table pipes and HTML tags; collapse whitespace."""
    text = text.replace("\\", "")                 # Markdown escapes (\[1\], \*) carry no words
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*_`|]", " ", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def facts():
    return (yaml.safe_load(FACT_SHEET.read_text(encoding="utf-8")) or {}).get("facts") or []


def document_for(source_id: str):
    """The extracted document of a source (installed corpus first, then the extraction folder)."""
    for folder in (CORPUS_DIR, EXTRACTED_DIR):
        hits = sorted(folder.glob(f"*_{source_id}.md")) if folder.exists() else []
        if hits:
            return hits[0]
    return None


def corpus_is_rebuilt() -> bool:
    docs = list(CORPUS_DIR.glob("*.md"))
    return bool(docs) and all("**Capture Method:**" in d.read_text(encoding="utf-8") for d in docs)


def test_sources_inventory_is_well_formed():
    sources = load_sources()
    assert sources
    for s in sources:
        assert s["capture"] in ("auto", "manual"), s["id"]
        assert s["policy_category"] in POLICY_DOMAINS, s["id"]
        assert s["url"].startswith("https://"), s["id"]
        assert s.get("publisher"), s["id"]


def test_fact_sheet_is_well_formed():
    ids = {s["id"] for s in load_sources()}
    seen = set()
    for f in facts():
        assert f["id"] not in seen, f["id"]
        seen.add(f["id"])
        assert f["source"] in ids, f"{f['id']}: unknown source {f['source']}"
        assert f["policy_category"] in POLICY_DOMAINS, f["id"]
        assert f["status"] in ("verified", "pending_capture"), f["id"]
        if f["status"] == "verified":
            assert f.get("quote") and f.get("fact") and f.get("key_clause"), f["id"]
        else:
            assert f.get("claim_to_check"), f["id"]


@pytest.mark.parametrize("fact", [f for f in facts() if f["status"] == "verified"], ids=lambda f: f["id"])
def test_verified_fact_is_quoted_from_its_page(fact):
    doc = document_for(fact["source"])
    if doc is None:
        pytest.skip(f"{fact['source']} not extracted (run fetch_sources.py and extract_sources.py)")
    text = normalise(doc.read_text(encoding="utf-8"))
    assert normalise(fact["quote"]) in text, f"{fact['id']}: quote not found in {doc.name}"
    for item in fact["key_clause"]:
        assert normalise(item) in text, f"{fact['id']}: key clause {item!r} not in {doc.name}"


def test_installed_corpus_has_provenance_and_no_stale_figures():
    if not corpus_is_rebuilt():
        pytest.skip("the rebuilt corpus is not installed yet (extract_sources.py --install)")
    for doc in sorted(CORPUS_DIR.glob("*.md")):
        text = doc.read_text(encoding="utf-8")
        for field in ("Source URL", "Publisher", "Retrieved", "Policy Category", "Content SHA256"):
            assert re.search(rf"^\*\*{field}:\*\* \S", text, re.M), f"{doc.name}: no {field}"
        for pattern in STALE_FIGURES:
            assert not re.search(pattern, text, re.I), f"{doc.name}: stale figure {pattern}"


def test_category_c_truth_is_on_its_official_pages():
    """Each answerable Category C question's key clauses appear on the official pages its truth names, so a correct
    answer is possible from the corpus (the old truth quoted the legacy documents, some of them wrong)."""
    import json
    from evaluation.run_rag_ablation import clean_text, is_clause_in_passage
    if not corpus_is_rebuilt():
        pytest.skip("the rebuilt corpus is not installed yet")
    pages = {}
    for doc in CORPUS_DIR.glob("*.md"):
        text = doc.read_text(encoding="utf-8")
        pages[text.splitlines()[0].lstrip("# ").strip()] = clean_text(text)
    facts_by_id = {f["id"]: f for f in facts()}
    cases = json.loads((PROJECT_ROOT / "evaluation" / "category_c_queries.json").read_text(encoding="utf-8"))
    for case in cases:
        gt = case["ground_truth_facts"]
        assert all(facts_by_id[f]["status"] == "verified" for f in gt["fact_ids"]), case["id"]
        if not gt["answerable"]:
            continue
        missing_pages = [t for t in gt["target_sections"] if t not in pages]
        assert not missing_pages, f"{case['id']}: no page titled {missing_pages}"
        scope = " ".join(pages[t] for t in gt["target_sections"])
        for clause in gt["key_clauses"]:
            assert is_clause_in_passage(clause, scope), f"{case['id']}: {clause} not on its pages"


def test_query_expansions_use_the_corpus_vocabulary():
    """The expansion terms are added to the question when no model rewrites it; a term the official pages never use
    (the old "15-month wait-out", "3% to 6%") pulls the search away from HDB's actual rules."""
    from mcp_server.policy_rag_tools import ACRONYM_EXPANSIONS
    if not corpus_is_rebuilt():
        pytest.skip("the rebuilt corpus is not installed yet")
    vocabulary = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", " ".join(
        d.read_text(encoding="utf-8").lower() for d in CORPUS_DIR.glob("*.md"))))
    for pattern, terms in ACRONYM_EXPANSIONS.items():
        missing = [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", terms.lower()) if w not in vocabulary]
        assert not missing, f"{pattern}: {missing} not in the corpus"
