"""
Formal Ragas Evaluation for HDB-Compass Regulatory Policy RAG Agent (Agent 5).
Location: evaluation/evaluate_ragas.py
Proposal Section 6.2 (Experiment 2, Page 11) & Section 6.3 (Pages 11-12).

Evaluates the Policy RAG pipeline using the official `ragas` library metrics:
  - Context Precision (Retrieval): Are relevant policy passages ranked high?
  - Context Recall (Retrieval): Did retrieval capture all ground-truth facts?
  - Faithfulness (Generation): Is the generated answer strictly grounded in retrieved passages?
  - Answer Relevancy (Generation): Does the response directly answer the user's regulatory query?

Usage:
  # Quick test on first 3 queries using default RAG-3 (production pipeline):
  python evaluation/evaluate_ragas.py --limit 3

  # Full evaluation on all 15 Category C Golden Queries:
  python evaluation/evaluate_ragas.py --pipeline RAG-3

  # Compare all 3 retrieval pipelines (RAG-1, RAG-2, RAG-3) with Ragas:
  python evaluation/evaluate_ragas.py --pipeline all --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
import yaml

# Ragas imports
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
import ragas
from langchain_core.embeddings import Embeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

from agents.policy_agent import PolicyRagAgent
from mcp_server.policy_rag_tools import execute_rag_pipeline, _get_embed_model

CATEGORY_C_FILE = PROJECT_ROOT / "evaluation" / "category_c_queries.json"
FACT_SHEET_FILE = PROJECT_ROOT / "evaluation" / "policy_fact_sheet.yaml"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "evaluation" / "ragas_evaluation_results.json"


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Wraps local sentence-transformers model as a Langchain Embeddings instance for Ragas."""

    def __init__(self, model: Any):
        self._model = model
        self.model = "all-MiniLM-L6-v2"  # String name expected by Ragas telemetry

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [list(map(float, vec)) for vec in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode(text, show_progress_bar=False)
        return list(map(float, embedding))


def get_ragas_llm_and_embeddings() -> Tuple[Any, Any]:
    """Initializes LLM (via OpenAI-compatible API) and local embeddings for Ragas."""
    from agents.llm import active_profile
    profile = active_profile()
    active_model = profile.get("model", "gemini-3.5-flash-lite")

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )
    if not api_key:
        raise ValueError("No LLM API key found in .env (expected GEMINI_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY)")

    # Prefer Google Gemini OpenAI-compatible endpoint
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model_name = os.getenv("RAGAS_MODEL", active_model)
    elif os.getenv("GROQ_API_KEY"):
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model_name = os.getenv("RAGAS_MODEL", "llama-3.3-70b-versatile")
    else:
        client = OpenAI(api_key=api_key)
        model_name = os.getenv("RAGAS_MODEL", "gpt-4o-mini")

    ragas_llm = llm_factory(model_name, provider="openai", client=client, max_tokens=8192)

    # Local embedding model from policy_rag_tools
    embed_model = _get_embed_model()
    ragas_emb = LangchainEmbeddingsWrapper(LocalSentenceTransformerEmbeddings(embed_model))

    return ragas_llm, ragas_emb


def load_ground_truth_references() -> Dict[str, str]:
    """Reconstructs authoritative ground-truth reference texts from policy_fact_sheet.yaml and category_c_queries.json."""
    facts_map: Dict[str, str] = {}
    if FACT_SHEET_FILE.exists():
        with open(FACT_SHEET_FILE, "r", encoding="utf-8") as f:
            sheet = yaml.safe_load(f) or {}
            for item in sheet.get("facts", []):
                fid = item.get("id")
                fact = item.get("fact", "")
                if fid and fact:
                    facts_map[fid] = fact

    references: Dict[str, str] = {}
    if CATEGORY_C_FILE.exists():
        with open(CATEGORY_C_FILE, "r", encoding="utf-8") as f:
            cases = json.load(f)
            for c in cases:
                qid = c.get("id")
                gt = c.get("ground_truth_facts", {})
                fact_ids = gt.get("fact_ids", [])
                
                # Combine matching facts from fact sheet
                combined_facts = [facts_map[fid] for fid in fact_ids if fid in facts_map]
                if combined_facts:
                    ref_text = " ".join(combined_facts)
                else:
                    # Fallback to key clauses or legacy ground truth
                    clauses = gt.get("key_clauses", [])
                    flattened = []
                    for cl in clauses:
                        if isinstance(cl, list):
                            flattened.append(" ".join(cl))
                        else:
                            flattened.append(str(cl))
                    ref_text = "; ".join(flattened) if flattened else "Official HDB policy applies."
                references[qid] = ref_text

    return references


def run_pipeline_for_evaluation(
    pipeline: str,
    limit: Optional[int] = None,
) -> Tuple[Dataset, List[Dict[str, Any]]]:
    """Runs the PolicyRagAgent on Category C queries to collect question, answer, contexts, and references."""
    with open(CATEGORY_C_FILE, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    if limit:
        queries_data = queries_data[:limit]

    references_map = load_ground_truth_references()
    agent = PolicyRagAgent(pipeline=pipeline)

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    metadata = []

    print(f"\n================================================================================")
    print(f" Executing Policy RAG Pipeline: {pipeline} on {len(queries_data)} queries")
    print(f"================================================================================")

    for idx, q_item in enumerate(queries_data, start=1):
        qid = q_item["id"]
        query = q_item["query"]
        reference = references_map.get(qid, "Official HDB regulatory criteria apply.")

        print(f"[{idx}/{len(queries_data)}] ({qid}) {query[:65]}...")
        t0 = time.monotonic()
        result = agent.run(query=query, use_llm=True)
        latency = time.monotonic() - t0

        generated_answer = result.get("answer", "")
        passages = result.get("passages", [])
        contexts = [f"[{p.get('section', '')}] {p.get('text', '')}" for p in passages]

        questions.append(query)
        answers.append(generated_answer)
        contexts_list.append(contexts if contexts else ["No context retrieved."])
        ground_truths.append(reference)

        metadata.append({
            "id": qid,
            "query": query,
            "pipeline": pipeline,
            "latency_s": round(latency, 2),
            "passages_count": len(passages),
            "reference": reference,
            "generated_answer": generated_answer,
            "retrieved_contexts": contexts,
        })

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    return dataset, metadata


def evaluate_pipeline_with_ragas(
    pipeline: str,
    ragas_llm: Any,
    ragas_emb: Any,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs evaluation using ragas library metrics."""
    dataset, meta_records = run_pipeline_for_evaluation(pipeline, limit=limit)

    print(f"\nEvaluating with Ragas metrics (Faithfulness, ContextRecall, ContextPrecision, AnswerRelevancy)...")

    # Set metric instances
    metrics = [
        faithfulness,
        context_recall,
        context_precision,
        answer_relevancy,
    ]

    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=1, timeout=60, max_retries=3)

    score_result = ragas.evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_config,
    )

    # Convert results to dataframe / dictionary
    df = score_result.to_pandas()
    per_query_results = []
    for i, row in df.iterrows():
        per_query_results.append({
            "id": meta_records[i]["id"],
            "query": meta_records[i]["query"],
            "faithfulness": round(float(row.get("faithfulness", 0.0) or 0.0), 4),
            "context_recall": round(float(row.get("context_recall", 0.0) or 0.0), 4),
            "context_precision": round(float(row.get("context_precision", 0.0) or 0.0), 4),
            "answer_relevancy": round(float(row.get("answer_relevancy", 0.0) or 0.0), 4),
            "latency_s": meta_records[i]["latency_s"],
        })

    avg_faithfulness = round(float(df["faithfulness"].mean() if "faithfulness" in df else 0.0), 4)
    avg_recall = round(float(df["context_recall"].mean() if "context_recall" in df else 0.0), 4)
    avg_precision = round(float(df["context_precision"].mean() if "context_precision" in df else 0.0), 4)
    avg_relevancy = round(float(df["answer_relevancy"].mean() if "answer_relevancy" in df else 0.0), 4)

    summary = {
        "pipeline": pipeline,
        "queries_evaluated": len(per_query_results),
        "mean_context_precision": avg_precision,
        "mean_context_recall": avg_recall,
        "mean_faithfulness": avg_faithfulness,
        "mean_answer_relevancy": avg_relevancy,
        "per_query": per_query_results,
    }

    return summary


def print_summary_card(summaries: List[Dict[str, Any]]):
    """Prints a clean ASCII scorecard table comparing pipelines."""
    print("\n" + "=" * 90)
    print("                     OFFICIAL RAGAS EVALUATION SCORECARD                            ")
    print("=" * 90)
    print(f"{'Pipeline':<18} | {'Context Prec.':<13} | {'Context Recall':<14} | {'Faithfulness':<12} | {'Answer Relevancy':<16}")
    print("-" * 90)
    for s in summaries:
        print(
            f"{s['pipeline']:<18} | "
            f"{s['mean_context_precision']:<13.4f} | "
            f"{s['mean_context_recall']:<14.4f} | "
            f"{s['mean_faithfulness']:<12.4f} | "
            f"{s['mean_answer_relevancy']:<16.4f}"
        )
    print("=" * 90)
    print(" Target Thresholds (Proposal §6.2): Context Recall > 0.88 | Context Precision > 0.88")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HDB-Compass Regulatory Policy RAG Agent using Ragas")
    parser.add_argument(
        "--pipeline",
        choices=["RAG-1", "RAG-2", "RAG-3", "all"],
        default="RAG-3",
        help="Which retrieval pipeline to evaluate: RAG-1 (naive), RAG-2 (hybrid), RAG-3 (production), or all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of Category C queries to evaluate (useful for fast verification).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path to write JSON evaluation results.",
    )
    args = parser.parse_args()

    ragas_llm, ragas_emb = get_ragas_llm_and_embeddings()

    pipelines_to_run = ["RAG-1", "RAG-2", "RAG-3"] if args.pipeline == "all" else [args.pipeline]

    all_summaries = []
    for pl in pipelines_to_run:
        summary = evaluate_pipeline_with_ragas(
            pipeline=pl,
            ragas_llm=ragas_llm,
            ragas_emb=ragas_emb,
            limit=args.limit,
        )
        all_summaries.append(summary)

    print_summary_card(all_summaries)

    # Export to JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"Results successfully saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()

