"""RAGAS-library scoring for Experiment 2E (evaluation/run_rag_ablation.py): four `ragas` metrics, two judges.

Generation, judged by a model from another family than the generator (Groq openai/gpt-oss-120b; Gemini writes):
  - Faithfulness: ragas splits the answer into statements and asks the judge whether the retrieved contexts support
    each one; score = supported statements / statements.
  - Answer Relevancy: the judge writes a question the answer would answer; ragas embeds it and the user's question and
    takes their cosine similarity (one generated question per answer, strictness=1, to fit the free tier; ragas's
    default is 3). A vague or off-topic answer scores low.
Retrieval, judged against a reference answer built from the fact sheet (evaluation/policy_fact_sheet.yaml); these
grade the retrieved passages, not any model's answer, so a smaller judge with its own quota is used (Groq
openai/gpt-oss-20b):
  - Context Recall: share of the reference's statements the retrieved contexts support.
  - Context Precision: whether the contexts that help answer the reference are ranked above those that don't.
The embeddings are the local all-MiniLM-L6-v2 the policy store already uses, so no embedding API is called.

Compatibility: ragas 0.4.3 imports two Vertex AI classes that langchain-community 0.4 removed. ragas only uses them
in isinstance checks, so empty stand-ins are registered before ragas is imported. Usage telemetry is switched off.
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
from typing import Any, Dict, List, Optional

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


def _patch_langchain_community() -> None:
    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            stub.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules[name] = stub
    import langchain_community.llms as llms
    if not hasattr(llms, "VertexAI"):
        llms.VertexAI = type("VertexAI", (), {})


_patch_langchain_community()

from langchain_core.embeddings import Embeddings  # noqa: E402
from ragas import SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,  # noqa: E402
                           ResponseRelevancy)
from ragas.run_config import RunConfig  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402

# ragas sends each statement check as its own judge call; this bounds one sample's wait on a slow free tier
SAMPLE_TIMEOUT_S = float(os.getenv("RAGAS_SAMPLE_TIMEOUT_S", "180"))


class _MiniLMEmbeddings(Embeddings):
    """The policy store's sentence-transformer, as a LangChain Embeddings object for ragas."""

    def __init__(self) -> None:
        from mcp_server.policy_rag_tools import _get_embed_model
        self.encoder = _get_embed_model()
        self.model = "all-MiniLM-L6-v2"   # ragas reads .model as the model's name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.encoder.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


_EMBEDDINGS: Optional[LangchainEmbeddingsWrapper] = None
_METRICS: Dict[int, Dict[str, Any]] = {}


def _metrics_for(llm: Any, label: str) -> Dict[str, Any]:
    """The four ragas metrics bound to this judge model (built once per model)."""
    global _EMBEDDINGS
    key = id(llm)
    if key not in _METRICS:
        llm.callbacks = [*(llm.callbacks or []), _UsageCounter(label)]   # each judge is its own client object
        if _EMBEDDINGS is None:
            _EMBEDDINGS = LangchainEmbeddingsWrapper(_MiniLMEmbeddings())
        # bypass_n: ragas may ask for several completions in one request; Groq allows n=1, so they are sent separately.
        # max_retries=1: ragas's own retry (10 attempts, any exception) hid quota errors until the sample timed out;
        # _score below retries per-minute limits itself and stops at once on a daily limit.
        judge = LangchainLLMWrapper(llm, bypass_n=True,
                                    run_config=RunConfig(timeout=int(SAMPLE_TIMEOUT_S), max_retries=1))
        _METRICS[key] = {
            "faithfulness": Faithfulness(llm=judge),
            "relevancy": ResponseRelevancy(llm=judge, embeddings=_EMBEDDINGS, strictness=1),
            "context_recall": LLMContextRecall(llm=judge),
            "context_precision": LLMContextPrecisionWithReference(llm=judge),
        }
    return _METRICS[key]


RATE_LIMIT_RETRIES = 3        # free tiers allow a few thousand tokens a minute: wait for the window to reset, then retry
RATE_LIMIT_WAIT_S = 60

# Groq's free tier also caps each model at 200,000 tokens a day. Once a judge hits a daily cap, retrying cannot help:
# its remaining scores are left missing for this run and the next run (the cache keeps every finished score) fills them.
_EXHAUSTED: Dict[str, str] = {}
_USAGE: Dict[str, Dict[str, int]] = {}


class _UsageCounter(BaseCallbackHandler):
    """Counts each judge's calls and tokens, so a run reports what it used of the daily quota."""

    def __init__(self, label: str) -> None:
        self.label = label

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        u = _USAGE.setdefault(self.label, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        u["calls"] += 1
        for gens in response.generations:
            for g in gens:
                meta = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                u["prompt_tokens"] += int(meta.get("input_tokens", 0))
                u["completion_tokens"] += int(meta.get("output_tokens", 0))


def token_usage() -> Dict[str, Dict[str, int]]:
    return {k: dict(v) for k, v in _USAGE.items()}


def _rate_limited(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return any(s in text for s in ("RateLimit", "429", "RESOURCE_EXHAUSTED", "rate_limit"))


def _daily_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(s in text for s in ("per day", "(tpd)", "(rpd)", "perday"))


async def _score(metric: Any, sample: SingleTurnSample, label: str) -> Optional[float]:
    if label in _EXHAUSTED:
        return None
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            value = await asyncio.wait_for(metric.single_turn_ascore(sample), timeout=SAMPLE_TIMEOUT_S)
            return None if value is None or value != value else round(float(value), 4)   # NaN -> missing
        except Exception as exc:  # noqa: BLE001 - a failed judge call is recorded as missing, never as 0 or 1
            if _rate_limited(exc) and _daily_limit(exc):
                _EXHAUSTED[label] = str(exc)[:300]
                print(f"    [ragas] {label}: DAILY quota reached; its remaining scores are left for the next run")
                return None
            if _rate_limited(exc) and attempt < RATE_LIMIT_RETRIES:
                print(f"    [ragas] {metric.name}: rate limited; retrying in {RATE_LIMIT_WAIT_S}s")
                await asyncio.sleep(RATE_LIMIT_WAIT_S)
                continue
            print(f"    [ragas] {metric.name}: {type(exc).__name__}: {str(exc)[:120]}; scored as missing")
            return None
    return None


def exhausted_judges() -> Dict[str, str]:
    return dict(_EXHAUSTED)


def generation_scores(llm: Any, label: str, question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    """RAGAS Faithfulness and Answer Relevancy of one LLM answer."""
    if llm is None or not answer.strip():
        return {"llm_faithfulness": None, "llm_answer_relevance": None, "judge": None}
    metrics = _metrics_for(llm, label)
    sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts or [""])

    async def both():
        return await _score(metrics["faithfulness"], sample, label), await _score(metrics["relevancy"], sample, label)

    faithfulness, relevancy = asyncio.run(both())
    return {"llm_faithfulness": faithfulness, "llm_answer_relevance": relevancy, "judge": f"ragas + {label}"}


def retrieval_scores(llm: Any, label: str, question: str, contexts: List[str], reference: str) -> Dict[str, Any]:
    """RAGAS Context Recall and Context Precision of one question's retrieved passages, against the reference."""
    if llm is None or not contexts or not reference.strip():
        return {"ragas_context_recall": None, "ragas_context_precision": None, "retrieval_judge": None}
    metrics = _metrics_for(llm, label)
    sample = SingleTurnSample(user_input=question, retrieved_contexts=contexts, reference=reference)

    async def both():
        return (await _score(metrics["context_recall"], sample, label),
                await _score(metrics["context_precision"], sample, label))

    recall, precision = asyncio.run(both())
    return {"ragas_context_recall": recall, "ragas_context_precision": precision, "retrieval_judge": f"ragas + {label}"}
