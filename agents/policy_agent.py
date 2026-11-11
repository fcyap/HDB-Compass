"""
Agent 5: Regulatory Policy RAG Specialist (agents/policy_agent.py)
Proposal Section 4.3 (Page 9).
Owned by Member 3: Advanced RAG & Regulatory Knowledge Base (Category C).

Responsibilities:
1. Answers qualitative regulatory, quota, and classification queries using the
   selected retrieval pipeline (RAG-2 hybrid + reranking or RAG-3 advanced retrieval).
2. Enforces strict grounding on retrieved context:
   - Prohibits temporal drift (e.g. legacy mature vs non-mature rules)
   - Accurately cites official HDB InfoWEB & CPF Board source URLs
   - Takes every figure (MOP years, clawback rates, income ceilings, age exemptions) from the retrieved
     passages only; the system prompt deliberately holds no policy figures, so the model cannot answer
     from memory (Topic 8: the "LLM ignores the grounding rule" failure).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from agents.llm import get_llm
from agents.tool_registry import ToolRunner
from mcp_server.policy_rag_tools import configured_pipeline, retrieve_decomposed, retrieve_hdb_policy_knowledge

# @mcp.tool() wraps the function in a FunctionTool, which is not callable; call the plain function underneath
_retrieve = getattr(retrieve_hdb_policy_knowledge, "fn", retrieve_hdb_policy_knowledge)

POLICY_AGENT_SYSTEM = """You are the Regulatory Policy RAG Agent for HDB-Compass, specializing in Singapore public housing policies, eligibility schemes, CPF housing grants, and financing regulations.

CRITICAL POLICY GROUNDING RULES:
1. Evidence only: Every figure, age, percentage, period and eligibility condition in your answer must come from the retrieved passages. Do not use rules you remember from training, even if you believe them correct: they may be out of date. (The policy figures that used to be written here were removed for that reason; the knowledge base is the single source of truth.)
2. Current framework: Flats are classified as Standard, Plus or Prime. Do not describe them with the legacy Mature vs Non-Mature estate rules unless a passage does.
3. Missing facts: If the passages do not state something the user asked, say plainly that it is not in the retrieved HDB documents and suggest checking hdb.gov.sg or an HFE letter. Never fill the gap with a guess.
4. Conflicts: If two passages disagree, quote both with their sources and their effective dates instead of choosing one silently.
5. Citations: Every regulatory statement MUST cite the official source URL provided in the retrieved passages.
6. Answer-first writing: Begin with one short sentence that directly answers the user's exact question. If the question names several policy topics, cover those topics in the opening sentence without introducing an unasked topic. For yes/no questions, start with "Yes" or "No" and state the key condition. Then state every condition, number, time limit and exception in the passages that bears on the question, each with its citation; leave out passages about topics the user did not ask. Do not begin with a heading, a summary of the passages, or a generic policy introduction.
"""

# Rule 6 and the answer instruction ask for every condition the passages hold, not a short answer. Experiment 2E (11 Nov
# 2026) compared this with the earlier "only the details the user needs ... in a concise response" wording on the same
# passages: Category C fact coverage was equal (0.68) with more citations (0.93 vs 0.80); Category E faithfulness rose
# 0.67 -> 0.78 and fact coverage 0.38 -> 0.45.
ANSWER_INSTRUCTION = ("Then state every condition, figure, time limit and exception from the passages that bears on the "
                      "question, one per line with its citation; do not drop a condition to keep the answer short, and "
                      "do not add topics the question does not ask about. ")


class PolicyRagAgent:
    """Specialist Policy RAG agent that executes retrieval and synthesizes answers."""

    def __init__(self, pipeline: Optional[str] = None, decompose: bool = True):
        self.pipeline = pipeline or configured_pipeline()   # config policy_rag.active_pipeline
        # decompose: a question spanning several policy documents gets one extra search per document (Topic 8)
        self.decompose = decompose
        self.tool_runner = ToolRunner("retrieve_hdb_policy_knowledge")

    @staticmethod
    def _fallback_opening(query: str, passages: List[Dict[str, Any]]) -> str:
        """Grounded non-LLM fallback when no model is available; it does not classify query shapes."""
        for passage in passages:
            for line in str(passage.get("text", "")).splitlines():
                clean = re.sub(r"^(?:[-*]\s*|\d+[.)]\s*)", "", line).strip()
                if clean and not clean.startswith(("[", "#", "|", "---")):
                    return f"The relevant HDB rule is: {clean}"
        return "The retrieved HDB rules relevant to your question are summarized below."

    @staticmethod
    def synthesize_grounded_answer(query: str, passages: List[Dict[str, Any]]) -> str:
        """
        Synthesizes a strictly grounded, citation-enforced policy answer directly from retrieved passages.
        Extracts the highest-signal regulatory statements and table rows matching query intent.
        """
        if not passages:
            return "No official HDB or CPF policy passages were retrieved for this query."

        q_words = {
            w.lower() for w in re.findall(r"\b[A-Za-z0-9-]+\b", query)
            if len(w) > 2 and w.lower() not in {"what", "are", "the", "how", "can", "for", "and", "with", "from", "does"}
        }

        sections_out = []
        for idx, p in enumerate(passages[:3], start=1):
            sec = p.get("section", "HDB Policy")
            url = p.get("source_url", "https://www.hdb.gov.sg")
            raw_lines = [
                ln.strip() for ln in p.get("text", "").splitlines()
                if ln.strip() and not ln.strip().startswith("[") and not re.match(r"^\|[\s:-]+\|$", ln.strip())
            ]
            # Rank lines by query word overlap, keeping top lines + context
            scored_lines = []
            for ln in raw_lines:
                ln_lower = ln.lower()
                overlap = sum(1 for w in q_words if w in ln_lower)
                scored_lines.append((overlap, ln))
            scored_lines.sort(key=lambda x: x[0], reverse=True)
            selected_lines = [ln for _, ln in scored_lines[:5]]
            body_excerpt = " ".join(selected_lines)
            sections_out.append(f"{idx}. **{sec}**: {body_excerpt} ([Official Source]({url}))")

        disclaimer = (
            "\n\n*Regulatory Notice: All eligibility, grant, and financing figures are subject to official "
            "HDB Flat Eligibility (HFE) assessment.*"
        )
        opening = PolicyRagAgent._fallback_opening(query, passages)
        return (opening + "\n\n" if opening else "") + "### Official HDB & CPF Regulatory Analysis\n\n" + "\n\n".join(sections_out) + disclaimer

    @staticmethod
    def generate_llm_answer(query: str, passages: List[Dict[str, Any]], context_override: Optional[str] = None,
                            llm: Any = None) -> str:
        """The grounded LLM answer the user sees ("" when no model is configured or the call fails).
        context_override replaces the numbered passages with other context (the ablation's long-context baseline,
        which sends every policy document instead of retrieved chunks). llm: a client to write with instead of the active profile's (Experiment 2E
        compares generator models)."""
        llm = llm or get_llm()
        if llm is None:
            return ""
        try:
            context_str = context_override if context_override is not None else "\n\n".join([
                f"Passage [{i+1}] (Source: {p['source_url']}) (Section: {p['section']})"
                + (f" (Page retrieved: {p['retrieved']})" if p.get("retrieved") else "") + f":\n{p['text']}"
                for i, p in enumerate(passages)
            ])
            prompt = (
                f"User Question: {query}\n\n"
                f"Retrieved Official Policy Context:\n{context_str}\n\n"
                "Answer the user's exact question directly in the first sentence, before any explanation. "
                "If the question names several policy topics, cover those topics in that first sentence and omit unasked topics. "
                "For a yes/no question, begin with Yes or No and include the key condition in that sentence. "
                + ANSWER_INSTRUCTION +
                "Do not start with a heading or a generic introduction about the policies or retrieved passages. "
                "Use only the retrieved context, and include inline citations with exact URLs for each factual claim. "
                "If the context does not state something the user asked, say it is not in the retrieved HDB documents."
            )
            response = llm.invoke([SystemMessage(content=POLICY_AGENT_SYSTEM), HumanMessage(content=prompt)])
            content = response.content
            if isinstance(content, list):
                content = " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
            return str(content or "")
        except Exception:
            return ""

    def run(self, query: str, policy_domain: Optional[str] = None, use_llm: bool = True) -> Dict[str, Any]:
        """
        Executes policy retrieval and synthesizes a grounded answer with citations.
        """
        # Step 1: Execute RAG retrieval tool (decomposed into per-domain searches for a multi-topic question,
        # unless the caller pinned one domain)
        trajectory: List[Dict[str, Any]] = []
        if self.decompose and not policy_domain:
            retrieval_output, trajectory = retrieve_decomposed(query, pipeline=self.pipeline)
        else:
            retrieval_output = _retrieve(query, policy_domain=policy_domain, pipeline=self.pipeline)
            trajectory = [{"query": query[:120], "policy_domain": policy_domain, "returned": len(retrieval_output.get("passages", [])),
                           "new_passages": len(retrieval_output.get("passages", []))}]
        passages = retrieval_output.get("passages", [])

        # Step 2: Synthesize grounded response using LLM (if active) or deterministic grounded synthesizer
        answer = self.generate_llm_answer(query, passages) if use_llm else ""

        if not answer:
            answer = self.synthesize_grounded_answer(query, passages)

        return {
            "query": query,
            "pipeline": retrieval_output.get("pipeline", self.pipeline),
            "passages": passages,
            "answer": answer,
            "retrieval_trajectory": trajectory,
        }


# Global default instance
policy_agent = PolicyRagAgent()


if __name__ == "__main__":
    q = "What are the exact differences in MOP and clawback between Prime and Plus flats?"
    res = policy_agent.run(q)
    print(f"Policy Agent Result:\nPipeline: {res['pipeline']}\nPassages count: {len(res['passages'])}")
    print(res["answer"])
