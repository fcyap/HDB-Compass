"""
Agent 5: Regulatory Policy RAG Specialist (agents/policy_agent.py)
Proposal Section 4.3 (Page 9).
Owned by Member 3: Advanced RAG & Regulatory Knowledge Base (Category C).

Responsibilities:
1. Answers qualitative regulatory, quota, and classification queries using the
   advanced 3-stage hybrid RAG pipeline (BM25 + dense vector + cross-encoder reranker).
2. Enforces strict grounding on retrieved context:
   - Prohibits temporal drift (e.g. legacy mature vs non-mature rules)
   - Accurately cites official HDB InfoWEB & CPF Board source URLs
   - Explicitly cites MOP years, clawback percentages (6-9% Prime vs 3-6% Plus),
     income ceilings, and age exemption rules (senior 55+ right-sizing).
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
from mcp_server.policy_rag_tools import retrieve_hdb_policy_knowledge

# @mcp.tool() wraps the function in a FunctionTool, which is not callable; call the plain function underneath
_retrieve = getattr(retrieve_hdb_policy_knowledge, "fn", retrieve_hdb_policy_knowledge)

POLICY_AGENT_SYSTEM = """You are the Regulatory Policy RAG Agent for HDB-Compass, specializing in Singapore public housing policies, eligibility schemes, CPF housing grants, and financing regulations.

CRITICAL POLICY GROUNDING RULES:
1. Flat Classification: Under the October 2024 framework, flats are Standard (5-year MOP, 0% clawback, whole flat rental allowed), Plus (10-year MOP, 3-6% clawback, no whole flat rental), and Prime (10-year MOP, 6-9% clawback, no whole flat rental). NEVER confuse this with legacy Mature vs Non-Mature estate rules.
2. Singles Scheme: Unmarried Singapore Citizens aged 35 and above can buy 2-room Flexi BTO flats in ANY estate across all tiers (Standard, Plus, Prime), or resale flats of any size (up to Executive) on the open market.
3. Private Property Wait-Out: 15-month wait-out applies to private property owners buying non-subsidised resale flats, but seniors aged 55 and above right-sizing to 4-room or smaller resale flats are STRICTLY EXEMPT.
4. Grants: EHG provides up to $120,000 for families (income ceiling $9,000) and up to $60,000 for singles (income ceiling $4,500). First-timer resale buyers can stack EHG + Family Grant + PHG up to $230,000.
5. Citations: Every regulatory statement MUST cite the official source URL provided in the retrieved passages.
"""


class PolicyRagAgent:
    """Specialist Policy RAG agent that executes retrieval and synthesizes answers."""

    def __init__(self, pipeline: str = "RAG-3"):
        self.pipeline = pipeline
        self.tool_runner = ToolRunner("retrieve_hdb_policy_knowledge")

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
        return "### Official HDB & CPF Regulatory Analysis\n\n" + "\n\n".join(sections_out) + disclaimer

    def run(self, query: str, policy_domain: Optional[str] = None, use_llm: bool = True) -> Dict[str, Any]:
        """
        Executes policy retrieval and synthesizes a grounded answer with citations.
        """
        # Step 1: Execute RAG retrieval tool
        retrieval_output = _retrieve(query, policy_domain=policy_domain, pipeline=self.pipeline)
        passages = retrieval_output.get("passages", [])

        # Step 2: Synthesize grounded response using LLM (if active) or deterministic grounded synthesizer
        answer = ""
        if use_llm:
            llm = get_llm()
            if llm is not None:
                try:
                    context_str = "\n\n".join([
                        f"Passage [{i+1}] (Source: {p['source_url']}) (Section: {p['section']}):\n{p['text']}"
                        for i, p in enumerate(passages)
                    ])
                    prompt = (
                        f"User Question: {query}\n\n"
                        f"Retrieved Official Policy Context:\n{context_str}\n\n"
                        "Synthesize a clear, concise, and structured answer. "
                        "Include inline citations with exact URLs for each factual claim."
                    )
                    response = llm.invoke([SystemMessage(content=POLICY_AGENT_SYSTEM), HumanMessage(content=prompt)])
                    answer = response.content
                except Exception:
                    answer = ""

        if not answer:
            answer = self.synthesize_grounded_answer(query, passages)

        return {
            "query": query,
            "pipeline": retrieval_output.get("pipeline", self.pipeline),
            "passages": passages,
            "answer": answer,
        }


# Global default instance
policy_agent = PolicyRagAgent(pipeline="RAG-3")


if __name__ == "__main__":
    q = "What are the exact differences in MOP and clawback between Prime and Plus flats?"
    res = policy_agent.run(q)
    print(f"Policy Agent Result:\nPipeline: {res['pipeline']}\nPassages count: {len(res['passages'])}")
    print(res["answer"])
