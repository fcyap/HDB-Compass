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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from agents.llm import get_llm
from agents.tool_registry import ToolRunner
from mcp_server.policy_rag_tools import retrieve_hdb_policy_knowledge

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

    def run(self, query: str, policy_domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes policy retrieval and synthesizes a grounded answer with citations.
        """
        # Step 1: Execute RAG retrieval tool
        retrieval_output = retrieve_hdb_policy_knowledge(query, policy_domain=policy_domain, pipeline=self.pipeline)
        passages = retrieval_output.get("passages", [])

        # Step 2: Synthesize grounded response using LLM (if active) or deterministic template
        llm = get_llm()
        answer = ""
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
                llm = None

        if not answer:
            # Deterministic fallback synthesis
            top_passages_summary = "\n".join([
                f"- **{p['section']}**: {p['text'].splitlines()[1] if len(p['text'].splitlines()) > 1 else p['text'][:120]}... [Source]({p['source_url']})"
                for p in passages[:3]
            ])
            answer = f"### Policy Analysis\n\n{top_passages_summary}"

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
