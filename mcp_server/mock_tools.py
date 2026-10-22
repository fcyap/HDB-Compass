"""
Mock MCP tools for Agents 2, 4 and 5 (mcp_server/mock_tools.py). Owned by Member 4.

Purpose: let the orchestration layer (Variants A/B/C, supervisor, auditor hook,
eval runner) be built and tested before Members 1 and 3 deliver real tools.
Every function has the exact signature from proposal §4.3 and returns a fixture
that validates against agents/schemas.py. Deterministic: same input -> same output.

Member 2's spatial tools are real already and are NOT mocked here.

Swap-out: agents/tool_registry.py picks real or mock per tool from
config/experiment_config.yaml -> tools.use_mock.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import FastMCP

from agents.schemas import (
    CalculateBsdAndUpfrontOutlayOutput,
    CalculateCpfHousingGrantsOutput,
    CalculateMaximumAffordabilityOutput,
    CompareBtoLaunchVsResaleOutput,
    ComputeTownPriceTimeseriesOutput,
    QueryResaleBlocksOutput,
    RetrieveHdbPolicyKnowledgeOutput,
)

mcp = FastMCP("HDB-Compass-Mock-Tools")

# ---------------------------------------------------------------------------
# Fixture: a small block universe. Streets/blocks chosen so a subset also
# appears in Member 2's real spatial index (Bishan St 13, Tampines St 21 ...),
# which makes the supervisor's set intersection non-trivial even on mocks.
# ---------------------------------------------------------------------------
_FIXTURE_BLOCKS: List[Dict[str, Any]] = [
    # town, block, street, postal, lat, lng, flat_type, median_price, ppsm, lease, txns
    ("BISHAN", "123", "BISHAN ST 13", "570123", 1.3487, 103.8512, "4 ROOM", 735000, 7900, 62.5, 9),
    ("BISHAN", "151", "BISHAN ST 11", "570151", 1.3526, 103.8471, "4 ROOM", 712000, 7650, 61.0, 6),
    ("BISHAN", "233", "BISHAN ST 22", "570233", 1.3577, 103.8395, "4 ROOM", 698000, 7500, 63.8, 7),
    ("TAMPINES", "201", "TAMPINES ST 21", "520201", 1.3545, 103.9533, "4 ROOM", 585000, 6300, 66.2, 14),
    ("TAMPINES", "432", "TAMPINES ST 41", "520432", 1.3589, 103.9478, "4 ROOM", 572000, 6150, 67.0, 11),
    ("TAMPINES", "875", "TAMPINES ST 84", "520875", 1.3491, 103.9375, "4 ROOM", 610000, 6500, 72.5, 8),
    ("QUEENSTOWN", "53", "STRATHMORE AVE", "142053", 1.2924, 103.8075, "5 ROOM", 842000, 8800, 64.1, 5),
    ("QUEENSTOWN", "91", "TANGLIN HALT RD", "141091", 1.3023, 103.7988, "5 ROOM", 798000, 8400, 55.3, 4),
    ("BUKIT MERAH", "128", "BT MERAH VIEW", "150128", 1.2836, 103.8232, "5 ROOM", 815000, 8600, 60.7, 6),
    ("ANG MO KIO", "308", "ANG MO KIO AVE 1", "560308", 1.3661, 103.8422, "4 ROOM", 520000, 5900, 58.9, 12),
    ("JURONG EAST", "245", "JURONG EAST ST 24", "600245", 1.3376, 103.7351, "4 ROOM", 545000, 5800, 62.0, 10),
    ("BUKIT MERAH", "25", "TIONG BAHRU RD", "160025", 1.2857, 103.8276, "5 ROOM", 1050000, 9900, 52.0, 2),
    # --- rows below exist in Member 2's real spatial index, so SQL ∩ spatial is non-empty on mocks ---
    ("BISHAN", "513", "BISHAN ST 13", "570513", 1.3495, 103.8496, "4 ROOM", 745000, 8000, 66.0, 8),
    ("BISHAN", "510", "BISHAN ST 13", "570510", 1.3488, 103.8497, "4 ROOM", 728000, 7850, 66.2, 7),
    ("BISHAN", "312", "SHUNFU RD", "570312", 1.3512, 103.8387, "4 ROOM", 690000, 7400, 60.5, 5),
    ("TAMPINES", "924", "TAMPINES ST 91", "520924", 1.3468, 103.9408, "4 ROOM", 598000, 6400, 70.1, 12),
    ("TAMPINES", "927", "TAMPINES ST 91", "520927", 1.3463, 103.9404, "4 ROOM", 605000, 6450, 70.3, 9),
    ("TAMPINES", "935", "TAMPINES ST 91", "520935", 1.3457, 103.9411, "4 ROOM", 612000, 6500, 69.8, 10),
    ("QUEENSTOWN", "182A", "STIRLING RD", "141182", 1.2955, 103.8046, "5 ROOM", 830000, 8700, 63.0, 6),
    ("BUKIT MERAH", "9B", "BOON TIONG RD", "163009", 1.2869, 103.8289, "5 ROOM", 845000, 8900, 61.5, 4),
    ("BUKIT MERAH", "11B", "BOON TIONG RD", "163011", 1.2861, 103.8295, "5 ROOM", 838000, 8850, 61.4, 5),
]


def _rows() -> List[Dict[str, Any]]:
    keys = ["town", "block", "street", "postal", "lat", "lng", "flat_type",
            "median_resale_price", "median_price_per_sqm",
            "median_remaining_lease_years", "transaction_count"]
    return [dict(zip(keys, r)) for r in _FIXTURE_BLOCKS]


# ---------------------------------------------------------------------------
# Agent 2 — Market & Valuation SQL (mock of Member 1)
# ---------------------------------------------------------------------------
@mcp.tool()
def query_resale_blocks_by_criteria(
    towns: List[str],
    flat_types: List[str],
    max_price: float,
    min_lease_years: int,
) -> Dict[str, Any]:
    """Returns aggregated block-level resale medians, transaction volumes and
    remaining leases (mock of DuckDB query)."""
    towns_u = {t.upper() for t in towns}
    types_u = {f.upper() for f in flat_types}
    out = []
    for r in _rows():
        if towns_u and r["town"] not in towns_u:
            continue
        if types_u and r["flat_type"] not in types_u:
            continue
        if r["median_resale_price"] > max_price:
            continue
        if r["median_remaining_lease_years"] < min_lease_years:
            continue
        out.append({**r, "last_transaction_month": "2026-09"})
    sql = ("SELECT ... FROM resale WHERE town IN (?) AND flat_type IN (?) "
           "AND resale_price <= ? AND remaining_lease_years >= ? GROUP BY block, street")
    return QueryResaleBlocksOutput(blocks=out, sql=sql).model_dump()


@mcp.tool()
def compute_town_price_timeseries(
    towns: List[str],
    flat_type: str,
    lookback_years: int = 3,
) -> Dict[str, Any]:
    """Quarterly median prices, YoY appreciation and price-per-sqm per town (mock)."""
    base = {"TAMPINES": 540000, "BEDOK": 505000, "BISHAN": 690000, "QUEENSTOWN": 790000}
    series, summary = [], []
    quarters = [f"{y}Q{q}" for y in range(2026 - lookback_years + 1, 2027) for q in (1, 2, 3, 4)][: lookback_years * 4]
    for t in towns:
        tu = t.upper()
        b = base.get(tu, 560000)
        for i, qtr in enumerate(quarters):
            price = round(b * (1 + 0.012 * i), -3)
            series.append({"town": tu, "quarter": qtr, "median_price": price,
                           "median_price_per_sqm": round(price / 93, 1), "transaction_count": 40 + i})
        latest = series[-1]["median_price"]
        summary.append({"town": tu, "latest_median_price": latest,
                        "yoy_appreciation_pct": 4.9, "median_price_per_sqm": round(latest / 93, 1)})
    return ComputeTownPriceTimeseriesOutput(series=series, summary=summary).model_dump()


@mcp.tool()
def compare_bto_launch_vs_resale(bto_project_name: str, radius_meters: int = 500) -> Dict[str, Any]:
    """Benchmarks indicative BTO launch prices against nearby recent resale flats (mock)."""
    return CompareBtoLaunchVsResaleOutput(
        bto_project_name=bto_project_name,
        bto_indicative_price_range={"min": 430000.0, "max": 575000.0},
        resale_median_nearby=705000.0,
        resale_sample_size=37,
        price_gap_pct=-28.7,
    ).model_dump()


# ---------------------------------------------------------------------------
# Agent 4 — Financial & Grant Calculator (mock of Member 1)
# Numbers are plausible but NOT official. Member 1 replaces with real tables.
# ---------------------------------------------------------------------------
@mcp.tool()
def calculate_cpf_housing_grants(
    flat_category: str,
    citizenship_pair: str,
    first_timer_status: bool,
    monthly_income: float,
    near_parents_4km: bool = False,
) -> Dict[str, Any]:
    """Deterministic EHG / Family Grant / PHG lookup (mock brackets)."""
    notes: List[str] = []
    if not first_timer_status:
        return CalculateCpfHousingGrantsOutput(
            enhanced_housing_grant=0, family_grant=0, proximity_housing_grant=0,
            total_grants=0, eligibility_notes=["Second-timer: no first-timer grants"]).model_dump()
    # EHG: stepped by income, max 120k at <=1500, 0 above 9000
    if monthly_income <= 1500:
        ehg = 120000.0
    elif monthly_income >= 9000:
        ehg = 0.0
    else:
        ehg = round(120000.0 - (monthly_income - 1500) / 500 * 7500, -2)
    family = 0.0
    if flat_category == "resale":
        family = 80000.0 if citizenship_pair.upper() == "SC_SC" else 70000.0 if citizenship_pair.upper() == "SC_PR" else 0.0
        if citizenship_pair.upper() == "SC_SINGLE":
            family = 40000.0
    else:
        notes.append("Family Grant applies to resale flats only")
    phg = 30000.0 if (near_parents_4km and flat_category == "resale") else 0.0
    if citizenship_pair.upper() == "SC_SINGLE" and phg:
        phg = 15000.0
    return CalculateCpfHousingGrantsOutput(
        enhanced_housing_grant=ehg, family_grant=family, proximity_housing_grant=phg,
        total_grants=ehg + family + phg, eligibility_notes=notes).model_dump()


@mcp.tool()
def calculate_maximum_affordability(
    monthly_income: float,
    cpf_oa_balance: float,
    cash_savings: float,
    loan_type: str,
    interest_rate: float = 2.6,
    tenure_years: int = 25,
) -> Dict[str, Any]:
    """30% MSR, TDSR and LTV capping -> max purchase price and instalment (mock, real formula)."""
    msr_cap = 0.30 * monthly_income
    r = interest_rate / 100 / 12
    n = tenure_years * 12
    max_loan = msr_cap * (1 - (1 + r) ** -n) / r
    ltv = 0.75 if loan_type == "bank" else 0.80
    price_by_ltv = max_loan / ltv
    price_by_down = (cpf_oa_balance + cash_savings) / (1 - ltv)
    binding = "MSR" if price_by_ltv <= price_by_down else "DOWNPAYMENT"
    max_price = min(price_by_ltv, price_by_down)
    return CalculateMaximumAffordabilityOutput(
        max_loan_amount=round(max_loan, 2), max_purchase_price=round(max_price, 2),
        monthly_instalment=round(min(msr_cap, max_price * ltv * r / (1 - (1 + r) ** -n)), 2),
        msr_cap_monthly=round(msr_cap, 2), ltv_ratio=ltv, binding_constraint=binding).model_dump()


@mcp.tool()
def calculate_bsd_and_upfront_outlay(
    purchase_price: float, loan_type: str, total_grants: float = 0.0
) -> Dict[str, Any]:
    """Tiered BSD, minimum cash downpayment, CPF usage, legal/valuation fees (mock tiers)."""
    tiers = [(180000, 0.01), (180000, 0.02), (640000, 0.03), (500000, 0.04), (1500000, 0.05)]
    remaining, bsd = purchase_price, 0.0
    for width, rate in tiers:
        slab = min(remaining, width)
        bsd += slab * rate
        remaining -= slab
        if remaining <= 0:
            break
    bsd += max(remaining, 0) * 0.06
    down_pct = 0.25 if loan_type == "bank" else 0.20
    min_cash = purchase_price * (0.05 if loan_type == "bank" else 0.0)
    cpf_usable = purchase_price * down_pct - min_cash
    fees = 2500.0 + 300.0
    total_upfront = purchase_price * down_pct + bsd + fees
    return CalculateBsdAndUpfrontOutlayOutput(
        buyer_stamp_duty=round(bsd, 2), min_cash_downpayment=round(min_cash, 2),
        cpf_oa_usable=round(cpf_usable, 2), legal_and_valuation_fees=fees,
        total_upfront=round(total_upfront, 2),
        cash_required=round(max(min_cash + bsd + fees - total_grants, 0.0), 2)).model_dump()


# ---------------------------------------------------------------------------
# Agent 5 — Regulatory Policy RAG (mock of Member 3)
# ---------------------------------------------------------------------------
_POLICY_FIXTURE = {
    "classification": [
        ("Prime flats: 10-year Minimum Occupation Period; subsidy recovery of 6% to 9% of resale price "
         "or valuation on first resale; whole-flat rental not permitted after MOP.",
         "https://www.hdb.gov.sg/residential/buying-a-flat/new-flats/flat-classification", "Prime"),
        ("Plus flats: 10-year MOP; subsidy recovery of 3% to 6% on first resale; whole-flat rental not permitted.",
         "https://www.hdb.gov.sg/residential/buying-a-flat/new-flats/flat-classification", "Plus"),
        ("Standard flats: 5-year MOP; no subsidy recovery; whole-flat rental allowed after MOP.",
         "https://www.hdb.gov.sg/residential/buying-a-flat/new-flats/flat-classification", "Standard"),
    ],
    "eligibility": [
        ("Singles aged 35 and above may buy a 2-room Flexi BTO flat in any estate, or a resale flat of any type.",
         "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan", None),
        ("Private property owners must wait 15 months after disposal before buying a resale flat; seniors "
         "aged 55 and above buying a 4-room or smaller flat are exempt.",
         "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan", None),
    ],
    "grants": [
        ("Enhanced CPF Housing Grant (EHG) of up to $120,000 for first-timer families with average "
         "monthly household income not exceeding $9,000.",
         "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan/schemes-and-grants", None),
    ],
    "eip": [
        ("Ethnic Integration Policy sets block and neighbourhood quotas by ethnic group; a resale purchase "
         "is blocked if it would cause the buyer's group to exceed its quota.",
         "https://www.hdb.gov.sg/residential/buying-a-flat/resale/eligibility/ethnic-integration-policy", None),
    ],
}


@mcp.tool()
def retrieve_hdb_policy_knowledge(query: str, policy_domain: Optional[str] = None) -> Dict[str, Any]:
    """Hybrid RAG + reranker (mock): returns top-5 passages with citations."""
    q = query.lower()
    domain = policy_domain or (
        "classification" if any(k in q for k in ("prime", "plus", "standard", "mop", "clawback")) else
        "grants" if any(k in q for k in ("grant", "ehg", "phg")) else
        "eip" if "eip" in q or "ethnic" in q else "eligibility")
    passages = []
    for i, (text, url, cls) in enumerate(_POLICY_FIXTURE.get(domain, [])):
        passages.append({"text": text, "source_url": url, "section": domain.title(),
                         "policy_category": domain, "flat_classification": cls,
                         "effective_date": "2024-10-01", "rerank_score": round(0.95 - 0.07 * i, 3)})
    return RetrieveHdbPolicyKnowledgeOutput(passages=passages[:5], pipeline="RAG-3").model_dump()


MOCK_TOOLS = {
    "query_resale_blocks_by_criteria": query_resale_blocks_by_criteria,
    "compute_town_price_timeseries": compute_town_price_timeseries,
    "compare_bto_launch_vs_resale": compare_bto_launch_vs_resale,
    "calculate_cpf_housing_grants": calculate_cpf_housing_grants,
    "calculate_maximum_affordability": calculate_maximum_affordability,
    "calculate_bsd_and_upfront_outlay": calculate_bsd_and_upfront_outlay,
    "retrieve_hdb_policy_knowledge": retrieve_hdb_policy_knowledge,
}

if __name__ == "__main__":
    mcp.run()
