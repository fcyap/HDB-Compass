"""
Category C Golden Queries Builder (evaluation/build_category_c.py).
Owned by Member 3: Advanced RAG & Regulatory Knowledge Base (Category C).

Generates the 15 stratified Category C test cases (5 per sub-use case: C1, C2, C3)
matching Proposal §2 (Pages 3-4), §6.1 (Page 10), and §6.2 (Page 11).

Sub-use cases:
- C1: Standard / Plus / Prime Classification Rules (MOP, clawback, renting)
- C2: Scheme & Age Eligibility (Singles 35, Joint Singles, Fiancé/Fiancée)
- C3: Wait-Out Periods & EIP Quotas (15-month senior 55 exemption, 30-month, EIP buyback)

Outputs:
- evaluation/category_c_queries.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_JSON = PROJECT_ROOT / "evaluation" / "category_c_queries.json"

CATEGORY_C_CASES = [
    # -----------------------------------------------------------------------
    # C1: Standard / Plus / Prime Flat Classification Rules
    # -----------------------------------------------------------------------
    {
        "id": "C1-01",
        "category": "C",
        "sub_use_case": "C1",
        "query": "We are comparing a Prime BTO vs. a Plus resale flat. What are the exact differences in MOP, subsidy clawback percentages, and whole-flat rental restrictions?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "Prime vs Plus MOP subsidy clawback whole-flat rental"
                }
            }
        ],
        "ground_truth_facts": {
            "prime_mop_years": 10,
            "plus_mop_years": 10,
            "prime_clawback_pct_range": "6% to 9%",
            "plus_clawback_pct_range": "3% to 6%",
            "whole_flat_rental_allowed": False,
            "key_clauses": [
                ["10", "mop"],
                ["clawback", "6%", "9%"],
                ["clawback", "3%", "6%"],
                ["whole-flat", "rental", "prohibited"]
            ],
            "target_sections": ["Detailed Rules for Prime Flats", "Detailed Rules for Plus Flats", "Key Policy Comparison Matrix"]
        }
    },
    {
        "id": "C1-02",
        "category": "C",
        "sub_use_case": "C1",
        "query": "Can I rent out my whole Prime flat after the 10-year Minimum Occupation Period is over?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "Prime flat rent out whole flat after 10 year MOP"
                }
            }
        ],
        "ground_truth_facts": {
            "whole_flat_rental_allowed": False,
            "room_rental_allowed": True,
            "key_clauses": [
                "whole flat is strictly prohibited for life",
                "spare bedrooms may do so only after completing the 10-year MOP"
            ],
            "target_sections": ["Detailed Rules for Prime Flats", "Rental Restrictions"]
        }
    },
    {
        "id": "C1-03",
        "category": "C",
        "sub_use_case": "C1",
        "query": "What is the subsidy recovery clawback rate for a Plus flat upon its first resale, and does it apply to subsequent buyers?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "Plus flat subsidy recovery clawback rate first resale subsequent buyers"
                }
            }
        ],
        "ground_truth_facts": {
            "clawback_pct_range": "3% to 6%",
            "applies_to_subsequent_buyers": False,
            "key_clauses": [
                "3% to 6% subsidy recovery",
                "applies only to the first resale"
            ],
            "target_sections": ["Detailed Rules for Plus Flats", "Subsidy Recovery Mechanism"]
        }
    },
    {
        "id": "C1-04",
        "category": "C",
        "sub_use_case": "C1",
        "query": "Are singles allowed to buy Prime or Plus resale flats on the open market, and what are the restrictions on flat size and income ceiling?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "singles buy Prime Plus resale flat size income ceiling"
                }
            }
        ],
        "ground_truth_facts": {
            "singles_eligible": True,
            "max_flat_size": "2-room only",
            "income_ceiling_sgd": 7000,
            "key_clauses": [
                "2-room resale flats only",
                "$7,000/month (Singles)",
                "35 and above"
            ],
            "target_sections": ["Key Policy Comparison Matrix", "Resale Market Eligibility Conditions"]
        }
    },
    {
        "id": "C1-05",
        "category": "C",
        "sub_use_case": "C1",
        "query": "What are the key differences between Standard and Plus flats regarding MOP length and rental regulations?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "Standard vs Plus flats MOP rental regulations"
                }
            }
        ],
        "ground_truth_facts": {
            "standard_mop_years": 5,
            "plus_mop_years": 10,
            "standard_whole_flat_rental": True,
            "plus_whole_flat_rental": False,
            "key_clauses": [
                "Standard flats: 5 years",
                "Plus flats: 10 years",
                "Whole flat can be rented out on the open market once the 5-year MOP is completed"
            ],
            "target_sections": ["Key Policy Comparison Matrix", "Detailed Rules for Standard Flats"]
        }
    },

    # -----------------------------------------------------------------------
    # C2: Scheme & Age Eligibility
    # -----------------------------------------------------------------------
    {
        "id": "C2-01",
        "category": "C",
        "sub_use_case": "C2",
        "query": "I am a 34-year-old single citizen. How do my flat options and grant rules differ if I apply for a 2-room Flexi BTO now vs. a resale flat at age 35?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "34-year-old single citizen 2-room Flexi BTO vs resale age 35"
                }
            }
        ],
        "ground_truth_facts": {
            "age_34_eligible": False,
            "age_35_eligible": True,
            "eligible_flat_types_at_35": "2-room Flexi BTO in any estate, or resale flat of any size",
            "key_clauses": [
                "Singles under 35 are ineligible to buy a BTO flat under the SSC scheme",
                "Singapore Citizen aged 35 and above",
                "2-room Flexi BTO flats in any estate",
                "any flat size (2-room, 3-room, 4-room, 5-room, or Executive)"
            ],
            "target_sections": ["Single Singapore Citizen Scheme (SSC)", "Minimum Age and Citizenship Restrictions Summary"]
        }
    },
    {
        "id": "C2-02",
        "category": "C",
        "sub_use_case": "C2",
        "query": "What is the minimum age and income ceiling for an unmarried Singapore Citizen applying for a 2-room Flexi BTO flat under the Single Singapore Citizen Scheme?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "Single Singapore Citizen Scheme minimum age income ceiling 2-room Flexi BTO"
                }
            }
        ],
        "ground_truth_facts": {
            "min_age": 35,
            "income_ceiling_sgd": 7000,
            "flat_type": "2-room Flexi BTO",
            "key_clauses": [
                "aged 35 and above",
                "Income ceiling for 2-room Flexi BTO is $7,000/month"
            ],
            "target_sections": ["Single Singapore Citizen Scheme (SSC)"]
        }
    },
    {
        "id": "C2-03",
        "category": "C",
        "sub_use_case": "C2",
        "query": "Can two unmarried Singapore Citizens aged 35 buy a resale flat together under the Joint Singles Scheme?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "Joint Singles Scheme JSS two unmarried citizens aged 35 resale flat"
                }
            }
        ],
        "ground_truth_facts": {
            "eligible": True,
            "scheme": "Joint Singles Scheme (JSS)",
            "key_clauses": [
                "2 to 4 unmarried Singapore Citizens aged 35 and above",
                "purchase any size resale flat on the open market"
            ],
            "target_sections": ["Joint Singles Scheme (JSS)"]
        }
    },
    {
        "id": "C2-04",
        "category": "C",
        "sub_use_case": "C2",
        "query": "What schemes are available for a Singapore Citizen married to a foreigner on a Long-Term Visit Pass?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "Singapore citizen married to foreigner LTVP Non-Citizen Spouse Scheme"
                }
            }
        ],
        "ground_truth_facts": {
            "scheme_name": "Non-Citizen Spouse Scheme",
            "resale_options": "Any flat size on open market",
            "bto_options": "2-room Flexi if aged 35+",
            "key_clauses": [
                "Non-Citizen Spouse Scheme",
                "Singapore Citizen applicant aged 21 and above with a non-citizen spouse on a valid Long-Term Visit Pass (LTVP)"
            ],
            "target_sections": ["Non-Citizen Spouse Scheme"]
        }
    },
    {
        "id": "C2-05",
        "category": "C",
        "sub_use_case": "C2",
        "query": "If a couple applies under the Fiancé/Fiancée Scheme, by when must they submit their marriage certificate?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "Fiancé Fiancée Scheme marriage certificate submission deadline"
                }
            }
        ],
        "ground_truth_facts": {
            "deadline_months": 3,
            "trigger_event": "taking possession / key collection",
            "key_clauses": [
                "submit an official Marriage Certificate within 3 months of taking possession of the flat"
            ],
            "target_sections": ["Fiancé / Fiancée Scheme"]
        }
    },

    # -----------------------------------------------------------------------
    # C3: Wait-Out Periods & EIP Quotas
    # -----------------------------------------------------------------------
    {
        "id": "C3-01",
        "category": "C",
        "sub_use_case": "C3",
        "query": "My parents (aged 58) are right-sizing from a private condo to a 4-room resale flat. Are they exempt from the 15-month wait-out period?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "parents aged 58 right-sizing private condo 4-room resale exempt 15-month wait-out"
                }
            }
        ],
        "ground_truth_facts": {
            "is_exempt": True,
            "min_senior_age": 55,
            "max_flat_size_for_exemption": "4-room or smaller",
            "key_clauses": [
                "Seniors aged 55 and above (and their spouses) who are right-sizing from a private property to an HDB resale flat are strictly EXEMPT",
                "applies only when purchasing a 4-room or smaller resale flat"
            ],
            "target_sections": ["Senior Citizen Exemption from 15-Month Wait-Out"]
        }
    },
    {
        "id": "C3-02",
        "category": "C",
        "sub_use_case": "C3",
        "query": "I sold my private residential property last month and I am 45 years old. How long must I wait before buying a 4-room HDB resale flat?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "sold private property aged 45 wait-out period 4-room resale"
                }
            }
        ],
        "ground_truth_facts": {
            "wait_out_months": 15,
            "is_exempt": False,
            "reason": "Under 55 years of age",
            "key_clauses": [
                ["15", "months", "disposal", "private"],
                ["below 55", "wait-out"]
            ],
            "target_sections": ["The 15-Month Wait-Out Period for Open Market Resale Flats"]
        }
    },
    {
        "id": "C3-03",
        "category": "C",
        "sub_use_case": "C3",
        "query": "What is the 30-month wait-out period for former private property owners, and does it apply to BTO flats and CPF housing grants?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "30-month wait-out period former private property BTO CPF housing grants"
                }
            }
        ],
        "ground_truth_facts": {
            "wait_out_months": 30,
            "applies_to_bto": True,
            "applies_to_grants": True,
            "key_clauses": [
                "30-month wait-out period from the disposal of private residential property applies to",
                "Applying for a new Build-To-Order (BTO) flat",
                "Applying for CPF Housing Grants"
            ],
            "target_sections": ["The 30-Month Wait-Out Period"]
        }
    },
    {
        "id": "C3-04",
        "category": "C",
        "sub_use_case": "C3",
        "query": "How does the Ethnic Integration Policy (EIP) block quota affect a resale transaction if both buyer and seller are of the same ethnicity?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "EIP block quota buyer seller same ethnicity exception"
                }
            }
        ],
        "ground_truth_facts": {
            "transaction_allowed": True,
            "exception_rule": "Same-ethnicity transactions do not alter the ethnic balance and are always permitted.",
            "key_clauses": [
                "Same-Ethnicity Exception",
                "If both the seller and the buyer belong to the same ethnic group, the transaction is always permitted"
            ],
            "target_sections": ["Ethnic Integration Policy (EIP)", "Transaction Restrictions"]
        }
    },
    {
        "id": "C3-05",
        "category": "C",
        "sub_use_case": "C3",
        "query": "Under what conditions can a flat owner apply for the HDB EIP Buyback Assistance Scheme if they cannot sell their flat due to ethnic quotas?",
        "expected_agents": ["policy_rag"],
        "expected_mcp_calls": [
            {
                "agent": "policy_rag",
                "tool": "retrieve_hdb_policy_knowledge",
                "required_params": {
                    "query": "HDB EIP Buyback Assistance Scheme conditions marketing period MOP"
                }
            }
        ],
        "ground_truth_facts": {
            "mop_completed": True,
            "active_marketing_months": 6,
            "buyback_pricing": "Prevailing fair market valuation",
            "key_clauses": [
                "EIP Buyback Assistance Scheme",
                "Must have completed the Minimum Occupation Period (MOP)",
                "actively marketed on major property portals for at least 6 continuous months",
                "HDB buys back the flat directly"
            ],
            "target_sections": ["EIP Buyback Assistance Scheme", "Eligibility Requirements"]
        }
    }
]


# ---------------------------------------------------------------------------------------------------------------------
# Official ground truth (Nov 2026). The ground_truth_facts above were written from the legacy hand-written corpus, which
# was later found wrong in places (Plus/Prime subsidy-recovery rates, the singles $7,000 ceiling, the 15-month wait-out).
# Each question's truth is now taken from the captured official pages via evaluation/policy_fact_sheet.yaml (`fact_ids`).
# key_clauses: the words a correct answer (and a relevant passage) must contain; target_sections: titles of the official
# pages that hold the answer (matched against the chunk's section breadcrumb). answerable=False: no captured page states
# the rule, so the correct answer says it is not in the documents (a test of the grounding rule).
# The old truth is kept under "legacy_ground_truth" for the old-vs-new corpus comparison.
# ---------------------------------------------------------------------------------------------------------------------
FRAMEWORK = "Standard, Plus, and Prime Housing Framework"
OFFICIAL_TRUTH = {
    "C1-01": dict(fact_ids=["mop_by_classification", "plus_prime_subsidy_recovery_rate",
                            "subsidy_recovery_not_on_open_market_resale", "no_whole_flat_rental_plus_prime"],
                  key_clauses=[["10 years"], ["subsidy recovery"], ["open market"], ["whole flat", "not allowed"]],
                  target_sections=[FRAMEWORK]),
    "C1-02": dict(fact_ids=["no_whole_flat_rental_plus_prime"],
                  key_clauses=[["whole flat"], ["not allowed"]],
                  target_sections=[FRAMEWORK, "Conditions After Buying a New Flat"]),
    "C1-03": dict(fact_ids=["plus_prime_subsidy_recovery_rate", "subsidy_recovery_not_on_open_market_resale"],
                  key_clauses=[["percentage"], ["launched"], ["open market"]],
                  target_sections=[FRAMEWORK]),
    "C1-04": dict(fact_ids=["singles_new_flat_type", "singles_flexi_income_ceiling", "singles_age_35"],
                  key_clauses=[["2-room"], ["$8,000"], ["35"]],
                  target_sections=["Singles"]),
    "C1-05": dict(fact_ids=["mop_by_classification", "no_whole_flat_rental_plus_prime"],
                  key_clauses=[["5 years"], ["10 years"], ["Allowed after MOP"], ["Not allowed"]],
                  target_sections=[FRAMEWORK]),
    "C2-01": dict(fact_ids=["singles_age_35", "singles_new_flat_type"],
                  key_clauses=[["35"], ["2-room Flexi"]],
                  target_sections=["Singles"]),
    "C2-02": dict(fact_ids=["singles_age_35", "singles_flexi_income_ceiling"],
                  key_clauses=[["35"], ["$8,000"]],
                  target_sections=["Singles"]),
    "C2-03": dict(fact_ids=["two_singles_together", "singles_new_flat_type", "singles_age_35"],
                  key_clauses=[["two or more singles"], ["35"]],
                  target_sections=["Singles"]),
    "C2-04": dict(fact_ids=["non_resident_spouse_flats", "non_resident_spouse_grant_age"],
                  key_clauses=[["non-resident spouse"], ["21"]],
                  target_sections=["CPF Housing Grant for Singles Buying Resale Flats", "Couples and Families"]),
    "C2-05": dict(fact_ids=["fiance_solemnise_new_flat", "fiance_solemnise_resale"],
                  key_clauses=[["solemnise"], ["3 months"]],
                  target_sections=["Key Collection", "Couples and Families"],
                  note="HDB's pages state the marriage must be solemnised within 3 months of key collection (new flat) "
                       "or of completion (resale); an overseas marriage certificate is submitted within the same 3 months."),
    "C3-01": dict(fact_ids=["wait_out_removed_2026", "wait_out_senior_exemption"],
                  key_clauses=[["no longer"], ["15-month wait-out"], ["55 and above"]],
                  target_sections=["Removal of the 15-month Wait-out Period for Private Residential Property Owners "
                                   "Purchasing Non-Subsidised HDB Resale Flats"],
                  note="HDB removed the 15-month wait-out on 27 July 2026 for non-subsidised resale purchases without an "
                       "HDB loan; before that, citizens aged 55+ moving to a 4-room or smaller resale flat were exempt."),
    "C3-02": dict(fact_ids=["private_property_30_months"],
                  key_clauses=[["30 months"], ["6 months"]],
                  target_sections=["Couples and Families"]),
    "C3-03": dict(fact_ids=["private_property_30_months"],
                  key_clauses=[["30 months"], ["grant"]],
                  target_sections=["Couples and Families"]),
    "C3-04": dict(fact_ids=["eip_same_ethnicity"],
                  key_clauses=[["same ethnic group"]],
                  target_sections=["Ethnic Integration Policy (EIP) and Singapore Permanent Resident Quota"]),
    "C3-05": dict(fact_ids=["eip_buyback_criteria", "eip_buyback_marketing_period"],
                  key_clauses=[["Minimum Occupation Period"], ["10 years"], ["six months"]],
                  target_sections=["More Support for Flat Owners Constrained by the Ethnic Integration Policy"],
                  note="Source is HDB's 2022 announcement of the EIP buyback; the conditions may have changed since."),
}


def official_cases():
    """The cases with their official ground truth (the legacy truth kept beside it)."""
    cases = []
    for case in CATEGORY_C_CASES:
        case = json.loads(json.dumps(case))
        truth = OFFICIAL_TRUTH[case["id"]]
        case["legacy_ground_truth"] = case.pop("ground_truth_facts")
        case["ground_truth_facts"] = {"source": "evaluation/policy_fact_sheet.yaml", "answerable": True, **truth}
        cases.append(case)
    return cases


def build():
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(official_cases(), f, indent=2, ensure_ascii=False)
    print(f"Generated {len(CATEGORY_C_CASES)} Category C queries -> {OUTPUT_JSON}")


if __name__ == "__main__":
    build()
