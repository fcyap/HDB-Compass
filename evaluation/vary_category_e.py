"""
Varied phrasings of the 15 Category E questions (evaluation/vary_category_e.py). Member 4.

The grounded questions (ground_category_e.py) all use one careful template, so a perfect score on them says little
about how the assistant copes with real users. This rewrites every question in four styles a real user might write,
from the same spec, so each rewrite states exactly the same facts (or leaves out the same one) and keeps the stored
ground truth and follow-up of the original:

  casual     lower-case chat with contractions, "k" amounts and "yrs"
  shorthand  Singlish shorthand: "SC couple", "4rm", "AMK", "PHG", "≤400m"
  reordered  search first, facts second, money as "$0.75m" / "650 thousand"
  bullets    a structured list

    .venv/bin/python -m evaluation.vary_category_e        # writes evaluation/category_e_varied.json (60 questions)
    LLM_PROFILE=vllm .venv/bin/python evaluation/run_category_e_prelim.py --dataset varied --tag varied1 --variants C
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ground_category_e import HOUSEHOLDS, SPECS  # noqa: E402

SOURCE = PROJECT_ROOT / "evaluation" / "category_e_queries.json"
OUT = PROJECT_ROOT / "evaluation" / "category_e_varied.json"

TOWN_SHORT = {"ANG MO KIO": "AMK", "TOA PAYOH": "TPY", "JURONG EAST": "JE"}   # only abbreviations people really use
WHAT_AMENITY = {"mrt": "MRT", "hawker": "hawker centre", "supermarket": "supermarket"}


def town(spec, short=False) -> str:
    return TOWN_SHORT.get(spec["town"], spec["town"].title()) if short else spec["town"].title()


def k(n: int) -> str:
    return f"{n / 1000:g}k" if n % 100 == 0 else f"{n:,}"   # 8500 -> "8.5k", 650000 -> "650k"


def millions(n: int) -> str:
    return f"${n / 1_000_000:g}m" if n >= 500_000 else f"{n // 1000} thousand"


def rooms(spec, style: str) -> str:
    n = spec["ft"].split()[0]
    return {"casual": f"{n}-room", "shorthand": f"{n}rm", "reordered": f"{n}-room", "bullets": f"{n}-room"}[style]


def household(spec, style: str) -> Dict[str, str]:
    """Pieces of the household description in each style; empty when the spec has no household (search-only)."""
    if "hh" not in spec:
        return {}
    h, omit = HOUSEHOLDS[spec["hh"]], set(spec.get("omit", []))
    old, young = h["ages"]
    single, pr = h["pair"] == "SC_Single", h["pair"] == "SC_PR"
    ages = "" if "ages" in omit else (f"{old}" if single else f"{old} and {young}")
    if style == "casual":
        who = ("i'm single, a singapore citizen" + (f", {ages}" if ages else "") + ", first timer, no kids") if single else \
              ("i'm a singapore citizen and my husband is a PR" if pr else "me and my wife are both singapore citizens") + \
              (f", we're {ages}" if ages else "") + ", first timers, no kids"
    elif style == "shorthand":
        who = ("single SC" + (f" aged {ages}" if ages else "") + ", 1st timer, no kids") if single else \
              ("SC + PR couple" if pr else "SC couple") + (f" aged {ages.replace(' and ', ' & ')}" if ages else "") + \
              ", both 1st timers, no kids"
    else:
        who = ("a single Singapore citizen" + (f" aged {ages}" if ages else "") + ", first-time buyer, no children") if single \
            else ("a married couple, one Singapore citizen and one permanent resident" if pr
                  else "a married Singapore citizen couple") + (f" aged {ages}" if ages else "") + \
            ", both first-timers with no children"
    cash = None if "cash" in omit else h["cash"]
    return dict(who=who, income=h["income"], cpf=h["cpf"], cash=cash, single=single)


def search_parts(spec, style: str) -> Dict[str, str]:
    radii = spec["radii"]
    if style == "shorthand":
        near = ", ".join(f"≤{m}m {WHAT_AMENITY[a]}" for a, m in radii.items())
    elif style == "casual":
        near = " and ".join(f"within {m}m of {'an mrt' if a == 'mrt' else 'a ' + WHAT_AMENITY[a]}" for a, m in radii.items())
    else:
        near = " and ".join(f"no more than {m} metres from {'an MRT station' if a == 'mrt' else 'a ' + WHAT_AMENITY[a]}"
                            for a, m in radii.items())
    return dict(near=near)


def casual(spec) -> str:
    s, hh = search_parts(spec, "casual"), household(spec, "casual")
    search = (f"looking for a resale {rooms(spec, 'casual')} in {town(spec).lower()} under {k(spec['budget'])}, at least "
              f"{spec['lease']} yrs lease left, {s['near']}")
    if not hh:
        return f"hi! any {search[len('looking for a '):]}? if nothing fits, what should we relax?"
    money = f"household income {k(hh['income'])} a month (HFE assessed), {k(hh['cpf'])} in cpf oa we can use"
    money += f", plus {k(hh['cash'])} cash" if hh["cash"] else ""
    return (f"hey, {hh['who']}. {money}. no other loans, been working continuously for the last 12 months, and our "
            f"grant and hdb loan eligibility is already confirmed. no proximity grant. {search}. use {spec['lease']} "
            f"yrs remaining lease for the calc. what grants do we get and what's our max budget? show them on the map pls")


def shorthand(spec) -> str:
    s, hh = search_parts(spec, "shorthand"), household(spec, "shorthand")
    want = (f"{rooms(spec, 'shorthand')} resale {town(spec, short=True)} <{k(spec['budget'])}, lease {spec['lease']}+ yrs, "
            f"{s['near']}")
    if not hh:
        return f"{want}. got any? if none, which one to relax"
    money = f"HFE income {k(hh['income'])}/mth, CPF OA {k(hh['cpf'])}" + (f", cash {k(hh['cash'])}" if hh["cash"] else "")
    return (f"{hh['who']}. {money}, no other debts, employed 12 mths continuously, grant + HDB loan eligibility "
            f"confirmed, no PHG. want {want}. calc w {spec['lease']} yrs lease. grants & max budget? map pls")


def reordered(spec) -> str:
    s, hh = search_parts(spec, "reordered"), household(spec, "reordered")
    search = (f"I'm searching for {rooms(spec, 'reordered')} resale flats in {town(spec)}: budget up to "
              f"{millions(spec['budget'])}, a remaining lease of {spec['lease']} years or more, {s['near']}.")
    if not hh:
        return search + " If none qualify, tell me the smallest change that would give me options."
    i_or_we = "I am" if hh["single"] else "We are"
    money = (f"Monthly HFE-assessed income is ${hh['income']:,}; CPF OA available is ${hh['cpf']:,}"
             + (f"; cash savings are ${hh['cash']:,}" if hh["cash"] else "") + ".")
    return (f"{search} About {'me' if hh['single'] else 'us'}: {i_or_we} {hh['who']}. {money} There are no other "
            f"debts, {'I have' if hh['single'] else 'we have'} been in continuous employment for the past 12 months, "
            f"grant and HDB loan eligibility have been confirmed, and no proximity grant applies. Please work out the "
            f"grants and maximum budget using a {spec['lease']}-year remaining lease, and map the matches.")


def bullets(spec) -> str:
    hh = household(spec, "bullets")
    near = [f"- within {m}m of {'an MRT station' if a == 'mrt' else 'a ' + WHAT_AMENITY[a]}" for a, m in spec["radii"].items()]
    looking = [f"- {rooms(spec, 'bullets')} resale in {town(spec)}", f"- under ${spec['budget']:,}",
               f"- {spec['lease']}+ years of lease left", *near]
    if not hh:
        return "Looking for:\n" + "\n".join(looking) + "\nAnything? If not, what's the best single thing to relax?"
    facts = [f"- {hh['who']}", f"- HFE income: ${hh['income']:,}/month", f"- CPF OA: ${hh['cpf']:,}"]
    facts += [f"- cash: ${hh['cash']:,}"] if hh["cash"] else []
    facts += ["- no other debts; continuously employed for 12 months",
              "- grant and HDB loan eligibility confirmed; no proximity grant"]
    return ("Household:\n" + "\n".join(facts) + "\nLooking for:\n" + "\n".join(looking)
            + f"\nAssume {spec['lease']} years of remaining lease for the calculation. What are the grants and max budget? "
              f"Map the matches.")


STYLES: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "casual": casual, "shorthand": shorthand, "reordered": reordered, "bullets": bullets}

# the follow-up reply to the assistant's question, per style
FOLLOW_UP = {
    "cash": {"casual": "we've got 25k in cash", "shorthand": "cash 25k", "reordered": "Our cash savings come to $25,000.",
             "bullets": "- cash: $25,000"},
    "ages": {"casual": "we're 32 and 30", "shorthand": "aged 32 & 30", "reordered": "We are aged 32 and 30.",
             "bullets": "- ages: 32 and 30"},
}


def build() -> List[Dict[str, Any]]:
    originals = {i["id"]: i for i in json.loads(SOURCE.read_text())}
    out: List[Dict[str, Any]] = []
    for spec in SPECS:
        base = originals[spec["id"]]
        for style, write in STYLES.items():
            item = copy.deepcopy(base)
            item["id"] = f"{spec['id']}~{style}"
            item["style"] = style
            item["original_id"] = spec["id"]
            item["query"] = write(spec)
            if item.get("follow_up"):
                item["follow_up"]["answer"] = FOLLOW_UP[spec["omit"][0]][style]
            out.append(item)
    return out


if __name__ == "__main__":
    items = build()
    OUT.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} varied questions to {OUT.relative_to(PROJECT_ROOT)}")
    for i in items[:4]:
        print(f"\n[{i['id']}]\n{i['query']}")
