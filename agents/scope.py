"""Scope check: is this message about Singapore HDB housing at all, and is it something the app can do?

Pure code, no model. It runs before any extraction so that "hi", "what's the weather" or "ignore your instructions"
get a short fixed reply instead of a search over HDB data. It is the stand-in for Member 5's input guardrail: the
supervisor calls it only while that hook is not installed (docs/INTERFACES.md section 10).

The rule is generous on purpose: one housing word anywhere keeps a message in scope, unless it is clearly about
private property, a price forecast or an attempt to change the instructions. A wrongly blocked housing question costs
more than a wrongly accepted stray one (that one just gets the menu).
"""
import re
from typing import Dict, List, Optional

GREETING = "greeting"
THANKS = "thanks"
OFF_TOPIC = "off_topic"
PRIVATE = "private_property"
FORECAST = "forecast"
INJECTION = "injection"
IN_SCOPE = "in_scope"

_GREETING = re.compile(r"^(?:hi|hello|hey|heya|hiya|yo|howdy|greetings|good (?:morning|afternoon|evening)|sup|hi there|hello there|hey there)"
                       r"(?:\s+(?:there|again|compass|hdb[- ]?compass|everyone|all))?$", re.I)
_THANKS = re.compile(r"^(?:(?:ok(?:ay)?|great|cool|nice|perfect|awesome)[, ]+)?(?:thanks|thank you|thank u|thx|ty|cheers|tq|tks)"
                     r"(?:\s+(?:so much|a lot|very much|a million|again|lah|ah))?$", re.I)

_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\b\W+(?:\w+\W+){0,3}?(?:instructions?|rules?|prompts?|guidelines?|programming|restrictions?)\b"
    r"|\b(?:system|developer|hidden|initial) (?:prompt|message|instructions?)\b"
    r"|\b(?:reveal|show|print|repeat|leak|tell me)\b\W+(?:\w+\W+){0,3}?(?:prompt|instructions)\b"
    r"|\bjail ?break\b|\bDAN mode\b|\bdeveloper mode\b|\bpretend (?:you are|to be|you're)\b|\byou are now\b|\bact as (?:an? )?(?!buyer|couple)",
    re.I)

_PRIVATE = re.compile(r"\b(?:condos?|condominiums?|ecs?|executive condominiums?|landed|bungalows?|semi-?detached|semi-?d|terrace houses?|"
                      r"private (?:property|properties|condos?|apartments?|housing|residential|homes?)|shophouses?)\b", re.I)
_HDB_WORDS = re.compile(r"\b(?:hdbs?|btos?|sbf|resale|flats?|maisonettes?|sers|dbss)\b", re.I)

_FORECAST = re.compile(
    r"\b(?:predict\w*|forecast\w*|project(?:ion|ed)s?|crystal ball)\b[^.?!]*\b(?:prices?|market|values?|resale|hdb|flats?|bto|rates?)\b"
    r"|\bwill\b[^.?!]*\b(?:prices?|market|values?|resale|hdb|flats?|housing)\b[^.?!]*\b(?:crash\w*|collapse\w*|rise|rising|fall|falling|drop\w*|"
    r"go (?:up|down)|increase|decrease|double|keep (?:rising|going|climbing)|bubble|burst)\b"
    r"|\b(?:crash|bubble)\b[^.?!]*\b(?:next year|soon|coming|this year|in 20\d\d)\b"
    r"|\b(?:prices?|market)\b[^.?!]*\b(?:next year|in the next \d+ years?|years from now|in 20(?:2[7-9]|[3-9]\d))\b[^.?!]*\b(?:be|go|rise|fall|crash)\b",
    re.I)

_HOUSING = re.compile(
    r"\b(?:hdbs?|btos?|sbf|resale|sales?|sold|transactions?|comparables?|coordinates?|latitude|longitude|flats?|rooms?|rm|executive|maisonettes?|studio|blocks?|blks?|storey|floor|unit|units|apartments?|"
    r"houses?|housing|homes?|propert(?:y|ies)|estate|neighbou?rhoods?|towns?|region|"
    r"mrt|lrt|bus|train|station|school|schools|primary|hawker|amenit\w*|supermarket|clinic|park|mall|commute|"
    r"grants?|cpf|oa|loan|loans|mortgage|hfe|mop|lease|price|prices|priced|cost|costs|afford\w*|budget|income|salary|earn|cash|savings|"
    r"down ?payment|upfront|stamp duty|bsd|absd|valuation|cov|"
    r"buyers?|first[- ]?times?|pr|citizens?|singaporeans?|permanent residents?|foreigners?|singles?|couples?|married|divorce\w*|widow\w*|fianc\w*|first[- ]?timers?|"
    r"second[- ]?timers?|eligib\w*|scheme\w*|quota\w*|ethnic\w*|prime|plus|standard|rent|renting|rental|sublet|landlord|tenant|"
    r"sers|upgrading|resettlement|sell(?:ing)?|buy(?:ing)?|purchase|apply(?:ing)?|launch(?:es)?|ballot|queue|postal)\b"
    r"|\b\d[- ]?(?:room|rm)\b|\b\d{6}\b", re.I)


def _town_pattern() -> "re.Pattern[str]":
    from agents import supervisor as sv   # imported late: the supervisor imports this module
    names = list(sv.HDB_TOWNS) + list(sv.TOWN_ALIASES)
    return re.compile("|".join(re.escape(n).replace(r"\ ", r"\s+") for n in names), re.I)


_TOWNS: Optional["re.Pattern[str]"] = None


def _has_housing_word(text: str) -> bool:
    global _TOWNS
    if _HOUSING.search(text):
        return True
    if _TOWNS is None:
        _TOWNS = _town_pattern()
    return bool(_TOWNS.search(text))


def classify(text: str, mid_conversation: bool = False) -> str:
    """One of the kind constants above. mid_conversation: there is an open question or earlier turns, so a short reply
    such as "4 room" or "yes" or "$8,000" is an answer, not an off-topic message."""
    t = (text or "").strip()
    bare = re.sub(r"[^\w\s'-]", "", t).strip()
    if _GREETING.match(bare):
        return GREETING
    if _THANKS.match(bare):
        return THANKS
    if _INJECTION.search(t):
        return INJECTION
    if _FORECAST.search(t):
        return FORECAST
    if _PRIVATE.search(t) and not _HDB_WORDS.search(t):
        return PRIVATE
    if mid_conversation or _has_housing_word(t):
        return IN_SCOPE
    return OFF_TOPIC


CAPABILITIES: List[str] = [
    "Find resale flats by budget, town and distance to MRT, schools or amenities",
    "Check resale prices and how they have moved",
    "Compare a BTO launch with nearby resale prices",
    "Estimate your grants and budget",
    "Explain HDB rules (MOP, eligibility, Prime/Plus)",
    "Find schools and amenities near an address",
]


def _capabilities() -> str:
    return "\n".join(f"- {c}" for c in CAPABILITIES)


REPLIES: Dict[str, str] = {
    GREETING: "Hi! I'm HDB-Compass, an assistant for Singapore HDB public housing. I can:\n\n" + _capabilities()
              + "\n\nWhat would you like to know?",
    THANKS: "You're welcome! I'm HDB-Compass; if you have more HDB questions, I can:\n\n" + _capabilities(),
    OFF_TOPIC: "I can only help with Singapore HDB public housing. I can:\n\n" + _capabilities(),
    PRIVATE: "I cover HDB flats only; condos, ECs and landed homes aren't in my data. For HDB flats I can:\n\n" + _capabilities(),
    FORECAST: "I can't predict future prices. I can show how they have moved, e.g. 'How have 4-room prices in Tampines "
              "changed over the last 5 years?'",
    INJECTION: "I can't change or reveal my instructions. Ask me about HDB housing.",
}


def scope_reply(text: str, mid_conversation: bool = False) -> Optional[str]:
    """The fixed reply for a message that is not an HDB request, or None when the message goes on to the agents."""
    return REPLIES.get(classify(text, mid_conversation))


# ---------------------------------------------------------------------------
# Menu: an in-scope message that names nothing the tools can act on ("what flat should I buy?")
# ---------------------------------------------------------------------------
_OPTIONS = {
    "launch": "Compare a BTO launch with nearby resale prices",
    "budget": "Estimate your grants and budget",
    "rules": "Explain BTO rules (MOP, Prime/Plus, eligibility)",
    "towns": "Compare towns by resale price",
    "find": "Find resale flats by budget, town and distance",
    "resale_rules": "Explain HDB rules (MOP, eligibility, Prime/Plus)",
}
_MENUS = {
    "bto": ["launch", "budget", "rules", "towns"],
    "resale": ["find", "towns", "budget", "resale_rules"],
    "general": ["find", "launch", "budget", "resale_rules"],
}
# words that pick an option when the reply is short; the option number and the exact label always work
_OPTION_WORDS = {
    "launch": r"\blaunch\w*|\bcompar\w*|\bnearby\b|\bproject\b",
    "budget": r"\bgrants?\b|\bbudget\b|\bafford\w*|\bestimate\b|\bloan\b|\bcpf\b",
    "rules": r"\brules?\b|\bmop\b|\bprime\b|\bplus\b|\beligib\w*|\bexplain\b",
    "resale_rules": r"\brules?\b|\bmop\b|\bprime\b|\bplus\b|\beligib\w*|\bexplain\b",
    "towns": r"\btowns?\b|\bcompar\w*|\bcheaper\b|\bresale prices?\b",
    "find": r"\bfind\b|\bsearch\b|\bflats?\b|\bresale\b",
}


# Words that name an HDB rule. A model sometimes files a vague question ("Should I buy BTO or resale?") as a policy
# question; without one of these the policy search only returns loosely related passages, so the menu is asked instead.
_POLICY_TOPIC = re.compile(
    r"\b(?:mop|minimum occupation|occupy|occupation|eligib\w*|qualif\w*|grants?|ehg|phg|subsid\w*|schemes?|quotas?|eip|"
    r"ethnic\w*|spr|prs?|permanent residents?|citizens?|citizenship|foreigners?|singles?|unmarried|married|fianc\w*|"
    r"divorc\w*|widow\w*|second[- ]?timers?|first[- ]?timers?|income (?:ceiling|cap|limit)s?|wait[- ]?out|"
    r"rent(?:ing)? out|sublet\w*|prime|plus|standard flats?|clawback|resale levy|levy|hfe|"
    r"loan|ltv|msr|tdsr|cpf|ownership|own(?:ing)? (?:a |another |private )|private propert\w*|rules?|allowed|"
    r"permitted|policy|policies|restrict\w*|requirements?|criteria|age limit|how (?:long|many times))\b", re.I)


# A question about figures in the resale data that no specific intent recognised: the market agent chooses how to
# answer it with the analysis tools (agents/supervisor.py, "analysis mode"). A message without these words gets the menu.
_DATA_WORDS = re.compile(
    r"\b(?:prices?|priced|pricey|sold|sales?|transactions?|median|statistics|stats|trends?|expensive|cheap\w*|premium|"
    r"compar\w+|breakdown|listings?|average|how much|per sqm|psf|distribution|highest|lowest|costs?|costliest|dearest|"
    r"priciest|affordable|histor\w+|gone up|gone down|appreciat\w+|cheaper|pricier)\b", re.I)


def mentions_data(text: str) -> bool:
    return bool(_DATA_WORDS.search(text or ""))


def mentions_policy_topic(text: str) -> bool:
    return bool(_POLICY_TOPIC.search(text or ""))


def menu_variant(query: str) -> str:
    bto = bool(re.search(r"\bbtos?\b", query, re.I))
    resale = bool(re.search(r"\bresale\b", query, re.I))
    return "bto" if bto and not resale else "resale" if resale and not bto else "general"


def menu_for(query: str) -> Dict[str, object]:
    """The numbered menu for a message with nothing to act on: {"question", "options": [{"key", "label", "text"}]}.
    "text" is what a clickable option sends back, and it is the label itself, so it reads as the choice."""
    variant = menu_variant(query)
    options = [{"key": k, "label": _OPTIONS[k], "text": _OPTIONS[k]} for k in _MENUS[variant]]
    intro = {"bto": "Happy to help with a BTO. What would you like to do?",
             "resale": "Happy to help with a resale flat. What would you like to do?",
             "general": "Happy to help you look at HDB flats. What would you like to do?"}[variant]
    question = intro + "\n\n" + "\n".join(f"{i}. {o['label']}" for i, o in enumerate(options, 1)) + \
        "\n\nReply with a number, or ask your own question."
    return {"question": question, "options": options}


def bto_projects() -> List[str]:
    try:
        from mcp_server.bto_tools import load_launches
        return list(dict.fromkeys(r["project"] for r in load_launches()))
    except Exception:  # noqa: BLE001 — the launch data is optional
        return []


def match_launch(text: str) -> Optional[str]:
    """The BTO launch a reply names: the full name anywhere in it, or a unique part of a name ("Portico")."""
    projects = bto_projects()
    t = " ".join(text.upper().split())
    full = [p for p in projects if p.upper() in t]
    if full:
        return max(full, key=len)
    part = re.sub(r"[^A-Z0-9 ]", "", t).strip()
    hits = [p for p in projects if len(part) >= 4 and part in p.upper()]
    return hits[0] if len(hits) == 1 else None


def _followup(key: str) -> Dict[str, object]:
    if key == "launch":
        names = bto_projects()
        listed = ", ".join(names) if names else "none loaded"
        # one chip per launch; its text is a complete question, so a click runs the comparison on its own
        options = [{"label": n, "text": f"How does {n} compare with resale prices?"} for n in names]
        return {"kind": "bto_launch", "needed": [], "question": f"Which launch? I have data for: {listed}.",
                "options": options}
    if key == "budget":
        return {"kind": "financial",
                "needed": ["citizenship_pair", "first_timer_status", "monthly_income", "youngest_buyer_age", "oldest_buyer_age"],
                "question": "To estimate your grants and budget, please tell me each buyer's citizenship, whether you are "
                            "first-timers, your HFE-assessed monthly household income and the age of each buyer."}
    if key in ("rules", "resale_rules"):
        return {"kind": "rules", "needed": [],
                "question": "Which rule? For example: the MOP for a Prime flat, who can apply for a BTO, how much grant a "
                            "first-timer gets, or how many times you can apply."}
    if key == "towns":
        return {"kind": "towns", "needed": [],
                "question": "Which towns and flat type? For example: 'Is Woodlands cheaper than Yishun for 4-room flats?'"}
    return {"kind": "search", "needed": [],
            "question": "Tell me the town, flat type and budget, for example: '4-room flats in Tampines under $600k "
                        "within 500m of an MRT'."}


def _picked_option(options: List[Dict[str, str]], text: str) -> Optional[str]:
    t = text.strip()
    m = re.fullmatch(r"(?:option\s*|no\.?\s*|number\s*|#)?(\d)[.)]?", t, re.I)
    if m and 1 <= int(m.group(1)) <= len(options):
        return options[int(m.group(1)) - 1]["key"]
    norm = " ".join(t.lower().rstrip(".!").split())
    for o in options:
        if norm == o["label"].lower():
            return o["key"]
    words = t.split()
    if len(words) <= 4 and not t.endswith("?") and not re.match(r"(?:what|which|how|can|is|are|do|does|will|who|where)\b", t, re.I):
        for o in options:
            if re.search(_OPTION_WORDS[o["key"]], t, re.I):
                return o["key"]
    return None


def resolve_menu_reply(pending: Optional[Dict[str, object]], text: str) -> Optional[Dict[str, object]]:
    """What a reply to a menu (or to a question a menu option asked) means, or None when it is just a new question.
    {"ask": clarification}: the option's own follow-up question, answered at once without searching.
    {"query": text}: the reply completed into a full question the normal pipeline can answer."""
    kind = (pending or {}).get("kind")
    if kind == "menu":
        key = _picked_option(list((pending or {}).get("options") or []), text)
        return {"ask": _followup(key)} if key else None
    if kind == "bto_launch":
        project = match_launch(text)
        return {"query": f"How does {project} compare with nearby resale prices?"} if project else None
    if kind == "rules" and len(text.split()) <= 8:
        return {"query": f"HDB rules: {text.strip()}"}
    if kind == "towns":
        towns = _towns_in(text)
        ft = re.search(r"\b([2-5])[- ]?(?:room|rm)\b", text, re.I)
        flat = f"{ft.group(1)}-room" if ft else None
        if len(towns) >= 2:
            return {"query": f"Is {towns[0].title()} cheaper than {towns[1].title()} for {flat or '4-room'} flats?"}
        if len(towns) == 1:
            return {"query": f"How much is a {flat or '4-room'} flat in {towns[0].title()}?"}
    return None


def _towns_in(text: str) -> List[str]:
    from agents import supervisor as sv
    up = text.upper()
    found = sorted((up.find(t), t) for t in sv.HDB_TOWNS if t in up)
    return [t for _, t in found]


# ---------------------------------------------------------------------------
# Asks the tools cannot do yet
# ---------------------------------------------------------------------------
_COMMUTE_TIME_ASK = re.compile(
    r"\b(?:how long|how far|how much time|how many minutes|travel(?:l?ing)? time|journey time|commute time)\b[^.?!]*"
    r"\b(?:commute|get to|getting to|travel|reach|journey|take|from\b[^.?!]*\bto|to work)\b"
    r"|\bcommute (?:from|to)\b[^.?!]*\b(?:to|from)\b|\bhow (?:long|far) (?:is|does) (?:the |my )?commute\b", re.I)
_SEARCH_WORD = re.compile(r"\b(?:find|show|list|search|recommend|suggest|under|within|below|at most|max(?:imum)?)\b", re.I)
SINGLE_COMMUTE_REPLY = ("I can't estimate a single commute yet: the commute check uses town-level averages and does not "
                        "route to your workplace. I can compare two workplaces' commutes for flats in a town, or find "
                        "flats near an MRT station.")


def unsupported_reply(text: str, destinations: int) -> Optional[str]:
    """A fixed reply for an ask the tools cannot do, or None. destinations: how many workplaces the message names (the
    commute check needs two). A search with limits ("find flats ... under 40 min") is left to the agents."""
    if destinations < 2 and _COMMUTE_TIME_ASK.search(text) and not _SEARCH_WORD.search(text):
        return SINGLE_COMMUTE_REPLY
    return None
