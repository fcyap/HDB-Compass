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
    r"pr|citizens?|singaporeans?|permanent residents?|foreigners?|singles?|couples?|married|divorce\w*|widow\w*|fianc\w*|first[- ]?timers?|"
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
