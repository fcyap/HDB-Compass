"""
Research intents for the supervisor (agents/research_intents.py). Owned by Member 1.

The original graph answered one question shape: "blocks matching price, lease and
distance limits". This module lets the same graph answer research questions with
the tools in mcp_server/market_research_tools.py, bto_tools.py and location_tools.py:

  extract_research_fields   regex extraction of addresses, asking price, intents and
                            stated assumptions (also the safety net under LLM extraction)
  research_plan             deterministic tool calls per specialist for those intents
  is_block_search           whether the classic block search should also run
  research_lines            template answer sections for the research tool outputs
  research_map_payload      map markers for resolved places and compared options
  compact_tool_result       valid, size-bounded JSON for ToolMessages
  unverified_coordinates    rejects model-supplied coordinates no tool produced
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_FLAT_TYPE = "4 ROOM"
VAGUE_RADIUS_METERS = 500
MARKET_RESEARCH_TOOLS = ("get_block_transaction_history", "get_market_snapshot", "rank_towns_by_value",
                         "compare_resale_options", "assess_resale_price", "compare_bto_launch_vs_resale",
                         "find_resale_blocks_near", "get_area_resale_statistics", "get_area_price_trend",
                         "query_resale_statistics", "describe_resale_data")
LOCATION_TOOLS = ("resolve_location", "get_nearby_amenities")
COORDINATE_TOOLS = {"query_comparable_resale_transactions": ("latitude", "longitude"),
                    "get_nearby_amenities": ("latitude", "longitude"),
                    "find_resale_blocks_near": ("latitude", "longitude")}
# Tools whose coordinates sit inside an `area` argument.
AREA_TOOLS = ("get_area_resale_statistics", "get_area_price_trend", "get_block_transaction_history")

_STREET = (r"(?:st|street|ave|avenue|rd|road|dr|drive|cres|crescent|ctrl|central|lane|ln|cl|close|pl|place|way|"
           r"walk|view|link|loop|ring|rise|hts|heights|ter|terrace|nth|north|sth|south|lor|lorong|jln|jalan|"
           r"vale|grove|green|garden|gdns|circle|cir|hill|pk|park|wk|ctr|centre|square|sq)")
_STOP = (r"(?=\s*(?:[,;?!()]|\.(?:\s|$)|$|\s(?:vs\.?|versus|or|and|for|is|are|was|with|near|compared|against|in|at|"
         r"that|which|under|below|to|a|an|the|good|worth|sold|flat|flats|unit|units|price|prices|cost|fair)\b))")
_BLK_STREET = re.compile(r"\b(?:blk|block)\s*(\d{1,4}[a-z]?\s+(?:[a-z]+\s){0,3}?" + _STREET + r"(?:\s+\d{1,3}[a-z]?)?)\b", re.I)
_BLK_ADDRESS = re.compile(r"\b(?:blk|block)\s*(\d{1,4}[a-z]?(?:\s+[a-z0-9']+){1,3}?)" + _STOP, re.I)
_BARE_ADDRESS = re.compile(
    r"(?<![\w$-])(\d{1,4}[a-z]?)\s+(?!(?:room|rm|years?|yrs?|mins?|minutes|km|m|sqm|storey|bedroom)\b)"
    r"((?:[a-z]+\s){0,3}?" + _STREET + r"(?:\s+\d{1,3}[a-z]?)?)\b", re.I)
_POSTAL = re.compile(r"(?<![\d$,])(\d{6})(?![\d,])")
_PLACE = re.compile(r"\b(?:near|around|close to|beside|next to|from)\s+((?:[a-z][a-z'’-]*\s){0,4}?"
                    r"(?:mrt(?:\s+station)?|lrt(?:\s+station)?|station|interchange|hawker centre|food centre|market|"
                    r"primary school|school))\b", re.I)

_RX = {
    "history": r"\b(histor(?:y|ical)|past (?:sales|transactions|prices)|sold for|sell for|go(?:ne)? for|"
               r"previous(?:ly)? sold|recent sales|"
               r"transactions? (?:at|for|in)|transacted)\b",
    "assessment": r"\b(fair(?:ly)?(?: price)?|reasonable|over-?priced|under-?priced|good (?:price|deal)|worth|"
                  r"valuation|too (?:much|high|expensive)|asking|offer(?:ed)?|is \$\s?\d)",
    "comparison": r"\b(vs\.?|versus|compare|comparison|compared|which is better|better (?:choice|option|buy))\b|\bor\b",
    "nearby": r"\b(what(?:'s|s| is| are)?\s+(?:near(?:by)?|around|close to)|nearby (?:amenities|facilities)|"
              r"amenities|facilities|what is around|near it|around it)\b",
    "snapshot": r"\b(market (?:now|today|currently|like|status|update|conditions?|overview|snapshot)|current(?:ly)? market|"
                r"state of the (?:resale )?market|where (?:is|are) (?:the )?(?:resale )?(?:market|prices)|"
                r"latest (?:resale )?prices|how (?:is|are) (?:the )?(?:(?:resale|hdb|property)\s+){0,2}(?:market|prices)|"
                r"prices? (?:now|today|currently))\b",
    "ranking": r"\b(which towns?|what towns?|best[- ]value|cheapest towns?|most affordable towns?|"
               r"rank(?:ing)? (?:the |all )?towns|towns? (?:to|should (?:i|we)) (?:buy|look|consider)|"
               r"where (?:should|can|could) (?:i|we) (?:buy|afford|look))\b",
    "search": r"\b(find|show|list|search|look(?:ing)? for|recommend|suggest|which blocks|map (?:the )?matches)\b",
    # "around Blk 277", "within 300m of Toh Guan MRT", "nearby flats": a question about an area, not one block.
    "area": r"\b(around|near(?:by)?|close to|within \d+(?:\.\d+)?\s*k?m\b|vicinity|surrounding|neighbou?r(?:ing|hood)|"
            r"in the area|same (?:area|estate|street))",
    "stats": r"\b(prices?|priced|sold|sell(?:s|ing)? for|sales|transactions?|transacted|statistics|stats|median|psf|"
             r"psm|per sqm|costs?|valu(?:e|es|ation)|trends?|gone up|appreciat\w*|resale market)\b",
    "area_trend": r"\b(trends?|appreciat\w*|over (?:the )?(?:last|past)|changed|moved|risen|fallen|gone (?:up|down)|"
                  r"increas\w*|grow\w*|year[- ]on[- ]year)\b",
}
# "How much is a 4-room in Tampines?", "median price of 5-room flats in Punggol", "is Woodlands cheaper than Yishun":
# a question about what flats cost in a town, answered with the town's price figures rather than a list of blocks.
_PRICE_QUESTION = re.compile(
    r"\bhow much (?:is|are|does|do|would|will|did|might)\b|\b(?:median|average|avg|typical|mean|going|current) (?:resale )?(?:prices?|rates?)\b"
    r"|\bprices? (?:of|for)\b|\bwhat(?:'s|s| is| are) the (?:resale )?prices?\b|\bhow (?:much|expensive)\b"
    r"|\b(?:cheaper|pricier|dearer|costlier|more expensive|less expensive|more affordable)\b", re.I)
_AREA_RADIUS = re.compile(r"\b(?:within|in)\s+(?:a\s+)?(\d+(?:\.\d+)?)\s*(km|m)\b(?:\s+radius)?"
                          r"(?!\s+(?:of|from)\s+(?:an?\s+|the\s+)?(?:\w+\s+){0,2}?"
                          r"(?:mrt|lrt|station|hawker|food centre|supermarket|school|bus stop)s?\b)", re.I)
_GENERIC_SCHOOL_WORDS = {"good", "top", "popular", "best", "reputable", "nearby", "branded", "any", "a", "the", "some",
                         "great", "elite", "famous", "decent", "nice", "local", "primary", "school", "schools", "sap"}


SCHOOL_CLARIFICATION = ("Which primary school do you have in mind? I can check blocks within 1 km or 1-2 km of a "
                        "named school; school quality is not something the data ranks.")


def _has(kind: str, text: str) -> bool:
    return bool(re.search(_RX[kind], text, re.I))


def _money(text: str) -> Optional[float]:
    m = re.match(r"(\d[\d,]*(?:\.\d+)?)\s*(k|m)?", text.replace(" ", ""), re.I)
    if not m:
        return None
    unit = (m.group(2) or "").lower()
    return float(m.group(1).replace(",", "")) * (1000 if unit == "k" else 1_000_000 if unit == "m" else 1)


def extract_addresses(query: str) -> List[str]:
    """User-supplied block addresses, postal codes and named places, in order of mention."""
    found: List[Tuple[int, str]] = []
    taken: List[Tuple[int, int]] = []

    def add(start: int, end: int, text: str) -> None:
        if any(s < end and start < e for s, e in taken):
            return
        taken.append((start, end))
        found.append((start, " ".join(text.split())))

    for pattern in (_BLK_STREET, _BLK_ADDRESS):
        for m in pattern.finditer(query):
            add(m.start(), m.end(), "Blk " + m.group(1))
    for m in _BARE_ADDRESS.finditer(query):
        add(m.start(), m.end(), f"{m.group(1)} {m.group(2)}")
    for m in _POSTAL.finditer(query):
        add(m.start(), m.end(), m.group(1))
    for m in _PLACE.finditer(query):
        if not generic_amenity_phrase(m.group(1)):
            add(m.start(1), m.end(1), m.group(1))
    return [text for _, text in sorted(found)]


def known_bto_project(query: str) -> Optional[str]:
    try:
        from mcp_server.bto_tools import load_launches
        projects = {r["project"] for r in load_launches()}
    except Exception:  # noqa: BLE001 — the dataset is optional for extraction
        return None
    q = " ".join(query.upper().split())
    hits = [p for p in projects if p.upper() in q]
    return max(hits, key=len) if hits else None


# "near an MRT", "near MRT stations", "near the hawker centre": a kind of place, not a place. Only a phrase that
# adds a name ("Bishan MRT", "Toa Payoh hawker centre", "Ai Tong School") points at one.
_KIND_ONLY = {"mrt", "lrt", "station", "stations", "interchange", "hawker", "food", "centre", "centres", "market",
              "primary", "school", "schools", "supermarket", "supermarkets", "train", "bus"}
_DETERMINERS = {"a", "an", "the", "any", "some", "each", "every", "nearby", "nearest", "closest", "local", "my", "our"}


def generic_amenity_phrase(phrase: str) -> bool:
    words = [w for w in re.findall(r"[a-z]+", (phrase or "").lower()) if w not in _DETERMINERS]
    return all(w in _KIND_ONLY for w in words)


def generic_school_phrase(name: Optional[str]) -> bool:
    words = re.findall(r"[a-z]+", (name or "").lower())
    return bool(words) and all(w in _GENERIC_SCHOOL_WORDS for w in words)


def _place_intents(q: str, addresses: List[str], has_point: bool = False) -> Dict[str, Any]:
    """Intents that need a place: the user's addresses, or a map pin (has_point)."""
    blocks = [a for a in addresses if re.match(r"(?:blk\s+)?\d", a, re.I)]
    out: Dict[str, Any] = {"addresses": addresses}
    out["wants_block_history"] = bool(blocks) and _has("history", q)
    out["wants_price_assessment"] = bool(blocks) and _has("assessment", q)
    out["wants_option_comparison"] = len(blocks) >= 2 and (_has("comparison", q) or not _has("history", q))
    out["wants_nearby_amenities"] = bool(addresses or has_point) and _has("nearby", q)
    out["wants_area_statistics"] = bool(addresses or has_point) and _has("area", q) and _has("stats", q)
    m = _AREA_RADIUS.search(q)
    if out["wants_area_statistics"] and m:
        out["area_radius_meters"] = int(min(max(float(m.group(1)) * (1000 if m.group(2).lower() == "km" else 1), 50), 3000))
    if out["wants_price_assessment"]:
        for m in re.finditer(r"\$\s?(\d[\d,]*(?:\.\d+)?\s*[km]?)(?!\s*(?:/|per|a)\s*month)", q, re.I):
            out["asking_price_sgd"] = _money(m.group(1))
            break
    return out


# Phrases that point at the map pin rather than naming a place.
_PIN_REF = re.compile(r"\b(here|this (?:flat|block|unit|place|location|spot|area|address|one|point)|"
                      r"that (?:flat|block|unit|place|location|spot)|(?:the |my )?pin(?:ned)?(?: location| spot| block)?|"
                      r"(?:selected|marked|clicked) (?:block|location|place|point|spot)|around me)\b", re.I)


def apply_map_pin(c: Dict[str, Any], query: str, pin: Optional[Tuple[float, float]],
                  radius_meters: Optional[float] = None) -> None:
    """Use the user's map pin as the place when the question points at it, or asks a
    place question without naming one. A pin on an HDB block becomes that block's
    address (so price tools can use it); elsewhere it is a coordinate for nearby searches."""
    if not pin:
        return
    refers = bool(_PIN_REF.search(query))
    place_question = any(_has(k, query) for k in ("history", "assessment", "nearby")) or (
        _has("area", query) and _has("stats", query))
    addresses = list(c.get("addresses") or [])
    if not (refers or (place_question and not addresses)):
        return
    if radius_meters:
        c["pin_radius_meters"] = int(radius_meters)  # the radius the map's facility panel already shows
    from mcp_server.location_tools import nearest_block  # lazy: keeps agents importable without the index
    lat, lng = float(pin[0]), float(pin[1])
    block = nearest_block(lat, lng)
    assumptions = c.setdefault("assumptions", [])
    if block:
        label = f"Blk {block['block']} {block['street']}"
        addresses = [label] + [a for a in addresses if a.upper() != label.upper()]
        assumptions.append(f"Used your map pin: {label} ({block['distance_meters']:.0f} m from the pin).")
        c.update(_place_intents(query, addresses))
    else:
        c["map_pin"] = {"latitude": lat, "longitude": lng}
        note = f"Used your map pin at ({lat:.5f}, {lng:.5f})"
        if any(_has(k, query) for k in ("history", "assessment")):
            note += "; it is not on an HDB block, so pin a block to check its prices"
        assumptions.append(note + ".")
        c.update(_place_intents(query, addresses, has_point=True))


# ---------------------------------------------------------------------------
# Which place a question is about: the map pin, a place named now, or one from earlier turns
# ---------------------------------------------------------------------------
# Constraints that belong to one question. A new question starts without them; a reply to an open question
# (pending clarification) keeps them, since it completes that request. See supervisor._finalize_constraints.
TURN_ONLY_KEYS = frozenset({
    "addresses", "wants_block_history", "wants_market_snapshot", "wants_town_ranking", "wants_option_comparison",
    "wants_price_assessment", "wants_nearby_amenities", "wants_area_statistics", "asking_price_sgd",
    "bto_project_name", "area_radius_meters", "map_pin", "pin_radius_meters", "nearby_schools",
    "wants_group_stats", "stats_group_by", "stats_near", "stats_distance_m", "stats_bands", "stats_trend",
    "stats_lookback_years", "stats_since_year", "stats_sort", "stats_order", "stats_listings", "analysis_mode",
    "wants_data_catalog",
})
# Never carried from an earlier turn, not even into a reply: they are worked out afresh for every message.
ALWAYS_FRESH_KEYS = frozenset({"clarifications", "assumptions", "extraction_source", "requested_tools", "place_options"})

# Words pointing at the pin ("here", "this block") and words that could mean the pin or a place named earlier.
_NEAR_REF = re.compile(r"\b(here|this (?:flat|block|unit|place|location|spot|area|address|one|point)|"
                       r"(?:the |my )?pin(?:ned)?(?: location| spot| block)?|"
                       r"(?:selected|marked|clicked) (?:block|location|place|point|spot)|around me)\b", re.I)
_FAR_REF = re.compile(r"\bthat (?:flat|block|unit|place|location|spot|area|address)\b|"
                      r"\bthe same (?:place|area|block|location|address)\b|"
                      r"(?<!\bis )(?<!\bare )(?<!\bwas )(?<!\bwere )\bthere\b(?!\s*(?:is|are|was|were|'s|be|any|an?)\b)",
                      re.I)
_PICK_PIN = re.compile(r"\b(?:pin(?:ned)?|map|marker|marked|here|first|1st|option (?:1|one)|the first one)\b", re.I)
_PICK_EARLIER = re.compile(r"\b(?:earlier|before|previous(?:ly)?|mentioned|second|2nd|option (?:2|two)|"
                           r"the other one|the second one)\b", re.I)


_POINTER_ONLY = re.compile(
    r"(?:the |my |this |that |your )?(?:pin(?:ned)?(?: (?:location|spot|block|place|one))?|map(?: pin)?|marker|here|there|"
    r"this (?:flat|block|unit|place|location|spot|area|address|one|point)|that (?:flat|block|unit|place|location|spot|area|address|one)|"
    r"(?:the )?(?:earlier|previous|other|first|second) (?:one|place|block|address)|(?:the )?same (?:place|area|block|location|address)|"
    r"(?:selected|marked|clicked) (?:block|location|place|point|spot))", re.I)


def is_place_question(query: str) -> bool:
    """A question that needs a place to answer (its history, a price check, what is near it, prices around it)."""
    return any(_has(k, query) for k in ("history", "assessment", "nearby")) or (_has("area", query) and _has("stats", query))


def _pin_option(pin: Tuple[float, float]) -> Dict[str, Any]:
    from mcp_server.location_tools import nearest_block  # lazy: keeps agents importable without the index
    lat, lng = float(pin[0]), float(pin[1])
    block = nearest_block(lat, lng)
    label = f"Blk {block['block']} {block['street']}" if block else f"your map pin ({lat:.5f}, {lng:.5f})"
    return {"source": "pin", "label": label, "latitude": lat, "longitude": lng}


def location_question(options: List[Dict[str, Any]]) -> str:
    pin, earlier = options[0]["label"], options[1]["label"]
    return (f"Do you mean the place you pinned on the map (**{pin}**) or **{earlier}**, which you mentioned earlier? "
            "Reply 'the pin' or 'the earlier one'.")


def _use_earlier(c: Dict[str, Any], query: str, place: str, named: List[str]) -> None:
    addresses = named + [place] if place.upper() not in {a.upper() for a in named} else named
    c.update(_place_intents(query, addresses))
    c.setdefault("assumptions", []).append(f"Used {place}, which you mentioned earlier.")


def _ask_which(c: Dict[str, Any], query: str, pin: Tuple[float, float], place: str) -> None:
    # keep what the user wants to know (nearby amenities, past sales...): the reply only names the place, and these
    # carry into it with the open question (TURN_ONLY_KEYS are kept for a reply). Nothing runs until then.
    wanted = _place_intents(query, [place])
    c.update({k: v for k, v in wanted.items() if k != "addresses" and v})
    options = [_pin_option(pin), {"source": "conversation", "label": place}]
    c["place_options"] = options
    c["clarifications"] = list(dict.fromkeys((c.get("clarifications") or []) + [location_question(options)]))


def resolve_place(c: Dict[str, Any], query: str, pin: Optional[Tuple[float, float]], earlier: List[str],
                  pending: Optional[Dict[str, Any]] = None, radius_meters: Optional[float] = None) -> None:
    """Decide which place this message is about and record it in c (addresses / map_pin / assumptions), or ask.

    1. A reply to "the pin or the earlier place?" picks one; a newly named place answers it too.
    2. "here", "this block", "the pin" with a pin: the pin (a place named alongside it is kept as well).
    3. "there", "that block", "the same area": the pin and/or the earlier place; if both are possible, ask.
    4. A place named in this message, with no pointing word: that place; the pin is ignored.
    5. A place question naming no place: the pin if there is one, else the earlier place; if both exist and
       differ, ask. Every choice not stated by the user is written to assumptions."""
    # a model sometimes copies a pointing phrase ("the pin", "there", "the earlier one") into addresses: not a place
    named = [a for a in c.get("addresses") or [] if not _POINTER_ONLY.fullmatch(a.strip(" .?!'\""))]
    if named != list(c.get("addresses") or []):
        c["addresses"] = named
    earlier_place = next((a for a in earlier if a.upper() not in {n.upper() for n in named}), None)
    if pending and pending.get("kind") == "location":
        if named:
            return
        options = pending.get("options") or []
        pin_opt = next((o for o in options if o.get("source") == "pin"), None)
        earlier_opt = next((o for o in options if o.get("source") == "conversation"), None)
        q = query.lower()
        wants_earlier = earlier_opt and (earlier_opt["label"].lower() in q or
                                         (_PICK_EARLIER.search(query) and not _PICK_PIN.search(query)))
        if wants_earlier:
            c.update(_place_intents(query, [earlier_opt["label"]]))
        elif pin_opt:
            # the pin is chosen, or the reply was unclear: asked once, the pin is then used and said so
            if not _PICK_PIN.search(query):
                c.setdefault("assumptions", []).append(f"Used your map pin ({pin_opt['label']}).")
            apply_map_pin(c, "here " + query, (pin_opt["latitude"], pin_opt["longitude"]), radius_meters)
        return
    near, far = bool(_NEAR_REF.search(query)), bool(_FAR_REF.search(query))
    if pin and near:
        apply_map_pin(c, query, pin, radius_meters)
        return
    if far:
        if pin and earlier_place:
            if _pin_option(pin)["label"].upper() == earlier_place.upper():
                apply_map_pin(c, "here " + query, pin, radius_meters)
            else:
                _ask_which(c, query, pin, earlier_place)
        elif pin:
            apply_map_pin(c, "here " + query, pin, radius_meters)
        elif earlier_place:
            _use_earlier(c, query, earlier_place, named)
        return
    if named or not is_place_question(query):
        return
    if pin and earlier_place and _pin_option(pin)["label"].upper() != earlier_place.upper():
        _ask_which(c, query, pin, earlier_place)
    elif pin:
        apply_map_pin(c, query, pin, radius_meters)
    elif earlier_place:
        _use_earlier(c, query, earlier_place, named)


def _asks_town_price(q: str, base: Dict[str, Any]) -> bool:
    """A price question about a town with no limit to search by (budget, lease, distance) and no search word."""
    limits = base.get("max_budget_sgd") or base.get("min_remaining_lease_years") or base.get("amenity_radii_meters")
    return bool(base.get("towns") and _PRICE_QUESTION.search(q) and not limits and not _has("search", q)
                and not extract_addresses(q))



# ---------------------------------------------------------------------------
# Broad market questions: grouped statistics (query_resale_statistics)
# ---------------------------------------------------------------------------
# "What are the historical statistics of HDBs sold near MRT stations?", "Which MRT stations have the most expensive
# resale flats nearby?", "Sales statistics across Singapore". No named place, no budget: a question about many blocks,
# answered by one grouped query that code runs and code writes up (research_lines). Named places, one block, block
# searches with limits and policy questions are not read as this.
_AMEN = (r"(?:(?:mrt|lrt|train)(?:\s*(?:or|and|/)\s*(?:mrt|lrt))?(?:\s+stations?)?|mrts|lrts|"
         r"hawker\s+(?:centres?|centers?)|food\s+(?:centres?|centers?)|supermarkets?|(?:primary\s+)?schools?)")
_NEAR_GENERIC = re.compile(
    r"\b(?:near(?:by)?|close\s+to|next\s+to|beside|around|(?:far|away)\s+from|walking\s+distance\s+(?:to|from)|"
    r"within\s+\d+(?:\.\d+)?\s*(?:km|m|metres?|meters?)\s+(?:of|from)|distance\s+(?:to|from)|"
    r"\d+(?:\.\d+)?\s*(?:km|m|metres?|meters?)\s+(?:of|from))\s+(?:an?\s+|the\s+|any\s+|some\s+|each\s+|every\s+)?"
    rf"(?P<a>{_AMEN})\b", re.I)
_STATS_WORDS = re.compile(r"\b(statistics|stats|prices?|priced|sold|sales?|sells?|transactions?|history|historical|median|"
                          r"average|costs?|expensive|cheap\w*|pricey|premium|how much|listings?|for sale|resale|"
                          r"compar\w+|breakdown|trends?|affordable|dearest|priciest)\b", re.I)
_SEARCH_VERBS = re.compile(r"\b(find|list|search|look(?:ing)? for|recommend|suggest|which blocks|map)\b", re.I)
_STATS_STRONG = re.compile(r"\b(statistics|stats|median|average|breakdown|historical)\b", re.I)
_STATION_GROUP = re.compile(
    r"\b(?:by|per|each|every|which|what|top|best|worst|cheapest|priciest|dearest)\s+(?:(?!near\b|town|by\b)\w+\s+){0,2}?(?:mrt\s+|lrt\s+)?stations?\b"
    r"|\b(?:mrt\s+|lrt\s+)?stations?\s+(?:have|has|with|in|are|is)\b"
    r"|\b(?:most|least)\s+\w+\s+(?:mrt\s+)?stations?\b|\bstations?\s+(?:of|in)\s+\w+", re.I)
_TOWN_GROUP = re.compile(
    r"\b(?:by|per|each|every|across|which|what|top|best|worst)\s+(?:\w+\s+){0,2}?towns?\b|\bby\s+(?:estate|region)\b"
    r"|\btowns?\s+(?:have|has|with|are|is)\b|\bacross\s+(?:all\s+)?towns\b", re.I)
_ISLAND = re.compile(r"\b(?:across\s+(?:all\s+of\s+)?singapore|island[- ]?wide|nation[- ]?wide|throughout singapore|"
                     r"whole of singapore|all (?:of )?singapore|entire island|singapore[- ]wide|overall)\b", re.I)
_CATALOG = re.compile(r"\b(?:what|which)\s+(?:data|breakdowns?|statistics|stats|towns|flat types|stations)\b[^.?]{0,40}"
                      r"\b(?:do you have|can you|available|cover|analy[sz]e|offer)\b"
                      r"|\bdescribe (?:the |your )?(?:resale |hdb )?data\b|\bwhat can you (?:tell me|analy[sz]e|show) about the (?:resale )?data\b", re.I)
_DESC = re.compile(r"\b(?:most expensive|highest|priciest|dearest|costliest|pricey|most)\b", re.I)
_ASC = re.compile(r"\b(?:cheapest|lowest|least expensive|most affordable|least|cheaper)\b", re.I)
_DISTANCE = re.compile(r"(?<![$\d.,])(\d+(?:\.\d+)?)\s*(km|m|metres?|meters?)\b(?!\s*(?:k|illion))", re.I)
STATS_MIN_M, STATS_MAX_M = 50, 3000
_NEAR_BY_WORD = {"hawker": "hawker", "food": "hawker", "supermarket": "supermarket", "school": "school"}


def _stats_near(amenity: str) -> str:
    a = amenity.lower()
    if re.search(r"hawker|food", a):
        return "hawker"
    if "supermarket" in a:
        return "supermarket"
    if "school" in a:
        return "school"
    if "lrt" in a and re.search(r"mrt|train", a):
        return "mrt_lrt"
    return "lrt" if "lrt" in a else "mrt"


def _stated_bands(q: str) -> Optional[List[int]]:
    """Distance bands the user wrote out: '0-300m, 300-800m, beyond 800m' or 'bands of 200/500/1000'."""
    m = re.search(r"\bbands?\s*(?:of|at|:)?\s*((?:\d+\s*(?:m|metres?)?\s*[/,]\s*)+\d+)\s*(?:m|metres?)?", q, re.I)
    edges: List[int] = []
    if m:
        edges = [int(n) for n in re.findall(r"\d+", m.group(1))]
    else:
        ranges = re.findall(r"\b(\d{1,4})\s*(?:m|metres?)?\s*(?:-|–|to)\s*(\d{2,4})\s*(?:m|metres?)\b", q, re.I)
        tail = re.findall(r"\b(?:beyond|over|more than|above|>)\s*(\d{2,4})\s*(?:m|metres?)\b", q, re.I)
        if len(ranges) + len(tail) >= 2:
            edges = [int(b) for _, b in ranges] + [int(t) for t in tail]
    edges = sorted(set(edges))
    return edges if len(edges) >= 1 and edges[0] >= STATS_MIN_M and edges[-1] <= STATS_MAX_M and len(edges) <= 6 \
        and (m or len(edges) >= 2) else None


def extract_group_stats(q: str, base: Dict[str, Any], addresses: List[str]) -> Dict[str, Any]:
    """The grouped-statistics intent and its parameters, or {} when the question is about something else."""
    if _CATALOG.search(q) and not addresses:
        return {"wants_data_catalog": True}
    if (addresses or base.get("max_budget_sgd") or base.get("min_remaining_lease_years") or base.get("school_name")
            or base.get("policy_question") or base.get("work_destinations") or base.get("household_income")
            or base.get("cpf_oa_balance")):
        return {}
    near_m = _NEAR_GENERIC.search(q)
    sm, tm = _STATION_GROUP.search(q), _TOWN_GROUP.search(q)
    station_group, town_group = bool(sm), bool(tm)
    if sm and tm:                     # "by town near MRT stations" / "stations by town": the first one named is the grouping
        station_group, town_group = sm.start() < tm.start(), tm.start() < sm.start()
    island = bool(_ISLAND.search(q)) and not base.get("towns")
    stats_word = bool(_STATS_WORDS.search(q))
    strong = bool(_STATS_STRONG.search(q))
    if not stats_word:
        return {}
    group_cue = station_group or town_group
    if _SEARCH_VERBS.search(q) and not (group_cue or strong):
        return {}
    near = _stats_near(near_m.group("a")) if near_m else None
    price_ask = bool(re.search(r"\b(?:expensive|cheap\w*|price\w*|priciest|dearest|costs?|affordable|median|average|sold|sales?)\b", q, re.I))
    if near is None and station_group and (price_ask or strong):
        near = "mrt" if not re.search(r"\blrt\b", q, re.I) or re.search(r"\bmrt\b", q, re.I) else "lrt"
    if near is None and not (island and strong) and not (town_group and strong and not base.get("towns")):
        return {}
    if near is None and station_group:
        return {}
    out: Dict[str, Any] = {"wants_group_stats": True, "stats_near": near,
                           "stats_listings": bool(re.search(r"\blistings?\b|\bfor sale\b", q, re.I))}
    notes: List[str] = []
    # what to group by
    if station_group and near in ("mrt", "mrt_lrt", "lrt"):
        group = ["station"]
    elif town_group or (island and near is None):
        group = ["town"]
    else:
        group = ["distance_band"]
    if near is None and group == ["town"]:
        out["stats_near"] = None
    out["stats_group_by"] = group
    if _DESC.search(q) and group != ["distance_band"] and (price_ask or station_group):
        out["stats_sort"], out["stats_order"] = "median_price", "desc"
    if _ASC.search(q) and group != ["distance_band"]:
        out["stats_sort"], out["stats_order"] = "median_price", "asc"
    if group == ["town"] and not out.get("stats_sort"):
        out["stats_sort"], out["stats_order"] = "median_price", "desc"
    if group == ["station"] and not out.get("stats_sort"):
        out["stats_sort"], out["stats_order"] = "median_price", "desc"
    # a distance the user stated: "within 300m of an MRT", "near MRT (400 m)"
    bands = _stated_bands(q)
    if bands:
        out["stats_bands"] = bands
    else:
        d = None
        for m in _DISTANCE.finditer(q):
            metres = float(m.group(1)) * (1000 if m.group(2).lower() == "km" else 1)
            if metres >= 20:
                d = metres
                break
        if d is not None and near is not None:
            clamped = int(min(max(round(d), STATS_MIN_M), STATS_MAX_M))
            if clamped != round(d):
                notes.append(f"A distance of {d:,.0f} m is outside the 50-3,000 m range this analysis supports, so "
                             f"{clamped:,} m was used.")
            out["stats_distance_m"] = clamped
    # the period: "over the last 5 years", "since 2019", "historical"
    m = re.search(r"(?:last|past|over|previous)\s+(\d{1,2}|two|three|four|five|ten)\s+years?", q, re.I)
    if m:
        words = {"two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
        out["stats_lookback_years"] = min(max(int(words.get(m.group(1).lower(), m.group(1))), 1), 10)
        out["stats_trend"] = True
    elif re.search(r"(?:last|past) decade", q, re.I):
        out["stats_lookback_years"], out["stats_trend"] = 10, True
    m = re.search(r"\bsince\s+(?:\w+\s+)?(20[12]\d)\b", q, re.I)
    if m:
        out["stats_since_year"], out["stats_trend"] = int(m.group(1)), True
    if re.search(r"\b(?:histor\w+|over the years|all[- ]time|over time|trends?)\b", q, re.I) and not out.get("stats_lookback_years") \
            and not out.get("stats_since_year"):
        out["stats_lookback_years"], out["stats_trend"] = 10, True
    out["wants_price_trend"] = False
    out["wants_market_snapshot"] = False
    out["wants_town_ranking"] = False
    out["trend_years"] = None
    if notes:
        out["assumptions"] = notes
    return out


def extract_research_fields(query: str, base: Dict[str, Any]) -> Dict[str, Any]:
    """Research fields for ExtractedConstraints, given the classic regex fields in `base`."""
    q = query
    out: Dict[str, Any] = {"assumptions": [], "clarifications": []}
    out.update(_place_intents(q, extract_addresses(q)))
    out["wants_market_snapshot"] = _has("snapshot", q) or _asks_town_price(q, base)
    out["wants_town_ranking"] = _has("ranking", q) and not base.get("towns")
    stats = extract_group_stats(q, base, out["addresses"])
    if stats.get("wants_group_stats"):
        # a grouped question owns its distance words: no vague 500 m default, no block search, no spatial agent
        radii = {k: v for k, v in (base.get("amenity_radii_meters") or {}).items() if k not in ("mrt", "hawker", "supermarket", "school")}
        out.update(stats)
        out["amenity_radii_meters"] = radii
        out["bto_project_name"] = known_bto_project(q)
        return out
    out.update(stats)
    m = re.search(r"(?:last|past|over|previous)\s+(\d{1,2}|two|three|four|five|ten)\s+years?", q, re.I)
    if m:
        words = {"two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
        out["trend_years"] = int(words.get(m.group(1).lower(), m.group(1)))
    elif re.search(r"(?:last|past) decade", q, re.I):
        out["trend_years"] = 10
    out["bto_project_name"] = known_bto_project(q)

    # Vague proximity: "near an MRT" with no distance gets an explicit, disclosed default.
    radii = dict(base.get("amenity_radii_meters") or {})
    if not (out["wants_nearby_amenities"] or out["wants_area_statistics"]):
        for m in re.finditer(r"\b(?:near|close to|walking distance (?:to|from)|next to|beside)\s+(?:an?\s+|the\s+)?"
                             r"(?:[a-z]+\s+){0,2}?(mrt|lrt|train station|hawker|food centre|supermarket)\b", q, re.I):
            key = {"lrt": "mrt", "train station": "mrt", "food centre": "hawker"}.get(m.group(1).lower(), m.group(1).lower())
            if key not in radii:
                radii[key] = VAGUE_RADIUS_METERS
                label = {"mrt": "an MRT/LRT station", "hawker": "a hawker centre", "supermarket": "a supermarket"}[key]
                out["assumptions"].append(
                    f"'Near' was taken as within {VAGUE_RADIUS_METERS} m straight-line of {label} (the nearest one, "
                    f"not a specific named station). Say a different distance to change this.")
    if radii != (base.get("amenity_radii_meters") or {}):
        out["amenity_radii_meters"] = radii

    # "near good schools" names no school; ask instead of matching a school containing "good".
    if generic_school_phrase(base.get("school_name")):
        out["school_name"] = None
        radii.pop("school", None)
        out["amenity_radii_meters"] = radii
        out["clarifications"].append(SCHOOL_CLARIFICATION)
    return out


def is_block_search(c: Dict[str, Any], query: str) -> bool:
    """Whether the classic town block search should run alongside research tools."""
    other = any(c.get(k) for k in ("wants_price_trend", "wants_bto_comparison", "wants_block_history",
                                   "wants_price_assessment", "wants_option_comparison", "wants_nearby_amenities",
                                   "wants_market_snapshot", "wants_town_ranking", "wants_area_statistics",
                                   "bto_project_name"))   # a named launch is its own question, not a town search
    stated = c.get("max_budget_sgd") or c.get("min_remaining_lease_years") or \
        {k: v for k, v in (c.get("amenity_radii_meters") or {}).items() if k != "school"}
    if c.get("wants_group_stats") or c.get("wants_data_catalog") or c.get("analysis_mode"):
        return bool(stated)     # "show me prices by town near MRT" is a statistics question, not a flat search
    return bool(stated) or _has("search", query) or not other


def needs_market_agent(c: Dict[str, Any]) -> bool:
    has_blocks = any(re.match(r"(?:blk\s+)?\d", a, re.I) for a in c.get("addresses") or [])
    if c.get("wants_area_statistics") and (c.get("addresses") or c.get("map_pin")):
        return True
    return bool(c.get("wants_market_snapshot") or c.get("wants_town_ranking") or c.get("bto_project_name")
                or c.get("wants_group_stats") or c.get("wants_data_catalog") or c.get("analysis_mode")
                or (c.get("wants_price_trend") and not c.get("towns"))
                or (has_blocks and (c.get("wants_block_history") or c.get("wants_price_assessment")
                                    or c.get("wants_option_comparison"))))


def needs_spatial_agent(c: Dict[str, Any]) -> bool:
    return bool(c.get("wants_nearby_amenities") and (c.get("addresses") or c.get("map_pin")))


def _flat_type(c: Dict[str, Any], assumptions: List[str], purpose: str) -> str:
    types = c.get("flat_types") or []
    if types:
        return types[0]
    note = f"No flat type was given, so {purpose} uses {DEFAULT_FLAT_TYPE} flats."
    if note not in assumptions:   # the plan is computed more than once per answer (gating, the run, the evidence gate)
        assumptions.append(note)
    return DEFAULT_FLAT_TYPE


def _area_arg(c: Dict[str, Any], assumptions: List[str]) -> Optional[Dict[str, Any]]:
    """The circle for an area question: the first named place (or the map pin) and its radius."""
    if not c.get("wants_area_statistics"):
        return None
    place: Dict[str, Any] = ({"address": c["addresses"][0]} if c.get("addresses")
                             else dict(c["map_pin"]) if c.get("map_pin") else {})
    if not place:
        return None
    radius = c.get("area_radius_meters")
    if not radius:
        radius = VAGUE_RADIUS_METERS
        note = (f"'Around' was taken as within {radius} m straight-line of the place. "
                f"Say a different distance (e.g. 'within 300m') to change this.")
        if note not in assumptions:
            assumptions.append(note)
    return {**place, "radius_meters": int(radius)}


def stats_call_args(c: Dict[str, Any], assumptions: List[str]) -> Dict[str, Any]:
    """The query_resale_statistics call for a grouped-statistics question: what the user asked, with disclosed defaults."""
    group = [g for g in (c.get("stats_group_by") or []) if g][:2] or ["distance_band"]
    near = c.get("stats_near")
    args: Dict[str, Any] = {"group_by": group}
    if c.get("towns"):
        args["towns"] = list(c["towns"])
    if c.get("flat_types"):
        args["flat_types"] = list(c["flat_types"])
    if near:
        args["near"] = near
    distance = c.get("stats_distance_m")
    if c.get("stats_bands") and "distance_band" in group:
        args["bands"] = list(c["stats_bands"])
    elif distance and "distance_band" in group:
        args["bands"] = [int(distance)]          # "within 300 m" against "further than 300 m"
    elif "distance_band" not in group and near:
        args["max_distance_m"] = int(distance or VAGUE_RADIUS_METERS)
    if c.get("stats_since_year"):
        args["start_month"] = f"{int(c['stats_since_year'])}-01"
    elif c.get("stats_lookback_years"):
        args["lookback_years"] = int(c["stats_lookback_years"])
    if c.get("stats_trend"):
        args["include_trend"] = True
    if c.get("stats_sort"):
        args["sort_by"] = c["stats_sort"]
        args["order"] = c.get("stats_order") or "desc"
    return args


def research_plan(agent: str, c: Dict[str, Any], query: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Deterministic calls for the research intents; assumptions are recorded in c['assumptions']."""
    plan: List[Tuple[str, Dict[str, Any]]] = []
    assumptions: List[str] = c.setdefault("assumptions", [])
    blocks = [a for a in c.get("addresses") or [] if re.match(r"(?:blk\s+)?\d", a, re.I)]
    if agent == "market_sql":
        if c.get("wants_data_catalog"):
            plan.append(("describe_resale_data", {}))
        if c.get("wants_group_stats"):
            plan.append(("query_resale_statistics", stats_call_args(c, assumptions)))
        area = _area_arg(c, assumptions)
        if area:
            flat = (c.get("flat_types") or [None])[0]
            plan.append(("find_resale_blocks_near", dict(**area, flat_type=flat)))
            plan.append(("get_area_resale_statistics", dict(area=area, flat_type=flat)))
            if c.get("trend_years") or _has("area_trend", query):
                plan.append(("get_area_price_trend", dict(area=area, flat_type=_flat_type(c, assumptions, "the area trend"),
                                                          lookback_years=c.get("trend_years") or 3)))
            if c.get("wants_block_history"):
                plan.append(("get_block_transaction_history", dict(area=area, flat_type=flat)))
        if c.get("wants_option_comparison") and len(blocks) >= 2:
            plan.append(("compare_resale_options", dict(addresses=blocks[:4],
                                                        flat_type=_flat_type(c, assumptions, "the comparison"))))
        if c.get("wants_price_assessment") and blocks:
            plan.append(("assess_resale_price", dict(address=blocks[0], asking_price=c.get("asking_price_sgd"),
                                                     flat_type=_flat_type(c, assumptions, "the price check"))))
        if c.get("wants_block_history") and blocks and not area:
            for address in blocks[:3]:
                plan.append(("get_block_transaction_history", dict(address=address, flat_type=(c.get("flat_types") or [None])[0])))
        if c.get("wants_town_ranking"):
            # "cheapest" asks about price paid; "best value" about price per sqm; "afford" about the budget.
            rank_by = ("share_within_budget" if c.get("max_budget_sgd") and re.search(r"\bafford", query, re.I)
                       else "median_price" if re.search(r"\bcheap|lowest price|least expensive|most affordable", query, re.I)
                       else "price_per_sqm")
            plan.append(("rank_towns_by_value", dict(flat_type=_flat_type(c, assumptions, "the town ranking"),
                                                     max_budget=c.get("max_budget_sgd"),
                                                     min_lease_years=c.get("min_remaining_lease_years"),
                                                     rank_by=rank_by)))
        if c.get("wants_market_snapshot") or (c.get("wants_price_trend") and not c.get("towns") and not area):
            plan.append(("get_market_snapshot", dict(towns=c.get("towns") or [], flat_types=c.get("flat_types") or [])))
        if c.get("bto_project_name"):
            plan.append(("compare_bto_launch_vs_resale", dict(bto_project_name=c["bto_project_name"],
                                                              flat_type=_flat_type(c, assumptions, "the BTO comparison"))))
    elif agent == "spatial" and needs_spatial_agent(c):
        radius = max([v for k, v in (c.get("amenity_radii_meters") or {}).items() if k != "school"]
                     or [c.get("pin_radius_meters") or 1000])
        radius = int(min(max(radius, 50), 5000))
        for address in (c.get("addresses") or [])[:2]:
            plan.append(("get_nearby_amenities", dict(address=address, radius_meters=radius)))
        if not c.get("addresses") and c.get("map_pin"):
            plan.append(("get_nearby_amenities", dict(**c["map_pin"], radius_meters=radius)))
    return plan


# ---------------------------------------------------------------------------
# Tool-message compaction and coordinate provenance
# ---------------------------------------------------------------------------
def compact_tool_result(result: Any, max_chars: int = 12000) -> str:
    """Valid JSON within max_chars: long lists are shortened with an explicit note."""
    text = json.dumps(result, default=str)
    if len(text) <= max_chars:
        return text
    for keep in (50, 25, 10, 5, 2):
        def trim(value: Any) -> Any:
            if isinstance(value, list) and len(value) > keep:
                return [trim(v) for v in value[:keep]] + [{"_omitted_items": len(value) - keep,
                                                           "_note": "list shortened for the model; full result kept in state"}]
            if isinstance(value, list):
                return [trim(v) for v in value]
            if isinstance(value, dict):
                return {k: trim(v) for k, v in value.items() if k != "sql"}
            return value
        text = json.dumps(trim(result), default=str)
        if len(text) <= max_chars:
            return text
    return json.dumps({"_truncated": True, "_note": "result too large for the model; summary only",
                       "keys": list(result)[:20] if isinstance(result, dict) else None,
                       "length": len(result) if isinstance(result, (list, dict)) else None})


def coordinates_in(value: Any) -> Iterable[Tuple[float, float]]:
    if isinstance(value, dict):
        for lat_key, lng_key in (("lat", "lng"), ("latitude", "longitude")):
            lat, lng = value.get(lat_key), value.get(lng_key)
            if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
                yield float(lat), float(lng)
        for v in value.values():
            yield from coordinates_in(v)
    elif isinstance(value, list):
        for v in value:
            yield from coordinates_in(v)


def unverified_coordinates(tool: str, args: Dict[str, Any], known: List[Tuple[float, float]]) -> Optional[str]:
    """Error text when a model passes coordinates that no tool, pin or user statement produced."""
    keys = COORDINATE_TOOLS.get(tool)
    if tool in AREA_TOOLS and isinstance(args.get("area"), dict):
        args, keys = args["area"], ("latitude", "longitude")
    if not keys or args.get(keys[0]) is None or args.get(keys[1]) is None:
        return None
    lat, lng = float(args[keys[0]]), float(args[keys[1]])
    for k_lat, k_lng in known:
        if math.hypot((lat - k_lat) * 111_320, (lng - k_lng) * 111_320 * math.cos(math.radians(lat))) <= 30:
            return None
    return ("Coordinates rejected: they did not come from resolve_location, a tool result or the user's map pin. "
            "Never estimate coordinates. Use assess_resale_price / get_block_transaction_history with the user's "
            "address, or resolve_location first.")


# ---------------------------------------------------------------------------
# Template answer sections and map payload
# ---------------------------------------------------------------------------
def _m(v: Optional[float]) -> str:
    return f"${v:,.0f}" if isinstance(v, (int, float)) else "n/a"


def _pct(v: Optional[float]) -> str:
    return f"{v:+.1f}%" if isinstance(v, (int, float)) else "n/a"


def _candidates(result: Dict[str, Any]) -> str:
    names = [c.get("name") for c in result.get("candidates") or []][:5]
    return f" Did you mean: {'; '.join(names)}?" if names else ""


def research_lines(state: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    sql = state.get("sql_market_results") or {}
    loc = state.get("location_results") or {}
    c = state.get("user_constraints") or {}

    cmp_ = sql.get("compare_resale_options")
    if cmp_ and "error" not in cmp_:
        lines.append(f"**Option comparison ({cmp_['flat_type']}, {cmp_['metadata']['window_start_month']} to "
                     f"{cmp_['metadata']['window_end_month']}):**")
        lines.append("| Option | Sales | Median price | Per sqm | Lease left | Nearest MRT | Hawker | Supermarket | Primary school |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for o in cmp_["options"]:
            if o["status"] not in ("ok", "no_transactions"):
                lines.append(f"| {o['address']} | address not resolved |  |  |  |  |  |  |  |")
                continue
            near = o.get("nearest") or {}
            dist = [f"{near[k]['name']} {near[k]['distance_meters']:.0f} m" if near.get(k) else "n/a"
                    for k in ("mrt", "hawker", "supermarket", "school")]
            lease = f"{o['median_remaining_lease_years']:.0f} yrs" if o.get("median_remaining_lease_years") else "n/a"
            ppsm = f"${o['median_price_per_sqm']:,.0f}" if o.get("median_price_per_sqm") else "n/a"
            lines.append(f"| {o['location']['name']} | {o['transaction_count']} | {_m(o.get('median_price'))} | {ppsm} | "
                         f"{lease} | " + " | ".join(dist) + " |")
        for o in cmp_["options"]:
            if o["status"] not in ("ok", "no_transactions"):
                lines.append(f"- '{o['address']}' could not be matched to one HDB block.{_candidates(o)}")

    assess = sql.get("assess_resale_price")
    if assess and "error" not in assess:
        if assess["status"] in ("ambiguous", "not_found"):
            lines.append(f"- Price check: {assess['message']}{_candidates(assess)}")
        else:
            n = assess.get("nearby")
            lines.append(f"**Price check for {assess['location']['name']} ({assess['flat_type']}):**")
            if n:
                lines.append(f"- {n['transaction_count']} nearby sales: median {_m(n['median'])}, middle 50% "
                             f"{_m(n['lower_quartile'])}-{_m(n['upper_quartile'])}, range {_m(n['minimum'])}-{_m(n['maximum'])}.")
            b = assess.get("same_block")
            if b:
                lines.append(f"- Same block: {b['transaction_count']} sale(s), median {_m(b['median'])}.")
            if assess.get("asking_price") and assess.get("position"):
                where = {"below_lower_quartile": "below the middle 50% of nearby sales",
                         "within_middle_50": "within the middle 50% of nearby sales",
                         "above_upper_quartile": "above the middle 50% of nearby sales"}[assess["position"]]
                lines.append(f"- Asking {_m(assess['asking_price'])} is {where} ({_pct(assess['asking_vs_nearby_median_pct'])} "
                             f"vs the median; higher than {assess['asking_percentile']:.0f}% of them).")
            lines.append(f"- {assess['message']} " + " ".join(assess["metadata"]["notes"]))

    lines.extend(_area_lines(sql))

    catalog = sql.get("describe_resale_data")
    if catalog and "error" not in catalog:
        lines.extend(_catalog_lines(catalog))
    stats = sql.get("query_resale_statistics")
    if stats and "error" not in stats:
        lines.extend(stats_lines(stats, c))

    histories = sql.get("get_block_transaction_history")
    for h in ([histories] if isinstance(histories, dict) else histories or []):
        if "error" in h:
            continue
        if h["status"] in ("ambiguous", "not_found"):
            lines.append(f"- Block history: {h['message']}{_candidates(h)}")
            continue
        lines.append(f"**Sales history: {h['message']}**")
        for y in h.get("yearly", []):
            lines.append(f"- {y['year']} {y['flat_type']}: {y['transaction_count']} sale(s), median {_m(y['median_price'])} "
                         f"(${y['median_price_per_sqm']:,.0f}/sqm)")
        for t in h.get("transactions", [])[:5]:
            where = f"Blk {t['block']} {t['street']}, " if h.get("scope", "block") != "block" and t.get("block") else ""
            lines.append(f"  - {t['month']}: {where}{t['flat_type']}, storey {t['storey_range']}, {t['floor_area_sqm']:.0f} sqm, "
                         f"{_m(t['resale_price'])}")

    rank = sql.get("rank_towns_by_value")
    if rank and "error" not in rank and rank.get("rows"):
        budget = c.get("max_budget_sgd")
        lines.append(f"**Towns ranked by {rank['ranking_basis']} ({rank['flat_type']}, {rank['metadata']['window_start_month']} "
                     f"to {rank['metadata']['window_end_month']}):**")
        for r in rank["rows"][:10]:
            share = f", {r['share_within_budget_pct']:.0f}% of sales within {_m(budget)}" if r.get("share_within_budget_pct") is not None else ""
            lines.append(f"{r['rank']}. {r['town']}: median {_m(r['median_price'])}, ${r['median_price_per_sqm']:,.0f}/sqm, "
                         f"{r['median_remaining_lease_years']:.0f} yrs lease, {r['transaction_count']} sales{share}")
        lines.append("- Price per sqm ignores location, condition and amenities.")

    snap = sql.get("get_market_snapshot")
    if snap and "error" not in snap and snap.get("rows"):
        lines.append(f"**Resale market, {snap['latest_complete_quarter']} (latest complete quarter):**")
        for r in snap["rows"]:
            lines.append(f"- {r['town'].title()} {r['flat_type']}: median {_m(r['median_price'])} "
                         f"(${r['median_price_per_sqm']:,.0f}/sqm, {r['transaction_count']} sales); "
                         f"vs prior quarter {_pct(r['qoq_change_pct'])}, 1 yr {_pct(r['yoy_change_pct'])}, "
                         f"3 yrs {_pct(r['change_3y_pct'])}, 5 yrs {_pct(r['change_5y_pct'])}")
        lines.extend(_cheaper_town_lines(snap["rows"]))

    bto = sql.get("compare_bto_launch_vs_resale")
    if bto and "error" not in bto and bto.get("status") in ("not_found", "flat_type_not_offered"):
        if bto["status"] == "not_found":
            lines.append(f"- BTO project '{bto['bto_project_name']}' is not in the launch dataset. Covered: "
                         + ", ".join(bto.get("available_projects") or []) + ".")
        else:
            lines.append(f"- {bto['bto_project_name']} does not offer {bto['flat_type']}; it offers "
                         + ", ".join(bto.get("available_flat_types") or []) + ".")

    for result in _as_list(loc.get("get_nearby_amenities")):
        if result["status"] != "ok":
            lines.append(f"- Nearby amenities: {result['message']}{_candidates(result)}")
            continue
        lines.append(f"**Near {result['location']['name']} (within {result['radius_meters']} m):**")
        labels = {"mrt": "MRT/LRT", "hawker": "Hawker centres", "supermarket": "Supermarkets", "school": "Primary schools"}
        for cat, label in labels.items():
            if cat not in result["counts"]:
                continue
            inside = [a for a in result["amenities"] if a["category"] == cat]
            nearest = result["nearest"].get(cat)
            if inside:
                lines.append(f"- {label} ({result['counts'][cat]}): " + "; ".join(
                    f"{a['name']} {a['distance_meters']:.0f} m (~{a['walking_minutes']} min walk)" for a in inside[:4]))
            elif nearest:
                lines.append(f"- {label}: none within {result['radius_meters']} m; nearest is {nearest['name']} "
                             f"{nearest['distance_meters']:.0f} m away.")
        lines.append(f"- {result['message']}")

    for note in dict.fromkeys(c.get("assumptions") or []):
        lines.append(f"_Assumption: {note}_")
    for question in c.get("clarifications") or []:
        # a menu is a numbered list of its own and reads better without a "Question:" label
        lines.append(question if c.get("menu_options") else f"**Question:** {question}")
    return lines


# ---------------------------------------------------------------------------
# Grouped statistics: the answer, the drill-down chips and the map (query_resale_statistics)
# ---------------------------------------------------------------------------
NEAR_A = {"mrt": "an MRT station", "mrt_lrt": "an MRT or LRT station", "lrt": "an LRT station", "hawker": "a hawker centre",
          "supermarket": "a supermarket", "school": "a primary school"}
NEAR_PLURAL = {"mrt": "MRT stations", "mrt_lrt": "MRT or LRT stations", "lrt": "LRT stations", "hawker": "hawker centres",
               "supermarket": "supermarkets", "school": "primary schools"}
DIM_HEADER = {"distance_band": "Distance", "town": "Town", "station": "Nearest station", "street": "Street",
              "flat_type": "Flat type", "year": "Year", "storey_band": "Storey", "lease_band": "Lease left"}
DRILL_RADII = (200, 500, 1000)


def _flat_words(flats: List[str]) -> str:
    words = [f"{f.split()[0]}-room" if re.match(r"\d ROOM$", f) else f.lower() for f in flats]
    return " and ".join(words)


def _within(label: str) -> str:
    """'0-200 m' -> 'within 200 m'."""
    m = re.match(r"0-([\d,]+ m)$", label)
    return f"within {m.group(1)}" if m else label


def _gap(a: float, b: float) -> str:
    pct = (a / b - 1) * 100 if b else 0.0
    if abs(pct) < 0.05:
        return "the same as"
    return f"{abs(pct):.1f}% {'above' if pct > 0 else 'below'}"


def _gl(r: Dict[str, Any], g: Dict[str, Any]) -> str:
    """A group's label as shown: towns in title case (the data holds them upper case)."""
    return g["label"].title() if r["group_by"] == ["town"] else g["label"]


def _stats_headline(r: Dict[str, Any]) -> str:
    groups = [g for g in r.get("groups") or [] if g.get("median_price")]
    if not groups:
        return ""
    near = NEAR_A.get(r.get("near") or "", "")
    base = r.get("baseline") or {}
    if r["group_by"] == ["distance_band"] and near:
        first, last = groups[0], groups[-1]
        if len(groups) >= 2 and last["label"].startswith(">") and first["label"].startswith("0-"):
            far = last["label"][1:] + " away"
            return (f"Flats {_within(first['label'])} of {near} sold at a median of {_m(first['median_price'])}, "
                    f"{_gap(first['median_price'], last['median_price'])} flats more than {far} ({_m(last['median_price'])}).")
        return f"Flats {_within(first['label'])} of {near} sold at a median of {_m(first['median_price'])}."
    desc = r.get("order", "desc") == "desc"
    if r["group_by"][0] in ("year", "storey_band", "lease_band") and len(groups) >= 2:
        a, b = groups[0], groups[-1]
        return f"The median was {_m(a['median_price'])} for {a['label']} and {_m(b['median_price'])} for {b['label']} ({_pct(b['median_price'] / a['median_price'] * 100 - 100)})."
    top = groups[0]
    word = {"count": "most sales", "median_psm": "highest price per sqm" if desc else "lowest price per sqm"}.get(
        r.get("sort_by") or "", "highest median price" if desc else "lowest median price")
    line = f"{_gl(r, top)} has the {word}"
    if (r.get("sort_by") or "median_price") == "count":
        line += f" ({top['count']:,} sales)"
    else:
        line += f" ({_m(top['median_price'])})"
    if base.get("median_price"):
        line += f", {_gap(top['median_price'], base['median_price'])} the {_m(base['median_price'])} median for {base['label']}"
    return line + "."


def _radius_note(r: Dict[str, Any]) -> str:
    """The distance the answer used, always stated, with how to change it."""
    near = NEAR_A.get(r.get("near") or "")
    if not near:
        return ""
    lrt = " (LRT stations excluded)" if r.get("near") == "mrt" else ""
    if r.get("bands") and "distance_band" in r["group_by"]:
        labels = ", ".join(g["label"] for g in r["groups"]) if r.get("groups") else ""
        return (f"Distance bands ({labels}) are straight-line metres to the nearest {near.split(' ', 1)[1]}{lrt}"
                " - say a different distance to change this.")
    if r.get("max_distance_m"):
        return (f"Within {r['max_distance_m']:,} m straight-line of {near}{lrt} - say a different distance to change this.")
    return ""


def stats_lines(r: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """Any query_resale_statistics result as an answer: headline, table, trend, notes. All figures are the tool's."""
    if r.get("status") == "invalid_request":
        lines = [f"- That breakdown could not be run: {r.get('message')}"]
        for key, vals in (r.get("allowed") or {}).items():
            lines.append(f"  - {key}: {', '.join(vals[:15])}{' ...' if len(vals) > 15 else ''}")
        return lines
    p = r.get("period") or {}
    if r.get("status") == "no_data":
        return [f"- No group had at least {r.get('min_sales', 10)} sales between {p.get('start_month')} and {p.get('end_month')}; "
                "try a wider distance, a longer period or fewer filters."]
    lines: List[str] = []
    if c.get("stats_listings"):
        lines.append("_Live listings are not available in this app; these figures are past resale sales._")
    head = _stats_headline(r)
    flat = _flat_words(r.get("flat_types") or [])
    scope = ", ".join(t.title() for t in r.get("towns") or []) or "Singapore"
    title = f"**Resale statistics{' for ' + flat + ' flats' if flat else ''} in {scope}, {p.get('start_month')} to {p.get('end_month')}**"
    lines.append(title)
    if head:
        lines.append(head)
    groups = r["groups"]
    show_psm = any(g.get("median_psm") for g in groups)
    show_1y = any(g.get("change_1y_pct") is not None for g in groups)
    show_pp = any(g.get("change_period_pct") is not None for g in groups)
    header = " · ".join(DIM_HEADER[d] for d in r["group_by"])
    cols = ["Sales", "Median price", "Middle 50%"] + (["Per sqm"] if show_psm else []) + (["Last 12 mths vs year before"] if show_1y else []) \
        + (["Last 12 mths vs first 12"] if show_pp else [])
    lines.append("| " + " | ".join([header] + cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for g in groups:
        row = [_gl(r, g), f"{g['count']:,}", _m(g.get("median_price")), f"{_m(g.get('p25'))} - {_m(g.get('p75'))}"]
        if show_psm:
            row.append(f"${g['median_psm']:,.0f}" if g.get("median_psm") else "n/a")
        if show_1y:
            row.append(_pct(g.get("change_1y_pct")))
        if show_pp:
            row.append(_pct(g.get("change_period_pct")))
        lines.append("| " + " | ".join(row) + " |")
    base = r.get("baseline") or {}
    if base.get("count"):
        row = [f"_{base['label']}_", f"{base['count']:,}", _m(base.get("median_price")), ""]
        if show_psm:
            row.append(f"${base['median_psm']:,.0f}" if base.get("median_psm") else "")
        row += [""] * (show_1y + show_pp)
        lines.append("| " + " | ".join(row) + " |")
    if r.get("truncated"):
        lines.append(f"- ...and {r['total_groups'] - len(groups)} more group(s) not shown (ask for a different sort or a narrower filter).")
    if r.get("include_trend"):
        lines.append("**Median price by year** (first and last year may be part-years):")
        for g in groups[:6]:
            if g.get("trend"):
                lines.append(f"- {_gl(r, g)}: " + " → ".join(f"{t['year']} {_m(t['median_price'])}" for t in g["trend"]))
    note = _radius_note(r)
    if note:
        lines.append(f"_{note}_")
    if p:
        lines.append(f"_Period: {p['start_month']} to {p['end_month']} ({p['months']} months). Medians pool every sale in the group; "
                     f"groups with fewer than {r.get('min_sales', 10)} sales are left out"
                     + (f" ({r['excluded_small_groups']} left out)" if r.get("excluded_small_groups") else "") + "._")
    for n in r.get("assumptions") or []:
        lines.append(f"_Assumption: {n}_")
    for n in r.get("notes") or []:
        if not n.startswith("Historical transactions") or not c.get("stats_listings"):
            lines.append(f"_{n}_")
    return lines


def _catalog_lines(d: Dict[str, Any]) -> List[str]:
    p = d.get("period_covered") or {}
    lines = [f"**What the resale data covers** ({p.get('first_month')} to {p.get('last_complete_month')}, past sales only):",
             "- Break sales down by: " + "; ".join(f"{k.replace('_', ' ')} ({v})" for k, v in (d.get("dimensions") or {}).items()),
             "- Distances can be measured to: " + "; ".join(f"{v}" for v in (d.get("near_options") or {}).values()),
             f"- Towns: {', '.join(t.title() for t in d.get('towns') or [])}",
             f"- Flat types: {', '.join(d.get('flat_types') or [])}",
             f"- Figures: " + "; ".join((d.get("metrics") or {}).values())]
    if d.get("mrt_stations"):
        lines.append(f"- {len(d['mrt_stations'])} MRT and {len(d.get('lrt_stations') or [])} LRT stations are covered, e.g. "
                     + ", ".join(d["mrt_stations"][:5]))
    lines.extend(f"_{x}_" for x in d.get("definitions") or [])
    return lines


def _chip_period(c: Dict[str, Any]) -> str:
    """The period words that put a chip's question back on the period the user asked for."""
    if c.get("stats_since_year"):
        return f" since {int(c['stats_since_year'])}"
    if (c.get("stats_lookback_years") or 0) > 1:
        return f" over the last {int(c['stats_lookback_years'])} years"
    return ""


def stats_drill_text(r: Dict[str, Any], g: Dict[str, Any], c: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """The complete question that goes one level below a town or station group (None for other groups)."""
    flats = _flat_words(r.get("flat_types") or [])
    fl = f"{flats} " if flats else ""
    near = r.get("near")
    words = NEAR_PLURAL.get(near or "", "")
    period = _chip_period(c or {})
    dist = f"within {r['max_distance_m']:,} m of " if r.get("max_distance_m") and r["max_distance_m"] != VAGUE_RADIUS_METERS else "near "
    if r["group_by"] == ["town"]:
        t = g["key"]["town"].title()
        if near in ("mrt", "mrt_lrt", "lrt"):
            text = f"{fl}resale prices by station {dist}{words} in {t}{period}"
        elif near:
            text = f"{fl}resale prices {dist}{words} in {t}{period}"
        else:
            text = f"{fl}resale prices by station near MRT stations in {t}{period}"
    elif r["group_by"] == ["station"]:
        radius = r.get("max_distance_m") or VAGUE_RADIUS_METERS
        text = f"{fl}resale prices near {g['key']['station']}, within {radius}m"
    else:
        return None
    return text[:1].upper() + text[1:]


def drilldown_suggestions(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """Follow-up chips, each a complete question (no state needed to answer it):
    island/bands -> town -> station -> the area around a station -> a block's sales."""
    sql = state.get("sql_market_results") or {}
    c = state.get("user_constraints") or {}
    chips: List[Dict[str, str]] = []
    cap = lambda t: t[:1].upper() + t[1:]   # noqa: E731
    r = sql.get("query_resale_statistics")
    if isinstance(r, dict) and r.get("status") == "ok" and r.get("groups"):
        flats = _flat_words(r.get("flat_types") or [])
        fl = f"{flats} " if flats else ""
        near = r.get("near")
        words = NEAR_PLURAL.get(near or "", "")
        since = c.get("stats_since_year")
        period = _chip_period(c)
        dist = f"within {r['max_distance_m']:,} m of " if r.get("max_distance_m") and r["max_distance_m"] != VAGUE_RADIUS_METERS else "near "
        town_list = r.get("towns") or []
        town_text = f" in {town_list[0].title()}" if len(town_list) == 1 else ""
        gb = r["group_by"]
        if gb == ["distance_band"] and near:
            chips.append({"label": "By town", "text": cap(f"{fl}resale prices by town {dist}{words}{town_text}{period}")})
            if near in ("mrt", "mrt_lrt", "lrt"):
                chips.append({"label": "By station", "text": cap(f"{fl}resale prices by station {dist}{words}{town_text}{period}")})
            if not r.get("include_trend") and not since:
                chips.append({"label": "Trend since 2017", "text": cap(f"{fl}resale prices near {words}{town_text} since 2017")})
            if not flats:
                chips.append({"label": "4-room only", "text": cap(f"4-room resale prices near {words}{town_text}{period}")})
        elif gb == ["town"]:
            for g in r["groups"][:4]:
                chips.append({"label": g["key"]["town"].title(), "text": stats_drill_text(r, g, c)})
            if near:
                chips.append({"label": "Back to distance bands", "text": cap(f"{fl}resale prices near {words}{period}")})
            else:
                chips.append({"label": "Near MRT vs far", "text": cap(f"{fl}resale prices near MRT stations vs far from MRT{period}")})
        elif gb == ["station"]:
            for g in r["groups"][:4]:
                chips.append({"label": g["key"]["station"], "text": stats_drill_text(r, g, c)})
            chips.append({"label": "Back to towns", "text": cap(f"{fl}resale prices by town {dist}{words}{period}")})
    area = sql.get("find_resale_blocks_near")
    if isinstance(area, dict) and area.get("status") == "ok" and area.get("blocks"):
        loc = area.get("location") or {}
        blocks = [b for b in area["blocks"] if b.get("transaction_count")]
        flat = area.get("flat_type")
        fl = f"{_flat_words([flat])} " if flat else ""
        for b in sorted(blocks, key=lambda b: -b["transaction_count"])[:4]:
            chips.append({"label": f"Blk {b['block']} {b['street'].title()}",
                          "text": f"What did {fl}flats at Blk {b['block']} {b['street']} sell for recently?"})
        if loc.get("kind") == "mrt" and blocks:
            towns = {}
            for b in blocks:
                towns[b["town"]] = towns.get(b["town"], 0) + 1
            town = max(towns, key=towns.get).title()
            chips.append({"label": f"All stations in {town}",
                          "text": cap(f"{fl}resale prices by station near MRT stations in {town}")})
    return chips



def _cheaper_town_lines(rows: List[Dict[str, Any]]) -> List[str]:
    """Which of two or more named towns is cheaper, by median price, for each flat type they share. Code states it, so
    the user gets a direct answer to "is X cheaper than Y". The island-wide row is not a town."""
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("town") != "ALL TOWNS" and r.get("median_price"):
            by_type.setdefault(r["flat_type"], []).append(r)
    lines: List[str] = []
    for flat_type, rs in by_type.items():
        if len(rs) < 2:
            continue
        rs = sorted(rs, key=lambda r: r["median_price"])
        low, high = rs[0], rs[-1]
        gap = high["median_price"] - low["median_price"]
        if len(rs) == 2:
            if gap == 0:
                lines.append(f"- {low['town'].title()} and {high['town'].title()} have the same {flat_type} median.")
            else:
                lines.append(f"- {low['town'].title()} is cheaper than {high['town'].title()} for {flat_type} by "
                             f"${gap:,.0f} ({gap / high['median_price'] * 100:.1f}% below {high['town'].title()}'s median).")
        else:
            lines.append(f"- For {flat_type}, {low['town'].title()} has the lowest median and {high['town'].title()} the highest "
                         f"(${gap:,.0f} apart).")
    return lines


def _area_lines(sql: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    near = sql.get("find_resale_blocks_near")
    stats = sql.get("get_area_resale_statistics")
    trend = sql.get("get_area_price_trend")
    for r in (near, stats, trend):
        if r and "error" not in r and r.get("status") in ("ambiguous", "not_found"):
            lines.append(f"- Area: {r['message']}{_candidates(r)}")
            return lines
    if stats and "error" not in stats and stats.get("status") in ("ok", "no_transactions"):
        md = stats["metadata"]
        where = (f"{stats['block_count']} blocks within {stats['radius_meters']} m of {stats['location']['name']}"
                 if stats["scope"] == "area" else f"{stats['block_count']} selected blocks")
        lines.append(f"**Resale prices, {where} ({md['window_start_month']} to {md['window_end_month']}):**")
        if not stats["by_flat_type"]:
            lines.append(f"- {stats['message']}")
        target = {t["flat_type"]: t for t in stats.get("target_by_flat_type") or []}
        for f in stats["by_flat_type"]:
            d = f["distribution"]
            lines.append(f"- {f['flat_type']}: {d['transaction_count']} sales in {f['block_count']} blocks, median {_m(d['median'])}, "
                         f"middle 50% {_m(d['lower_quartile'])}-{_m(d['upper_quartile'])}, range {_m(d['minimum'])}-{_m(d['maximum'])}, "
                         f"${d['median_price_per_sqm']:,.0f}/sqm, {f['median_floor_area_sqm']:.0f} sqm, "
                         f"{f['median_remaining_lease_years']:.0f} yrs lease left")
            t = target.get(f["flat_type"])
            if t and d["median"]:
                td = t["distribution"]
                gap = 100 * (td["median"] / d["median"] - 1)
                lines.append(f"  - {stats['location']['name']} itself: {td['transaction_count']} sale(s), median {_m(td['median'])} "
                             f"({_pct(gap)} vs the area median)")
        lines.append("- " + " ".join(md["notes"][:1]))
    if trend and "error" not in trend and trend.get("status") == "ok":
        lines.append(f"**{trend['flat_type']} price trend, same area:**")
        if trend.get("period_change_pct") is not None:
            cagr = f", {_pct(trend['period_cagr_pct'])}/yr" if trend.get("period_cagr_pct") is not None else ""
            lines.append(f"- {trend['period_start_quarter']} to {trend['period_end_quarter']}: median price {_pct(trend['period_change_pct'])} "
                         f"(per sqm {_pct(trend['period_price_per_sqm_change_pct'])}{cagr}).")
        lines.append(f"- Quarterly medians{' (last 8 quarters)' if len(trend['points']) > 8 else ''}: " + "; ".join(
            f"{p['quarter']} {_m(p['median_price'])} ({p['transaction_count']})" for p in trend["points"][-8:]))
        lines.append("- " + " ".join(trend["metadata"]["notes"]))
    if near and "error" not in near and near.get("status") == "ok":
        label = f"{near['flat_type']} " if near.get("flat_type") else ""
        md = near["metadata"]
        lines.append(f"**Blocks within {near['radius_meters']} m of {near['location']['name']}: {near['blocks_with_sales']} of "
                     f"{near['block_count']} had {near['transaction_count']} {label}sale(s), {md['window_start_month']} to "
                     f"{md['window_end_month']}:**")
        lines.append("| Block | Distance | Sales | Median price | Per sqm | Lease left |")
        lines.append("|---|---|---|---|---|---|")
        for b in [b for b in near["blocks"] if b["transaction_count"]][:10]:
            mark = " (this block)" if b["is_target"] else ""
            ppsm = f"${b['median_price_per_sqm']:,.0f}" if b.get("median_price_per_sqm") else "n/a"
            lease = f"{b['median_remaining_lease_years']:.0f} yrs" if b.get("median_remaining_lease_years") else "n/a"
            lines.append(f"| Blk {b['block']} {b['street']}{mark} | {b['distance_meters']:.0f} m | {b['transaction_count']} | "
                         f"{_m(b.get('median_price'))} | {ppsm} | {lease} |")
        if near["blocks_with_sales"] > 10:
            lines.append(f"- Showing the 10 nearest of {near['blocks_with_sales']} blocks with sales; all are on the map.")
    return lines


def _as_list(value: Any) -> List[Dict[str, Any]]:
    return [value] if isinstance(value, dict) else list(value or [])


def research_map_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    sql = state.get("sql_market_results") or {}
    loc = state.get("location_results") or {}

    def point(lat: float, lng: float, props: Dict[str, Any]) -> None:
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lng, lat]}, "properties": props})

    places: List[Dict[str, Any]] = []
    for o in (sql.get("compare_resale_options") or {}).get("options", []) or []:
        if o.get("location"):
            places.append(o["location"])
            point(o["location"]["lat"], o["location"]["lng"], {"name": o["location"]["name"], "category": "hdb_candidate",
                                                               "median_resale_price": o.get("median_price"),
                                                               "remaining_lease_years": o.get("median_remaining_lease_years")})
    for key in ("assess_resale_price", "get_block_transaction_history"):
        for r in _as_list(sql.get(key)):
            if r.get("location") and r.get("scope", "block") == "block":
                places.append(r["location"])
                point(r["location"]["lat"], r["location"]["lng"], {"name": r["location"]["name"], "category": "hdb_candidate"})
    area_radius = None
    near = sql.get("find_resale_blocks_near") or {}
    if near.get("status") == "ok":
        places.append(near["location"])
        area_radius = near["radius_meters"]
        for b in near["blocks"]:
            if b["transaction_count"]:
                point(b["lat"], b["lng"], {"name": f"Blk {b['block']} {b['street']}", "category": "hdb_candidate",
                                           "median_resale_price": b.get("median_price"),
                                           "remaining_lease_years": b.get("median_remaining_lease_years"),
                                           "transaction_count": b["transaction_count"], "is_target": b["is_target"]})
        centre = near["location"]
        if centre["kind"] in ("mrt", "hawker", "supermarket", "school"):  # a pin or block is already on the map
            point(centre["lat"], centre["lng"], {"name": centre["name"], "category": centre["kind"],
                                                 "distance_meters": 0, "walking_mins": 0})
    stats = sql.get("query_resale_statistics")
    if isinstance(stats, dict) and stats.get("status") == "ok":
        c_now = state.get("user_constraints") or {}
        kind = "station" if stats["group_by"] == ["station"] else "town" if stats["group_by"] == ["town"] else "band"
        for g in stats["groups"]:
            if kind != "band" and g.get("lat") is not None:
                places.append({"lat": g["lat"], "lng": g["lng"]})
                point(g["lat"], g["lng"], {"name": g["label"], "category": "hdb_candidate", "kind": kind,
                                           "median_resale_price": g["median_price"], "transaction_count": g["count"],
                                           "drill_question": stats_drill_text(stats, g, c_now)})
        for b in stats.get("map_blocks") or []:
            places.append({"lat": b["lat"], "lng": b["lng"]})
            point(b["lat"], b["lng"], {"name": b["block_key"].title(), "category": "hdb_candidate", "kind": "block",
                                       "band": b["group"], "median_resale_price": b["median_price"],
                                       "transaction_count": b["count"], "town": b["town"]})
    for r in _as_list(loc.get("get_nearby_amenities")):
        if r.get("status") == "ok":
            places.append(r["location"])
            point(r["location"]["lat"], r["location"]["lng"], {"name": r["location"]["name"], "category": "hdb_candidate"})
            for a in r["amenities"]:
                point(a["lat"], a["lng"], {"name": a["name"], "category": a["category"],
                                           "distance_meters": a["distance_meters"], "walking_mins": a["walking_minutes"]})
    if not features:
        return {}
    lat = sum(p["lat"] for p in places) / len(places) if places else features[0]["geometry"]["coordinates"][1]
    lng = sum(p["lng"] for p in places) / len(places) if places else features[0]["geometry"]["coordinates"][0]
    payload: Dict[str, Any] = {"markers": {"type": "FeatureCollection", "features": features},
                               "viewport": {"lat": lat, "lng": lng, "zoom": 15 if len(places) == 1 else 12.5},
                               "candidate_blocks": [p["name"] for p in places if p.get("kind") == "hdb_block"]}
    if area_radius:
        payload["candidate_blocks"] = [f"Blk {b['block']} {b['street']}" for b in near["blocks"] if b["transaction_count"]]
        payload["viewport"]["zoom"] = 16 if area_radius <= 300 else 15 if area_radius <= 700 else 14
    radius = area_radius or next((r["radius_meters"] for r in _as_list(loc.get("get_nearby_amenities"))
                                  if r.get("status") == "ok"), None)
    places = list({(p["lat"], p["lng"]): p for p in places}.values())
    if radius and len(places) == 1:
        from mcp_server.spatial_onemap_tools import create_buffer_geojson
        payload["buffer_polygon"] = create_buffer_geojson(lat=lat, lng=lng, radius_meters=float(radius))
    return payload


def place_clarification(c: Dict[str, Any]) -> str:
    place = (c.get("addresses") or ["your pinned location"])[0]
    return (f"What would you like to know about {place}? For example: its past sales, resale prices around it "
            f"(say a radius, e.g. 'within 500m'), the price trend nearby, whether an asking price is fair, or "
            f"what amenities are near it.")


# ---------------------------------------------------------------------------
# Fast path: questions the regex fully understands need no model round trip
# ---------------------------------------------------------------------------
FAST_INTENTS = ("wants_group_stats", "wants_data_catalog", "wants_nearby_amenities", "wants_block_history", "wants_price_assessment", "wants_area_statistics",
                "wants_option_comparison", "wants_market_snapshot", "wants_town_ranking", "wants_price_trend")


def answerable_without_model(c: Dict[str, Any], query: str) -> bool:
    """True when every tool the question needs is fixed by the extracted intents, so the
    deterministic plan is exactly what a model would call. Anything with financial facts,
    policy, commutes, school radii, comparables or a classic filter search keeps the model,
    as do questions where no intent was recognised (the model may understand the phrasing)."""
    needs_model = (c.get("financial_inputs") or c.get("household_income") or c.get("cpf_oa_balance")
                   or c.get("cash_savings") or c.get("citizenship_scheme") or c.get("policy_question")
                   or c.get("work_destinations") or c.get("school_name") or c.get("comparable_resale")
                   or (c.get("wants_bto_comparison") and not c.get("bto_project_name")))
    if needs_model:
        return False
    if not (any(c.get(k) for k in FAST_INTENTS) or c.get("bto_project_name")):
        return False
    return not is_block_search(c, query)
