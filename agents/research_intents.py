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
                         "find_resale_blocks_near", "get_area_resale_statistics", "get_area_price_trend")
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
    "history": r"\b(histor(?:y|ical)|past (?:sales|transactions|prices)|sold for|previous(?:ly)? sold|recent sales|"
               r"transactions? (?:at|for|in)|transacted)\b",
    "assessment": r"\b(fair(?:ly)?(?: price)?|reasonable|over-?priced|under-?priced|good (?:price|deal)|worth|"
                  r"valuation|too (?:much|high|expensive)|asking|offer(?:ed)?|is \$\s?\d)",
    "comparison": r"\b(vs\.?|versus|compare|comparison|compared|which is better|better (?:choice|option|buy))\b|\bor\b",
    "nearby": r"\b(what(?:'s|s| is| are)?\s+(?:near(?:by)?|around|close to)|nearby (?:amenities|facilities)|"
              r"amenities|facilities|what is around|near it|around it)\b",
    "snapshot": r"\b(market (?:now|today|currently|like|status|update|conditions?|overview|snapshot)|current(?:ly)? market|"
                r"state of the (?:resale )?market|where (?:is|are) (?:the )?(?:resale )?(?:market|prices)|"
                r"latest (?:resale )?prices|how (?:is|are) (?:the )?(?:resale |hdb )?(?:market|prices)|"
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


def extract_research_fields(query: str, base: Dict[str, Any]) -> Dict[str, Any]:
    """Research fields for ExtractedConstraints, given the classic regex fields in `base`."""
    q = query
    out: Dict[str, Any] = {"assumptions": [], "clarifications": []}
    out.update(_place_intents(q, extract_addresses(q)))
    out["wants_market_snapshot"] = _has("snapshot", q)
    out["wants_town_ranking"] = _has("ranking", q) and not base.get("towns")
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
                                   "wants_market_snapshot", "wants_town_ranking", "wants_area_statistics"))
    stated = c.get("max_budget_sgd") or c.get("min_remaining_lease_years") or \
        {k: v for k, v in (c.get("amenity_radii_meters") or {}).items() if k != "school"}
    return bool(stated) or _has("search", query) or not other


def needs_market_agent(c: Dict[str, Any]) -> bool:
    has_blocks = any(re.match(r"(?:blk\s+)?\d", a, re.I) for a in c.get("addresses") or [])
    if c.get("wants_area_statistics") and (c.get("addresses") or c.get("map_pin")):
        return True
    return bool(c.get("wants_market_snapshot") or c.get("wants_town_ranking") or c.get("bto_project_name")
                or (c.get("wants_price_trend") and not c.get("towns"))
                or (has_blocks and (c.get("wants_block_history") or c.get("wants_price_assessment")
                                    or c.get("wants_option_comparison"))))


def needs_spatial_agent(c: Dict[str, Any]) -> bool:
    return bool(c.get("wants_nearby_amenities") and (c.get("addresses") or c.get("map_pin")))


def _flat_type(c: Dict[str, Any], assumptions: List[str], purpose: str) -> str:
    types = c.get("flat_types") or []
    if types:
        return types[0]
    assumptions.append(f"No flat type was given, so {purpose} uses {DEFAULT_FLAT_TYPE} flats.")
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


def research_plan(agent: str, c: Dict[str, Any], query: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Deterministic calls for the research intents; assumptions are recorded in c['assumptions']."""
    plan: List[Tuple[str, Dict[str, Any]]] = []
    assumptions: List[str] = c.setdefault("assumptions", [])
    blocks = [a for a in c.get("addresses") or [] if re.match(r"(?:blk\s+)?\d", a, re.I)]
    if agent == "market_sql":
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

    for note in c.get("assumptions") or []:
        lines.append(f"_Assumption: {note}_")
    for question in c.get("clarifications") or []:
        lines.append(f"**Question:** {question}")
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
FAST_INTENTS = ("wants_nearby_amenities", "wants_block_history", "wants_price_assessment", "wants_area_statistics",
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
