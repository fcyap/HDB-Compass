"""Resolve user-supplied places and summarise what is near them (mcp_server/location_tools.py).

Agents must never guess coordinates. Every tool that needs a location calls
resolve_place() on the user's own text: an HDB block address or postal code from
the spatial index, or a named MRT/LRT station, hawker centre, supermarket or
primary school. OneMap search is used only as a last resort, and only when
OneMap credentials are configured. Ambiguity is returned, never silently resolved.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from agents.research_schemas import (
    GetNearbyAmenitiesInput,
    GetNearbyAmenitiesOutput,
    LocationCandidate,
    NearbyAmenity,
    ResolveLocationInput,
    ResolveLocationOutput,
)
from mcp_server.spatial_onemap_tools import geocode_onemap, get_spatial_index, wgs84_to_svy21

WALKING_METERS_PER_MINUTE = 80.0
CATEGORIES = ("mrt", "hawker", "supermarket", "school")

# Both the query and the index are normalised with the same abbreviations.
_ALIASES = {
    "STREET": "ST", "ROAD": "RD", "AVENUE": "AVE", "DRIVE": "DR", "CRESCENT": "CRES",
    "NORTH": "NTH", "SOUTH": "STH", "BUKIT": "BT", "UPPER": "UPP", "LORONG": "LOR",
    "JALAN": "JLN", "TAMAN": "TMN", "PLACE": "PL", "CLOSE": "CL", "LANE": "LN",
    "CENTRAL": "CTRL", "TERRACE": "TER", "WALK": "WK", "HEIGHTS": "HTS", "PARK": "PK",
    "GARDENS": "GDNS", "MOUNT": "MT", "SAINT": "ST", "CENTER": "CENTRE",
}
_IGNORED = {"BLK", "BLOCK", "IN", "AT", "SINGAPORE", "POSTAL", "CODE", "THE", "OF", "NEAR", "S"}
# Words that say what kind of place it is, not which one.
_KIND_WORDS = {
    "mrt": {"MRT", "LRT", "STATION", "INTERCHANGE"},
    "school": {"PRIMARY", "SCHOOL"},
    "hawker": {"HAWKER", "FOOD", "CENTRE", "MARKET"},
    "supermarket": {"SUPERMARKET"},
}
_GENERIC = set().union(*_KIND_WORDS.values())
_BLOCK_ID = re.compile(r"^\d{1,4}[A-Z]?$")


def tokens(value: str) -> set:
    return {_ALIASES.get(t, t) for t in re.findall(r"[A-Z0-9]+", (value or "").upper()) if t not in _IGNORED}


def display_name(raw: str, category: str) -> str:
    """Consumer-facing names for registry entries such as 'COLD STORAGE SINGAPORE (1983) PTE LTD'."""
    name = (raw or "").replace("?", " ").strip()
    if category == "supermarket":
        # Licence names carry unit numbers and legal suffixes: keep the brand and a readable branch.
        name = re.sub(r"\([^()]*#[^()]*\)|\(UNITS?:[^()]*\)", " ", name, flags=re.I)
        branches = [b for b in re.findall(r"\(([^()]*)\)", name)
                    if not re.search(r"^(?:SINGAPORE|1983)$|CO-OPERATIVE|PTE|LTD|LIMITED", b.strip(), re.I)]
        name = re.sub(r"\([^()]*\)", " ", name)
        name = re.sub(r"\b(?:SINGAPORE|CO-OPERATIVE|PTE\.?|LTD\.?|LIMITED|CO\.?)\b", " ", name, flags=re.I)
        upper = name.upper()
        for brand, label in (("FAIRPRICE", "NTUC FairPrice"), ("COLD STORAGE", "Cold Storage"),
                             ("SHENG SIONG", "Sheng Siong"), ("GIANT", "Giant"),
                             ("DON DON DONKI", "Don Don Donki"), ("PRIME SUPERMARKET", "Prime Supermarket")):
            if brand in upper:
                name = label
                break
        else:
            name = " ".join(re.sub(r"\s+[.,]+(?=\s|$)", " ", name).split()).title()
        branch = " ".join(branches[0].split()).title() if branches else ""
        return f"{name} ({branch})" if branch else name
    return " ".join(name.split()).title()


def _candidate(kind: str, record: Dict[str, Any], score: float, name: Optional[str] = None) -> LocationCandidate:
    if kind == "hdb_block":
        name = f"Blk {record['block']} {record['street']}"
    return LocationCandidate(
        kind=kind, name=name or display_name(record.get("name", ""), kind),
        block=record.get("block") if kind == "hdb_block" else None,
        street=record.get("street") if kind == "hdb_block" else None,
        town=record.get("town"), postal=record.get("postal"),
        lat=float(record["lat"]), lng=float(record["lng"]),
        source="local spatial index", score=round(score, 3),
    )


def _usable(record: Dict[str, Any]) -> bool:
    lat, lng = record.get("lat"), record.get("lng")
    return (isinstance(lat, (int, float)) and isinstance(lng, (int, float))
            and math.isfinite(lat) and math.isfinite(lng) and 1.1 <= lat <= 1.5 and 103.5 <= lng <= 104.1)


def _match_blocks(query_tokens: set, payload: Dict[str, Any]) -> List[LocationCandidate]:
    postal = next((t for t in query_tokens if re.fullmatch(r"\d{6}", t)), None)
    if postal:
        return [_candidate("hdb_block", b, 1.0) for b in payload.get("blocks", [])
                if str(b.get("postal")) == postal and _usable(b)]
    identifiers = [t for t in query_tokens if _BLOCK_ID.match(t)]
    found: Dict[Tuple[str, str], LocationCandidate] = {}
    for identifier in identifiers:
        rest = query_tokens - {identifier}
        if not rest:
            continue
        for b in payload.get("blocks", []):
            street = str(b.get("street") or "")
            if str(b.get("block", "")).upper() != identifier or not street or street.upper().startswith("STREET IN "):
                continue
            street_tokens = tokens(street)
            if rest <= street_tokens | tokens(str(b.get("town") or "")) and _usable(b):
                score = len(rest & street_tokens) / len(street_tokens)
                found.setdefault((b["block"], street), _candidate("hdb_block", b, score))
        if found:
            break
    return sorted(found.values(), key=lambda c: (-c.score, c.name))


def _match_named(query_tokens: set, payload: Dict[str, Any]) -> List[LocationCandidate]:
    kinds = [k for k, words in _KIND_WORDS.items() if query_tokens & words] or list(CATEGORIES)
    core = query_tokens - _GENERIC
    if not core:
        return []
    pool: List[Tuple[str, Dict[str, Any], List[str]]] = []
    for a in payload.get("amenities", []):
        if a.get("category") in kinds:
            pool.append((a["category"], a, [a.get("name", "")]))
    if "school" in kinds:
        pool.extend(("school", s, [s.get("name", ""), s.get("alias", "")]) for s in payload.get("schools", []))
    found: Dict[str, LocationCandidate] = {}
    for kind, record, names in pool:
        best = 0.0
        for name in filter(None, names):
            name_tokens = tokens(name) - _GENERIC
            if name_tokens and core <= name_tokens:
                best = max(best, len(core) / len(name_tokens))
        if best and _usable(record):
            c = _candidate(kind, record, best)
            if c.name not in found or found[c.name].score < best:
                found[c.name] = c
    return sorted(found.values(), key=lambda c: (-c.score, c.name))


def resolve_place(query: str, limit: int = 5) -> Dict[str, Any]:
    args = ResolveLocationInput(query=query, limit=limit)
    try:
        payload = get_spatial_index()["payload"]
    except (FileNotFoundError, OSError):
        return ResolveLocationOutput(status="unavailable", query=args.query,
                                     message="The local spatial index is unavailable; rebuild it first.").model_dump()
    query_tokens = tokens(args.query)
    candidates = _match_blocks(query_tokens, payload) or _match_named(query_tokens, payload)
    if not candidates and os.getenv("ONEMAP_EMAIL") and os.getenv("ONEMAP_PASSWORD"):
        geo = geocode_onemap(args.query)
        if geo and _usable(geo):
            candidates = [LocationCandidate(kind="onemap", name=geo.get("address") or geo.get("name") or args.query,
                                            postal=geo.get("postal"), lat=geo["lat"], lng=geo["lng"],
                                            source="OneMap search (first result)", score=0.6)]
    if not candidates:
        return ResolveLocationOutput(
            status="not_found", query=args.query,
            message="No indexed HDB block, MRT/LRT station, hawker centre, supermarket or primary school matches. "
                    "Ask for the block number and street, or a six-digit postal code.").model_dump()
    top = [c for c in candidates if c.score == candidates[0].score]
    # A weak partial match ("good school" inside a longer school name) needs confirmation.
    if len(top) == 1 and top[0].score >= 0.5:
        return ResolveLocationOutput(status="ok", query=args.query, candidates=[top[0]],
                                     message=f"Resolved to {top[0].name} ({top[0].source}).").model_dump()
    return ResolveLocationOutput(
        status="ambiguous", query=args.query, candidates=candidates[:args.limit],
        message=f"{len(candidates)} places match; ask the user which one they mean.").model_dump()


def resolve_location(query: str, limit: int = 5) -> Dict[str, Any]:
    """Resolve a user-supplied place to verified coordinates before any location-based analysis.

    Accepts an HDB block and street ('Blk 288B Jurong East St 21'), a six-digit postal
    code, or a named MRT/LRT station, hawker centre, supermarket or primary school.
    Returns status ok with one candidate, ambiguous with several (ask the user), or
    not_found. Use ONLY these coordinates (or the user's map pin) for other tools;
    never estimate coordinates or substitute a town centre.
    """
    return resolve_place(query, limit)


def nearest_block(lat: float, lng: float, max_meters: float = 60.0) -> Optional[Dict[str, Any]]:
    """The indexed HDB block nearest a map pin, if one lies within max_meters."""
    try:
        index = get_spatial_index()
    except (FileNotFoundError, OSError):
        return None
    blocks = index["payload"].get("blocks", [])
    x, y = wgs84_to_svy21(lat, lng)
    tree = index.get("block_kdtree")
    if tree is not None and blocks:
        d, i = tree.query([x, y])
        return {**blocks[int(i)], "distance_meters": round(float(d), 1)} if d <= max_meters else None
    best, best_d = None, max_meters
    for b in blocks:
        if not _usable(b):
            continue
        bx, by = (b["x"], b["y"]) if "x" in b else wgs84_to_svy21(b["lat"], b["lng"])
        d = math.hypot(bx - x, by - y)
        if d <= best_d:
            best, best_d = b, d
    return None if best is None else {**best, "distance_meters": round(best_d, 1)}


def _amenity_records(payload: Dict[str, Any], categories: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    records = [(a["category"], a) for a in payload.get("amenities", []) if a.get("category") in categories]
    if "school" in categories:
        records.extend(("school", s) for s in payload.get("schools", []))
    return records


def amenities_near(lat: float, lng: float, radius_meters: float, categories: List[str],
                   max_per_category: int = 5) -> Tuple[Dict[str, int], Dict[str, Optional[NearbyAmenity]], List[NearbyAmenity]]:
    payload = get_spatial_index()["payload"]
    x, y = wgs84_to_svy21(lat, lng)
    by_category: Dict[str, List[NearbyAmenity]] = {c: [] for c in categories}
    seen = set()
    for category, record in _amenity_records(payload, categories):
        name = display_name(record.get("name", ""), category)
        rx, ry = (record["x"], record["y"]) if "x" in record else wgs84_to_svy21(record["lat"], record["lng"])
        distance = math.hypot(rx - x, ry - y)
        if (category, name.upper(), round(distance)) in seen:
            continue
        seen.add((category, name.upper(), round(distance)))
        by_category[category].append(NearbyAmenity(
            name=name, category=category, lat=record["lat"], lng=record["lng"],
            distance_meters=round(distance, 1),
            walking_minutes=max(1, round(distance / WALKING_METERS_PER_MINUTE))))
    counts, nearest, within = {}, {}, []
    for category, items in by_category.items():
        items.sort(key=lambda a: (a.distance_meters, a.name))
        inside = [a for a in items if a.distance_meters <= radius_meters]
        counts[category] = len(inside)
        nearest[category] = items[0] if items else None
        within.extend(inside[:max_per_category])
    within.sort(key=lambda a: a.distance_meters)
    return counts, nearest, within


def get_nearby_amenities(
    address: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_meters: int = 1000,
    categories: Optional[List[str]] = None,
    max_per_category: int = 5,
) -> Dict[str, Any]:
    """List MRT/LRT stations, hawker centres, supermarkets and primary schools near a place.

    Prefer the user's address text; the tool resolves it and returns ambiguity rather
    than guessing. Coordinates are accepted only from resolve_location or a map pin.
    Reports counts within the straight-line radius, the nearest of each category
    (even beyond the radius) and walking-time estimates at 80 m/min. Clinics, parks
    and bus stops are not in the local index.
    """
    args = GetNearbyAmenitiesInput(address=address, latitude=latitude, longitude=longitude,
                                   radius_meters=radius_meters, categories=categories or [],
                                   max_per_category=max_per_category)
    location: Optional[LocationCandidate] = None
    if args.address:
        resolved = resolve_place(args.address)
        if resolved["status"] != "ok":
            return GetNearbyAmenitiesOutput(status=resolved["status"], message=resolved["message"],
                                            candidates=resolved["candidates"],
                                            radius_meters=args.radius_meters).model_dump()
        location = LocationCandidate(**resolved["candidates"][0])
        lat, lng = location.lat, location.lng
    else:
        lat, lng = args.latitude, args.longitude
        location = LocationCandidate(kind="onemap", name=f"Pinned location ({lat:.5f}, {lng:.5f})",
                                     lat=lat, lng=lng, source="supplied coordinates", score=1.0)
    try:
        counts, nearest, within = amenities_near(lat, lng, args.radius_meters, list(args.categories or CATEGORIES),
                                                 args.max_per_category)
    except (FileNotFoundError, OSError):
        return GetNearbyAmenitiesOutput(status="unavailable", message="The local spatial index is unavailable.",
                                        radius_meters=args.radius_meters).model_dump()
    return GetNearbyAmenitiesOutput(
        status="ok", location=location, radius_meters=args.radius_meters, counts=counts, nearest=nearest,
        amenities=within,
        message=("Straight-line distances from indexed points; walking minutes assume 80 m/min. "
                 "Clinics, parks and bus stops are not covered by the local index."),
    ).model_dump()
