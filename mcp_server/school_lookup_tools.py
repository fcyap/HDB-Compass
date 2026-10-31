"""Resolve a residential address before querying local primary-school proximity."""

import math
import re

from agents.schemas import (
    BlockRef,
    FindSchoolsNearAddressInput,
    FindSchoolsNearAddressOutput,
    NearbySchoolRecord,
)
from mcp_server.spatial_onemap_tools import get_spatial_index, wgs84_to_svy21


def address_tokens(value):
    aliases = {"STREET": "ST", "ROAD": "RD", "AVENUE": "AVE", "DRIVE": "DR",
               "CRESCENT": "CRES", "NORTH": "NTH", "SOUTH": "STH"}
    ignored = {"BLK", "BLOCK", "IN", "AT", "SINGAPORE", "POSTAL", "CODE"}
    return {aliases.get(t, t) for t in re.findall(r"[A-Z0-9]+", value.upper()) if t not in ignored}


def find_primary_schools_near_address(address: str, radius_meters: float = 2000):
    """Find primary schools near one indexed HDB address. Resolve ambiguity before
    measuring straight-line distances. Unknown/placeholder addresses require clarification;
    an empty school list is meaningful only when status is ok. No walking-route or
    official registration-boundary determination is provided.
    """
    args = FindSchoolsNearAddressInput(address=address, radius_meters=radius_meters)
    result = FindSchoolsNearAddressOutput(
        status="needs_information", radius_meters=args.radius_meters,
        message="Please provide the full block and street address, or a six-digit postal code.",
    )
    tokens = address_tokens(args.address)
    identifier = re.search(r"\b\d+[A-Z]?\b", args.address.upper())
    if identifier is None:
        return result.model_dump()
    try:
        payload = get_spatial_index()["payload"]
    except (FileNotFoundError, OSError):
        result.status = "unavailable"
        result.message = "The local spatial index is unavailable. Rebuild it before searching for nearby schools."
        return result.model_dump()

    matches = {}
    for block in payload.get("blocks", []):
        identity_field = "postal" if re.fullmatch(r"\d{6}", identifier[0]) else "block"
        if str(block.get(identity_field, "")).upper() != identifier[0]:
            continue
        identity = " ".join(str(block.get(k) or "") for k in ("block", "street", "town", "postal"))
        if tokens <= address_tokens(identity):
            key = tuple(block.get(k) for k in ("block", "street", "postal", "lat", "lng"))
            matches[key] = block
    if not matches:
        result.message = (
            "I couldn't resolve that address in the local index. This does not mean there are no schools nearby. "
            "Please provide the full block and street address, or a six-digit postal code."
        )
        return result.model_dump()
    if len(matches) > 1:
        result.address_candidates = [
            BlockRef(**b) for b in matches.values()
            if b.get("street") and not b["street"].upper().startswith("STREET IN ")
        ][:10]
        result.message = (
            f"{len(matches)} indexed addresses match; up to 10 named addresses are shown. "
            "Please resend the school search with one full address or postal code."
        )
        return result.model_dump()

    block = next(iter(matches.values()))
    if not block.get("street") or block["street"].upper().startswith("STREET IN "):
        result.message = (
            "The matching index entry has a placeholder street name, so I can't verify this address. "
            "Please provide the actual street address; this entry needs address verification before I can calculate nearby schools."
        )
        return result.model_dump()
    lat, lng = block.get("lat"), block.get("lng")
    if (not isinstance(lat, (int, float)) or not isinstance(lng, (int, float))
            or not math.isfinite(lat) or not math.isfinite(lng)
            or not (1 <= lat <= 1.6 and 103.5 <= lng <= 104.2)):
        result.status = "unavailable"
        result.message = "The address has no usable coordinates in the local index. Its location needs verification."
        return result.model_dump()
    if not payload.get("schools"):
        result.status = "unavailable"
        result.message = "The local index has no primary-school data. Rebuild it before searching."
        return result.model_dump()

    x, y = wgs84_to_svy21(lat, lng)
    for school in payload["schools"]:
        sx, sy = wgs84_to_svy21(school["lat"], school["lng"])
        distance = math.hypot(sx - x, sy - y)
        if distance <= args.radius_meters:
            result.schools.append(NearbySchoolRecord(
                name=school["name"], address=school.get("address", ""),
                lat=school["lat"], lng=school["lng"], distance_meters=round(distance, 1),
            ))
    result.schools.sort(key=lambda s: (s.distance_meters, s.name))
    result.status = "ok"
    result.location = BlockRef(**block)
    result.message = (
        "Distances use indexed points and a straight-line radius, not walking routes or official MOE registration boundaries. "
        "Only primary schools in the local snapshot are covered."
    )
    return result.model_dump()
