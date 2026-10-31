"""Address-centred school requests and evidence-only result presentation."""

import re

from mcp_server.spatial_onemap_tools import create_buffer_geojson


def school_location_request(query):
    # Direction matters: schools near an address is different from flats near a school.
    match = re.search(
        r"^\s*(?:(?:please|find|show|list|me|the|what|which|are|there|any|primary|secondary)\s+)*"
        r"schools?\s+(?:are\s+)?(?:near(?:by)?|around|close to|within\s+"
        r"(?P<radius>\d+(?:\.\d+)?)\s*(?P<unit>km|m)\s+of)\s+(?P<address>.+)",
        query, re.I,
    )
    if not match:
        return None
    address = match["address"].strip().rstrip("?.!")
    radius = float(match["radius"]) * (1000 if match["unit"].lower() == "km" else 1) if match["radius"] else 2000
    trailing_radius = re.search(r"\s+within\s+(\d+(?:\.\d+)?)\s*(km|m)$", address, re.I)
    if trailing_radius:
        radius = float(trailing_radius[1]) * (1000 if trailing_radius[2].lower() == "km" else 1)
        address = address[:trailing_radius.start()].strip()
    return {"address": address, "radius_meters": radius}


def present_school_result(result):
    """Use the same structured evidence for the answer and map; never add resale cards."""
    payload = {"markers": {"type": "FeatureCollection", "features": []}}
    if result.get("status") != "ok":
        lines = [result.get("message") or "The school lookup is unavailable. Please try again after checking the spatial index."]
        for block in result.get("address_candidates", []):
            lines.append(f"- Blk {block['block']} {block['street']} ({block.get('postal') or block['town']})")
        return "\n".join(lines), payload

    location = result["location"]
    radius = result["radius_meters"]
    schools = result["schools"]
    name = f"Blk {location['block']} {location['street']}"
    lines = [f"**Primary schools near {name}**", f"Search radius: {radius:g} m, straight-line."]
    if schools:
        lines.extend(f"- {s['name']}: {s['distance_meters']:.0f} m — {s['address']}" for s in schools)
    else:
        lines.append("No primary schools in the local index fall within this radius.")
    lines.extend([result["message"], result["source"] + "."])
    features = [{
        "type": "Feature", "geometry": {"type": "Point", "coordinates": [location["lng"], location["lat"]]},
        "properties": {"name": name, "category": "hdb_location"},
    }]
    features.extend({
        "type": "Feature", "geometry": {"type": "Point", "coordinates": [s["lng"], s["lat"]]},
        "properties": {"name": s["name"], "category": "school", "distance_meters": s["distance_meters"]},
    } for s in schools)
    payload.update(
        markers={"type": "FeatureCollection", "features": features},
        buffer_polygon=create_buffer_geojson(location["lat"], location["lng"], radius),
        viewport={"lat": location["lat"], "lng": location["lng"], "zoom": 14},
    )
    return "\n".join(lines), payload
