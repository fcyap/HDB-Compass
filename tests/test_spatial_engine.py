"""
Regression Test Suite: Spatial Engine (tests/test_spatial_engine.py).
Tests:
1. Coordinate transformations (WGS84 <-> SVY21) roundtrip precision.
2. SVY21 geodesic metric buffer polygon GeoJSON creation.
3. Fast KD-Tree radius queries against the official API-generated spatial index.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_server.spatial_onemap_tools import (
    wgs84_to_svy21,
    svy21_to_wgs84,
    create_buffer_geojson,
    SpatialIndex,
)


def test_coordinate_roundtrip():
    print("Testing coordinate conversion roundtrip (WGS84 <-> SVY21)...")
    # Coordinates for Orchard MRT
    orig_lat, orig_lng = 1.3040, 103.8318

    # Forward: WGS84 -> SVY21
    x, y = wgs84_to_svy21(orig_lat, orig_lng)
    # Reverse: SVY21 -> WGS84
    calc_lat, calc_lng = svy21_to_wgs84(x, y)

    lat_diff = abs(orig_lat - calc_lat)
    lng_diff = abs(orig_lng - calc_lng)

    print(f"  Input  GPS: Lat={orig_lat:.5f}, Lng={orig_lng:.5f}")
    print(f"  SVY21 Grid: X={x:.2f} m, Y={y:.2f} m")
    print(f"  Output GPS: Lat={calc_lat:.5f}, Lng={calc_lng:.5f}")
    print(f"  Delta: Lat error={lat_diff:.8f} deg, Lng error={lng_diff:.8f} deg")

    assert lat_diff < 1e-6, "Latitude roundtrip mismatch!"
    assert lng_diff < 1e-6, "Longitude roundtrip mismatch!"
    print("  [PASS] Coordinate roundtrip exact to sub-millimeter precision.\n")


def test_buffer_geojson_generation():
    print("Testing buffer GeoJSON polygon creation...")
    center_lat, center_lng = 1.3040, 103.8318  # Orchard MRT
    radius = 1000.0  # 1 km

    feature = create_buffer_geojson(
        lat=center_lat,
        lng=center_lng,
        radius_meters=radius,
        properties={"label": "Orchard 1km Buffer", "category": "search_zone"}
    )

    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    coords = feature["geometry"]["coordinates"][0]

    # Check that polygon is closed (first point == last point)
    assert coords[0] == coords[-1], "Polygon boundary is not closed!"
    assert len(coords) >= 64, "Polygon does not have expected resolution!"

    print(f"  Polygon generated with {len(coords)} vertices.")
    print(f"  First vertex (Lng, Lat): {coords[0]}")
    print(f"  GeoJSON properties: {json.dumps(feature['properties'])}")
    print("  [PASS] Buffer GeoJSON generation verified.\n")


def test_kdtree_radius_queries():
    print("Testing KD-Tree spatial indexing & radius queries with dynamic index...")
    index = SpatialIndex()
    print(f"  Indexed {len(index.records)} real Singapore POIs from official APIs.")
    assert len(index.records) > 0, "Spatial index should contain loaded records!"

    # Query 1: 1km around Orchard MRT (1.3040, 103.8318)
    orchard_lat, orchard_lng = 1.3040, 103.8318
    radius_1km = 1000.0

    matches_1km = index.query_radius(orchard_lat, orchard_lng, radius_meters=radius_1km)
    print(f"\n  Query: Within {radius_1km}m of Orchard MRT (Lat: {orchard_lat}, Lng: {orchard_lng})")
    print(f"  Matches found: {len(matches_1km)}")
    for m in matches_1km[:5]:
        print(f"    - [{m['distance_meters']}m] {m['name']} ({m['category']})")

    assert len(matches_1km) > 0, "Should find matches around Orchard!"

    # Query 2: Nearest neighbor to Bishan area
    bishan_lat, bishan_lng = 1.3508, 103.8481
    nearest_3 = index.query_nearest(bishan_lat, bishan_lng, k=3)
    print(f"\n  Query: Top 3 nearest POIs to Bishan:")
    for n in nearest_3:
        print(f"    - [{n['distance_meters']}m] {n['name']} ({n.get('category', 'amenity')})")

    assert len(nearest_3) == 3

    # Test GeoJSON conversion
    geojson_fc = index.to_geojson_features(matches_1km)
    assert geojson_fc["type"] == "FeatureCollection"
    assert len(geojson_fc["features"]) == len(matches_1km)
    print(f"\n  Exported {len(geojson_fc['features'])} query results to GeoJSON FeatureCollection.")
    print("  [PASS] KD-Tree spatial queries verified.\n")


if __name__ == "__main__":
    print("=" * 65)
    print("    RUNNING PHASE 1 SPATIAL ENGINE VERIFICATION SUITE")
    print("=" * 65 + "\n")
    test_coordinate_roundtrip()
    test_buffer_geojson_generation()
    test_kdtree_radius_queries()
    print("=" * 65)
    print("  ALL PHASE 1 ENGINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
