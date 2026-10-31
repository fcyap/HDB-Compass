"""
FastMCP Spatial Tools Regression Suite (tests/test_spatial_tools.py).
Tests all 5 FastMCP Spatial Tools mandated for Member 2 (Proposal Pages 8 & 13):
1. verify_primary_school_distance(school_name, bracket)
2. filter_blocks_by_amenity_proximity(towns, amenity_constraints)
3. compute_dual_transit_commute(candidate_blocks, work_dest_a, work_dest_b, max_minutes)
4. sync_interactive_map_ui(candidate_blocks, buffer_radii, center_latlon)
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_server.spatial_onemap_tools import (
    verify_primary_school_distance,
    filter_blocks_by_amenity_proximity,
    compute_dual_transit_commute,
    sync_interactive_map_ui,
)


def run_tests():
    print("=" * 70)
    print("     TESTING FASTMCP SPATIAL TOOLS (MEMBER 2 DELIVERABLES)")
    print("=" * 70)

    # Test 1: Catholic High Primary Priority Admission
    print("\n[TEST 1] verify_primary_school_distance('Catholic High', bracket='1km'):")
    catholic_blocks = verify_primary_school_distance("Catholic High", bracket="1km")
    print(f"  -> Qualifying HDB Blocks within 1km: {len(catholic_blocks)}")
    for b in catholic_blocks[:5]:
        print(f"     * {b}")
    assert len(catholic_blocks) > 0, "Catholic High should have blocks within 1km in Bishan!"
    print("  [PASS] School priority boundary verified.")

    # Test 2: Ai Tong School Priority Admission
    print("\n[TEST 2] verify_primary_school_distance('Ai Tong School', bracket='1km'):")
    aitong_blocks = verify_primary_school_distance("Ai Tong School", bracket="1km")
    print(f"  -> Qualifying HDB Blocks within 1km: {len(aitong_blocks)}")
    for b in aitong_blocks[:5]:
        print(f"     * {b}")
    assert len(aitong_blocks) > 0, "Ai Tong School should have qualifying blocks in Sin Ming/Bishan!"
    print("  [PASS] Ai Tong School priority boundary verified.")

    # Test 3: Multi-layer Amenity Proximity Filter
    print("\n[TEST 3] filter_blocks_by_amenity_proximity(towns=['ANG MO KIO'], amenity_constraints={'mrt': 400, 'hawker': 300}):")
    amk_blocks = filter_blocks_by_amenity_proximity(
        towns=["ANG MO KIO"],
        amenity_constraints={"mrt": 400, "hawker": 300}
    )
    print(f"  -> Matching Blocks satisfying both constraints: {len(amk_blocks)}")
    for b in amk_blocks[:5]:
        print(f"     * Blk {b['block']} {b['street']}: {b['distances']}")
    assert len(amk_blocks) > 0, "Should find blocks in AMK near both MRT and Hawker!"
    print("  [PASS] Multi-layer POI spatial intersection verified.")

    # Test 4: Dual-Transit Commute Travel-Time Minimizer
    print("\n[TEST 4] compute_dual_transit_commute(work_dest_a='Raffles Place', work_dest_b='one-north', max_minutes=35):")
    transit_results = compute_dual_transit_commute(
        candidate_blocks=[],
        work_dest_a="Raffles Place",
        work_dest_b="one-north",
        max_minutes=35
    )
    print(f"  -> Blocks where both commutes <= 35 mins: {len(transit_results)}")
    for r in transit_results[:4]:
        print(f"     * Blk {r['block']} {r['street']} ({r['town']}) -> A: {r['commute_to_dest_a_mins']}m, B: {r['commute_to_dest_b_mins']}m (Total: {r['total_joint_commute_mins']}m)")
    assert len(transit_results) > 0, "Should find central blocks with both commutes <= 35 mins!"
    print("  [PASS] Dual-transit commute travel-time minimizer verified.")

    # Test 5: Interactive Map UI State Sync
    print("\n[TEST 5] sync_interactive_map_ui(candidate_blocks, buffer_radii, center_latlon):")
    map_payload = sync_interactive_map_ui(
        candidate_blocks=aitong_blocks[:10],
        buffer_radii={"primary_school": 1000},
        center_latlon=[1.3606, 103.8330]
    )
    assert map_payload["status"] in ("synced", "synchronized")
    assert "buffer_polygon" in map_payload
    assert len(map_payload["buffer_polygon"]["geometry"]["coordinates"][0]) >= 64
    print(f"  -> Generated MapLibre GeoJSON payload with {len(map_payload['buffer_polygon']['geometry']['coordinates'][0])} polygon vertices.")
    print("  [PASS] Interactive map UI state synchronization verified.")

    print("\n" + "=" * 70)
    print("  ALL 5 SPATIAL FASTMCP TOOL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


def test_spatial_tools_suite():
    run_tests()


def test_dual_transit_commute_depends_on_work_destinations():
    """Different workplace destination pairs must produce destination-specific commute times and block sets."""
    central_west = compute_dual_transit_commute(
        candidate_blocks=["Blk 18A HOLLAND DR", "Blk 201D TAMPINES ST 21"],
        work_dest_a="Raffles Place",
        work_dest_b="one-north",
        max_minutes=60,
    )
    east = compute_dual_transit_commute(
        candidate_blocks=["Blk 18A HOLLAND DR", "Blk 201D TAMPINES ST 21"],
        work_dest_a="Changi Business Park",
        work_dest_b="Tampines",
        max_minutes=60,
    )
    by_blk_cw = {r["block"]: r for r in central_west}
    by_blk_east = {r["block"]: r for r in east}

    assert by_blk_cw["18A"]["total_joint_commute_mins"] < by_blk_cw["201D"]["total_joint_commute_mins"]
    assert by_blk_east["201D"]["total_joint_commute_mins"] < by_blk_east["18A"]["total_joint_commute_mins"]
    assert by_blk_cw["18A"]["commute_to_dest_a_mins"] != by_blk_east["18A"]["commute_to_dest_a_mins"]
    assert by_blk_cw["201D"]["commute_to_dest_b_mins"] != by_blk_east["201D"]["commute_to_dest_b_mins"]


if __name__ == "__main__":
    run_tests()
