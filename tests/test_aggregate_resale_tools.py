"""Grouped resale statistics (mcp_server/aggregate_resale_tools.py): medians and counts agree with an independent query,
bands and filters behave, wrong values come back as structured errors, and the drill-down handles are the next call."""
from __future__ import annotations

import os

os.environ["LLM_PROFILE"] = "none"

import statistics  # noqa: E402

import pytest  # noqa: E402

from agents.research_schemas import DescribeResaleDataOutput, QueryResaleStatisticsOutput  # noqa: E402
from data.market_database import open_market_database  # noqa: E402
from mcp_server import block_features as bf  # noqa: E402
from mcp_server import location_tools  # noqa: E402
from mcp_server.aggregate_resale_tools import describe_resale_data, query_resale_statistics  # noqa: E402
from tests.test_research_tools import AMENITIES, BLOCKS, SCHOOLS, research_env, snapshot  # noqa: E402,F401


@pytest.fixture
def env(research_env, monkeypatch):
    amenities = AMENITIES + [
        {"name": "BISHAN INTERCHANGE MRT STATION", "category": "mrt", "lat": 1.3503, "lng": 103.8503},
        {"name": "TAMPINES MRT STATION", "category": "mrt", "lat": 1.3560, "lng": 103.9450},   # ~220 m from block 5
        {"name": "BISHAN LRT STATION", "category": "mrt", "lat": 1.3511, "lng": 103.8500},
    ]
    payload = {"blocks": BLOCKS, "amenities": amenities, "schools": SCHOOLS}
    monkeypatch.setattr(location_tools, "get_spatial_index", lambda: {"payload": payload})
    bf._CACHE.clear()
    yield
    bf._CACHE.clear()


def oracle(keys, start="2021-01-01", end="2026-09-01", flat=None):
    """Count and median price of the sales at these blocks, from a plain query that shares no code with the tool."""
    with open_market_database() as con:
        sql = "SELECT resale_price FROM resale_transactions WHERE block_key IN (SELECT unnest(?::VARCHAR[])) AND month BETWEEN ? AND ?"
        params = [keys, start, end]
        if flat:
            sql += " AND flat_type = ?"
            params.append(flat)
        prices = [r[0] for r in con.execute(sql, params).fetchall()]
    return len(prices), (statistics.median(prices) if prices else None)


def group(result, label):
    return next(g for g in result["groups"] if g["label"] == label)


def test_distance_bands_match_an_independent_query(env):
    r = query_resale_statistics(group_by=["distance_band"], near="mrt", lookback_years=10, min_sales=1)
    QueryResaleStatisticsOutput.model_validate(r)
    assert r["status"] == "ok" and r["bands"] == [200, 500, 1000]
    feats = bf.get_block_features()
    first_band = [k for k in feats.keys if feats.row(k)["mrt_m"] <= 200]       # block 1 and 2 sit within 200 m of Bishan
    far = [k for k in feats.keys if feats.row(k)["mrt_m"] > 200]
    g0, g1 = group(r, "0-200 m"), group(r, "200-500 m")
    n, med = oracle(first_band)
    assert (g0["count"], g0["median_price"]) == (n, med)
    assert g1["count"] == oracle([k for k in far if feats.row(k)["mrt_m"] <= 500])[0]
    assert r["baseline"]["count"] == oracle(feats.keys)[0] and r["period"]["end_month"] == "2026-09"


def test_town_groups_and_period_and_flat_type_filters(env):
    r = query_resale_statistics(group_by=["town"], flat_types=["4-room"], lookback_years=10, min_sales=1)
    assert r["status"] == "ok" and r["flat_types"] == ["4 ROOM"]
    bishan = group(r, "BISHAN")
    keys = ["1 BISHAN ST 1", "2 BISHAN ST 1"]
    assert (bishan["count"], bishan["median_price"]) == (lambda n_m: n_m)(oracle(keys, flat="4 ROOM"))
    assert bishan["p25"] <= bishan["median_price"] <= bishan["p75"]
    one_year = query_resale_statistics(group_by=["town"], flat_types=["4 ROOM"], min_sales=1)
    assert one_year["period"] == {"start_month": "2025-10", "end_month": "2026-09", "months": 12}
    assert group(one_year, "BISHAN")["count"] == oracle(keys, "2025-10-01", flat="4 ROOM")[0]
    # months after the last complete one (the 9,000,000 sale in 2026-10) never count
    assert max(g["median_price"] for g in one_year["groups"]) < 1_000_000


def test_station_grouping_uses_merged_stations_and_a_default_radius(env):
    r = query_resale_statistics(group_by=["station"], near="mrt", lookback_years=10, min_sales=1)
    labels = {g["label"] for g in r["groups"]}
    assert labels == {"Bishan MRT", "Tampines MRT"}
    assert r["max_distance_m"] == 500 and any("500 m" in a for a in r["assumptions"])
    t = group(r, "Tampines MRT")
    assert t["count"] == oracle(["5 TAMPINES ST 1"])[0] and t["lat"] and t["drill"]["tool"] == "get_area_resale_statistics"
    assert t["drill"]["params"]["area"] == {"address": "Tampines MRT", "radius_meters": 500}
    only = query_resale_statistics(group_by=["station"], near="mrt", station="bishan", lookback_years=10, min_sales=1)
    assert [g["label"] for g in only["groups"]] == ["Bishan MRT"]


def test_lrt_is_excluded_from_mrt_and_counted_in_mrt_lrt(env):
    mrt = query_resale_statistics(group_by=["distance_band"], near="mrt", bands=[50], lookback_years=10, min_sales=1)
    both = query_resale_statistics(group_by=["distance_band"], near="mrt_lrt", bands=[50], lookback_years=10, min_sales=1)
    assert "LRT stations are not counted" in " ".join(mrt["notes"]) and "together" in " ".join(both["notes"])
    # block 2 is ~11 m from the LRT stop and ~50 m from the MRT point: only mrt_lrt puts it inside 50 m... of the LRT
    assert group(both, "0-50 m")["count"] >= group(mrt, "0-50 m")["count"]


def test_trend_sort_and_two_dimensions(env):
    r = query_resale_statistics(group_by=["town", "flat_type"], lookback_years=10, include_trend=True, min_sales=1,
                                sort_by="count", order="asc", top_n=2)
    assert r["truncated"] and len(r["groups"]) == 2 and r["total_groups"] > 2
    assert [g["count"] for g in r["groups"]] == sorted(g["count"] for g in r["groups"])
    assert all(g["trend"] for g in r["groups"])
    years = query_resale_statistics(group_by=["year"], lookback_years=10, min_sales=1)
    assert [g["label"] for g in years["groups"]] == sorted(g["label"] for g in years["groups"])
    assert all(g["change_1y_pct"] is None for g in years["groups"])


def test_small_groups_are_left_out_and_counted(env):
    r = query_resale_statistics(group_by=["town"], flat_types=["4 ROOM"], lookback_years=10, min_sales=10)
    assert [g["label"] for g in r["groups"]] == ["TAMPINES"] and r["excluded_small_groups"] == 1


@pytest.mark.parametrize("kwargs, key", [
    ({"group_by": ["colour"]}, "group_by"), ({"near": "airport"}, "near"), ({"sort_by": "vibes"}, "sort_by"),
    ({"towns": ["ATLANTIS"]}, "town"), ({"flat_types": ["9 ROOM"]}, "flat_type"), ({"metrics": ["mean"]}, "metrics"),
    ({"group_by": ["station"], "near": "hawker"}, "near"), ({"top_n": 500}, "group_by"),
    ({"group_by": ["distance_band"], "bands": [500, 200]}, None)])
def test_invalid_requests_are_structured_errors_with_allowed_values(env, kwargs, key):
    r = query_resale_statistics(**kwargs)
    assert r["status"] == "invalid_request" and r["message"]
    if key:
        assert r["allowed"] and key in r["allowed"]


def test_unknown_station_suggests_close_names(env):
    r = query_resale_statistics(group_by=["distance_band"], station="Bishon MRT")
    assert r["status"] == "invalid_request" and "Bishan MRT" in r["message"]


def test_town_drill_goes_to_stations_and_band_drill_to_towns(env):
    towns = query_resale_statistics(group_by=["town"], near="mrt", max_distance_m=1000, lookback_years=10, min_sales=1)
    drill = group(towns, "BISHAN")["drill"]
    assert drill["params"]["group_by"] == ["station"] and drill["params"]["towns"] == ["BISHAN"]
    assert drill["params"]["near"] == "mrt" and drill["params"]["max_distance_m"] == 1000
    followed = query_resale_statistics(**drill["params"], min_sales=1)
    assert followed["status"] == "ok" and [g["label"] for g in followed["groups"]] == ["Bishan MRT"]
    band = query_resale_statistics(group_by=["distance_band"], near="mrt", lookback_years=10, min_sales=1)
    d = group(band, "0-200 m")["drill"]
    assert d["params"]["group_by"] == ["town"] and d["params"]["max_distance_m"] == 200
    last = query_resale_statistics(group_by=["distance_band"], near="mrt", bands=[100], lookback_years=10, min_sales=1)
    assert group(last, ">100 m")["drill"] is None


def test_describe_resale_data_lists_the_valid_values(env):
    d = describe_resale_data()
    DescribeResaleDataOutput.model_validate(d)
    assert d["towns"] == ["BISHAN", "TAMPINES"] and "4 ROOM" in d["flat_types"]
    assert d["mrt_stations"] == ["Bishan MRT", "Tampines MRT"] and d["lrt_stations"] == ["Bishan LRT"]
    assert set(d["near_options"]) == {"mrt", "mrt_lrt", "lrt", "hawker", "supermarket", "school"}
    assert describe_resale_data(include_stations=False)["mrt_stations"] == []
