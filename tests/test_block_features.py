"""Block features (mcp_server/block_features.py): nearest stations with LRT kept apart, merged interchange
duplicates, other amenity distances, school counts, and the one-time cache."""
from __future__ import annotations

import os

os.environ["LLM_PROFILE"] = "none"

import pytest  # noqa: E402

from data.market_database import open_market_database  # noqa: E402
from mcp_server import block_features as bf  # noqa: E402
from mcp_server import location_tools  # noqa: E402
from tests.test_research_tools import AMENITIES, BLOCKS, SCHOOLS, research_env, snapshot  # noqa: E402,F401


@pytest.fixture
def features(research_env, monkeypatch):
    amenities = AMENITIES + [
        # the same interchange twice, 30 m apart, and an LRT stop that is nearer to block 2 than any MRT
        {"name": "BISHAN INTERCHANGE MRT STATION", "category": "mrt", "lat": 1.3503, "lng": 103.8503},
        {"name": "BISHAN MRT STATION", "category": "mrt", "lat": 1.3520, "lng": 103.8520},
        {"name": "BISHAN LRT STATION", "category": "mrt", "lat": 1.3511, "lng": 103.8500},
    ]
    payload = {"blocks": BLOCKS, "amenities": amenities, "schools": SCHOOLS}
    monkeypatch.setattr(location_tools, "get_spatial_index", lambda: {"payload": payload})
    bf._CACHE.clear()
    yield bf.get_block_features()
    bf._CACHE.clear()


def test_interchange_duplicates_are_one_station(features):
    assert sorted(features.stations) == ["Bishan LRT", "Bishan MRT"]
    assert features.stations["Bishan MRT"]["points"] == 3


def test_nearest_mrt_ignores_lrt_and_uses_the_nearest_duplicate(features):
    r = features.row("1 BISHAN ST 1")
    assert r["mrt_station"] == "Bishan MRT" and 30 < r["mrt_m"] < 70     # the 1.3505 point, ~55 m
    near_lrt = features.row("2 BISHAN ST 1")                              # LRT 11 m away, MRT ~50 m
    assert near_lrt["lrt_station"] == "Bishan LRT" and near_lrt["lrt_m"] < 20
    assert near_lrt["mrt_m"] > near_lrt["lrt_m"] and near_lrt["rail_station"] == "Bishan LRT"
    assert near_lrt["rail_m"] == near_lrt["lrt_m"]


def test_other_amenities_and_schools(features):
    r = features.row("1 BISHAN ST 1")
    assert 1000 < r["hawker_m"] < 1200 and r["supermarket_m"] < 60
    assert r["schools_1km"] == 1                                          # Ai Tong School is ~550 m away
    assert features.row("5 TAMPINES ST 1")["schools_1km"] == 0
    assert features.row("5 TAMPINES ST 1")["mrt_m"] > 5000
    assert features.row("NO SUCH BLOCK") is None


def test_cached_per_snapshot_and_fast(features):
    assert bf.get_block_features() is features
    assert features.build_seconds < 5


def test_features_join_in_sql(features):
    with open_market_database() as con:
        features.load_into(con)
        n, within = con.execute("SELECT count(*), count(*) FILTER (WHERE mrt_m < 200) FROM block_features").fetchone()
    assert n == 3 and within >= 1          # only blocks with sales are in the database
