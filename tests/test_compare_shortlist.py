import pytest

from agents import research_intents as ri
from agents.research_intents import extract_addresses, comparison_takeaway, comparison_chips, research_lines

A, B, C, D = ("Blk 231 TAMPINES ST 21", "Blk 440 TAMPINES ST 43", "Blk 154 SIMEI ST 1", "Blk 123 BEDOK NORTH ST 2")


def _text(names):
    if len(names) == 1:
        return names[0]
    return f"Compare {', '.join(names[:-1])} and {names[-1]} for 4 ROOM"


@pytest.mark.parametrize("names", [[A, B], [A, B, C], [A, B, C, D]])
def test_extracts_all_capscased_blocks(names):
    got = extract_addresses(_text(names))
    assert [g.lower() for g in got] == [n.lower() for n in names]


def _opt(name, n, price=None, ppsm=None, lease=None, mrt=None):
    return {"address": name, "location": {"name": name}, "status": "ok" if n else "no_transactions",
            "transaction_count": n, "median_price": price, "median_price_per_sqm": ppsm,
            "median_remaining_lease_years": lease,
            "nearest": {"mrt": {"name": "X Mrt Station", "distance_meters": mrt}}}


def _cmp():
    return {"flat_type": "4 ROOM", "metadata": {"window_start_month": "2024-10", "window_end_month": "2026-09", "notes": []},
            "options": [_opt(A, 1, 553888, 5956, 57, 531), _opt(B, 3, 510000, 6071, 59, 616), _opt(C, 0, mrt=100)]}


def test_takeaway_computed_from_output():
    t = comparison_takeaway(_cmp())
    assert "cheapest median: Blk 440 TAMPINES ST 43 ($510,000)" in t
    assert "most lease left: Blk 440 TAMPINES ST 43 (59 yrs)" in t
    assert "best price per sqm: Blk 231 TAMPINES ST 21 ($5,956)" in t
    assert "closest MRT: Blk 154 SIMEI ST 1 (100 m" in t   # zero-sales blocks still compete on distance


def test_zero_sales_block_marked_not_n_a():
    lines = research_lines({"sql_market_results": {"compare_resale_options": _cmp()}, "user_constraints": {}})
    text = "\n".join(lines)
    assert "| Blk 154 SIMEI ST 1 | 0 sales | no 4 ROOM sales in this window" in text
    assert "had no 4 ROOM sales" in text
    assert "n/a" not in text.split("Blk 154 SIMEI ST 1")[1].splitlines()[0]


def test_chips_price_check_cheapest():
    chips = comparison_chips({"sql_market_results": {"compare_resale_options": _cmp()}})
    assert chips[0]["text"] == f"Is the price fair for a 4 ROOM at {B}?"
    assert chips[1]["text"] == f"Can we afford a 4 ROOM at {B}?"


def test_fast_path_routes_to_compare_without_model():
    import warnings
    warnings.filterwarnings("ignore")
    from agents.arch_3_specialist_5agent import build_graph
    import agents.supervisor as sv
    g = build_graph()
    q = f"Compare {A}, {B} and {C} for 4 ROOM"
    out = g.invoke(sv.initial_state(query=q, variant="C", previous=None, fast_path=True))
    tools = [l.get("tool") for l in out.get("tool_call_log") or []]
    assert tools == ["compare_resale_options"]
    args = out["tool_call_log"][0].get("args") or out["tool_call_log"][0].get("arguments")
    assert len(args["addresses"]) == 3 and args["flat_type"] == "4 ROOM"
    assert "**Takeaway:**" in out["draft_response"]
