"""
School lookup pack (Member 1, on Member 2's school index): primary schools near a residential address.

The reverse question, blocks near a named school, stays Member 2's core verify_primary_school_distance.
The request itself is read by agents/school_queries.py (school_location_request), which fills nearby_schools.
"""
from __future__ import annotations

from typing import Any, Dict

from agents.addons import ToolAddon, ToolPack


def _plan(agent: str, c: Dict[str, Any], query: str):
    return [("find_primary_schools_near_address", c["nearby_schools"])] if agent == "spatial" and c.get("nearby_schools") else []


PACK = ToolPack(
    name="schools",
    tools=(
        ToolAddon("find_primary_schools_near_address", ("mcp_server.school_lookup_tools", "find_primary_schools_near_address"),
                  "spatial", "agents.schemas:FindSchoolsNearAddressInput", "agents.schemas:FindSchoolsNearAddressOutput",
                  lambda c: bool(c.get("nearby_schools")),
                  "primary schools near a residential address (straight-line distances, not MOE registration distances)",
                  "Primary schools near the address", "schools near 570147"),
    ),
    plan=_plan,
)
