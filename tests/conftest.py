"""Mock tool configuration for deterministic contract and graph tests."""

import pytest


@pytest.fixture
def mock_domain_tools(monkeypatch):
    from agents.llm import load_config
    from mcp_server.mock_tools import MOCK_TOOLS

    modes = load_config()["tools"]["use_mock"]
    for name in MOCK_TOOLS:
        monkeypatch.setitem(modes, name, "mock")
