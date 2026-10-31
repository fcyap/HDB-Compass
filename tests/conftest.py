"""Mock tool configuration for deterministic contract and graph tests."""

import os

import pytest

# Tests never upload to LangSmith, whatever .env says: set before anything imports langsmith, and re-asserted per test
# (a test that exercises tracing monkeypatches the variables itself and the fixture restores them).
for _name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2"):
    os.environ[_name] = "false"
os.environ.pop("LANGSMITH_AGENT_ENVIRONMENT", None)


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2"):
        monkeypatch.setenv(name, "false")
    try:
        from langsmith import utils as ls_utils
        ls_utils.get_env_var.cache_clear()   # langsmith memoises environment reads
    except Exception:  # noqa: BLE001
        pass
    yield
    try:
        from langsmith import utils as ls_utils
        ls_utils.get_env_var.cache_clear()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture
def mock_domain_tools(monkeypatch):
    from agents.llm import load_config
    from mcp_server.mock_tools import MOCK_TOOLS

    modes = load_config()["tools"]["use_mock"]
    for name in MOCK_TOOLS:
        monkeypatch.setitem(modes, name, "mock")
