"""LangSmith setup checks (agents/tracing.py): region, destination, verdicts. No network: _http_get is faked."""
import json

import pytest

from agents import tracing as T

ENV_NAMES = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT", "LANGSMITH_PROJECT",
             "LANGCHAIN_PROJECT", "LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT", "LANGSMITH_WORKSPACE_ID")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for n in ENV_NAMES:
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_fakefakefake")
    monkeypatch.setattr(T, "_status", None)
    monkeypatch.setattr(T, "_load_env", lambda: None)
    yield


def fake_http(monkeypatch, by_host, default=(403, b"")):
    calls = []

    def get(url, headers, timeout):
        calls.append(url)
        for host, resp in by_host.items():
            if url.startswith(host):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return default
    monkeypatch.setattr(T, "_http_get", get)
    return calls


def test_ok_us_default(monkeypatch):
    fake_http(monkeypatch, {T.US_ENDPOINT: (200, b"[]")})
    pf = T.preflight()
    assert pf["ok"] and pf["verdict"] == T.OK and pf["destination"]["kind"] == "default"
    assert pf["fix"] == ""


def test_403_with_apac_accepting(monkeypatch):
    calls = fake_http(monkeypatch, {T.US_ENDPOINT: (403, b""), T.EU_ENDPOINT: (403, b""), T.APAC_ENDPOINT: (200, b"[]")})
    pf = T.preflight()
    assert not pf["ok"] and pf["verdict"] == T.FORBIDDEN
    assert pf["accepted_region"]["region"] == "APAC"
    assert "LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com" in pf["fix"]
    assert all("/api/v1/sessions" in c for c in calls)   # read-only probes


def test_403_nowhere(monkeypatch):
    fake_http(monkeypatch, {})
    pf = T.preflight()
    assert not pf["ok"] and pf["accepted_region"] is None and "revoked" in pf["fix"]


def test_401(monkeypatch):
    fake_http(monkeypatch, {T.US_ENDPOINT: (401, b"")}, default=(401, b""))
    pf = T.preflight()
    assert pf["verdict"] == T.INVALID_KEY and not pf["ok"] and "new key" in pf["fix"]


def test_404_project_missing_is_ok(monkeypatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", "p")
    fake_http(monkeypatch, {T.US_ENDPOINT: (404, b"")})
    assert T.preflight()["ok"]


def test_project_missing_empty_list(monkeypatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", "p")
    fake_http(monkeypatch, {T.US_ENDPOINT: (200, b"[]")})
    pf = T.preflight()
    assert pf["verdict"] == T.PROJECT_MISSING and pf["ok"]


def test_network_error(monkeypatch):
    fake_http(monkeypatch, {T.US_ENDPOINT: OSError("boom")})
    pf = T.preflight()
    assert pf["verdict"] == T.NETWORK and not pf["ok"]


def test_no_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY")
    assert T.preflight()["verdict"] == T.NO_KEY


def test_endpoint_respected(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ENDPOINT", T.APAC_ENDPOINT + "/")
    calls = fake_http(monkeypatch, {T.APAC_ENDPOINT: (200, json.dumps([{"id": 1}]).encode())})
    pf = T.preflight()
    assert pf["ok"] and pf["region"] == "APAC" and pf["endpoint_set"]
    assert calls and all(c.startswith(T.APAC_ENDPOINT) for c in calls)


def test_half_address(monkeypatch):
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "local")
    fake_http(monkeypatch, {T.US_ENDPOINT: (200, b"[]")})
    pf = T.preflight()
    assert not pf["ok"] and "LANGSMITH_AGENT_ID" in pf["fix"] and pf["destination"]["kind"] == "invalid"


def test_address_and_project(monkeypatch):
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "hdb-compass")
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "local")
    monkeypatch.setenv("LANGSMITH_PROJECT", "hdb-compass")
    fake_http(monkeypatch, {T.US_ENDPOINT: (200, b"[]")})
    pf = T.preflight()
    assert not pf["ok"] and "Remove the LANGSMITH_PROJECT line" in pf["fix"]


def test_valid_address(monkeypatch):
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "hdb-compass")
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "local")
    fake_http(monkeypatch, {T.US_ENDPOINT: (200, b"[]")})
    pf = T.preflight()
    assert pf["ok"] and pf["destination"]["kind"] == "agent"


def test_configure_disables_once_with_one_line(monkeypatch, caplog):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    fake_http(monkeypatch, {T.APAC_ENDPOINT: (200, b"[]")})
    with caplog.at_level("INFO", logger="hdb_compass.tracing"):
        st = T.configure_tracing()
        T.configure_tracing()
    assert not st["enabled"]
    assert len([r for r in caplog.records if "DISABLED" in r.message]) == 1
    import os
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_configure_ok_leaves_tracing_on(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "hdb-compass")
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "local")
    fake_http(monkeypatch, {T.US_ENDPOINT: (200, b"[]")})
    st = T.configure_tracing()
    import os
    assert st["enabled"] and os.environ["LANGSMITH_TRACING"] == "true" and "LANGSMITH_PROJECT" not in os.environ


def test_format_check_has_fix(monkeypatch):
    fake_http(monkeypatch, {T.APAC_ENDPOINT: (200, b"[]")})
    out = T.format_check(T.preflight())
    assert "exact fix" in out and "FAILED" in out and "APAC" in out


def test_run_config_contents(monkeypatch):
    monkeypatch.setattr(T, "code_commit", lambda: "abc123")
    cfg = T.run_config("C", session_id="s1", question_id="q7", tags=("x",), metadata={"extra": 1})
    assert cfg["run_name"] == "hdb-compass-C:q7"
    assert cfg["tags"] == ["variant:C", "eval", "x"]
    m = cfg["metadata"]
    assert m["variant"] == "C" and m["session_id"] == "s1" and m["thread_id"] == "s1"
    assert m["question_id"] == "q7" and m["commit"] == "abc123" and m["extra"] == 1
    app = T.run_config("A")
    assert app["run_name"] == "hdb-compass-A" and "app" in app["tags"] and "thread_id" not in app["metadata"]


def test_tool_span_only_when_enabled(monkeypatch):
    seen = []

    class Fake:
        def __init__(self, **kw):
            seen.append(kw)

        def __call__(self, f):
            return f
    monkeypatch.setattr(T, "traceable", Fake)
    monkeypatch.setattr(T, "tracing_enabled", lambda: False)
    assert T.traced_tool_call("t", lambda x: x + 1, {"x": 1}) == 2
    assert seen == []
    monkeypatch.setattr(T, "tracing_enabled", lambda: True)
    assert T.traced_tool_call("t", lambda x: x + 1, {"x": 1}) == 2
    assert seen and seen[0]["run_type"] == "tool" and seen[0]["name"] == "t"
