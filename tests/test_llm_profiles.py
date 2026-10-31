"""Provider selection and real client serialization without external requests."""

import json
from types import SimpleNamespace

import httpx
import pytest
import requests

from agents import llm as L


@pytest.fixture(autouse=True)
def isolated_model_environment(monkeypatch):
    for name in (
        "LLM_PROFILE",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_AUTO_PROFILES",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "OLLAMA_MODEL",
        "OLLAMA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    L._auto_profile.cache_clear()
    yield
    L._auto_profile.cache_clear()


def test_none_needs_no_provider_and_reports_no_model(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "none")
    monkeypatch.setattr(
        L.requests, "get", lambda *a, **kw: pytest.fail("No network for none")
    )
    assert L.get_llm() is None
    assert L.model_status(check=True)["mode"] == "deterministic"
    assert L.main(["--test-tools"]) == 1


@pytest.mark.parametrize(
    "profile,key_env,model",
    [
        ("groq", "GROQ_API_KEY", "openai/gpt-oss-20b"),
        ("openai", "OPENAI_API_KEY", "gpt-4.1-mini"),
    ],
)
def test_hosted_profile_requires_its_own_key(monkeypatch, profile, key_env, model):
    monkeypatch.setenv("LLM_PROFILE", profile)
    with pytest.raises(L.ModelConfigurationError, match=key_env):
        L.get_llm()
    monkeypatch.setenv(key_env, "test-secret")
    p = L.active_profile()
    assert p["api_key"] == "test-secret" and p["model"] == model
    assert "test-secret" not in json.dumps(L.model_status())
    client = L.get_llm()
    assert client.model_name == model and client.stream_usage is False


def test_explicit_profile_does_not_probe_or_switch(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "ollama")
    monkeypatch.setenv("GROQ_API_KEY", "test-secret")
    monkeypatch.setattr(
        L.requests,
        "get",
        lambda *a, **kw: pytest.fail("Explicit profile must not auto-detect"),
    )
    assert L.active_profile()["name"] == "ollama"
    assert L.get_llm().reasoning_effort == "none"


def test_generic_endpoint_and_model_override(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "compatible")
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9000/v1/")
    monkeypatch.setenv("LLM_MODEL", "custom-tool-model")
    p = L.active_profile()
    assert p["base_url"] == "http://localhost:9000/v1"
    assert p["model"] == "custom-tool-model"


def test_custom_profile_requires_endpoint_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "compatible")
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    assert L.model_status()["status"] == "error"
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9000/v1")
    with pytest.raises(L.ModelConfigurationError, match="LLM_MODEL"):
        L.get_llm()


def test_non_qwen_override_does_not_inherit_qwen_parameters(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "vllm")
    monkeypatch.setenv("LLM_MODEL", "another-model")
    p = L.active_profile()
    assert not p["system_suffix"] and "extra_body" not in p


def test_auto_prefers_local_and_retains_selection(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "auto")
    monkeypatch.setenv("GROQ_API_KEY", "test-secret")
    calls = []

    def discover(p):
        calls.append(p["name"])
        return True, "Available"

    monkeypatch.setattr(L, "_probe_profile", discover)
    assert L.active_profile()["name"] == "ollama"
    assert L.active_profile()["name"] == "ollama"
    assert calls == ["ollama"]


def test_auto_uses_hosted_only_after_local_unavailable(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "auto")
    monkeypatch.setenv("GROQ_API_KEY", "test-secret")
    monkeypatch.setattr(
        L, "_probe_profile", lambda p: (p["name"] == "groq", "Unavailable")
    )
    assert L.active_profile()["name"] == "groq"


def test_auto_can_be_restricted_to_local_and_never_silently_disables_model(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "auto")
    monkeypatch.setenv("LLM_AUTO_PROFILES", "ollama")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setattr(L, "_probe_profile", lambda p: (False, "Unavailable"))
    status = L.model_status()
    assert status["status"] == "error" and "LLM_PROFILE=none" in status["message"]
    assert "test-secret" not in json.dumps(status)


def test_auto_skips_invalid_key(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "auto")
    monkeypatch.setenv("GROQ_API_KEY", "invalid-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")

    def get(url, **kwargs):
        if "openai.com" in url:
            return SimpleNamespace(
                status_code=200, json=lambda: {"data": [{"id": "gpt-4.1-mini"}]}
            )
        return SimpleNamespace(status_code=401)

    monkeypatch.setattr(L.requests, "get", get)
    assert L.active_profile()["name"] == "openai"


def test_auto_rejects_ambiguous_endpoint_override(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "auto")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9000/v1")
    assert "unset LLM_BASE_URL" in L.model_status()["message"]


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
def test_connection_check_returns_safe_provider_errors(monkeypatch, status):
    monkeypatch.setenv("LLM_PROFILE", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-secret")
    monkeypatch.setattr(
        L.requests, "get", lambda *a, **kw: SimpleNamespace(status_code=status)
    )
    report = L.model_status(check=True)
    assert report["status"] == "unavailable"
    assert "test-secret" not in json.dumps(report)


def test_model_probe_checks_exact_model_and_does_not_redirect(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "ollama")
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(
            status_code=200, json=lambda: {"data": [{"id": "some-other-model"}]}
        )

    monkeypatch.setattr(L.requests, "get", get)
    assert L.model_status(check=True)["status"] == "unavailable"
    assert calls[0][0].endswith("/v1/models")
    assert calls[0][1]["allow_redirects"] is False
    assert calls[0][1]["timeout"] == (2, 3)


def test_model_probe_network_errors_cannot_leak_keys(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-secret")

    def fail(*args, **kwargs):
        raise requests.ConnectionError("Authorization: test-secret")

    monkeypatch.setattr(L.requests, "get", fail)
    assert "test-secret" not in json.dumps(L.model_status(check=True))


@pytest.mark.parametrize("profile", ["groq", "openai", "compatible"])
def test_real_client_tool_round_trip_serialization(monkeypatch, profile, capsys):
    monkeypatch.setenv("LLM_PROFILE", profile)
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(key, "test-secret")
    if profile == "compatible":
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9000/v1")
        monkeypatch.setenv("LLM_MODEL", "test-tool-model")
    payloads = []

    def handler(request):
        assert request.headers["authorization"] == "Bearer test-secret"
        payload = json.loads(request.content)
        payloads.append(payload)
        assert payload["tools"][0]["function"]["name"] == "echo_probe"
        assert payload["parallel_tool_calls"] is False
        if len(payloads) == 1:
            reply = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "probe-1",
                        "type": "function",
                        "function": {
                            "name": "echo_probe",
                            "arguments": '{"value":"compass-check"}',
                        },
                    }
                ],
            }
        else:
            assert payload["messages"][-1] == {
                "role": "tool",
                "content": "compass-check",
                "tool_call_id": "probe-1",
            }
            reply = {"role": "assistant", "content": "compass-check"}
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if len(payloads) == 1 else "stop",
                        "message": reply,
                    }
                ],
            },
        )

    original = L.get_llm
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(L, "get_llm", lambda: original(http_client=client))
        assert L.main(["--test-tools"]) == 0
    assert len(payloads) == 2
    assert "test-secret" not in capsys.readouterr().out


def test_backend_status_distinguishes_configuration_from_connectivity(monkeypatch):
    from backend.main import get_model_status

    monkeypatch.setenv("LLM_PROFILE", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-secret")
    calls = []

    def get(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(status_code=401)

    monkeypatch.setattr(L.requests, "get", get)

    configured = get_model_status()
    assert configured["status"] == "configured"
    assert not calls
    unavailable = get_model_status(check=True)
    assert unavailable["status"] == "unavailable"
    assert len(calls) == 1
    for response in (configured, unavailable):
        text = json.dumps(response)
        assert "test-secret" not in text
        assert "base_url" not in text and "api_key" not in text
