"""Gemini profile resolution and client-side rate limiting (no network, no real sleeping)."""
from datetime import date, timedelta

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from agents import llm, rate_limit as rl


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for v in ("LLM_PROFILE", "LLM_BASE_URL", "LLM_MODEL", "LLM_RPM", "LLM_TPM", "LLM_RPD",
              "LLM_RATE_LIMIT_MAX_WAIT_S", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(v, raising=False)
    rl.reset_limiters()
    yield
    rl.reset_limiters()


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def make(limits, max_wait=60, clock=None, today=None):
    c = clock or Clock()
    kw = {"today": today} if today else {}
    return rl.RateLimiter("t", limits, max_wait, clock=c.now, sleep=c.sleep, **kw), c


# ---- Gemini profile -------------------------------------------------------
def test_gemini_profile_resolution(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k-test")
    p = llm.active_profile()
    assert p["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert p["model"] == "gemini-3.5-flash-lite"
    assert p["api_key"] == "k-test"
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    assert llm.active_profile()["model"] == "gemini-2.5-flash-lite"


def test_gemini_google_api_key_fallback_and_missing_key(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "gemini")
    with pytest.raises(llm.ModelConfigurationError, match="GEMINI_API_KEY"):
        llm.active_profile()
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    assert llm.active_profile()["api_key"] == "g-test"


def test_gemini_uses_googles_own_sdk_with_thinking_level(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k-test")
    m = llm.get_llm()
    assert type(m).__mro__[1].__name__ == "ChatGoogleGenerativeAI"
    assert m.model == "gemini-3.5-flash-lite" and m.thinking_level == "low"
    assert m.max_output_tokens == 2048 and m.seed == 42
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")      # 2.5 models take a token budget instead of a level
    assert llm.get_llm().thinking_budget == 1024


def test_groq_uses_groqs_own_sdk(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "k-test")
    m = llm.get_llm()
    assert type(m).__name__ == "ChatGroq" and m.model_name == "openai/gpt-oss-20b"
    assert m.model_kwargs == {"seed": 42}


def test_provider_must_be_known():
    with pytest.raises(llm.ModelConfigurationError, match="Unknown provider"):
        llm.provider_of({"name": "x", "provider": "nope"})


def test_google_reply_blocks_become_plain_text():
    blocks = [{"type": "text", "text": "Hello ", "extras": {"signature": "abc"}},
              {"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "world"}]
    assert llm._plain_text(blocks) == "Hello world"
    assert llm._plain_text("already text") == "already text"
    assert llm._plain_text([]) == ""


def test_gemini_not_in_experiment_config():
    assert "gemini" not in llm.load_config()["llm"]["profiles"]
    assert "gemini" not in llm.load_config()["llm"].get("auto_profiles", [])


# ---- limit resolution -----------------------------------------------------
def test_env_overrides_yaml_limits():
    yaml_l = {"requests_per_minute": 10, "tokens_per_minute": 1000, "requests_per_day": 5}
    assert rl.resolve_limits(yaml_l, env={}) == rl.Limits(10, 1000, 5)
    assert rl.resolve_limits(yaml_l, env={"LLM_RPM": "3", "LLM_TPM": "0", "LLM_RPD": ""}) == rl.Limits(3, 0, 5)
    assert rl.resolve_limits(None, env={}).unlimited


def test_profile_defaults_and_unlimited_local(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    s = llm.model_status()["rate_limits"]
    assert (s["requests_per_minute"], s["requests_per_day"]) == (15, 250)
    monkeypatch.setenv("LLM_RPM", "0")
    assert llm.model_status()["rate_limits"]["requests_per_minute"] is None
    monkeypatch.setenv("LLM_PROFILE", "ollama")
    assert llm.model_status()["rate_limits"] == {"enabled": False}


# ---- limiter behaviour ----------------------------------------------------
def test_rpm_blocks_until_window_frees():
    lim, c = make(rl.Limits(requests_per_minute=2))
    lim.acquire(); lim.acquire()
    assert c.slept == []
    lim.acquire()
    assert len(c.slept) == 1 and c.slept[0] == pytest.approx(60.0)
    assert lim.status()["requests_last_minute"] == 1


def test_tpm_accounting_and_usage_correction():
    lim, c = make(rl.Limits(tokens_per_minute=1000))
    slot = lim.acquire(900)
    lim.acquire(50)
    assert lim.status()["tokens_last_minute"] == 950
    lim.record_usage(slot, 100)  # actual usage far below the estimate
    assert lim.status()["tokens_last_minute"] == 150
    lim.acquire(800)  # fits now, no sleep
    assert c.slept == []
    lim.acquire(300)  # 150+800+300 > 1000 -> must wait
    assert c.slept


def test_oversized_single_request_allowed_on_empty_window():
    lim, c = make(rl.Limits(tokens_per_minute=100))
    lim.acquire(5000)
    assert c.slept == []


def test_rpd_exhaustion_raises_immediately_and_resets_next_day():
    day = [date(2026, 1, 1)]
    lim, c = make(rl.Limits(requests_per_day=2), today=lambda: day[0])
    lim.acquire(); lim.acquire()
    with pytest.raises(rl.RateLimitError, match="Daily request limit"):
        lim.acquire()
    assert c.slept == []
    day[0] += timedelta(days=1)
    lim.acquire()


def test_max_wait_error_names_limit():
    lim, c = make(rl.Limits(requests_per_minute=1), max_wait=5)
    lim.acquire()
    with pytest.raises(rl.RateLimitError, match="requests per minute"):
        lim.acquire()
    assert c.slept == []


# ---- wiring through get_llm ----------------------------------------------
def test_get_llm_attaches_callback_and_merges_callbacks(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    m = llm.get_llm()
    assert any(isinstance(cb, rl.RateLimitCallback) for cb in m.callbacks)
    from langchain_core.callbacks import BaseCallbackHandler

    class Mine(BaseCallbackHandler):
        pass

    m2 = llm.get_llm(callbacks=[Mine()])
    kinds = {type(cb) for cb in m2.callbacks}
    assert Mine in kinds and rl.RateLimitCallback in kinds
    # the same limiter is shared process-wide
    a = next(cb for cb in m.callbacks if isinstance(cb, rl.RateLimitCallback))
    b = next(cb for cb in m2.callbacks if isinstance(cb, rl.RateLimitCallback))
    assert a.limiter is b.limiter


def test_unlimited_profile_has_no_callback(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "ollama")
    m = llm.get_llm()
    assert not any(isinstance(cb, rl.RateLimitCallback) for cb in (m.callbacks or []))


def _fake_with_callback(limiter, replies):
    model = GenericFakeChatModel(messages=iter(replies))
    model.callbacks = [rl.RateLimitCallback(limiter, max_output_tokens=10)]
    return model


def test_callback_acquires_on_invoke_bind_tools_and_structured_output():
    lim, c = make(rl.Limits(requests_per_day=2))
    model = _fake_with_callback(lim, [AIMessage(content="a"), AIMessage(content="b"), AIMessage(content="c")])
    model.invoke([HumanMessage(content="hi")])
    assert lim.status()["requests_today"] == 1
    # Survives bind (what bind_tools / with_structured_output build on).
    model.bind(stop=None).invoke([HumanMessage(content="hi")])
    assert lim.status()["requests_today"] == 2
    with pytest.raises(rl.RateLimitError, match="Daily request limit"):
        model.invoke([HumanMessage(content="hi")])


def test_bind_tools_and_structured_output_keep_callbacks(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls = []

    class Out(BaseModel):
        x: int

    def tool(a: int) -> int:
        """t"""
        return a

    import httpx

    def fail(request):  # no network: the HTTP layer errors after callbacks have run
        raise httpx.ConnectError("blocked")

    m = llm.get_llm(max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(fail)))
    cb = next(c for c in m.callbacks if isinstance(c, rl.RateLimitCallback))
    monkeypatch.setattr(cb.limiter, "acquire", lambda tokens=0: calls.append(tokens) or (0.0, tokens))
    for runnable in (m, m.bind_tools([tool]), m.with_structured_output(Out)):
        with pytest.raises(Exception):
            runnable.invoke([HumanMessage(content="hello world")])
    assert len(calls) == 3
    assert all(t >= 2048 for t in calls)  # estimate includes max_tokens


# ---- limits follow the model ---------------------------------------------
def test_limits_follow_the_selected_model(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    known = llm.active_profile()
    assert known["rate_limits"]["requests_per_minute"] == 15 and known["limiter_key"] == "gemini:gemini-3.5-flash-lite"
    monkeypatch.setenv("GEMINI_MODEL", "some-other-model")
    other = llm.active_profile()
    assert other["rate_limits"]["requests_per_minute"] == 10     # profile fallback
    assert other["limiter_key"] == "gemini:some-other-model"      # its own counters
    monkeypatch.setenv("LLM_RPM", "7")                            # .env still overrides any model
    assert llm.model_status()["rate_limits"]["requests_per_minute"] == 7


def test_groq_default_model_limits(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    p = llm.active_profile()
    assert p["model"] == "openai/gpt-oss-20b"
    assert (p["rate_limits"]["requests_per_minute"], p["rate_limits"]["tokens_per_minute"]) == (30, 8000)
