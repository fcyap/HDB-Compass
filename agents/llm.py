"""Shared model factory for local and OpenAI-compatible hosted providers.

Explicit profiles preserve one model across experiment variants. Opt-in auto
selection checks model availability once per process, before user inference.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import requests
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "experiment_config.yaml"
load_dotenv(PROJECT_ROOT / ".env")
_AUTO_SELECTION_LOCK = Lock()


class ModelConfigurationError(ValueError):
    """Configuration failure whose message contains no credentials."""


HOSTED_PROFILES_PATH = PROJECT_ROOT / "config" / "hosted_models.yaml"


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _hosted_profiles() -> Dict[str, Any]:
    """Hosted profiles for the app (config/hosted_models.yaml); the experiment config names local models only."""
    if not HOSTED_PROFILES_PATH.exists():
        return {}
    with open(HOSTED_PROFILES_PATH, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("profiles") or {}


def all_profiles() -> Dict[str, Any]:
    return {**_hosted_profiles(), **load_config()["llm"]["profiles"]}


def active_profile_name() -> str:
    return (os.getenv("LLM_PROFILE") or load_config()["llm"]["active"]).strip()


def _configured_profile(name: str) -> Dict[str, Any]:
    profiles = all_profiles()
    if name not in profiles:
        raise ModelConfigurationError(
            "Unknown LLM_PROFILE. Choose ollama, vllm, groq, openai, gemini, compatible, auto or none."
        )
    p = dict(profiles[name])
    p["base_url"] = (
        (os.getenv("LLM_BASE_URL") or p.get("base_url") or "").strip().rstrip("/")
    )
    p["model"] = (
        os.getenv("LLM_MODEL")
        or os.getenv(p.get("model_env", ""))
        or p.get("model")
        or ""
    ).strip()
    # free-tier limits belong to a model, not a provider: a model listed under model_rate_limits uses its own numbers,
    # any other model uses the profile's rate_limits (LLM_RPM / LLM_TPM / LLM_RPD override either, see rate_limit.py)
    per_model = p.get("model_rate_limits") or {}
    p["rate_limits"] = per_model.get(p["model"]) or p.get("rate_limits")
    p["limiter_key"] = f"{name}:{p['model']}"
    key_env = p.get("api_key_env", "LLM_API_KEY")
    p["api_key"] = (os.getenv(key_env) or p.get("api_key") or "").strip()
    for fallback in p.get("api_key_env_fallbacks") or []:
        if not p["api_key"]:
            p["api_key"] = (os.getenv(fallback) or "").strip()
    if p.get("requires_api_key") and not p["api_key"]:
        raise ModelConfigurationError(f"Set {key_env} in .env for the {name} profile.")
    if not p["api_key"]:
        p["api_key"] = "not-needed"
    try:
        url = urlsplit(p["base_url"])
    except ValueError as exc:
        raise ModelConfigurationError(
            "Set LLM_BASE_URL to a valid HTTP(S) API base URL."
        ) from exc
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ModelConfigurationError(
            "Set LLM_BASE_URL to an HTTP(S) API base URL without credentials, query or fragment."
        )
    if not p["model"]:
        raise ModelConfigurationError(
            "Set LLM_MODEL to a model supporting tool calling."
        )
    p["name"] = name
    p.setdefault("system_suffix", "")
    if "qwen3" not in p["model"].lower():
        p["system_suffix"] = ""
        if p.get("extra_body") == {"chat_template_kwargs": {"enable_thinking": False}}:
            p.pop("extra_body")
    return p


def _probe_profile(profile: Dict[str, Any]) -> tuple[bool, str]:
    """List models without inference or user conversation data."""
    try:
        response = requests.get(
            profile["base_url"] + "/models",
            headers={"Authorization": f"Bearer {profile['api_key']}"},
            timeout=(2, 3),
            allow_redirects=False,
        )
        if response.status_code in (401, 403):
            return False, "Provider rejected credentials or model-list access."
        if not 200 <= response.status_code < 300:
            return False, f"Model discovery returned HTTP {response.status_code}."
        data = response.json().get("data", [])
        if not isinstance(data, list):
            return False, "Provider returned an invalid model list."
        if not any(
            isinstance(row, dict)
            and (row.get("id") == profile["model"] or str(row.get("id", "")).removeprefix("models/") == profile["model"])
            for row in data
        ):
            return (
                False,
                "Configured model is not advertised by the provider; check the model name or pull it in Ollama.",
            )
        return (
            True,
            "Model is advertised by the provider. Run --test-tools to check inference and tool calling.",
        )
    except (requests.RequestException, ValueError, AttributeError, TypeError):
        return (
            False,
            "Cannot discover models. Check the endpoint, network and provider service.",
        )


@lru_cache(maxsize=1)
def _auto_profile() -> Dict[str, Any]:
    if os.getenv("LLM_BASE_URL") or os.getenv("LLM_MODEL"):
        raise ModelConfigurationError(
            "For auto, unset LLM_BASE_URL/LLM_MODEL and use provider-specific model variables; select an explicit profile for endpoint overrides."
        )
    names = os.getenv("LLM_AUTO_PROFILES")
    names = (
        names.split(",")
        if names
        else load_config()["llm"].get("auto_profiles", ["ollama", "groq", "openai"])
    )
    failures = []
    for name in dict.fromkeys(n.strip() for n in names if n.strip()):
        try:
            p = _configured_profile(name)
            ok, detail = _probe_profile(p)
        except ModelConfigurationError as exc:
            failures.append(f"{name}: {exc}")
            continue
        if ok:
            return p
        failures.append(f"{name}: {detail}")
    raise ModelConfigurationError(
        "No auto profile is available. "
        + " ".join(failures)
        + " Start a configured model, supply a provider key, or explicitly select LLM_PROFILE=none."
    )


def active_profile() -> Dict[str, Any]:
    name = active_profile_name()
    if name == "none":
        return {"name": "none", "model": None, "base_url": None, "api_key": None}
    if name == "auto":
        with _AUTO_SELECTION_LOCK:
            return dict(_auto_profile())
    return _configured_profile(name)


def model_status(check: bool = False) -> Dict[str, Any]:
    """Public metadata; never expose keys, URLs or provider error bodies."""
    requested = active_profile_name()
    try:
        p = active_profile()
    except ModelConfigurationError as exc:
        return {
            "requested_profile": requested,
            "profile": None,
            "model": None,
            "mode": "unavailable",
            "status": "error",
            "message": str(exc),
        }
    if p["name"] == "none":
        return {
            "requested_profile": requested,
            "profile": "none",
            "model": None,
            "mode": "deterministic",
            "status": "disabled",
            "message": "Rule-based mode; no language model is running.",
        }
    ok, detail = (
        _probe_profile(p)
        if check
        else (
            True,
            "Model configured; inference and tool calling have not been checked.",
        )
    )
    return {
        "requested_profile": requested,
        "profile": p["name"],
        "model": p["model"],
        "mode": "model",
        "status": ("available" if ok else "unavailable") if check else "configured",
        "message": detail,
        "rate_limits": _rate_limit_status(p),
    }


def _rate_limit_status(p: Dict[str, Any]) -> Dict[str, Any]:
    """Configured limits plus live in-memory counters (no secrets)."""
    from agents import rate_limit as rl

    limits = rl.resolve_limits(p.get("rate_limits"))
    if limits.unlimited:
        return {"enabled": False}
    live = rl.limiter_status(p.get("limiter_key", p["name"])) or {}
    return {
        "enabled": True,
        "requests_per_minute": limits.requests_per_minute or None,
        "tokens_per_minute": limits.tokens_per_minute or None,
        "requests_per_day": limits.requests_per_day or None,
        "max_wait_s": rl.max_wait_s(),
        **{k: v for k, v in live.items() if k.startswith(("requests_last", "tokens_last", "requests_today", "requests_left"))},
    }


# Which SDK talks to which provider. Hosted providers use their own SDK (through its LangChain wrapper), so a model
# a provider lists works the way the provider documents it; a compatibility layer (e.g. Gemini's OpenAI endpoint)
# drops features such as Gemini 3's thought signatures, which broke multi-step tool calls. "openai" is also the
# protocol vLLM, Ollama and the "compatible" profile serve, so those stay on ChatOpenAI.
PROVIDERS = ("openai", "groq", "google")


def provider_of(p: Dict[str, Any]) -> str:
    provider = (p.get("provider") or "openai").strip().lower()
    if provider not in PROVIDERS:
        raise ModelConfigurationError(f"Unknown provider '{provider}' for the {p.get('name')} profile. Choose {', '.join(PROVIDERS)}.")
    return provider


def _google_thinking(effort: Any, model: str) -> Dict[str, Any]:
    """The profile's reasoning_effort as Gemini's own thinking setting: a level on Gemini 3.x, a token budget on 2.5."""
    if effort is None:
        return {}
    effort = str(effort).lower()
    if model.startswith("gemini-2.5"):
        return {"thinking_budget": {"none": 0, "minimal": 128, "low": 1024, "medium": 8192, "high": 24576}.get(effort, -1)}
    return {"thinking_level": "minimal" if effort == "none" else effort}


def _plain_text(content: Any) -> Any:
    """Gemini 3 returns a reply as signed content blocks; the rest of the app reads message.content as text."""
    if not isinstance(content, list):
        return content
    return "".join(b if isinstance(b, str) else (b.get("text") or "") for b in content
                   if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text"))


_GOOGLE_CLASS: Any = None


def _google_chat_class():
    """ChatGoogleGenerativeAI whose replies carry plain-string content. Tool calls and the function-call thought
    signatures Gemini 3 needs on the next step stay as they are (they live on tool_calls / additional_kwargs)."""
    global _GOOGLE_CLASS
    if _GOOGLE_CLASS is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        class PlainTextChatGoogle(ChatGoogleGenerativeAI):
            def _generate(self, *args, **kwargs):
                result = super()._generate(*args, **kwargs)
                for g in result.generations:
                    g.message.content = _plain_text(g.message.content)
                    g.text = g.message.content if isinstance(g.message.content, str) else g.text
                return result

            async def _agenerate(self, *args, **kwargs):
                result = await super()._agenerate(*args, **kwargs)
                for g in result.generations:
                    g.message.content = _plain_text(g.message.content)
                    g.text = g.message.content if isinstance(g.message.content, str) else g.text
                return result

        _GOOGLE_CLASS = PlainTextChatGoogle
    return _GOOGLE_CLASS


def _build_client(p: Dict[str, Any], common: Dict[str, Any], cfg: Dict[str, Any], overrides: Dict[str, Any]):
    """The provider's own chat client. `common`: model, max_tokens, timeout, max_retries, callbacks."""
    provider = provider_of(p)
    temperature = cfg["global"]["temperature"] if p.get("supports_temperature", True) else None
    seed = cfg["global"]["seed"] if p.get("supports_seed", True) else None
    explicit_base = (os.getenv("LLM_BASE_URL") or "").strip().rstrip("/")
    if provider == "groq":
        from langchain_groq import ChatGroq
        kwargs: Dict[str, Any] = dict(model=common["model"], api_key=p["api_key"], max_tokens=common["max_tokens"],
                                      timeout=common["timeout"], max_retries=common["max_retries"],
                                      callbacks=common.get("callbacks"))
        if explicit_base:
            kwargs["base_url"] = explicit_base          # otherwise the SDK's own endpoint
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["model_kwargs"] = {"seed": seed}
        if p.get("reasoning_effort") is not None:
            kwargs["reasoning_effort"] = p["reasoning_effort"]
        kwargs.update(overrides)
        return ChatGroq(**kwargs)
    if provider == "google":
        kwargs = dict(model=common["model"], api_key=p["api_key"], max_tokens=common["max_tokens"],
                      timeout=common["timeout"], max_retries=common["max_retries"], callbacks=common.get("callbacks"))
        if explicit_base:
            kwargs["base_url"] = explicit_base
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        kwargs.update(_google_thinking(p.get("reasoning_effort"), common["model"]))
        kwargs.update(overrides)
        return _google_chat_class()(**kwargs)
    from langchain_openai import ChatOpenAI
    kwargs = dict(model=common["model"], base_url=p["base_url"], api_key=p["api_key"], max_tokens=common["max_tokens"],
                  timeout=common["timeout"], max_retries=common["max_retries"], stream_usage=False,
                  callbacks=common.get("callbacks"))
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    if p.get("extra_body"):
        kwargs["extra_body"] = p["extra_body"]
    if p.get("reasoning_effort") is not None:
        kwargs["reasoning_effort"] = p["reasoning_effort"]
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)


def get_llm(profile: Optional[str] = None, model: Optional[str] = None, **overrides: Any):
    """Construct the shared client; never switch providers mid-run.
    profile: build this named profile instead of the active one (e.g. an evaluation judge from another model family,
    so the judge does not grade its own model's answers). model: use this model of that profile instead of its
    configured one. Each profile:model pair keeps its own rate limiter and its own free-tier limits."""
    p = _configured_profile(profile) if profile else active_profile()
    if p["name"] == "none":
        return None
    if model:
        p = dict(p, model=model, limiter_key=f"{p['name']}:{model}",
                 rate_limits=(p.get("model_rate_limits") or {}).get(model) or p.get("rate_limits"))
    from agents import rate_limit as rl

    cfg = load_config()
    common: Dict[str, Any] = dict(
        model=p["model"],
        max_tokens=int(overrides.get("max_tokens") or p.get("max_tokens", 2048)),
        # LLM_TIMEOUT_S / LLM_MAX_RETRIES: the live demo lowers these so a stalled server fails over in seconds
        timeout=float(os.getenv("LLM_TIMEOUT_S", p.get("timeout_s", 120))),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", 2)),
    )
    overrides = dict(overrides)
    cbs = list(overrides.pop("callbacks", None) or [])
    limiter = rl.get_limiter(
        p.get("limiter_key", p["name"]), rl.resolve_limits(p.get("rate_limits")), rl.max_wait_s()
    )
    if limiter is not None:
        cbs.append(rl.RateLimitCallback(limiter, common["max_tokens"]))
    common["callbacks"] = cbs or None
    return _build_client(p, common, cfg, overrides)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the configured model without exposing credentials."
    )
    parser.add_argument(
        "--check", action="store_true", help="Check model discovery without inference"
    )
    parser.add_argument(
        "--test-tools",
        action="store_true",
        help="Make two small inference requests to verify tool calling",
    )
    args = parser.parse_args(argv)
    status = model_status(check=args.check)
    print(json.dumps(status, indent=2))
    if status["status"] in ("error", "unavailable"):
        return 1
    if not args.test_tools:
        return 0
    if status["mode"] == "deterministic":
        print("Select a model profile before using --test-tools.")
        return 1
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    def echo_probe(value: str) -> str:
        """Echo the supplied value to test local tool execution."""
        return value

    try:
        prof = active_profile()
        bind_kwargs = {} if prof.get("provider") == "google" else {"parallel_tool_calls": False}
        bound = get_llm().bind_tools([echo_probe], **bind_kwargs)
        messages = [
            SystemMessage(
                content="Use the provided tool exactly as requested. "
                + active_profile().get("system_suffix", "")
            ),
            HumanMessage(
                content="Call echo_probe once with value compass-check. After receiving its result, reply with compass-check and do not call again."
            ),
        ]
        reply = bound.invoke(messages)
        calls = reply.tool_calls
        if (
            len(calls) != 1
            or calls[0]["name"] != "echo_probe"
            or calls[0]["args"] != {"value": "compass-check"}
        ):
            print(
                "Tool test failed: model did not request the expected tool and argument."
            )
            return 1
        messages.extend(
            [
                reply,
                ToolMessage(
                    content=echo_probe(**calls[0]["args"]), tool_call_id=calls[0]["id"]
                ),
            ]
        )
        final = bound.invoke(messages)
        if final.tool_calls or "compass-check" not in str(final.content):
            print(
                "Tool test failed: model did not finish using the returned tool result."
            )
            return 1
    except Exception:
        print(
            "Model/tool request failed. Check credentials, model tool support, quota and endpoint configuration."
        )
        return 1
    print(
        "Tool-call round trip passed. This does not establish housing-answer accuracy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
