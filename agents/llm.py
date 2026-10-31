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
from typing import Any, Dict
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
            "Unknown LLM_PROFILE. Choose ollama, vllm, groq, openai, compatible, auto or none."
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
    key_env = p.get("api_key_env", "LLM_API_KEY")
    p["api_key"] = (os.getenv(key_env) or p.get("api_key") or "").strip()
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
            isinstance(row, dict) and row.get("id") == profile["model"] for row in data
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
    }


def get_llm(**overrides: Any):
    """Construct the shared client; never switch providers mid-run."""
    p = active_profile()
    if p["name"] == "none":
        return None
    from langchain_openai import ChatOpenAI

    cfg = load_config()
    kwargs: Dict[str, Any] = dict(
        model=p["model"],
        base_url=p["base_url"],
        api_key=p["api_key"],
        max_tokens=p.get("max_tokens", 2048),
        # LLM_TIMEOUT_S / LLM_MAX_RETRIES: the live demo lowers these so a stalled server fails over in seconds
        timeout=float(os.getenv("LLM_TIMEOUT_S", p.get("timeout_s", 120))),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", 2)),
        stream_usage=False,
    )
    if p.get("supports_temperature", True):
        kwargs["temperature"] = cfg["global"]["temperature"]
    if p.get("supports_seed", True):
        kwargs["seed"] = cfg["global"]["seed"]
    if p.get("extra_body"):
        kwargs["extra_body"] = p["extra_body"]
    if p.get("reasoning_effort") is not None:
        kwargs["reasoning_effort"] = p["reasoning_effort"]
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)


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
        bound = get_llm().bind_tools([echo_probe], parallel_tool_calls=False)
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
