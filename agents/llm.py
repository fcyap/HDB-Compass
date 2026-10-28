"""
LLM factory (agents/llm.py). Owned by Member 4.

One place that turns config/experiment_config.yaml -> a LangChain chat model.
All variants (A, B, C), the auditor and the eval runner call get_llm() so the
benchmark is guaranteed to use one model.

Profiles (config: llm.profiles.<name>), selected by LLM_PROFILE env var or
config llm.active:
  ollama -> local dev on the laptop        (OpenAI-compatible, http://localhost:11434/v1)
  vllm   -> SCIS GPU cluster via SSH tunnel (OpenAI-compatible, http://localhost:8001/v1)
  none   -> deterministic mode, no model (rule-based fallbacks everywhere)

Both servers speak the OpenAI-compatible protocol, so one client class covers them.
No paid provider is configured by decision of the team (22 Oct 2026).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "experiment_config.yaml"
load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def active_profile_name() -> str:
    cfg = load_config()
    return os.getenv("LLM_PROFILE") or cfg["llm"]["active"]


def active_profile() -> Dict[str, Any]:
    cfg = load_config()
    name = active_profile_name()
    if name == "none":
        return {"name": "none", "model": None, "base_url": None, "api_key": None}
    profile = dict(cfg["llm"]["profiles"][name])
    # env overrides so the cluster endpoint/model can change without editing yaml
    profile["base_url"] = os.getenv("LLM_BASE_URL", profile.get("base_url"))
    profile["model"] = os.getenv("LLM_MODEL", profile.get("model"))
    profile["api_key"] = os.getenv(profile.get("api_key_env", "LLM_API_KEY"), profile.get("api_key", "not-needed"))
    profile["name"] = name
    profile.setdefault("system_suffix", "")
    return profile


def get_llm(**overrides: Any):
    """Return a ChatOpenAI configured for the active profile, or None when the
    profile is "none" (deterministic mode: every node falls back to rule-based
    logic so graphs and tests run with no model at all).
    temperature and seed come from config.global (proposal: temp=0.0, seed=42)."""
    if active_profile_name() == "none":
        return None
    from langchain_openai import ChatOpenAI  # imported lazily so tests without the package still import this module

    cfg = load_config()
    p = active_profile()
    kwargs: Dict[str, Any] = dict(
        model=p["model"],
        base_url=p["base_url"],
        api_key=p["api_key"],
        temperature=cfg["global"]["temperature"],
        seed=cfg["global"]["seed"],
        max_tokens=p.get("max_tokens", 2048),
        # LLM_TIMEOUT_S / LLM_MAX_RETRIES: the live demo lowers these so a stalled server fails over in seconds
        timeout=float(os.getenv("LLM_TIMEOUT_S", p.get("timeout_s", 120))),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", 2)),
    )
    if p.get("extra_body"):
        kwargs["extra_body"] = p["extra_body"]
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)


if __name__ == "__main__":
    p = active_profile()
    print(f"active profile: {p['name']}  model={p['model']}  base_url={p['base_url']}")
    try:
        llm = get_llm()
        print(llm.invoke("Reply with the single word OK.").content)
    except Exception as e:  # noqa: BLE001
        print(f"LLM not reachable: {e}")
