"""Per-chat model settings for the app's model dropdown (Member 4, 6 Nov).

The app's backend serves one process for every chat, and the model was chosen once, by the environment the backend
started with (LLM_PROFILE and friends, set by cluster/demo_start.sh). The dropdown lets each chat pick its own model:
backend/main.py runs each turn inside `use(MODEL_OPTIONS[choice]["settings"])`, and the readers of those settings
(agents/llm.py, agents/supervisor.py) ask `setting(name)`, which prefers the chat's choice over the environment. A
context variable carries it, so two chats running at once never see each other's choice; with no choice made (the
evaluation, the tests, the harness, the CLI) `setting` is exactly os.getenv.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional

_CHOSEN: ContextVar[Dict[str, str]] = ContextVar("hdb_chat_settings", default={})


def setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """The chat's own value for `name` when its model choice sets one, else the environment's."""
    chosen = _CHOSEN.get()
    if name in chosen:
        return chosen[name]
    return os.getenv(name, default)


@contextmanager
def use(settings: Dict[str, str]) -> Iterator[None]:
    """Run the block with these settings over the environment (for the turns of one chat)."""
    token = _CHOSEN.set({**_CHOSEN.get(), **{k: str(v) for k, v in settings.items()}})
    try:
        yield
    finally:
        _CHOSEN.reset(token)


def choose(settings: Dict[str, str]) -> None:
    """These settings for the rest of the current context: one websocket connection, which handles its chat's turns one
    at a time (backend/main.py). Each connection runs in its own task, so another chat's choice never reaches it."""
    _CHOSEN.set({k: str(v) for k, v in (settings or {}).items()})


def model_options() -> list:
    """The dropdown's choices, each with whether it can answer now: Claude and Gemini when their key is set (and the
    package installed), the cluster model when its tunnel answers, offline always. Calls no model."""
    from agents import llm as L
    out = []
    for key, opt in MODEL_OPTIONS.items():
        available, why = True, ""
        if key != "none":
            with use(opt["settings"]):
                try:
                    p = L.active_profile()
                    if key == "claude":
                        import langchain_anthropic  # noqa: F401
                    if key == "vllm":
                        import requests
                        r = requests.get(p["base_url"] + "/models", timeout=(1.0, 1.5),
                                         headers={"Authorization": f"Bearer {p['api_key']}"})
                        available, why = r.ok, ("" if r.ok else f"the cluster model answered HTTP {r.status_code}")
                except ImportError:
                    available, why = False, "its package is not installed"
                except L.ModelConfigurationError as e:
                    available, why = False, str(e)
                except Exception:  # noqa: BLE001 — no tunnel, no network
                    available, why = False, "not reachable (start the app with the cluster to use it)"
        out.append({"id": key, "label": opt["label"], "available": available, "reason": why})
    return out


def started_with() -> str:
    """The dropdown's starting choice: the model the backend was started with (cluster/demo_start.sh)."""
    name = (os.getenv("LLM_PROFILE") or "").strip().lower()
    return name if name in MODEL_OPTIONS else "none"


# The models the dropdown offers, each with the settings it brings. Claude answers in the conversational style the
# 6 Nov plan built for it; Gemini, the cluster model and offline keep the answers as measured (the template style).
MODEL_OPTIONS: Dict[str, Dict[str, object]] = {
    "claude": {"label": "Claude Haiku 5.5", "settings": {
        "LLM_PROFILE": "claude", "HDB_ANSWER_STYLE": "conversational", "HDB_READING_CONTEXT": "1",
        "HDB_TRUST_MODEL_READING": "0", "LLM_TIMEOUT_S": "60", "LLM_MAX_RETRIES": "1"}},
    "gemini": {"label": "Gemini 3.5 Flash-Lite", "settings": {
        "LLM_PROFILE": "gemini", "HDB_ANSWER_STYLE": "template", "HDB_READING_CONTEXT": "0",
        "HDB_TRUST_MODEL_READING": "0", "LLM_TIMEOUT_S": "60", "LLM_MAX_RETRIES": "1"}},
    "vllm": {"label": "Cluster · Qwen3-30B", "settings": {
        "LLM_PROFILE": "vllm", "HDB_ANSWER_STYLE": "template", "HDB_READING_CONTEXT": "0",
        "HDB_TRUST_MODEL_READING": "0", "LLM_TIMEOUT_S": "25", "LLM_MAX_RETRIES": "0"}},
    "none": {"label": "Offline (no model)", "settings": {
        "LLM_PROFILE": "none", "HDB_ANSWER_STYLE": "template", "HDB_READING_CONTEXT": "0",
        "HDB_TRUST_MODEL_READING": "0"}},
}
