"""
Add-on tool packs (agents/addons). Owned by Member 1.

A pack adds tools to the core multi-agent system without editing the core lists by hand. Each tool is declared
once, as a ToolAddon, and the core reads that declaration wherever it needs it:

  * agents/schemas.py       TOOL_OWNER (which specialist owns it) and TOOL_INPUT_MODEL (argument validation)
  * agents/tool_registry.py REAL_IMPL (where the implementation lives)
  * agents/supervisor.py    TASK_TOOLS (when Variant C offers it), deterministic_plan (the pack's plan steps, so the
                            result survives requested_outputs), AGENT_SYSTEM (one "when to use" line in the owner's
                            prompt) and NOT_CHECKED (what the answer says if its module cannot load)
  * mcp_server/server.py    registration with the tool's output schema

So a tool the agents cannot discover fails tests/test_addons.py instead of failing silently in the app.

Which runs see the add-ons is one switch, `addons.offer_in` in config/experiment_config.yaml (HDB_ADDONS overrides
it for one run, like HDB_CODE_WRITTEN_FACTS):
  app  (default) the app and Variant C; Variants A and B keep the original tool lists, so their measured Tool F1
       stays comparable with earlier runs
  all  every variant
  off  none: the code behaves as it did before the packs existed (the evaluation runner uses this by default)

This module must not import agents.schemas or agents.supervisor at import time: schemas imports it to register the
tools. Input and output models are therefore named as "module:attribute" strings and resolved on first use.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

Plan = List[Tuple[str, Dict[str, Any]]]
OFFER_MODES = ("app", "all", "off")


@dataclass(frozen=True)
class ToolAddon:
    name: str                               # the tool name the model calls
    impl: Tuple[str, str]                   # (module, attribute) of the real implementation
    owner: str                              # "market_sql" | "spatial" | "financial" | "policy_rag"
    input_model: str                        # "module:Class" validating the arguments (what the model sees)
    output_model: str                       # "module:Class" advertised to MCP clients
    offer_when: Callable[[Dict[str, Any]], bool]   # Variant C offers the tool only when this holds (TASK_TOOLS)
    prompt_hint: str                        # one line for the owner's system prompt: when to use it
    not_checked: str                        # "<What> could not be checked" when its module fails to import
    example_query: str                      # a question that must reach this tool (tests/test_addons.py)


@dataclass(frozen=True)
class ToolPack:
    name: str
    tools: Tuple[ToolAddon, ...]
    # deterministic plan steps for one specialist: plan(agent, constraints, query)
    plan: Callable[[str, Dict[str, Any], str], Plan] = field(default=lambda agent, c, query: [])
    # a general rule for the owner's prompt, added once before the tools' hints
    prompt_preamble: Dict[str, str] = field(default_factory=dict)


def _load(ref: str) -> Any:
    module, attr = ref.split(":")
    return getattr(importlib.import_module(module), attr)


def _packs() -> List[ToolPack]:
    from agents.addons import research, schools   # noqa: PLC0415 — packs load lazily (see module docstring)
    return [research.PACK, schools.PACK]


PACKS: List[ToolPack] = []


def packs() -> List[ToolPack]:
    if not PACKS:
        PACKS.extend(_packs())
    return PACKS


def all_addons() -> List[ToolAddon]:
    return [t for p in packs() for t in p.tools]


def addon(name: str) -> Optional[ToolAddon]:
    return next((t for t in all_addons() if t.name == name), None)


def is_addon(name: str) -> bool:
    return addon(name) is not None


def offer_mode() -> str:
    """addons.offer_in, or HDB_ADDONS for one run. Any other value is refused rather than guessed at."""
    env = os.getenv("HDB_ADDONS", "").strip().lower()
    if env:
        if env not in OFFER_MODES:
            raise ValueError(f"HDB_ADDONS must be one of {OFFER_MODES}, not {env!r}")
        return env
    from agents.llm import load_config   # noqa: PLC0415
    mode = str((load_config().get("addons") or {}).get("offer_in", "app")).lower()
    if mode not in OFFER_MODES:
        raise ValueError(f"addons.offer_in must be one of {OFFER_MODES}, not {mode!r}")
    return mode


def active() -> bool:
    """Whether the packs' plans run at all (any mode but off)."""
    return offer_mode() != "off"


def offered_in(variant: Optional[str], fast_path: bool = False) -> bool:
    """Whether a run of this variant may be offered add-on tools."""
    mode = offer_mode()
    return mode == "all" or (mode == "app" and (fast_path or (variant or "C") == "C"))


def visible_tools(names: List[str], variant: Optional[str], fast_path: bool = False) -> List[str]:
    """names without the add-on tools when this run may not see them."""
    if offered_in(variant, fast_path):
        return list(names)
    return [n for n in names if not is_addon(n)]


# --- what the core reads ----------------------------------------------------------------------------------------
def register_tools(tool_owner: Dict[str, str], tool_input_model: Dict[str, Any]) -> None:
    """Called once by agents/schemas.py: add each tool's owner and input model."""
    for t in all_addons():
        if t.name in tool_owner and tool_owner[t.name] != t.owner:
            raise ValueError(f"add-on {t.name} clashes with core tool owned by {tool_owner[t.name]}")
        tool_owner[t.name] = t.owner
        tool_input_model[t.name] = _load(t.input_model)


def real_impls() -> Dict[str, Tuple[str, str]]:
    return {t.name: t.impl for t in all_addons()}


def output_model(name: str) -> Any:
    return _load(addon(name).output_model)


def task_tools() -> Dict[str, Callable[[Dict[str, Any]], bool]]:
    return {t.name: t.offer_when for t in all_addons()}


def plan(agent: str, c: Dict[str, Any], query: str) -> Plan:
    if not active():
        return []
    return [step for p in packs() for step in p.plan(agent, c, query)]


def prompt_section(agent: str) -> str:
    """The owner's extra prompt lines: each pack's rule for this agent, then one line per tool it owns."""
    parts: List[str] = []
    for p in packs():
        hints = [f"{t.name}: {t.prompt_hint}" for t in p.tools if t.owner == agent]
        if hints:
            if p.prompt_preamble.get(agent):
                parts.append(p.prompt_preamble[agent])
            parts.extend(hints)
    return (" Also available when offered: " + "; ".join(parts) + ".") if parts else ""


def not_checked() -> Dict[str, str]:
    return {t.name: t.not_checked for t in all_addons()}
