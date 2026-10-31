"""
LangSmith tracing setup (agents/tracing.py): the one place that decides whether runs are traced.

Why a module of its own: LangChain/LangGraph trace themselves the moment LANGSMITH_TRACING=true, and when the key is
refused (revoked, wrong region, no workspace access) every run then logs an upload error and the agents keep working
untraced with a screenful of noise. So the process checks the credentials ONCE at startup (`preflight`), and
`configure_tracing` switches tracing off with a single clear log line if they fail, instead of failing on every run.

What it gives the rest of the code:
  * preflight() / tracing_status()  -- one read-only call to LangSmith, classified (ok, no_key, 401, 403, ...).
  * configure_tracing()             -- call once at backend start-up and from the runners.
  * run_config(...)                 -- the RunnableConfig (name, tags, metadata) for graph.invoke / astream.
  * traced_tool_call(...)           -- ToolRunner runs the plain Python tool through this so each tool is a child run.
  * `python -m agents.tracing --check` -- prints the diagnosis and the fix to try; exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("hdb_compass.tracing")

US_ENDPOINT = "https://api.smith.langchain.com"
EU_ENDPOINT = "https://eu.api.smith.langchain.com"
DEFAULT_PROJECT = "hdb-compass"
PREFLIGHT_TIMEOUT_S = 5.0
MAX_TRACED_CHARS = 4000   # a tool result (block lists, RAG passages) can be large; the trace keeps a readable head

# preflight verdicts
OK, NO_KEY, INVALID_KEY, FORBIDDEN, PROJECT_MISSING, NETWORK, ERROR = (
    "ok", "no_key", "invalid_key", "forbidden", "project_missing", "network_error", "error")
PASSING = (OK, PROJECT_MISSING)   # a missing project is created by the first run, so it does not block tracing

try:   # imported at module level so tests can replace it; absent langsmith just means no tool spans
    from langsmith import traceable
except Exception:  # noqa: BLE001  # pragma: no cover
    traceable = None  # type: ignore[assignment]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")   # does not override variables already set in the environment
    except Exception:  # noqa: BLE001  # pragma: no cover
        pass


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _env(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v is not None and v.strip():
            return v.strip()
    return None


def tracing_requested() -> bool:
    return _truthy(_env("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"))


def settings() -> Dict[str, Any]:
    return {
        "requested": tracing_requested(),
        "has_key": bool(_env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")),
        "endpoint": (_env("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT") or US_ENDPOINT).rstrip("/"),
        "workspace_id": _env("LANGSMITH_WORKSPACE_ID", "LANGCHAIN_WORKSPACE_ID"),
        "project": _env("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT") or DEFAULT_PROJECT,
        "agent_vars": sorted(k for k in os.environ if k.startswith("LANGSMITH_AGENT_")),
    }


def key_prefix() -> str:
    key = _env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY") or ""
    return key[:8] + "..." if key else ""


# ---------------------------------------------------------------------------------------------- the one HTTP call
def _http_get(url: str, headers: Mapping[str, str], timeout: float) -> int:
    """GET and return the status code; a network failure raises OSError. The only network use in this module's check,
    so tests replace this one function."""
    req = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            _http_get.last_body = body    # type: ignore[attr-defined]
            return resp.status
    except urllib.error.HTTPError as e:
        _http_get.last_body = b""         # type: ignore[attr-defined]
        return e.code
    except urllib.error.URLError as e:
        raise OSError(str(e.reason)) from e


_http_get.last_body = b""  # type: ignore[attr-defined]


def _headers() -> Dict[str, str]:
    h = {"x-api-key": _env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY") or "", "User-Agent": "hdb-compass-preflight"}
    ws = settings()["workspace_id"]
    if ws:
        h["x-tenant-id"] = ws
    return h


def _probe(endpoint: str, project: str, timeout: float) -> Dict[str, Any]:
    """Read the project (a list of at most one session): needs a valid key and workspace access, writes nothing."""
    from urllib.parse import quote
    url = f"{endpoint}/api/v1/sessions?limit=1&name={quote(project)}"
    try:
        code = _http_get(url, _headers(), timeout)
    except OSError as e:
        return {"verdict": NETWORK, "http": None, "detail": str(e)}
    if code == 200:
        try:
            found = bool(json.loads(_http_get.last_body or b"[]"))
        except ValueError:
            found = True
        return {"verdict": OK if found else PROJECT_MISSING, "http": 200}
    if code == 401:
        return {"verdict": INVALID_KEY, "http": 401}
    if code == 403:
        return {"verdict": FORBIDDEN, "http": 403}
    if code == 404:
        return {"verdict": PROJECT_MISSING, "http": 404}
    return {"verdict": ERROR, "http": code}


FIXES = {
    NO_KEY: "Set LANGSMITH_API_KEY in .env (LangSmith > Settings > API Keys).",
    INVALID_KEY: "LangSmith says the key is not valid (401). Create a new key in LangSmith > Settings > API Keys and "
                 "paste it into LANGSMITH_API_KEY; check it was copied whole, with no quotes or spaces.",
    FORBIDDEN: ("LangSmith recognises the request but refuses it (403). The usual causes, in order: (1) the key was "
                "revoked or has expired - create a new one; (2) the account lives in another region - try "
                "LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com (EU) or https://api.smith.langchain.com (US); "
                "(3) a self-hosted LangSmith - set LANGSMITH_ENDPOINT to its URL; (4) an organisation-scoped or "
                "service key - also set LANGSMITH_WORKSPACE_ID to the workspace id; (5) the key's user has no access to "
                "any workspace - ask the workspace admin."),
    NETWORK: "LangSmith could not be reached. Check the network/proxy and LANGSMITH_ENDPOINT.",
    ERROR: "LangSmith answered with an unexpected status; check LANGSMITH_ENDPOINT and https://status.smith.langchain.com.",
}


def preflight(timeout: float = PREFLIGHT_TIMEOUT_S) -> Dict[str, Any]:
    """Check the LangSmith credentials with ONE read-only call. Returns a dict with `verdict` (ok, no_key, invalid_key,
    forbidden, project_missing, network_error, error), `ok` (tracing can work), `message`, `fix` and the settings used.
    Never includes the key."""
    s = settings()
    out: Dict[str, Any] = {"endpoint": s["endpoint"], "project": s["project"], "workspace_id": s["workspace_id"],
                           "requested": s["requested"], "warnings": []}
    if s["agent_vars"]:
        out["warnings"].append(
            f"{', '.join(s['agent_vars'])} is set. In the installed langsmith these variables switch on the beta "
            "'agent address' routing, which fails every run (EnvAddressError / 'No indexed run ID') and leaves runs "
            "untraced. Remove every LANGSMITH_AGENT_* variable.")
    if not s["has_key"]:
        out.update(verdict=NO_KEY, http=None, detail="")
    else:
        out.update(_probe(s["endpoint"], s["project"], timeout))
        if out["verdict"] == FORBIDDEN:
            other = EU_ENDPOINT if s["endpoint"] == US_ENDPOINT else US_ENDPOINT
            if s["endpoint"] in (US_ENDPOINT, EU_ENDPOINT):
                alt = _probe(other, s["project"], timeout)
                out["other_region"] = {"endpoint": other, "verdict": alt["verdict"], "http": alt["http"]}
    v = out["verdict"]
    out["ok"] = v in PASSING and not s["agent_vars"]
    out["fix"] = FIXES.get(v, "")
    if v == FORBIDDEN and out.get("other_region"):
        o = out["other_region"]
        out["fix"] += (f" Probe of {o['endpoint']}: {'WORKED - set LANGSMITH_ENDPOINT to it' if o['verdict'] in PASSING else 'also refused (' + str(o['http']) + ')'}.")
    msgs = {
        OK: f"LangSmith accepts the key; project '{s['project']}' exists.",
        PROJECT_MISSING: f"LangSmith accepts the key; project '{s['project']}' does not exist yet and is created by the first traced run.",
        NO_KEY: "No LangSmith API key is set.",
        INVALID_KEY: f"LangSmith rejects the key (HTTP 401) at {s['endpoint']}.",
        FORBIDDEN: f"LangSmith refuses the key (HTTP 403 Forbidden) at {s['endpoint']}.",
        NETWORK: f"Cannot reach {s['endpoint']}: {out.get('detail')}",
        ERROR: f"Unexpected HTTP {out.get('http')} from {s['endpoint']}.",
    }
    out["message"] = msgs[v]
    return out


_status: Optional[Dict[str, Any]] = None


def tracing_status() -> Dict[str, Any]:
    """The result of the last configure_tracing()/preflight in this process (settings only if neither has run)."""
    return dict(_status) if _status else {"configured": False, **settings()}


def _set_tracing_env(on: bool) -> None:
    value = "true" if on else "false"
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        if name in os.environ or not on:
            os.environ[name] = value
    if on:
        os.environ["LANGSMITH_TRACING"] = "true"
    try:   # langsmith memoises environment reads; without this the switch is not seen
        from langsmith import utils as ls_utils
        ls_utils.get_env_var.cache_clear()
    except Exception:  # noqa: BLE001  # pragma: no cover
        pass


def tracing_enabled() -> bool:
    """Whether LangSmith will record a run started now (the cheap check the tool wrapper uses)."""
    try:
        from langsmith.utils import tracing_is_enabled
        return bool(tracing_is_enabled())
    except Exception:  # noqa: BLE001
        return False


def configure_tracing(force_check: bool = False, quiet: bool = False) -> Dict[str, Any]:
    """Call once per process. If tracing is requested but the credentials fail the preflight, turn it OFF for this
    process and log one line, instead of letting every run fail its upload. If it passes, leave native tracing on and
    set the project. Safe to call again (the check is not repeated unless force_check)."""
    global _status
    if _status is not None and not force_check:
        return dict(_status)
    _load_env()
    s = settings()
    if not s["requested"]:
        _status = {"configured": True, "enabled": False, "reason": "tracing not requested (LANGSMITH_TRACING is not true)",
                   "project": s["project"]}
        return dict(_status)
    pf = preflight()
    if pf["ok"]:
        os.environ["LANGSMITH_PROJECT"] = s["project"]
        _set_tracing_env(True)
        _status = {"configured": True, "enabled": True, "reason": pf["message"], "project": s["project"],
                   "endpoint": s["endpoint"], "verdict": pf["verdict"]}
        if not quiet:
            log.info("LangSmith tracing on: project '%s' at %s", s["project"], s["endpoint"])
        for w in pf["warnings"]:
            log.warning(w)
    else:
        _set_tracing_env(False)
        reason = pf["message"] if pf["verdict"] not in PASSING else pf["warnings"][0]
        _status = {"configured": True, "enabled": False, "reason": reason, "verdict": pf["verdict"],
                   "project": s["project"], "endpoint": s["endpoint"]}
        if not quiet:
            log.warning("LangSmith tracing DISABLED for this process: %s Fix: %s (run `python -m agents.tracing --check`)",
                        reason, pf["fix"] or "see the warning above")
    return dict(_status)


# --------------------------------------------------------------------------------------------- run configuration
_COMMIT: Optional[str] = None


def code_commit() -> str:
    global _COMMIT
    if _COMMIT is None:
        try:
            _COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                     text=True, timeout=3).stdout.strip() or "unknown"
        except Exception:  # noqa: BLE001
            _COMMIT = "unknown"
    return _COMMIT


def _model_profile() -> str:
    try:
        from agents.llm import active_profile
        return str(active_profile().get("name"))
    except Exception:  # noqa: BLE001
        return os.getenv("LLM_PROFILE", "unknown")


def _addons_mode() -> str:
    try:
        from agents import addons
        return addons.offer_mode()
    except Exception:  # noqa: BLE001
        return os.getenv("HDB_ADDONS", "unknown")


def run_config(variant: str, session_id: Optional[str] = None, question_id: Optional[str] = None,
               tags: Sequence[str] = (), metadata: Optional[Mapping[str, Any]] = None, source: Optional[str] = None
               ) -> Dict[str, Any]:
    """RunnableConfig for graph.invoke / astream: a run name that reads well in the trace list, tags to filter by
    (variant, app/eval) and the metadata needed to compare runs later (which model, which add-on mode, which code)."""
    source = source or ("eval" if question_id else "app")
    meta: Dict[str, Any] = {"variant": variant, "model_profile": _model_profile(), "addons": _addons_mode(),
                            "commit": code_commit(), "source": source}
    if session_id:
        meta["session_id"] = session_id
        meta["thread_id"] = session_id   # LangSmith groups runs into a thread by this key
    if question_id:
        meta["question_id"] = question_id
    meta.update(metadata or {})
    name = f"hdb-compass-{variant}" + (f":{question_id}" if question_id else "")
    return {"run_name": name, "tags": [f"variant:{variant}", source, *tags], "metadata": meta}


# -------------------------------------------------------------------------------------------------- tool spans
def _shrink(value: Any, limit: int = MAX_TRACED_CHARS) -> Any:
    """Keep a tool result readable in the UI: small values pass through, large ones become a head plus a size note."""
    try:
        text = json.dumps(value, default=str)
    except Exception:  # noqa: BLE001
        text = str(value)
    if len(text) <= limit:
        return value
    return {"_truncated": True, "chars": len(text), "head": text[:limit]}


def traced_tool_call(name: str, fn: Callable[..., Any], arguments: Dict[str, Any]) -> Any:
    """Run one tool as a LangSmith child run (type tool). When tracing is off - the default for tests and for a
    refused key - this is a plain call, so the cost is one environment check."""
    if traceable is None or not tracing_enabled():
        return fn(**arguments)
    wrapped = traceable(run_type="tool", name=name,
                        process_inputs=lambda inputs: {k: _shrink(v) for k, v in inputs.items()},
                        process_outputs=lambda out: {"output": _shrink(out)})(lambda **kw: fn(**kw))
    return wrapped(**arguments)


# ------------------------------------------------------------------------------------------------------- CLI
def format_check(pf: Dict[str, Any]) -> str:
    lines = ["LangSmith tracing check",
             f"  tracing requested : {'yes' if pf['requested'] else 'no (LANGSMITH_TRACING is not true)'}",
             f"  api key           : {key_prefix() or 'NOT SET'}",
             f"  endpoint          : {pf['endpoint']}",
             f"  workspace id      : {pf['workspace_id'] or '(not set)'}",
             f"  project           : {pf['project']}",
             f"  verdict           : {pf['verdict'].upper()}  {pf['message']}"]
    if pf.get("other_region"):
        o = pf["other_region"]
        lines.append(f"  other region      : {o['endpoint']} -> {o['verdict']} (HTTP {o['http']})")
    for w in pf["warnings"]:
        lines.append(f"  WARNING           : {w}")
    if pf["fix"]:
        lines.append(f"  what to try       : {pf['fix']}")
    lines.append("  result            : " + ("OK, runs will be traced" if pf["ok"] else "FAILED, runs will NOT be traced"))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LangSmith tracing diagnosis")
    ap.add_argument("--check", action="store_true", help="check the credentials and print the diagnosis")
    ap.parse_args(argv)
    _load_env()
    pf = preflight()
    print(format_check(pf))
    return 0 if pf["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
