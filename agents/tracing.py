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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("hdb_compass.tracing")

US_ENDPOINT = "https://api.smith.langchain.com"
EU_ENDPOINT = "https://eu.api.smith.langchain.com"
APAC_ENDPOINT = "https://apac.api.smith.langchain.com"
REGIONS = {"US": US_ENDPOINT, "EU": EU_ENDPOINT, "APAC": APAC_ENDPOINT}
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
    endpoint = _env("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT")
    return {
        "requested": tracing_requested(),
        "has_key": bool(_env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")),
        "endpoint": (endpoint or US_ENDPOINT).rstrip("/"),
        "endpoint_set": bool(endpoint),
        "workspace_id": _env("LANGSMITH_WORKSPACE_ID", "LANGCHAIN_WORKSPACE_ID"),
        "project": _env("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),   # None when unset
        "agent_id": _env("LANGSMITH_AGENT_ID"),
        "agent_env": _env("LANGSMITH_AGENT_ENVIRONMENT"),
    }


def key_prefix() -> str:
    key = _env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY") or ""
    return key[:8] + "..." if key else ""


def region_name(endpoint: str) -> str:
    for name, url in REGIONS.items():
        if endpoint.rstrip("/") == url:
            return name
    return "custom"


# ---------------------------------------------------------------------------------------------- destination
def destination(s: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Where runs will land, and whether the combination of variables is usable. kind is agent (an agent address:
    LANGSMITH_AGENT_ID + LANGSMITH_AGENT_ENVIRONMENT), project (LANGSMITH_PROJECT) or default (neither: LangSmith uses a
    project called "default"). ok=False with `problem` and `fix` for the combinations that leave runs untraced:
    half an address, or an address together with a project."""
    s = s or settings()
    aid, aenv, project = s["agent_id"], s["agent_env"], s["project"]
    if bool(aid) != bool(aenv):
        have, miss = ("LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT") if aid else ("LANGSMITH_AGENT_ENVIRONMENT", "LANGSMITH_AGENT_ID")
        return {"kind": "invalid", "ok": False, "label": "half an agent address",
                "problem": f"{have} is set without {miss}; the agent address needs both, and with only one every run fails "
                           "(EnvAddressError / 'No indexed run ID') and is not traced.",
                "fix": f"Set {miss} as well (the LangSmith agent page shows both), or remove {have}."}
    if aid and aenv:
        if project:
            return {"kind": "invalid", "ok": False, "label": f"agent address AND project '{project}'",
                    "problem": "an agent address and LANGSMITH_PROJECT are both set; langsmith refuses to choose between them "
                               "('Unset one of them') and the runs are not traced.",
                    "fix": "Remove the LANGSMITH_PROJECT line from .env (keeps the agent address), or remove "
                           "LANGSMITH_AGENT_ID and LANGSMITH_AGENT_ENVIRONMENT (keeps the project)."}
        try:   # langsmith's own validation, offline
            from langsmith import utils as ls_utils
            ls_utils.get_env_var.cache_clear()
            from langsmith.address import address_from_env
            address_from_env()
        except Exception as e:  # noqa: BLE001
            return {"kind": "invalid", "ok": False, "label": f"agent address {aid}/{aenv} (invalid)",
                    "problem": f"langsmith rejects the agent address: {e}",
                    "fix": "Correct LANGSMITH_AGENT_ID / LANGSMITH_AGENT_ENVIRONMENT as shown on the LangSmith agent page."}
        return {"kind": "agent", "ok": True, "label": f"agent '{aid}', environment '{aenv}'", "problem": "", "fix": ""}
    if project:
        return {"kind": "project", "ok": True, "label": f"project '{project}'", "problem": "", "fix": ""}
    return {"kind": "default", "ok": True, "label": "project 'default' (neither an agent address nor LANGSMITH_PROJECT is set)",
            "problem": "", "fix": "", "note": "Runs go to the LangSmith project called 'default'. Set LANGSMITH_PROJECT=<name> or an "
                                              "agent address (LANGSMITH_AGENT_ID + LANGSMITH_AGENT_ENVIRONMENT) to choose."}


# ---------------------------------------------------------------------------------------------- the one HTTP call
def _http_get(url: str, headers: Mapping[str, str], timeout: float) -> Tuple[int, bytes]:
    """GET and return (status, body); a network failure raises OSError. The only network use of the check, so tests
    replace this one function."""
    req = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except urllib.error.URLError as e:
        raise OSError(str(e.reason)) from e


def _headers() -> Dict[str, str]:
    h = {"x-api-key": _env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY") or "", "User-Agent": "hdb-compass-preflight"}
    ws = settings()["workspace_id"]
    if ws:
        h["x-tenant-id"] = ws
    return h


def _probe(endpoint: str, project: Optional[str], timeout: float) -> Dict[str, Any]:
    """Read at most one project (session): needs a valid key and workspace access, writes nothing. With a project name
    it also reports whether that project exists yet."""
    from urllib.parse import quote
    url = f"{endpoint}/api/v1/sessions?limit=1" + (f"&name={quote(project)}" if project else "")
    try:
        code, body = _http_get(url, _headers(), timeout)
    except OSError as e:
        return {"verdict": NETWORK, "http": None, "detail": str(e)}
    if code == 200:
        if not project:
            return {"verdict": OK, "http": 200}
        try:
            found = bool(json.loads(body or b"[]"))
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


def _fix_for(verdict: str, endpoint: str, accepted: Optional[Dict[str, str]]) -> str:
    if verdict in (FORBIDDEN, INVALID_KEY) and accepted:
        return (f"The key is valid but belongs to the {accepted['region']} region. Set "
                f"LANGSMITH_ENDPOINT={accepted['endpoint']} in .env (the SDK defaults to the US region).")
    return {
        NO_KEY: "Set LANGSMITH_API_KEY in .env (LangSmith > Settings > API Keys).",
        INVALID_KEY: f"LangSmith rejects the key (401) at {endpoint} and no other region accepts it. Create a new key in "
                     "LangSmith > Settings > API Keys and paste it whole into LANGSMITH_API_KEY (no quotes or spaces).",
        FORBIDDEN: f"LangSmith refuses the key (403) at {endpoint} and no other region accepts it: the key was revoked or "
                   "expired (create a new one), is organisation-scoped (also set LANGSMITH_WORKSPACE_ID), or the "
                   "account is self-hosted (set LANGSMITH_ENDPOINT to its URL).",
        NETWORK: "LangSmith could not be reached. Check the network/proxy and LANGSMITH_ENDPOINT.",
        ERROR: "LangSmith answered with an unexpected status; check LANGSMITH_ENDPOINT and https://status.smith.langchain.com.",
    }.get(verdict, "")


def preflight(timeout: float = PREFLIGHT_TIMEOUT_S) -> Dict[str, Any]:
    """Check the LangSmith credentials and destination. One read-only call to the configured endpoint; on 401/403 also
    a read-only probe of the US, EU and APAC regions to say which one accepts the key. Returns `verdict` (ok, no_key,
    invalid_key, forbidden, project_missing, network_error, error), `ok` (tracing can work), `message`, `fix` (the
    exact thing to change, "" when ok), `destination` and the settings used. Never includes the key."""
    s = settings()
    dest = destination(s)
    project = s["project"] if dest["kind"] == "project" else None
    out: Dict[str, Any] = {"endpoint": s["endpoint"], "region": region_name(s["endpoint"]), "endpoint_set": s["endpoint_set"],
                           "destination": dest, "workspace_id": s["workspace_id"], "requested": s["requested"],
                           "accepted_region": None}
    if not s["has_key"]:
        out.update(verdict=NO_KEY, http=None, detail="")
    else:
        out.update(_probe(s["endpoint"], project, timeout))
        if out["verdict"] in (FORBIDDEN, INVALID_KEY):
            for name, url in REGIONS.items():
                if url == s["endpoint"]:
                    continue
                alt = _probe(url, None, timeout)
                out.setdefault("region_probes", {})[name] = alt["http"] if alt["http"] else alt["verdict"]
                if alt["verdict"] in PASSING:
                    out["accepted_region"] = {"region": name, "endpoint": url}
                    break
    v = out["verdict"]
    out["ok"] = v in PASSING and dest["ok"]
    where = f"{out['region']} region, {s['endpoint']}"
    msgs = {
        OK: f"LangSmith accepts the key ({where}).",
        PROJECT_MISSING: f"LangSmith accepts the key ({where}); project '{s['project']}' does not exist yet and is created by the first traced run.",
        NO_KEY: "No LangSmith API key is set.",
        INVALID_KEY: f"LangSmith rejects the key (HTTP 401) at {where}.",
        FORBIDDEN: f"LangSmith refuses the key (HTTP 403 Forbidden) at {where}.",
        NETWORK: f"Cannot reach {s['endpoint']}: {out.get('detail')}",
        ERROR: f"Unexpected HTTP {out.get('http')} from {s['endpoint']}.",
    }
    out["message"] = msgs[v]
    fixes = []
    if v not in PASSING:
        fixes.append(_fix_for(v, s["endpoint"], out["accepted_region"]))
    if not dest["ok"]:
        fixes.append(dest["fix"])
    out["fix"] = " ".join(f for f in fixes if f)
    out["problems"] = ([out["message"]] if v not in PASSING else []) + ([dest["problem"]] if not dest["ok"] else [])
    return out


_status: Optional[Dict[str, Any]] = None


def tracing_status() -> Dict[str, Any]:
    """The result of the last configure_tracing() in this process (the settings only if it has not run)."""
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
    """Call once per process. If tracing is requested but the credentials or destination fail the preflight, turn it
    OFF for this process with ONE log line (instead of an upload error on every run). If it passes, native LangSmith
    tracing is left on exactly as configured (endpoint, agent address or project are not touched). Safe to call again
    (the check is not repeated unless force_check)."""
    global _status
    if _status is not None and not force_check:
        return dict(_status)
    _load_env()
    s = settings()
    if not s["requested"]:
        _status = {"configured": True, "enabled": False, "reason": "tracing not requested (LANGSMITH_TRACING is not true)",
                   "destination": destination(s)["label"]}
        return dict(_status)
    pf = preflight()
    dest = pf["destination"]
    if pf["ok"]:
        _set_tracing_env(True)
        _status = {"configured": True, "enabled": True, "reason": pf["message"], "destination": dest["label"],
                   "endpoint": s["endpoint"], "verdict": pf["verdict"]}
        if not quiet:
            log.info("LangSmith tracing on: %s at %s%s", dest["label"], s["endpoint"],
                     f" ({dest['note']})" if dest.get("note") else "")
    else:
        _set_tracing_env(False)
        reason = " ".join(pf["problems"])
        _status = {"configured": True, "enabled": False, "reason": reason, "verdict": pf["verdict"],
                   "destination": dest["label"], "endpoint": s["endpoint"]}
        if not quiet:
            log.warning("LangSmith tracing DISABLED for this process: %s Fix: %s (details: python -m agents.tracing --check)",
                        reason, pf["fix"])
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
    dest = pf["destination"]
    lines = ["LangSmith tracing check",
             f"  tracing requested : {'yes' if pf['requested'] else 'no (LANGSMITH_TRACING is not true)'}",
             f"  api key           : {key_prefix() or 'NOT SET'}",
             f"  endpoint          : {pf['endpoint']} ({pf['region']} region{'' if pf['endpoint_set'] else ', the default: LANGSMITH_ENDPOINT is not set'})",
             f"  workspace id      : {pf['workspace_id'] or '(not set)'}",
             f"  destination       : {dest['label']}" + ("" if dest["ok"] else "  <-- PROBLEM"),
             f"  key check         : {pf['verdict'].upper()}  {pf['message']}"]
    if pf.get("region_probes"):
        lines.append("  other regions     : " + ", ".join(f"{k} -> {v}" for k, v in pf["region_probes"].items()))
    if dest.get("note"):
        lines.append(f"  note              : {dest['note']}")
    if not dest["ok"]:
        lines.append(f"  destination issue : {dest['problem']}")
    if pf["fix"]:
        lines.append(f"  exact fix         : {pf['fix']}")
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
