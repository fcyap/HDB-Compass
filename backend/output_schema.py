"""Output layer: what the server sends to the browser is validated and cleaned first (proposal section 5, risks 4 and 5).

Every WebSocket message in backend/main.py goes through `send_clean`, which calls `clean_message`:

  1. SHAPE. A pydantic model per message type states what the frontend reads (frontend/src/App.tsx), e.g. `summary` is a
     string (the page calls `payload.summary.split(...)`), `logs` a list, `viewport` two numbers. A field with the wrong
     type is dropped and logged, so one bad field cannot crash the page or the message. Unknown fields pass through
     unchanged (the map and card payloads are large and owned by other members).
  2. TEXT. The text fields a person reads (the answer, the step logs, the follow-up question and its chips, a status
     message) lose personal and financial identifiers (agents/sanitizer.py, a second line behind the input sanitiser: a tool
     result or the model could still echo one) and any script, style or iframe block and `javascript:` link. Map and
     block data are left alone.
  3. ERRORS. An exception is never sent as text: `safe_error_message` returns a plain sentence and the real error stays in the
     server log (the Groq errors we saw carried an organisation id).

Nothing here raises: a failure falls back to the plain message with only the text cleaned.
"""
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from agents import sanitizer

_SCRIPT_BLOCK = re.compile(r"<\s*(script|style|iframe|object|embed)\b[^>]*>.*?<\s*/\s*\1\s*>", re.I | re.S)
_OPEN_TAG = re.compile(r"<\s*/?\s*(script|style|iframe|object|embed)\b[^>]*>", re.I)
_JS_LINK = re.compile(r"\((\s*javascript\s*:[^)]*)\)", re.I)
_HANDLER = re.compile(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*')", re.I)


def clean_text(text: Any) -> Any:
    """`text` without identifiers, script-like HTML and javascript: links. Anything that is not a string is returned as it is."""
    if not isinstance(text, str):
        return text
    out = _SCRIPT_BLOCK.sub("", text)
    out = _OPEN_TAG.sub("", out)
    out = _HANDLER.sub("", out)
    out = _JS_LINK.sub("(#)", out)
    return sanitizer.redact(out)[0]


# ---------------------------------------------------------------------------
# error text
# ---------------------------------------------------------------------------
BUSY = "The language model is busy right now. Please try again in a minute."
UNREACHABLE = "I couldn't reach the language model. Please try again in a moment."
GENERIC = "Something went wrong while answering. Please try again."


def safe_error_message(exc: Any) -> str:
    """A plain sentence for the user. The real exception stays in the server log, not in the browser."""
    name = type(exc).__name__ if isinstance(exc, BaseException) else ""
    text = f"{name} {exc}".lower()
    if "ratelimit" in text or "429" in text or "rate limit" in text or "quota" in text:
        return BUSY
    if any(k in text for k in ("timeout", "timed out", "connection", "unreachable", "unavailable")):
        return UNREACHABLE
    return GENERIC


# ---------------------------------------------------------------------------
# message shapes (what the frontend reads)
# ---------------------------------------------------------------------------
class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")     # unknown fields pass through unchanged


class StepLog(_Loose):
    step: str = ""
    message: str = ""
    status: str = "completed"


class Viewport(_Loose):
    lat: float
    lng: float
    zoom: Optional[float] = None


class StepUpdate(_Loose):
    type: str
    node: str = ""
    logs: List[StepLog] = []
    summary: Optional[str] = None
    viewport: Optional[Viewport] = None
    geojson_buffer: Optional[Dict[str, Any]] = None
    geojson_amenities: Optional[Dict[str, Any]] = None
    qualifying_blocks: Optional[List[Any]] = None
    nearby_amenities: Optional[List[Any]] = None
    market_summary: Optional[str] = None
    financial_summary: Optional[str] = None
    policy_notes: Optional[str] = None
    school_lookup: Optional[Dict[str, Any]] = None
    clarification: Optional[Dict[str, Any]] = None
    flat_type: Optional[str] = None


class ErrorMessage(_Loose):
    type: str
    error: str = GENERIC


class ModelStatus(_Loose):
    type: str
    status: str = ""
    message: Optional[str] = None


MODELS = {"STEP_UPDATE": StepUpdate, "ERROR": ErrorMessage, "MODEL_STATUS": ModelStatus}
TEXT_FIELDS = ("summary", "market_summary", "financial_summary", "policy_notes", "error", "message")


def _clean_clarification(c: Any) -> Any:
    if not isinstance(c, dict):
        return c
    out = dict(c)
    if "question" in out:
        out["question"] = clean_text(out["question"])
    if isinstance(out.get("options"), list):
        out["options"] = [{**o, **{k: clean_text(o[k]) for k in ("label", "text") if k in o}} if isinstance(o, dict) else o
                          for o in out["options"]]
    return out


def clean_texts(message: Dict[str, Any]) -> Dict[str, Any]:
    """`message` with only the fields people read cleaned; everything else is the same object."""
    out = dict(message)
    for key in TEXT_FIELDS:
        if key in out:
            out[key] = clean_text(out[key])
    if isinstance(out.get("logs"), list):
        out["logs"] = [{**log, **{k: clean_text(log[k]) for k in ("step", "message") if k in log}} if isinstance(log, dict) else log
                       for log in out["logs"]]
    if "clarification" in out:
        out["clarification"] = _clean_clarification(out["clarification"])
    return out


def _validated(message: Dict[str, Any], model: type, dropped: List[str]) -> Dict[str, Any]:
    """The message that fits `model`: a field with the wrong type is dropped (and named in `dropped`), then it is checked again."""
    current = dict(message)
    for _ in range(len(current) + 1):
        try:
            return model.model_validate(current).model_dump(exclude_unset=True)
        except ValidationError as e:
            bad = {err["loc"][0] for err in e.errors() if err.get("loc") and err["loc"][0] in current}
            if not bad:
                break
            for key in bad:
                dropped.append(str(key))
                current.pop(key, None)
    return {k: v for k, v in current.items() if k == "type"}


def clean_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """The message to send: text cleaned, shape checked. Never raises."""
    try:
        if not isinstance(message, dict):
            return {"type": "ERROR", "error": GENERIC}
        out = clean_texts(message)
        model = MODELS.get(str(out.get("type")))
        if model is not None:
            dropped: List[str] = []
            out = _validated(out, model, dropped)
            if dropped:
                print(f"[output schema] dropped fields with the wrong type from a {message.get('type')}: {sorted(set(dropped))}")
        return out
    except Exception as e:  # noqa: BLE001 — the answer matters more than the check
        print(f"[output schema] could not check a message ({type(e).__name__}); sent with its text cleaned")
        try:
            return clean_texts(message)
        except Exception:  # noqa: BLE001
            return message


async def send_clean(websocket: Any, message: Dict[str, Any]) -> None:
    await websocket.send_json(clean_message(message))
