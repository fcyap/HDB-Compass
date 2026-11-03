"""Member 5: input guardrail and runtime auditor (docs/INTERFACES.md section 3).

Layer 1, `input_guardrail_node`, runs before extraction. Rules only, no model: SQL / prompt injection, NRIC and phone
numbers (the message is refused, since a node cannot rewrite `messages`), requests for guaranteed or speculative profit,
then Member 1's scope check (greeting, off-topic, condo, forecast). Anything else passes with `{}`.

Layer 4, `runtime_auditor_node`, runs after `synthesize`. It judges what the model wrote (in Variant C, the opening before
the code-written facts) against the tool results in the state: first by rules (every figure, distance, lease and block
number must appear in the evidence; no speculative-gain or stigmatising wording), then, only when a model is configured
and the rules pass, by an LLM judge. Both nodes never raise: a failure is logged in `errors` and the answer passes.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from agents import sanitizer, scope
from agents.schemas import AuditorVerdict


class _Supervisor:
    """The supervisor imports this module while it loads, so it is imported late, on first use (as scope.py does)."""
    def __getattr__(self, name: str) -> Any:
        from agents import supervisor
        return getattr(supervisor, name)


sv = _Supervisor()

# ---------------------------------------------------------------------------
# Layer 1: input guardrail
# ---------------------------------------------------------------------------
_SQL = re.compile(
    r"\b(?:drop|truncate|alter)\s+(?:the\s+)?(?:\w+\s+)?(?:table|database|schema)\b"
    r"|\bdelete\s+from\b|\binsert\s+into\b|\bupdate\s+\w+\s+set\b"
    r"|\bunion\s+(?:all\s+)?select\b|\bselect\b[^.?!]{0,60}\bfrom\b[^.?!]{0,40}\b(?:transactions?|resale|duckdb|information_schema)\b"
    r"|;\s*--|'\s*or\s+'?1'?\s*=\s*'?1|\bor\s+1\s*=\s*1\b|\bxp_cmdshell\b|\battach\s+database\b|\bcopy\s+\w+\s+to\b",
    re.I)
_SPECULATIVE = re.compile(
    r"\bguarantee[sd]?\b[^.?!]{0,60}\b(?:profit|gains?|returns?|appreciat\w*|money|make|earn)\b"
    r"|\b(?:profit|gains?|returns?|appreciat\w*)\b[^.?!]{0,40}\bguarantee[sd]?\b"
    r"|\b(?:sure|risk[- ]?free|can'?t lose|no risk|certain|definite)\b[^.?!]{0,40}\b(?:profit|gains?|money|returns?|win|investment)\b"
    r"|\bwhich\b[^.?!]{0,60}\b(?:will|going to|gonna)\b[^.?!]{0,40}\b(?:make|earn|profit|appreciate|double)\b"
    r"|\b(?:make|earn)\b[^.?!]{0,20}\$?\s?\d[\d,.]*\s?(?:k|m|million|thousand)?\b[^.?!]{0,30}\b(?:profit|gain)\b"
    r"|\bbest\b[^.?!]{0,30}\b(?:investment|flip)\b|\bquick (?:flip|profit)\b"
    r"|\b(?:profit|gains?|returns?)\b[^.?!]{0,25}\b(?:i|we)\s+(?:will|would|can|could|'ll)\s+(?:make|earn|get|gain)\b"
    r"|\bhow much (?:profit|money)\b[^.?!]{0,30}\b(?:will|would|can)\b[^.?!]{0,15}\b(?:make|earn|gain)\b",
    re.I)
_OFF_TASK = re.compile(
    r"\b(?:write|create|generate|build|code|debug|fix)\b[^.?!]{0,30}\b(?:python|javascript|java|sql|html|css|script|program|code|"
    r"function|bot|scraper|essay|poem|story|email|cover letter)\b|\bscrape\w*\b[^.?!]{0,30}\b(?:web|website|listings?|sites?)\b",
    re.I)

SQL_REPLY = ("I can't act on that request. I can only help with questions about HDB housing, and I can't change my "
             "instructions or access the database directly. For example, you could ask: "
             "\"What is the median 4-room price in Tampines?\"")
_PII_NAMES = {"NRIC": "NRIC/FIN number", "PHONE": "phone number", "CARD": "card number",
              "BANK_ACCOUNT": "bank account number", "EMAIL": "email address", "SECRET": "password or PIN"}


def pii_reply(kinds: List[str]) -> str:
    """The refusal for a message that carried identifiers; it names each kind found, e.g. "a bank account number"."""
    names = [_PII_NAMES[k] for k in kinds if k in _PII_NAMES] or ["personal identifier"]
    found = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return (f"For your security, I've removed the {found} from your message and have not processed it. Please don't "
            "share personal or financial identifiers such as NRIC/FIN, phone, card or account numbers. I don't need them. "
            "To estimate your grants, just tell me your household income, citizenship status and ages.")


SPECULATIVE_REPLY = ("I can't promise or predict profit on any flat or BTO launch; resale prices depend on location, lease "
                     "and market conditions, and past prices don't guarantee future ones. What I can show is how prices "
                     "have moved: for example 'How have 4-room prices in Tampines changed over the last 5 years?'. "
                     "Note that BTO and subsidised flats carry a Minimum Occupation Period (MOP), and selling a subsidised "
                     "flat can mean clawing back part of the CPF housing grants. This is not financial advice.")


def _block(reason: str, reply: str) -> Dict[str, Any]:
    return {"draft_response": reply, "errors": [f"input_guardrail: {reason}"],
            "auditor_verdict": {"passed": True, "critique": "blocked_by_input_guardrail", "retry_count": 0}}


def classify_input(text: str, mid_conversation: bool = False) -> Tuple[str, Optional[str]]:
    """(kind, fixed reply); kind is "ok" when the message may go on. The order matters: attacks first, then privacy,
    then speculative asks, then the scope check."""
    t = text or ""
    if _SQL.search(t):
        return "sql_injection", SQL_REPLY
    if scope.classify(t, mid_conversation) == scope.INJECTION:
        return "prompt_injection", scope.REPLIES[scope.INJECTION]
    kinds = sanitizer.redact(t)[1] + [k for k, ph in sanitizer.PLACEHOLDERS.items() if ph in t]   # raw (evaluation) or redacted (the app)
    if kinds:
        return "pii", pii_reply(list(dict.fromkeys(kinds)))
    if _OFF_TASK.search(t):   # a coding or writing job that happens to mention property
        return scope.OFF_TOPIC, scope.REPLIES[scope.OFF_TOPIC]
    if scope.classify(t, mid_conversation) == scope.PRIVATE:   # before the speculative check: "best condo for investment"
        return scope.PRIVATE, scope.REPLIES[scope.PRIVATE]
    if _SPECULATIVE.search(t):
        return "speculative_advice", SPECULATIVE_REPLY
    kind = scope.classify(t, mid_conversation)
    if kind != scope.IN_SCOPE:
        return kind, scope.REPLIES[kind]
    return "ok", None


def input_guardrail_node(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        user_turns = [m for m in state.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"]
        mid = bool(state.get("pending_clarification")) or len(user_turns) > 1 or bool(state.get("map_ui_payload"))
        kind, reply = classify_input(sv.last_user_query(state), mid)
        return {} if reply is None else _block(kind, reply)
    except Exception as e:  # noqa: BLE001 — a guardrail fault must not stop the answer
        return {"errors": [f"input_guardrail: {type(e).__name__}"]}


# ---------------------------------------------------------------------------
# Layer 4: runtime auditor
# ---------------------------------------------------------------------------
EVIDENCE_KEYS = ("candidate_blocks_intersection", "relaxation", "sql_market_results", "spatial_block_candidates",
                 "financial_breakdown", "rag_policy_passages", "user_constraints", "nearby_school_results")

_NUM = r"\d[\d,]*(?:\.\d+)?"
_MONEY = re.compile(rf"\$\s?({_NUM})\s*(k|m|mil|million|thousand)?\b|\b({_NUM})\s*(k|million|mil)\b", re.I)
_METRES = re.compile(rf"\b({_NUM})\s*(m|metres?|meters?|km|kilomet(?:re|er)s?)\b(?!\w)", re.I)
_YEARS = re.compile(rf"\b({_NUM})\s*(?:years?|yrs?)\b", re.I)
_BLOCK = re.compile(r"\b(?:blk|block)\.?\s*(\d{1,3}[A-Z]?)\b", re.I)
_SPECULATIVE_CLAIM = re.compile(
    r"\bguarantee[sd]?\b|\brisk[- ]?free\b|\bcan'?t lose\b|\bsure (?:profit|gain|thing|bet)\b"
    r"|\bwill (?:definitely |surely |certainly )?(?:appreciate|increase in value|make you|double|go up|rise)\b"
    r"|\b(?:certain|assured) (?:to )?(?:profit|gain|appreciat\w*)\b",
    re.I)
_STIGMA = re.compile(r"\b(?:ghetto|slum|dangerous (?:area|neighbou?rhood|estate)|bad (?:area|neighbou?rhood)|"
                     r"undesirable|dodgy (?:area|estate)|rough (?:area|neighbou?rhood)|low[- ]class)\b", re.I)


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def _scale(unit: Optional[str]) -> float:
    u = (unit or "").lower()
    return 1e6 if u in ("m", "mil", "million") else 1e3 if u in ("k", "thousand") else 1.0


def evidence_text(state: Dict[str, Any]) -> str:
    return json.dumps({k: state.get(k) for k in EVIDENCE_KEYS if state.get(k) is not None}, default=str)


def _evidence_numbers(text: str) -> Set[float]:
    return {_to_float(m) for m in re.findall(_NUM, text)}


def _near(value: float, pool: Set[float], rel: float = 0.0, tol: Optional[float] = None) -> bool:
    """Whether `value` matches a figure in `pool`. `tol` is the allowed rounding difference: by default $1 for figures of
    100 or more and 0.05 below that; years and metres pass ROUNDING, because the answer shows 72.58 years as "73 yrs"."""
    tol = (1.0 if value >= 100 else 0.05) if tol is None else tol
    return any(abs(value - p) <= max(tol, rel * abs(p)) for p in pool)


ROUNDING = 0.99   # a lease or distance shown rounded (or cut) to a whole number is still the evidence's figure


def ungrounded_claims(draft: str, evidence: str, extra_allowed: str = "") -> List[str]:
    """Figures, distances, leases and block numbers in `draft` that appear nowhere in `evidence`. `extra_allowed` is text
    the user wrote: a figure the user stated may be repeated back. Abbreviated money ($650k, $1.2m) may differ from the
    exact figure by rounding (0.5%); every other figure must match exactly."""
    pool = _evidence_numbers(evidence + " " + extra_allowed)
    low = (evidence + " " + extra_allowed).lower()
    spaceless = re.sub(r"\s+", "", low)
    bad: List[str] = []
    for m in _MONEY.finditer(draft):
        if re.sub(r"\s+", "", m.group(0)).lower() in spaceless:   # written in the evidence word for word ("$21k extended")
            continue
        num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        value = _to_float(num) * _scale(unit)
        # "$748k" may differ from the exact figure by rounding (0.5%); an unabbreviated figure must match to the dollar
        if not _near(value, pool, rel=0.005 if unit else 0.0):
            bad.append(m.group(0).strip())
    for m in _METRES.finditer(draft):
        km = m.group(2).lower().startswith("k")
        number = _to_float(m.group(1))
        # the evidence holds metres; "0.42 km" is 420 m, and a km figure may also be what the evidence itself holds
        if not (_near(number * (1000 if km else 1), pool, tol=ROUNDING) or (km and _near(number, pool, tol=ROUNDING))):
            bad.append(m.group(0).strip())
    for m in _YEARS.finditer(draft):
        if not _near(_to_float(m.group(1)), pool, tol=ROUNDING):
            bad.append(m.group(0).strip())
    for m in _BLOCK.finditer(draft):
        if not re.search(rf"(?<![\w.]){re.escape(m.group(1).lower())}(?![\w.])", low):   # a whole token, not part of a larger number
            bad.append(m.group(0).strip())
    return list(dict.fromkeys(bad))


def compliance_issues(draft: str) -> List[str]:
    out = [f"speculative claim: '{m.group(0)}'" for m in _SPECULATIVE_CLAIM.finditer(draft)]
    out += [f"stigmatising wording: '{m.group(0)}'" for m in _STIGMA.finditer(draft)]
    return list(dict.fromkeys(out))


def audit_text(draft: str, evidence: str, user_text: str = "") -> Dict[str, Any]:
    """The rule-based verdict on one piece of model-written text: {"passed", "critique", "ungrounded_claims"}."""
    claims = ungrounded_claims(draft, evidence, user_text)
    issues = compliance_issues(draft)
    if not claims and not issues:
        return {"passed": True, "critique": "", "ungrounded_claims": []}
    parts = []
    if claims:
        parts.append("these figures are not in the tool results: " + ", ".join(claims) + "; remove them or use only figures from the data")
    parts += [f"{i}; remove it and report objective, historical facts only" for i in issues]
    return {"passed": False, "critique": "; ".join(parts), "ungrounded_claims": claims + issues}


_JUDGE_PROMPT = """You check an assistant's reply about Singapore HDB housing against the tool data it was written from.
Flag only claims in the REPLY that the DATA contradicts or does not support: invented prices, distances, lease years,
block numbers, rules or grants, a statement that a block meets a limit the data does not show, or advice that promises
profit. Do not flag wording, general knowledge that is not a specific figure, or things the user said.
Answer with JSON only: {{"passed": true|false, "ungrounded_claims": ["..."]}}

DATA:
{evidence}

USER MESSAGE:
{user}

REPLY:
{reply}"""


def llm_judge(draft: str, evidence: str, user_text: str) -> Optional[Dict[str, Any]]:
    """The model's verdict, or None when no model is configured or its answer cannot be read."""
    if os.getenv("HDB_AUDITOR_LLM", "on").lower() in ("off", "0", "false"):
        return None
    from agents.llm import get_llm
    llm = get_llm()
    if llm is None:
        return None
    raw = sv.strip_think(getattr(llm.invoke(_JUDGE_PROMPT.format(evidence=evidence[:6000], user=user_text, reply=draft)),
                                 "content", "") or "")
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    data = json.loads(m.group(0))
    claims = [str(c) for c in data.get("ungrounded_claims") or []]
    passed = bool(data.get("passed", not claims)) and not claims
    return {"passed": passed, "ungrounded_claims": claims,
            "critique": "" if passed else "the data does not support: " + "; ".join(claims)}


def _model_text(state: Dict[str, Any]) -> str:
    """What the auditor judges: the model's own sentences. `synthesize` records them in `model_opening` ("" when code
    wrote everything or the model wrote nothing: pass); code's answer sentence and the facts section are never judged
    (docs/INTERFACES.md section 3). A state without that key (Variants A and B, hand-built states) falls back to the
    draft before the code-written facts."""
    if "model_opening" in state:
        return (state.get("model_opening") or "").strip()
    draft = state.get("draft_response") or ""
    facts = sv.template_response(state) if sv.writes_facts_in_code(state) else ""
    return draft.split(facts, 1)[0].strip() if facts and facts in draft else draft


def _judge_data(state: Dict[str, Any]) -> str:
    """The data the LLM judge reads: the facts section the user sees (short, already checked against the tool results)
    when code writes it, else the evidence JSON."""
    facts = sv.template_response(state) if sv.writes_facts_in_code(state) else ""
    return facts or evidence_text(state)


def runtime_auditor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prev = state.get("auditor_verdict") or {}
    retries = int(prev.get("retry_count", 0))
    try:
        opening = _model_text(state)
        if not opening:   # the model wrote nothing (it failed or was off)
            return {"auditor_verdict": AuditorVerdict(passed=True, critique="", retry_count=retries).model_dump()}
        evidence, user = evidence_text(state), sv.last_user_query(state)
        verdict = audit_text(opening, evidence, user)
        errors: List[str] = []
        if verdict["passed"]:
            try:
                judged = llm_judge(opening, _judge_data(state), user)
            except Exception as e:  # noqa: BLE001 — a judge fault leaves the rule verdict (pass)
                judged, errors = None, [f"runtime_auditor: llm_judge {type(e).__name__}"]
            if judged is not None:
                verdict = judged
        out = AuditorVerdict(passed=verdict["passed"], critique=verdict["critique"],
                             retry_count=retries if verdict["passed"] else retries + 1,
                             ungrounded_claims=verdict["ungrounded_claims"]).model_dump()
        return {"auditor_verdict": out, **({"errors": errors} if errors else {})}
    except Exception as e:  # noqa: BLE001 — an auditor fault must not stop the answer
        return {"auditor_verdict": AuditorVerdict(passed=True, critique="auditor error (pass-through)",
                                                  retry_count=retries).model_dump(),
                "errors": [f"runtime_auditor: {type(e).__name__}"]}
