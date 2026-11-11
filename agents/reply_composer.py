"""The app's conversational reply (HDB_ANSWER_STYLE=conversational, the app on Claude). Member 4.

Code still works out every fact (the tools, the join, the calculators) and writes them as today's checked facts section
(supervisor.template_response). The model then writes the reply a user reads, in plain conversational prose, from those
facts; verify_reply checks it line by line before anyone sees it, and the checked facts are kept under it as Details.
A reply that fails is written once more with the list of problems; a second failure gives today's answer (code's
opening and the facts), so the worst case is what the app showed before. The cluster model and Experiment 1 never take
this path (supervisor.answer_style).
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage

DETAILS_SEP = "\n\n---\n\n**Details (checked figures)**\n\n"
REPLY_WORDS = 180
MAX_BLOCKS_NAMED = 5

COMPOSER_SYSTEM = (
    "You are HDB-Compass, a friendly and knowledgeable assistant for Singapore HDB resale flats, chatting with a home "
    "buyer. Write the reply they read: natural, warm and to the point, like an expert friend in a chat, not a report. "
    "Code has already worked out every fact you may use (searches, prices, distances, grants, budgets, rules) and you "
    "are given them as CHECKED FACTS. Use only figures, block names and rules found there or in what the user said, "
    "written as they are written there. Never work out, estimate or round a figure yourself. If the facts do not answer "
    "something, say so plainly instead of guessing.")

STYLE_RULES = (
    "How to write it:\n"
    "- First sentence: the direct answer to the latest message (yes/no, the figure, the best option, or what is missing).\n"
    "- Then a short, natural explanation in prose. About {words} words at most.\n"
    "- A small markdown table only when comparing 2 to 5 blocks or options side by side; otherwise no tables.\n"
    "- Name at most {blocks} blocks. When the facts list more, say the full list is in the details below and on the map.\n"
    "- No headings, no bold section titles, no notes about the HFE letter, taxes or fees (the details carry them).\n"
    "- Earlier turns are context: answer the latest message; do not repeat an earlier answer.\n"
    "- HDB rules: state only what the HDB rules lines say, with their conditions as written. Add no rule, condition, "
    "age, amount or exception from your own knowledge.\n"
    "- If the facts do not answer the question, say that first (do not open with a yes or no you cannot support).\n"
    "- Say a limit or check was not done only when the facts say so, and offer only a next step the facts name.\n"
    "- Do not list missing household facts yourself; the question below asks for them.\n"
    "- Counts: \"the N cheapest of M matching blocks\" means M blocks match; never say N match.\n"
    "- {ending}")


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
    return str(content or "")


def conversation_brief(state: Dict[str, Any], turns: int = 3) -> str:
    """The last few turns before this one: each user message and the opening of each earlier reply (not its details)."""
    from agents import supervisor as sv
    msgs = [m for m in state.get("messages") or [] if isinstance(m, dict)]
    earlier = msgs[:-1] if msgs and msgs[-1].get("role") == "user" else msgs
    lines: List[str] = []
    for m in earlier[-2 * turns:]:
        content = str(m.get("content") or "")
        if m.get("role") == "user":
            lines.append(f"User: {content[:400]}")
        else:
            head = content.split(DETAILS_SEP)[0].split("\n\n")[0]
            lines.append("Assistant: " + " ".join(sv._split_opening(head)[:2])[:400])
    return "\n".join(lines)


def stream_writer() -> Optional[Callable[[Dict[str, Any]], None]]:
    """LangGraph's custom-stream writer inside a graph run, or None (a unit test, the harness's invoke, or the auditor's
    model judge on: a sentence must not reach the page before that judge has seen the whole reply, PLAT-32). With the
    judge off, as in the demo, each streamed sentence has passed the same rule checks as the whole reply."""
    if os.getenv("HDB_AUDITOR_LLM", "on").strip().lower() not in ("off", "0", "false", "no"):
        return None
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except Exception:  # noqa: BLE001 — not inside a graph run
        return None


def _stream_checked(llm: Any, msgs: List[Any], check: Callable[[str], List[str]],
                    writer: Callable[[Dict[str, Any]], None]) -> str:
    """The reply, streamed to the page one checked sentence (or whole table) at a time; the first sentence that fails
    stops the streaming (the rest is still read, for the full check)."""
    full, pending, ok = "", "", True

    def flush(unit: str) -> None:
        nonlocal ok
        if ok and unit.strip():
            if check(unit):
                ok = False
            else:
                writer({"reply_delta": unit})

    for chunk in llm.stream(msgs):
        piece = _text(getattr(chunk, "content", chunk))
        full += piece
        pending += piece
        while True:
            lines = pending.split("\n")
            in_table = lines[0].lstrip().startswith("|")
            if in_table:   # a table goes out whole, once a line after it is not a row
                end = next((i for i, line in enumerate(lines[1:], 1) if line.strip() and not line.lstrip().startswith("|")), None)
                if end is None:
                    break
                unit, pending = "\n".join(lines[:end]) + "\n", "\n".join(lines[end:])
                flush(unit)
                continue
            m = re.search(r"(?<!\bBlk)(?<!\bNo)(?<!\bSt)(?<!\bAve)[.!?](?=\s)|\n", pending)
            if not m:
                break
            unit, pending = pending[:m.end()], pending[m.end():]
            flush(unit)
    flush(pending)
    return full


def compose_reply(llm: Any, state: Dict[str, Any], facts: str, evidence: Dict[str, Any],
                  clar: Optional[Dict[str, Any]], policy: Any, code_answer: str,
                  problems: Optional[List[str]] = None,
                  writer: Optional[Callable[[Dict[str, Any]], None]] = None) -> str:
    """One model call: the reply, from the checked facts. problems: what was wrong with the last attempt. writer: stream
    each checked sentence to the page as it is written."""
    from agents import research_intents as ri
    from agents import supervisor as sv
    question = (clar or {}).get("question")
    ending = (f"End with exactly this question to the user, word for word, and ask nothing else: \"{question}\""
              if question else "You may end with one short offer of a useful next step, as a question.")
    rules = STYLE_RULES.format(words=REPLY_WORDS, blocks=MAX_BLOCKS_NAMED, ending=ending)
    brief = conversation_brief(state)
    parts = []
    if brief:
        parts.append(f"Conversation so far:\n{brief}")
    parts.append(f"Latest message (answer this one):\n{sv.user_request(state)}")
    focus = (state.get("user_constraints") or {}).get("turn_focus")
    if focus == "other":
        parts.append("This message asks something other than the earlier flat search; the search is context only.")
    parts.append("CHECKED FACTS (written by code from the tools; shown to the user under your reply as Details, so do "
                 "not copy them all):\n---\n" + facts + "\n---")
    if policy:
        parts.append("HDB rules found for this question:\n---\n" + "\n".join(sv.policy_fact_lines(policy, sv.user_request(state)))
                     + "\n---")
    if code_answer:
        parts.append(f"Code's own answer, which your reply must agree with: \"{code_answer}\"")
    parts.append("Tool results (for context; every figure you state must still be in the CHECKED FACTS):\n"
                 + ri.compact_tool_result(evidence, 8000))
    parts.append(rules)
    if problems:
        parts.append("Your last reply was not shown because: " + "; ".join(problems[:8])
                     + ". Write it again without those problems.")
    msgs = [sv.sys_msg(COMPOSER_SYSTEM), HumanMessage(content="\n\n".join(parts))]
    if writer is not None and hasattr(llm, "stream"):
        check = lambda unit: verify_reply(unit, state, facts, clar, policy, shape=False)  # noqa: E731
        return sv.strip_think(_stream_checked(llm, msgs, check, writer)).strip()
    return sv.strip_think(_text(llm.invoke(msgs).content)).strip()


# ---------------------------------------------------------------------------
# The checks on a composed reply: deterministic, line by line
# ---------------------------------------------------------------------------
_PERCENT = re.compile(r"([+-]?\d+(?:\.\d+)?)\s?(?:%|per ?cent)", re.I)
_COUNT = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+)\s+(?:matching\s+|candidate\s+|recent\s+)?"
                    r"(?:blocks?|sales|transactions?|towns?|matches|options|flats?\b(?!\s*type))", re.I)
_MINUTES = re.compile(r"\b(\d+)\s*(?:-|\s)?min(?:ute)?s?\b", re.I)
_BLK = re.compile(r"\b(?:Blk|Block)\.?\s*(\d{1,4}[A-Z]?)\b", re.I)
_MEETS_ALL = re.compile(r"\bmeets?\s+(?:every|all)\s+(?:of\s+)?(?:your\s+|the\s+)?(?:limits?|constraints?|criteria|requirements?)\b", re.I)
_BUDGET_CLAIM = re.compile(r"\b(?:you can afford|your (?:maximum |max(?:imum)? )?(?:budget|ceiling|purchase price) (?:is|of|comes to)|"
                           r"(?:budget|ceiling) of)\b[^.]*\$\s?\d", re.I)


def _units(reply: str) -> List[str]:
    """Each sentence of prose and each table row."""
    from agents import supervisor as sv
    out: List[str] = []
    for line in reply.splitlines():
        s = line.strip()
        if not s or re.fullmatch(r"\|?[\s:|-]+\|?", s):
            continue
        if s.startswith("|"):
            out.append(s)
        else:
            out += sv._split_opening(re.sub(r"^[-*]\s+", "", s))
    return out


def _numbers(text: str) -> List[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


_RANKS = re.compile(r"\bthan\b|\b(?:cheapest|lowest|highest|nearest|closest|furthest|farthest|longest|shortest|largest|"
                    r"smallest|biggest|newest|oldest|priciest|best|most|least)\b", re.I)


def _comparison_wrong(unit: str, named: List[str], listed: List[Dict[str, Any]]) -> bool:
    """A block named as the cheapest, nearest or with the most lease (or as cheaper than another) when the listed
    blocks' figures say otherwise. The opening's stricter rule (any comparison with no named block is dropped) is for one
    or two sentences; in a whole reply a sentence naming no block, or a hedge ("one of the closest"), states nothing a
    check could refute, so it stands."""
    from agents import supervisor as sv
    if not named or re.search(r"\bone of the\b|\bamong the\b", unit, re.I) or not _RANKS.search(unit):
        return False
    return bool(sv._COMPARISON.search(unit)) and not sv.comparisons_hold(unit, listed)


_YEARS_OR_METRES = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-\s*)?(years?|yrs?|m\b|metres?|meters?)", re.I)
_NOT_CHECKED_CLAIM = re.compile(r"\b(?:has not|hasn't|have not|haven't|was not|wasn't|were not|weren't|not yet|couldn't|could not)"
                                r"\s+(?:been\s+|be\s+)?(?:checked|timed|run|worked out|measured)\b", re.I)
_MATCH_TOTAL = re.compile(r"cheapest of (\d[\d,]*) matching", re.I)
_MATCH_CLAIM = re.compile(r"\b(\d[\d,]*)\s+(?:[\w-]+\s+){0,3}?(?:blocks?|flats?)\s+(?:that\s+|which\s+)?(?:fit|match|meet)", re.I)


def _figures_on(lines: str, kind: str) -> List[float]:
    """The lease years or metres the facts lines give, as written there ("67 yrs lease", "mrt 241m")."""
    rx = r"(\d+(?:\.\d+)?)\s*(?:yrs?|years?)" if kind == "years" else r"(\d+(?:\.\d+)?)\s*m\b"
    return [float(x) for x in re.findall(rx, lines, re.I)]


def _plain_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9$%]+", (text or "").lower()))


def drop_claims(reply: str, claims: List[str]) -> str:
    """The reply without the sentences the auditor's judge named as ungrounded (each claim is a sentence or part of
    one, quoted from the reply). Table rows and paragraph breaks are kept; "" when nothing is left."""
    marks = [_plain_words(c)[:60] for c in claims if len(_plain_words(c)) >= 12]
    if not marks:
        return ""
    from agents import supervisor as sv
    paras = []
    for para in reply.split("\n\n"):
        if para.strip().startswith("|"):
            paras.append(para)
            continue
        kept = [u for u in sv._split_opening(para.strip())
                if not any(m in _plain_words(u) or _plain_words(u)[:60] in m for m in marks)]
        if kept:
            paras.append(" ".join(kept))
    return "\n\n".join(paras).strip()


def verify_reply(reply: str, state: Dict[str, Any], facts: str, clar: Optional[Dict[str, Any]] = None,
                 policy: Any = None, shape: bool = True) -> List[str]:
    """What is wrong with a composed reply; empty when it may be shown. Every figure must be in the checked facts, the
    tool results, the policy lines or the user's own words (Member 5's rule check, guardrails.audit_text, plus counts,
    percentages and minutes); every block named must be in the facts or the last list; a sentence about one block uses
    that block's own figures; the opening checks (comparisons, PR-couple grants, the ceiling, distances, a table cut
    short) hold; and its shape is a chat reply (no headings, only code's question when one is open)."""
    from agents import guardrails as g
    from agents import supervisor as sv
    problems: List[str] = []
    if not reply.strip():
        return ["the reply was empty"]
    users = " ".join(str(m.get("content", "")) for m in state.get("messages") or []
                     if isinstance(m, dict) and m.get("role") == "user")
    # earlier answers were checked when they were shown: their figures and blocks may be repeated ("as I said, raising the
    # budget to $715,000 unlocks 2 blocks" under a follow-up whose own facts are a different search)
    shown_before = " ".join(str(m.get("content", "")) for m in (state.get("messages") or [])[:-1]
                            if isinstance(m, dict) and m.get("role") == "assistant")
    users = (users + " " + shown_before).strip()
    policy_text = "\n".join(sv.policy_fact_lines(policy, sv.user_request(state))) if policy else ""
    # the tool results without the raw policy passages: a rule figure must be in the policy lines the answer shows
    # (a $10,000 top-up or an age-35 limit from a passage the user never sees had passed, 6 Nov grading)
    pool = "\n".join([facts, g.evidence_text({k: v for k, v in state.items() if k != "rag_policy_passages"}), policy_text])
    question = (clar or {}).get("question") or ""
    body = reply.replace(question, " ") if question else reply
    plain = body.replace("**", "")

    # 1. figures: Member 5's rule check (money, metres, years, block numbers), then counts, percentages and minutes
    verdict = g.audit_text(plain, pool, users)
    problems += [f"'{c}' is not in the checked facts" for c in verdict.get("ungrounded_claims") or []]
    pool_numbers = set(_numbers(pool + " " + users))
    pool_percents = [float(x) for x in _PERCENT.findall(pool + " " + users)]
    for m in _PERCENT.finditer(plain):
        if not any(abs(abs(float(m.group(1))) - abs(p)) <= 0.05 for p in pool_percents):
            problems.append(f"'{m.group(0).strip()}' is not in the checked facts")
    for rx in (_COUNT, _MINUTES):
        for m in rx.finditer(plain):
            if float(m.group(1).replace(",", "")) not in pool_numbers:
                problems.append(f"'{m.group(0).strip()}' is not in the checked facts")
    if policy:
        allowed = sv.policy_figures(policy, sv.user_request(state))
        for f in sv._figures(plain):
            if isinstance(f[0], tuple) and f not in allowed:
                problems.append("a range of figures the HDB rules do not give")
            elif not isinstance(f[0], tuple) and f[0] == "%" and f not in allowed and any(
                    isinstance(x[0], tuple) and f in x for x in allowed):
                problems.append(f"'{f[1]:g}%' is one end of a range the HDB rules give whole")

    # 2. blocks: only those code listed, shown earlier, or the user named
    known = {b.upper() for b in _BLK.findall(facts + " " + users + " " + " ".join(state.get("prior_blocks") or []))}
    for b in dict.fromkeys(x.upper() for x in _BLK.findall(body)):
        if b not in known:
            problems.append(f"Blk {b} is not in the checked facts")

    # 3. a sentence about one block uses that block's own figures; 4. the opening checks
    listed = (state.get("candidate_blocks_intersection") or [])[:sv.LISTED_BLOCKS]
    open_money = (clar or {}).get("kind", "financial") == "financial" and bool(question)
    for unit in _units(body):
        named = list(dict.fromkeys(x.upper() for x in _BLK.findall(unit)))
        if len(named) == 1 and "$" in unit:
            lines = "\n".join(line for line in facts.splitlines()
                              if re.search(rf"\b(?:Blk|Block)\.?\s*{re.escape(named[0])}\b", line, re.I))
            if lines:
                wrong = [c for c in g.ungrounded_claims(unit.replace("**", ""), lines, users) if c.startswith(("$", "S$"))]
                problems += [f"'{c}' is not Blk {named[0]}'s figure" for c in wrong]
        if len(named) == 1:
            # a block's lease and distance exactly as its facts line writes them (the rule check allows a year of
            # rounding against the raw data, so "66 years" passed for a block the facts show as "67 yrs lease")
            lines = "\n".join(line for line in facts.splitlines()
                              if re.search(rf"\b(?:Blk|Block)\.?\s*{re.escape(named[0])}\b", line, re.I))
            for m in _YEARS_OR_METRES.finditer(unit.replace("**", "")):
                kind = "years" if m.group(2).lower().startswith(("y", "yr")) else "metres"
                shown = _figures_on(lines, kind)
                if lines and shown and float(m.group(1)) not in shown:
                    problems.append(f"'{m.group(0).strip()}' is not Blk {named[0]}'s figure")
        if listed and _comparison_wrong(unit, named, listed):
            problems.append(f"a comparison the facts do not bear out: \"{unit[:80]}\"")
        if sv.pr_couple_grant_claim(unit, state):
            problems.append("two permanent residents get no CPF housing grant")
        if sv.ceiling_grants_claim_wrong(unit, state):
            problems.append("what the budget ceiling includes is stated wrongly")
        if re.search(r"\d+(?:\.\d+)?\s*(?:m|km|metres?|meters?|min(?:ute)?s?)\b", unit, re.I) and sv.unsupported_proximity(unit, facts):
            problems.append(f"a distance the facts do not give: \"{unit[:80]}\"")
        if sv.ranks_part_table(unit, facts):
            problems.append("a ranking of the whole set from a table that shows only part of it")
        if open_money and _BUDGET_CLAIM.search(unit):
            problems.append("a budget figure while household facts are still missing")
    total = _MATCH_TOTAL.search(facts)
    if total:
        for m in _MATCH_CLAIM.finditer(plain):
            if m.group(1).replace(",", "") != total.group(1).replace(",", ""):
                problems.append(f"{total.group(1)} blocks match, not {m.group(1)} (the facts list only the cheapest)")
    if not sv.not_checked_lines(state) and not sv.unavailable_tools(state) and _NOT_CHECKED_CLAIM.search(plain):
        problems.append("says a check was not done although every check ran")
    if sv.not_checked_lines(state) and _MEETS_ALL.search(plain):
        problems.append("says every limit is met although one was not checked")

    # 5. shape: a chat reply, not a report (not checked on a single streamed sentence)
    if not shape:
        return list(dict.fromkeys(problems))
    if re.search(r"^\s*#", reply, re.M):
        problems.append("headings")
    asks = [u for u in _units(body) if u.rstrip().endswith("?") and not u.startswith("|")]
    if question and (asks or question not in reply):
        problems.append("ask only the question given, word for word, at the end")
    elif not question and len(asks) > 1:
        problems.append("more than one question")
    rows = [line for line in reply.splitlines() if line.strip().startswith("|") and not re.fullmatch(r"\|?[\s:|-]+\|?", line.strip())]
    if rows and len(rows) < 3:   # a header and at least two rows
        problems.append("a table with fewer than two rows")
    return list(dict.fromkeys(problems))


def compose(llm: Any, state: Dict[str, Any], facts: str, evidence: Dict[str, Any], clar: Optional[Dict[str, Any]],
            policy: Any, code_answer: str) -> Tuple[Optional[str], List[str]]:
    """(reply, notes): a checked reply after at most two model calls, or None and why (the caller then shows today's
    answer). The notes avoid the words the app reads as a model failure."""
    problems: Optional[List[str]] = None
    writer = stream_writer()
    for attempt in (1, 2):
        try:
            reply = compose_reply(llm, state, facts, evidence, clar, policy, code_answer, problems, writer)
        except Exception as e:  # noqa: BLE001 — the caller shows today's answer
            if writer:
                writer({"reply_reset": True})
            return None, [f"synthesize: composed reply not written ({type(e).__name__}); checked facts used"]
        problems = verify_reply(reply, state, facts, clar, policy)
        if not problems:
            note = "synthesize: reply written from the checked facts" + (" on the second try" if attempt == 2 else "")
            return reply, [note]
        if writer:
            writer({"reply_reset": True})   # what streamed is withdrawn before the next try or today's answer
    return None, [f"synthesize: composed reply had unchecked items ({'; '.join(problems[:4])}); checked facts used"]
