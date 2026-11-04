"""Layer 0: remove personal and financial identifiers from a message before anything else reads it.

    clean, kinds = redact("My NRIC is S1234567D, call 9123 4567")
    # clean == "My NRIC is [NRIC], call [PHONE]", kinds == ["NRIC", "PHONE"]

It is applied where the text enters the system (the WebSocket handler in backend/main.py and `supervisor.initial_state`,
which every variant, the evaluation and the command line go through), so the graph, the model, the remembered session,
the chat log and any LangSmith trace only ever see the placeholder. Pure code, no model, no state; running it twice
changes nothing. Postal codes, money and ordinary figures are left alone (a benchmark test locks that).

Regexes cannot find names or home addresses; Presidio (proposal section 5, risk 5) would add that.
"""
import re
from typing import List, Tuple

PLACEHOLDERS = {
    "NRIC": "[NRIC]", "PHONE": "[PHONE]", "CARD": "[CARD]", "BANK_ACCOUNT": "[BANK_ACCOUNT]",
    "EMAIL": "[EMAIL]", "SECRET": "[SECRET]",
}
PLACEHOLDER_RE = re.compile("|".join(re.escape(p) for p in PLACEHOLDERS.values()))

# an NRIC or FIN as people type it: "S1234567D", "S 1234567 D", "S1234567-D", "S-1234567-D"
_NRIC = re.compile(r"(?<![A-Za-z\d])[STFGM][\s-]?\d{7}[\s-]?[A-Z](?![A-Za-z\d])", re.I)
_PHONE = re.compile(r"(?<![\d$,.])(?:\+?65[\s-]?)?[689]\d{3}[\s-]?\d{4}(?!\d|,\d)")
# "8000-9000" or "$8,000 to $9,000 income" is a range of money, not a phone number: both ends are round figures
_ROUND_RANGE = re.compile(r"^\+?(?:65[\s-]?)?\d{2}00\s*-\s*\d{2}00$|^\+?(?:65[\s-]?)?\d{1}000\s*-\s*\d{1}000$")
_MONEY_BEFORE = re.compile(r"(?:\$|\b(?:income|salary|earn\w*|budget|price|pay|paid|between|from|range|per month|monthly)\b)[^.?!\n]{0,25}$", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_CARD = re.compile(r"(?<![\d$,.])(?:\d[ -]?){12,18}\d(?!\d|,\d)")
# a number after a banking word, or a Singapore bank's own account layout (DBS 3-5-1, OCBC 3-6-3, UOB 3-3-3-1)
_BANK_KEYWORD = re.compile(
    r"(\b(?:bank\s+)?(?:account|acct|a/c|iban|swift|paynow|savings|current)\b(?:\s+(?:no\.?|number|num|#))?[^.\n\d]{0,25}?)"
    r"(\d[\d -]{6,20}\d)", re.I)
_BANK_LAYOUT = re.compile(r"(?<![\d-])(?:\d{3}-\d{5}-\d|\d{3}-\d{6}-\d{3}|\d{3}-\d{3}-\d{3}-\d)(?![\d-])")
_SECRET = re.compile(r"(\b(?:password|passcode|passphrase|otp|cvv|cvc|pin)\b\s*(?:is|[:=])\s*)\S+", re.I)


def _luhn(digits: str) -> bool:
    total, flip = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if flip:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        flip = not flip
    return total % 10 == 0


def redact(text: str) -> Tuple[str, List[str]]:
    """(text with identifiers replaced by placeholders, kinds found in order of first appearance)."""
    if not text:
        return text or "", []
    found: List[str] = []

    def note(kind: str) -> str:
        if kind not in found:
            found.append(kind)
        return PLACEHOLDERS[kind]

    def card(m: "re.Match[str]") -> str:
        digits = re.sub(r"[ -]", "", m.group(0))
        return note("CARD") if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)

    out = _SECRET.sub(lambda m: m.group(1) + note("SECRET"), text)
    out = _NRIC.sub(lambda m: note("NRIC"), out)
    out = _EMAIL.sub(lambda m: note("EMAIL"), out)
    out = _CARD.sub(card, out)
    out = _BANK_KEYWORD.sub(lambda m: m.group(1) + note("BANK_ACCOUNT"), out)
    out = _BANK_LAYOUT.sub(lambda m: note("BANK_ACCOUNT"), out)
    def phone(m: "re.Match[str]") -> str:
        if _ROUND_RANGE.match(m.group(0)) or (re.search(r"\d\s*-\s*\d", m.group(0)) and _MONEY_BEFORE.search(m.string[:m.start()])):
            return m.group(0)
        return note("PHONE")

    out = _PHONE.sub(phone, out)
    return out, found


def has_placeholder(text: str) -> bool:
    return bool(PLACEHOLDER_RE.search(text or ""))
