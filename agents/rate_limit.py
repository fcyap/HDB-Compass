"""Client-side rate limiting for free-tier LLM APIs (RPM / TPM / RPD).

One process-wide limiter per profile name, shared by every agent and thread.
State is in memory only: the daily counter resets on restart and is not shared
between processes (run one backend worker, or set conservative limits).

Wired in through a LangChain callback (``RateLimitCallback``) attached by
``agents.llm.get_llm``; callbacks survive ``bind_tools`` and
``with_structured_output`` because they live on the underlying model.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Deque, Dict, Optional, Tuple

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover
    BaseCallbackHandler = object  # type: ignore[assignment,misc]

WINDOW_S = 60.0
DEFAULT_MAX_WAIT_S = 60.0


class RateLimitError(RuntimeError):
    """Raised when a configured client-side limit cannot be satisfied in time."""


@dataclass(frozen=True)
class Limits:
    requests_per_minute: int = 0  # 0 = unlimited
    tokens_per_minute: int = 0
    requests_per_day: int = 0

    @property
    def unlimited(self) -> bool:
        return not (self.requests_per_minute or self.tokens_per_minute or self.requests_per_day)


def _as_int(value: Any) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return max(0, int(float(str(value).strip())))
    except ValueError:
        return None


def resolve_limits(profile_limits: Optional[Dict[str, Any]], env=None) -> Limits:
    """YAML ``rate_limits`` overridden by LLM_RPM / LLM_TPM / LLM_RPD (0 = unlimited)."""
    env = os.environ if env is None else env
    pl = profile_limits or {}
    out = {}
    for field, key, var in (
        ("requests_per_minute", "requests_per_minute", "LLM_RPM"),
        ("tokens_per_minute", "tokens_per_minute", "LLM_TPM"),
        ("requests_per_day", "requests_per_day", "LLM_RPD"),
    ):
        override = _as_int(env.get(var))
        out[field] = override if override is not None else (_as_int(pl.get(key)) or 0)
    return Limits(**out)


def max_wait_s(env=None) -> float:
    env = os.environ if env is None else env
    try:
        return float(env.get("LLM_RATE_LIMIT_MAX_WAIT_S") or DEFAULT_MAX_WAIT_S)
    except ValueError:
        return DEFAULT_MAX_WAIT_S


def estimate_tokens(text_chars: int, max_output_tokens: int = 0) -> int:
    """Cheap pre-call estimate: chars/4 for the prompt plus the output ceiling."""
    return text_chars // 4 + 1 + max(0, int(max_output_tokens or 0))


class RateLimiter:
    """Thread-safe sliding-window limiter with an injectable clock and sleep."""

    def __init__(
        self,
        name: str,
        limits: Limits,
        max_wait: float = DEFAULT_MAX_WAIT_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.name, self.limits, self.max_wait = name, limits, max_wait
        self._clock, self._sleep, self._today = clock, sleep, today
        self._lock = threading.Lock()
        self._requests: Deque[float] = deque()
        self._tokens: Deque[list] = deque()  # [timestamp, tokens]
        self._day = today()
        self._day_count = 0

    def _prune(self, now: float) -> None:
        cutoff = now - WINDOW_S
        while self._requests and self._requests[0] <= cutoff:
            self._requests.popleft()
        while self._tokens and self._tokens[0][0] <= cutoff:
            self._tokens.popleft()
        if self._today() != self._day:
            self._day, self._day_count = self._today(), 0

    def _wait_needed(self, now: float, tokens: int) -> Tuple[float, str]:
        lim = self.limits
        wait, which = 0.0, ""
        if lim.requests_per_minute and len(self._requests) >= lim.requests_per_minute:
            w = self._requests[-lim.requests_per_minute] + WINDOW_S - now
            if w > wait:
                wait, which = w, f"requests per minute ({lim.requests_per_minute})"
        if lim.tokens_per_minute and self._tokens:
            used = sum(t for _, t in self._tokens)
            # A single request larger than the whole budget is let through on an empty window.
            if used + tokens > lim.tokens_per_minute:
                need, w = used + tokens - lim.tokens_per_minute, 0.0
                for ts, t in self._tokens:
                    need -= t
                    w = ts + WINDOW_S - now
                    if need <= 0:
                        break
                if w > wait:
                    wait, which = w, f"tokens per minute ({lim.tokens_per_minute})"
        return max(wait, 0.0), which

    def acquire(self, tokens: int = 0) -> Tuple[float, int]:
        """Block until a request fits; returns (slot timestamp, tokens reserved)."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._prune(now)
                if self.limits.requests_per_day and self._day_count >= self.limits.requests_per_day:
                    raise RateLimitError(
                        f"Daily request limit reached for the {self.name} profile "
                        f"({self.limits.requests_per_day} requests per day, LLM_RPD). "
                        "Try again tomorrow, raise the limit, or switch LLM_PROFILE."
                    )
                wait, which = self._wait_needed(now, tokens)
                if wait <= 0:
                    self._requests.append(now)
                    self._tokens.append([now, tokens])
                    self._day_count += 1
                    return now, tokens
            if waited + wait > self.max_wait:
                raise RateLimitError(
                    f"Rate limit for the {self.name} profile: {which} would need a wait of "
                    f"{wait:.0f}s (max {self.max_wait:.0f}s, LLM_RATE_LIMIT_MAX_WAIT_S). "
                    "Retry shortly or raise the limit."
                )
            self._sleep(wait)
            waited += wait

    def record_usage(self, slot: Tuple[float, int], actual_tokens: int) -> None:
        """Replace the pre-call estimate with the provider-reported total."""
        ts, reserved = slot
        with self._lock:
            for entry in self._tokens:
                if entry[0] == ts and entry[1] == reserved:
                    entry[1] = max(0, int(actual_tokens))
                    return

    def status(self) -> Dict[str, Any]:
        with self._lock:
            self._prune(self._clock())
            lim = self.limits
            used_tokens = sum(t for _, t in self._tokens)
            return {
                "requests_per_minute": lim.requests_per_minute or None,
                "requests_last_minute": len(self._requests),
                "tokens_per_minute": lim.tokens_per_minute or None,
                "tokens_last_minute": used_tokens,
                "requests_per_day": lim.requests_per_day or None,
                "requests_today": self._day_count,
                "requests_left_today": (
                    max(0, lim.requests_per_day - self._day_count) if lim.requests_per_day else None
                ),
            }


_LIMITERS: Dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_limiter(name: str, limits: Limits, max_wait: float) -> Optional[RateLimiter]:
    """Process-wide limiter for a profile; None when the profile is unlimited."""
    if limits.unlimited:
        return None
    with _LIMITERS_LOCK:
        cur = _LIMITERS.get(name)
        if cur is None or cur.limits != limits:
            new = RateLimiter(name, limits, max_wait)
            if cur is not None:  # limits changed: keep history
                new._requests, new._tokens = cur._requests, cur._tokens
                new._day, new._day_count = cur._day, cur._day_count
            _LIMITERS[name] = cur = new
        cur.max_wait = max_wait
        return cur


def limiter_status(name: str) -> Optional[Dict[str, Any]]:
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(name)
    return lim.status() if lim else None


def reset_limiters() -> None:
    with _LIMITERS_LOCK:
        _LIMITERS.clear()


def _usage_total(response: Any) -> Optional[int]:
    try:
        for gen_list in response.generations:
            for gen in gen_list:
                um = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if um and um.get("total_tokens"):
                    return int(um["total_tokens"])
        tu = (response.llm_output or {}).get("token_usage") or {}
        if tu.get("total_tokens"):
            return int(tu["total_tokens"])
    except Exception:
        pass
    return None


class RateLimitCallback(BaseCallbackHandler):  # type: ignore[misc,valid-type]
    """Acquire a slot before each chat-model call; correct token usage after."""

    raise_error = True  # let RateLimitError abort the call instead of being logged
    run_inline = True

    def __init__(self, limiter: RateLimiter, max_output_tokens: int = 0) -> None:
        self.limiter = limiter
        self.max_output_tokens = max_output_tokens
        self._slots: Dict[Any, Tuple[float, int]] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        chars = sum(len(str(getattr(m, "content", ""))) for batch in messages for m in batch)
        slot = self.limiter.acquire(estimate_tokens(chars, self.max_output_tokens))
        with self._lock:
            self._slots[run_id] = slot

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        if run_id in self._slots:
            return
        slot = self.limiter.acquire(estimate_tokens(sum(len(p) for p in prompts), self.max_output_tokens))
        with self._lock:
            self._slots[run_id] = slot

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self._lock:
            slot = self._slots.pop(run_id, None)
        total = _usage_total(response)
        if slot is not None and total is not None:
            self.limiter.record_usage(slot, total)

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self._lock:
            self._slots.pop(run_id, None)
