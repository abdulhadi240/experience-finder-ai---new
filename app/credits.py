# app/credits.py
"""Client for the internal credits API — the enforcement layer.

The frontend also gates on its displayed balance, but that check is a UX
nicety only: it can be bypassed by direct API calls, and it goes stale during
rapid-fire messages. This module is the actual security boundary. Every
billable agent invocation must be preceded by a successful `reserve`.

Internal-only endpoint — never call this from a browser, and never surface the
shared secret in a response or a log line.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Literal, Optional

import httpx

from .config import settings

_log = logging.getLogger("app.credits")

# Dedicated client with a short timeout — this call is in front of TTFB for
# every chat message, so it must never inherit the 30s RAG budget.
_credits_client = httpx.AsyncClient(timeout=settings.credits_timeout_seconds)

UsageType = Literal["TRIP_PLANNING", "EXPLORE_TRAVEL", "CUSTOMER_SERVICE"]

# Placeholder copy — final wording/UX is being confirmed separately.
OUT_OF_CREDITS_MESSAGE = (
    "You've used your available AI credits. Upgrade to keep planning with me — "
    "or check back later once your credits refresh."
)
CREDITS_UNAVAILABLE_MESSAGE = (
    "I couldn't verify your available credits just now. Please try again in a moment."
)


def new_request_id() -> str:
    """Mint an idempotency key for ONE billable action.

    Call this once per chat message and hold the value for that message's whole
    lifetime — the same id must be reused on every retry of the reserve call and
    again on any later refund. Never call this from inside a retry loop.
    """
    return str(uuid.uuid4())


def usage_type_for(param: str, plan: bool = False) -> UsageType:
    """Map an incoming /chat request to the credits usageType.

    Decided from the request rather than from the validation agent's verdict,
    because the gate runs before any LLM call — see the reserve call site in
    helpers._main_stream for why that ordering matters.
    """
    if plan or param == "plan":
        return "TRIP_PLANNING"
    return "EXPLORE_TRAVEL"


@dataclass
class ReserveResult:
    """Outcome of a reserve attempt.

    `transport_failed` separates "the credits service said no" from "we never
    got an answer" — the two look identical in `allowed` but need different
    logging, alerting, and user-facing copy.
    """
    allowed: bool
    remaining: Optional[int] = None
    from_cache: bool = False
    reason_code: str = "UNKNOWN"
    transport_failed: bool = False

    @property
    def user_message(self) -> str:
        if self.transport_failed:
            return CREDITS_UNAVAILABLE_MESSAGE
        return OUT_OF_CREDITS_MESSAGE


def _identity(user_id: str, user_type: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Split a single incoming id into the (userId, vId) pair the API expects.

    Exactly one side is ever populated — sending both, or neither, is a
    400 missing_identity. Absent user_type is treated as a visitor so requests
    from a not-yet-updated frontend still resolve to one identity.
    """
    if user_type == "logged-in":
        return user_id, None
    return None, user_id


async def reserve(
    *,
    user_id: str,
    user_type: Optional[str],
    c_id: str,
    usage_type: UsageType,
    request_id: str,
) -> ReserveResult:
    """Reserve one credit. Returns allowed=False when the agent must NOT run.

    `request_id` is supplied by the caller and reused verbatim on the internal
    retry below — regenerating it here would defeat the idempotency protection
    and could double-charge a user for a single message.
    """
    if not settings.credits_enforce:
        _log.warning("[CREDITS] enforcement disabled via CREDITS_ENFORCE — allowing request")
        return ReserveResult(allowed=True, reason_code="ENFORCEMENT_DISABLED")

    user_ref, v_id = _identity(user_id, user_type)
    payload = {
        "userId": user_ref,
        "vId": v_id,
        "cId": c_id,
        "usageType": usage_type,
        "requestId": request_id,
    }

    result = await _post(
        "/internal/credits/reserve", payload, request_id, usage_type, log_tag="RESERVE"
    )
    if result is None:
        # Unreachable after retry. Fail CLOSED: an unverifiable request is an
        # unbounded one, and this gate is what caps usage. Flip CREDITS_ENFORCE
        # to fail open if an incident makes that the lesser evil.
        _log.error(
            "[CREDITS] reserve unreachable after retry — failing closed | requestId=%s", request_id
        )
        return ReserveResult(
            allowed=False, reason_code="CREDITS_UNAVAILABLE", transport_failed=True
        )

    allowed = bool(result.get("allowed"))
    reason_code = result.get("reasonCode") or "UNKNOWN"
    remaining = result.get("remaining")
    from_cache = bool(result.get("fromCache"))

    # DENIED_NOT_FOUND / DENIED_INVALID_IDENTITY mean a bug on one side of the
    # wire, not an out-of-credits user. Both are hard failures, but they need to
    # be loud so they get fixed rather than read as normal blocking.
    if reason_code in ("DENIED_NOT_FOUND", "DENIED_INVALID_IDENTITY"):
        _log.error(
            "[CREDITS] hard failure reasonCode=%s | requestId=%s | user_type=%s | cId=%s",
            reason_code, request_id, user_type, c_id,
        )
        allowed = False

    _log.info(
        "[CREDITS] reserve allowed=%s reasonCode=%s remaining=%s fromCache=%s "
        "| requestId=%s | user_type=%s | usageType=%s | cId=%s",
        allowed, reason_code, remaining, from_cache,
        request_id, user_type, usage_type, c_id,
    )
    return ReserveResult(
        allowed=allowed,
        remaining=remaining,
        from_cache=from_cache,
        reason_code=reason_code,
    )


async def refund(
    *,
    user_id: str,
    user_type: Optional[str],
    c_id: str,
    usage_type: UsageType,
    request_id: str,
    cost: Optional[int] = None,
) -> bool:
    """Hand back a credit charged for output the user never received.

    Must carry the SAME request_id as the original reserve — that is how the
    credits service identifies which charge to reverse.

    NOTE: the refund contract is still pending (separate issue). The payload
    below follows the fields named in the reserve spec; `cost` is assumed to be
    a unit count because the reserve response returns no cost to echo back.
    """
    if not settings.credits_enforce:
        return True

    user_ref, v_id = _identity(user_id, user_type)
    payload = {
        "userId": user_ref,
        "vId": v_id,
        "cId": c_id,
        "usageType": usage_type,
        "requestId": request_id,
        "cost": settings.credits_refund_cost if cost is None else cost,
    }

    result = await _post(
        "/internal/credits/refund", payload, request_id, usage_type, log_tag="REFUND"
    )
    if result is None:
        # A lost refund is a silently overcharged user — must be alertable.
        _log.error(
            "[CREDITS] REFUND FAILED — user charged for undelivered output | "
            "requestId=%s | user_type=%s | usageType=%s | cId=%s",
            request_id, user_type, usage_type, c_id,
        )
        return False

    _log.info("[CREDITS] refunded | requestId=%s | usageType=%s", request_id, usage_type)
    return True


async def refund_safe(**kwargs) -> None:
    """Fire-and-forget refund — never lets a credits failure break the stream."""
    try:
        await refund(**kwargs)
    except Exception as e:
        _log.error("[CREDITS] refund raised | requestId=%s | %s", kwargs.get("request_id"), e)


async def _post(
    path: str,
    payload: dict,
    request_id: str,
    usage_type: str,
    log_tag: str,
) -> Optional[dict]:
    """POST with one retry on transport/5xx failure. Returns None if unreachable.

    The retry deliberately re-sends the identical payload, request_id included.
    httpx does no automatic request retries, so this loop is the only retry in
    play — do not add transport-level retries without re-reading the idempotency
    rules in the spec.
    """
    url = f"{settings.credits_base_url.rstrip('/')}{path}"
    headers = {
        "Content-Type": "application/json",
        "X-Callback-Secret": settings.credits_callback_secret or "",
    }

    for attempt in (1, 2):
        try:
            # Full request body, every attempt — a retry logs again so the
            # reused requestId is visible in the trace. Headers are never
            # logged: they carry the shared secret.
            _log.info(
                "[CREDITS] %s INPUT (attempt %d) POST %s\n%s",
                log_tag, attempt, url, json.dumps(payload, indent=2),
            )
            response = await _credits_client.post(url, json=payload, headers=headers)

            # 4xx are our bugs — a retry sends the same broken payload, so stop.
            if 400 <= response.status_code < 500:
                _log.error(
                    "[CREDITS] %s OUTPUT (HTTP %d) — rejected, check config/payload | requestId=%s\n%s",
                    log_tag, response.status_code, request_id, _safe_body(response),
                )
                return None

            response.raise_for_status()
            body = response.json()
            _log.info(
                "[CREDITS] %s OUTPUT (HTTP %d)\n%s",
                log_tag, response.status_code, json.dumps(body, indent=2),
            )
            return body

        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            if attempt == 1:
                _log.warning(
                    "[CREDITS] %s attempt 1 failed (%s) — retrying with same requestId=%s",
                    log_tag, type(e).__name__, request_id,
                )
                await asyncio.sleep(0.15)
                continue
            _log.error(
                "[CREDITS] %s failed after retry (%s) | requestId=%s | usageType=%s",
                log_tag, type(e).__name__, request_id, usage_type,
            )
            return None
        except Exception as e:
            _log.error("[CREDITS] %s unexpected error: %s | requestId=%s", log_tag, e, request_id)
            return None

    return None


def _safe_body(response: httpx.Response) -> str:
    """Response body for logging — truncated, and never echoes the request headers."""
    try:
        return json.dumps(response.json(), indent=2)[:500]
    except Exception:
        return (response.text or "")[:500]
