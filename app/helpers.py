import asyncio
import json
import re
import time
from typing import AsyncGenerator, Dict, Any, Optional

import httpx
import pycountry
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from agents import Runner

from .schemas import QueryRequest
from .services import (
    _openai_client,
    get_complete_response,
    stream_agent_to_queue,
    summarize_for_rag,
    build_rag_payload,
    check_pii_fast,
    generate_loading_statements,
    get_instant_loading_message,
)
from .agents_ import validation_agent
from .memory import setup_user_session, add_message, get_user_memory_for_engage
from .region_metadata import REGION_METADATA
from . import redis_history as _redis_history
from .rag_cache import build_rag_cache_key, get_rag_cache, set_rag_cache
import os
import logging

_log = logging.getLogger("app.agent")

_REDIS_TTL      = int(os.getenv("REDIS_TTL_SECONDS", "600"))
_REDIS_MAX      = int(os.getenv("REDIS_OLD_INTERACTIONS_MAX", "30"))


# ─── Real-time query detector ─────────────────────────────────────
# Routes queries that need live web data to web_search_agent.
_REALTIME_SIGNALS = [
    # simple time-now phrases (catches "Dubai right now", "what to do right now", etc.)
    "right now", "right now?",
    "currently", "current conditions", "at the moment", "these days", "nowadays",
    # safety / news
    "is it safe", "is safe", "safe to travel", "travel safe", "safety",
    "should i travel", "should i go", "should i visit", "should i still travel",
    "should i still go", "is it ok to travel", "is it okay to travel",
    "ok to travel", "okay to travel", "safe to visit", "safe to go",
    "current situation", "situation now",
    "latest news", "latest update", "latest updates", "latest situation",
    "whats happening", "what is happening",
    "travel advisory", "travel warning", "travel alert",
    "any conflict", "any danger", "any protests", "any unrest",
    "is there war", "is there conflict", "is there fighting",
    # weather
    "weather today", "current weather", "weather forecast",
    # events / opening hours
    "events tonight", "events this week", "events today",
    "open now", "closing time today",
    # pricing
    "ticket price", "entry fee", "how much to enter",
    # entry / visa
    "entry requirements", "visa requirements", "border open",
]

def _is_realtime_query(message: str) -> bool:
    """Returns True when the query needs live web data, not just RAG place data."""
    # normalise apostrophes so "what's" and "what\u2019s" both match
    msg = message.lower().replace("\u2019", "'").replace("\u2018", "'")
    return any(signal in msg for signal in _REALTIME_SIGNALS)



# --- Conversational-reply detector ---
# When the user is answering a mindset follow-up question (not starting a new
# travel query), skip RAG entirely and inject [FOLLOW_UP_MODE] instead.

_QUERY_STARTS_FOLLOWUP = (
    "best ", "top ", "what ", "where ", "which ", "how ", "show ", "give ",
    "list ", "find ", "recommend", "suggest",
    "tell ", "can ", "is ", "are ", "do ", "does ",
)
_QUERY_CONTAINS_FOLLOWUP = (
    "things to do", "places to visit", "places to see",
    "what to do", "where to go", " vs ", " vs.", "which is better",
    "compare ", "or better",
)
_PERSONAL_OPENER_RE = re.compile(
    r"^(i\b|we\b|my\b|our\b|for\s+(me|us)\b)",
    re.IGNORECASE,
)


def _is_conversational_reply(message: str, history: list[dict] | None) -> bool:
    """
    Returns True when the user's message looks like a conversational answer
    to the agent's previous follow-up question rather than a new travel query.
    """
    if not history:
        return False

    # The last assistant turn must have ended with a question
    last_answer = (history[-1].get("answer") or "").strip()
    _clean = last_answer.rstrip('"').rstrip("}").strip()
    if not _clean.endswith("?"):
        return False

    # After 2 consecutive follow-up turns the planning question was already asked.
    # Force full validation flow so isTravelRelated can fire and trip planning can start.
    if _count_followup_turns(history) >= 2:
        return False

    msg = message.strip()
    msg_lower = msg.lower()

    # Strong travel-query signals -> treat as a new query, not a reply
    if any(msg_lower.startswith(q) for q in _QUERY_STARTS_FOLLOWUP):
        return False
    if any(q in msg_lower for q in _QUERY_CONTAINS_FOLLOWUP):
        return False
    if len(msg) > 400:
        return False

    # Personal opener -> almost certainly a reply
    if _PERSONAL_OPENER_RE.match(msg):
        return True

    # Short message with no strong query signals -> treat as a reply
    if len(msg) < 200:
        return True

    return False


def _count_followup_turns(history: list[dict] | None) -> int:
    # Count consecutive follow-up-mode responses at end of history.
    # A follow-up-mode response has no POI block.
    # T3 (most recent, i=0) may end without '?' (offer statement vs. question) — POI
    # absence alone is enough. Older items require '?' to confirm they were question turns.
    if not history:
        return 0
    count = 0
    for i, item in enumerate(reversed(history)):
        answer = (item.get("answer") or "").strip().rstrip('"').rstrip("}").strip()
        no_poi = "<poi" not in answer.lower() and "<pois>" not in answer.lower()
        if no_poi and (i == 0 or answer.endswith("?")):
            count += 1
        else:
            break
    return count


# ─── Instant Location Scope Gate ─────────────────────────────────

def _check_location_scope(message: str, reference: str) -> Optional[str]:
    """
    Instant Python-only location scope gate — no LLM, no I/O.

    Algorithm:
      1. Look up REGION_METADATA[reference]. If absent → allow (unscoped portal).
      2. If query mentions portal's country, any known location, or any known
         experience keyword → allow.
      3. Scan 1-gram and 2-gram tokens with pycountry. If a token resolves to a
         country that differs from the portal's country code → block.
      4. No foreign country detected → allow (generic query).

    Returns None to allow, or a user-facing string to block.
    """
    region_meta = REGION_METADATA.get(reference)
    if not region_meta:
        return None  # no scope restriction for this reference

    q_lower = message.lower()
    portal_country_cd = (region_meta.get("countryCd") or "").upper()

    # ── Pass 1: explicit in-scope match ──────────────────────────
    if region_meta.get("country", "").lower() in q_lower:
        return None
    for loc in region_meta.get("locations", []):
        if loc.lower() in q_lower:
            return None
    for exp in region_meta.get("experiences", []):
        if exp.lower() in q_lower:
            return None

    # ── Pass 2: detect a foreign country name ────────────────────
    # Build 1-grams and 2-grams from the raw message
    words = re.findall(r"[A-Za-z\-\']+", message)
    candidates: list[str] = list(words)
    for i in range(len(words) - 1):
        candidates.append(f"{words[i]} {words[i + 1]}")

    for token in candidates:
        token = token.strip(" '-")
        if not token:
            continue
        try:
            country_obj = pycountry.countries.lookup(token)
            if country_obj.alpha_2 != portal_country_cd:
                portal_country = region_meta.get("country", "this destination")
                return (
                    f"This assistant is set up for {portal_country}. "
                    f"Please ask about experiences, places, or activities in {portal_country}."
                )
        except LookupError:
            pass  # token doesn't match any country — continue

    # ── Pass 3: no location signals at all → generic query → allow ─
    return None


# ─── RAG Helper ──────────────────────────────────────────────────

# Shared async client — one connection pool for all requests, zero thread usage
_rag_client = httpx.AsyncClient(timeout=30.0)


async def rag(
    query: str,
    reference: str,
    thread_id: str = "",
    location: str = "",
    filters: str = "",
    top_k: int = 8,
    category: str = "places",
) -> Dict[str, Any]:
    """Fully async RAG call via shared httpx client (/chat endpoint)."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or None")

    payload = {
        "query": query.strip(),
        "reference": reference,
        "top_k": top_k,
        "category": category,
        "location": location or "",
        "filters": filters or "",
        "thread_id": thread_id,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    print("\n" + "=" * 80)
    print("[RAG /chat] >>> INPUT")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("=" * 80)

    try:
        response = await _rag_client.post(
            url="https://rag.hiptraveler.com/chat",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"success": True, "data": response.text, "status_code": response.status_code}

        print("\n" + "-" * 80)
        print("[RAG /chat] <<< OUTPUT")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("-" * 80 + "\n")
        return result

    except httpx.TimeoutException:
        raise Exception("RAG request timed out after 30 seconds")
    except httpx.ConnectError:
        raise Exception("Failed to connect to RAG webhook")
    except httpx.HTTPStatusError as e:
        raise Exception(f"RAG HTTP error: {e}")


async def rag_guide(
    query: str,
    reference: str,
    thread_id: str,
    location: str = "",
    filters: str = "",
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Destination-guide retrieval via the /chat/guide endpoint.
    Runs in parallel with rag(); returns guide content for the agent.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or None")

    payload = {
        "query": query.strip(),
        "reference": reference,
        "top_k": top_k,
        "location": location or "",
        "filters": filters or "",
        "thread_id": thread_id,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    print("\n" + "=" * 80)
    print("[GUIDE /chat/guide] >>> INPUT")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("=" * 80)

    try:
        response = await _rag_client.post(
            url="https://rag.hiptraveler.com/chat/guide",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"success": True, "data": response.text, "status_code": response.status_code}

        print("\n" + "-" * 80)
        print("[GUIDE /chat/guide] <<< OUTPUT")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("-" * 80 + "\n")
        return result

    except httpx.TimeoutException:
        raise Exception("Guide request timed out after 30 seconds")
    except httpx.ConnectError:
        raise Exception("Failed to connect to Guide webhook")
    except httpx.HTTPStatusError as e:
        raise Exception(f"Guide HTTP error: {e}")


# ─── Explore Context Extractor ───────────────────────────────────

def _extract_explore_context(history: list[dict] | None) -> tuple[str, list[str]]:
    """
    Parses the most recent explore response from Redis history.

    Returns:
      - context_str : a [PREVIOUS_EXPLORE_CONTEXT] block for the trip planner (activity hint)
      - pois        : list of place names extracted from the previous explore response.
                      These are injected directly into the TripPlan result in Python —
                      we do NOT rely on the LLM to populate pois from context, because
                      the UNIVERSAL RULE blocks extraction from assistant-side text.
    """
    if not history:
        return "", []

    # history is oldest→newest; most recent is last
    last = history[-1]

    # ── Extract place names ───────────────────────────────────────
    pois: list[str] = []
    answer = last.get("answer", "")
    # Parse <title> tags from XML POI format
    pois = re.findall(r'<title>([^<]+)</title>', answer)
    if not pois:
        # Fallback: extract **Bold Place Names** from web search text
        pois = re.findall(r'\*\*([^*]+)\*\*', answer)

    # ── Build context string for activity hint only ───────────────
    original_query = (last.get("question") or "").strip()
    if not original_query:
        return "", pois

    context_str = (
        "\n\n[PREVIOUS_EXPLORE_CONTEXT]\n"
        f"Previous user search: {original_query}\n"
        "[/PREVIOUS_EXPLORE_CONTEXT]"
    )
    return context_str, pois


# ─── Streaming-with-starter generator (isTravelRelated=False only) ──

async def streaming_with_loading(
    context: str,
    agent_name: str,
    final_message_with_ref: str,
    original_message: str,
    thread_id: str,
    param: str,
) -> AsyncGenerator[str, None]:
    """
    SSE generator used only when isTravelRelated=False.

    Two tasks fire at t=0:
      • stream_starter_to_queue — gpt-4.1-nano streams a 2-3 sentence human-sounding
                                   opener immediately so the user sees real content at once.
      • stream_agent_to_queue   — main agent runs in parallel, tokens land in token_queue.

    Phase 1: starter tokens stream to the client instantly (TTFB < 500ms).
    Bridge:  if the agent hasn't produced its first token within 800ms of the starter
             finishing, one loading stage fires to cover the gap.
    Phase 2: main agent tokens stream seamlessly after the starter.
    """
    start_time = time.time()

    yield f"data: {json.dumps({'start_time': start_time, 'status': 'started', 'threadId': thread_id})}\n\n"

    # Both fire at t=0 — zero added latency
    starter_queue = asyncio.Queue()
    token_queue   = asyncio.Queue()

    asyncio.create_task(
        stream_starter_to_queue(original_message, param, starter_queue)
    )
    asyncio.create_task(
        stream_agent_to_queue(
            agent_name=agent_name,
            final_message_with_ref=final_message_with_ref,
            original_message=original_message,
            thread_id=thread_id,
            queue=token_queue,
        )
    )

    # ── Phase 1: stream starter tokens ───────────────────────────
    ttfb_sent = False
    while True:
        token = await starter_queue.get()
        if token is None:
            break
        if not ttfb_sent:
            ttfb_sent = True
            yield f"data: {json.dumps({'time_to_first_byte': time.time() - start_time})}\n\n"
        yield f"data: {json.dumps({'content': token})}\n\n"

    # ── Bridge: wait silently for first agent token ───────────────
    while True:
        token = await token_queue.get()
        if token is None:
            end_time = time.time()
            yield f"data: {json.dumps({'done': True, 'total_time': end_time - start_time, 'threadId': thread_id, 'param': param})}\n\n"
            return
        if isinstance(token, Exception):
            yield f"data: {json.dumps({'error': str(token)})}\n\n"
            return
        yield f"data: {json.dumps({'content': token})}\n\n"
        break

    # ── Phase 2: stream main agent tokens ────────────────────────
    while True:
        token = await token_queue.get()
        if token is None:
            break
        if isinstance(token, Exception):
            yield f"data: {json.dumps({'error': str(token)})}\n\n"
            break
        yield f"data: {json.dumps({'content': token})}\n\n"

    end_time = time.time()
    yield f"data: {json.dumps({'done': True, 'total_time': end_time - start_time, 'threadId': thread_id, 'param': param})}\n\n"


# ─── Error / RAG-note stream helpers ─────────────────────────────

def get_rag_note_stream_response(note: str, thread_id: str, param: str):
    async def _gen() -> AsyncGenerator[str, None]:
        start_time = time.time()
        yield f"data: {json.dumps({'start_time': start_time, 'status': 'started'})}\n\n"
        yield f"data: {json.dumps({'time_to_first_byte': time.time() - start_time})}\n\n"
        yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n\n"
        for i, word in enumerate(note.split()):
            chunk = word if i == 0 else f" {word}"
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield f"data: {json.dumps({'content': '\"}'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total_time': time.time() - start_time, 'threadId': thread_id, 'param': param, 'response_type': 'rag_streaming'})}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def get_error_stream_response(reason: str, solution: str):
    async def _gen() -> AsyncGenerator[str, None]:
        start_time = time.time()
        yield f"data: {json.dumps({'start_time': start_time, 'status': 'started'})}\n\n"
        yield f"data: {json.dumps({'time_to_first_byte': time.time() - start_time})}\n\n"

        if len(solution) < 50:
            for chunk in [
                '{"', "answer", '":"',
                "Let", "'s", " keep", " it", " travel", "-focused", ".\n\n",
                "What", " would", " you", " like", " to", " explore", " next", "?",
                '"}',
            ]:
                if chunk:
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
        else:
            yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n\n"
            for i, word in enumerate(solution.split()):
                chunk = word if i == 0 else f" {word}"
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield f"data: {json.dumps({'content': '\"}'})}\n\n"

        yield f"data: {json.dumps({'done': True, 'total_time': time.time() - start_time, 'blocked': True})}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ─── Main Stream Generator ────────────────────────────────────────

async def _main_stream(
    request: QueryRequest,
    thread_id: str,
    param: str,
    final_message: str,
    history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    start_time = time.time()

    # ── t=0: client gets [STARTED] before any network calls ──────
    yield f"data: {json.dumps({'start_time': start_time, 'status': 'started', 'threadId': thread_id})}\n\n"

    # ── Save user question to Zep — always, non-blocking, never gates anything ──
    if request.user_id:
        async def _save_to_zep():
            try:
                await asyncio.to_thread(setup_user_session, request.user_id, thread_id)
                await asyncio.to_thread(add_message, role='user', thread_id=thread_id, message=request.message)
            except Exception as e:
                pass
        asyncio.create_task(_save_to_zep())

    # ── Fast-path: plan=True skips ALL middleware ────────────────
    # No PII check, no RAG, no validation, no loaders, no streaming.
    # Returns the trip plan JSON directly and exits.
    if request.plan:
        try:
            ctx_str, ctx_pois = _extract_explore_context(history)
            enriched = final_message + ctx_str
            response_content, timing_info = await get_complete_response(enriched, thread_id, param)
            if ctx_pois and not response_content.pois:
                response_content.pois = ctx_pois
            yield f"data: {json.dumps({'travel': [jsonable_encoder(response_content), jsonable_encoder(timing_info)], 'type': 'non-streaming', 'done': True})}\n\n"
            # ── Clear or save to Redis ──
            if not response_content.feedback:
                asyncio.create_task(_redis_history.clear_conversation(thread_id))
            elif response_content.summary:
                asyncio.create_task(_redis_history.append_interaction(
                    thread_id,
                    question=request.message,
                    answer=json.dumps(jsonable_encoder(response_content)),
                    max_items=_REDIS_MAX, ttl_seconds=_REDIS_TTL,
                ))
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return

    # ── Detect follow-up reply — skip RAG when user is answering a mindset question ──
    _is_followup = _is_conversational_reply(request.message, history)
    _fup_count_pre = _count_followup_turns(history)  # needed for USER_PREFERENCES injection at T4

    # ── Fire ALL tasks at t=0 — RAG, validation, PII, loaders, scope, Zep all in parallel ──
    # scope gate runs in a thread so pycountry I/O never blocks the event loop
    scope_task    = asyncio.create_task(asyncio.to_thread(_check_location_scope, request.message, request.reference))
    pii_task      = asyncio.create_task(check_pii_fast(request.message))

    async def _summarize_then_fetch() -> tuple[Any, Any]:
        # One nano call parses the message into a full structured payload
        # (query/category/top_k/location/filters); it then feeds both retrieval
        # endpoints, fired in parallel. Same single nano round-trip as before.
        p = await build_rag_payload(
            final_message,
            fallback_location=request.location or "",
            fallback_filters=request.filters or "",
        )
        # ── Redis cache check — skip RAG endpoint calls on hit ──
        _cache_key = build_rag_cache_key(
            p["query"], p["location"], p["category"], p["top_k"], request.reference
        )
        _cached = await get_rag_cache(_cache_key)
        if _cached is not None:
            return _cached  # (rag_result, guide_result) — no endpoint calls

        rag_res, guide_res = await asyncio.gather(
            rag(query=p["query"], reference=request.reference, thread_id=thread_id,
                location=p["location"], filters=p["filters"],
                top_k=p["top_k"], category=p["category"]),
            rag_guide(query=p["query"], reference=request.reference, thread_id=thread_id,
                      location=p["location"], filters=p["filters"]),
            return_exceptions=True,
        )
        # Cache result for future identical queries (skips on exceptions)
        await set_rag_cache(_cache_key, rag_res, guide_res)
        return rag_res, guide_res

    # Follow-up replies skip RAG and LLM loaders entirely
    if not _is_followup:
        loading_task  = asyncio.create_task(generate_loading_statements(final_message, "explore"))
        rag_task      = asyncio.create_task(_summarize_then_fetch())
    else:
        loading_task  = None
        rag_task      = None

    if not _is_followup:
        validation_task = asyncio.create_task(Runner.run(validation_agent, final_message))
    else:
        validation_task = None
    await asyncio.sleep(0)   # yield so all tasks go in-flight simultaneously

    # ── TTFB + instant first loader fire IMMEDIATELY ─────────────────
    _instant_loader = get_instant_loading_message("explore") if not _is_followup else "On it…"
    _shown_loaders: list[str] = [_instant_loader]
    yield f"data: {json.dumps({'time_to_first_byte': time.time() - start_time})}\n\n"
    yield f"data: {json.dumps({'loading': _instant_loader})}\n\n"

    if _is_followup:
        pass  # no RAG, no validation — nothing to wait for
    else:
        # ── Pre-fetch LLM loaders while rag+validation run in background ─
        _llm_msgs: list[str] = []
        _llm_msgs_ready = False
        _deep_research_sent = False
        try:
            _llm_msgs = await asyncio.wait_for(asyncio.shield(loading_task), timeout=1.3)
            _llm_msgs_ready = True
        except asyncio.TimeoutError:
            pass  # still running — will retry in the loop
        except Exception:
            _llm_msgs = []
            _llm_msgs_ready = True

        # ── Await RAG + validation, emitting one LLM loader per 1.5s tick ──
        _loader_index = 1
        _pending = {rag_task, validation_task}
        while _pending:
            _, _pending = await asyncio.wait(_pending, timeout=1.5)
            if _pending:  # still waiting — emit next loader
                if not _llm_msgs_ready:
                    try:
                        if loading_task.done():
                            _llm_msgs = loading_task.result() or []
                        else:
                            _llm_msgs = await asyncio.wait_for(
                                asyncio.shield(loading_task), timeout=0.5
                            )
                        _llm_msgs_ready = True
                    except asyncio.TimeoutError:
                        pass
                    except Exception:
                        _llm_msgs = []
                        _llm_msgs_ready = True

                _llm_pos = _loader_index - 1
                if _llm_pos < len(_llm_msgs):
                    _msg: str | None = _llm_msgs[_llm_pos]
                elif not _deep_research_sent:
                    _msg = "Doing some deep research on this — almost there…"
                    _deep_research_sent = True
                else:
                    _msg = None  # deep-research already shown — stay silent
                if _msg:
                    _shown_loaders.append(_msg)
                    yield f"data: {json.dumps({'loading': _msg})}\n\n"
                _loader_index += 1

    # ── Scope gate (resolved by now — fired at t=0 in a thread) ─────
    try:
        scope_error = scope_task.result() if scope_task.done() else await scope_task
    except Exception:
        scope_error = None
    if scope_error:
        yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n\n"
        for i, word in enumerate(scope_error.split()):
            yield f"data: {json.dumps({'content': word if i == 0 else ' ' + word})}\n\n"
        yield f"data: {json.dumps({'content': '\"}'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total_time': time.time() - start_time, 'threadId': thread_id, 'param': param, 'response_type': 'scope_gate'})}\n\n"
        return

    # ── PII check (resolved by now — fired at t=0 in parallel) ───────
    try:
        pii_detected = pii_task.result() if pii_task.done() else await pii_task
    except Exception:
        pii_detected = False
    if pii_detected:
        pii_message = (
            "To keep your information safe, please avoid sharing personal details "
            "like phone numbers, email addresses, or ID numbers in your messages. "
            "Feel free to ask me anything about travel destinations and I'll be happy to help!"
        )
        yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n\n"
        for i, word in enumerate(pii_message.split()):
            yield f"data: {json.dumps({'content': word if i == 0 else ' ' + word})}\n\n"
        yield f"data: {json.dumps({'content': '\"}'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'blocked': True, 'total_time': time.time() - start_time, 'reason': 'PII_DETECTED'})}\n\n"
        return

    # Both done — extract results.
    if rag_task is not None:
        try:
            rag_result, guide_result = rag_task.result()
        except Exception as e:
            rag_result = e
            guide_result = e
    else:
        rag_result = None
        guide_result = None

    if validation_task is not None:
        try:
            validation_result = validation_task.result()
        except Exception as e:
            validation_result = e
    else:
        validation_result = None

    # ── RAG result ───────────────────────────────────────────────
    if rag_result is None:
        pass  # follow-up mode — RAG intentionally skipped
    elif isinstance(rag_result, Exception):
        _log.warning("[RAG] failed: %s — falling back to web_search_agent", rag_result)
    else:
        _log.info("[RAG] returned %d chunk(s)", len(rag_result.get("chunks") or []))

    # ── Destination-guide result ─────────────────────────────────
    if guide_result is None:
        pass  # follow-up mode — guide intentionally skipped
    elif isinstance(guide_result, Exception):
        _log.warning("[GUIDE] failed: %s", guide_result)
    else:
        _log.info("[GUIDE] returned guide payload (%d keys)", len(guide_result or {}))

    # ── Validation check ─────────────────────────────────────────
    if isinstance(validation_result, Exception):
        yield f"data: {json.dumps({'error': str(validation_result)})}\n\n"
        return


    if (not _is_followup
            and validation_result is not None
            and not isinstance(validation_result, Exception)
            and not validation_result.final_output.isValid):
        solution = validation_result.final_output.solution
        if len(solution) < 50:
            solution = "Let's keep it travel-focused. What would you like to explore next?"
        yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n\n"
        for i, word in enumerate(solution.split()):
            yield f"data: {json.dumps({'content': word if i == 0 else ' ' + word})}\n\n"
        yield f"data: {json.dumps({'content': '\"}'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total_time': time.time() - start_time, 'blocked': True})}\n\n"
        return

    # ── RAG injection for streaming agents ───────────────────────
    # Only chunks signal a real RAG hit — other fields are metadata
    rag_chunks = []
    rag_data = {}
    if rag_result is not None and not isinstance(rag_result, Exception):
        rag_chunks = rag_result.get("chunks") or []
        if rag_chunks:
            rag_data = {
                k: rag_result.get(k, [])
                for k in ("entities", "chunks", "audience", "travel_style")
                if rag_result.get(k)
            }
            _log.info(
                "[RAG] injecting %d chunks: %s",
                len(rag_chunks),
                [f"{c.get('name')} ({c.get('id')})" for c in rag_chunks],
            )

    final_message_with_ref = final_message + "\n\nReference : " + request.reference

    # ── Trip planning (isTravelRelated=True) ─────────────────────
    # Route is decided purely by the validation agent's INTENT verdict —
    # no keyword/sentence backstops.
    if not _is_followup and validation_result.final_output.isTravelRelated:
        try:
            ctx_str, ctx_pois = _extract_explore_context(history)
            enriched = final_message + ctx_str
            response_content, timing_info = await get_complete_response(enriched, thread_id, param)
            if ctx_pois and not response_content.pois:
                response_content.pois = ctx_pois
            yield f"data: {json.dumps({'travel': [jsonable_encoder(response_content), jsonable_encoder(timing_info)], 'type': 'non-streaming', 'done': True})}\n\n"
            # ── Clear or save to Redis ────────────────────────────
            if not response_content.feedback:
                asyncio.create_task(_redis_history.clear_conversation(thread_id))
            elif response_content.summary:
                asyncio.create_task(_redis_history.append_interaction(
                    thread_id,
                    question=request.message,
                    answer=json.dumps(jsonable_encoder(response_content)),
                    max_items=_REDIS_MAX, ttl_seconds=_REDIS_TTL,
                ))
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return

    # ── Explore / General (isTravelRelated=False) ─────────────────
    # Realtime is decided by INTENT in the validation agent (isRealtime),
    # not by keyword matching. Fall back to the keyword heuristic only if
    # the field is somehow missing.
    if validation_result is not None and not isinstance(validation_result, Exception):
        is_realtime = getattr(validation_result.final_output, "isRealtime", None) or False
    else:
        is_realtime = _is_realtime_query(request.message)

    # Skip RAG injection for real-time queries — web_search_agent expects
    # no RAG data and its instruction says "RAG returned nothing".
    # Injecting irrelevant place chunks would cause it to format places
    # instead of fetching live information.
    if rag_data and not is_realtime:
        # Build a clear ID lookup table above the raw JSON so the agent
        # can only match IDs to exact names — no guessing.
        _chunks = rag_data.get("chunks") or []
        _id_table_lines = ["VERIFIED ID TABLE — only these IDs exist, matched to their exact name:"]
        for _c in _chunks:
            _cid   = _c.get("id", "")
            _cname = _c.get("name", "")
            _ctype = _c.get("type", "place")
            _cat   = {"activity": "activity", "restaurant": "restaurant", "food": "restaurant",
                      "dine": "restaurant", "hotel": "hotel", "accommodation": "hotel",
                      "stay": "hotel", "tour": "tour", "place": "place"}.get(_ctype, "place")
            _id_table_lines.append(f'  id="{_cid}" | name="{_cname}" | category="{_cat}"')
        _id_table = "\n".join(_id_table_lines)
        final_message_with_ref += (
            f"\n\n[RAG_RESULTS]\n"
            f"{_id_table}\n\n"
            f"FULL DATA:\n{json.dumps(rag_data)}\n"
            f"[/RAG_RESULTS]"
        )

    # ── Destination-guide injection (runs alongside RAG, same gate) ──
    # Guide content is editorial/destination context, not place-id data.
    # The agent must render it INSIDE the answer — a short guide section at the
    # TOP and another at the BOTTOM — in the same format as the rest of the answer.
    if not is_realtime and guide_result is not None and not isinstance(guide_result, Exception) and guide_result:
        final_message_with_ref += (
            f"\n\n[DESTINATION_GUIDE]\n"
            f"Structured destination guide data for this query. "
            f"Use this to make your response specific and informative — not generic. "
            f"Do NOT dump raw JSON or field names. Weave facts naturally into your answer voice.\n\n"
            f"HOW TO USE EACH FIELD:\n"
            f"- summary: foundation for your opening framing of the destination.\n"
            f"- known_for / highlights: what makes it worth visiting — reference these when describing the destination's identity.\n"
            f"- activities: use the name, intensity, duration, and bestSeasons of each activity to give concrete nuanced descriptions "
            f"(e.g. 'best in spring', 'full-day commitment', 'moderate intensity'). This makes POI descriptions specific, not generic.\n"
            f"- characteristics (pace, climate, familyFriendly, averageDailyBudget, recommendedMaxDays): weave into practical advice — "
            f"pacing suggestions, who the destination suits, rough budget framing, how many days it rewards.\n"
            f"- itinerary_hints: use mobility notes and logistics tips in your practical closing section.\n"
            f"- best_months / travel_styles: reference when discussing timing or matching the user's stated travel style.\n"
            f"- practical (language, timezone): mention when useful for first-time visitors.\n\n"
            f"STRUCTURE RULE: guide data should inform the BODY of your response, not just bookend sentences. "
            f"Every POI description and practical tip should be richer because of this data.\n\n"
            f"{json.dumps(guide_result)}\n"
            f"[/DESTINATION_GUIDE]"
        )

    if _is_followup:
        _fup_count = _count_followup_turns(history)
        if _fup_count >= 1:
            # User has answered Q1 and Q2 — this is Turn 3: advise + planning question
            _fup_instruction = (
                "\n\n[FOLLOW_UP_MODE: TURN 3 — ACKNOWLEDGE + ADVISE + PLANNING QUESTION]\n"
                "The user has answered both Q1 (experience priorities) and Q2 (travel pace/style). "
                "You now have a clear picture of what they want.\n\n"
                "Write a MEDIUM-LENGTH response (no <POIS> block):\n\n"
                "1. ACKNOWLEDGMENT (1 sentence) — pull together both Q1 and Q2 answers into one specific, "
                "warm sentence that shows you absorbed both.\n\n"
                "2. PERSONALIZED ADVICE (3–5 points) — show concretely how this destination fits their "
                "exact profile. Use your own travel knowledge and any [DESTINATION_GUIDE] data present "
                "(itinerary_hints, best_months, characteristics, known_for). Name specific neighborhoods, "
                "areas, timing, or approaches that suit their pace and priorities. Suggest a rough framing "
                "for the trip (e.g. how many days, which area to base in, what to anchor around). Be "
                "concrete and personal — not generic praise. Use bullet points when listing distinct tips, "
                "or flowing prose when it reads more naturally — whichever formats better.\n\n"
                "3. PLANNING QUESTION (1 sentence, must end with ?) —\n"
                "   • If the destination is clear from context: 'Based on what you've told me, "
                "[Destination] sounds like the perfect fit — want me to build the trip around that?'\n"
                "   • If destination is unclear: ask which destination they want to anchor the trip around.\n"
                "   One question only. Always ends with '?'.\n\n"
                "Do NOT output a <POIS> block.\n"
                "[/FOLLOW_UP_MODE]"
            )
        else:
            # User answered Q1 — this is Turn 2: advise + ask Q2
            _fup_instruction = (
                "\n\n[FOLLOW_UP_MODE: TURN 2 — ACKNOWLEDGE + ADVISE + ASK Q2]\n"
                "The user just answered Q1 (what they value / experience priorities).\n\n"
                "Write a MEDIUM-LENGTH response (no <POIS> block):\n\n"
                "1. ACKNOWLEDGMENT (1 sentence) — reflect what they said in a way that shows you "
                "understood, without just repeating their words.\n\n"
                "2. DESTINATION-SPECIFIC ADVICE (3–5 points) — connect their stated priorities "
                "to the destination using your own travel knowledge and any [DESTINATION_GUIDE] data "
                "present (activities with bestSeasons/intensity, characteristics, known_for, "
                "itinerary_hints). Be concrete: name neighborhoods, timing, what kind of experience "
                "they will get. Not generic ('you'll love it') but specific ('the Alfama lanes are "
                "quietest before 9am in spring'). Use bullet points when listing distinct tips or "
                "advice points, or flowing prose when it reads more naturally — whichever formats better.\n\n"
                "3. FOLLOW-UP QUESTION (1 sentence, ends with ?) — ask Q2: target their TRAVEL PACE "
                "AND STYLE: how they like their days to unfold, how much ground they cover, the rhythm "
                "that makes a trip feel right. Open-ended, no option lists.\n\n"
                "Do NOT output a <POIS> block.\n"
                "[/FOLLOW_UP_MODE]"
            )
        final_message_with_ref += _fup_instruction
    else:
        # If the user just completed the Q1+Q2 follow-up cycle (T4 = accepted planning),
        # extract their two preference answers and inject as hard constraints.
        if _fup_count_pre >= 2 and history and len(history) >= 2:
            _q1_ans = (history[-2].get("question") or "").strip()
            _q2_ans = (history[-1].get("question") or "").strip()
            if _q1_ans or _q2_ans:
                final_message_with_ref += (
                    "\n\n[USER_PREFERENCES]\n"
                    "Before accepting this plan, the user answered two follow-up questions.\n"
                    "Treat these as HARD CONSTRAINTS — every POI selection, daily pace, and\n"
                    "neighborhood choice must reflect both answers:\n\n"
                    f"Experience priorities (what they value): \"{_q1_ans}\"\n"
                    f"Travel pace and style (how they travel): \"{_q2_ans}\"\n"
                    "[/USER_PREFERENCES]"
                )
        # Loading context: agent opens without echoing the loader text
        _loaders_block = "\n".join(f"- {m}" for m in _shown_loaders)
        final_message_with_ref += (
            "\n\n[LOADING_CONTEXT]\n"
            f"While the user waited, they saw these loading messages:\n{_loaders_block}\n"
            f"Last message shown: \"{_shown_loaders[-1]}\"\n"
            "Do NOT reference, repeat, or echo any loading message text in your response. "
            "Jump directly into your answer content.\n"
            "[/LOADING_CONTEXT]"
        )



    final_message_with_ref += (
        "\n\n[INSTRUCTION] "
        "NEVER start with: 'I am HipTraveler', 'I\u2019m HipTraveler', 'Your name is', 'Hi', 'Hello', or any self-introduction or greeting. "
        "Jump directly into the content — bullet list, facts, or answer body. "
        "If the user only greeted you or shared their name, respond with a single short question about their travel plans. "
        "If [USER_PREFERENCES] data is present and the query is about preferences, answer specifically from that data. [/INSTRUCTION]"
    )

    # Routing:
    #  - memory queries        → rag_format_agent  (answers from [USER_PREFERENCES])
    #  - real-time queries     → web_search_agent  (mandatory live search, no RAG noise)
    #  - RAG chunks present    → rag_format_agent
    #  - no chunks, not r-t    → web_search_agent
    if _is_followup:
        agent_name = "rag_format_agent"
    elif (validation_result is not None
          and not isinstance(validation_result, Exception)
          and validation_result.final_output.isMemoryQuery):
        agent_name = "rag_format_agent"
    elif is_realtime:
        agent_name = "web_search_agent"
    elif rag_chunks:
        agent_name = "rag_format_agent"
    else:
        agent_name = "web_search_agent"

    # One-line routing decision — explains exactly why this agent was chosen.
    _log.info(
        "[ROUTE] agent=%s | isRealtime=%s | isMemoryQuery=%s | rag_chunks=%d | query=%r",
        agent_name,
        is_realtime,
        (validation_result.final_output.isMemoryQuery if validation_result and not isinstance(validation_result, Exception) else False),
        len(rag_chunks),
        request.message,
    )

    _log.info(
        "[AGENT INPUT] agent=%s | user=%s\n%s",
        agent_name, request.user_id, final_message_with_ref,
    )

    token_queue = asyncio.Queue()
    asyncio.create_task(
        stream_agent_to_queue(
            agent_name=agent_name,
            final_message_with_ref=final_message_with_ref,
            original_message=request.message,
            thread_id=thread_id,
            queue=token_queue,
            is_pro=request.is_pro,
        )
    )

    # ── Phase 2: stream main agent tokens ────────────────────────────
    # Loaders were already emitted during rag+validation wait — just stream now.
    redis_answer_parts: list[str] = []
    stream_error = False
    answer_prefix_sent = False

    while True:
        token = await token_queue.get()
        if token is None:
            break
        if isinstance(token, Exception):
            yield f"data: {json.dumps({'error': str(token)})}\n\n"
            stream_error = True
            break
        if not answer_prefix_sent:
            answer_prefix_sent = True
            yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n\n"
        yield f"data: {json.dumps({'content': token})}\n\n"
        redis_answer_parts.append(token)

    yield f"data: {json.dumps({'content': '\"}'})}\n\n"
    yield f"data: {json.dumps({'done': True, 'total_time': time.time() - start_time, 'threadId': thread_id, 'param': param})}\n\n"

    # ── Save explore Q&A to Redis (non-blocking, fire-and-forget) ─
    if redis_answer_parts and not stream_error:
        asyncio.create_task(_redis_history.append_interaction(
            thread_id,
            question=request.message, answer=''.join(redis_answer_parts),
            max_items=_REDIS_MAX, ttl_seconds=_REDIS_TTL,
        ))


# ─── Conversation Context Helpers ─────────────────────────────────

def clean_answer(answer: str) -> str:
    return answer.strip()


def build_conversation_context(request: QueryRequest, history: list[dict] | None = None) -> str:
    """Build the full message string from Redis history (oldest→newest order)."""
    if not history:
        return request.message

    if len(history) == 1:
        last = history[0]
        return (
            f"Last conversation (this is the continuation of the conversation):\n"
            f"User: {last['question']}\nAssistant: {clean_answer(last['answer'])}\n\n"
            f"New question asked: {request.message}"
        )

    # 2+ interactions: all but last are "previous", last is most recent
    lines = ["Previous conversations:"]
    for item in history[:-1]:
        lines.append(f"User: {item['question']}\nAssistant: {clean_answer(item['answer'])}")
    last = history[-1]
    lines.append(
        f"\nLast conversation:\n"
        f"User: {last['question']}\nAssistant: {clean_answer(last['answer'])}\n\n"
        f" (this is the continuation of the conversation)\n\n"
        f"User asked: {request.message}"
    )
    return "\n\n".join(lines)


# ─── Memory Engage Stream ─────────────────────────────────────────

async def generate_engage_stream(context: str) -> AsyncGenerator[str, None]:
    fallback = "Where would you like to travel next?"
    start_time = time.time()
    yield f"data: {json.dumps({'start_time': start_time, 'status': 'started'})}\n\n"

    if not context:
        yield f"data: {json.dumps({'done': True, 'total_time': time.time() - start_time})}\n\n"
        return

    try:
        stream = await _openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are HipTraveler, a professional AI travel assistant re-engaging a returning user. "
                        "The user's last topic is marked with [LAST_TOPIC] in the context — your message MUST be about that topic. "
                        "CRITICAL: Do NOT include '[LAST_TOPIC]' or any tags in your response. Output only the message text. "
                        "STYLE: Warm, professional, and concise — like a knowledgeable travel concierge checking in. "
                        "- Max 12 words. No filler words, no 'I can help', no 'let me'. "
                        "- Return ONLY the message text, nothing else. "
                        "Examples of the right tone: "
                        "'Still planning your Paris trip?' / "
                        "'Hope your Paris trip is coming together!' / "
                        "'Ready to continue exploring Paris?' / "
                        "'Still looking for restaurants in Paris?'"
                    ),
                },
                {"role": "user", "content": f"User travel history:\n{context}"},
            ],
            max_tokens=40,
            temperature=0.7,
            stream=True,
        )
        ttfb_sent = False
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                if not ttfb_sent:
                    ttfb_sent = True
                    yield f"data: {json.dumps({'time_to_first_byte': time.time() - start_time})}\n\n"
                    yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n\n"
                yield f"data: {json.dumps({'content': json.dumps(token)[1:-1]})}\n\n"
        yield f"data: {json.dumps({'content': '\"}'})}\n\n"
    except Exception as e:
        pass
        yield f"data: {json.dumps({'content': fallback})}\n\n"

    yield f"data: {json.dumps({'done': True, 'total_time': time.time() - start_time})}\n\n"


