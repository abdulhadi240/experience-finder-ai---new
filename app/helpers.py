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
    top_k: int = 10,
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
    top_k: int = 10,
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

    # ── Fire ALL tasks at t=0 — RAG, validation, PII, loaders, scope, Zep all in parallel ──
    # scope gate runs in a thread so pycountry I/O never blocks the event loop
    scope_task    = asyncio.create_task(asyncio.to_thread(_check_location_scope, request.message, request.reference))
    pii_task      = asyncio.create_task(check_pii_fast(request.message))
    # Loaders are always explore-style here: this streaming flow finds/shows
    # recommendations, it does NOT build an itinerary — never use plan/itinerary wording.
    loading_task  = asyncio.create_task(generate_loading_statements(final_message, "explore"))

    async def _summarize_then_fetch() -> tuple[Any, Any]:
        # One nano call parses the message into a full structured payload
        # (query/category/top_k/location/filters); it then feeds both retrieval
        # endpoints, fired in parallel. Same single nano round-trip as before.
        p = await build_rag_payload(
            final_message,
            fallback_location=request.location or "",
            fallback_filters=request.filters or "",
        )
        rag_res, guide_res = await asyncio.gather(
            rag(query=p["query"], reference=request.reference, thread_id=thread_id,
                location=p["location"], filters=p["filters"],
                top_k=p["top_k"], category=p["category"]),
            rag_guide(query=p["query"], reference=request.reference, thread_id=thread_id,
                      location=p["location"], filters=p["filters"]),
            return_exceptions=True,
        )
        return rag_res, guide_res

    rag_task        = asyncio.create_task(_summarize_then_fetch())
    validation_task = asyncio.create_task(Runner.run(validation_agent, final_message))
    await asyncio.sleep(0)   # yield so all tasks go in-flight simultaneously

    # ── TTFB + instant first loader fire IMMEDIATELY ─────────────────
    _instant_loader = get_instant_loading_message("explore")
    _shown_loaders: list[str] = [_instant_loader]
    yield f"data: {json.dumps({'time_to_first_byte': time.time() - start_time})}\n\n"
    yield f"data: {json.dumps({'loading': _instant_loader})}\n\n"

    # ── Pre-fetch LLM loaders while rag+validation run in background ─
    # Await up to 1.3s for the nano call. rag_task + validation_task keep
    # running regardless — we're just choosing when to check them.
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
            # Last-chance fetch if nano call wasn't ready after 1.3s
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

    # Both done — extract results. rag_task returns (rag_result, guide_result),
    # each of which may itself be an Exception (gather return_exceptions=True).
    try:
        rag_result, guide_result = rag_task.result()
    except Exception as e:
        rag_result = e
        guide_result = e

    try:
        validation_result = validation_task.result()
    except Exception as e:
        validation_result = e

    # ── RAG result ───────────────────────────────────────────────
    if isinstance(rag_result, Exception):
        _log.warning("[RAG] failed: %s — falling back to web_search_agent", rag_result)
    else:
        _log.info("[RAG] returned %d chunk(s)", len(rag_result.get("chunks") or []))

    # ── Destination-guide result ─────────────────────────────────
    if isinstance(guide_result, Exception):
        _log.warning("[GUIDE] failed: %s", guide_result)
    else:
        _log.info("[GUIDE] returned guide payload (%d keys)", len(guide_result or {}))

    # ── Validation check ─────────────────────────────────────────
    if isinstance(validation_result, Exception):
        yield f"data: {json.dumps({'error': str(validation_result)})}\n\n"
        return


    if not validation_result.final_output.isValid:
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
    if not isinstance(rag_result, Exception):
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
    if validation_result.final_output.isTravelRelated:
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
    is_realtime = getattr(validation_result.final_output, "isRealtime", None)
    if is_realtime is None:
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
    if not is_realtime and not isinstance(guide_result, Exception) and guide_result:
        final_message_with_ref += (
            f"\n\n[DESTINATION_GUIDE]\n"
            f"Editorial destination-guide content retrieved for this query. "
            f"You MUST include this guide in your answer body, in the SAME format/voice as the rest of the answer:\n"
            f"  • Open with a short guide-driven intro section at the TOP (1-2 sentences of the most useful context/framing from the guide), BEFORE the main recommendations.\n"
            f"  • Close with a short guide-driven section at the BOTTOM (a practical tip / local insight from the guide), AFTER the recommendations and BEFORE the closing question.\n"
            f"Weave it naturally — do not label it 'Destination Guide', do not dump raw JSON, and never attach place ids from here (it has none).\n"
            f"{json.dumps(guide_result)}\n"
            f"[/DESTINATION_GUIDE]"
        )

    # Loading context: agent opens in sync with the last loader the user saw
    _loaders_block = "\n".join(f"- {m}" for m in _shown_loaders)
    final_message_with_ref += (
        "\n\n[LOADING_CONTEXT]\n"
        f"While the user waited, they saw these loading messages:\n{_loaders_block}\n"
        f"Last message shown: \"{_shown_loaders[-1]}\"\n"
        "Your opening sentence must feel like a natural continuation of that last message "
        "as if you are picking up exactly where it left off. Do NOT repeat it word-for-word; "
        "flow directly into the answer content.\n"
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
    if validation_result.final_output.isMemoryQuery:
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
        validation_result.final_output.isMemoryQuery,
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
