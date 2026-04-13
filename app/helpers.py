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
    stream_starter_to_queue,
    summarize_for_rag,
    check_pii_fast,
)
from .agents_ import validation_agent
from .memory import setup_user_session, add_message, get_user_memory_for_engage, get_user_preferences
from .region_metadata import REGION_METADATA
from . import redis_history as _redis_history
import os
import logging

_log = logging.getLogger("app.agent")

_REDIS_TTL      = int(os.getenv("REDIS_TTL_SECONDS", "120"))
_REDIS_MAX      = int(os.getenv("REDIS_OLD_INTERACTIONS_MAX", "30"))


# ─── Real-time query detector ─────────────────────────────────────
# Routes queries that need live web data to web_search_agent.
_REALTIME_SIGNALS = [
    # simple time-now phrases (catches "Dubai right now", "what to do right now", etc.)
    "right now", "right now?",
    "currently", "current conditions", "at the moment", "these days", "nowadays",
    # safety / news
    "is it safe", "is safe", "safe to travel", "travel safe", "safety",
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


async def rag(query: str, reference: str) -> Dict[str, Any]:
    """Fully async RAG call via shared httpx client."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or None")

    payload = {"query": query.strip(), "reference": reference}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    try:
        response = await _rag_client.post(
            url="https://rag.hiptraveler.com/chat",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        pass
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"success": True, "data": response.text, "status_code": response.status_code}

    except httpx.TimeoutException:
        raise Exception("RAG request timed out after 30 seconds")
    except httpx.ConnectError:
        raise Exception("Failed to connect to RAG webhook")
    except httpx.HTTPStatusError as e:
        raise Exception(f"RAG HTTP error: {e}")


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
    # Prefer $$$$$ metadata block (RAG) — "name" values are authoritative
    pois: list[str] = []
    answer = last.get("answer", "")
    if "$$$$$" in answer:
        parts = answer.split("$$$$$")
        if len(parts) >= 2:
            metadata_block = parts[1]
            pois = re.findall(r'"name"\s*:\s*"([^"]+)"', metadata_block)
    if not pois:
        # web_search_agent — no metadata block, extract **Bold Place Names** from text
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
    conversation_id: Optional[str] = None,
    history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    start_time = time.time()

    # ── t=0: client gets [STARTED] before any network calls ──────
    yield f"data: {json.dumps({'start_time': start_time, 'status': 'started', 'threadId': thread_id})}\n\n"

    # ── Instant location scope gate (pure Python, no LLM, no I/O) ──
    scope_error = _check_location_scope(request.message, request.reference)
    if scope_error:
        yield f"data: {json.dumps({'time_to_first_byte': time.time() - start_time})}\n\n"
        yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n\n"
        for i, word in enumerate(scope_error.split()):
            yield f"data: {json.dumps({'content': word if i == 0 else ' ' + word})}\n\n"
        yield f"data: {json.dumps({'content': '\"}'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total_time': time.time() - start_time, 'threadId': thread_id, 'param': param, 'response_type': 'scope_gate'})}\n\n"
        return

    # ── Save user question to Zep — always, non-blocking, never gates anything ──
    if request.user_id:
        async def _save_to_zep():
            try:
                await asyncio.to_thread(setup_user_session, request.user_id, thread_id)
                await asyncio.to_thread(add_message, role='user', thread_id=thread_id, message=request.message)
            except Exception as e:
                pass
        asyncio.create_task(_save_to_zep())

    # ── Fast-path: plan queries skip all middleware ───────────────
    if request.plan:
        try:
            ctx_str, ctx_pois = _extract_explore_context(history)
            enriched = final_message + ctx_str
            response_content, timing_info = await get_complete_response(enriched, thread_id, param)
            if ctx_pois and not response_content.pois:
                response_content.pois = ctx_pois
            yield f"data: {json.dumps({'travel': [jsonable_encoder(response_content), jsonable_encoder(timing_info)], 'type': 'non-streaming', 'done': True})}\n\n"
            # ── Clear or save to Redis ──
            if request.user_id and conversation_id:
                if response_content.feedback is None:
                    asyncio.create_task(_redis_history.clear_conversation(
                        request.user_id, conversation_id,
                    ))
                elif response_content.summary:
                    asyncio.create_task(_redis_history.append_interaction(
                        request.user_id, conversation_id,
                        question=request.message,
                        answer=json.dumps(jsonable_encoder(response_content)),
                        max_items=_REDIS_MAX, ttl_seconds=_REDIS_TTL,
                    ))
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return

    # ── Fire starter + light PII check simultaneously ────────────
    starter_queue = asyncio.Queue()
    asyncio.create_task(stream_starter_to_queue(final_message, param, starter_queue))
    pii_task = asyncio.create_task(check_pii_fast(request.message))
    await asyncio.sleep(0)   # yield once so both HTTP calls go in-flight immediately

    # ── Fire remaining tasks in parallel ─────────────────────────
    async def _summarize_then_rag() -> Dict[str, Any]:
        query = await summarize_for_rag(final_message)
        return await rag(query=query, reference=request.reference)

    rag_task        = asyncio.create_task(_summarize_then_rag())
    validation_task = asyncio.create_task(Runner.run(validation_agent, final_message))
    zep_prefs_task  = asyncio.create_task(asyncio.to_thread(get_user_preferences, request.user_id)) if request.user_id else None

    # ── PII check — resolves in ~150-200ms, starter first token ~300ms ──
    # Awaiting here adds zero real latency since starter hasn't produced a token yet
    if await pii_task:
        pii_message = (
            "To keep your information safe, please avoid sharing personal details "
            "like phone numbers, email addresses, or ID numbers in your messages. "
            "Feel free to ask me anything about travel destinations and I'll be happy to help!"
        )
        yield f"data: {json.dumps({'time_to_first_byte': time.time() - start_time})}\n\n"
        yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n\n"
        for i, word in enumerate(pii_message.split()):
            yield f"data: {json.dumps({'content': word if i == 0 else ' ' + word})}\n\n"
        yield f"data: {json.dumps({'content': '\"}'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'blocked': True, 'total_time': time.time() - start_time, 'reason': 'PII_DETECTED'})}\n\n"
        return

    # ── Phase 1: stream starter tokens immediately ────────────────
    ttfb_sent = False
    while True:
        token = await starter_queue.get()
        if token is None:
            break
        if not ttfb_sent:
            ttfb_sent = True
            yield f"data: {json.dumps({'time_to_first_byte': time.time() - start_time})}\n\n"
            yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n\n"
        # JSON-string-encode the token so the frontend can accumulate all chunks
        # into a valid JSON string and call JSON.parse() at the end.
        # json.dumps(token)[1:-1] gives the JSON escape sequences without outer quotes
        # e.g. "it's" → it's  |  '"hello"' → \"hello\"  |  '\n' → \n
        yield f"data: {json.dumps({'content': json.dumps(token)[1:-1]})}\n\n"
        await asyncio.sleep(0.18)   # throttle starter so main agent is ready by the time it ends

    # ── Separator: use JSON-string newline escapes, not literal newlines ──
    # Literal '\n\n' would break JSON.parse on the frontend.
    # '\\n\\n' (Python: backslash-n x2) → wire: "\\n\\n" → frontend buffer: \n\n (valid JSON escapes)
    yield f"data: {json.dumps({'content': '\\n\\n'})}\n\n"

    # ── Await both tasks (likely already done while starter was streaming) ──
    try:
        rag_result = await rag_task
    except Exception as e:
        rag_result = e

    try:
        validation_result = await validation_task
    except Exception as e:
        validation_result = e

    # ── RAG result (scope note check moved to instant gate above) ──
    if isinstance(rag_result, Exception):
        pass

    # ── Validation check ─────────────────────────────────────────
    if isinstance(validation_result, Exception):
        yield f"data: {json.dumps({'error': str(validation_result)})}\n\n"
        return


    if not validation_result.final_output.isValid:
        solution = validation_result.final_output.solution
        if len(solution) < 50:
            solution = "Let's keep it travel-focused. What would you like to explore next?"
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

    final_message_with_ref = final_message + "\n\nReference : " + request.reference

    # ── Await Zep prefs (in-flight since t=0, inject into agent context) ──────
    zep_prefs = None
    if zep_prefs_task:
        try:
            zep_prefs = await zep_prefs_task
        except Exception:
            zep_prefs = None

    # ── Trip planning (isTravelRelated=True) ─────────────────────
    if validation_result.final_output.isTravelRelated:
        try:
            ctx_str, ctx_pois = _extract_explore_context(history)
            enriched = final_message + ctx_str
            response_content, timing_info = await get_complete_response(enriched, thread_id, param)
            if ctx_pois and not response_content.pois:
                response_content.pois = ctx_pois
            yield f"data: {json.dumps({'travel': [jsonable_encoder(response_content), jsonable_encoder(timing_info)], 'type': 'non-streaming', 'done': True})}\n\n"
            # ── Clear or save to Redis ────────────────────────────
            if request.user_id and conversation_id:
                if response_content.feedback is None:
                    # Plan is complete (no more questions) → clear history in background
                    asyncio.create_task(_redis_history.clear_conversation(
                        request.user_id, conversation_id,
                    ))
                elif response_content.summary:
                    asyncio.create_task(_redis_history.append_interaction(
                        request.user_id, conversation_id,
                        question=request.message,
                        answer=json.dumps(jsonable_encoder(response_content)),
                        max_items=_REDIS_MAX, ttl_seconds=_REDIS_TTL,
                    ))
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return

    # ── Explore / General (isTravelRelated=False) ─────────────────
    is_realtime = _is_realtime_query(request.message)

    # Skip RAG injection for real-time queries — web_search_agent expects
    # no RAG data and its instruction says "RAG returned nothing".
    # Injecting irrelevant place chunks would cause it to format places
    # instead of fetching live information.
    if rag_data and not is_realtime:
        final_message_with_ref += f"\n\n[RAG_RESULTS]\n{json.dumps(rag_data)}\n[/RAG_RESULTS]"

    if zep_prefs:
        final_message_with_ref += f"\n\n[USER_PREFERENCES]\n{zep_prefs}\n[/USER_PREFERENCES]"

    final_message_with_ref += (
        "\n\n[INSTRUCTION] A lead-in has already been shown to the user. "
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

    # ── Phase 2: stream main agent tokens ────────────────────────
    redis_answer_parts: list[str] = []
    stream_error = False
    while True:
        token = await token_queue.get()
        if token is None:
            break
        if isinstance(token, Exception):
            yield f"data: {json.dumps({'error': str(token)})}\n\n"
            stream_error = True
            break
        yield f"data: {json.dumps({'content': token})}\n\n"
        redis_answer_parts.append(token)

    yield f"data: {json.dumps({'content': '\"}'})}\n\n"
    yield f"data: {json.dumps({'done': True, 'total_time': time.time() - start_time, 'threadId': thread_id, 'param': param})}\n\n"

    # ── Save explore Q&A to Redis (non-blocking, fire-and-forget) ─
    if request.user_id and conversation_id and redis_answer_parts and not stream_error:
        asyncio.create_task(_redis_history.append_interaction(
            request.user_id, conversation_id,
            question=request.message, answer=''.join(redis_answer_parts),
            max_items=_REDIS_MAX, ttl_seconds=_REDIS_TTL,
        ))


# ─── Conversation Context Helpers ─────────────────────────────────

def clean_answer(answer: str) -> str:
    return answer.split("$$$$$")[0].strip()


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
