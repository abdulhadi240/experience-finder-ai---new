import asyncio
import json
import time
import uuid
import httpx
from typing import AsyncGenerator, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from agents import Runner

from .schemas import QueryRequest, UserCreateRequest
from .services import (
    get_complete_response,
    stream_agent_to_queue,
    stream_starter_to_queue,
    summarize_for_rag,
    check_pii_fast,
)
from .agents_ import validation_agent
from .memory import delete_user, create_new_user, setup_user_session, add_message

router = APIRouter()


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
        print("RAG Response:", response.json())
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
) -> AsyncGenerator[str, None]:
    start_time = time.time()

    # ── t=0: client gets [STARTED] before any network calls ──────
    yield f"data: {json.dumps({'start_time': start_time, 'status': 'started', 'threadId': thread_id})}\n\n"

    # ── Fire starter + light PII check simultaneously ────────────
    starter_queue = asyncio.Queue()
    asyncio.create_task(stream_starter_to_queue(request.message, param, starter_queue))
    pii_task = asyncio.create_task(check_pii_fast(request.message))
    await asyncio.sleep(0)   # yield once so both HTTP calls go in-flight immediately

    # ── Fire remaining tasks in parallel ─────────────────────────
    async def _summarize_then_rag() -> Dict[str, Any]:
        query = await summarize_for_rag(final_message)
        return await rag(query=query, reference=request.reference)

    zep_task        = asyncio.create_task(asyncio.to_thread(setup_user_session, request.user_id, thread_id))
    rag_task        = asyncio.create_task(_summarize_then_rag())
    validation_task = asyncio.create_task(Runner.run(validation_agent, final_message))

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
        yield f"data: {json.dumps({'content': token})}\n\n"
        await asyncio.sleep(0.10)   # throttle starter so main agent is ready by the time it ends

    # ── Separator: blank line between starter paragraph and agent list ──
    yield f"data: {json.dumps({'content': '\n\n'})}\n\n"

    # ── Await both tasks (likely already done while starter was streaming) ──
    try:
        rag_result = await rag_task
    except Exception as e:
        rag_result = e

    try:
        validation_result = await validation_task
    except Exception as e:
        validation_result = e

    # ── RAG check ────────────────────────────────────────────────
    note = ""
    if isinstance(rag_result, Exception):
        print(f"RAG failed, falling back: {rag_result}")
    else:
        note = rag_result.get("note", "")

    if "This portal is scoped to" in note and "Please ask about that destination" in note:
        yield f"data: {json.dumps({'time_to_first_byte': time.time() - start_time})}\n\n"
        for i, word in enumerate(note.split()):
            yield f"data: {json.dumps({'content': word if i == 0 else ' ' + word})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total_time': time.time() - start_time, 'threadId': thread_id, 'param': param, 'response_type': 'rag_note'})}\n\n"
        return

    # ── Validation check ─────────────────────────────────────────
    if isinstance(validation_result, Exception):
        yield f"data: {json.dumps({'error': str(validation_result)})}\n\n"
        return

    print("validation_result:", validation_result.final_output)

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

    # ── Ensure Zep session is ready before agent calls add_message ──
    await zep_task

    # ── Trip planning (isTravelRelated=True) ─────────────────────
    if validation_result.final_output.isTravelRelated:
        try:
            response_content, timing_info = await get_complete_response(final_message, thread_id, param)
            yield f"data: {json.dumps({'travel': [jsonable_encoder(response_content), jsonable_encoder(timing_info)], 'type': 'non-streaming', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return

    # ── Explore / General (isTravelRelated=False) ─────────────────
    if rag_data:
        final_message_with_ref += f"\n\n[RAG_RESULTS]\n{json.dumps(rag_data)}\n[/RAG_RESULTS]"

    final_message_with_ref += "\n\n[INSTRUCTION] Begin your response with one short natural sentence that introduces the recommendations (e.g. 'Here are the best things to do in Tokyo:' or 'A few great spots to check out in Rome:'). Make it specific to the query. Then continue with your list. [/INSTRUCTION]"

    # chunks present = RAG has real content → format agent; no chunks = fall back to web
    agent_name = "rag_format_agent" if rag_chunks else "web_search_agent"

    token_queue = asyncio.Queue()
    asyncio.create_task(
        stream_agent_to_queue(
            agent_name=agent_name,
            final_message_with_ref=final_message_with_ref,
            original_message=request.message,
            thread_id=thread_id,
            queue=token_queue,
        )
    )

    # ── Phase 2: stream main agent tokens ────────────────────────
    while True:
        token = await token_queue.get()
        if token is None:
            break
        if isinstance(token, Exception):
            yield f"data: {json.dumps({'error': str(token)})}\n\n"
            break
        yield f"data: {json.dumps({'content': token})}\n\n"

    yield f"data: {json.dumps({'content': '\"}'})}\n\n"
    yield f"data: {json.dumps({'done': True, 'total_time': time.time() - start_time, 'threadId': thread_id, 'param': param})}\n\n"


# ─── Main Chat Route ──────────────────────────────────────────────

@router.post("/chat")
async def unified_chat(request: QueryRequest):
    try:
        thread_id     = uuid.uuid4().hex          # instant — no network call
        param         = request.param
        final_message = build_conversation_context(request)

        return StreamingResponse(
            _main_stream(request, thread_id, param, final_message),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Helper Functions ─────────────────────────────────────────────

def clean_answer(answer: str) -> str:
    return answer.split("$$$$$")[0].strip()


def build_conversation_context(request: QueryRequest) -> str:
    if not request.old_interactions:
        return request.message

    recent = request.old_interactions[:3]

    if len(recent) >= 3:
        old      = recent[2]
        previous = recent[1]
        last     = recent[0]
        return (
            f"Previous conversations:\n"
            f"User: {old.question}\nAssistant: {clean_answer(old.answer)}\n\n"
            f"User: {previous.question}\nAssistant: {clean_answer(previous.answer)}\n\n"
            f"Last conversation:\n"
            f"User: {last.question}\nAssistant: {clean_answer(last.answer)}\n\n"
            f" (this is the continuation of the conversation)\n\n"
            f"User asked: {request.message}"
        )
    elif len(recent) == 2:
        previous = recent[1]
        last     = recent[0]
        return (
            f"Previous conversation:\n"
            f"User: {previous.question}\nAssistant: {clean_answer(previous.answer)}\n\n"
            f"Last conversation:\n"
            f"User: {last.question}\nAssistant: {clean_answer(last.answer)}\n\n"
            f" (this is the continuation of the conversation)\n\n"
            f"User asked: {request.message}"
        )
    elif len(recent) == 1:
        last = recent[0]
        return (
            f"Last conversation (this is the continuation of the conversation):\n"
            f"User: {last.question}\nAssistant: {clean_answer(last.answer)}\n\n"
            f"New question asked: {request.message}"
        )

    return request.message


# ─── User Management Routes ───────────────────────────────────────

@router.get("/delete_user")
async def delete_user_route(user_id: int = Query(..., description="The ID of the user to delete")):
    try:
        result = delete_user(user_id)
        return {"message": f"User {user_id} deleted successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/create_user")
async def create_user_route(user: UserCreateRequest):
    try:
        result = create_new_user(
            user_id=user.user_id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
        )
        return {"message": "User created successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agent-streaming-api"}
