import asyncio
import json
import time
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
    generate_loading_statements,
    get_loading_message,
    _STAGE_TIMINGS_MS,
)
from .agents_ import validation_agent
from .memory import delete_user, create_new_user, check_user, add_message

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


# ─── Streaming-with-loading generator (isTravelRelated=False only) ──

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

    Two concurrent tasks start immediately:
      • statements_task — gpt-4.1-nano generates 6 personalised loading messages
                          from the full conversation context (history + destination).
      • stream_agent_to_queue — agent tokens land in token_queue as they arrive.

    Loading stages fire on a timer until the FIRST token arrives in the queue,
    at which point loading stops immediately and tokens stream live (≤50 ms delay).

    statements[0-2] used at stages 0-2 (0.5 s, 3 s, 6 s)
    statements[3-5] used at stages 3-5 (10 s, 15 s, 20 s)
    Static fallback pool used if LLM call not ready in time.
    """
    start_time = time.time()

    yield f"data: {json.dumps({'start_time': start_time, 'status': 'started', 'threadId': thread_id})}\n\n"

    # Both tasks start at t=0 — zero added latency
    statements_task = asyncio.create_task(generate_loading_statements(context, param))

    token_queue = asyncio.Queue()
    asyncio.create_task(
        stream_agent_to_queue(
            agent_name=agent_name,
            final_message_with_ref=final_message_with_ref,
            original_message=original_message,
            thread_id=thread_id,
            queue=token_queue,
        )
    )

    stage_index = 0
    loop_start  = time.time()
    statements  = []
    first_token = False

    while True:
        # Non-blocking poll for next agent token
        try:
            token = token_queue.get_nowait()

            if token is None:                       # sentinel — stream complete
                break
            if isinstance(token, Exception):
                yield f"data: {json.dumps({'error': str(token)})}\n\n"
                break

            # First token → interrupt loading immediately
            if not first_token:
                first_token = True
                ttfb = time.time() - start_time
                yield f"data: {json.dumps({'time_to_first_byte': ttfb})}\n\n"

            yield f"data: {json.dumps({'content': token})}\n\n"

        except asyncio.QueueEmpty:
            # No token yet — advance loading stages while waiting
            if not first_token:
                elapsed_ms = (time.time() - loop_start) * 1000

                if stage_index < len(_STAGE_TIMINGS_MS) and elapsed_ms >= _STAGE_TIMINGS_MS[stage_index]:
                    if not statements and statements_task.done():
                        try:
                            statements = statements_task.result()
                        except Exception:
                            statements = []

                    msg = (
                        statements[stage_index]
                        if statements and stage_index < len(statements)
                        else get_loading_message(stage_index, None, param)
                    )
                    yield f"data: {json.dumps({'type': 'loading', 'message': msg})}\n\n"
                    stage_index += 1

            await asyncio.sleep(0.05)

    if not statements_task.done():
        statements_task.cancel()

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


# ─── Main Chat Route ──────────────────────────────────────────────

@router.post("/chat")
async def unified_chat(request: QueryRequest):
    try:
        thread_id     = check_user(request.user_id)
        param         = request.param
        final_message = build_conversation_context(request)

        # RAG
        try:
            rag_response = await rag(query=request.message, reference=request.reference)
            note         = rag_response.get("note", "")
        except Exception as e:
            print(f"RAG failed, falling back: {e}")
            note = ""

        is_scoped_note = (
            "This portal is scoped to" in note
            and "Please ask about that destination" in note
        )
        if is_scoped_note:
            return get_rag_note_stream_response(note, thread_id, param)

        # Validation
        print("validation")
        validation_result = await Runner.run(validation_agent, final_message)
        print("validation_result:", validation_result.final_output)

        if not validation_result.final_output.isValid:
            return get_error_stream_response(
                validation_result.final_output.reason,
                validation_result.final_output.solution,
            )

        final_message_with_ref = final_message + "\n\nReference : " + request.reference

        # isTravelRelated=True → original JSON response, no loading messages
        if validation_result.final_output.isTravelRelated:
            response_content, timing_info = await get_complete_response(
                final_message, thread_id, param
            )
            
            # This is where we combine them into the array you want
            return JSONResponse(content={
                "response": [
                    jsonable_encoder(response_content),
                    jsonable_encoder(timing_info)
                ],
                "type": "non-streaming",
            })
        # isTravelRelated=False → streaming with loading messages
        agent_name = "general_agent" if param == "plan" else "explore_agent"

        return StreamingResponse(
            streaming_with_loading(
                context=final_message,
                agent_name=agent_name,
                final_message_with_ref=final_message_with_ref,
                original_message=request.message,
                thread_id=thread_id,
                param=param,
            ),
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
