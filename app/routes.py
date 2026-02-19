import asyncio
import json
import time
import httpx
from typing import AsyncGenerator, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
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

# Module-level shared client — one connection pool reused across all requests.
# httpx.AsyncClient is fully async: no threads consumed, scales to thousands of
# concurrent requests without exhausting any thread pool.
_rag_client = httpx.AsyncClient(timeout=30.0)

async def rag(query: str, reference: str) -> Dict[str, Any]:
    """Async RAG call. Uses the shared httpx client — zero thread usage."""
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


# ─── Pipeline Decision (RAG + validation only, no agent call) ────

async def run_pipeline_decision(request: QueryRequest) -> dict:
    """
    Runs RAG (non-blocking) → validation → routing decision.

    For the streaming path it returns ROUTING INFO only — the agent is NOT
    called here.  The caller starts the agent separately so it can interrupt
    loading messages the moment the first token arrives.

    For non-streaming paths (trip planning, error, rag-note) the full result
    is produced here so loading covers the entire wait.
    """
    thread_id     = check_user(request.user_id)
    param         = request.param
    final_message = build_conversation_context(request)

    # RAG — fully async via httpx, no thread pool needed
    try:
        rag_response = await rag(query=request.message, reference=request.reference)
        note = rag_response.get("note", "")
    except Exception as e:
        print(f"RAG failed, falling back: {e}")
        note = ""

    # Scoped-note short-circuit
    if "This portal is scoped to" in note and "Please ask about that destination" in note:
        return {"type": "rag_note", "note": note, "thread_id": thread_id, "param": param}

    # Validation
    print("validation")
    validation_result = await Runner.run(validation_agent, final_message)
    print("validation_result:", validation_result.final_output)

    if not validation_result.final_output.isValid:
        return {
            "type": "error",
            "reason": validation_result.final_output.reason,
            "solution": validation_result.final_output.solution,
            "thread_id": thread_id,
            "param": param,
        }

    final_message_with_ref = final_message + "\n\nReference : " + request.reference

    if validation_result.final_output.isTravelRelated:
        # Trip planning: run the full agent here; loading covers the entire wait
        response_content, timing_info = await get_complete_response(final_message, thread_id, param)
        return {
            "type": "non_streaming",
            "data": jsonable_encoder(response_content),
            "thread_id": thread_id,
            "param": param,
            "timing_info": timing_info,
        }

    # Streaming path: return routing info only — agent starts in chat_stream_generator
    return {
        "type": "streaming",
        "agent_name": "general_agent" if param == "plan" else "explore_agent",
        "final_message_with_ref": final_message_with_ref,
        "thread_id": thread_id,
        "param": param,
    }


# ─── Unified SSE Generator ────────────────────────────────────────

async def chat_stream_generator(request: QueryRequest) -> AsyncGenerator[str, None]:

    start_time = time.time()
    param      = request.param

    # ── t=0: signal receipt — zero latency ──────────────────────
    yield f"data: {json.dumps({'start_time': start_time, 'status': 'started'})}\n\n"

    # ── Two concurrent tasks fire immediately ────────────────────
    statements_task = asyncio.create_task(generate_loading_statements(request.message, param))
    decision_task   = asyncio.create_task(run_pipeline_decision(request))

    stage_index = 0
    loop_start  = time.time()
    statements  = []   # filled once statements_task completes

    # ── PHASE 1: loading while pipeline decision runs ────────────
    while not decision_task.done():
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

        await asyncio.sleep(0.1)

    # ── Retrieve decision ────────────────────────────────────────
    try:
        result = decision_task.result()
    except Exception as e:
        if not statements_task.done():
            statements_task.cancel()
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total_time': time.time() - start_time})}\n\n"
        return

    r_type    = result["type"]
    thread_id = result.get("thread_id", "")

    # ── PHASE 2 (streaming path only) ───────────────────────────
    if r_type == "streaming":
        token_queue = asyncio.Queue()

        # Start agent — tokens land in queue as they arrive
        asyncio.create_task(
            stream_agent_to_queue(
                agent_name             = result["agent_name"],
                final_message_with_ref = result["final_message_with_ref"],
                original_message       = request.message,
                thread_id              = thread_id,
                queue                  = token_queue,
            )
        )

        first_token = False

        while True:
            # ── Non-blocking queue poll ──────────────────────────
            try:
                token = token_queue.get_nowait()

                if token is None:                       # sentinel — stream complete
                    break
                if isinstance(token, Exception):
                    yield f"data: {json.dumps({'error': str(token)})}\n\n"
                    break

                # First token arrived → interrupt loading immediately
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

                # 50 ms poll — keeps interrupt latency low
                await asyncio.sleep(0.05)

    # ── Non-streaming / error / rag-note responses ───────────────
    elif r_type == "rag_note":
        note = result["note"]
        yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n\n"
        for i, word in enumerate(note.split()):
            chunk = word if i == 0 else f" {word}"
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield f"data: {json.dumps({'content': '\"}'})}\n\n"

    elif r_type == "error":
        solution = result["solution"]
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

    elif r_type == "non_streaming":
        yield f"data: {json.dumps({'response': result['data'], 'type': 'non-streaming', 'threadId': thread_id, 'param': result['param']})}\n\n"

    # ── Cleanup + done event ─────────────────────────────────────
    if not statements_task.done():
        statements_task.cancel()

    end_time     = time.time()
    done_payload = {
        "done":          True,
        "total_time":    end_time - start_time,
        "threadId":      thread_id,
        "param":         param,
        "response_type": r_type,
    }
    if r_type == "error":
        done_payload["blocked"] = True
    if r_type == "non_streaming" and "timing_info" in result:
        done_payload.update(result["timing_info"])

    yield f"data: {json.dumps(done_payload)}\n\n"


# ─── Main Chat Route ──────────────────────────────────────────────

@router.post("/chat")
async def unified_chat(request: QueryRequest):
    return StreamingResponse(
        chat_stream_generator(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ─── Helper Functions ─────────────────────────────────────────────

def clean_answer(answer: str) -> str:
    """Remove POI metadata block (between $$$$$) from answer."""
    return answer.split("$$$$$")[0].strip()


def build_conversation_context(request: QueryRequest) -> str:
    """Build the final message with conversation history (max 3)."""
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
