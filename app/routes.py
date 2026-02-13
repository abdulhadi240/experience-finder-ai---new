import os
import json
import time
from typing import AsyncGenerator, Optional
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from agents import Runner

from .schemas import QueryRequest, UserCreateRequest
from .services import generate_stream, get_complete_response, get_complete_response_explore
from .agents_ import validation_agent, explore_travel_agent
from .memory import delete_user, create_new_user
from .memory import check_user

logger = logging.getLogger("chat.redis")
# If you don't have logging configured elsewhere, this helps during local dev:
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ✅ Redis history helpers main
# from .redis_history import (
#     redis_enabled,
#     get_or_create_conversation_id,
#     fetch_recent_interactions,
#     append_interaction,
# )

# testing - hardcoded values
from .redis_history_local import (
	get_redis,
	redis_enabled,
    get_or_create_conversation_id,
    fetch_recent_interactions,
    append_interaction,
)

OLD_INTERACTIONS_LIMIT = int(os.getenv("REDIS_OLD_INTERACTIONS_LIMIT", "3"))
OLD_INTERACTIONS_MAX   = int(os.getenv("REDIS_OLD_INTERACTIONS_MAX", "30"))
REDIS_TTL              = int(os.getenv("REDIS_TTL_SECONDS", "3600"))
IDLE_CUTOFF_SECONDS    = int(os.getenv("REDIS_IDLE_CUTOFF_SECONDS", "1200"))

router = APIRouter()


@router.post("/chat")
async def unified_chat(request: QueryRequest):
    try:
        thread_id = check_user(request.user_id)
        param = request.param

        # ✅ per-user conversation separation (idle window or explicit request.conversation_id)
        conversation_id: Optional[str] = None

        try:
            enabled = redis_enabled()
            logger.info(
                "Redis enabled check: %s (user_id=%s, request_conversation_id=%s)",
                enabled,
                request.user_id,
                getattr(request, "conversation_id", None),
            )

            if enabled:
                # Optional: ping Redis to ensure connectivity
                try:
                    r = await get_redis()  # import get_redis from your redis module
                    pong = await r.ping()
                    logger.info("Redis ping ok: %s", pong)
                except Exception as ping_err:
                    logger.exception(
                        "Redis ping failed; will continue without Redis. err=%s",
                        ping_err,
                    )
                    enabled = False

            if enabled:
                conversation_id = await get_or_create_conversation_id(
                    user_id=str(request.user_id),
                    idle_cutoff_seconds=IDLE_CUTOFF_SECONDS,
                    ttl_seconds=REDIS_TTL,
                    explicit_conversation_id=getattr(request, "conversation_id", None),
                )
                logger.info("Redis conversation_id selected: %s", conversation_id)
            else:
                logger.info("Redis disabled for this request; using request-only context.")

        except Exception as e:
            logger.exception(
                "Unexpected Redis init error; proceeding without Redis. err=%s",
                e,
            )
            conversation_id = None

        final_message_with_current = await build_conversation_context(
            request,
            conversation_id=conversation_id,
        )

        logger.info(
            "Context built (conversation_id=%s, context_len=%d)",
            conversation_id,
            len(final_message_with_current or ""),
        )

        validation_result = await Runner.run(validation_agent, final_message_with_current)

        if not validation_result.final_output.isValid:
            return get_error_stream_response(
                validation_result.final_output.reason,
                validation_result.final_output.solution,
            )

        if validation_result.final_output.isTravelRelated:
            response_content = await get_complete_response(
                final_message_with_current, thread_id, param
            )

            # ✅ save Q/A to Redis (non-streaming path)
            try:
                assistant_answer = _extract_assistant_text(response_content)
                if conversation_id and assistant_answer:
                    await append_interaction(
                        user_id=str(request.user_id),
                        conversation_id=conversation_id,
                        question=request.message,
                        answer=assistant_answer,
                        max_items=OLD_INTERACTIONS_MAX,
                        ttl_seconds=REDIS_TTL,
                    )
            except Exception as e:
                logger.exception("Redis append_interaction failed: %s", e)

            return JSONResponse(
                content={
                    "response": jsonable_encoder(response_content),
                    "type": "non-streaming",
                    "conversation_id": conversation_id,
                }
            )

        # ✅ streaming path — pass conversation_id so generate_stream can persist assistant answer
        agent = "general_agent" if param == "plan" else "explore_agent"
        return StreamingResponse(
            generate_stream(
                message=request.message,
                thread_id=thread_id,
                reference=request.reference,
                agent=agent,
                final_message=final_message_with_current,
                user_id=str(request.user_id),
                conversation_id=conversation_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def clean_answer(answer: str) -> str:
    return answer.split('$$$$$')[0].strip()


async def build_conversation_context(request: QueryRequest, conversation_id: Optional[str]) -> str:
    interactions = []
    if redis_enabled() and conversation_id:
        try:
            interactions = await fetch_recent_interactions(
                user_id=str(request.user_id),
                conversation_id=conversation_id,
                limit=OLD_INTERACTIONS_LIMIT
            )
        except Exception as e:
            print("Redis fetch_recent_interactions failed:", str(e))

    if not interactions:
        return request.message

    blocks = []
    for item in interactions:
        q = (item.get("question") or "").strip()
        a = clean_answer((item.get("answer") or "").strip())
        if q and a:
            blocks.append(f"User: {q}\nAssistant: {a}")
        elif q:
            blocks.append(f"User: {q}")

    if not blocks:
        return request.message

    history_text = "\n\n".join(blocks)
    return (
        f"Previous conversations:\n{history_text}\n\n"
        f"(this is the continuation of the conversation)\n\n"
        f"User asked: {request.message}"
    )


def _extract_assistant_text(response_content) -> Optional[str]:
    if response_content is None:
        return None
    if isinstance(response_content, dict):
        for k in ("answer", "response", "content", "message"):
            v = response_content.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None
    for attr in ("answer", "response", "content", "message"):
        v = getattr(response_content, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def get_error_stream_response(reason: str, solution: str):
    """Generate a streaming error response for invalid/non-travel queries."""
    async def error_stream_generator() -> AsyncGenerator[str, None]:
        start_time = time.time()
        yield f"data: {json.dumps({'start_time': start_time, 'status': 'started'})}\n\n"

        first_chunk_time = time.time()
        ttfb = first_chunk_time - start_time
        yield f"data: {json.dumps({'time_to_first_byte': ttfb})}\n\n"

        chunks = [
            '{"', "answer", '":"',
            "Let", "'s", " keep", " it", " travel", "-focused", " ✨", ".\n\n",
            "I", " can", " help", " you", " explore", " destinations", ",",
            " discover", " experiences", ",", " and", " plan", " your", " trip", ".\n\n",
            "What", " would", " you", " like", " to", " explore", " next", "?",
            '"}', ""
        ]

        for chunk in chunks:
            yield f"data: {json.dumps({'content': chunk})}\n\n"

        end_time = time.time()
        yield f"data: {json.dumps({'done': True, 'total_time': end_time - start_time, 'blocked': True})}\n\n"

    return StreamingResponse(
        error_stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


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
            last_name=user.last_name
        )
        return {"message": "User created successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "agent-streaming-api"}
