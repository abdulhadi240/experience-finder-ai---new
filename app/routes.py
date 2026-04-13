import asyncio
import os
import uuid

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from .schemas import QueryRequest, UserCreateRequest
from .memory import delete_user, create_new_user, get_user_memory_for_engage
from .helpers import build_conversation_context, _main_stream, generate_engage_stream
from . import redis_history

router = APIRouter()

_REDIS_TTL          = int(os.getenv("REDIS_TTL_SECONDS", "600"))
_REDIS_IDLE_CUTOFF  = int(os.getenv("REDIS_IDLE_CUTOFF_SECONDS", "600"))
_REDIS_CTX_LIMIT    = int(os.getenv("REDIS_OLD_INTERACTIONS_LIMIT", "5"))


# ─── Main Chat Route ──────────────────────────────────────────────

@router.post("/chat")
async def unified_chat(request: QueryRequest):
    try:
        thread_id = uuid.uuid4().hex
        param     = request.param

        # ── Resolve Redis conversation ID ─────────────────────────
        # Uses request.threadId when the frontend tracks the session,
        # otherwise falls back to idle-cutoff logic in redis_history.
        conversation_id = await redis_history.get_or_create_conversation_id(
            request.user_id,
            idle_cutoff_seconds=_REDIS_IDLE_CUTOFF,
            ttl_seconds=_REDIS_TTL,
            explicit_conversation_id=request.threadId or None,
        )

        # ── Load conversation history from Redis, fallback to old_interactions ──
        history: list[dict] = []
        if conversation_id:
            history = await redis_history.fetch_recent_interactions(
                request.user_id, conversation_id, limit=_REDIS_CTX_LIMIT
            )

        # Fallback: if Redis returned nothing, use old_interactions from frontend
        if not history and request.old_interactions:
            history = [
                {"question": item.question, "answer": item.answer}
                for item in request.old_interactions[-_REDIS_CTX_LIMIT:]
            ]

        final_message = build_conversation_context(request, history)

        return StreamingResponse(
            _main_stream(request, thread_id, param, final_message, conversation_id=conversation_id, history=history),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# ─── Memory Re-engagement Endpoint ───────────────────────────────

@router.get("/memory/engage")
async def memory_engage(user_id: str = Query(..., description="The user's ID")):
    """
    Stream a personalised re-engagement question token by token.
    Fetches Zep user summary, then streams gpt-4.1-nano output as SSE.
    """
    context = await asyncio.to_thread(get_user_memory_for_engage, user_id)
    return StreamingResponse(generate_engage_stream(context), media_type="text/event-stream")
