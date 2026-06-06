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

_REDIS_TTL       = int(os.getenv("REDIS_TTL_SECONDS", "600"))
_REDIS_CTX_LIMIT = int(os.getenv("REDIS_OLD_INTERACTIONS_LIMIT", "20"))


# ─── Main Chat Route ──────────────────────────────────────────────

@router.post("/chat")
async def unified_chat(request: QueryRequest):
    try:
        # Reuse the thread_id the frontend sends, or create a new one for a new conversation
        thread_id = request.threadId or uuid.uuid4().hex
        param     = request.param

        # ── Load conversation history from Redis keyed by thread_id ──
        history: list[dict] = await redis_history.fetch_recent_interactions(
            thread_id, limit=_REDIS_CTX_LIMIT
        )

        # Fallback: if Redis returned nothing, use old_interactions from frontend
        if not history and request.old_interactions:
            history = [
                {"question": item.question, "answer": item.answer}
                for item in request.old_interactions[-_REDIS_CTX_LIMIT:]
            ]

        final_message = build_conversation_context(request, history)

        return StreamingResponse(
            _main_stream(request, thread_id, param, final_message, history=history),
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
