# for testing purposes

import json
import time
import uuid
from typing import Dict, List, Optional

from redis.asyncio import Redis

# ✅ HARD-CODED LOCAL SETTINGS (TEMP)
REDIS_URL = "redis://localhost:6379/0"
REDIS_TTL_SECONDS = 3600
REDIS_IDLE_CUTOFF_SECONDS = 10 #20 * 60   # 20 minutes
REDIS_OLD_INTERACTIONS_LIMIT = 3
REDIS_OLD_INTERACTIONS_MAX = 30

_redis: Optional[Redis] = None


def redis_enabled() -> bool:
    return True


async def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
            retry_on_timeout=True,
        )
    return _redis


def _key_active_conv(user_id: str) -> str:
    return f"chat:active_conv:{user_id}"


def _key_interactions(user_id: str, conversation_id: str) -> str:
    return f"chat:interactions:{user_id}:{conversation_id}"


async def get_or_create_conversation_id(
    user_id: str,
    *,
    idle_cutoff_seconds: int,
    ttl_seconds: int,
    explicit_conversation_id: Optional[str] = None,
) -> str:
    """
    Uses explicit_conversation_id if provided; otherwise uses rolling idle window.
    """
    if explicit_conversation_id:
        return explicit_conversation_id

    r = await get_redis()
    if not r:
        # no redis -> single ephemeral conversation
        return "no-redis"

    key = _key_active_conv(user_id)
    raw = await r.get(key)
    now = int(time.time())

    if raw:
        try:
            obj = json.loads(raw)
            conv_id = obj.get("conversation_id")
            last_seen = int(obj.get("last_seen", 0))
            if conv_id and (now - last_seen) <= idle_cutoff_seconds:
                await r.set(key, json.dumps({"conversation_id": conv_id, "last_seen": now}), ex=ttl_seconds)
                return conv_id
        except Exception:
            pass

    conv_id = str(uuid.uuid4())
    await r.set(key, json.dumps({"conversation_id": conv_id, "last_seen": now}), ex=ttl_seconds)
    return conv_id


async def fetch_recent_interactions(user_id: str, conversation_id: str, limit: int = REDIS_OLD_INTERACTIONS_LIMIT) -> List[Dict[str, str]]:
    r = await get_redis()
    key = _key_interactions(user_id, conversation_id)
    raw = await r.lrange(key, -limit, -1)

    out: List[Dict[str, str]] = []
    for item in raw or []:
        try:
            obj = json.loads(item)
            if isinstance(obj, dict):
                out.append({
                    "question": str(obj.get("question", "")),
                    "answer": str(obj.get("answer", "")),
                })
        except Exception:
            continue
    return out


async def append_interaction(
    user_id: str,
    conversation_id: str,
    question: str,
    answer: str,
    max_items: int,
    ttl_seconds: int,
) -> None:
    r = await get_redis()
    if not r or not conversation_id:
        return

    key = _key_interactions(user_id, conversation_id)
    payload = json.dumps({"question": question, "answer": answer})

    pipe = r.pipeline(transaction=True)
    pipe.rpush(key, payload)
    pipe.ltrim(key, -max_items, -1)
    pipe.expire(key, ttl_seconds)
    await pipe.execute()
