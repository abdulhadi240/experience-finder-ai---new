"""
redis_history.py

Production-grade Redis-backed conversation memory:
- Supports Render/Upstash via REDIS_URL
- Supports AWS ElastiCache via REDIS_HOST/REDIS_PORT + REDIS_AUTH_TOKEN (+ TLS)
- Kill switch via REDIS_ENABLED
- Separates conversations per user via rolling idle cutoff (active conversation id)
- Stores interactions (question+answer) per (user_id, conversation_id)

Env vars (recommended):
  REDIS_ENABLED=true|false            (default: false)
  REDIS_URL=redis://... or rediss://... (optional, preferred when available)
  REDIS_HOST=...                     (used when REDIS_URL not set)
  REDIS_PORT=6379                    (default: 6379)
  REDIS_AUTH_TOKEN=...               (optional; required for ElastiCache with auth token)
  REDIS_USE_TLS=true|false           (default: true)
"""

from __future__ import annotations

import json
import os
import time
import uuid
import logging
from typing import Dict, List, Optional

from redis.asyncio import Redis
from redis.asyncio.client import Redis as RedisType

logger = logging.getLogger("app.redis_history")

# Singleton async client
_redis: Optional[RedisType] = None


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def get_redis_url() -> Optional[str]:
    """
    Prefer a full URL if provided (Render/Upstash).
    Otherwise build from host/port/auth/tls (AWS ElastiCache friendly).
    """
    url = os.getenv("REDIS_URL")
    if url:
        return url

    host = os.getenv("REDIS_HOST")
    if not host:
        return None

    port = os.getenv("REDIS_PORT", "6379")
    token = os.getenv("REDIS_AUTH_TOKEN")
    use_tls = _env_bool("REDIS_USE_TLS", True)

    scheme = "rediss" if use_tls else "redis"

    # If token exists, include it; else no auth
    if token:
        return f"{scheme}://:{token}@{host}:{port}/0"
    return f"{scheme}://{host}:{port}/0"


def redis_enabled() -> bool:
    """
    Kill-switch + config presence.
    Default is disabled to avoid accidental prod breakage.
    """
    if not _env_bool("REDIS_ENABLED", False):
        return False
    return bool(get_redis_url())


async def get_redis() -> Optional[RedisType]:
    """
    Returns a connected Redis client or None.
    - Caches a singleton client.
    - Validates connection with PING on first creation and on reconnect.
    - If PING fails, it resets the client and returns None.
    """
    global _redis

    if not redis_enabled():
        return None

    url = get_redis_url()
    if not url:
        return None

    # Create client if not created
    if _redis is None:
        _redis = Redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
            retry_on_timeout=True,
            health_check_interval=30,  # helps keep connections healthy
        )

    # Validate connection; reconnect once if needed
    try:
        await _redis.ping()
        return _redis
    except Exception as e:
        logger.warning("Redis ping failed; reconnecting once. err=%s", e)

    # Reconnect once
    try:
        try:
            await _redis.aclose()
        except Exception:
            pass

        _redis = Redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        await _redis.ping()
        return _redis
    except Exception as e:
        logger.error("Redis reconnect failed; disabling Redis for this call. err=%s", e)
        return None


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
) -> Optional[str]:
    """
    Determine which conversation this request belongs to.

    If explicit_conversation_id is provided, use it.
    Otherwise:
      - Reuse last active conversation if last_seen within idle_cutoff_seconds
      - Else create a new UUID conversation_id

    Returns:
      conversation_id (str) if Redis available,
      None if Redis disabled/unavailable.
    """
    if explicit_conversation_id:
        return explicit_conversation_id

    r = await get_redis()
    if not r:
        return None

    # Defensive bounds
    idle_cutoff_seconds = max(1, int(idle_cutoff_seconds))
    ttl_seconds = max(60, int(ttl_seconds))

    key = _key_active_conv(user_id)
    raw = await r.get(key)
    now = int(time.time())

    if raw:
        try:
            obj = json.loads(raw)
            conv_id = obj.get("conversation_id")
            last_seen = int(obj.get("last_seen", 0))
            if conv_id and (now - last_seen) <= idle_cutoff_seconds:
                await r.set(
                    key,
                    json.dumps({"conversation_id": conv_id, "last_seen": now}),
                    ex=ttl_seconds,
                )
                return str(conv_id)
        except Exception:
            # corrupted value -> create new
            pass

    conv_id = str(uuid.uuid4())
    await r.set(
        key,
        json.dumps({"conversation_id": conv_id, "last_seen": now}),
        ex=ttl_seconds,
    )
    return conv_id


async def fetch_recent_interactions(
    user_id: str,
    conversation_id: Optional[str],
    *,
    limit: int,
) -> List[Dict[str, str]]:
    """
    Fetch last N interactions for the given user+conversation.
    Returns [] if Redis unavailable.
    """
    if not conversation_id:
        return []

    r = await get_redis()
    if not r:
        return []

    limit = max(1, int(limit))
    key = _key_interactions(user_id, conversation_id)

    raw = await r.lrange(key, -limit, -1)
    out: List[Dict[str, str]] = []

    for item in raw or []:
        try:
            obj = json.loads(item)
            if isinstance(obj, dict):
                out.append(
                    {
                        "question": str(obj.get("question", "")),
                        "answer": str(obj.get("answer", "")),
                    }
                )
        except Exception:
            continue

    return out


async def append_interaction(
    user_id: str,
    conversation_id: Optional[str],
    *,
    question: str,
    answer: str,
    max_items: int,
    ttl_seconds: int,
) -> None:
    """
    Append a Q/A interaction to Redis list and enforce trim+ttl.
    No-op if Redis unavailable.
    """
    if not conversation_id:
        return

    r = await get_redis()
    if not r:
        return

    max_items = max(1, int(max_items))
    ttl_seconds = max(60, int(ttl_seconds))

    key = _key_interactions(user_id, conversation_id)
    payload = json.dumps({"question": question, "answer": answer})

    pipe = r.pipeline(transaction=True)
    pipe.rpush(key, payload)
    pipe.ltrim(key, -max_items, -1)
    pipe.expire(key, ttl_seconds)
    await pipe.execute()


async def close_redis() -> None:
    """
    Optional: call this on app shutdown if you wire it into FastAPI lifespan.
    """
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception:
            pass
        _redis = None
