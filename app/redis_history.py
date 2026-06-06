"""
redis_history.py

Production-grade Redis-backed conversation memory:
- Supports Render/Upstash via REDIS_URL
- Supports AWS ElastiCache via REDIS_HOST/REDIS_PORT + REDIS_AUTH_TOKEN (+ TLS)
- Kill switch via REDIS_ENABLED
- Keyed by thread_id — one thread per conversation, managed by the frontend

Env vars (recommended):
  REDIS_ENABLED=true|false            (default: false) - set to true for Render staging
  REDIS_URL=redis://... or rediss://... (optional, preferred when available) - set (Upstash/Render Redis URL)
  REDIS_HOST=...                     (used when REDIS_URL not set)
  REDIS_PORT=6379                    (default: 6379)
  REDIS_AUTH_TOKEN=...               (optional; required for ElastiCache with auth token)
  REDIS_USE_TLS=true|false           (default: true)
"""

from __future__ import annotations

import json
import os
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


def _key_interactions(thread_id: str) -> str:
    return f"chat:interactions:{thread_id}"


async def fetch_recent_interactions(
    thread_id: str,
    *,
    limit: int,
) -> List[Dict[str, str]]:
    """
    Fetch last N interactions for the given thread.
    Returns [] if Redis unavailable.
    """
    if not thread_id:
        return []

    r = await get_redis()
    if not r:
        return []

    limit = max(1, int(limit))
    key = _key_interactions(thread_id)

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

    if out:
        logger.info(
            "[Redis RETRIEVED] thread=%s | interactions=%d",
            thread_id, len(out),
        )
        for i, item in enumerate(out):
            logger.info(
                "[Redis RETRIEVED] [%d]\nQ: %s\nA: %s",
                i + 1,
                item["question"],
                item["answer"],
            )
    else:
        logger.info("[Redis RETRIEVED] thread=%s | no history found", thread_id)

    return out


async def append_interaction(
    thread_id: str,
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
    if not thread_id:
        return

    r = await get_redis()
    if not r:
        return

    max_items = max(1, int(max_items))
    ttl_seconds = max(60, int(ttl_seconds))

    key = _key_interactions(thread_id)
    payload = json.dumps({"question": question, "answer": answer})

    pipe = r.pipeline(transaction=True)
    pipe.rpush(key, payload)
    pipe.ltrim(key, -max_items, -1)
    pipe.expire(key, ttl_seconds)
    await pipe.execute()

    logger.info("[Redis SAVED] thread=%s | ttl=%ds", thread_id, ttl_seconds)
    logger.info("[Redis SAVED] Q: %s", question)
    logger.info("[Redis SAVED] A: %s", answer)


async def clear_conversation(thread_id: str) -> None:
    """
    Delete all interactions for the given thread.
    No-op if Redis unavailable.
    """
    if not thread_id:
        return

    r = await get_redis()
    if not r:
        return

    try:
        await r.delete(_key_interactions(thread_id))
        logger.info("[Redis CLEARED] thread=%s", thread_id)
    except Exception as e:
        logger.warning("Redis clear failed: %s", e)


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