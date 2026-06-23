"""
rag_cache.py

Short-lived Redis cache for RAG + destination-guide results.
Keyed on the structured query payload so identical destination queries
within the TTL window skip the RAG endpoint calls entirely.

Key format:  rag:{reference}:{query_norm}:{location_norm}:{category}:{top_k}
TTL:         600 s (10 minutes)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Tuple

from app.redis_history import get_redis

_log = logging.getLogger("app.rag_cache")

RAG_CACHE_TTL = 600  # 10 minutes

_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").lower().strip())


def build_rag_cache_key(
    query: str,
    location: str,
    category: str,
    top_k: int,
    reference: str,
) -> str:
    return f"rag:{_norm(reference)}:{_norm(query)}:{_norm(location)}:{_norm(category)}:{top_k}"


async def get_rag_cache(key: str) -> Optional[Tuple[Any, Any]]:
    """
    Returns (rag_result, guide_result) from cache, or None on miss / error.
    """
    try:
        r = await get_redis()
        if r is None:
            return None
        raw = await r.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        _log.info("[RAG CACHE HIT] %s", key)
        return data["rag"], data["guide"]
    except Exception as e:
        _log.debug("[RAG CACHE] read error: %s", e)
        return None


async def set_rag_cache(
    key: str,
    rag_result: Any,
    guide_result: Any,
) -> None:
    """
    Stores (rag_result, guide_result) only when neither is an Exception.
    Silently no-ops on any Redis error.
    """
    if isinstance(rag_result, Exception) or isinstance(guide_result, Exception):
        _log.warning("[RAG CACHE] skipping set — result is Exception: rag=%s guide=%s",
                     type(rag_result).__name__, type(guide_result).__name__)
        return
    try:
        r = await get_redis()
        if r is None:
            _log.warning("[RAG CACHE] Redis unavailable — skipping set")
            return
        payload = json.dumps({"rag": rag_result, "guide": guide_result}, ensure_ascii=False)
        await r.set(key, payload, ex=RAG_CACHE_TTL)
        _log.info("[RAG CACHE SET] ttl=%ds | %s", RAG_CACHE_TTL, key)
    except Exception as e:
        _log.warning("[RAG CACHE] write error: %s", e)
