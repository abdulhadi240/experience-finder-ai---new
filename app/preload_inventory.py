# app/preload_inventory.py
"""Fire-and-forget inventory preload for every location the agent detects.

Whenever the Experience Finder flow pins down a destination — the trip planner
emits `destinations`, or the explore-path RAG payload carries a `location` —
we tell the backend so it can start warming inventory for that city before
the user gets there:

    POST {PRELOAD_INVENTORY_BASE_URL}/trip-planner/preloadInventory
    X-Callback-Secret: <PRELOAD_INVENTORY_CALLBACK_SECRET>
    {"cityId": null, "formattedAddr": "San Francisco, CA, United States",
     "cityType": "locality", "countryCode": "US", "lang": "en"}

The agent only knows a free-text name ("San Diego", "Karachi"), so each name
is resolved through Google Geocoding to get the state / country pieces the
`formattedAddr` contract needs:

    US city      → "City, ST, United States"   e.g. "New York, NY, United States"
    any other    → "City, Country"             e.g. "Karachi, Pakistan"

Everything here is best-effort and off the request path: call sites use
`preload_locations_background`, which schedules a task and returns at once.
Nothing in this module raises into the chat stream. The backend answers 200
immediately and does the real work asynchronously, so a response is never
awaited for anything but logging.

Internal-only endpoint — never surface the shared secret in a response or log.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import httpx

from .config import settings

_log = logging.getLogger("app.preload")

_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
_PRELOAD_PATH = "/trip-planner/preloadInventory"

# One shared pool for both the geocode lookup and the preload POST. Short
# timeout on purpose: this never blocks a user, but a stalled backend must not
# accumulate open sockets across many chat turns either.
_client = httpx.AsyncClient(timeout=settings.preload_inventory_timeout_seconds)

# Geocode results at these levels are not a city the backend can preload.
# The planner deliberately consolidates "Paris, London, Rome" to countries and
# "Europe" to a continent — those must not trigger a preload of a whole nation.
_SKIP_TYPES = frozenset({"country", "continent"})

# Address-component types that can name the location, most specific first.
# Tokyo geocodes as administrative_area_level_1 and Singapore as a country +
# locality, so the list is wider than just "locality".
_NAME_TYPES = (
    "locality",
    "postal_town",
    "sublocality_level_1",
    "sublocality",
    "administrative_area_level_3",
    "administrative_area_level_2",
    "administrative_area_level_1",
    "colloquial_area",
    "natural_feature",
)

# The chat request carries no language, and every detected name is already
# normalised to English by the planner / nano payload builder.
DEFAULT_LANG = "en"


@dataclass(frozen=True)
class PreloadLocation:
    """The three fields of the preload contract that we can actually supply."""
    formatted_addr: str
    city_type: str
    country_code: str

    def payload(self, lang: str = DEFAULT_LANG) -> dict:
        return {
            # The AI side has no city ids — the backend resolves it from formattedAddr.
            "cityId": None,
            "formattedAddr": self.formatted_addr,
            "cityType": self.city_type,
            "countryCode": self.country_code,
            "lang": lang,
        }


# ─── Pure: geocode result → contract fields ──────────────────────────

def build_location(result: dict) -> Optional[PreloadLocation]:
    """Turn one Google Geocoding result into the preload payload fields.

    Returns None when the result is not something worth preloading: a country
    or continent, or a result with no country / no nameable component.
    """
    types = [t for t in (result.get("types") or []) if t != "political"]
    # Prefer a city-like type when several apply: Singapore comes back as
    # ["country", "locality", "political"] and is a city we DO want to preload.
    city_type = next((t for t in _NAME_TYPES if t in types), types[0] if types else "")
    if not city_type or city_type in _SKIP_TYPES:
        return None

    by_type: dict[str, dict] = {}
    for component in result.get("address_components") or []:
        for t in component.get("types") or []:
            by_type.setdefault(t, component)

    country = by_type.get("country")
    if not country:
        return None
    country_code = (country.get("short_name") or "").strip().upper()
    country_name = (country.get("long_name") or "").strip()
    if not country_code:
        return None

    name = ""
    for t in _NAME_TYPES:
        candidate = (by_type.get(t) or {}).get("long_name") or ""
        if candidate.strip():
            name = candidate.strip()
            break
    if not name:
        return None

    parts = [name]
    if country_code == "US":
        state = ((by_type.get("administrative_area_level_1") or {}).get("short_name") or "").strip()
        if state and state != name:
            parts.append(state)
    if country_name and country_name != name:
        parts.append(country_name)

    return PreloadLocation(
        formatted_addr=", ".join(parts),
        city_type=city_type,
        country_code=country_code,
    )


# ─── Geocoding with a small in-process cache ─────────────────────────

# Destination names repeat constantly ("Paris" on every turn of a planning
# interview), so the geocode result is memoised. Failures are NOT cached — a
# transient outage should not pin a city to "unresolvable" for the process.
_geocode_cache: dict[str, Optional[PreloadLocation]] = {}
_GEOCODE_CACHE_MAX = 2000


async def resolve_location(name: str) -> Optional[PreloadLocation]:
    key = (name or "").strip().lower()
    if not key:
        return None
    if key in _geocode_cache:
        return _geocode_cache[key]

    api_key = settings.google_maps_api_key
    if not api_key:
        _log.warning("[PRELOAD] GOOGLE_MAPS_API_KEY not set — cannot resolve %r", name)
        return None

    try:
        response = await _client.get(
            _GEOCODE_URL, params={"address": name, "language": "en", "key": api_key}
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        _log.warning("[PRELOAD] geocode failed for %r: %s", name, type(e).__name__)
        return None

    status = data.get("status")
    results = data.get("results") or []
    if status in ("OVER_QUERY_LIMIT", "REQUEST_DENIED", "UNKNOWN_ERROR"):
        # Quota / key problems are transient or config bugs — don't cache them.
        _log.warning("[PRELOAD] geocode status=%s for %r", status, name)
        return None

    location = build_location(results[0]) if status == "OK" and results else None
    if location is None:
        _log.info("[PRELOAD] %r is not a preloadable city (status=%s)", name, status)

    if len(_geocode_cache) >= _GEOCODE_CACHE_MAX:
        _geocode_cache.clear()
    _geocode_cache[key] = location
    return location


# ─── The POST ────────────────────────────────────────────────────────

_missing_secret_warned = False


async def post_preload(location: PreloadLocation, lang: str = DEFAULT_LANG) -> bool:
    """Send one preload request. Returns True on a 2xx, False otherwise. Never raises."""
    global _missing_secret_warned
    secret = settings.preload_inventory_callback_secret
    if not secret:
        if not _missing_secret_warned:
            _log.warning("[PRELOAD] PRELOAD_INVENTORY_CALLBACK_SECRET not set — preload disabled")
            _missing_secret_warned = True
        return False

    url = f"{settings.preload_inventory_base_url.rstrip('/')}{_PRELOAD_PATH}"
    headers = {"Content-Type": "application/json", "X-Callback-Secret": secret}
    payload = location.payload(lang)

    # Full request body on the console, same shape as the credits trace.
    # Headers are never logged: they carry the shared secret.
    _log.info("[PRELOAD] INPUT POST %s\n%s", url, json.dumps(payload, indent=2))

    try:
        response = await _client.post(url, json=payload, headers=headers)
    except Exception as e:
        _log.warning(
            "[PRELOAD] OUTPUT request failed for %r: %s", location.formatted_addr, type(e).__name__
        )
        return False

    if response.status_code >= 400:
        # 4xx means our payload or secret is wrong — loud so it gets fixed.
        _log.warning(
            "[PRELOAD] OUTPUT (HTTP %d) rejected for %r — check secret/payload\n%s",
            response.status_code, location.formatted_addr, (response.text or "")[:300],
        )
        return False

    _log.info(
        "[PRELOAD] OUTPUT (HTTP %d) for %r\n%s",
        response.status_code, location.formatted_addr, (response.text or "")[:300],
    )
    return True


# ─── Per-location dedupe within one conversation ─────────────────────

# A planning interview re-emits the same `destinations` on every turn, and an
# explore thread often repeats the same city across several questions. Each of
# those is the same selection, not a new one, so the same (thread, city) pair
# is sent once per window. A different thread — another user, another session —
# is a fresh selection and goes through: the request is per location, not
# per session.
_DEDUPE_WINDOW_SECONDS = 600.0
_recent_sends: dict[tuple[str, str], float] = {}
_RECENT_MAX = 5000


def _should_send(thread_id: str, formatted_addr: str) -> bool:
    now = time.monotonic()
    if len(_recent_sends) >= _RECENT_MAX:
        for k, ts in list(_recent_sends.items()):
            if now - ts > _DEDUPE_WINDOW_SECONDS:
                _recent_sends.pop(k, None)
        if len(_recent_sends) >= _RECENT_MAX:
            _recent_sends.clear()
    key = (thread_id or "", formatted_addr)
    last = _recent_sends.get(key)
    if last is not None and now - last <= _DEDUPE_WINDOW_SECONDS:
        return False
    _recent_sends[key] = now
    return True


# ─── Entry points ────────────────────────────────────────────────────

async def preload_locations(
    names: Iterable[str],
    *,
    thread_id: str = "",
    lang: str = DEFAULT_LANG,
) -> int:
    """Resolve and preload every distinct name. Returns how many were sent. Never raises."""
    if not settings.preload_inventory_enabled:
        return 0

    seen: set[str] = set()
    distinct: list[str] = []
    for raw in names or []:
        if not isinstance(raw, str):
            continue
        name = raw.strip()
        key = name.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        distinct.append(name)
    if not distinct:
        return 0

    async def _one(name: str) -> bool:
        location = await resolve_location(name)
        if location is None:
            return False
        if not _should_send(thread_id, location.formatted_addr):
            return False
        return await post_preload(location, lang)

    results = await asyncio.gather(*(_one(n) for n in distinct), return_exceptions=True)
    sent = 0
    for name, r in zip(distinct, results):
        if isinstance(r, BaseException):
            _log.warning("[PRELOAD] unexpected error for %r: %s", name, r)
        elif r:
            sent += 1
    return sent


# asyncio only holds a weak reference to tasks — keep them alive until done.
_background_tasks: set[asyncio.Task] = set()


def preload_locations_background(
    names: Iterable[str],
    *,
    thread_id: str = "",
    lang: str = DEFAULT_LANG,
    source: str = "",
) -> Optional[asyncio.Task]:
    """Schedule the preload and return immediately — the fire-and-forget entry point.

    `source` is only a log tag ("plan" / "explore") so a trace shows which
    detection path fired.
    """
    if not settings.preload_inventory_enabled:
        return None
    names = [n for n in (names or []) if isinstance(n, str) and n.strip()]
    if not names:
        return None

    _log.info("[PRELOAD] detected %s via %s | thread=%s", names, source or "?", thread_id)
    task = asyncio.create_task(preload_locations(names, thread_id=thread_id, lang=lang))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
