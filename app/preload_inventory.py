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
import re
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
# The planner keeps every city the user named ("Paris, London, Rome" stays as
# three cities, each preloaded on its own), but a user who says "Japan" or
# "Europe" yields a country / continent — those must not trigger a preload of
# a whole nation.
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
    """The three fields of the preload contract that we can actually supply.

    `name`, `state_name`, `state_code` and `country_name` are not sent; they
    let a split list like "San Francisco, CA" recognise that "CA" is the
    state of the city next to it rather than a second destination.
    """
    formatted_addr: str
    city_type: str
    country_code: str
    name: str = ""
    state_name: str = ""
    state_code: str = ""
    country_name: str = ""

    def is_parent_of(self, other: "PreloadLocation") -> bool:
        """True when this location is the state / country that `other` sits in."""
        me = self.name.strip().lower()
        if not me or self.formatted_addr == other.formatted_addr:
            return False
        return me in {
            other.state_name.strip().lower(),
            other.state_code.strip().lower(),
            other.country_name.strip().lower(),
        } - {""}

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

    state_component = by_type.get("administrative_area_level_1") or {}
    state_code = (state_component.get("short_name") or "").strip()
    state_name = (state_component.get("long_name") or "").strip()

    # A result that IS the state (e.g. "California") must not repeat its own
    # abbreviation: "California, United States", not "California, CA, United States".
    is_state = city_type == "administrative_area_level_1" or (
        bool(state_name) and state_name.lower() == name.lower()
    )

    parts = [name]
    if country_code == "US" and not is_state:
        if state_code and state_code.lower() != name.lower():
            parts.append(state_code)
    if country_name and country_name.lower() != name.lower():
        parts.append(country_name)

    return PreloadLocation(
        formatted_addr=", ".join(parts),
        city_type=city_type,
        country_code=country_code,
        name=name,
        state_name=state_name,
        state_code=state_code,
        country_name=country_name,
    )


# ─── Splitting a multi-city string ───────────────────────────────────

# The explore path's nano payload carries ONE `location` string, and for a
# multi-city question that string is "Barcelona, Madrid". Geocoding it whole
# resolves to only one of them (Madrid), so it is split into its parts first.
_SPLIT_RE = re.compile(r"\s*(?:,|;|/|&|\+|\band\b)\s*", re.IGNORECASE)


def split_location_names(text: str) -> list[str]:
    """'Barcelona, Madrid' → ['Barcelona', 'Madrid']; 'Paris' → ['Paris'].

    Order is kept and exact duplicates dropped. Whether a part is a real
    second city or the state / country of its neighbour ("San Francisco, CA",
    "Paris, France") is decided after geocoding by `drop_parents`.
    """
    seen: set[str] = set()
    out: list[str] = []
    for part in _SPLIT_RE.split(text or ""):
        part = part.strip(" .")
        key = part.lower()
        if not part or key in seen:
            continue
        seen.add(key)
        out.append(part)
    return out


def drop_parents(locations: list[PreloadLocation]) -> list[PreloadLocation]:
    """Remove any location that is just the state / country of another in the list.

    "San Francisco, CA" splits into two names that both geocode — but "CA"
    is California, the state San Francisco sits in, not a second destination.
    """
    return [
        loc for loc in locations
        if not any(loc.is_parent_of(other) for other in locations if other is not loc)
    ]


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


# ─── Formatting the plan's `destinations` for the backend ────────────

async def format_destinations(names: Iterable[str]) -> list[str]:
    """Rewrite each planner destination into the backend's address shape.

        "Barcelona"      → "Barcelona, Spain"
        "San Francisco"  → "San Francisco, CA, United States"   (US: state code)
        "Japan"          → "Japan"                               (country: unchanged)

    Anything that cannot be resolved (no API key, geocode failure, a country
    or continent, an unknown name) is passed through untouched, so the plan
    never loses a destination over a lookup. Order is kept, duplicates
    collapse once formatted (e.g. "SF" and "San Francisco"). Never raises.
    """
    raw: list[str] = []
    for n in names or []:
        if isinstance(n, str) and n.strip():
            raw.append(n.strip())
    if not raw:
        return []

    async def _one(name: str) -> str:
        try:
            location = await asyncio.wait_for(
                resolve_location(name), timeout=settings.preload_inventory_timeout_seconds
            )
        except Exception as e:  # noqa: BLE001 — a lookup must never cost the plan
            _log.warning("[PRELOAD] format lookup failed for %r: %s", name, type(e).__name__)
            return name
        return location.formatted_addr if location else name

    formatted = await asyncio.gather(*(_one(n) for n in raw))
    seen: set[str] = set()
    out: list[str] = []
    for addr in formatted:
        key = addr.lower()
        if key not in seen:
            seen.add(key)
            out.append(addr)
    return out


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
    split: bool = True,
) -> int:
    """Resolve and preload every distinct name. Returns how many were sent. Never raises.

    Each name may itself be a multi-city string ("Barcelona, Madrid" from the
    explore path) — it is split, every part geocoded, and any part that turns
    out to be the state / country of a sibling ("San Francisco, CA") dropped.
    With split=False each name is one address ("Chicago, IL, United States",
    from the planner) and is geocoded whole.
    """
    if not settings.preload_inventory_enabled:
        return 0

    # name → its parts; parts are geocoded individually.
    groups: list[list[str]] = []
    seen: set[str] = set()
    for raw in names or []:
        if not isinstance(raw, str):
            continue
        candidates = split_location_names(raw) if split else [raw.strip()]
        parts = [p for p in candidates if p and p.lower() not in seen]
        if not parts:
            continue
        seen.update(p.lower() for p in parts)
        groups.append(parts)
    if not groups:
        return 0

    async def _resolve_whole(address: str) -> Optional[PreloadLocation]:
        # A stale or wrong state in a three-part address (e.g. "Hoi An, Quang Nam, Vietnam":
        # Quang Nam merged into Da Nang in 2025) makes Google fall back to the
        # bare country. Retry as "City, Country" before giving up.
        location = await resolve_location(address)
        pieces = [p.strip() for p in address.split(",") if p.strip()]
        if location is None and len(pieces) >= 3:
            location = await resolve_location(f"{pieces[0]}, {pieces[-1]}")
        return location

    async def _resolve_group(parts: list[str]) -> list[PreloadLocation]:
        lookup = resolve_location if split else _resolve_whole
        resolved = await asyncio.gather(*(lookup(p) for p in parts), return_exceptions=True)
        found: list[PreloadLocation] = []
        for part, r in zip(parts, resolved):
            if isinstance(r, BaseException):
                _log.warning("[PRELOAD] unexpected error for %r: %s", part, r)
            elif r is not None:
                found.append(r)
        # Only a split string can carry a city next to its own state / country.
        return drop_parents(found) if len(parts) > 1 else found

    group_results = await asyncio.gather(*(_resolve_group(g) for g in groups))
    targets: dict[str, PreloadLocation] = {}
    for locations in group_results:
        for loc in locations:
            targets.setdefault(loc.formatted_addr, loc)

    async def _send(location: PreloadLocation) -> bool:
        if not _should_send(thread_id, location.formatted_addr):
            return False
        return await post_preload(location, lang)

    results = await asyncio.gather(*(_send(loc) for loc in targets.values()), return_exceptions=True)
    sent = 0
    for addr, r in zip(targets, results):
        if isinstance(r, BaseException):
            _log.warning("[PRELOAD] unexpected error for %r: %s", addr, r)
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
    split: bool = True,
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
    task = asyncio.create_task(preload_locations(names, thread_id=thread_id, lang=lang, split=split))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
