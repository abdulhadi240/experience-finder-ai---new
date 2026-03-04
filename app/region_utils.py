# app/region_utils.py
from typing import Optional, Dict, Tuple
from .region_metadata import REGION_METADATA
import hashlib
import json
import re
import pycountry

try:
    from .rag.region_detector import detect_region_from_query
except ImportError:
    def detect_region_from_query(query: str) -> Optional[str]:  # type: ignore[misc]
        return None

_OVERRIDES = {
    "the bahamas": "BS",
    "bahamas": "BS",
    "bahamas, the": "BS",
    "u.s.": "US",
    "usa": "US",
    "united states": "US",
    "uk": "GB",
    "u.k.": "GB",
    "viet nam": "VN",
}

def to_country_code(country: str | None, country_code: str | None = None) -> str | None:
    if country_code and isinstance(country_code, str) and len(country_code.strip()) == 2:
        return country_code.strip().upper()

    if not country:
        return None

    s = country.strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()

    key = s.lower()
    if key in _OVERRIDES:
        return _OVERRIDES[key]

    try:
        c = pycountry.countries.lookup(s)
        return getattr(c, "alpha_2", None)
    except LookupError:
        return None

def detect_location_from_query(query: str, region_meta: Optional[dict] = None) -> Optional[str]:
    """
    Detects a specific sub-location mentioned in the query.
    Handles compound names like 'Eleuthera & Harbour Island' by allowing partial matches.
    """
    if not region_meta:
        return None

    q_lower = query.lower()

    for loc in region_meta.get("locations", []):
        loc_lower = loc.lower()
        # Split compound names like "Eleuthera & Harbour Island" into individual words
        parts = re.split(r"[,&/]", loc_lower)
        parts = [p.strip() for p in parts if p.strip()]

        # Match full name OR partial segment
        if re.search(rf"\b{re.escape(loc_lower)}\b", q_lower):
            return loc
        for part in parts:
            if re.search(rf"\b{re.escape(part)}\b", q_lower):
                return loc

    return None

def detect_locations_from_query(query: str, region_meta: Optional[dict] = None) -> list[str]:
    """
    Detects all valid location names from the query (not just one).
    """
    found = []
    if not region_meta:
        return found

    q_lower = query.lower()
    for loc in region_meta.get("locations", []):
        if re.search(rf"\b{re.escape(loc.lower())}\b", q_lower):
            found.append(loc)
    return found


def make_cache_key(query: str, reference: str, country: str = "", city: str = "") -> str:
    """
    Cache key must include scope because /chat results depend on it.
    """
    raw = f"{reference}|{norm_country(country)}|{(city or '').strip().lower()}|{query.strip().lower()}"
    key_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"chat:{key_hash}"


def get_region_meta(reference: str) -> Optional[dict]:
    return REGION_METADATA.get(reference)

def normalize(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def query_mentions_valid_location(query: str, region_meta: dict) -> bool:

    if not region_meta:
        return False

    q_lower = query.lower()

    # First pass: check direct matches
    if region_meta["country"].lower() in q_lower:
        return True
    for loc in region_meta["locations"]:
        if loc.lower() in q_lower:
            return True
    for exp in region_meta["experiences"]:
        if exp.lower() in q_lower:
            return True

    # Fallback: detect region using LLM
    detected = detect_region_from_query(query)
    if not detected:
        return False

    detected_normalized = detected.lower()
    country_match = detected_normalized == region_meta["country"].lower()
    alias_match = detected_normalized in [loc.lower() for loc in region_meta["locations"]]

    return country_match or alias_match

def detect_and_validate_region(query: str, reference: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Returns a tuple: (region_meta, detected_country)

    - If a region in REGION_METADATA matches the detected signal, returns (region_meta, None).
    - If no REGION_METADATA match but reference == "hiptraveler", returns (None, detected_country)
      so the caller can still filter RAG by country.
    - Otherwise returns (None, None).
    """
    detected = detect_region_from_query(query)
    if not detected:
        return None, None

    detected_norm = normalize(detected)

    # Try to map detection to a known region
    for region in REGION_METADATA.values():
        country = normalize(region.get("country"))
        country_cd = normalize(region.get("countryCd"))
        locations = [normalize(loc) for loc in region.get("locations", [])]

        if (
            detected_norm == country
            or detected_norm == country_cd
            or any(_loc_matches(detected_norm, loc) for loc in locations)
        ):
            return region, None

    # Not in REGION_METADATA
    if reference == "hiptraveler":
        # Still allow filtering by detected country in RAG
        return None, detected

    return None, None

def _loc_matches(detected_norm: str, loc: str) -> bool:
    if detected_norm == loc:
        return True
    # allow partial only if loc contains separators and detected matches a full segment
    parts = [p.strip() for p in re.split(r"[,&/]", loc) if p.strip()]
    return detected_norm in parts

def normalize_country_to_iso2(value: str) -> str:
    if not value:
        return ""

    v = value.strip()

    # If already ISO-2 (FJ, MX, US)
    if len(v) == 2 and v.isalpha():
        return v.upper()

    # If ISO-3 (FJI)
    if len(v) == 3 and v.isalpha():
        try:
            return pycountry.countries.get(alpha_3=v.upper()).alpha_2
        except Exception:
            return ""

    # Try resolve name → ISO-2
    try:
        return pycountry.countries.lookup(v).alpha_2
    except LookupError:
        return ""

def norm_country(value: str) -> str:
    if not value:
        return ""
    v = value.strip()
    iso2 = normalize_country_to_iso2(v)
    return iso2 or v   # keep original if lookup fails

def detect_city_from_query_simple(query: str) -> Optional[str]:
    """
    Heuristic city extraction for queries like:
      - "top 4 restaurants in Paris"
      - "things to do near London"
    """
    q = (query or "").strip()
    if not q:
        return None

    q_lc = q.lower()

    # Prefer patterns like "in X" / "near X" / "around X"
    m = re.search(r"\b(?:in|near|around)\s+([a-zA-Z][a-zA-Z\s\-\']{2,})\b", q_lc)
    if m:
        city = m.group(1).strip(" .,!?:;")
        # Title-case but preserve apostrophes/dashes reasonably
        return " ".join([w.capitalize() for w in city.split()])

    return None
