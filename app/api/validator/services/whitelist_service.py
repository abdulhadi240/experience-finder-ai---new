"""
Whitelist Domain Finder Service
================================
Discovers verified, trusted domains for a given category + country by
querying OpenAI, Perplexity, and Gemini in parallel and finding consensus.

A domain is considered "verified" only when all three LLMs independently
name it as a top source for the same category + country pair.

Flow:
    1. Extract (category, country) from the incoming query.
    2. Fire OpenAI / Perplexity / Gemini in parallel — each returns 5 domains.
    3. Compute the intersection across all three result sets.
    4. Upsert every verified domain into the `whitelist_domains` Supabase table.
"""

import os
import re
import json
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from supabase import create_client


# ── Prompts ────────────────────────────────────────────────────────────────────
# Each prompt is intentionally slightly differently worded so each LLM reasons
# independently.  All three must agree on a domain for it to be verified.

_OPENAI_SYSTEM = (
    "You are a travel research expert. "
    "Your only job is to identify the most authoritative and widely-used websites "
    "that list and review travel businesses."
)

_OPENAI_USER = """\
Task: For the category "{category}" in the country "{country}", name exactly 5 trusted \
websites that provide authentic user reviews and comprehensive business listings.

Rules:
- Return ONLY root domain names (e.g. tripadvisor.com, yelp.com, foodpanda.com).
- Include both global platforms AND country-specific platforms that are widely \
  used in {country}.
- The domains must be well-known review or listing platforms — NOT blogs, news \
  sites, government sites, or social media.
- Do NOT include: facebook.com, instagram.com, twitter.com, tiktok.com, youtube.com.
- Normalise each domain to lowercase with no "www." prefix.
- Order from most authoritative to least.

Respond with ONLY a valid JSON array of exactly 5 strings — no explanation, no markdown:
["domain1.com", "domain2.com", "domain3.com", "domain4.com", "domain5.com"]"""

_PERPLEXITY_SYSTEM = (
    "You are a travel industry expert specialising in trusted information sources. "
    "Answer concisely with only the requested data."
)

_PERPLEXITY_USER = """\
Which 5 websites do travellers trust most to find and review {category} in the \
country of {country}?

Requirements:
- Legitimate review or listing platforms with real user reviews (NOT social media, \
  NOT blogs, NOT news sites).
- Include both international platforms and country-specific ones popular in {country}.
- Return ONLY lowercase root domain names (no www., no paths, no descriptions).
- Exactly 5 domains, ranked by trustworthiness and popularity.

Output — a JSON array only, nothing else:
["domain1.com", "domain2.com", "domain3.com", "domain4.com", "domain5.com"]"""

_GEMINI_SYSTEM = (
    "You are a travel research assistant. "
    "Provide precise, factual answers without any extra text."
)

_GEMINI_USER = """\
List the 5 most authoritative and widely-used websites for finding and reviewing \
{category} in the country of {country}.

Criteria:
- Must be a review or listing platform (not a blog, news outlet, or social media site).
- Must have verified user reviews and comprehensive listings.
- Include country-specific platforms popular in {country} alongside global ones.
- Lowercase root domain only (no www., no paths).
- Exactly 5 entries.

Respond with ONLY a JSON array — no prose, no markdown, no code fences:
["domain1.com", "domain2.com", "domain3.com", "domain4.com", "domain5.com"]"""


# ── Country extraction ─────────────────────────────────────────────────────────

def _extract_country_from_query(query: str, openai_key: str) -> Optional[str]:
    """
    Use a lightweight OpenAI call to extract the country from the query.
    Resolves city names to their country (e.g. "karachi" → "pakistan").
    Returns lowercase country name, or None if it cannot be determined.
    """
    if not openai_key:
        return None

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4.1-nano",
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract the country from travel queries. "
                    "Resolve any city, region, or destination to its country. "
                    "Always respond with ONLY a JSON object."
                ),
            },
            {
                "role": "user",
                "content": (
                    f'Travel query: "{query}"\n\n'
                    "What country is this query about? "
                    "If the query mentions a city (e.g. Karachi → Pakistan, "
                    "Paris → France, New York → United States), resolve it to the country. "
                    "If no location is found, return null.\n\n"
                    'Respond with ONLY: {"country": "country name in lowercase"} '
                    'or {"country": null}'
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        result = json.loads(resp.json()["choices"][0]["message"]["content"])
        country = result.get("country")
        if country and isinstance(country, str):
            return country.strip().lower()
        return None
    except Exception as e:
        print(f"  ⚠️  Country extraction failed: {e}")
        return None


# ── Category keyword mapping ──────────────────────────────────────────────────
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "restaurant": [
        "restaurant", "restaurants", "food", "dining", "cafe", "cafes",
        "eat", "eating", "cuisine", "diner", "eatery", "bistro", "bar",
        "brunch", "lunch", "dinner", "breakfast",
    ],
    "hotel": [
        "hotel", "hotels", "accommodation", "resort", "resorts", "motel",
        "hostel", "stay", "lodging", "inn", "villa", "guesthouse", "airbnb",
        "apartment", "suite",
    ],
    "activity": [
        "activity", "activities", "tour", "tours", "experience", "experiences",
        "adventure", "excursion", "excursions", "trip", "class", "classes",
        "workshop", "lesson", "sport", "sports", "hiking", "diving", "surfing",
        "skiing", "cycling", "things to do",
    ],
    "place": [
        "place", "places", "attraction", "attractions", "landmark", "landmarks",
        "sight", "sights", "museum", "park", "beach", "monument", "temple",
        "church", "mosque", "gallery", "zoo", "garden", "market", "plaza",
        "square", "castle", "fort", "palace",
    ],
}


def _extract_category(query: str) -> str:
    """
    Derive a normalised category from the query text using keyword matching.
    Falls back to 'place' if nothing matches.
    """
    lower = query.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return category
    return "place"


def _normalise_domain(raw: str) -> Optional[str]:
    """
    Normalise a raw string to a lowercase root domain with no 'www.' prefix.
    Handles both bare domains ("Yelp.com") and full URLs ("https://www.yelp.com/...").
    Returns None if the string cannot be parsed as a valid domain.
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip().lower()

    if raw.startswith("http"):
        try:
            parsed = urlparse(raw)
            raw = parsed.netloc or raw
        except Exception:
            pass

    raw = re.sub(r"^www\.", "", raw)

    if "." not in raw or " " in raw:
        return None

    raw = raw.split("/")[0].split("?")[0].split("#")[0]
    return raw if raw else None


def _parse_domain_list(text: str) -> List[str]:
    """
    Extract a list of domain strings from an LLM response.
    Handles clean JSON arrays and recovers gracefully from minor formatting issues.
    """
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group(0))
            if isinstance(items, list):
                result = []
                for item in items:
                    d = _normalise_domain(str(item))
                    if d:
                        result.append(d)
                return result
        except json.JSONDecodeError:
            pass

    result = []
    for m in re.finditer(r'"([^"]+)"', text):
        d = _normalise_domain(m.group(1))
        if d:
            result.append(d)
    return result


# ── LLM callers (synchronous, designed for ThreadPoolExecutor) ─────────────────

def _call_openai(category: str, country: str, api_key: str) -> Dict[str, Any]:
    """Ask OpenAI GPT-4.1-mini to return 5 trusted domains."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4.1-mini",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _OPENAI_SYSTEM},
            {
                "role": "user",
                "content": _OPENAI_USER.format(category=category, country=country),
            },
        ],
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        domains = _parse_domain_list(text)
        print(f"  🤖 OpenAI whitelist [{category}/{country}]: {domains}")
        return {"llm": "openai", "success": True, "domains": domains}
    except Exception as e:
        print(f"  ⚠️  OpenAI whitelist call failed: {e}")
        return {"llm": "openai", "success": False, "domains": []}


def _call_perplexity(category: str, country: str, api_key: str) -> Dict[str, Any]:
    """Ask Perplexity Sonar to return 5 trusted domains."""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _PERPLEXITY_SYSTEM},
            {
                "role": "user",
                "content": _PERPLEXITY_USER.format(category=category, country=country),
            },
        ],
        "return_citations": False,
        "return_related_questions": False,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        domains = _parse_domain_list(text)
        print(f"  🔍 Perplexity whitelist [{category}/{country}]: {domains}")
        return {"llm": "perplexity", "success": True, "domains": domains}
    except Exception as e:
        print(f"  ⚠️  Perplexity whitelist call failed: {e}")
        return {"llm": "perplexity", "success": False, "domains": []}


def _call_gemini(category: str, country: str, api_key: str) -> Dict[str, Any]:
    """Ask Gemini 2.0 Flash to return 5 trusted domains."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    headers = {"Content-Type": "application/json"}
    payload = {
        "system_instruction": {"parts": [{"text": _GEMINI_SYSTEM}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": _GEMINI_USER.format(category=category, country=country)
                    }
                ],
            }
        ],
        "generationConfig": {"temperature": 0},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        parts = resp.json()["candidates"][0]["content"]["parts"]
        text = " ".join(p.get("text", "") for p in parts)
        domains = _parse_domain_list(text)
        print(f"  💎 Gemini whitelist [{category}/{country}]: {domains}")
        return {"llm": "gemini", "success": True, "domains": domains}
    except Exception as e:
        print(f"  ⚠️  Gemini whitelist call failed: {e}")
        return {"llm": "gemini", "success": False, "domains": []}


# ── Cap ────────────────────────────────────────────────────────────────────────
MAX_WHITELIST_PER_CATEGORY = 5


# ── Supabase helpers ───────────────────────────────────────────────────────────

def _fetch_whitelist_sync(
    supabase_client, category: str, country: str
) -> List[str]:
    """
    Synchronously fetch verified domains for a category + country from Supabase.
    Returns a list of domain strings (e.g. ["foodpanda.com", "tripadvisor.com"]).
    """
    try:
        response = (
            supabase_client
            .table("whitelist_domains")
            .select("source")
            .eq("category", category.lower())
            .eq("country", country.lower())
            .execute()
        )
        return [row["source"] for row in (response.data or [])]
    except Exception as e:
        print(f"  ⚠️  Whitelist fetch failed: {e}")
        return []


# ── Main service class ─────────────────────────────────────────────────────────

class WhitelistDomainFinder:
    """
    Finds and stores verified whitelist domains for a category + country
    by reaching consensus across OpenAI, Perplexity, and Gemini.
    """

    def __init__(self):
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY", "")
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        supabase_url = os.getenv("SUPABASE_URL", "")
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

        if not supabase_url or not service_role_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

        self._supabase = create_client(supabase_url, service_role_key)

    # ── Discovery ──────────────────────────────────────────────────────────────

    def find_and_store(self, query: str) -> Dict[str, Any]:
        """
        Extract category + country from the query, fire all 3 LLMs in parallel,
        find consensus domains, and upsert them into whitelist_domains.
        """
        category = _extract_category(query)
        country = _extract_country_from_query(query, self.openai_key)

        if not country:
            print(f"⚠️  Whitelist: could not extract country from '{query}' — skipping")
            return {"skipped": True, "reason": "no_country", "query": query}

        print(f"\n{'─'*60}")
        print(f"🔐 WHITELIST DOMAIN FINDER")
        print(f"   Query   : {query}")
        print(f"   Category: {category}")
        print(f"   Country : {country}")
        print(f"{'─'*60}")

        # ── Fire all 3 LLMs in parallel ────────────────────────────────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            f_openai     = pool.submit(_call_openai,     category, country, self.openai_key)
            f_perplexity = pool.submit(_call_perplexity, category, country, self.perplexity_key)
            f_gemini     = pool.submit(_call_gemini,     category, country, self.gemini_key)

            r_openai     = f_openai.result()
            r_perplexity = f_perplexity.result()
            r_gemini     = f_gemini.result()

        results    = [r_openai, r_perplexity, r_gemini]
        successful = [r for r in results if r["success"] and r["domains"]]

        if len(successful) < 2:
            print(f"⚠️  Whitelist: fewer than 2 LLMs succeeded — skipping upsert")
            return {
                "skipped": True,
                "reason": "insufficient_llm_responses",
                "category": category,
                "country": country,
            }

        # ── Consensus: domain must appear in ALL successful LLM responses ──────
        domain_sets: List[Set[str]] = [set(r["domains"]) for r in successful]
        verified_domains: Set[str] = domain_sets[0].intersection(*domain_sets[1:])

        print(f"\n  📊 Consensus ({len(successful)}/3 LLMs):")
        for r in successful:
            print(f"       {r['llm']:12s} → {r['domains']}")
        print(f"     ✅ Verified (in ALL): {sorted(verified_domains)}")

        if not verified_domains:
            print(f"  ℹ️  No consensus domains — nothing to store")
            return {
                "category": category,
                "country": country,
                "verified": [],
                "stored": 0,
            }

        # ── Cap: max 5 verified domains per category + country ────────────────
        existing = _fetch_whitelist_sync(self._supabase, category, country)
        slots_available = MAX_WHITELIST_PER_CATEGORY - len(existing)

        print(f"  📦 Existing: {len(existing)}/{MAX_WHITELIST_PER_CATEGORY} — "
              f"slots available: {slots_available}")

        if slots_available <= 0:
            print(f"  🔒 Cap reached ({MAX_WHITELIST_PER_CATEGORY}) for "
                  f"[{category}/{country}] — no new domains stored")
            return {
                "category": category,
                "country": country,
                "verified": sorted(verified_domains),
                "stored": 0,
                "capped": True,
                "existing": existing,
            }

        # Only take as many new domains as there are open slots.
        # Exclude domains already stored, then take up to slots_available.
        new_domains = sorted(verified_domains - set(existing))[:slots_available]

        stored = self._upsert_domains(category, country, set(new_domains))
        print(f"  💾 Stored {stored}/{len(new_domains)} new domains "
              f"(cap: {MAX_WHITELIST_PER_CATEGORY})")
        print(f"{'─'*60}\n")

        return {
            "category": category,
            "country": country,
            "verified": sorted(verified_domains),
            "stored": stored,
            "existing": existing,
            "total_after": len(existing) + stored,
        }

    def _upsert_domains(
        self, category: str, country: str, domains: Set[str]
    ) -> int:
        """
        Upsert each verified domain into whitelist_domains.
        ON CONFLICT updates verified_at so we know when it was last re-confirmed.
        """
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        inserted = 0

        for domain in domains:
            row = {
                "category":    category.lower(),
                "country":     country.lower(),
                "source":      domain.lower(),
                "verified_at": now_iso,
            }
            try:
                response = (
                    self._supabase
                    .table("whitelist_domains")
                    .upsert(row, on_conflict="category,country,source")
                    .execute()
                )
                if response.data:
                    inserted += 1
                    print(f"    ✅ Upserted: {domain}")
            except Exception as e:
                print(f"    ⚠️  Failed to upsert '{domain}': {e}")

        return inserted

    # ── Lookup helpers (used by the validation pipeline) ──────────────────────

    def get_whitelist_for(self, category: str, country: str) -> List[str]:
        """Return all verified domains for a given category + country."""
        return _fetch_whitelist_sync(self._supabase, category, country)

    def is_whitelisted(self, source: str, category: str, country: str) -> bool:
        """Check if a domain is verified for this category + country."""
        domain = _normalise_domain(source)
        if not domain:
            return False
        return domain in set(self.get_whitelist_for(category, country))


# ── Module-level singleton + async wrapper ────────────────────────────────────

_finder: Optional[WhitelistDomainFinder] = None


def _get_finder() -> WhitelistDomainFinder:
    global _finder
    if _finder is None:
        _finder = WhitelistDomainFinder()
    return _finder


async def run_whitelist_discovery(query: str) -> Dict[str, Any]:
    """
    Async wrapper — runs the synchronous whitelist finder in a thread so it
    doesn't block the event loop.  Fire-and-forget from async helpers.
    """
    try:
        finder = _get_finder()
        return await asyncio.to_thread(finder.find_and_store, query)
    except Exception as e:
        print(f"⚠️  run_whitelist_discovery failed: {e}")
        return {"skipped": True, "reason": str(e)}


def fetch_whitelist_domains(category: str, country: str) -> List[str]:
    """
    Synchronous helper for use inside validator_service.py (which runs in
    a ThreadPoolExecutor).  Returns verified domains or [] on any error.
    """
    try:
        finder = _get_finder()
        return finder.get_whitelist_for(category, country)
    except Exception as e:
        print(f"⚠️  fetch_whitelist_domains failed: {e}")
        return []
