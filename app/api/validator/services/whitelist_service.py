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
- Do NOT include: facebook.com, instagram.com, twitter.com, tiktok.com, youtube.com, google.com.
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
- Do NOT include: google.com.

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
- Do NOT include: google.com.

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


_BLOCKED_DOMAINS: Set[str] = {
    "google.com",
    "facebook.com", "instagram.com", "twitter.com", "tiktok.com", "youtube.com",
}


def _parse_domain_list(text: str) -> List[str]:
    """
    Extract a list of domain strings from an LLM response.
    Handles clean JSON arrays and recovers gracefully from minor formatting issues.
    Blocked domains (e.g. google.com) are silently dropped.
    """
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group(0))
            if isinstance(items, list):
                result = []
                for item in items:
                    d = _normalise_domain(str(item))
                    if d and d not in _BLOCKED_DOMAINS:
                        result.append(d)
                return result
        except json.JSONDecodeError:
            pass

    result = []
    for m in re.finditer(r'"([^"]+)"', text):
        d = _normalise_domain(m.group(1))
        if d and d not in _BLOCKED_DOMAINS:
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
        return {"llm": "openai", "success": True, "domains": domains}
    except Exception as e:
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
        return {"llm": "perplexity", "success": True, "domains": domains}
    except Exception as e:
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
        return {"llm": "gemini", "success": True, "domains": domains}
    except Exception as e:
        return {"llm": "gemini", "success": False, "domains": []}


# ── Cap ────────────────────────────────────────────────────────────────────────
MAX_WHITELIST_PER_CATEGORY = 5
MAX_RETRY_ATTEMPTS = 3


# ── Domain web-search verification ────────────────────────────────────────────

def _verify_domain_has_content(
    domain: str, category: str, country: str, api_key: str
) -> bool:
    """
    Use OpenAI web search to confirm that the domain actually contains
    listings/reviews for the given category in the given country.
    Returns True if confirmed, False otherwise.
    """
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    prompt = (
        f"Visit the website {domain} and check whether it contains actual listings, "
        f"reviews, or business information for '{category}' in '{country}'. "
        f"Answer ONLY with a JSON object: {{\"has_content\": true}} or {{\"has_content\": false}}. "
        f"Return true only if the site has real, relevant content for this category and country. "
        f"No explanation, no markdown."
    )
    payload = {
        "model": "gpt-4.1-mini",
        "tools": [{"type": "web_search_preview"}],
        "tool_choice": "required",
        "input": prompt,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        # Extract text from output blocks
        output = resp.json().get("output", [])
        text = ""
        for block in output:
            if isinstance(block, dict) and block.get("type") == "message":
                for part in block.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text += part.get("text", "")
        # Parse the JSON answer
        match = re.search(r'\{.*?"has_content"\s*:\s*(true|false).*?\}', text, re.IGNORECASE | re.DOTALL)
        if match:
            result = match.group(1).lower() == "true"
            status = "✅" if result else "❌"
            return result
        return False
    except Exception as e:
        return False


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

    def _run_llm_consensus(self, category: str, country: str) -> Set[str]:
        """
        Fire all 3 LLMs in parallel and return the consensus domain set.
        Returns an empty set if fewer than 2 LLMs succeed.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            f_openai     = pool.submit(_call_openai,     category, country, self.openai_key)
            f_perplexity = pool.submit(_call_perplexity, category, country, self.perplexity_key)
            f_gemini     = pool.submit(_call_gemini,     category, country, self.gemini_key)

            results = [f_openai.result(), f_perplexity.result(), f_gemini.result()]

        successful = [r for r in results if r["success"] and r["domains"]]

        if len(successful) < 2:
            return set()

        domain_sets: List[Set[str]] = [set(r["domains"]) for r in successful]
        consensus: Set[str] = domain_sets[0].intersection(*domain_sets[1:])

        return consensus

    def _web_verify_domains(
        self, domains: Set[str], category: str, country: str
    ) -> Dict[str, bool]:
        """
        Run OpenAI web-search verification for each domain in parallel.
        Returns {domain: True/False}.
        """
        results: Dict[str, bool] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(domains) or 1) as pool:
            futures = {
                pool.submit(
                    _verify_domain_has_content, d, category, country, self.openai_key
                ): d
                for d in domains
            }
            for fut, domain in futures.items():
                results[domain] = fut.result()
        confirmed = [d for d, ok in results.items() if ok]
        rejected  = [d for d, ok in results.items() if not ok]
        return results

    def find_and_store(self, query: str) -> Dict[str, Any]:
        """
        Extract category + country from the query, run LLM consensus to find
        trusted domains, web-verify each one, and upsert only confirmed domains.
        If no domain passes web verification, retries the full LLM consensus
        up to MAX_RETRY_ATTEMPTS times before giving up.
        """
        category = _extract_category(query)
        country  = _extract_country_from_query(query, self.openai_key)

        if not country:
            return {"skipped": True, "reason": "no_country", "query": query}


        # ── Cap check upfront ─────────────────────────────────────────────────
        existing = _fetch_whitelist_sync(self._supabase, category, country)
        slots_available = MAX_WHITELIST_PER_CATEGORY - len(existing)

        if slots_available <= 0:
            return {
                "category": category,
                "country": country,
                "stored": 0,
                "capped": True,
                "existing": existing,
            }

        # ── Retry loop: consensus → web-verify → store ────────────────────────
        all_verified_ever: Set[str] = set()
        confirmed_domains: Set[str] = set()

        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):

            consensus = self._run_llm_consensus(category, country)
            if not consensus:
                continue

            all_verified_ever.update(consensus)

            # Only web-verify domains not already confirmed or known bad
            new_candidates = consensus - confirmed_domains - set(existing)
            if not new_candidates:
                continue

            verify_results = self._web_verify_domains(new_candidates, category, country)
            newly_confirmed = {d for d, ok in verify_results.items() if ok}
            confirmed_domains.update(newly_confirmed)

            if confirmed_domains:
                break


        # ── Store confirmed domains up to available slots ─────────────────────
        if not confirmed_domains:
            return {
                "category": category,
                "country": country,
                "consensus_found": sorted(all_verified_ever),
                "web_confirmed": [],
                "stored": 0,
            }

        new_domains = sorted(confirmed_domains - set(existing))[:slots_available]
        stored = self._upsert_domains(category, country, set(new_domains))


        return {
            "category": category,
            "country": country,
            "consensus_found": sorted(all_verified_ever),
            "web_confirmed": sorted(confirmed_domains),
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
            except Exception:
                pass

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
        return []
