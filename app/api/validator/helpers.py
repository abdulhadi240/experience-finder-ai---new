import os
import asyncio
import json
import uuid
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import httpx
import requests
from dotenv import load_dotenv

from app.api.validator.models.schemas import ValidatorRequest
from app.api.validator.services.openai_service import OpenAIService
from app.api.validator.services.validator_service import validate_research
from app.api.validator.services.conversion import convert_research_to_attraction, pick_best_citation
from app.api.validator.services.supabase_service import SupabaseService
from app.api.validator.services.whitelist_service import run_whitelist_discovery


# -------------------------------------------------------
# Helper: Google Maps Lookup
# -------------------------------------------------------
async def get_google_maps_data(location_query: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Calls the Google Geocoding API to get data for a location."""
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location_query, "key": api_key}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "OK" and data.get("results"):
                return data["results"][0]
            return None
    except httpx.HTTPStatusError as e:
        return None
    except Exception as e:
        return None


# -------------------------------------------------------
# Helper: Google Place Photo
# -------------------------------------------------------
async def get_place_photo_url(place_id: str, api_key: str) -> Optional[str]:
    """
    Fetches the first photo for a place using the Place Details API.
    Follows Google's redirect to return the real CDN URL (lh3.googleusercontent.com)
    which is shorter and contains no API key.
    """
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "photos",
        "key": api_key,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(details_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "OK":
                photos = data.get("result", {}).get("photos", [])
                if photos:
                    photo_ref = photos[0].get("photo_reference")
                    if photo_ref:
                        maps_url = (
                            f"https://maps.googleapis.com/maps/api/place/photo"
                            f"?maxwidth=800&photoreference={photo_ref}&key={api_key}"
                        )
                        # Follow the redirect — Google returns the real CDN URL
                        # e.g. https://lh3.googleusercontent.com/places/...
                        redirect_resp = await client.get(maps_url, follow_redirects=False)
                        cdn_url = redirect_resp.headers.get("location")
                        if cdn_url:
                            return cdn_url
                        # Fallback: return the maps URL if redirect didn't happen
                        return maps_url
    except Exception:
        pass
    return None


async def get_place_details(place_id: str, api_key: str) -> Dict[str, Any]:
    """
    Fetches place details (photo URL, rating, num reviews, opening hours) in a single
    Place Details API call. Returns a dict with keys: photo_url, rating, num_reviews, hours.
    """
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "photos,rating,user_ratings_total,opening_hours,business_status,types",
        "key": api_key,
    }
    result: Dict[str, Any] = {"photo_url": None, "rating": None, "num_reviews": None, "hours": None, "business_status": None, "types": []}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(details_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "OK":
                place = data.get("result", {})

                # Photo
                photos = place.get("photos", [])
                if photos:
                    photo_ref = photos[0].get("photo_reference")
                    if photo_ref:
                        maps_url = (
                            f"https://maps.googleapis.com/maps/api/place/photo"
                            f"?maxwidth=800&photoreference={photo_ref}&key={api_key}"
                        )
                        redirect_resp = await client.get(maps_url, follow_redirects=False)
                        cdn_url = redirect_resp.headers.get("location")
                        result["photo_url"] = cdn_url if cdn_url else maps_url

                # Rating & reviews
                if place.get("rating") is not None:
                    result["rating"] = place["rating"]
                if place.get("user_ratings_total") is not None:
                    result["num_reviews"] = place["user_ratings_total"]

                # Opening hours — store as {day: hours_string} dict
                opening_hours = place.get("opening_hours", {})
                weekday_text = opening_hours.get("weekday_text", [])
                if weekday_text:
                    hours_dict = {}
                    for entry in weekday_text:
                        # e.g. "Monday: 9:00 AM – 5:00 PM"
                        if ": " in entry:
                            day, times = entry.split(": ", 1)
                            hours_dict[day] = times
                    if hours_dict:
                        result["hours"] = hours_dict

                # Business status (OPERATIONAL, CLOSED_TEMPORARILY, CLOSED_PERMANENTLY)
                result["business_status"] = place.get("business_status")

                # Place types from Google
                result["types"] = place.get("types", [])

            else:
                pass
    except Exception:
        pass
    return result


# -------------------------------------------------------
# Helper: RAG Upsert
# -------------------------------------------------------
async def upsert_to_rag(formatted_data: Dict[str, Any], maps_data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Upsert formatted attraction data to the RAG system.
    Uses Google Maps place_id as ID, or generates a random UUID if not available.

    Args:
        formatted_data: The formatted attraction data from conversion
        maps_data: Optional Google Maps data containing place_id

    Returns:
        True if upsert was successful, False otherwise
    """

    # Determine the ID: use Google Maps place_id or generate UUID
    doc_id = None
    if maps_data and maps_data.get("place_id"):
        doc_id = maps_data["place_id"]
    else:
        doc_id = str(uuid.uuid4())

    # Get current timestamp in ISO-8601 format
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Extract meta_obj fields
    meta_obj = formatted_data.get("meta_obj", {})

    # Build the RAG upsert payload
    payload = {
        "id": doc_id,
        "title": formatted_data.get("title", ""),
        "content": formatted_data.get("content", ""),
        "query": formatted_data.get("query", ""),
        "category": formatted_data.get("category", ""),
        "country": formatted_data.get("country", ""),
        "city": formatted_data.get("city", ""),
        "meta_obj": {
            "address": meta_obj.get("location", "") or meta_obj.get("address", "")
        },
        "updated_at": updated_at,
        "language": formatted_data.get("language", "en")
    }

    # Add optional fields if available
    if formatted_data.get("tags"):
        payload["tags"] = formatted_data["tags"]
    if formatted_data.get("source"):
        payload["source"] = formatted_data["source"]
    if formatted_data.get("region_code"):
        payload["region_code"] = formatted_data["region_code"]
    if formatted_data.get("latitude"):
        payload["latitude"] = formatted_data["latitude"]
    if formatted_data.get("longitude"):
        payload["longitude"] = formatted_data["longitude"]
    if formatted_data.get("image"):
        payload["image"] = formatted_data["image"]

    # Add optional meta_obj fields
    if meta_obj.get("ranking"):
        payload["meta_obj"]["ranking"] = meta_obj["ranking"]
    if meta_obj.get("audience"):
        payload["meta_obj"]["audience"] = meta_obj["audience"]
    if meta_obj.get("price_level") or meta_obj.get("priceLevel"):
        payload["meta_obj"]["priceLevel"] = meta_obj.get("price_level") or meta_obj.get("priceLevel")
    if meta_obj.get("hours"):
        payload["meta_obj"]["hours"] = meta_obj["hours"]
    if meta_obj.get("rating") is not None:
        payload["meta_obj"]["rating"] = meta_obj["rating"]
    if meta_obj.get("numReviews") is not None:
        payload["meta_obj"]["numReviews"] = meta_obj["numReviews"]


    # Send to RAG API
    rag_url = "https://rag.hiptraveler.com/upsert"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer 96717f5a-e562-4fd5-a14c-499794e8328f"
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(rag_url, json=payload, headers=headers)
            response.raise_for_status()

            return True

    except httpx.HTTPStatusError as e:
        return False

    except httpx.TimeoutException:
        return False

    except Exception as e:
        return False


# -------------------------------------------------------
# Helper: RAG Check
# -------------------------------------------------------
def has_rag_answer(rag_response: Dict[str, Any]) -> bool:
    """
    Check if RAG response contains a valid answer based on the actual response structure.
    RAG has data if any of these arrays are non-empty: entities, chunks, audience, travel_style
    """
    if not rag_response:
        return False

    # Check if any of the data arrays have content
    entities = rag_response.get("entities", [])
    chunks = rag_response.get("chunks", [])
    audience = rag_response.get("audience", [])
    travel_style = rag_response.get("travel_style", [])

    # Debug: Show what we found

    # RAG has data if any array is non-empty
    has_data = (
        (isinstance(entities, list) and len(entities) > 0) or
        (isinstance(chunks, list) and len(chunks) > 0) or
        (isinstance(audience, list) and len(audience) > 0) or
        (isinstance(travel_style, list) and len(travel_style) > 0)
    )

    if has_data:
        if isinstance(entities, list) and len(entities) > 0:
            pass
        if isinstance(chunks, list) and len(chunks) > 0:
            pass
        if isinstance(audience, list) and len(audience) > 0:
            pass
        if isinstance(travel_style, list) and len(travel_style) > 0:
            pass
    else:
        pass

    return has_data


def rag(query: str, reference: str) -> Dict[str, Any]:
    """
    Send a query to the webhook for RAG processing.

    Args:
        query (str): The query string to send to the RAG system
        reference (str): The reference/original query

    Returns:
        Dict[str, Any]: Response from the webhook

    Raises:
        requests.exceptions.RequestException: If the request fails
        ValueError: If the query is empty or None
    """

    # Validate input
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or None")

    # Webhook URL
    url = "https://rag.hiptraveler.com/chat"

    # Prepare the payload
    payload = {
        "query": query.strip(),
        "reference": reference
    }

    # Headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        # Send POST request
        response = requests.post(
            url=url,
            json=payload,
            headers=headers,
            timeout=30  # 30 second timeout
        )

        # Raise an exception for bad status codes
        response.raise_for_status()

        # Try to parse JSON response
        try:
            return response.json()
        except json.JSONDecodeError:
            # If response is not JSON, return the text content
            return {
                "success": True,
                "data": response.text,
                "status_code": response.status_code
            }

    except requests.exceptions.Timeout:
        raise requests.exceptions.RequestException("Request timed out after 30 seconds")
    except requests.exceptions.ConnectionError:
        raise requests.exceptions.RequestException("Failed to connect to the webhook")
    except requests.exceptions.HTTPError as e:
        raise requests.exceptions.RequestException(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Request failed: {e}")


# -------------------------------------------------------
# Helper: Duplicate Query Check via OpenAI
# -------------------------------------------------------
async def is_duplicate_query(new_query: str, supabase_service: SupabaseService) -> bool:
    """
    Fetches all non-expired saved queries from Supabase, then asks OpenAI 4o
    whether the new query is the same or very similar to any existing one.
    Returns True if duplicate found, False otherwise.
    """
    load_dotenv()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return False

    # 1. Fetch all non-expired saved queries
    saved_rows = await supabase_service.get_all_saved_queries()
    if not saved_rows:
        return False

    existing_queries = [row["query_text"] for row in saved_rows if row.get("query_text")]
    if not existing_queries:
        return False


    # 2. Ask OpenAI 4o for similarity
    prompt = (
        "You are a duplicate-query detector. I will give you a NEW query and a list of EXISTING queries.\n"
        "Determine if the NEW query is the same as, or very similar in intent/meaning to, ANY of the existing queries.\n\n"
        f"NEW QUERY: \"{new_query}\"\n\n"
        f"EXISTING QUERIES:\n"
        + "\n".join(f"- \"{q}\"" for q in existing_queries)
        + "\n\nRespond with ONLY valid JSON: {\"result\": \"yes\"} if duplicate/very similar, or {\"result\": \"no\"} if not."
    )

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o",
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            is_dup = result.get("result", "no").strip().lower() == "yes"

            return is_dup

    except Exception as e:
        return False


# -------------------------------------------------------
# Helper: Research Processing
# -------------------------------------------------------
async def process_query_research(
    query: str,
    original_query: str,
    query_type: str,
    supabase_service: SupabaseService,
    reference: str,
    skip_storage: bool = False,
) -> Dict[str, Any]:
    """Runs validation, converts to attraction format, and saves to Supabase.
    When skip_storage=True, skips all RAG upserts and Supabase inserts."""
    load_dotenv()
    t_total = time.time()

    # ============================================
    # STEP 1+2: RAG check and web research in parallel
    # ============================================
    t0 = time.time()

    rag_raw, result = await asyncio.gather(
        asyncio.to_thread(rag, query, reference),
        asyncio.to_thread(validate_research, query),
        return_exceptions=True,
    )

    # ── Process RAG result ────────────────────────────────────────
    rag_response = None
    if not isinstance(rag_raw, Exception):
        if has_rag_answer(rag_raw):
            rag_response = rag_raw
        else:
            pass

    # ── Process validate_research result ─────────────────────────
    if isinstance(result, Exception):
        return None

    # ============================================
    # SCORE CHECK: Skip if score is 0
    # ============================================
    score = result.get("score", "0/3")
    score_value = float(score.split("/")[0]) if "/" in str(score) else 0
    if score_value == 0:
        return None

    location = result.get("location")

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    maps_data = None
    place_image_url = None
    t1 = time.time()
    if location and api_key:
        maps_data = await get_google_maps_data(location, api_key)
        if maps_data:
            place_id = maps_data.get("place_id")
            if place_id:
                place_image_url = await get_place_photo_url(place_id, api_key)
        else:
            pass
    conversion_input = {
        "type": query_type,
        "original_query": original_query,
        "queries": [query],
        "rag_context": rag_response,  # 🔥 Pass RAG data to exclude from research
        "results": [
            {
                "query": query,
                "score": result.get("score"),
                "research": result.get("research"),
                "citations": result.get("citations"),
                "location": location,
                "maps_data": maps_data,
            }
        ],
    }

    if rag_response:
        pass
    else:
        pass

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    try:
        t2 = time.time()
        formatted_data_list = await asyncio.to_thread(
            convert_research_to_attraction,
            conversion_input,
            openai_api_key,
        )

        if formatted_data_list:
            # ── Override source with best citation (TripAdvisor → Yelp → other) ──
            best_src = pick_best_citation(result.get("citations", []))

            # Process all converted results (may be multiple when research covers multiple places)
            all_formatted = []
            for item in formatted_data_list:
                formatted_data = item.model_dump()

                if best_src:
                    formatted_data["source"] = best_src

                # ── Per-place Google Maps lookup using specific title + city ──
                place_maps_data = maps_data  # fall back to city-level if lookup fails
                place_image = place_image_url

                place_title = formatted_data.get("title", "")
                place_city = formatted_data.get("city", "")
                if api_key and place_title and place_city:
                    specific_query = f"{place_title}, {place_city}"
                    specific_maps = await get_google_maps_data(specific_query, api_key)
                    if specific_maps and specific_maps.get("place_id"):
                        place_maps_data = specific_maps
                        place_details = await get_place_details(specific_maps["place_id"], api_key)

                        # Skip closed businesses
                        biz_status = place_details.get("business_status")
                        if biz_status in ("CLOSED_TEMPORARILY", "CLOSED_PERMANENTLY"):
                            continue

                        # Skip hotels — we use a third-party API for hotel data
                        place_types = place_details.get("types", [])
                        _HOTEL_TYPES = {"lodging", "hotel", "motel", "hostel", "resort_hotel", "extended_stay_hotel"}
                        if _HOTEL_TYPES & set(place_types):
                            continue

                        place_image = place_details["photo_url"]
                        # Update lat/lng from specific place
                        geom = specific_maps.get("geometry", {}).get("location", {})
                        if geom.get("lat"):
                            formatted_data["latitude"] = str(geom["lat"])
                        if geom.get("lng"):
                            formatted_data["longitude"] = str(geom["lng"])
                        # Update meta_obj location with specific address
                        if specific_maps.get("formatted_address"):
                            formatted_data.setdefault("meta_obj", {})["location"] = specific_maps["formatted_address"]
                        # Populate meta_obj with rating, numReviews, hours from Place Details
                        meta = formatted_data.setdefault("meta_obj", {})
                        if place_details["rating"] is not None:
                            meta["rating"] = place_details["rating"]
                        if place_details["num_reviews"] is not None:
                            meta["numReviews"] = place_details["num_reviews"]
                        if place_details["hours"]:
                            meta["hours"] = place_details["hours"]
                    else:
                        pass

                # Skip if the assigned category is hotel-related
                cat_lower = formatted_data.get("category", "").lower()
                if any(h in cat_lower for h in ("hotel", "hostel", "motel", "lodging", "resort")):
                    continue

                # ============================================
                # SCORE-BASED ROUTING
                # ============================================

                formatted_data["score_value"] = score_value
                formatted_data["id"] = place_maps_data.get("place_id") if place_maps_data else None
                if place_image:
                    formatted_data["image"] = place_image

                if skip_storage:
                    formatted_data["rag_upserted"] = False
                    formatted_data["db_id"] = None

                elif score_value >= 2:
                    t_upsert = time.time()
                    rag_success = await upsert_to_rag(formatted_data, place_maps_data)
                    formatted_data["rag_upserted"] = rag_success
                    formatted_data["db_id"] = None
                    if rag_success:
                        pass
                    else:
                        pass

                elif score_value >= 1:
                    try:
                        db_record = await supabase_service.insert_research_insight(formatted_data)
                        formatted_data["db_id"] = db_record.get("id")
                        formatted_data["created_at"] = db_record.get("created_at")
                    except Exception as e:
                        formatted_data["db_id"] = None
                        formatted_data["db_error"] = str(e)
                    formatted_data["rag_upserted"] = False
                else:
                    formatted_data["rag_upserted"] = False

                all_formatted.append(formatted_data)

            # Return the first result for backwards compatibility (callers expect a single dict)
            return all_formatted[0] if len(all_formatted) == 1 else all_formatted[0]
        return conversion_input["results"][0]
    except Exception as e:
        return conversion_input["results"][0]


# -------------------------------------------------------
# Background Processing
# -------------------------------------------------------
async def process_in_background(query: str, reference: str, classification: Optional[Dict[str, Any]] = None):
    """Process the research in the background.
    If `classification` is already computed (e.g. ran in parallel with dedup), it is reused
    to skip the classify_query call entirely.
    """
    try:
        t_bg = time.time()

        supabase_service = SupabaseService()

        if classification is None:
            openai_service = OpenAIService()
            t_c = time.time()
            classification = await openai_service.classify_query(query)
        else:
            pass

        formatted_results: List[Dict[str, Any]] = []

        if classification.get("queries"):
            sub_queries = classification["queries"]

            # ── Fire whitelist discovery in parallel (non-blocking) ───────────
            # One whitelist task per unique sub-query — runs concurrently with
            # the full research pipeline and does NOT block returning results.
            for q in sub_queries:
                asyncio.create_task(run_whitelist_discovery(q))

            if len(sub_queries) == 1:
                result = await process_query_research(
                    sub_queries[0],
                    query,
                    classification["type"],
                    supabase_service,
                    reference
                )
                if result is not None:
                    formatted_results.append(result)
            else:
                tasks = [
                    process_query_research(q, query, classification["type"], supabase_service, reference)
                    for q in sub_queries
                ]
                all_results = await asyncio.gather(*tasks)
                formatted_results = [r for r in all_results if r is not None]


    except Exception as e:
        import traceback


# -------------------------------------------------------
# Deep Validation Processing (awaitable, returns filtered results)
# -------------------------------------------------------
async def process_deep_validation(query: str, reference: str) -> List[Dict[str, Any]]:
    """Run the same research pipeline as process_in_background but await results
    and return only items with score >= 2."""

    openai_service = OpenAIService()
    supabase_service = SupabaseService()

    classification = await openai_service.classify_query(query)

    all_results: List[Dict[str, Any]] = []

    if classification.get("queries"):
        if len(classification["queries"]) == 1:
            result = await process_query_research(
                classification["queries"][0],
                query,
                classification["type"],
                supabase_service,
                reference,
                skip_storage=True,
            )
            if result is not None:
                all_results.append(result)
        else:
            tasks = [
                process_query_research(q, query, classification["type"], supabase_service, reference, skip_storage=True)
                for q in classification["queries"]
            ]
            gathered = await asyncio.gather(*tasks)
            all_results = [r for r in gathered if r is not None]

    # Filter: keep only items with score >= 2
    high_score_results = [r for r in all_results if r.get("score_value", 0) >= 2]


    return high_score_results
