"""
Validator API routes — integrated into main FastAPI app.
Handles classification, research validation, OpenAI conversion, and Supabase storage.
"""
import os
import asyncio
import httpx
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from app.api.validator.models.schemas import ValidatorRequest, RAGUpsertRequest, RAGUpsertResponse
from app.api.validator.services.openai_service import OpenAIService
from app.api.validator.services.validator_service import validate_research
from app.api.validator.services.conversion import convert_research_to_attraction
from app.api.validator.services.supabase_service import SupabaseService
from dotenv import load_dotenv
import requests
import json


router = APIRouter()

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
            print(f"Google Maps API Error: {data.get('status')} - {data.get('error_message')}")
            return None
    except httpx.HTTPStatusError as e:
        print(f"HTTP error calling Google Maps API: {e}")
        return None
    except Exception as e:
        print(f"An error occurred calling Google Maps API: {e}")
        return None


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
    print(f"\n{'🔄'*40}")
    print("🔄 STARTING RAG UPSERT")
    print(f"{'🔄'*40}\n")

    # Determine the ID: use Google Maps place_id or generate UUID
    doc_id = None
    if maps_data and maps_data.get("place_id"):
        doc_id = maps_data["place_id"]
        print(f"📍 Using Google Maps place_id: {doc_id}")
    else:
        doc_id = str(uuid.uuid4())
        print(f"🔀 Generated random UUID: {doc_id}")

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

    print(f"📦 RAG Upsert Payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    # Send to RAG API
    rag_url = "https://rag.hiptraveler.com/upsert"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(rag_url, json=payload, headers=headers)
            response.raise_for_status()

            print(f"\n✅ RAG UPSERT SUCCESS")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            print(f"{'🔄'*40}\n")
            return True

    except httpx.HTTPStatusError as e:
        print(f"\n❌ RAG UPSERT FAILED: HTTP {e.response.status_code}")
        print(f"Response: {e.response.text}")
        print(f"{'🔄'*40}\n")
        return False

    except httpx.TimeoutException:
        print(f"\n❌ RAG UPSERT FAILED: Request timed out")
        print(f"{'🔄'*40}\n")
        return False

    except Exception as e:
        print(f"\n❌ RAG UPSERT FAILED: {str(e)}")
        print(f"{'🔄'*40}\n")
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
        print("🔍 RAG Response Check: No response received")
        return False
    
    # Check if any of the data arrays have content
    entities = rag_response.get("entities", [])
    chunks = rag_response.get("chunks", [])
    audience = rag_response.get("audience", [])
    travel_style = rag_response.get("travel_style", [])
    
    # Debug: Show what we found
    print("\n" + "="*80)
    print("🔍 RAG RESPONSE ANALYSIS")
    print("="*80)
    print(f"📊 Entities count: {len(entities) if isinstance(entities, list) else 'N/A'}")
    print(f"📊 Chunks count: {len(chunks) if isinstance(chunks, list) else 'N/A'}")
    print(f"📊 Audience count: {len(audience) if isinstance(audience, list) else 'N/A'}")
    print(f"📊 Travel Style count: {len(travel_style) if isinstance(travel_style, list) else 'N/A'}")
    
    # RAG has data if any array is non-empty
    has_data = (
        (isinstance(entities, list) and len(entities) > 0) or
        (isinstance(chunks, list) and len(chunks) > 0) or
        (isinstance(audience, list) and len(audience) > 0) or
        (isinstance(travel_style, list) and len(travel_style) > 0)
    )
    
    if has_data:
        print("\n✅ RAG HAS DATA - Details:")
        if isinstance(entities, list) and len(entities) > 0:
            print(f"   • Entities ({len(entities)}): {json.dumps(entities, indent=6)}")
        if isinstance(chunks, list) and len(chunks) > 0:
            print(f"   • Chunks ({len(chunks)}): {json.dumps(chunks, indent=6)}")
        if isinstance(audience, list) and len(audience) > 0:
            print(f"   • Audience ({len(audience)}): {json.dumps(audience, indent=6)}")
        if isinstance(travel_style, list) and len(travel_style) > 0:
            print(f"   • Travel Style ({len(travel_style)}): {json.dumps(travel_style, indent=6)}")
    else:
        print("\n❌ RAG HAS NO DATA - All arrays are empty")
    
    print("="*80 + "\n")
    
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
        print("⚠️  OPENAI_API_KEY not set — skipping duplicate check")
        return False

    # 1. Fetch all non-expired saved queries
    saved_rows = await supabase_service.get_all_saved_queries()
    if not saved_rows:
        print("ℹ️  No saved queries in DB — no duplicates possible")
        return False

    existing_queries = [row["query_text"] for row in saved_rows if row.get("query_text")]
    if not existing_queries:
        return False

    print(f"\n🔎 Checking new query against {len(existing_queries)} saved queries...")

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

            if is_dup:
                print(f"🔁 DUPLICATE DETECTED — query already exists: '{new_query}'")
            else:
                print(f"✅ No duplicate found for: '{new_query}'")
            return is_dup

    except Exception as e:
        print(f"⚠️  Duplicate check failed ({type(e).__name__}: {e}) — proceeding as non-duplicate")
        return False
    
    

# -------------------------------------------------------
# Helper: Research Processing
# -------------------------------------------------------
async def process_query_research(
    query: str,
    original_query: str,
    query_type: str,
    supabase_service: SupabaseService,
    reference: str
) -> Dict[str, Any]:
    """Runs validation, converts to attraction format, and saves to Supabase."""
    load_dotenv()
    
    # ============================================
    # STEP 1: Check RAG system first
    # ============================================
    print("\n" + "🔍"*40)
    print("🔍 STARTING RAG CHECK")
    print("🔍"*40)
    print(f"📤 Query sent to RAG: '{query}'")
    print(f"📤 Reference: '{reference}'")
    print(f"📤 Original Query: '{original_query}'")
    print("🔍"*40 + "\n")
    
    rag_response = None  # Initialize to store RAG data
    
    try:
        # Call RAG function asynchronously
        rag_response = await asyncio.to_thread(rag, query, reference)
        
        print("\n" + "📥"*40)
        print("📥 RAG RESPONSE RECEIVED")
        print("📥"*40)
        print("📥 Full RAG Response:")
        print(json.dumps(rag_response, indent=2, ensure_ascii=False))
        print("📥"*40 + "\n")
        
        # Check if RAG has a valid answer
        if has_rag_answer(rag_response):
            print("\n" + "✅"*40)
            print("✅ FOUND IN RAG - WILL EXCLUDE THIS CONTENT")
            print("✅"*40)
            print("📋 Content that will be EXCLUDED from new research:")
            print(json.dumps({
                "entities": rag_response.get("entities", []),
                "chunks": rag_response.get("chunks", []),
                "audience": rag_response.get("audience", []),
                "travel_style": rag_response.get("travel_style", [])
            }, indent=2, ensure_ascii=False))
            print("✅"*40 + "\n")
            # DON'T return None - continue with research but pass RAG context
        else:
            print("\n" + "❌"*40)
            print("❌ NOT IN RAG - PROCEEDING WITH FULL RESEARCH")
            print("❌"*40 + "\n")
            rag_response = None  # Clear it if no valid data
            
    except Exception as e:
        # If RAG fails, log it but continue with normal flow
        print("\n" + "⚠️"*40)
        print("⚠️ RAG SYSTEM ERROR")
        print("⚠️"*40)
        print(f"⚠️ Error: {e}")
        print(f"⚠️ Error Type: {type(e).__name__}")
        print("⚠️ Continuing with normal research flow...")
        print("⚠️"*40 + "\n")
        rag_response = None
    
    # ============================================
    # STEP 2: Proceed with research (with RAG context if available)
    # ============================================
    print("\n" + "🔬"*40)
    print("🔬 STARTING RESEARCH VALIDATION")
    print("🔬"*40 + "\n")
    
    result = await asyncio.to_thread(validate_research, query)
    
    # ============================================
    # SCORE CHECK: Skip if score is 0
    # ============================================
    score = result.get("score", "0/3")
    score_value = float(score.split("/")[0]) if "/" in str(score) else 0
    if score_value == 0:
        print(f"\n❌ Score is 0/3 for query: '{query}' — skipping entirely")
        return None
    
    location = result.get("location")

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    maps_data = None
    if location and api_key:
        print(f"🗺️  Looking up Google Maps data for: {location}")
        maps_data = await get_google_maps_data(location, api_key)
        if maps_data:
            print(f"✅ Google Maps data retrieved successfully")
        else:
            print(f"🗺️  Looking up Google Maps data for: {location}")
    elif not api_key:
        print("⚠️  Warning: GOOGLE_MAPS_API_KEY not set. Skipping maps lookup.")

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

    print("\n" + "🔄"*40)
    print("🔄 CONVERSION INPUT PREPARED")
    print("🔄"*40)
    if rag_response:
        print("✅ RAG context included - OpenAI will exclude this content")
        print(f"📋 RAG Context Keys: {list(rag_response.keys())}")
    else:
        print("ℹ️  No RAG context - Full research will be converted")
    print("🔄"*40 + "\n")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    try:
        print("🤖 Calling OpenAI for conversion...")
        formatted_data_list = await asyncio.to_thread(
            convert_research_to_attraction,
            conversion_input,
            openai_api_key,
        )

        if formatted_data_list:
            formatted_data = formatted_data_list[0].model_dump()

            # ============================================
            # SCORE-BASED ROUTING:
            # Score >= 2: Save to Supabase AND upsert to RAG
            # Score == 1: Only save to Supabase
            # Score == 0: Already skipped above
            # ============================================
            print(f"\n{'📊'*40}")
            print(f"📊 SCORE-BASED ROUTING")
            print(f"{'📊'*40}")
            print(f"📊 Score Value: {score_value}/3")

            if score_value >= 2:
                print(f"📊 Action: Upsert to RAG only (skip database)")
                print(f"{'📊'*40}\n")

                # Upsert to RAG only - no database insert
                rag_success = await upsert_to_rag(formatted_data, maps_data)
                formatted_data["rag_upserted"] = rag_success
                formatted_data["db_id"] = None
                if rag_success:
                    print(f"✅ Successfully upserted to RAG")
                else:
                    print(f"⚠️  Failed to upsert to RAG")

            elif score_value >= 1:
                print(f"📊 Action: Save to Supabase only (score too low for RAG)")
                print(f"{'📊'*40}\n")

                # Only save to Supabase
                try:
                    db_record = await supabase_service.insert_research_insight(formatted_data)
                    formatted_data["db_id"] = db_record.get("id")
                    formatted_data["created_at"] = db_record.get("created_at")
                    print(f"✅ Successfully saved to Supabase with ID: {db_record.get('id')}")
                except Exception as e:
                    formatted_data["db_id"] = None
                    formatted_data["db_error"] = str(e)
                    print(f"⚠️  Error saving to Supabase: {e}")

                formatted_data["rag_upserted"] = False
            else:
                print(f"📊 Action: Skip (score is 0)")
                print(f"{'📊'*40}\n")
                formatted_data["rag_upserted"] = False

            return formatted_data
        return conversion_input["results"][0]
    except Exception as e:
        print(f"\n❌ Error converting data with OpenAI: {e}")
        return conversion_input["results"][0]


# -------------------------------------------------------
# Background Processing
# -------------------------------------------------------
async def process_in_background(query: str, reference: str):
    """Process the research in the background."""
    try:
        print(f"\n{'='*80}")
        print(f"🚀 Background Research Started")
        print(f"Query: {query}")
        print(f"Reference: {reference}")
        print(f"{'='*80}\n")
        
        openai_service = OpenAIService()
        print("✅ OpenAI service initialized")
        
        supabase_service = SupabaseService()
        print("✅ Supabase service initialized")

        classification = await openai_service.classify_query(query)
        print(f"✅ Classification completed: {classification.get('type')}")
        
        formatted_results: List[Dict[str, Any]] = []

        if classification.get("queries"):
            if len(classification["queries"]) == 1:
                result = await process_query_research(
                    classification["queries"][0],
                    query,
                    classification["type"],
                    supabase_service,
                    reference
                )
                print("✅ Single query research completed")
                if result is not None:
                    formatted_results.append(result)
            else:
                tasks = [
                    process_query_research(q, query, classification["type"], supabase_service, reference)
                    for q in classification["queries"]
                ]
                all_results = await asyncio.gather(*tasks)
                formatted_results = [r for r in all_results if r is not None]
                print(f"✅ Multiple queries research completed: {len(formatted_results)} results")

        print(f"\n{'='*80}")
        print(f"✅ Background Research Completed Successfully")
        print(f"{'='*80}")
        print(f"Type: {classification.get('type')}")
        print(f"Results: {len(formatted_results)} items processed")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ Background Research Failed")
        print(f"{'='*80}")
        print(f"Error: {str(e)}")
        print(f"Error Type: {type(e).__name__}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        print(f"{'='*80}\n")


# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@router.get("/")
async def validator_root():
    """Root endpoint with API information."""
    return {
        "message": "Travel Query Validator API",
        "version": "2.0.0",
        "endpoints": {
            "validator": "/validator (POST)",
            "insights": "/insights (GET)",
            "insights_by_location": "/insights/location (GET)",
            "insights_by_category": "/insights/category (GET)",
            "health": "/health (GET)",
            "examples": "/examples (GET)",
            "rag_upsert": "/rag/upsert (POST)",
        },
    }


@router.get("/health")
async def validator_health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Travel Query Validator API",
        "database": "Supabase Connected",
    }


@router.post("/process")
async def validate_query(request: ValidatorRequest):
    """Main validation endpoint - checks for duplicate query, then starts research in background."""
    print(f"\n{'='*80}")
    print("📨 Request Received")
    print(f"{'='*80}")
    print(f"Query: {request.query}")
    print(f"Reference: {request.reference}")
    print(f"{'='*80}\n")

    try:
        # ── Step 1: Duplicate check ───────────────────────────
        supabase_service = SupabaseService()
        is_dup = await is_duplicate_query(request.query, supabase_service)

        if is_dup:
            print("🔁 Query is a duplicate — skipping research\n")
            return {
                "message": "Similar query already exists. Skipping research.",
                "duplicate": True,
            }

        # ── Step 2: Save the new query ────────────────────────
        try:
            await supabase_service.insert_saved_query(request.query)
            print(f"✅ Query saved to saved_queries table")
        except Exception as e:
            print(f"⚠️  Could not save query (continuing anyway): {e}")

        # ── Step 3: Start background research ─────────────────
        asyncio.create_task(
            process_in_background(
                query=request.query,
                reference=request.reference,
            )
        )

        print("✅ Background task started, returning response to client\n")
        return {
            "message": "Research has started",
            "duplicate": False,
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting research: {str(e)}",
        )
        
        
@router.get("/saved-queries")
async def get_saved_queries():
    """Return all non-expired saved queries."""
    try:
        supabase_service = SupabaseService()
        rows = await supabase_service.get_all_saved_queries()
        return {
            "count": len(rows),
            "queries": rows,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching saved queries: {str(e)}")
    
@router.delete("/saved-queries")
async def delete_all_saved_queries():
    """Delete all records from saved_queries table."""
    try:
        supabase_service = SupabaseService()
        count = await supabase_service.delete_all_saved_queries()
        return {
            "message": "All saved queries deleted",
            "deleted_count": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting saved queries: {str(e)}")


# -------------------------------------------------------
# RAG Upsert Endpoint
# -------------------------------------------------------
@router.post("/rag/upsert", response_model=RAGUpsertResponse)
async def rag_upsert(request: RAGUpsertRequest):
    """
    Upsert a single document into the RAG index.

    This endpoint validates the payload and forwards it to the RAG system
    at https://rag.hiptraveler.com/upsert
    """
    print(f"\n{'='*80}")
    print("📤 RAG UPSERT REQUEST RECEIVED")
    print(f"{'='*80}")
    print(f"ID: {request.id}")
    print(f"Title: {request.title}")
    print(f"Category: {request.category}")
    print(f"Location: {request.city}, {request.country}")
    print(f"Query: {request.query}")
    print(f"{'='*80}\n")

    # Validate language code
    allowed_languages = ["en", "fr", "it", "de", "es", "zh"]
    language = request.language if request.language in allowed_languages else "en"

    # Build the payload for RAG API
    payload = {
        "id": request.id,
        "title": request.title,
        "content": request.content,
        "query": request.query,
        "category": request.category,
        "country": request.country,
        "city": request.city,
        "meta_obj": {
            "address": request.meta_obj.address
        },
        "updated_at": request.updated_at,
        "language": language
    }

    # Add optional fields if provided
    if request.tags:
        payload["tags"] = request.tags
    if request.source:
        payload["source"] = request.source
    if request.region_code:
        payload["region_code"] = request.region_code
    if request.latitude:
        payload["latitude"] = request.latitude
    if request.longitude:
        payload["longitude"] = request.longitude

    # Add optional meta_obj fields
    if request.meta_obj.ranking:
        payload["meta_obj"]["ranking"] = request.meta_obj.ranking
    if request.meta_obj.audience:
        payload["meta_obj"]["audience"] = request.meta_obj.audience
    if request.meta_obj.priceLevel:
        payload["meta_obj"]["priceLevel"] = request.meta_obj.priceLevel
    if request.meta_obj.hours:
        payload["meta_obj"]["hours"] = request.meta_obj.hours
    if request.meta_obj.rating is not None:
        payload["meta_obj"]["rating"] = request.meta_obj.rating
    if request.meta_obj.numReviews is not None:
        payload["meta_obj"]["numReviews"] = request.meta_obj.numReviews

    print(f"📦 Payload to send to RAG:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    # Send to RAG API
    rag_url = "https://rag.hiptraveler.com/upsert"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(rag_url, json=payload, headers=headers)
            response.raise_for_status()

            print(f"\n✅ RAG UPSERT SUCCESS")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}\n")

            return RAGUpsertResponse(
                success=True,
                message="Document upserted successfully",
                id=request.id
            )

    except httpx.HTTPStatusError as e:
        error_msg = f"RAG API returned error: {e.response.status_code} - {e.response.text}"
        print(f"\n❌ RAG UPSERT FAILED: {error_msg}\n")
        raise HTTPException(status_code=e.response.status_code, detail=error_msg)

    except httpx.TimeoutException:
        error_msg = "RAG API request timed out after 30 seconds"
        print(f"\n❌ RAG UPSERT FAILED: {error_msg}\n")
        raise HTTPException(status_code=504, detail=error_msg)

    except Exception as e:
        error_msg = f"Failed to upsert to RAG: {str(e)}"
        print(f"\n❌ RAG UPSERT FAILED: {error_msg}\n")
        raise HTTPException(status_code=500, detail=error_msg)