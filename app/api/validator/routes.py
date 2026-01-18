"""
Validator API routes — integrated into main FastAPI app.
Handles classification, research validation, OpenAI conversion, and Supabase storage.
"""
import os
import asyncio
import httpx
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from app.api.validator.models.schemas import ValidatorRequest
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
    location = result.get("location")

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    maps_data = None
    if location and api_key:
        print(f"🗺️  Looking up Google Maps data for: {location}")
        maps_data = await get_google_maps_data(location, api_key)
        if maps_data:
            print(f"✅ Google Maps data retrieved successfully")
        else:
            print(f"❌ Google Maps lookup failed")
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
            try:
                db_record = await supabase_service.insert_research_insight(formatted_data)
                formatted_data["db_id"] = db_record.get("id")
                formatted_data["created_at"] = db_record.get("created_at")
                print(f"\n✅ Successfully saved to Supabase with ID: {db_record.get('id')}")
            except Exception as e:
                formatted_data["db_id"] = None
                formatted_data["db_error"] = str(e)
                print(f"\n⚠️  Error saving to Supabase: {e}")
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
                formatted_results.append(result)
            else:
                tasks = [
                    process_query_research(q, query, classification["type"], supabase_service, reference)
                    for q in classification["queries"]
                ]
                formatted_results = await asyncio.gather(*tasks)
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
    """Main validation endpoint - starts research in background and returns immediately."""
    print(f"\n{'='*80}")
    print("📨 Request Received")
    print(f"{'='*80}")
    print(f"Query: {request.query}")
    print(f"Reference: {request.reference}")
    print(f"{'='*80}\n")
    
    try:
        # Start the background task
        asyncio.create_task(
            process_in_background(
                query=request.query,
                reference=request.reference
            )
        )
        
        # Return immediately
        print("✅ Background task started, returning response to client\n")
        return {
            "message": "Research has started"
        }

    except Exception as e:
        print(f"❌ Error starting background task: {str(e)}\n")
        raise HTTPException(
            status_code=500, 
            detail=f"Error starting research: {str(e)}"
        )