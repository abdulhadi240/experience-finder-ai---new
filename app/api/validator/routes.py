"""
Validator API routes — integrated into main FastAPI app.
Handles classification, research validation, OpenAI conversion, and Supabase storage.
"""
import asyncio
import json

from fastapi import APIRouter, HTTPException
from app.api.validator.models.schemas import ValidatorRequest, RAGUpsertRequest, RAGUpsertResponse
from app.api.validator.services.supabase_service import SupabaseService
from app.api.validator.helpers import is_duplicate_query, process_in_background, process_deep_validation

import httpx

router = APIRouter()


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
        # Extract the question-only portion for dedup/storage.
        # Web search path sends "question\n\nAnswer:\n..." — we only store the question.
        question_only = request.query.split("\n\nAnswer:\n")[0].strip()

        # ── Step 1: Duplicate check (question only) ───────────
        supabase_service = SupabaseService()
        is_dup = await is_duplicate_query(question_only, supabase_service)

        if is_dup:
            print("🔁 Query is a duplicate — skipping research\n")
            return {
                "message": "Similar query already exists. Skipping research.",
                "duplicate": True,
            }

        # ── Step 2: Save the question only ────────────────────
        try:
            await supabase_service.insert_saved_query(question_only)
            print(f"✅ Query saved to saved_queries table")
        except Exception as e:
            print(f"⚠️  Could not save query (continuing anyway): {e}")

        # ── Step 3: Start background research ─────────────────
        # Pass the FULL query (with Answer: block if present) so the classifier
        # can extract the specific place names the web search returned.
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


@router.post("/deep_validation")
async def deep_validation(request: ValidatorRequest):
    """Deep validation endpoint - runs the full research pipeline synchronously
    and returns JSON results only for items with score >= 2."""
    print(f"\n{'='*80}")
    print("🔬 Deep Validation Request Received")
    print(f"{'='*80}")
    print(f"Query: {request.query}")
    print(f"Reference: {request.reference}")
    print(f"{'='*80}\n")

    try:
        high_score_results = await process_deep_validation(
            query=request.query,
            reference=request.reference,
        )

        _strip = {"db_id", "score_value", "rag_upserted"}
        clean_results = [
            {k: v for k, v in r.items() if k not in _strip}
            for r in high_score_results
        ]

        return {
            "query": request.query,
            "reference": request.reference,
            "count": len(clean_results),
            "results": clean_results,
        }

    except Exception as e:
        print(f"❌ Deep Validation Error: {str(e)}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Error during deep validation: {str(e)}",
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
    if request.image:
        payload["image"] = request.image

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
        "Accept": "application/json",
        "Authorization": "Bearer 96717f5a-e562-4fd5-a14c-499794e8328f"
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
