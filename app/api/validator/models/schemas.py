"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List
from enum import Enum
from typing import List, Optional, Literal
from datetime import datetime

class QueryType(str, Enum):
    """Enum for query classification types"""
    GENERIC = "generic"
    SPECIFIC = "specific"
    IGNORE = "ignore"


class QueryClassification(BaseModel):
    """
    Model for OpenAI structured output.
    Used to parse and validate OpenAI's response.
    """
    type: QueryType = Field(
        description="Type of query: 'generic' for broad travel queries, 'specific' for particular places/services, 'ignore' for time-sensitive or irrelevant queries"
    )
    queries: List[str] = Field(
        description="List of rewritten queries. For generic: 5 sub-queries. For specific: 1 rewritten query. For ignore: empty list"
    )


class ValidatorRequest(BaseModel):
    """
    Request model for validator endpoint
    """
    query: str = Field(..., description="The user's travel query to validate and classify")
    reference: str = Field(... , description="The reference given by the user")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the best places to visit in Karachi?",
                "reference": "hiptraveler"
            }
        }


class ResearchResult(BaseModel):
    """
    Research validation result for a single query
    """
    query: str = Field(..., description="The query that was researched")
    score: str = Field(..., description="Similarity score out of 3")
    research: str = Field(..., description="Combined research from all sources")
    citations: List[str] = Field(default_factory=list, description="List of citation URLs")
    location: Optional[str] = Field(None, description="Location (e.g., [Name city country]) or null") 
    maps_data: Optional[Dict[str, Any]] = None     
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Best historical sites in Karachi",
                "score": "2.5/3",
                "research": "Karachi has several notable historical sites including...",
                "citations": [
                    "https://example.com/karachi-history",
                    "https://example.com/historical-sites"
                ],
                "location": "Karachi, Pakistan",
                "maps_data":""
            }
        }


class ValidatorResponse(BaseModel):
    """
    Response model for validator endpoint with research results
    """
    type: str = Field(..., description="Classification type: 'generic', 'specific', or 'ignore'")
    original_query: str = Field(..., description="The original query submitted by user")
    queries: List[str] = Field(default_factory=list, description="Generated or rewritten queries")
    results: List[ResearchResult] = Field(default_factory=list, description="Research validation results for each query")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "generic",
                "original_query": "What are the best places to visit in Karachi?",
                "queries": [
                    "Best historical sites in Karachi",
                    "Top beaches in Karachi",
                    "Family-friendly attractions in Karachi"
                ],
                "results": [
                    {
                        "query": "Best historical sites in Karachi",
                        "score": "2.5/3",
                        "research": "Karachi has several notable historical sites...",
                        "citations": ["https://example.com/source1"]
                    },
                    {
                        "query": "Top beaches in Karachi",
                        "score": "2.8/3",
                        "research": "Karachi's coastline features beautiful beaches...",
                        "citations": ["https://example.com/source2"]
                    }
                ]
            }
        }
        
        
class MetaObject(BaseModel):
    audience: List[str]
    location: str
    ranking: Optional[str] = None
    price_level: Optional[str] = None

class AttractionOutput(BaseModel):
    country: str
    city: str
    meta_obj: MetaObject
    latitude: str
    language: str
    id: str
    category: str
    source: str
    title: str
    content: str
    region_code: str
    tags: str
    longitude: str
    query: str


# -------------------------------------------------------
# RAG Upsert Models
# -------------------------------------------------------
class LanguageCode(str, Enum):
    """Allowed language codes for RAG upsert"""
    EN = "en"
    FR = "fr"
    IT = "it"
    DE = "de"
    ES = "es"
    ZH = "zh"


class RAGMetaObject(BaseModel):
    """Meta object for RAG upsert request"""
    address: str = Field(..., description="Required address field")
    ranking: Optional[str] = Field(None, description="Ranking info (e.g., '#1 of 10 Things to Do')")
    audience: Optional[List[str]] = Field(None, description="Target audience (e.g., ['Romantic', 'Families'])")
    priceLevel: Optional[str] = Field(None, description="Price level (e.g., '$', '$$', '$$$', '$$$$')")
    hours: Optional[Dict[str, str]] = Field(None, description="Operating hours as key-value pairs")
    rating: Optional[float] = Field(None, description="Average rating (double)")
    numReviews: Optional[int] = Field(None, description="Number of reviews")

    class Config:
        json_schema_extra = {
            "example": {
                "address": "Sandy Point, Great Abaco Island, Bahamas",
                "ranking": "#1 of 5 Things to Do",
                "audience": ["Romantic", "Families"],
                "priceLevel": "$$",
                "hours": {"Monday": "9:00 AM - 5:00 PM"},
                "rating": 4.5,
                "numReviews": 120
            }
        }


class RAGUpsertRequest(BaseModel):
    """Request model for RAG upsert endpoint"""
    id: str = Field(..., description="Unique identifier for the document (UUID acceptable)")
    title: str = Field(..., description="Title of the document")
    content: str = Field(..., description="Main content/description")
    tags: Optional[str] = Field(None, description="Comma-separated tags (not an array)")
    source: Optional[str] = Field(None, description="URL, source identifier, or Google Place ID")
    query: str = Field(..., description="The query associated with this upsert record")
    category: str = Field(..., description="Category: place, tour, restaurant, hotel")
    country: str = Field(..., description="Country name")
    city: str = Field(..., description="City name")
    region_code: Optional[str] = Field(None, description="Region code if available")
    meta_obj: RAGMetaObject = Field(..., description="Metadata object with address and optional fields")
    updated_at: str = Field(..., description="ISO-8601 timestamp (e.g., 2026-02-03T18:45:12Z)")
    latitude: Optional[str] = Field(None, description="Latitude as string")
    longitude: Optional[str] = Field(None, description="Longitude as string")
    language: Optional[str] = Field("en", description="Language code: en, fr, it, de, es, zh. Defaults to 'en'")
    image: Optional[str] = Field(None, description="Image URL for the place (e.g. from Google Places Photo API)")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Sandy Point Beach",
                "content": "A beautiful secluded beach on Great Abaco Island...",
                "tags": "beach,romantic,nature,bahamas",
                "source": "https://example.com/sandy-point",
                "query": "best beaches in Bahamas",
                "category": "place",
                "country": "Bahamas",
                "city": "Great Abaco Island",
                "region_code": "ABA",
                "meta_obj": {
                    "address": "Sandy Point, Great Abaco Island, Abaco Islands, Bahamas",
                    "audience": ["Romantic", "Nature Lovers"],
                    "priceLevel": "$",
                    "rating": 4.8,
                    "numReviews": 85
                },
                "updated_at": "2026-02-03T18:45:12Z",
                "latitude": "26.0167",
                "longitude": "-77.3833",
                "language": "en"
            }
        }


class RAGUpsertResponse(BaseModel):
    """Response model for RAG upsert endpoint"""
    success: bool = Field(..., description="Whether the upsert was successful")
    message: str = Field(..., description="Status message")
    id: Optional[str] = Field(None, description="ID of the upserted document")
    error: Optional[str] = Field(None, description="Error message if failed")


class FrontendRAGMetaObject(BaseModel):
    """Meta object sent by the frontend for RAG upsert"""
    location: str = Field(..., description="Address/location string")
    audience: Optional[List[str]] = Field(None, description="Target audience list")
    rating: Optional[float] = Field(None, description="Average rating")
    numReviews: Optional[int] = Field(None, description="Number of reviews")
    ranking: Optional[str] = Field(None, description="Ranking info")
    priceLevel: Optional[str] = Field(None, description="Price level")
    hours: Optional[Dict[str, str]] = Field(None, description="Operating hours")


class FrontendRAGUpsertRequest(BaseModel):
    """
    Request model for the frontend RAG upsert endpoint.
    updated_at is auto-generated by the backend — do not send it.
    """
    id: str = Field(..., description="Unique identifier (e.g. TripAdvisor ID or UUID)")
    title: str = Field(..., description="Title of the place")
    content: str = Field(..., description="Main content/description")
    query: str = Field(..., description="Query for RAG indexing")
    category: str = Field(..., description="Category e.g. Hip Place, restaurant, hotel")
    country: str = Field(..., description="Country name")
    city: str = Field(..., description="City name")
    meta_obj: FrontendRAGMetaObject = Field(..., description="Metadata object")
    latitude: Optional[str] = Field(None, description="Latitude as string")
    longitude: Optional[str] = Field(None, description="Longitude as string")
    language: Optional[str] = Field("en", description="Language code: en, fr, it, de, es, zh")
    tags: Optional[str] = Field(None, description="Comma-separated tags")
    source: Optional[str] = Field(None, description="Source URL or identifier")
    region_code: Optional[str] = Field(None, description="Region code")
    image: Optional[str] = Field(None, description="Image URL")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "g295414-d2419975",
                "title": "Port Grand Pakistan",
                "content": "Port Grand is a vibrant waterfront entertainment and dining complex...",
                "query": "Port Grand Pakistan Karachi",
                "category": "Hip Place",
                "country": "Pakistan",
                "city": "Karachi",
                "meta_obj": {
                    "location": "Port Grand Food Street road opposite PNSC Building, Karachi 74000 Pakistan",
                    "audience": ["FAMILY", "FRIENDS GETAWAY"],
                    "rating": 4.3,
                    "numReviews": 343
                },
                "latitude": "24.844986",
                "longitude": "66.99099",
                "language": "en",
                "tags": "Shopping Malls, Cultural landmark, Scenic Outlook",
                "source": "https://www.tripadvisor.com/Attraction_Review-g295414-d2419975-Reviews-Port_Grand_Pakistan-Karachi_Sindh_Province.html",
                "region_code": "",
                "image": "https://media-cdn.tripadvisor.com/media/photo-m/1280/15/32/e5/a3/a-grand-view-of-our-port.jpg"
            }
        }
