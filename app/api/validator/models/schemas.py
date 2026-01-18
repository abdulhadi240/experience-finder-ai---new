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
