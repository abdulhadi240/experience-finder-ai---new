import os
import asyncio
from supabase import create_client
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv


class SupabaseService:
    """Service for interacting with Supabase database"""

    def __init__(self):
        """Initialize Supabase client"""
        load_dotenv()

        # ✅ Use SUPABASE_URL instead of PROJECT_ID
        supabase_url = os.getenv("SUPABASE_URL")
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not service_role_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

        # ✅ Correct v2.8.1 initialization (no Client import)
        self.client = create_client(supabase_url, service_role_key)
        self.table_name = "research_insights"

    async def insert_research_insight(self, attraction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a single research insight into Supabase"""
        insert_data = {
            "query": attraction_data.get("query"),
            "title": attraction_data.get("title"),
            "content": attraction_data.get("content"),
            "category": attraction_data.get("category"),
            "country": attraction_data.get("country"),
            "city": attraction_data.get("city"),
            "region_code": attraction_data.get("region_code"),
            "latitude": float(attraction_data.get("latitude")) if attraction_data.get("latitude") else None,
            "longitude": float(attraction_data.get("longitude")) if attraction_data.get("longitude") else None,
            "language": attraction_data.get("language", "en"),
            "tags": attraction_data.get("tags"),
            "source": "web search",
            "meta_obj": attraction_data.get("meta_obj"),
        }
        insert_data = {k: v for k, v in insert_data.items() if v is not None}

        try:
            # ✅ Supabase v2 insert pattern
            response = await asyncio.to_thread(
                lambda: self.client.table(self.table_name).insert(insert_data).execute()
            )
            if response.data:
                return response.data[0]
            raise RuntimeError("No data returned from insert operation")
        except Exception as e:
            print(f"Error inserting into Supabase: {e}")
            raise

    async def get_insights_by_query(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve research insights by query text"""
        try:
            response = await asyncio.to_thread(
                lambda: self.client.table(self.table_name)
                .select("*")
                .ilike("query", f"%{query}%")
                .execute()
            )
            return response.data or []
        except Exception as e:
            print(f"Error fetching from Supabase: {e}")
            return []
