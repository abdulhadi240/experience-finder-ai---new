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
        
        
    # -------------------------------------------------------
    # saved_queries table helpers
    # -------------------------------------------------------
    async def get_all_saved_queries(self) -> List[Dict[str, Any]]:
        """Fetch all non-expired rows from saved_queries."""
        try:
            from datetime import datetime, timezone
            now_iso = datetime.now(timezone.utc).isoformat()
            response = await asyncio.to_thread(
                lambda: self.client.table("saved_queries")
                .select("*")
                .gte("expiry_at", now_iso)
                .execute()
            )
            return response.data or []
        except Exception as e:
            print(f"Error fetching saved_queries: {e}")
            return []

    async def insert_saved_query(self, query_text: str, expiry_days: int = 7) -> Dict[str, Any]:
        """Insert a new query into saved_queries with an expiry."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        expiry_at = now + timedelta(days=expiry_days)
        insert_data = {
            "query_text": query_text,
            "expiry_days": expiry_days,
            "expiry_at": expiry_at.isoformat(),
        }
        try:
            response = await asyncio.to_thread(
                lambda: self.client.table("saved_queries").insert(insert_data).execute()
            )
            if response.data:
                return response.data[0]
            raise RuntimeError("No data returned from insert operation")
        except Exception as e:
            print(f"Error inserting into saved_queries: {e}")
            raise
        
    async def delete_all_saved_queries(self) -> int:
        """Delete all rows from saved_queries table."""
        try:
            response = await asyncio.to_thread(
                lambda: self.client.table("saved_queries")
                .delete()
                .neq("id", "00000000-0000-0000-0000-000000000000")
                .execute()
            )
            deleted = len(response.data) if response.data else 0
            print(f"🗑️ Deleted {deleted} rows from saved_queries")
            return deleted
        except Exception as e:
            print(f"Error deleting saved_queries: {e}")
            raise

    # -------------------------------------------------------
    # whitelist_domains table helpers
    # -------------------------------------------------------
    async def get_whitelist_domains(
        self, category: Optional[str] = None, country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch verified whitelist domains.
        Optionally filter by category and/or country (case-insensitive).
        Returns a list of rows: {id, category, country, source, verified_at, created_at}
        """
        try:
            query = self.client.table("whitelist_domains").select("*")
            if category:
                query = query.eq("category", category.lower())
            if country:
                query = query.eq("country", country.lower())
            response = await asyncio.to_thread(lambda: query.execute())
            return response.data or []
        except Exception as e:
            print(f"Error fetching whitelist_domains: {e}")
            return []

    async def is_source_whitelisted(
        self, source: str, category: str, country: str
    ) -> bool:
        """
        Return True if the given domain is in the whitelist for category + country.
        """
        rows = await self.get_whitelist_domains(category=category, country=country)
        return source.lower() in {r["source"] for r in rows}

    async def insert_whitelist_domain(
        self, category: str, country: str, source: str
    ) -> Dict[str, Any]:
        """
        Directly upsert a single domain into whitelist_domains.
        ON CONFLICT (category, country, source) updates verified_at.
        """
        from datetime import datetime, timezone
        row = {
            "category": category.lower(),
            "country": country.lower(),
            "source": source.lower(),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            response = await asyncio.to_thread(
                lambda: self.client.table("whitelist_domains")
                .upsert(row, on_conflict="category,country,source")
                .execute()
            )
            if response.data:
                return response.data[0]
            raise RuntimeError("No data returned from upsert operation")
        except Exception as e:
            print(f"Error upserting whitelist_domain: {e}")
            raise

    async def delete_whitelist_domain(
        self, category: str, country: str, source: str
    ) -> int:
        """Delete a specific domain from whitelist_domains. Returns deleted count."""
        try:
            response = await asyncio.to_thread(
                lambda: self.client.table("whitelist_domains")
                .delete()
                .eq("category", category.lower())
                .eq("country", country.lower())
                .eq("source", source.lower())
                .execute()
            )
            return len(response.data) if response.data else 0
        except Exception as e:
            print(f"Error deleting whitelist_domain: {e}")
            raise
