import os
import requests
import json
from typing import Dict, List, Any
import time
import concurrent.futures
from dotenv import load_dotenv
from app.api.validator.services.supabase_service import SupabaseService
from app.api.validator.services.whitelist_service import (
    _extract_category,
    _extract_country_from_query,
    fetch_whitelist_domains,
)
from supabase import create_client
import asyncio



class ResearchValidator:
    def __init__(self):
        load_dotenv()
        """Initialize the validator with API keys from environment variables."""
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not all([self.openai_key, self.perplexity_key, self.gemini_key]):
            raise ValueError(
                "Missing API keys. Please set OPENAI_API_KEY, PERPLEXITY_API_KEY, and GEMINI_API_KEY environment variables."
            )

        # Initialize Supabase client
        self.supabase_service = create_client(self.supabase_url, self.service_role_key)
        self.blacklist_domains: List[str] = []  # initialize empty list

        # Immediately fetch and print domains
        try:
            asyncio.run(self._initialize_blacklist_domains())
        except Exception as e:
            print(f"⚠️ Error initializing blacklist domains: {e}")

        # Whitelist domains are fetched per-query (category + country specific)
        # so we only store the current query's whitelist here temporarily.
        self._current_whitelist: List[str] = []

    async def get_all_blacklist_domains(self) -> List[str]:
        """Retrieve all blacklist domains from the Supabase table 'blacklist_domains'"""
        try:
            response = await asyncio.to_thread(
                lambda: self.supabase_service.table("blacklist").select("domain").execute()
            )
            return [row["domain"] for row in response.data] if response.data else []
        except Exception as e:
            print(f"⚠️ Error fetching blacklist domains: {e}")
            return []

    async def _initialize_blacklist_domains(self):
        """Fetch blacklist domains and print them immediately"""
        self.blacklist_domains = await self.get_all_blacklist_domains()
        if self.blacklist_domains:
            print(f"🔒 Loaded {len(self.blacklist_domains)} blacklist domains:")
            for domain in self.blacklist_domains:
                print(f"  - {domain}")
        else:
            print("🔒 No blacklist domains found.")

        
        
    def search_openai(self, query: str) -> Dict[str, Any]:
        """Perform web search using OpenAI API."""
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}",
        }

        # Construct blacklist note
        if self.blacklist_domains:
            blocked = ", ".join(self.blacklist_domains)
            blacklist_note = f"\nDO NOT USE any of the following sources or domains under any circumstance: {blocked}."
        else:
            blacklist_note = ""

        # Construct whitelist note — these are verified trusted sources for this query
        if self._current_whitelist:
            trusted = ", ".join(self._current_whitelist)
            whitelist_note = (
                f"\nTRUSTED SOURCES: The following domains have been verified as the most "
                f"authoritative sources for this category and country. Prioritise them when "
                f"available, in addition to Tripadvisor and Yelp: {trusted}."
            )
        else:
            whitelist_note = ""

        system_prompt = (
            "You are an expert internet research assistant. Your primary goal is to gather accurate, up-to-date, and verifiable information from the web about locations, attractions, and businesses. "
            "Prioritize sources in this order: 1) Tripadvisor, 2) Yelp, 3) official or reputable travel/review sites (Google Travel, Lonely Planet, etc.). "
            "For each entity, collect: name, location, ranking info (e.g., '#1 of 1 Things to Do in Sandy Point'), price level ($-$$$$), average rating, top reviews or highlights, category/type, and source URLs for citation. "
            "Present findings in structured format (bullets or table), cite Tripadvisor or Yelp first, list top 3 results if relevant, and note when primary sources are unavailable. "
            "Be factual, concise, and analytical; extract ranking and price level verbatim from the source; use the most recent data available."
            + whitelist_note
            + blacklist_note
        )
        
        data = {
            "model": "gpt-4.1",
            "tools": [{"type": "web_search"}],
            "tool_choice": "auto",
            "include": ["web_search_call.action.sources"],
            "input": query + system_prompt
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "source": "OpenAI",
                    "data": result,
                    "content": self._extract_openai_content(result),
                    "citations": self._extract_openai_citations(result)
                }
            else:
                return {
                    "success": False,
                    "source": "OpenAI",
                    "error": f"Error {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "source": "OpenAI",
                "error": str(e)
            }
    
    def search_perplexity(self, query: str) -> Dict[str, Any]:
        """Perform web search using Perplexity API."""
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.perplexity_key}",
            "Content-Type": "application/json",
        }

        blacklist_note = ""
        if self.blacklist_domains:
            blocked = ", ".join(self.blacklist_domains)
            blacklist_note = f" Avoid these domains: {blocked}."

        whitelist_note = ""
        if self._current_whitelist:
            trusted = ", ".join(self._current_whitelist)
            whitelist_note = (
                f" TRUSTED SOURCES: The following domains are verified as the most "
                f"authoritative for this category and country — prioritise them alongside "
                f"Tripadvisor and Yelp when available: {trusted}."
            )

        system_content = (
            "You are an expert internet research assistant. Your primary goal is to gather accurate, up-to-date, and verifiable information from the web about locations, attractions, and businesses. "
            "Prioritize sources in this order: 1) Tripadvisor, 2) Yelp, 3) official or reputable travel/review sites (Google Travel, Lonely Planet, etc.). "
            "For each entity, collect: name, location, ranking info (e.g., '#1 of 1 Things to Do in Sandy Point'), price level ($-$$$$), average rating, top reviews or highlights, category/type, and source URLs for citation. "
            "Present findings in structured format (bullets or table), cite Tripadvisor or Yelp first, list top 3 results if relevant, and note when primary sources are unavailable. "
            "Be factual, concise, and analytical; extract ranking and price level verbatim from the source; use the most recent data available."
            + whitelist_note
            + blacklist_note
        )

        data = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ],
            "return_citations": True,
            "return_related_questions": False,
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                citations_list = result.get("citations", [])
                citations = []
                for c in citations_list:
                    if isinstance(c, str):
                        citations.append(c)
                    elif isinstance(c, dict) and c.get('url'):
                        citations.append(c['url'])
                
                return {
                    "success": True,
                    "source": "Perplexity",
                    "data": result,
                    "content": content,
                    "citations": citations
                }
            else:
                return {
                    "success": False,
                    "source": "Perplexity",
                    "error": f"Error {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "source": "Perplexity",
                "error": str(e)
            }
    
    def search_gemini(self, query: str) -> Dict[str, Any]:
        """Perform web search using Google Gemini API with Google Search grounding."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={self.gemini_key}"
        )
        headers = {"Content-Type": "application/json"}

        if self.blacklist_domains:
            blocked = ", ".join(self.blacklist_domains)
            blacklist_note = f"\nDO NOT reference or cite any content from these domains: {blocked}."
        else:
            blacklist_note = ""

        if self._current_whitelist:
            trusted = ", ".join(self._current_whitelist)
            whitelist_note = (
                f"\nTRUSTED SOURCES: The following domains are verified as the most authoritative "
                f"for this category and country — prioritise them alongside Tripadvisor and Yelp "
                f"when available: {trusted}."
            )
        else:
            whitelist_note = ""

        system_prompt = (
            "You are an expert internet research assistant. Gather accurate, up-to-date, and verifiable "
            "information from the web about locations, attractions, and businesses. "
            "Prioritize sources in this order: 1) Tripadvisor, 2) Yelp, 3) official or reputable travel/review sites "
            "(Google Travel, Lonely Planet, etc.). "
            "For each entity, collect: name, location, ranking info (e.g., '#1 of 1 Things to Do in Sandy Point'), "
            "price level ($-$$$$), average rating, top reviews or highlights, category/type, and source URLs for citation. "
            "Present findings in structured format, cite Tripadvisor or Yelp first, and note when primary sources "
            "are unavailable. Be factual, concise, and analytical; use the most recent data available."
            + whitelist_note
            + blacklist_note
        )

        data = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": query}], "role": "user"}],
            "tools": [{"google_search": {}}],
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "source": "Gemini",
                    "data": result,
                    "content": self._extract_gemini_content(result),
                    "citations": self._extract_gemini_citations(result),
                }
            else:
                return {
                    "success": False,
                    "source": "Gemini",
                    "error": f"Error {response.status_code}: {response.text}",
                }
        except Exception as e:
            return {
                "success": False,
                "source": "Gemini",
                "error": str(e),
            }
    
    def _extract_openai_content(self, result: Dict) -> str:
        """Extract text content from OpenAI Responses API output."""
        try:
            if "output" in result:
                output = result["output"]
                if isinstance(output, list):
                    text_parts = []
                    for item in output:
                        if isinstance(item, dict) and item.get("type") == "message":
                            content = item.get("content", [])
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "output_text":
                                        text_parts.append(block.get("text", ""))
                            elif isinstance(content, str):
                                text_parts.append(content)
                        elif isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                        elif isinstance(item, str):
                            text_parts.append(item)
                    combined = "\n".join(text_parts).strip()
                    if combined:
                        return combined
                elif isinstance(output, str):
                    return output
            if "content" in result:
                return result["content"] if isinstance(result["content"], str) else json.dumps(result["content"], indent=2)
            return json.dumps(result, indent=2)
        except Exception as e:
            print(f"⚠️ Error extracting OpenAI content: {e}")
            return str(result)
    
    def _extract_openai_citations(self, result: Dict) -> List[str]:
        """Extract citations from OpenAI Responses API output."""
        citations = []
        try:
            output = result.get("output", [])
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict) and item.get("type") == "web_search_call":
                        sources = item.get("action", {}).get("sources", [])
                        for s in sources:
                            url = s.get("url") if isinstance(s, dict) else None
                            if url:
                                citations.append(url)
            if not citations and "sources" in result:
                citations = [s.get("url") for s in result["sources"] if isinstance(s, dict) and s.get("url")]
        except Exception as e:
            print(f"⚠️ Error extracting OpenAI citations: {e}")
        return citations
    
    def _extract_gemini_content(self, result: Dict) -> str:
        """Extract text content from Gemini API response."""
        try:
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text_parts = [p.get("text", "") for p in parts if p.get("text")]
                return "\n".join(text_parts).strip()
        except Exception as e:
            print(f"⚠️ Error extracting Gemini content: {e}")
        return ""

    def _extract_gemini_citations(self, result: Dict) -> List[str]:
        """Extract source URLs from Gemini grounding metadata."""
        citations = []
        try:
            candidates = result.get("candidates", [])
            if candidates:
                grounding = candidates[0].get("groundingMetadata", {})
                for chunk in grounding.get("groundingChunks", []):
                    uri = chunk.get("web", {}).get("uri")
                    if uri:
                        citations.append(uri)
        except Exception as e:
            print(f"⚠️ Error extracting Gemini citations: {e}")
        return citations

    def calculate_similarity_and_synthesize(self, research_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Calculate similarity score (out of 3), synthesize research, and extract location."""
        successful_results = [r for r in research_results if r["success"] and r.get("content")]
        
        if len(successful_results) < 2:
            return {
                "success": False,
                "error": "Need at least 2 successful sources with content to synthesize."
            }
        
        n = len(successful_results)
        analysis_text = f"""You are analyzing research from {n} sources.

Original Query: {query}

"""
        for i, result in enumerate(successful_results, 1):
            analysis_text += f"\n{'='*60}\nSOURCE {i} - {result['source']}:\n{'='*60}\n{result['content']}\n\n"

        if n >= 3:
            score_rubric = """Task 1: Calculate Similarity Score (out of 3).
Analyze how similar these research results are:
- 3/3 = All 3 sources strongly agree on the same information
- 2.5/3 = All 3 sources agree, minor differences in details
- 2/3 = Two sources agree, one provides different information
- 1.5/3 = Partial agreement, notable differences
- 1/3 = Sources mostly provide different information
- 0.5/3 = Sources significantly contradict each other
- 0/3 = Complete contradiction"""
        else:
            score_rubric = """Task 1: Calculate Similarity Score (out of 3).
You have 2 sources. Score based on their agreement:
- 3/3 = Both sources strongly agree on the same information
- 2.5/3 = Both sources agree, minor differences in details
- 2/3 = Partial agreement, some notable differences
- 1/3 = Sources mostly provide different information
- 0/3 = Sources significantly contradict each other"""

        analysis_text += score_rubric

        analysis_text += """
Task 2: Synthesize Combined Research.
Create a comprehensive, cohesive answer that combines the best factual information from all sources.
Ensure inclusion of:
- Name
- Location
- Ranking details (e.g., '#1 of 1 Things to Do in Sandy Point')
- Price range or level ($-$$$$)
- Average rating
- Notable highlights or reviews
- Category/type
Do NOT mention or reference any website or platform names.
Maintain a factual, analytical, and neutral tone.

Task 3: Extract Location.
Based on BOTH the original query AND the research content, identify the primary location:
- If the query is about a specific business or place: format as "Specific Place Name, City, Country"
- If the query is about a city or area: format as "City, Country"
- If the query is about a landmark: format as "Landmark Name, City, Country"
- If the location cannot be determined, use "Unknown"

CRITICAL:
You MUST provide a location field.
Never return null or leave it empty.

Provide your response strictly in JSON format:
{
  "similarity_score": 2.5,
  "similarity_explanation": "Brief explanation of why this score",
  "combined_research": "Comprehensive synthesized answer combining all sources with price, ranking, rating, and key highlights but without mentioning any websites",
  "location": "Name, City, Country"
}
"""
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}",
        }
        
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are an expert research analyst who synthesizes information from multiple sources and extracts location data. Always provide a location field, never null."},
                {"role": "user", "content": analysis_text}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            if response.status_code == 200:
                result = response.json()
                synthesis_text = result["choices"][0]["message"]["content"]
                synthesis = json.loads(synthesis_text)
                
                # Debug: Print what OpenAI returned
                print(f"\n🔍 DEBUG - OpenAI Synthesis Response:")
                print(json.dumps(synthesis, indent=2))
                
                return {
                    "success": True,
                    "synthesis": synthesis
                }
            else:
                return {
                    "success": False,
                    "error": f"Error {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_validated_research(self, query: str) -> Dict[str, Any]:
        """
        Main function to get validated research.
        Loads whitelist domains for this query category + country, then
        fires all 3 web searches in parallel via ThreadPoolExecutor.
        """
        print(f"\n{'='*80}")
        print(f"🔎 PARALLEL WEB RESEARCH — 3 sources firing simultaneously")
        print(f"📤 Query: '{query}'")
        print(f"{'='*80}")

        # ── Load whitelist for this query ─────────────────────────────────────
        category = _extract_category(query)
        country  = _extract_country_from_query(query, self.openai_key)

        if country:
            self._current_whitelist = fetch_whitelist_domains(category, country)
            if self._current_whitelist:
                print(f"✅ Whitelist loaded [{category}/{country}]: {self._current_whitelist}")
            else:
                print(f"ℹ️  No whitelist entries for [{category}/{country}]")
        else:
            self._current_whitelist = []
            print(f"ℹ️  Could not resolve country — whitelist skipped")

        t_start = time.time()

        # ── Fire all 3 searches in parallel; start synthesis as soon as 2 arrive ──
        research_results = []
        synthesis_future = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            search_futures = {
                executor.submit(self.search_openai,     query): "OpenAI",
                executor.submit(self.search_perplexity, query): "Perplexity",
                executor.submit(self.search_gemini,     query): "Gemini",
            }

            for future in concurrent.futures.as_completed(search_futures):
                source = search_futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"success": False, "source": source, "error": str(e), "content": "", "citations": []}
                research_results.append(result)
                ok      = result.get("success") and result.get("content")
                content = str(result.get("content", ""))
                status  = "✅" if ok else "❌"
                print(f"  {status} {source} done at {time.time() - t_start:.1f}s — "
                      f"{len(content)} chars | {len(result.get('citations', []))} citations")
                if ok:
                    print(f"     Preview: {content[:300]}")
                else:
                    print(f"     Error: {result.get('error')}")

                # Start synthesis as soon as 2 results are ready — overlaps with slowest search
                if len(research_results) == 2 and synthesis_future is None:
                    print(f"🚀 Starting early synthesis with 2/3 results at {time.time() - t_start:.1f}s...")
                    synthesis_future = executor.submit(
                        self.calculate_similarity_and_synthesize,
                        research_results.copy(),
                        query,
                    )

            # Fallback: start synthesis if somehow not triggered above
            if synthesis_future is None:
                synthesis_future = executor.submit(
                    self.calculate_similarity_and_synthesize,
                    research_results,
                    query,
                )

            synthesis_result = synthesis_future.result()

        print(f"⚡ All 3 searches + synthesis completed in {time.time() - t_start:.1f}s")

        successful = [r for r in research_results if r.get('success') and r.get('content')]
        print(f"\n📊 {len(successful)}/3 sources with content")
        print(f"{'='*80}\n")

        print(f"🧪 Synthesis success: {synthesis_result.get('success')}")
        if not synthesis_result.get('success'):
            print(f"🧪 Synthesis error: {synthesis_result.get('error')}")
        
        # Aggregate all citations
        all_citations = []
        for result in research_results:
            if result.get("success") and result.get("citations"):
                all_citations.extend(result["citations"])
        
        # Remove duplicates while preserving order
        unique_citations = []
        seen = set()
        for citation in all_citations:
            if citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)
        
        # Create final output
        if synthesis_result["success"]:
            synthesis = synthesis_result["synthesis"]
            location = synthesis.get("location", "Unknown")
            if not location or location in [None, "null", "None", "", "N/A", "n/a"]:
                location = "Unknown"
            if isinstance(location, str):
                location = location.strip()
                if not location:
                    location = "Unknown"
            
            score = f"{synthesis.get('similarity_score', 0)}/3"
            print(f"\n✅ Final location value: '{location}'")
            print(f"✅ Final score: {score}")
            print(f"✅ Citations count: {len(unique_citations)}")
            
            return {
                "score": score,
                "research": synthesis.get("combined_research", ""),
                "location": location,
                "citations": unique_citations
            }
        else:
            print(f"\n❌ Synthesis failed — returning score 0/3")
            return {
                "score": "0/3",
                "research": f"Error: {synthesis_result.get('error', 'Unknown error')}",
                "location": "Unknown",
                "citations": unique_citations
            }


# Global instance
_validator = None

def validate_research(query: str) -> Dict[str, Any]:
    """
    Validate research from multiple sources and return synthesized results.
    
    Args:
        query: The research question
    
    Returns:
        Dictionary with keys:
        - score: Similarity score (X/3)
        - research: Combined research from all sources
        - location: Location string (never null, defaults to "Unknown")
        - citations: List of all unique citation URLs
    """
    global _validator
    if _validator is None:
        try:
            _validator = ResearchValidator()
        except ValueError as e:
            return {
                "score": "0/3",
                "research": f"Initialization Error: {e}",
                "location": "Unknown",
                "citations": []
            }
    
    return _validator.get_validated_research(query)