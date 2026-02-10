import os
import requests
import json
from typing import Dict, List, Any
import time
from dotenv import load_dotenv
from app.api.validator.services.supabase_service import SupabaseService
from supabase import create_client
import asyncio



class ResearchValidator:
    def __init__(self):
        load_dotenv()
        """Initialize the validator with API keys from environment variables."""
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not all([self.openai_key, self.perplexity_key, self.tavily_key]):
            raise ValueError(
                "Missing API keys. Please set OPENAI_API_KEY, PERPLEXITY_API_KEY, and TAVILY_API_KEY environment variables."
            )

        # Initialize Supabase client
        self.supabase_service = create_client(self.supabase_url, self.service_role_key)
        self.blacklist_domains: List[str] = []  # initialize empty list

        # Immediately fetch and print domains
        try:
            asyncio.run(self._initialize_blacklist_domains())
        except Exception as e:
            print(f"⚠️ Error initializing blacklist domains: {e}")

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
        
        # Construct blacklist prompt
        if self.blacklist_domains:
            blocked = ", ".join(self.blacklist_domains)
            blacklist_note = f"\nDO NOT USE any of the following sources or domains under any circumstance: {blocked}."
        else:
            blacklist_note = ""
        
        system_prompt = (
        "You are an expert internet research assistant. Your primary goal is to gather accurate, up-to-date, and verifiable information from the web about locations, attractions, and businesses. "
        "Prioritize sources in this order: 1) Tripadvisor, 2) Yelp, 3) official or reputable travel/review sites (Google Travel, Lonely Planet, etc.). "
        "For each entity, collect: name, location, ranking info (e.g., '#1 of 1 Things to Do in Sandy Point'), price level ($-$$$$), average rating, top reviews or highlights, category/type, and source URLs for citation. "
        "Present findings in structured format (bullets or table), cite Tripadvisor or Yelp first, list top 3 results if relevant, and note when primary sources are unavailable. "
        "Be factual, concise, and analytical; extract ranking and price level verbatim from the source; use the most recent data available."
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
            "Content-Type": "application/json"
        }
        
        blacklist_note = ""
        if self.blacklist_domains:
            blocked = ", ".join(self.blacklist_domains)
            blacklist_note = f" Avoid these domains: {blocked}."
        
        data = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "You are an expert internet research assistant. Your primary goal is to gather accurate, up-to-date, and verifiable information from the web about locations, attractions, and businesses. Prioritize sources in this order: 1) Tripadvisor, 2) Yelp, 3) official or reputable travel/review sites (Google Travel, Lonely Planet, etc.). For each entity, collect: name, location, ranking info (e.g., '#1 of 1 Things to Do in Sandy Point'), price level ($-$$$$), average rating, top reviews or highlights, category/type, and source URLs for citation. Present findings in structured format (bullets or table), cite Tripadvisor or Yelp first, list top 3 results if relevant, and note when primary sources are unavailable. Be factual, concise, and analytical; extract ranking and price level verbatim from the source; use the most recent data available." + blacklist_note},
                {"role": "user", "content": query}
            ],
            "return_citations": True,
            "return_related_questions": False
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
    
    def search_tavily(self, query: str) -> Dict[str, Any]:
        """Perform web search using Tavily API."""
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.tavily_key}"
        }
        
        data = {
            "query": query,
            "include_answer": "advanced",
            "search_depth": "advanced",
            "include_raw_content": True,
            "include_domains": [],
            "exclude_domains": self.blacklist_domains
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "")
                results = result.get("results", [])
                citations = [r.get("url") for r in results if r.get("url")]
                
                return {
                    "success": True,
                    "source": "Tavily",
                    "data": result,
                    "content": answer,
                    "citations": citations,
                    "raw_results": results
                }
            else:
                return {
                    "success": False,
                    "source": "Tavily",
                    "error": f"Error {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "source": "Tavily",
                "error": str(e)
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
    
    def calculate_similarity_and_synthesize(self, research_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Calculate similarity score (out of 3), synthesize research, and extract location."""
        successful_results = [r for r in research_results if r["success"] and r.get("content")]
        
        if len(successful_results) < 2:
            return {
                "success": False,
                "error": "Need at least 2 successful sources with content to synthesize."
            }
        
        analysis_text = f"""You are analyzing research from multiple sources. 

Original Query: {query}

"""
        for i, result in enumerate(successful_results, 1):
            analysis_text += f"\n{'='*60}\nSOURCE {i} - {result['source']}:\n{'='*60}\n{result['content']}\n\n"
        
        analysis_text += """
Task 1: Calculate Similarity Score (out of 3).
Analyze how similar these research results are:
- 3/3 = All sources strongly agree on the same information
- 2.5/3 = All sources agree, minor differences in details
- 2/3 = Two sources agree, one provides different information
- 1.5/3 = Partial agreement, notable differences
- 1/3 = Sources mostly provide different information
- 0.5/3 = Sources significantly contradict each other
- 0/3 = Complete contradiction

Task 2: Synthesize Combined Research.
Create a comprehensive, cohesive answer that combines the best factual information from all sources.
Ensure inclusion of:
- Name
- Location
- Ranking details (e.g., '#1 of 1 Things to Do in Sandy Point')
- Price range or level ($–$$$$)
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
        """
        # Collect research from all sources
        research_results = []
        
        # OpenAI
        print(f"\n{'='*80}")
        print(f"🔎 RESEARCH SOURCE 1: OpenAI Web Search")
        print(f"{'='*80}")
        print(f"📤 Query: '{query}'")
        openai_result = self.search_openai(query)
        research_results.append(openai_result)
        print(f"✅ Success: {openai_result.get('success')}")
        if openai_result.get('success'):
            content = openai_result.get('content', '')
            content_type = type(content).__name__
            content_str = str(content)
            print(f"📝 Content type: {content_type}")
            print(f"📝 Content length: {len(content_str)} chars")
            print(f"📝 Content preview:\n{content_str[:1000]}")
            print(f"🔗 Citations ({len(openai_result.get('citations', []))}): {openai_result.get('citations', [])}")
        else:
            print(f"❌ Error: {openai_result.get('error')}")
        print(f"{'='*80}\n")
        time.sleep(1)
        
        # Perplexity
        print(f"\n{'='*80}")
        print(f"🔎 RESEARCH SOURCE 2: Perplexity")
        print(f"{'='*80}")
        print(f"📤 Query: '{query}'")
        perplexity_result = self.search_perplexity(query)
        research_results.append(perplexity_result)
        print(f"✅ Success: {perplexity_result.get('success')}")
        if perplexity_result.get('success'):
            content = perplexity_result.get('content', '')
            print(f"📝 Content type: {type(content).__name__}")
            print(f"📝 Content length: {len(str(content))} chars")
            print(f"📝 Content preview:\n{str(content)[:1000]}")
            print(f"🔗 Citations ({len(perplexity_result.get('citations', []))}): {perplexity_result.get('citations', [])}")
        else:
            print(f"❌ Error: {perplexity_result.get('error')}")
        print(f"{'='*80}\n")
        time.sleep(1)
        
        # Tavily
        print(f"\n{'='*80}")
        print(f"🔎 RESEARCH SOURCE 3: Tavily")
        print(f"{'='*80}")
        print(f"📤 Query: '{query}'")
        tavily_result = self.search_tavily(query)
        research_results.append(tavily_result)
        print(f"✅ Success: {tavily_result.get('success')}")
        if tavily_result.get('success'):
            content = tavily_result.get('content', '')
            print(f"📝 Content type: {type(content).__name__}")
            print(f"📝 Content length: {len(str(content))} chars")
            print(f"📝 Content preview:\n{str(content)[:1000]}")
            print(f"🔗 Citations ({len(tavily_result.get('citations', []))}): {tavily_result.get('citations', [])}")
        else:
            print(f"❌ Error: {tavily_result.get('error')}")
        print(f"{'='*80}\n")
        
        # Summary before synthesis
        successful = [r for r in research_results if r.get('success') and r.get('content')]
        failed = [r for r in research_results if not r.get('success') or not r.get('content')]
        print(f"\n📊 RESEARCH SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Successful sources with content: {len(successful)}/3")
        print(f"❌ Failed or empty sources: {len(failed)}/3")
        for r in research_results:
            has_content = bool(r.get('content'))
            content_len = len(str(r.get('content', ''))) if has_content else 0
            status = "✅" if r.get('success') and has_content else "❌"
            print(f"   {status} {r.get('source')}: success={r.get('success')}, has_content={has_content}, content_length={content_len}")
        print(f"{'='*80}\n")
        
        # Calculate similarity and synthesize
        print(f"🧪 Starting synthesis with {len(successful)} successful sources...")
        synthesis_result = self.calculate_similarity_and_synthesize(research_results, query)
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