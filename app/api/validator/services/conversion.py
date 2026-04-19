"""
Data Converter Service
Converts research data to attraction format using OpenAI
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai import OpenAI
import json
import os
import re

# ── Compiled patterns for source prioritisation ──────────────────────────────
_RE_TRIPADVISOR = re.compile(r'tripadvisor\.', re.IGNORECASE)
_RE_YELP        = re.compile(r'\byelp\.com\b',  re.IGNORECASE)


def pick_best_citation(citations: List[str]) -> str:
    """
    Scan all citation URLs and return the highest-priority source:
      1. TripAdvisor  (tripadvisor.com / tripadvisor.co.uk / etc.)
      2. Yelp         (yelp.com)
      3. First other  third-party URL
      4. ""           if citations is empty
    """
    ta_url   = next((u for u in citations if isinstance(u, str) and _RE_TRIPADVISOR.search(u)), None)
    if ta_url:
        return ta_url

    yelp_url = next((u for u in citations if isinstance(u, str) and _RE_YELP.search(u)), None)
    if yelp_url:
        return yelp_url

    return next((u for u in citations if isinstance(u, str) and u.startswith("http")), "")


# Output Schema Models
class MetaObject(BaseModel):
    audience: List[str] = []
    location: Optional[str] = None
    ranking: Optional[str] = None
    price_level: Optional[str] = None


class AttractionOutput(BaseModel):
    country: str = ""
    city: str = ""
    meta_obj: MetaObject = MetaObject()
    latitude: Optional[str] = None
    language: str = "en"
    category: str = ""
    source: str = ""
    title: str = ""
    content: str = ""
    region_code: Optional[str] = None
    tags: str = ""
    longitude: Optional[str] = None
    query: str = ""


def convert_research_to_attraction(
    research_data: Dict[str, Any], 
    openai_api_key: str
) -> List[AttractionOutput]:
    """
    Convert research data to attraction format using OpenAI API.
    
    Args:
        research_data: Dictionary containing research results (any format)
        openai_api_key: OpenAI API key for authentication
        
    Returns:
        List of AttractionOutput objects
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    # 🔥 NEW: Check for RAG context to exclude
    rag_context = research_data.get("rag_context")
    rag_exclusion_text = ""
    
    if rag_context:
        
        # Extract what's already known from RAG
        existing_entities = rag_context.get("entities", [])
        existing_chunks = rag_context.get("chunks", [])
        existing_audience = rag_context.get("audience", [])
        existing_travel_style = rag_context.get("travel_style", [])
        
        # Build exclusion instruction
        rag_exclusion_text = f"""

⚠️ CRITICAL EXCLUSION INSTRUCTION:
The following information is ALREADY KNOWN from our existing database and must be EXCLUDED from your conversion:

Existing Entities: {json.dumps(existing_entities, indent=2)}
Existing Content Chunks: {json.dumps(existing_chunks, indent=2)}
Existing Audience Types: {json.dumps(existing_audience, indent=2)}
Existing Travel Styles: {json.dumps(existing_travel_style, indent=2)}

DO NOT include, repeat, or reference any of the above information in your output.
Focus ONLY on NEW information from the research that is not already covered above.
Find complementary insights, additional details, or alternative perspectives not present in the existing data.
"""
    
    # Prepare the prompt for OpenAI
    prompt = f"""
You are a data transformation expert. Convert the following research data into a structured attraction/place format.

{rag_exclusion_text}

INPUT DATA:
{json.dumps(research_data, indent=2)}

INSTRUCTIONS:
1. Extract the country code (2-letter ISO code) from the maps_data address_components
2. Extract the city from the location or maps_data
3. Create a meta_obj with:
   - audience: List of audience types based on the research content (e.g., ["ROMANTIC", "FAMILY", "FRIENDS GETAWAY", "SOLO", "BUSINESS", "ADVENTURE"])
   - location: Full formatted address from maps_data (if available)
   - ranking: If mentioned in research, otherwise null
   - price_level: Extract from research if mentioned (e.g., "$", "$$", "$$$", "$$$$"), otherwise null
4. Extract latitude and longitude from maps_data geometry location as strings
5. Set language to "en"
6. Generate a unique id using the place_id from maps_data or create a descriptive one
7. Determine category using STRICT rules — pick the MOST SPECIFIC match:
   - If the place serves food/drinks (restaurant, café, bar, bakery, food stall, diner, bistro, pub, brewery, pizzeria): category = "Restaurant"
   - If it is a nature spot (park, beach, garden, trail, mountain, lake, waterfall, forest, nature reserve): category = "Nature & Parks"
   - If it is a museum, gallery, historic site, monument, or cultural center: category = "Museum & Culture"
   - If it is a shopping destination (mall, market, bazaar, souk): category = "Shopping"
   - If it is nightlife (nightclub, lounge, comedy club): category = "Nightlife"
   - If it is an adventure/sport activity (diving, skydiving, rafting, zip-line, surfing): category = "Adventure"
   - For anything else (landmarks, viewpoints, entertainment venues, unique experiences): category = "Attraction"
   NEVER use generic labels like "Place", "Hip Place", "Food & Dining", or "Landmark". Always use one of the specific categories above.
8. Use the first citation as source
9. TITLE RULE - CRITICAL: Use ONLY the exact proper name of the specific place as the title.
   - CORRECT examples: "Gary Danko", "Mall of America", "Eiffel Tower", "Nobu Miami"
   - WRONG examples: "Top Dining Experiences in Orlando", "Diverse Dining in San Francisco", "Mall of America: A Premier Shopping Destination"
   - NEVER use descriptive phrases, category summaries, or add subtitles after a colon
10. MULTIPLE PLACES RULE - CRITICAL: If the research mentions MULTIPLE specific places (e.g., several restaurants, hotels, attractions), you MUST create ONE separate result entry per individual place. Do NOT merge them into a single generic entry. Each entry gets the exact name of that specific place as its title.
11. Extract region_code from address_components (state/province/administrative_area_level_1)
12. Create tags as a comma-separated string with relevant keywords from the research
13. Include the original query

IMPORTANT:
- If maps_data is null or missing, extract location info from the location string
- If specific details are missing, make reasonable inferences from the research text
- Ensure all required fields are filled
- The title MUST be just the specific place name — never a descriptive summary or category label
- Content should be concise but informative, focused on the specific place
- When research covers multiple places, produce multiple result objects (one per place)
{'- EXCLUDE all information mentioned in the EXCLUSION INSTRUCTION above' if rag_context else ''}

OUTPUT FORMAT:
Return ONLY a valid JSON object with a "results" array like this:
{{
  "results": [
    {{
      "country": "string (2-letter code like PK, US, etc.)",
      "city": "string",
      "meta_obj": {{
        "audience": ["string"],
        "location": "string",
        "ranking": "string or null",
        "price_level": "string or null"
      }},
      "latitude": "string",
      "language": "en",
      "id": "string",
      "category": "string (MUST be one of: Restaurant, Nature & Parks, Museum & Culture, Shopping, Nightlife, Adventure, Attraction)",
      "source": "string (URL)",
      "title": "string",
      "content": "string",
      "region_code": "string",
      "tags": "string (comma-separated)",
      "longitude": "string",
      "query": "string"
    }}
  ]
}}

Convert each result in the input data into a separate object in the results array.
"""

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a data transformation expert. Always return valid JSON only, with no additional text or markdown formatting. If exclusion instructions are provided, strictly avoid repeating any of the excluded content."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    # Extract and parse the response
    response_content = response.choices[0].message.content
    
    # Handle if response is wrapped in markdown code blocks
    if response_content.strip().startswith("```"):
        response_content = response_content.strip()
        response_content = response_content.split("```")[1]
        if response_content.startswith("json"):
            response_content = response_content[4:]
        response_content = response_content.strip()
    
    # Parse JSON response
    parsed_data = json.loads(response_content)
    
    # Extract the results array
    if isinstance(parsed_data, dict) and "results" in parsed_data:
        results_array = parsed_data["results"]
    elif isinstance(parsed_data, list):
        results_array = parsed_data
    elif isinstance(parsed_data, dict):
        # Try other common key names
        for key in ['attractions', 'data', 'items']:
            if key in parsed_data and isinstance(parsed_data[key], list):
                results_array = parsed_data[key]
                break
        else:
            # If still no array found, wrap the dict in a list
            results_array = [parsed_data]
    else:
        results_array = [parsed_data]
    
    # Validate and convert to AttractionOutput objects
    attractions = [AttractionOutput(**item) for item in results_array]
    
    return attractions

# Example usage and testing
if __name__ == "__main__":
    # Sample input data
    sample_input = {
        "type": "specific",
        "original_query": "Is biryani ka best food in karachi?",
        "queries": ["Where can I find the best biryani in Karachi?"],
        "results": [
            {
                "query": "Where can I find the best biryani in Karachi?",
                "score": "2.5/3",
                "research": "Karachi is renowned for its diverse and flavorful biryani offerings. Top recommendations include Biryani Centre, located at Plot No. 12‑C, 26th Commercial Street, Tauheed Commercial Area, known for its generous portions and spicy, juicy mutton biryani.",
                "citations": [
                    "https://faizantechcore.com/biryani-in-karachi/"
                ],
                "location": "Karachi, Pakistan",
                "maps_data": {
                    "address_components": [
                        {"long_name": "Karachi", "short_name": "Karachi", "types": ["locality", "political"]},
                        {"long_name": "Sindh", "short_name": "Sindh", "types": ["administrative_area_level_1", "political"]},
                        {"long_name": "Pakistan", "short_name": "PK", "types": ["country", "political"]}
                    ],
                    "formatted_address": "Karachi, Pakistan",
                    "geometry": {
                        "location": {"lat": 24.8607343, "lng": 67.0011364}
                    },
                    "place_id": "ChIJv0sdZQY-sz4RIwxaVUQv-Zw"
                }
            }
        ]
    }
    
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Convert the data
    try:
        attractions = convert_research_to_attraction(sample_input, api_key)
        
    except Exception:
        pass