# app/tools.py
from agents import (
    function_tool
)
import requests
import json
import threading
from typing import Dict, Any, List

@function_tool
def customer_rag_n8n(query: str) -> Dict[str, Any]:
    """
    Send a query to the n8n webhook for customer RAG processing.
    
    Args:
        query (str): The query string to send to the RAG system
        
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
    url = "https://cs-automation.hiptraveler.com/webhook/customer_service"
    
    # Prepare the payload
    payload = {
        "message": query.strip()
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

    
    

@function_tool
def rag(query: str , reference: str) -> Dict[str, Any]:
    """
    Send a query to the  webhook for  RAG processing.
    
    Args:
        query (str): The query string to send to the RAG system
        
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
        "reference":reference,
        "top_k" : 10

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
    
    

def place_search(queries: List[Dict[str, str]], reference: str = "hiptraveler") -> Dict[str, Any]:
    """
    Send a batch of place queries to the HipTraveler Places API webhook for processing.

    Args:
        queries (List[Dict[str, str]]): A list of query objects. 
                                        Each object must have "query", "location", and "type".
                                        Example: 
                                        [
                                            {"query": "Cable beach", "location": "Nassau, Bahamas", "type": "place"},
                                            {"query": "Wild Thyme", "location": "Nassau, Bahamas", "type": "restaurant"}
                                        ]
        reference (str): The reference source ID (default: "hiptraveler").

    Returns:
        Dict[str, Any]: Response from the webhook containing search results.

    Raises:
        requests.exceptions.RequestException: If the request fails.
        ValueError: If the queries list is empty.
    """

    # Validate input
    if not queries or not isinstance(queries, list):
        raise ValueError("Queries must be a non-empty list of dictionary objects.")

    # Webhook URL (Batch Endpoint)
    url = "https://rag.hiptraveler.com/places/batch/search"

    # Prepare the payload strictly matching the required schema
    payload = {
        "reference": reference,
        "queries": queries
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
            timeout=45  # Increased timeout for batch processing
        )

        # Raise an exception for bad status codes
        response.raise_for_status()
        
        # Try to parse JSON response
        try:
            data = response.json()
            return data
        except json.JSONDecodeError:
            # If response is not JSON, return a wrapped text response
            return {
                "status": "error",
                "message": "Invalid JSON response from server",
                "raw_content": response.text,
                "status_code": response.status_code
            }

    except requests.exceptions.Timeout:
        raise requests.exceptions.RequestException("Request timed out after 45 seconds")
    except requests.exceptions.ConnectionError:
        raise requests.exceptions.RequestException("Failed to connect to the webhook")
    except requests.exceptions.HTTPError as e:
        raise requests.exceptions.RequestException(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Request failed: {e}")
    
def research_further(query: str, reference: str = "hiptraveler"):
    """
    Kick off the validation pipeline in a background thread without any HTTP round-trip.
    Calls the validation logic directly in-process so logs are visible and environment
    (dev / prod / local) makes no difference.
    """
    def _run(query_inner: str):
        if not query_inner or not query_inner.strip():
            return

        import asyncio
        from app.api.validator.helpers import is_duplicate_query, process_in_background
        from app.api.validator.services.supabase_service import SupabaseService
        from app.api.validator.services.openai_service import OpenAIService

        async def _pipeline():
            question_only = query_inner.split("\n\nAnswer:\n")[0].strip()

            supabase_service = SupabaseService()
            openai_service   = OpenAIService()

            # Run dedup check and classify_query in parallel — saves ~2-3s
            is_dup, classification = await asyncio.gather(
                is_duplicate_query(question_only, supabase_service),
                openai_service.classify_query(query_inner.strip()),
                return_exceptions=True,
            )

            if isinstance(is_dup, Exception):
                is_dup = False
            if isinstance(classification, Exception):
                return

            if is_dup:
                return

            try:
                await supabase_service.insert_saved_query(question_only)
            except Exception:
                pass

            # Pass the pre-computed classification so process_in_background skips the classify call
            await process_in_background(query=query_inner.strip(), reference=reference, classification=classification)

        asyncio.run(_pipeline())

    thread = threading.Thread(target=_run, args=(query,), daemon=True)
    thread.start()
    