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
        "reference":reference

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
        print(response.json())
        
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

    print(f"--- Sending Request ---\nURL: {url}\nPayload: {json.dumps(payload, indent=2)}")

    try:
        # Send POST request
        response = requests.post(
            url=url,
            json=payload,
            headers=headers,
            timeout=45  # Increased timeout for batch processing
        )

        # Print Raw Response info for debugging
        print(f"--- Response Received ---\nStatus Code: {response.status_code}")
        print(f"Raw Text: {response.text[:500]}...") # Print first 500 chars to avoid clutter

        # Raise an exception for bad status codes
        response.raise_for_status()
        
        # Try to parse JSON response
        try:
            data = response.json()
            # print(f"Parsed JSON: {json.dumps(data, indent=2)}") # Optional: Print full parsed JSON
            return data
        except json.JSONDecodeError as json_err:
            print(f"!!! JSON Decode Error: {json_err}")
            # If response is not JSON, return a wrapped text response
            return {
                "status": "error",
                "message": "Invalid JSON response from server",
                "raw_content": response.text,
                "status_code": response.status_code
            }

    except requests.exceptions.Timeout:
        print("!!! Error: Request timed out after 45 seconds")
        raise requests.exceptions.RequestException("Request timed out after 45 seconds")
    except requests.exceptions.ConnectionError:
        print("!!! Error: Failed to connect to the webhook")
        raise requests.exceptions.RequestException("Failed to connect to the webhook")
    except requests.exceptions.HTTPError as e:
        print(f"!!! HTTP Error: {e}")
        raise requests.exceptions.RequestException(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"!!! General Request Error: {e}")
        raise requests.exceptions.RequestException(f"Request failed: {e}")
    
def research_further(query: str):
    """
    Send a query to the webhook for research purposes.
    This version runs in the background and does not block the main workflow.
    
    Args:
        query (str): The query string to send to the RAG system
    """
    
    def _send_request(query_inner):
        # Validate input
        if not query_inner or not query_inner.strip():
            print("Query cannot be empty or None")
            return
        
        url = "https://ai.hiptraveler.com/validator/process"
        payload = {"query": query_inner.strip()}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=1000)
            response.raise_for_status()
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {"success": True, "data": response.text, "status_code": response.status_code}
            
            # Optionally log the response or store it somewhere
            print("research_further completed:", data)
        
        except requests.exceptions.RequestException as e:
            print(f"research_further request failed: {e}")
    
    # Run the request in a separate thread
    thread = threading.Thread(target=_send_request, args=(query,))
    thread.daemon = True  # so it won't block program exit
    thread.start()
    