from queue import Full
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from agents import Runner
import json
import time
import requests
from typing import AsyncGenerator, Dict, Any
from .schemas import QueryRequest, UserCreateRequest
from .services import generate_stream, get_complete_response, get_complete_response_explore
from .agents_ import validation_agent, explore_travel_agent
from .memory import delete_user, create_new_user
from .memory import check_user, get_message, add_message

router = APIRouter()


# ─── RAG Function ───────────────────────────────────────────────

def rag(query: str, reference: str) -> Dict[str, Any]:
    """Send a query to the webhook for RAG processing."""
    
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or None")
    
    url = "https://rag.hiptraveler.com/chat"
    
    payload = {
        "query": query.strip(),
        "reference": reference
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        response = requests.post(
            url=url,
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        print("RAG Response:", response.json())
        
        try:
            return response.json()
        except json.JSONDecodeError:
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


def get_rag_note_stream_response(note: str, thread_id: str, param: str):
    """Stream the RAG note response when no chunks are found."""
    async def note_stream_generator() -> AsyncGenerator[str, None]:
        start_time = time.time()
        yield f"data: {json.dumps({'start_time': start_time, 'status': 'started'})}\n\n"

        first_chunk_time = time.time()
        ttfb = first_chunk_time - start_time
        yield f"data: {json.dumps({'time_to_first_byte': ttfb})}\n\n"

        # Stream the note as JSON wrapper word by word (same format as your error stream)
        yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n\n"

        words = note.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else f" {word}"
            yield f"data: {json.dumps({'content': chunk})}\n\n"

        yield f"data: {json.dumps({'content': '\"}'})}\n\n"

        end_time = time.time()
        yield f"data: {json.dumps({'done': True, 'total_time': end_time - start_time, 'threadId': thread_id, 'param': param, 'response_type': 'rag_streaming'})}\n\n"

    return StreamingResponse(
        note_stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# ─── Main Chat Route ────────────────────────────────────────────

@router.post("/chat")
async def unified_chat(request: QueryRequest):
    try:
        # Step 1: User and Thread setup
        thread_id = check_user(request.user_id)
        param = request.param

        # Step 2: Build conversation context
        final_message_with_current = build_conversation_context(request)
        print(final_message_with_current)

        # Step 3: Ask RAG first (applies to ALL modes)
        try:
            rag_response = rag(query=request.message, reference=request.reference)
            chunks = rag_response.get("chunks", [])
            audience = rag_response.get("audience", [])
            travel_style = rag_response.get("travel_style", [])
            note = rag_response.get("note", "")
        except Exception as e:
            print(f"RAG failed, falling back to agent: {e}")
            chunks = ["fallback"]
            audience = []
            travel_style = []
            note = ""

        # Step 4: RAG has no useful data — stream the note back
        rag_has_data = bool(chunks or audience or travel_style)
        rag_country_detected = "Answer retrieved for detected country" in note

        if not rag_has_data and not rag_country_detected and note:
            add_message(role='assistant', thread_id=thread_id, message=note)
            return get_rag_note_stream_response(note, thread_id, param)

        # Step 5: RAG is good — proceed with validation
        print("validation")
        validation_result = await Runner.run(validation_agent, final_message_with_current)
        print("validation_result:", validation_result.final_output)

        # Step 6: Check Validity
        if not validation_result.final_output.isValid:
            return get_error_stream_response(
                validation_result.final_output.reason,
                validation_result.final_output.solution
            )

        # Step 7: Route based on travel relevance
        if validation_result.final_output.isTravelRelated:
            response_content = await get_complete_response(
                final_message_with_current, thread_id, param
            )
            return JSONResponse(content={
                "response": jsonable_encoder(response_content),
                "type": "non-streaming"
            })
        else:
            agent = 'general_agent' if param == 'plan' else 'explore_agent'
            print("Stream")
            return StreamingResponse(
                generate_stream(
                    request.message, thread_id, request.reference,
                    agent, final_message_with_current
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Helper Functions (unchanged) ───────────────────────────────

def clean_answer(answer: str) -> str:
    """Remove POI metadata block (between $$$$$) from answer."""
    return answer.split('$$$$$')[0].strip()


def build_conversation_context(request: QueryRequest) -> str:
    """Build the final message with conversation history (max 3)."""
    if not request.old_interactions:
        return request.message

    recent = request.old_interactions[:3]

    if len(recent) >= 3:
        last = recent[0]
        previous = recent[1]
        old = recent[2]

        old_conversation = f"User: {old.question}\nAssistant: {clean_answer(old.answer)}"
        previous_conversation = f"User: {previous.question}\nAssistant: {clean_answer(previous.answer)}"
        last_conversation = f"User: {last.question}\nAssistant: {clean_answer(last.answer)}"
        return (
            f"Previous conversations:\n{old_conversation}\n\n"
            f"{previous_conversation}\n\n"
            f"Last conversation:\n{last_conversation}\n\n (this is the continuation of the conversation)\n\n"
            f"User asked: {request.message}"
        )
    elif len(recent) == 2:
        last = recent[0]
        previous = recent[1]
        previous_conversation = f"User: {previous.question}\nAssistant: {clean_answer(previous.answer)}"
        last_conversation = f"User: {last.question}\nAssistant: {clean_answer(last.answer)}"
        return (
            f"Previous conversation:\n{previous_conversation}\n\n"
            f"Last conversation:\n{last_conversation}\n\n (this is the continuation of the conversation)\n\n"
            f"User asked: {request.message}"
        )
    elif len(recent) == 1:
        last = recent[0]
        last_conversation = f"User: {last.question}\nAssistant: {clean_answer(last.answer)}"
        return (
            f"Last conversation (this is the continuation of the conversation):\n{last_conversation}\n\n"
            f"New question asked: {request.message}"
        )

    return request.message


def get_error_stream_response(reason: str, solution: str):
    """Generate a streaming error response for invalid/non-travel queries."""
    async def error_stream_generator() -> AsyncGenerator[str, None]:
        start_time = time.time()
        yield f"data: {json.dumps({'start_time': start_time, 'status': 'started'})}\n\n"

        first_chunk_time = time.time()
        ttfb = first_chunk_time - start_time
        yield f"data: {json.dumps({'time_to_first_byte': ttfb})}\n\n"

        if len(solution) < 50:
            chunks = [
                '{"', "answer", '":"',
                "Let", "'s", " keep", " it", " travel", "-focused", " ✨", ".\n\n",
                "I", " can", " help", " you", " explore", " destinations", ",",
                " discover", " experiences", ",", " and", " plan", " your", " trip", ".\n\n",
                "What", " would", " you", " like", " to", " explore", " next", "?",
                '"}', ""
            ]
            for chunk in chunks:
                if chunk:
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
        else:
            yield f"data: {json.dumps({'content': '{\"answer\":\"'})}\n"
            words = solution.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == 0 else f" {word}"
                yield f"data: {json.dumps({'content': chunk})}\n"
            yield f"data: {json.dumps({'content': '\"}'})}\n"

        end_time = time.time()
        yield f"data: {json.dumps({'done': True, 'total_time': end_time - start_time, 'blocked': True})}\n\n"

    return StreamingResponse(
        error_stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.get("/delete_user")
async def delete_user_route(user_id: int = Query(..., description="The ID of the user to delete")):
    try:
        result = delete_user(user_id)
        return {"message": f"User {user_id} deleted successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/create_user")
async def create_user_route(user: UserCreateRequest):
    try:
        result = create_new_user(
            user_id=user.user_id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name
        )
        return {"message": "User created successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agent-streaming-api"}