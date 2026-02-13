# app/services.py
import json
import os
import time
import asyncio
from typing import AsyncGenerator, Optional
from agents import Runner
from openai.types.responses import ResponseTextDeltaEvent
from .agents_ import general_agent , trip_planning_agent , explore_planning_agent , explore_agent # Import the configured agents
from .memory import check_user, add_message, get_message
from .tools import research_further

#from .redis_history import append_interaction #main

from .redis_history_local import append_interaction # testing

OLD_INTERACTIONS_MAX = int(os.getenv("REDIS_OLD_INTERACTIONS_MAX", "30"))
REDIS_TTL = int(os.getenv("REDIS_TTL_SECONDS", "3600"))

async def generate_stream(
    message: str,
    thread_id: str,
    reference: str,
    agent: str,
    final_message: str,
    user_id: str,
    conversation_id: Optional[str],
) -> AsyncGenerator[str, None]:
    """Generates a streaming response in Server-Sent Events (SSE) format."""
    start_time = time.time()
    first_chunk_time = None
    full_response_content = ""

    try:
        add_message(role='user', thread_id=thread_id, message=message)

        final_message_with_current = final_message + "\n\n Reference : " + reference

        research_further(final_message_with_current)

        if agent == 'general_agent':
            result = Runner.run_streamed(general_agent, final_message_with_current)
        else:
            result = Runner.run_streamed(explore_agent, final_message_with_current)

        yield f"data: {json.dumps({'start_time': start_time, 'status': 'started', 'threadId': thread_id, 'conversation_id': conversation_id})}\n\n"

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                chunk = event.data.delta
                if chunk:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        ttfb = first_chunk_time - start_time
                        yield f"data: {json.dumps({'time_to_first_byte': ttfb})}\n\n"

                    full_response_content += chunk
                    yield f"data: {json.dumps({'content': chunk})}\n\n"

        end_time = time.time()
        yield f"data: {json.dumps({'done': True, 'total_time': end_time - start_time})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    finally:
        # Persist to your existing memory system
        if thread_id and full_response_content:
            try:
                add_message(role='assistant', thread_id=thread_id, message=full_response_content)
            except Exception as e:
                print("add_message failed:", str(e))

        # ✅ Persist Q/A to Redis for this conversation (if enabled/available)
        if conversation_id and full_response_content:
            try:
                await append_interaction(
                    user_id=str(user_id),
                    conversation_id=conversation_id,
                    question=message,
                    answer=full_response_content,
                    max_items=OLD_INTERACTIONS_MAX,
                    ttl_seconds=REDIS_TTL,
                )
            except Exception as e:
                print("Redis append_interaction failed:", str(e))

async def get_complete_response(message: str, thread_id: str , mode: str) -> tuple[str, dict]:
    """Generates a complete, non-streamed response and provides timing info."""
    start_time = time.time()
    
    try:        
        result = await Runner.run(trip_planning_agent, message) 
        
        # Access the actual response data
        full_response = result.final_output  
        add_message(role='assistant', thread_id=thread_id, message=full_response) 
        
        end_time = time.time()
        total_time = end_time - start_time
        
        timing_info = {
            "param" : mode,
            "threadId":thread_id,
            "total_time": f"{total_time:.2f} seconds",
            "response_type": "non_streaming"
        }    
            
        return full_response, timing_info

    except Exception as e:
        raise Exception(f"Agent error: {str(e)}") from e
    
    
    
async def get_complete_response_explore(message: str, thread_id: str , mode: str) -> tuple[str, dict]:
    """Generates a complete, non-streamed response and provides timing info."""
    research_further(message)
    start_time = time.time()
    
    try:        
        # Append the latest message to final_message before sending to agent    
        result = await Runner.run(explore_planning_agent, message) 
        
        # Access the actual response data
        full_response = result.final_output  
        add_message(role='assistant', thread_id=thread_id, message=full_response) 
        
        end_time = time.time()
        total_time = end_time - start_time
        
        timing_info = {
            "param" : mode,
            "threadId":thread_id,
            "total_time": f"{total_time:.2f} seconds",
            "response_type": "non_streaming"
        }    
            
        return full_response, timing_info

    except Exception as e:
        raise Exception(f"Agent error: {str(e)}") from e