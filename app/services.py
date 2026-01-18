# app/services.py
import json
import time
import asyncio
from typing import AsyncGenerator
from agents import Runner
from openai.types.responses import ResponseTextDeltaEvent
from .agents_ import general_agent , trip_planning_agent , explore_planning_agent # Import the configured agents
from .memory import check_user, add_message, get_message
from .tools import research_further

async def generate_stream(message: str, thread_id: str , reference: str) -> AsyncGenerator[str, None]:
    """Generates a streaming response in Server-Sent Events (SSE) format."""
    start_time = time.time()
    first_chunk_time = None
    full_response_content = ""
    
    try:
        
        add_message(role='user', thread_id=thread_id, message=message)
        final_message = get_message(thread_id=thread_id)
        research_further(final_message)
        
        # Append the latest message to final_message before sending to agent
        final_message_with_current = final_message + "\n\n Question : " + message + "\n\n Reference : " + reference
        
        result = Runner.run_streamed(general_agent, final_message_with_current)
        
        yield f"data: {json.dumps({'start_time': start_time, 'status': 'started' , 'threadId': thread_id})}\n\n"
        
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                chunk = event.data.delta
                if chunk:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        ttfb = first_chunk_time - start_time
                        yield f"data: {json.dumps({'time_to_first_byte': ttfb})}\n\n"
                    
                    # Accumulate the chunk for the full response
                    full_response_content += chunk
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
        
        end_time = time.time()
        yield f"data: {json.dumps({'done': True, 'total_time': end_time - start_time})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    finally:
        # Add the assistant's response to memory after the streaming is complete
        if thread_id and full_response_content:
            async def save_message():
                add_message(role='assistant', thread_id=thread_id, message=full_response_content)
            asyncio.create_task(save_message())

async def get_complete_response(message: str, thread_id: str , mode: str) -> tuple[str, dict]:
    """Generates a complete, non-streamed response and provides timing info."""
    start_time = time.time()
    
    try:        
        # Append the latest message to final_message before sending to agent    
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