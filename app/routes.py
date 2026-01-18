from queue import Full
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from agents import Runner
import json
import time
from typing import AsyncGenerator
from .schemas import QueryRequest, UserCreateRequest
from .services import generate_stream, get_complete_response, get_complete_response_explore
from .agents_ import validation_agent, explore_travel_agent
from .memory import delete_user, create_new_user
from .memory import check_user, get_message, add_message

router = APIRouter()

@router.post("/chat")
async def unified_chat(request: QueryRequest):
    """
    This endpoint intelligently routes requests based on content validation.
    - If param is 'explore', it uses specific logic for explore agent.
    - Otherwise, it uses the standard validation logic.
    """
    try:
        # Step 1: User and Thread setup
        thread_id = None
        thread_id = check_user(request.user_id)
        
        param = request.param 
        final_message_with_current = request.message

        # Helper function to generate the error stream
        def get_error_stream_response(reason, solution):
            async def error_stream_generator() -> AsyncGenerator[str, None]:
                start_time = time.time()
                yield f"data: {json.dumps({'start_time': start_time, 'status': 'started'})}\n\n"

                # Set TTFB
                first_chunk_time = time.time()
                ttfb = first_chunk_time - start_time
                yield f"data: {json.dumps({'time_to_first_byte': ttfb})}\n\n"

                chunks = [
                    '{"', "answer", '":"',
                    "Your", " message", " was", " blocked", " by", " our", " content",
                    " policy", " because", " it", " was", " flagged", " as",
                    " inappropriate. \n\n", 
                    " Reason:", f" {reason} \n\n",
                    "Solution:", f" {solution}", 
                    ""
                ]

                for chunk in chunks:
                    yield f"data: {json.dumps({'content': chunk})}\n\n"

                end_time = time.time()
                yield f"data: {json.dumps({'done': True, 'total_time': end_time - start_time, 'blocked': True})}\n\n"

            return StreamingResponse(
                error_stream_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )

        # --- EXPLORE SYSTEM ---
        if request.param == "explore":
            # 1. Run Explore Agent
            explore_travel_agent_result = await Runner.run(explore_travel_agent, final_message_with_current)
            output = explore_travel_agent_result.final_output

            # 2. Check Validity
            if not output.isValid:
                return get_error_stream_response(output.reason, output.solution)

            # 3. PRIORITY CHECK: isPlanRelated
            if getattr(output, 'isPlanRelated', False):
                # If True, simply get complete response (standard) and return. Nothing else.
                response_content = await get_complete_response(request.message, thread_id, param)
                return JSONResponse(content={
                    "response": jsonable_encoder(response_content),
                    "type": "non-streaming"
                })
            
            # 4. ELSE: Check travel_type logic
            # If we are here, isPlanRelated was False. Now check specific-search-query.
            elif (output.isTravelRelated and getattr(output, 'travel_type', '') == "specific-search-query"):
                response_content = await get_complete_response_explore(request.message, thread_id, param)
                return JSONResponse(content={
                    "response": jsonable_encoder(response_content),
                    "type": "non-streaming"
                })
            
            # 5. Default Fallback -> Stream
            else:
                return StreamingResponse(
                    generate_stream(request.message, thread_id, request.reference),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )

        # --- DEFAULT VALIDATION SYSTEM ---
        else:
            # 1. Run Validation Agent
            validation_result = await Runner.run(validation_agent, final_message_with_current)

            # 2. Check Validity
            if not validation_result.final_output.isValid:
                return get_error_stream_response(
                    validation_result.final_output.reason, 
                    validation_result.final_output.solution
                )
            
            # 3. Check Travel logic (Standard)
            if validation_result.final_output.isTravelRelated:
                response_content = await get_complete_response(request.message, thread_id, param)
                return JSONResponse(content={
                    "response": jsonable_encoder(response_content),
                    "type": "non-streaming"
                })
            else:
                return StreamingResponse(
                    generate_stream(request.message, thread_id, request.reference),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    """Health check endpoint."""
    return {"status": "healthy", "service": "agent-streaming-api"}