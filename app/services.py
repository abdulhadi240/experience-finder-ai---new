# app/services.py
import json
import time
import asyncio
import random
from typing import AsyncGenerator
from openai import AsyncOpenAI
from agents import Runner
from openai.types.responses import ResponseTextDeltaEvent
from .agents_ import general_agent , trip_planning_agent , explore_agent # Import the configured agents
from .config import settings
from .memory import check_user, add_message, get_message
from .tools import research_further

# Module-level singleton — one shared client, one connection pool, reused across all requests
_openai_client = AsyncOpenAI(api_key=settings.openai_api_key)


# ─── LLM-Generated Loading Statements ───────────────────────────

async def generate_loading_statements(message: str, param: str) -> list:
    """
    Call gpt-4.1-nano to produce 6 personalised loading statements for this specific query.
    Runs as a concurrent asyncio.Task — adds zero wall-clock latency to the pipeline.

    Returns a list of up to 6 strings.
    Falls back to [] on any failure; caller uses the static pool as fallback.

    Layout (caller responsibility):
      statements[0-2]  → shown at stages 0-2  (0.5 s, 3 s, 6 s)
      statements[3-5]  → shown at stages 3-5  (10 s, 15 s, 20 s)
    """
    mode = "trip planner" if param == "plan" else "explore"

    prompt = (
        f"Travel AI conversation context (may include history):\n{message}\n\n"
        f"Mode: {mode}\n\n"
        "Using the full context above (destination, topic, conversation history), "
        "write exactly 6 short loading messages (5–10 words each) to display while the AI thinks.\n"
        "Progression: acknowledgment → researching → personalising → reassuring → anticipating → patience.\n"
        "Naturally reference the destination or topic from the conversation where it fits. "
        "Never use the words 'searching', 'scanning', 'processing', 'computing', or imply heavy server work.\n"
        "Tone: calm, confident, curated — the AI is thinking, not processing.\n"
        'Return ONLY valid JSON: {"messages": ["…", "…", "…", "…", "…", "…"]}'
    )

    try:
        response = await _openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.8,
        )
        raw      = response.choices[0].message.content.strip()
        data     = json.loads(raw)
        stmts    = data.get("messages", [])
        if isinstance(stmts, list) and len(stmts) >= 3:
            return [str(s) for s in stmts[:6]]
    except Exception as e:
        print(f"generate_loading_statements failed: {e}")

    return []


# ─── Instant First Loading Statement (fires at t=0) ──────────────

_INSTANT_STATEMENTS = {
    'explore': [
        "On it — one sec…",
        "Let me check that for you…",
        "Looking into this now…",
        "Right on it…",
        "Give me a moment…",
    ],
    'plan': [
        "On it — one sec…",
        "Great question — checking that now…",
        "Smart move — let me look into that…",
        "Right, let me pull that up…",
        "Give me just a moment…",
    ],
}


def get_instant_loading_message(param: str) -> str:
    mode = 'plan' if param == 'plan' else 'explore'
    return random.choice(_INSTANT_STATEMENTS[mode])


# ─── Loading Message Pools ───────────────────────────────────────

_LOADING_STAGES = {
    'explore': [
        {  # Stage 0 — 0.5s — Acknowledgment
            'generic':  ["Looking into this…", "Let me check…", "On it."],
            'injected': ["Exploring {topic}…", "Looking into {topic}…"],
        },
        {  # Stage 1 — 3s — Signal Depth
            'generic':  ["Filtering the good stuff…", "Comparing the smart picks…", "Sorting through options…"],
            'injected': ["Checking what stands out for {topic}…", "Reviewing what makes {topic} special…"],
        },
        {  # Stage 2 — 6s — Personalisation Cue
            'generic':  ["Matching this to your travel style…", "Prioritising quality over hype…", "Narrowing it down…"],
            'injected': ["Finding the best fit for {topic}…", "Looking at timing, vibe, and value for {topic}…"],
        },
        {  # Stage 3 — 10s — Reassurance
            'generic':  ["Still with you…", "This one's worth getting right…", "Making sure this is solid…", "Avoiding the tourist traps…"],
            'injected': ["Making sure {topic} works for your goals…", "Double-checking the smartest options for {topic}…"],
        },
        {  # Stage 4 — 15s — Anticipation
            'generic':  ["Almost ready.", "Pulling it together…", "Final touches…", "This is shaping up nicely…"],
            'injected': ["Finalising the best picks in {topic}…", "Locking in the strongest options for {topic}…"],
        },
        {  # Stage 5 — 20s — Patience Reinforcement
            'generic':  ["Thanks for your patience — quality takes a second.", "Good answers beat fast answers.", "Nearly there — promise."],
            'injected': [],
        },
    ],
    'plan': [
        {  # Stage 0
            'generic':  ["Great question — checking that now…", "Smart move asking that…", "On it — one sec…"],
            'injected': ["Checking the best timing for {topic}…", "Looking into {topic} now…"],
        },
        {  # Stage 1
            'generic':  ["Pulling the details…", "Reviewing the specifics…", "Comparing seasonal factors…"],
            'injected': ["Pulling details on {topic}…"],
        },
        {  # Stage 2
            'generic':  ["Making sure this fits your trip…", "Factoring in crowds and value…", "Looking at what really matters…"],
            'injected': [],
        },
        {  # Stage 3
            'generic':  ["Almost there…", "Then we'll jump back to your itinerary…", "Making sure you plan this right…"],
            'injected': [],
        },
        {  # Stage 4
            'generic':  ["Wrapping it up now…", "Finalising your answer…", "Bringing it together…"],
            'injected': [],
        },
        {  # Stage 5
            'generic':  ["Thanks for hanging with me.", "Worth the wait.", "This is how pros plan trips."],
            'injected': [],
        },
    ],
}

# Millisecond thresholds at which each stage fires
_STAGE_TIMINGS_MS = [500, 3000, 6000, 10000, 15000, 20000]


def get_loading_message(stage: int, topic, param: str) -> str:
    """Return a random loading message for the given stage, param, and optional topic."""
    mode = 'plan' if param == 'plan' else 'explore'
    stages = _LOADING_STAGES[mode]
    stage_data = stages[min(stage, len(stages) - 1)]

    injected = stage_data.get('injected', [])
    generic  = stage_data.get('generic', ["Working on it…"])

    if topic and injected:
        msg = random.choice(injected)
        return msg.replace('{topic}', topic)

    return random.choice(generic)


# ─── Agent → Queue Streamer ──────────────────────────────────────

async def stream_agent_to_queue(
    agent_name: str,
    final_message_with_ref: str,
    original_message: str,
    thread_id: str,
    queue: asyncio.Queue,
) -> None:
    """
    Run the agent stream and push every token into `queue` as it arrives.
    The caller polls the queue while simultaneously advancing loading stages;
    the moment the first token lands the caller drops loading and streams live.

    Sentinel protocol:
      • Each text chunk   → str put into queue
      • Error             → Exception instance put into queue
      • Stream complete   → None put into queue  (always sent via finally)
    """
    try:
        add_message(role='user', thread_id=thread_id, message=original_message)
        research_further(final_message_with_ref)

        if agent_name == 'general_agent':
            result = Runner.run_streamed(general_agent, final_message_with_ref)
        else:
            result = Runner.run_streamed(explore_agent, final_message_with_ref)

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                chunk = event.data.delta
                if chunk:
                    await queue.put(chunk)

    except Exception as e:
        await queue.put(e)          # consumer yields error event then breaks
    finally:
        await queue.put(None)       # sentinel — always fired

async def generate_stream(message: str, thread_id: str , reference: str , agent: str , final_message: str) -> AsyncGenerator[str, None]:
    """Generates a streaming response in Server-Sent Events (SSE) format."""
    start_time = time.time()
    first_chunk_time = None
    full_response_content = ""
    
    try:
        
        add_message(role='user', thread_id=thread_id, message=message)
        print("Stream Start")
        # Append the latest message to final_message before sending to agent
        final_message_with_current = final_message + "\n\n Reference : " + reference
        print("Stream Start " + final_message_with_current)

        research_further(final_message_with_current)
        
        if agent == 'general_agent':
            result = Runner.run_streamed(general_agent, final_message_with_current)
        else:
            result = Runner.run_streamed(explore_agent, final_message_with_current)

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
    
    #finally:
        # Add the assistant's response to memory after the streaming is complete
        #if thread_id and full_response_content:
            #async def save_message():
                #add_message(role='assistant', thread_id=thread_id, message=full_response_content)
            #asyncio.create_task(save_message())

async def get_complete_response(message: str, thread_id: str , mode: str) -> tuple[str, dict]:
    """Generates a complete, non-streamed response and provides timing info."""
    start_time = time.time()
    
    try:        
        result = await Runner.run(trip_planning_agent, message) 
        
        # Access the actual response data
        full_response = result.final_output  
        #add_message(role='assistant', thread_id=thread_id, message=full_response) 
        
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
        #add_message(role='assistant', thread_id=thread_id, message=full_response) 
        
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