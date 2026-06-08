# app/services.py
import json
import re
import time
import asyncio
import random
from openai import AsyncOpenAI
from agents import Runner
from openai.types.responses import ResponseTextDeltaEvent
from .agents_ import trip_planning_agent, rag_format_agent, web_search_agent
from .config import settings
from .memory import check_user, add_message, get_message
from .tools import research_further

# Module-level singleton — one shared client, one connection pool, reused across all requests
_openai_client = AsyncOpenAI(api_key=settings.openai_api_key)


# ─── RAG Query Summarizer ────────────────────────────────────────

async def summarize_for_rag(message: str) -> str:
    """
    Ultra-fast gpt-4.1-nano call that converts the user message (which may
    contain conversation history or pronouns like 'there'/'it') into a clean,
    standalone search query for RAG.
    Falls back to the original message on any failure.
    """
    try:
        response = await _openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a travel query processor. Follow these steps in order:\n"
                        "1. If the message is not in English, translate it to English first.\n"
                        "2. Convert it into a concise standalone search query for a travel database. Resolve any pronouns or references using context.\n"
                        "3. Silently fix any misspelled destination, city, or place names (e.g. 'Karahic' → 'Karachi').\n"
                        "Return ONLY the final English query, no explanation, no punctuation at the end."
                    ),
                },
                {"role": "user", "content": message},
            ],
            max_tokens=40,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip()
        return result if result else message
    except Exception:
        return message


# ─── LLM-Generated Loading Statements ───────────────────────────

async def generate_loading_statements(message: str, param: str) -> list:
    """
    Call gpt-4.1-nano to produce 5 personalised loading statements for this specific query.
    Runs as a concurrent asyncio.Task — adds zero wall-clock latency to the pipeline.

    Returns a list of up to 5 strings.
    Falls back to [] on any failure; caller uses the static instant message + a
    deep-research fallback when these run out.
    """
    mode = "trip planner" if param == "plan" else "explore"

    prompt = (
        f"Travel AI conversation context (may include prior turns; newest message is last):\n{message}\n\n"
        f"Mode: {mode}\n\n"
        "Write exactly 5 short loading messages (6–10 words each) to display while the AI finds the answer.\n"
        "Rules:\n"
        "- FIRST infer what the user actually wants RIGHT NOW from the conversation. If earlier turns "
        "established a destination or focus (e.g. they were exploring Pakistan and just said 'islamabad' "
        "or 'yes'), the loaders MUST be about THAT destination/focus — never a different place.\n"
        "- Reference the specific destination or topic by name (e.g. 'Pulling up the best of Islamabad…'). "
        "Never mention a destination that isn't in the conversation.\n"
        "- Sound like a knowledgeable travel friend thinking out loud, not a chatbot.\n"
        "- Each message should feel like progress: looking → finding → comparing → narrowing → almost ready.\n"
        "- NEVER use: 'acknowledging', 'processing', 'computing', 'scanning', 'searching', or stage labels.\n"
        "- NEVER start with a capital-letter action word like 'Acknowledging' or 'Researching'.\n"
        "- Tone: warm, calm, confident — like the AI genuinely knows the answer and is retrieving it.\n"
        'Return ONLY valid JSON: {"messages": ["…", "…", "…", "…", "…"]}'
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
        if isinstance(stmts, list) and len(stmts) >= 1:
            return [str(s) for s in stmts[:5]]
    except Exception:
        pass

    return []


# ─── Instant First Loading Statement (fires at t=0) ──────────────

_INSTANT_STATEMENTS = {
    'explore': [
        "Pulling up the best picks for you…",
        "Finding what's worth your time here…",
        "Getting the good stuff together…",
        "Checking what stands out here…",
        "Rounding up the top options…",
        "Sorting through what's actually worth it…",
    ],
    'plan': [
        "Putting your trip together…",
        "Working out the details for you…",
        "Getting your itinerary started…",
        "Mapping this out for you…",
        "Building your trip plan now…",
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


# ─── Human-sounding Starter → Queue Streamer ─────────────────────

async def check_pii_fast(message: str) -> bool:
    """
    Lightweight PII-only guardrail using gpt-4.1-nano.
    Fires at t=0 in parallel with the starter — completes in ~150-200ms.
    Returns True if PII is detected (block the response).
    """
    try:
        response = await _openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a PII detector. Reply with exactly YES or NO.\n"
                        "Reply YES if the message contains any: phone numbers, email addresses, "
                        "physical addresses, passport or ID numbers, credit/debit card numbers, "
                        "or social security numbers.\n"
                        "Reply NO for everything else including names, cities, or travel queries."
                    ),
                },
                {"role": "user", "content": message[:600]},
            ],
            max_tokens=3,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().upper()
        return result.startswith("YES")
    except Exception:
        return False   # fail open — never block on error


async def stream_starter_to_queue(message: str, param: str, queue: asyncio.Queue) -> None:
    """
    Streams a 1–2 sentence human-sounding opener into the queue at t=0.
    Runs in parallel with the main agent — gives the user instant real content
    while the agent processes RAG and generates recommendations.
    """
    prompt = (
    f"You are HipTraveler AI. Read the conversation below and write ONE short bridge sentence for the user while the main system processes their request. Focus on the LAST user message only.\n\n{message}\n\n"
    "Write exactly ONE sentence. Pick the matching case and output ONLY a sentence like the examples shown — never output the instructions themselves:\n"
    "- Unsafe/off-topic/spam → 'I can only assist with travel-related topics.'\n"
    "- Greeting → 'Hi there! Where are you dreaming of traveling to?'\n"
    "- Preferences/history question → 'Here is what your travel profile shows:'\n"
    "- Trip planning/itinerary → 'On it — putting your trip together!' or 'Great choice — let me plan that out for you!'\n"
    "- Safety/weather/live info → 'Checking current conditions in [destination] for you.' or 'Let me look into that for you.'\n"
    "- Best time/season → 'Good question — the timing can really shape the experience.'\n"
    "- Recommendations/best-of → 'There are some great options — let me pull up the best ones for you.'\n"
    "- Anything else about a destination → 'That destination has a lot to offer — pulling up the details now.'\n"
    "Rules: Output ONLY the bridge sentence. Never echo instructions. Never list places. Never ask questions. Never say 'are you looking for' or offer choices. Never introduce yourself. Never start with Certainly/Great/Sure/Absolutely."
    )
    try:
        stream = await _openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.7,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                await queue.put(delta)
    except Exception:
        pass
    finally:
        await queue.put(None)  # sentinel — always fired


# ─── Agent → Queue Streamer ──────────────────────────────────────

async def stream_agent_to_queue(
    agent_name: str,
    final_message_with_ref: str,
    original_message: str,
    thread_id: str,
    queue: asyncio.Queue,
    is_pro: bool = False,
) -> None:
    """
    Run the agent stream and push every token into `queue` as it arrives.

    Sentinel protocol:
      • Each text chunk   → str put into queue
      • Error             → Exception instance put into queue
      • Stream complete   → None put into queue  (always sent via finally)

    Validation routing:
      • rag_format_agent  → fire research_further(question) immediately — validator
                            researches places for that query broadly.
      • web_search_agent  → accumulate full answer, then fire
                            research_further(question + answer) so the validator
                            targets only the specific places web search returned.

    Zep saves are handled upstream in _main_stream — not here.
    """
    try:
        # RAG path: send just the question to the validator right away
        if agent_name == 'rag_format_agent':
            research_further(original_message)

        if agent_name == 'rag_format_agent':
            result = Runner.run_streamed(rag_format_agent, final_message_with_ref)
        else:
            result = Runner.run_streamed(web_search_agent, final_message_with_ref)

        # Strip the {"answer":"..."} JSON wrapper — handles compact and pretty-printed JSON.
        _PREFIX_RE  = re.compile(r'\{\s*"answer"\s*:\s*"')
        _SUFFIX_LEN = 10
        prefix_buf  = ""
        prefix_done = False
        suffix_buf  = ""

        # Web search path: collect all emitted tokens so we can send them to the validator
        answer_parts = [] if agent_name == 'web_search_agent' else None

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                chunk = event.data.delta
                if not chunk:
                    continue

                # ── Strip prefix ────────────────────────────────────
                if not prefix_done:
                    prefix_buf += chunk
                    m = _PREFIX_RE.search(prefix_buf)
                    if m:
                        prefix_done = True
                        chunk = prefix_buf[m.end():]
                        prefix_buf = ""
                        if not chunk:
                            continue
                    elif len(prefix_buf) > 40:
                        prefix_done = True
                        chunk = prefix_buf
                        prefix_buf = ""
                    else:
                        continue

                # ── Rolling suffix buffer to strip trailing "} or "\n} ──
                pending = suffix_buf + chunk
                if len(pending) > _SUFFIX_LEN:
                    emit = pending[:-_SUFFIX_LEN]
                    emit = re.sub(r'\s*Source:\s*\S+', '', emit)
                    if emit:
                        await queue.put(emit)
                        if answer_parts is not None:
                            answer_parts.append(emit)
                    suffix_buf = pending[-_SUFFIX_LEN:]
                else:
                    suffix_buf = pending

        # Flush suffix
        if suffix_buf:
            cleaned = re.sub(r'"?\s*\}?\s*$', '', suffix_buf)
            cleaned = re.sub(r'\s*Source:\s*\S+', '', cleaned)
            if cleaned:
                await queue.put(cleaned)
                if answer_parts is not None:
                    answer_parts.append(cleaned)

        # Web search path: now that the full answer is ready, send question + answer
        # to the validator so it researches only the places web search returned.
        # Strip <POIS>...</POIS> XML blocks and any stray "Source: ..." annotations
        # the agent may have appended despite instructions.
        if answer_parts:
            full_answer = ''.join(answer_parts)
            full_answer = re.sub(r'<POIS>.*?</POIS>', '', full_answer, flags=re.DOTALL)
            full_answer = re.sub(r'\s*Source:\s*\S+', '', full_answer)
            full_answer = full_answer.strip()
            research_further(f"{original_message}\n\nAnswer:\n{full_answer}")

        # Assistant answers are not saved to Zep — only user questions are saved

    except Exception as e:
        await queue.put(e)
    finally:
        await queue.put(None)

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
            "param": mode,
            "threadId": thread_id,
            "total_time": f"{total_time:.2f} seconds",
            "response_type": "non_streaming",
            "plan": True
        }
            
        return full_response, timing_info

    except Exception as e:
        raise Exception(f"Agent error: {str(e)}") from e
    
    
    
