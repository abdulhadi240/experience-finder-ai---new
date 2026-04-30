# HipTraveler Experience Finder AI — System Documentation

> **Version**: prod-v1  
> **Last Updated**: 2026-04-21  
> **Stack**: Python · FastAPI · OpenAI Agents · Zep Cloud · Supabase · Redis · AWS ECS

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Chat System](#3-chat-system)
4. [Validation System](#4-validation-system)
5. [Memory & Conversation State](#5-memory--conversation-state)
6. [Agent Definitions](#6-agent-definitions)
7. [API Reference](#7-api-reference)
8. [Data Models & Schemas](#8-data-models--schemas)
9. [External Integrations](#9-external-integrations)
10. [Infrastructure & Deployment](#10-infrastructure--deployment)
11. [Security Architecture](#11-security-architecture)
12. [Performance & Concurrency Model](#12-performance--concurrency-model)
13. [Configuration Reference](#13-configuration-reference)

---

## 1. System Overview

The **Experience Finder AI** is a streaming AI backend that powers travel recommendations on the HipTraveler platform. Users interact through a chat interface; the system responds with personalized place recommendations, trip plans, and real-time travel information.

**Core Capabilities**

| Capability | Description |
|------------|-------------|
| Explore Mode | RAG-backed place recommendations scoped to a portal's geography |
| Plan Mode | Multi-turn conversation to extract a structured `TripPlan` object |
| Web Search Mode | Real-time queries (weather, safety, events, pricing) via live web search |
| Research Validation | Background pipeline that validates and indexes new place data |
| Conversation Memory | Dual-layer memory: Zep Cloud (long-term) + Redis (session) |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HIPTRAVELER PLATFORM                                │
│                                                                             │
│   ┌────────────────┐                                          ┌──────────┐  │
│   │  Web Frontend  │◄── SSE Stream (chunked JSON tokens) ────│          │  │
│   │  (Next.js)     │─── POST /chat ─────────────────────────►│          │  │
│   └────────────────┘                                          │  FastAPI │  │
│                                                               │  (8080)  │  │
│   ┌────────────────┐                                          │          │  │
│   │  Admin / Tools │─── POST /validator/* ──────────────────►│          │  │
│   └────────────────┘                                          └────┬─────┘  │
└────────────────────────────────────────────────────────────────────┼────────┘
                                                                     │
              ┌──────────────────────────────────────────────────────┤
              │                                                       │
     ┌────────▼────────┐    ┌────────────────┐    ┌─────────────────▼────────┐
     │  CHAT PIPELINE  │    │  MEMORY LAYER  │    │  VALIDATION PIPELINE     │
     │                 │    │                │    │                          │
     │ ┌─────────────┐ │    │ ┌────────────┐ │    │ ┌──────────────────────┐ │
     │ │   Starter   │ │    │ │ Zep Cloud  │ │    │ │  Query Classifier    │ │
     │ │  Stream     │ │    │ │ (long-term)│ │    │ │  (generic/specific)  │ │
     │ └─────────────┘ │    │ └────────────┘ │    │ └──────────────────────┘ │
     │ ┌─────────────┐ │    │ ┌────────────┐ │    │ ┌──────────────────────┐ │
     │ │ Main Agent  │ │    │ │   Redis    │ │    │ │  Research Engine     │ │
     │ │ (RAG/Web)   │ │    │ │ (session)  │ │    │ │ OpenAI+Perplexity+  │ │
     │ └─────────────┘ │    │ └────────────┘ │    │ │      Gemini         │ │
     │ ┌─────────────┐ │    └────────────────┘    │ └──────────────────────┘ │
     │ │ Validation  │ │                           │ ┌──────────────────────┐ │
     │ │  Guard      │ │                           │ │  Supabase Upsert     │ │
     │ └─────────────┘ │                           │ └──────────────────────┘ │
     └─────────────────┘                           └──────────────────────────┘
              │                                                       │
     ┌────────▼───────────────────────────────────────────────────────▼──────┐
     │                          EXTERNAL SERVICES                            │
     │                                                                        │
     │  OpenAI API   Perplexity   Google Gemini   Google Maps   Zep Cloud    │
     │  RAG Webhook   Supabase       Redis           AWS ECS                 │
     └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chat System

### 3.1 Request Lifecycle

```
Client POST /chat
│
├── 1. RESOLVE CONVERSATION ID
│       Redis: get_or_create_conversation_id(user_id)
│       └── Returns existing ID (if within idle cutoff) or creates new UUID
│
├── 2. LOAD HISTORY
│       Redis: fetch_recent_interactions(user_id, conv_id, limit=20)
│       └── Fallback: old_interactions from request body
│
├── 3. BUILD FINAL MESSAGE
│       helpers.build_final_message(message, history, portal_reference)
│       └── Injects: conversation history, explore context, Zep preferences
│
├── 4. LOCATION SCOPE GATE  [Python only, ~0ms]
│       helpers._check_location_scope(final_message, reference)
│       ├── Absent portal → ALLOW
│       ├── In-scope country match → ALLOW
│       ├── Foreign country detected → BLOCK (with helpful message)
│       └── No location signal → ALLOW
│
├── 5. PARALLEL FIRE AT t=0
│       ├── [A] Starter Stream    → gpt-4.1-nano opener (TTFB ~300ms)
│       ├── [B] Main Agent        → RAG or Web Search agent
│       ├── [C] Input Validation  → validation_agent guardrails
│       └── [D] Zep Preferences   → async fetch user context
│
├── 6. PII CHECK [~150ms]
│       gpt-4.1-nano scans for: phone, email, address, credit card
│       └── BLOCK if PII found
│
├── 7. STREAM TOKENS TO CLIENT
│       SSE: {"content": token, "done": false}
│       ├── Starter tokens first (with 0.18s throttle)
│       └── Main agent tokens follow
│
└── 8. POST-STREAM SAVE
        Redis: append_interaction(user_id, conv_id, {question, answer})
        Zep: add_message(thread_id, user_message)  [async, non-blocking]
```

### 3.2 Streaming Pipeline (Concurrent Architecture)

```
t=0ms ──────────────────────────────────────────────────────────────────────►
        │
        ├──[STARTER]────────────────────────────►
        │   gpt-4.1-nano                         │ ~15-20 tokens
        │   "Great question! Let me research..."  │ throttled 0.18s/token
        │                                         ▼ ~3 seconds
        │
        ├──[MAIN AGENT]──────────────────────────────────────────────────────►
        │   ├── RAG lookup (explore)              │
        │   │   └── Webhook: rag.hiptraveler.com  │
        │   └── Web search (realtime queries)     │
        │       └── OpenAI WebSearchTool           │ ~3-8 seconds
        │                                          ▼
        ├──[VALIDATION]──────────────────────►
        │   validation_agent guardrails            │ ~200-500ms
        │   └── BLOCK path if invalid             ▼
        │
        └──[ZEP PREFS]────────────────────►
            User preference fetch                  │ ~400ms
            └── Injected into main agent context  ▼

        LOADING STATEMENTS (shown to user while waiting):
        t=0.5s: "Exploring your request..."
        t=3s:   "Searching the best spots..."
        t=6s:   "Personalising your results..."
        t=10s:  "Checking local insights..."
        t=15s:  "Almost there..."
        t=20s:  "Just a moment more..."
```

### 3.3 Mode Routing Decision Tree

```
POST /chat
│
├── param == "plan" ?
│   YES ──► Trip Planner Fast-Path
│           └── trip_planning_agent (skip validation middleware)
│               └── Returns: TripPlan JSON
│
└── param == "explore"
    │
    ├── _is_realtime_query(message) ?
    │   YES ──► web_search_agent
    │           ├── Signals: "weather", "is it safe", "right now", "events tonight"
    │           └── Returns: natural language with **bold** place names
    │
    └── NO ──► rag_format_agent
                ├── Builds RAG query from final_message
                └── Returns: formatted places with $$$$ metadata block
```

### 3.4 Loading Statements System

```
Loading Statement Generation (per request):
│
├── Fire gpt-4.1-nano with query context (t=0.5s intervals)
│   ├── Stage 1 (~0.5s):  Acknowledge user query
│   ├── Stage 2 (~3s):    "Researching" phase
│   ├── Stage 3 (~6s):    "Personalising" phase
│   ├── Stage 4 (~10s):   "Reassuring" phase
│   ├── Stage 5 (~15s):   "Anticipating" phase
│   └── Stage 6 (~20s):   "Patience" phase
│
└── Fallback: Static pool of 6 generic messages (any LLM error)
```

---

## 4. Validation System

The validator is a **standalone sub-application** mounted at `/validator`. It runs an independent research pipeline to discover, validate, and index travel destinations into the RAG system.

### 4.1 Validation Pipeline Overview

```
POST /validator/process  OR  POST /validator/deep_validation
│
├── 1. DUPLICATE CHECK
│       Supabase: query saved_queries WHERE query_text = input
│       └── If found and not expired → SKIP (return cached result)
│
├── 2. QUERY CLASSIFICATION  [LLM]
│       Classifier: "generic" | "specific" | "ignore"
│       │
│       ├── ignore  → Early exit (irrelevant or time-sensitive)
│       │
│       ├── generic → Generate 5 sub-queries
│       │             "best restaurants in Paris" →
│       │             ["top rated Paris restaurants 2024",
│       │              "must try Paris bistros",
│       │              "famous Paris brasseries", ...]
│       │
│       └── specific → Rewrite 1 query
│                       "Eiffel Tower" → "Eiffel Tower Paris visitor guide"
│
├── 3. PARALLEL RESEARCH  [3 engines simultaneously]
│       │
│       ├──[OpenAI]──────────────────────────────────────────────────────────►
│       │   Model: gpt-4.1 with web_search tool
│       │   Output: structured JSON with place names, ratings, sources
│       │
│       ├──[Perplexity]──────────────────────────────────────────────────────►
│       │   Model: sonar
│       │   Output: research text + citations array
│       │
│       └──[Gemini]───────────────────────────────────────────────────────────►
│           Model: gemini-2.5-flash + Google Search grounding
│           Output: grounded response with sources
│
├── 4. DOMAIN FILTERING
│       ├── Remove blacklisted sources
│       └── Prioritize whitelisted sources
│
├── 5. SCORING [per result]
│       Score 0–3:
│       ├── 3: Verified, authoritative, well-structured
│       ├── 2: Acceptable quality
│       ├── 1: Low quality
│       └── 0: Rejected
│
├── 6. CONVERSION TO ATTRACTION FORMAT  [gpt-4o]
│       ResearchResult → AttractionOutput
│       ├── Extract: title, category, lat/lng, tags
│       ├── Resolve: city, country code (Google Maps API)
│       ├── Generate: audience types, content summary
│       └── Exclude: items already in RAG (no duplication)
│
├── 7. SUPABASE UPSERT
│       Table: research_insights
│       └── Insert/update per attraction (Google Place ID as key)
│
└── 8. RAG WEBHOOK PUSH  [only score >= 2]
        POST https://rag.hiptraveler.com/chat
        └── Payload: AttractionOutput array
```

### 4.2 Input Guardrails (Per-Chat Validation)

```
Every /chat request runs validation_agent in parallel:

validation_agent checks:
│
├── PII Detection [gpt-4.1-nano, ~150ms]
│   Blocks: phone numbers, emails, addresses, credit cards, national IDs
│
├── Topic Relevance
│   ├── ALLOW: travel, food, accommodation, activities, transport
│   └── BLOCK: politics, medical, financial advice, adult content
│
├── Spam / Junk Detection
│   └── BLOCK: repeated nonsense, test strings, gibberish
│
└── Output Model: global_input_guardrail
    ├── isValid: bool
    ├── reason: str
    ├── isTravelRelated: bool
    ├── isMemoryQuery: bool  (triggers Zep lookup)
    └── solution: str       (rewrite suggestion if invalid)
```

### 4.3 Research Engine Detail

```
ResearchValidator
│
├── OpenAI Research
│   ├── Model: gpt-4.1
│   ├── Tool: web_search (built-in)
│   ├── System prompt: Extract places, ratings, descriptions
│   └── Output: { query, score, research, citations, location }
│
├── Perplexity Research
│   ├── Model: sonar
│   ├── API: api.perplexity.ai/chat/completions
│   ├── Messages: system (researcher role) + user (query)
│   └── Output: { query, score, research, citations: [...URLs] }
│
├── Gemini Research
│   ├── Model: gemini-2.5-flash
│   ├── Tools: Google Search grounding
│   ├── System instruction: Research assistant persona
│   └── Output: { query, score, research, sources: [...] }
│
└── Merge Strategy
    ├── Deduplicate by place name
    ├── Score 0–3 per source
    └── Pass score >= 2 to conversion
```

### 4.4 Domain Whitelist/Blacklist System

```
Domain Classification:
│
├── Whitelist (trusted sources)
│   Table: whitelist_domains
│   ├── Fields: category, country, source (domain), verified_at
│   ├── API: GET/POST/DELETE /validator/whitelist
│   └── Effect: Prioritized in research merge
│
└── Blacklist (blocked sources)
    ├── Maintained in code/config
    └── Effect: Excluded from research before scoring
```

---

## 5. Memory & Conversation State

### 5.1 Dual-Layer Memory Architecture

```
Memory System
│
├── LAYER 1: Zep Cloud  [Long-term, semantic]
│   │
│   ├── User Node
│   │   ├── user_id (external reference)
│   │   ├── email, first_name, last_name
│   │   └── node_summary (auto-generated by Zep AI)
│   │
│   ├── Thread Node
│   │   ├── thread_id (per conversation)
│   │   └── linked to user
│   │
│   ├── Messages
│   │   ├── role: user | assistant
│   │   └── content: text
│   │
│   └── Context Block
│       ├── Auto-summarized by Zep
│       └── Retrieved as context string per request
│
└── LAYER 2: Redis  [Short-term, session]

    Key: chat:active_conv:{user_id}
    Value: { conversation_id, last_seen_timestamp }
    TTL: REDIS_IDLE_CUTOFF_SECONDS (default: 600s)
    │
    Key: chat:interactions:{user_id}:{conversation_id}
    Value: [ {question, answer}, ... ]  (JSON list, max 30 items)
    TTL: REDIS_TTL_SECONDS (default: 600s)
```

### 5.2 Conversation ID Lifecycle

```
Request arrives with user_id
│
├── Redis ENABLED?
│   NO  → Return None (use old_interactions from request)
│   YES ↓
│
├── Key exists: chat:active_conv:{user_id}
│   NO  → Create new UUID, store with timestamp → RETURN new_id
│   YES ↓
│
├── age = now - last_seen
│   age > REDIS_IDLE_CUTOFF? (default 600s)
│   YES → Delete old key, create new UUID → RETURN new_id
│   NO  ↓
│
└── Update last_seen timestamp → RETURN existing_id
```

### 5.3 Memory Read/Write Flow Per Request

```
Incoming Request
│
├── READ [at request start]
│   ├── Redis: fetch_recent_interactions() → up to 20 Q&A pairs
│   ├── Zep:   get_user_preferences()     → user context summary
│   └── Zep:   get_message(thread_id)    → thread context string
│
├── INJECT [into agent context]
│   ├── Conversation history → final_message prefix
│   ├── Zep context          → agent system prompt
│   └── User preferences     → personalization hints
│
└── WRITE [after response complete]
    ├── Redis: append_interaction()  → {question, answer} pair
    └── Zep:   add_message()        → user question only (async)
```

### 5.4 Re-Engagement Flow

```
GET /memory/engage?user_id={id}
│
├── Zep: get_user_memory_for_engage(user_id)
│   └── Returns: user_summary + last_topic
│
├── Generate re-engagement question:
│   "Based on your interest in [last_topic], have you considered..."
│
└── Return SSE stream of the generated question
```

---

## 6. Agent Definitions

### 6.1 Agent Roster

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          AGENT REGISTRY                                 │
├───────────────────┬──────────────┬────────────────────────────────────┤
│ Agent             │ Model        │ Purpose                             │
├───────────────────┼──────────────┼────────────────────────────────────┤
│ trip_planning_    │ gpt-4o       │ Multi-turn structured trip          │
│ agent             │              │ extraction → TripPlan JSON           │
├───────────────────┼──────────────┼────────────────────────────────────┤
│ rag_format_agent  │ gpt-4o       │ RAG search + formatted              │
│                   │              │ recommendations                     │
├───────────────────┼──────────────┼────────────────────────────────────┤
│ web_search_agent  │ gpt-4.1      │ Real-time web search for            │
│                   │              │ weather/safety/events/pricing       │
├───────────────────┼──────────────┼────────────────────────────────────┤
│ validation_agent  │ gpt-4.1-nano │ Input guardrails (PII, topic,       │
│                   │              │ spam, off-topic)                    │
├───────────────────┼──────────────┼────────────────────────────────────┤
│ starter_agent     │ gpt-4.1-nano │ 1-2 sentence human opener           │
│                   │              │ (fast TTFB)                         │
├───────────────────┼──────────────┼────────────────────────────────────┤
│ loading_agent     │ gpt-4.1-nano │ Contextual loading statements       │
│                   │              │ shown during processing             │
└───────────────────┴──────────────┴────────────────────────────────────┘
```

### 6.2 Trip Planning Agent Logic

```
trip_planning_agent — Extraction Algorithm
│
├── INPUT: full conversation history + current message
│
├── STEP 1: Build mental summary
│   "User wants to go to [dest] for [n] days, [travelers]..."
│   Rules:
│   ├── Read ALL user messages (not just latest)
│   ├── States are PERMANENT until explicitly overridden
│   └── Assistant question BEFORE user answer = context for parsing
│
├── STEP 2: Extract fields
│   ├── destinations: list[str]  (cities, regions, countries)
│   ├── startDate: "MM-dd-yyyy"  (null if negative constraint)
│   ├── endDate: "MM-dd-yyyy"    (null if negative constraint)
│   ├── numDays: int             (derived from start/end if possible)
│   ├── month: str               (if only month mentioned)
│   ├── pax: { adults, children, infants, elderly }
│   ├── activities: list[str]
│   ├── travelStyle: list[str]
│   ├── themes: list[str]
│   ├── experienceTypes: list[str]
│   └── pois: list[str]  ← AUTO-POPULATED from LLM knowledge (never ask user)
│
├── STEP 3: Generate feedback (ordered priority)
│   ├── startDate  (unless user said "flexible" / "no dates")
│   ├── numDays    (if null and startDate known)
│   ├── destinations (if empty)
│   ├── pax        (parse: "couple" → adults:2, "solo" → adults:1)
│   ├── travelStyle
│   └── activities
│   Note: pois, themes, endDate, month, experienceTypes NEVER in feedback
│
├── STEP 4: Auto-populate POIs
│   "Paris" → ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral", ...]
│   ├── min 5 POIs per destination
│   └── Use LLM world knowledge (not RAG)
│
└── STEP 5: Build summary (next question for user)
    summary = format_question(feedback[0])
    "What dates are you thinking for your trip?"
```

### 6.3 RAG Format Agent Response Structure

```
RAG Agent Output Format:

[Natural language recommendation text]

$$$$
{
  "places": [
    {
      "name": "Place Name",
      "category": "Restaurant|Nature|Museum|...",
      "description": "...",
      "location": { "city": "...", "country": "..." },
      "tags": ["tag1", "tag2"],
      "rating": 4.5,
      "price_level": "$$"
    }
  ],
  "total_count": 5,
  "query_context": "..."
}
$$$$
```

---

## 7. API Reference

### 7.1 Chat Endpoints

```
Base URL: https://api.hiptraveler.com  (production)
         http://localhost:8080          (local)
```

#### `POST /chat`

Stream a chat response.

**Request Body**
```json
{
  "message": "Best things to do in Barcelona",
  "user_id": "user_abc123",
  "reference": "5ec4e10998603520ceb30fe5",
  "param": "explore",
  "threadId": "thread_optional",
  "old_interactions": [
    { "question": "...", "answer": "..." }
  ],
  "is_pro": false,
  "plan": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | User's message |
| `user_id` | string | Yes | Unique user identifier |
| `reference` | string | Yes | Portal reference (scopes geography) |
| `param` | string | Yes | `"explore"` or `"plan"` |
| `threadId` | string | No | Zep thread ID for memory |
| `old_interactions` | array | No | Fallback history if Redis unavailable |
| `is_pro` | bool | No | Pro user flag |
| `plan` | bool | No | If true, saves interaction to Zep |

**Response**: `text/event-stream`
```
data: {"content": "Great question!", "done": false}
data: {"content": " Here are", "done": false}
data: {"content": " the top spots...", "done": false}
data: {"content": "", "done": true, "total_time": 4.2, "mode": "explore"}
```

**Error Responses**

| Status | Condition |
|--------|-----------|
| 400 | PII detected in message |
| 400 | Location out of portal scope |
| 400 | Input validation failed |
| 422 | Invalid request body |
| 500 | Internal server error |

---

#### `GET /health`

Health check for load balancer.

**Response** `200 OK`
```json
{ "status": "ok" }
```

---

#### `POST /create_user`

Register a new user in Zep memory system.

**Request Body**
```json
{
  "user_id": "user_abc123",
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe"
}
```

---

#### `GET /delete_user?user_id={id}`

Remove user from Zep memory.

---

#### `GET /memory/engage?user_id={id}`

Generate a re-engagement message based on user's memory.

**Response**: `text/event-stream` (SSE tokens)

---

### 7.2 Validator Endpoints

#### `POST /validator/process`

Start background research for a query (async, deduplication-checked).

**Request Body**
```json
{
  "query": "best rooftop bars in Bangkok",
  "reference": "5ec4e10998603520ceb30fe5"
}
```

**Response** `202 Accepted`
```json
{ "status": "processing", "query": "best rooftop bars in Bangkok" }
```

---

#### `POST /validator/deep_validation`

Run full synchronous validation pipeline, return results.

**Request Body** (same as `/process`)

**Response** `200 OK`
```json
{
  "status": "success",
  "results": [
    {
      "title": "Sky Bar at Lebua",
      "category": "Nightlife",
      "city": "Bangkok",
      "country": "TH",
      "score": "3",
      "source": "https://...",
      "latitude": "13.7234",
      "longitude": "100.5130",
      "tags": "rooftop,cocktails,views,luxury",
      "meta_obj": {
        "audience": ["couples", "adults"],
        "ranking": "Top 10",
        "price_level": "$$$$"
      }
    }
  ]
}
```

---

#### Whitelist Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/validator/whitelist` | List all whitelisted domains |
| POST | `/validator/whitelist` | Add domain to whitelist |
| DELETE | `/validator/whitelist` | Remove domain from whitelist |
| GET | `/validator/whitelist/all` | Get full whitelist with metadata |
| DELETE | `/validator/whitelist/all` | Clear entire whitelist |
| GET | `/validator/whitelist/check?domain=X` | Check if domain is trusted |

---

#### Query History Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/validator/saved-queries` | View all saved queries |
| DELETE | `/validator/saved-queries` | Clear saved queries |

---

## 8. Data Models & Schemas

### 8.1 Chat Request / Response

```python
class Interaction(BaseModel):
    question: str
    answer: str

class QueryRequest(BaseModel):
    message: str
    user_id: str
    reference: str
    param: str               # "explore" | "plan"
    threadId: Optional[str]
    old_interactions: Optional[list[Interaction]]
    is_pro: Optional[bool] = False
    plan: Optional[bool] = False
```

### 8.2 Trip Plan Output

```python
class Pax(BaseModel):
    adults: Optional[int]
    children: Optional[int]
    infants: Optional[int]
    elderly: Optional[int]

class TripPlan(BaseModel):
    startDate: Optional[str]          # "MM-dd-yyyy"
    endDate: Optional[str]            # "MM-dd-yyyy"
    numDays: Optional[int]
    destinations: list[str]
    month: Optional[str]
    pax: Optional[Pax]
    experienceTypes: Optional[list[str]]
    travelStyle: Optional[list[str]]
    activities: Optional[list[str]]
    themes: Optional[list[str]]
    pois: list[str]                   # Always auto-populated
    feedback: Optional[list[str]]     # Missing fields
    summary: str                      # Next question for user
```

### 8.3 Validation Input Guardrail

```python
class global_input_guardrail(BaseModel):
    isValid: bool
    reason: str
    isTravelRelated: bool
    isMemoryQuery: bool
    solution: str           # Rewrite suggestion if invalid
```

### 8.4 Research / Attraction Models

```python
class ResearchResult(BaseModel):
    query: str
    score: str              # "0"–"3"
    research: str
    citations: list[str]
    location: Optional[str]
    maps_data: Optional[dict]

class MetaObj(BaseModel):
    audience: list[str]     # ["couples", "families", "solo", ...]
    location: str
    ranking: str
    price_level: str        # "$" | "$$" | "$$$" | "$$$$"

class AttractionOutput(BaseModel):
    title: str              # Exact place name only
    content: str
    category: str           # See categories below
    country: str            # ISO 2-letter code
    city: str
    source: str             # URL
    meta_obj: MetaObj
    latitude: str
    longitude: str
    tags: str               # Comma-separated
    region_code: Optional[str]
    query: str
```

**Valid Categories**:
`Restaurant` · `Nature & Parks` · `Museum & Culture` · `Shopping` · `Nightlife` · `Adventure` · `Attraction`

### 8.5 Supabase Table Schemas

**`research_insights`**
```sql
CREATE TABLE research_insights (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query        TEXT,
    title        TEXT NOT NULL,
    content      TEXT,
    category     TEXT,
    country      TEXT,
    city         TEXT,
    region_code  TEXT,
    latitude     FLOAT,
    longitude    FLOAT,
    language     TEXT DEFAULT 'en',
    tags         TEXT,
    source       TEXT,
    image        TEXT,
    meta_obj     JSONB,
    created_at   TIMESTAMPTZ DEFAULT now(),
    updated_at   TIMESTAMPTZ DEFAULT now()
);
```

**`saved_queries`**
```sql
CREATE TABLE saved_queries (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text   TEXT NOT NULL,
    expiry_days  INT DEFAULT 7,
    expiry_at    TIMESTAMPTZ,
    created_at   TIMESTAMPTZ DEFAULT now()
);
```

**`whitelist_domains`**
```sql
CREATE TABLE whitelist_domains (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category     TEXT,
    country      TEXT,
    source       TEXT NOT NULL,
    verified_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ DEFAULT now()
);
```

---

## 9. External Integrations

### 9.1 Integration Map

```
                     ┌──────────────────┐
                     │  Experience      │
                     │  Finder AI       │
                     └────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   LLM Services          Data Services         Search Services
        │                     │                     │
   ┌────▼─────┐          ┌────▼─────┐          ┌────▼─────┐
   │ OpenAI   │          │Supabase  │          │Perplexity│
   │ GPT-4o   │          │(Postgres)│          │  sonar   │
   │ GPT-4.1  │          └──────────┘          └──────────┘
   │ nano     │               │                     │
   └──────────┘          ┌────▼─────┐          ┌────▼─────┐
        │                │  Redis   │          │  Google  │
   ┌────▼─────┐          │(Upstash/ │          │  Gemini  │
   │   Zep    │          │ElastiC.) │          │2.5-flash │
   │  Cloud   │          └──────────┘          └──────────┘
   └──────────┘               │
        │                ┌────▼─────┐
   ┌────▼─────┐          │  Google  │
   │ RAG      │          │  Maps    │
   │ Webhook  │          │   API    │
   └──────────┘          └──────────┘
```

### 9.2 Service Details

| Service | Endpoint | Auth Method | Used For |
|---------|----------|-------------|----------|
| OpenAI API | `api.openai.com` | API Key | All agents, embeddings |
| Perplexity | `api.perplexity.ai` | API Key | Research validation |
| Google Gemini | `generativelanguage.googleapis.com` | API Key | Research + search grounding |
| Google Maps | `maps.googleapis.com` | API Key | Geocoding, place details |
| Zep Cloud | Cloud endpoint | API Key | Long-term user memory |
| Supabase | `*.supabase.co` | Service Role Key | Attraction database |
| Redis/Upstash | `*.upstash.io:6379` | Auth token (TLS) | Session cache |
| RAG Webhook | `rag.hiptraveler.com/chat` | None | Place search (explore) |
| n8n Webhook | `cs-automation.hiptraveler.com` | None | Customer RAG upsert |

### 9.3 OpenAI Model Usage

| Model | Temperature | Max Tokens | Tasks |
|-------|-------------|------------|-------|
| `gpt-4o` | 0.7–0.8 | unlimited | Trip planner, explore agent |
| `gpt-4.1` | 0.7 | unlimited | Web search agent, research |
| `gpt-4.1-nano` | 0.0–0.8 | 40–250 | Starter, PII check, loading, RAG summary |

---

## 10. Infrastructure & Deployment

### 10.1 AWS Architecture

```
                         Internet
                             │
                    ┌────────▼─────────┐
                    │ Application Load  │
                    │ Balancer (HTTPS)  │
                    │  Port 443 (ACM)   │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼──────────┐      ┌──────────▼──────────┐
    │  ECS Task (AZ-a)   │      │  ECS Task (AZ-b)    │
    │  CPU: 512           │      │  CPU: 512            │
    │  Memory: 1024MB     │      │  Memory: 1024MB      │
    │  Port: 8080         │      │  Port: 8080          │
    └─────────┬──────────┘      └──────────┬──────────┘
              │                             │
              └──────────┬──────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
    ┌─────────▼─────┐    ┌─────────▼─────────────┐
    │ ElastiCache   │    │  AWS SSM Parameter     │
    │ Redis 7.1     │    │  Store (secrets)       │
    │ cache.t4g.med │    └────────────────────────┘
    └───────────────┘

VPC: 10.0.0.0/16
  Subnet AZ-a: 10.0.0.0/20   (public)
  Subnet AZ-b: 10.0.16.0/20  (public)
```

### 10.2 Auto-Scaling Configuration

```
ECS Service Auto-Scaling:
  Min capacity:  2 tasks
  Max capacity:  6 tasks
  Scale-out:    CPU > 60% for 2 minutes → +1 task
  Scale-in:     CPU < 60% for 5 minutes → -1 task
  Cooldown:     60 seconds
```

### 10.3 CI/CD Pipeline

```
Git Push → main branch
│
├── GitHub Actions Trigger
│
├── STEP 1: Checkout code
│
├── STEP 2: AWS Authentication
│   └── OIDC (keyless, no stored credentials)
│       Role: arn:aws:iam::087307346846:role/github-actions-oidc
│
├── STEP 3: Build & Push Docker Image
│   ├── Platform: linux/amd64
│   ├── ECR Registry: 087307346846.dkr.ecr.us-east-1.amazonaws.com
│   ├── Repository: agentic-api
│   └── Tag: {COMMIT_SHA:7}  (e.g., "a1b2c3d")
│
├── STEP 4: Terraform Init
│   └── Backend: S3 bucket agentic-terraform-state-ai
│
├── STEP 5: Terraform Apply
│   └── Updates ECS task definition with new image URI
│
├── STEP 6: Force ECS Deployment
│   └── aws ecs update-service --force-new-deployment
│
└── STEP 7: Wait for stability
    └── aws ecs wait services-stable (timeout: 10 min)
```

### 10.4 Container Definition

```dockerfile
FROM python:3.12-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential gcc curl

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefer-binary -r requirements.txt

COPY . .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 11. Security Architecture

### 11.1 Security Layers

```
Incoming Request
│
├── LAYER 1: CORS Middleware
│   Allowed origins:
│   ├── http://localhost:3000
│   └── *.hiptraveler.com  (regex match)
│
├── LAYER 2: Location Scope Gate  [Python, ~0ms]
│   ├── Checks portal reference against REGION_METADATA
│   ├── Uses pycountry for country detection
│   └── Blocks off-geography queries instantly
│
├── LAYER 3: PII Detection  [gpt-4.1-nano, ~150ms]
│   Blocks: phone numbers, emails, addresses,
│           national IDs, credit card numbers
│
├── LAYER 4: Input Guardrails  [validation_agent, ~300ms]
│   Blocks: off-topic, spam, unsafe content
│
├── LAYER 5: Domain Filtering  [validator pipeline]
│   ├── Blacklist: excluded source domains
│   └── Whitelist: trusted source prioritization
│
└── LAYER 6: Secrets Management
    ├── Local: .env file
    └── Production: AWS SSM Parameter Store
        ├── /agentic/OPENAI_API_KEY
        ├── /agentic/ZEP_API_KEY
        ├── /agentic/SUPABASE_*
        ├── /agentic/GOOGLE_MAPS_API_KEY
        ├── /agentic/PERPLEXITY_API_KEY
        └── /agentic/REDIS_AUTH_TOKEN
```

### 11.2 Known Security Gaps

| Gap | Description | Risk |
|-----|-------------|------|
| No Auth | Routes accept any `user_id` without verification | Medium |
| No Rate Limiting | No per-user request throttling | Medium |
| Public Endpoints | `/validator/process` has no auth | Low |
| RAG Webhook | Unauthenticated calls to external RAG | Low |

---

## 12. Performance & Concurrency Model

### 12.1 Latency Targets

| Metric | Target | Description |
|--------|--------|-------------|
| TTFB | < 500ms | First token via starter agent |
| Starter duration | ~3s | 15–20 tokens at 0.18s/token |
| Main agent (RAG) | 3–6s | RAG lookup + formatting |
| Main agent (Web) | 5–10s | Web search + synthesis |
| PII check | < 200ms | gpt-4.1-nano classification |
| Validation | < 500ms | Input guardrails |
| Total response | 5–10s | End-to-end |

### 12.2 Concurrency Architecture

```
AsyncIO Event Loop
│
├── HTTP: aiohttp / httpx (non-blocking)
├── Redis: aioredis (non-blocking)
├── Supabase: supabase-py async client
│
├── Blocking calls wrapped in thread pools:
│   ├── Zep SDK (synchronous library)
│   └── pycountry lookups
│
└── Parallel task execution (asyncio.gather):
    ├── Starter stream
    ├── Main agent stream
    ├── Input validation
    └── Zep preference fetch
```

### 12.3 Connection Pooling

```
Redis Client:
├── Singleton pattern (module-level)
├── Connection pool via aioredis
└── TLS for all connections (REDIS_USE_TLS=true)

HTTP Clients:
├── Shared httpx.AsyncClient per service
└── Connection reuse across requests
```

---

## 13. Configuration Reference

### 13.1 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `ZEP_API_KEY` | Yes | — | Zep Cloud API key |
| `SUPABASE_URL` | No | — | Supabase project URL |
| `SUPABASE_KEY` | No | — | Supabase anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | No | — | Supabase service role key |
| `GOOGLE_MAPS_API_KEY` | No | — | Google Maps API key |
| `PERPLEXITY_API_KEY` | No | — | Perplexity AI key |
| `GEMINI_API_KEY` | No | — | Google Gemini API key |
| `TAVILY_API_KEY` | No | — | Tavily search key |
| `REDIS_ENABLED` | No | `false` | Enable Redis caching |
| `REDIS_URL` | No | — | Upstash Redis URL (preferred) |
| `REDIS_HOST` | No | — | ElastiCache host (fallback) |
| `REDIS_PORT` | No | `6379` | Redis port |
| `REDIS_AUTH_TOKEN` | No | — | Redis auth token |
| `REDIS_USE_TLS` | No | `true` | Enable TLS for Redis |
| `REDIS_TTL_SECONDS` | No | `600` | Data TTL (seconds) |
| `REDIS_IDLE_CUTOFF_SECONDS` | No | `600` | Conversation timeout |
| `REDIS_OLD_INTERACTIONS_LIMIT` | No | `20` | Max history items fetched |
| `REDIS_OLD_INTERACTIONS_MAX` | No | `30` | Max history items stored |

### 13.2 Portal Reference (region scope)

The `reference` field in chat requests is a portal identifier that scopes responses to a specific geography. Metadata for each portal is defined in `region_metadata.py`.

```python
REGION_METADATA = {
    "5ec4e10998603520ceb30fe5": {
        "country": "ES",
        "country_name": "Spain",
        "locations": ["Barcelona", "Madrid", "Seville", ...],
        "experiences": [...]
    },
    # ... more portals
}
```

### 13.3 Application Startup

```python
# main.py
app = FastAPI(title="Experience Finder AI")

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_origin_regex=r"https?://.*\.hiptraveler\.com|https?://hiptraveler\.com",
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(chat_router)         # /chat, /health, /memory
app.include_router(validator_router,    # /validator/*
    prefix="/validator")

@app.on_event("startup")
async def startup():
    # Initialize Redis connection pool
    await init_redis()
```

---

## Appendix A — File Reference

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app entry point, router mounting |
| `app/config.py` | Settings loader (pydantic BaseSettings) |
| `app/schemas.py` | Pydantic request/response models |
| `app/routes.py` | Chat + memory HTTP routes |
| `app/services.py` | Streaming pipeline, loading statements |
| `app/helpers.py` | Location gate, RAG helpers, stream orchestration |
| `app/agents_.py` | Agent definitions + complex instruction prompts |
| `app/tools.py` | Function tools for agents |
| `app/memory.py` | Zep Cloud integration |
| `app/redis_history.py` | Redis session cache |
| `app/region_metadata.py` | Portal geography metadata |
| `app/region_utils.py` | Country/region utility functions |
| `app/api/validator/routes.py` | Validator HTTP routes |
| `app/api/validator/helpers.py` | Validation helper functions |
| `app/api/validator/services/validator_service.py` | Core research engine |
| `app/api/validator/services/openai_service.py` | OpenAI API wrapper |
| `app/api/validator/services/conversion.py` | Research → Attraction format |
| `app/api/validator/services/supabase_service.py` | Database CRUD |
| `app/api/validator/services/whitelist_service.py` | Domain whitelist management |
| `app/api/validator/models/schemas.py` | Validator Pydantic models |
| `app/api/validator/config/prompt.py` | Validator LLM prompts |
| `infra/main.tf` | AWS ECS/VPC/Redis Terraform |
| `.github/workflows/deploy.yml` | CI/CD pipeline |
| `Dockerfile` | Container image |
| `requirements.txt` | Python dependencies |

---

## Appendix B — Data Flow Diagrams

### B.1 Full Explore Request Flow

```
User                  FastAPI              Redis         Agents          External
─────                 ───────              ─────         ──────          ────────
  │                      │                   │              │               │
  │── POST /chat ────────►│                   │              │               │
  │                      │                   │              │               │
  │              ┌───────┴────────┐           │              │               │
  │              │ Location Gate  │           │              │               │
  │              │ (Python, ~0ms) │           │              │               │
  │              └───────┬────────┘           │              │               │
  │                      │── get_conv_id ────►│              │               │
  │                      │◄─ conv_id ─────────│              │               │
  │                      │── fetch_history ──►│              │               │
  │                      │◄─ interactions ────│              │               │
  │                      │                   │              │               │
  │                      │── fire parallel tasks at t=0 ────────────────────│
  │                      │                   │ [A] starter_agent ──────────►│ gpt-4.1-nano
  │◄─ SSE: token1 ───────│◄─ stream ─────────│◄─ stream ──────────────────── │
  │◄─ SSE: token2 ───────│                   │ [B] validation_agent ───────►│ gpt-4.1-nano
  │◄─ SSE: token3 ───────│                   │ [C] zep_preferences ────────►│ Zep Cloud
  │                      │                   │ [D] rag_format_agent ───────►│ RAG Webhook
  │◄─ SSE: loading1 ─────│                   │              │               │
  │◄─ SSE: loading2 ─────│                   │◄─ validation ok ─────────────│
  │                      │                   │◄─ zep context ───────────────│
  │◄─ SSE: main_token1 ──│◄─ rag stream ─────│◄─ rag results ───────────────│
  │◄─ SSE: main_token2 ──│                   │              │               │
  │◄─ SSE: {done:true} ──│                   │              │               │
  │                      │── append_interaction ──────────►│              │
  │                      │── add_message (Zep async) ──────────────────────►│
  │                      │                   │              │               │
```

### B.2 Validator Research Flow

```
Trigger              Classifier         Research Engines        Database
───────              ──────────         ────────────────        ────────
   │                     │                     │                    │
   │── POST /process ────►│                     │                    │
   │                     │── check duplicate ──────────────────────►│
   │                     │◄─ exists? ──────────────────────────────│
   │                     │                     │                    │
   │              ┌──────┴──────┐              │                    │
   │              │  Classify   │              │                    │
   │              │  generic /  │              │                    │
   │              │  specific / │              │                    │
   │              │  ignore     │              │                    │
   │              └──────┬──────┘              │                    │
   │                     │── parallel research ►│                    │
   │                     │                OpenAI │───────────────────►
   │                     │              Perplexity ──────────────────►
   │                     │                Gemini ─────────────────────►
   │                     │◄─────────────── merged results ──────────│
   │                     │                     │                    │
   │              ┌──────┴──────┐              │                    │
   │              │   Domain    │              │                    │
   │              │  Filtering  │              │                    │
   │              └──────┬──────┘              │                    │
   │                     │                     │                    │
   │              ┌──────┴──────┐              │                    │
   │              │  Conversion │              │                    │
   │              │  gpt-4o     │──── Maps ──►│                    │
   │              └──────┬──────┘              │                    │
   │                     │── upsert ──────────────────────────────►│ research_insights
   │                     │── RAG push (score>=2) ──────────────────►│ RAG Webhook
   │                     │                     │                    │
   │◄── 202 Accepted ────│                     │                    │
```

---

*Documentation generated from source code analysis of `prod-v1` branch.*
