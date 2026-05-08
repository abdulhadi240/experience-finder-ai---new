# Experience Finder AI — System Documentation

**Version:** 1.0  
**Platform:** HipTraveler  
**Stack:** Python 3.12 · FastAPI · OpenAI Agents · Zep Cloud · Redis · Supabase

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Core Components](#3-core-components)
4. [Chat Endpoint (`POST /chat`)](#4-chat-endpoint-post-chat)
5. [Validation Endpoint (`POST /validator/process`)](#5-validation-endpoint-post-validatorprocess)
6. [Agent Definitions (`agents_.py`)](#6-agent-definitions-agents_py)
7. [Data Flow Diagrams](#7-data-flow-diagrams)
8. [Supporting Services](#8-supporting-services)
9. [API Reference Summary](#9-api-reference-summary)

---

## 1. System Overview

The **Experience Finder AI** is a multi-agent conversational travel assistant built for the HipTraveler platform. It handles two primary interaction modes:

| Mode | Parameter | Description |
|------|-----------|-------------|
| **Plan** | `param = "plan"` | Extracts structured trip data from conversation and builds a travel plan |
| **Explore** | `param = "explore"` | Retrieves curated travel recommendations via RAG or live web search |

The system is built around a **streaming-first architecture** — all responses are delivered as Server-Sent Events (SSE) to the client, enabling token-by-token real-time output. Every request passes through a multi-stage AI pipeline: guardrail validation → intent classification → agent routing → response streaming.

### Key Capabilities

- **Conversational trip planning** with persistent memory across sessions
- **RAG-powered recommendations** from a curated travel knowledge base
- **Live web search fallback** for real-time data (events, pricing, visa requirements)
- **Portal geofencing** — each client portal is scoped to a specific geographic region
- **Multi-tenant** — supports multiple regional portals via the `reference` field
- **Memory re-engagement** — personalised outreach based on user's travel preferences

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT (Frontend)                          │
│                     POST /chat  ·  SSE Stream                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                          │
│                           main.py  · port 8080                      │
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────────────────────────┐   │
│  │   /chat router   │    │        /validator router             │   │
│  │  app/routes.py   │    │   app/api/validator/routes.py        │   │
│  └────────┬─────────┘    └──────────────────────────────────────┘   │
└───────────┼─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Core Orchestrator (helpers.py)                  │
│                          _main_stream()                             │
│                                                                     │
│  ┌───────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│  │ Redis History │  │  Zep Memory  │  │   Region Metadata      │   │
│  │  (Cache TTL)  │  │  (Long-term) │  │  (Portal Geofencing)   │   │
│  └───────────────┘  └──────────────┘  └────────────────────────┘   │
└───────────┬─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AI Agent Layer                              │
│                                                                     │
│  ┌──────────────────┐  ┌─────────────────┐  ┌───────────────────┐  │
│  │  Guardrail Agent │  │ Trip Planning   │  │  Customer Service │  │
│  │  (Input Safety)  │  │     Agent       │  │      Agent        │  │
│  └──────────────────┘  └─────────────────┘  └───────────────────┘  │
│                                                                     │
│  ┌──────────────────┐  ┌─────────────────┐  ┌───────────────────┐  │
│  │  Explore Travel  │  │  RAG Format     │  │  Web Search       │  │
│  │     Agent        │  │     Agent       │  │     Agent         │  │
│  └──────────────────┘  └─────────────────┘  └───────────────────┘  │
└───────────┬─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      External Integrations                          │
│                                                                     │
│   OpenAI API    ·    RAG Service    ·   n8n Webhook   ·  Supabase  │
│   (GPT-4/5)          (RAG API)       (Customer FAQ)    (Database)   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

| File | Role |
|------|------|
| `main.py` | FastAPI app init, CORS config, router registration |
| `app/routes.py` | HTTP route handlers (`/chat`, `/create_user`, `/delete_user`, `/memory/engage`) |
| `app/helpers.py` | Main orchestrator — `_main_stream()`, query detection, context building |
| `app/agents_.py` | All AI agent definitions and system prompts |
| `app/services.py` | OpenAI client wrappers — streaming, summarisation, loading messages |
| `app/tools.py` | Function tools registered to agents (RAG calls, n8n webhook) |
| `app/memory.py` | Zep Cloud integration — user sessions, threads, conversation memory |
| `app/redis_history.py` | Short-term conversation cache via Redis (Upstash / ElastiCache) |
| `app/schemas.py` | Pydantic models for all request/response types |
| `app/config.py` | Environment settings loaded from `.env` |
| `app/region_metadata.py` | Portal-to-geography mapping for geofencing |
| `app/api/validator/` | Standalone research validation sub-API |

---

## 4. Chat Endpoint (`POST /chat`)

### Purpose

The `/chat` endpoint is the primary entry point for all user interactions. It accepts a user message and streams a real-time AI response back as Server-Sent Events.

### Request Schema

```json
{
  "message": "string",
  "user_id": "string",
  "reference": "string",
  "param": "plan | explore",
  "threadId": "string (optional)",
  "old_interactions": [ { "question": "...", "answer": "..." } ],
  "is_pro": false,
  "plan": false
}
```

| Field | Description |
|-------|-------------|
| `message` | The user's raw message |
| `user_id` | Unique user identifier |
| `reference` | Portal reference code (determines geographic scope) |
| `param` | `"plan"` for trip planning mode, `"explore"` for discovery mode |
| `threadId` | Optional client session ID |
| `old_interactions` | Fallback conversation history if Redis is empty |
| `is_pro` | Pro-tier flag for feature access |
| `plan` | When `true`, enables long-term Zep memory save/load |

### Response

**Type:** `StreamingResponse`  
**Content-Type:** `text/event-stream`  
**Format:** Server-Sent Events (SSE), one JSON object per event

```
data: {"type": "token", "content": "Hello"}
data: {"type": "token", "content": " there"}
data: {"type": "done"}
```

### Processing Pipeline

```
POST /chat
    │
    ├─ 1. Generate unique thread_id
    │
    ├─ 2. Redis: get or create conversation_id
    │         (idle cutoff: 600s → starts fresh conversation)
    │
    ├─ 3. Redis: fetch recent interactions (up to 20)
    │         (fallback to old_interactions if Redis empty)
    │
    ├─ 4. Build final_message with conversation context
    │
    └─ 5. _main_stream() → StreamingResponse (SSE)
```

### Conversation Context & Memory

The system uses a **two-layer memory architecture**:

```
┌────────────────────────────────────────────────┐
│          SHORT-TERM: Redis (Session)            │
│  • Stores raw Q&A pairs per conversation_id     │
│  • TTL: 600 seconds (configurable)              │
│  • Idle cutoff: 600 seconds → new conversation  │
│  • Limit: last 20 interactions per session      │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│          LONG-TERM: Zep Cloud (User)            │
│  • Stores thread per user_id                    │
│  • Summarises conversation over time            │
│  • Retrieves travel preferences & history       │
│  • Only active when plan=true in request        │
└────────────────────────────────────────────────┘
```

---

## 5. Validation Endpoint (`POST /validator/process`)

### Purpose

The `/validator/process` endpoint is a background research pipeline. When a travel query comes in, it classifies the query, checks for duplicates, and triggers an async research task to enrich the knowledge base. This feeds the RAG system used by the explore flow.

### Request Schema

```json
{
  "query": "string",
  "reference": "string"
}
```

### Response

```json
{
  "message": "Research has started",
  "duplicate": false
}
```

### Processing Flow

```
POST /validator/process
    │
    ├─ 1. Extract clean question (strips "Answer:" block if present)
    │
    ├─ 2. Generate embedding for duplicate detection
    │
    ├─ 3. Supabase: check similarity against stored queries
    │         (cosine similarity threshold)
    │
    ├─ 4. If unique: insert query into Supabase
    │
    └─ 5. Launch async background research task
              │
              ├─ Classify query: "generic" or "specific"
              │
              ├─ Generic  → expand to 5 research sub-queries
              │   Specific → rewrite into 1 focused query
              │
              ├─ For each sub-query: research + score (0/3 to 3/3)
              │
              └─ Store results in Supabase research insights table
```

### Query Classification

| Type | Behaviour | Sub-queries |
|------|-----------|-------------|
| `generic` | Broad travel topic | Expanded into 5 targeted research queries |
| `specific` | Concrete place/experience | Rewritten as 1 precise query |
| `ignore` | Irrelevant or non-actionable | Discarded — no research triggered |

### Additional Validator Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `GET /validator/` | GET | API info and endpoint listing |
| `GET /validator/health` | GET | Health check |
| `POST /validator/deep_validation` | POST | Synchronous full research pipeline |
| `GET /validator/whitelist` | GET | List whitelisted domains |
| `POST /validator/whitelist` | POST | Add trusted domain |
| `DELETE /validator/whitelist` | DELETE | Remove domain |
| `GET /validator/saved-queries` | GET | List non-expired research queries |
| `POST /validator/rag/upsert` | POST | Upsert document into RAG knowledge base |
| `POST /validator/insights/sample` | POST | Insert sample research insight |

### RAG Upsert (`POST /validator/rag/upsert`)

Used to push travel content (experiences, venues, destinations) into the RAG index.

```json
{
  "id": "unique-doc-id",
  "title": "Baha Mar Resort",
  "content": "Full description...",
  "query": "luxury resorts Nassau",
  "category": "stay",
  "country": "The Bahamas",
  "city": "Nassau",
  "language": "en",
  "tags": "luxury,beach,resort",
  "latitude": 25.083,
  "longitude": -77.337,
  "meta_obj": {
    "rating": 4.8,
    "reviews": 3200,
    "price_range": "$$$"
  }
}
```

Supported languages: `en`, `fr`, `it`, `de`, `es`, `zh`

---

## 6. Agent Definitions (`agents_.py`)

The file `app/agents_.py` defines all six AI agents used in the system. Each agent is configured with a model, a structured output schema, and a detailed system prompt.

---

### Agent 1 — Trip Planning Agent

| Property | Value |
|----------|-------|
| **Model** | `gpt-5.4` |
| **Output** | `TripPlan` (structured JSON) |
| **Mode** | `param = "plan"` |

**Purpose:** Conducts a guided conversational interview to extract all trip planning data. Reads the full conversation history before asking any question to avoid repetition.

**Output Schema:**

```json
{
  "startDate": "MM-dd-yyyy or null",
  "endDate": "MM-dd-yyyy or null",
  "numDays": 7,
  "destinations": ["Nassau", "Freeport"],
  "month": "July",
  "pax": {
    "adults": 2,
    "children": 1,
    "infants": 0,
    "elderly": 0
  },
  "experienceTypes": ["beach", "culture"],
  "travelStyle": ["luxury"],
  "activities": ["snorkelling", "local food"],
  "themes": ["family"],
  "pois": ["Atlantis Paradise Island"],
  "feedback": ["numDays"],
  "summary": "Sounds wonderful! How many days are you planning to travel?"
}
```

**Key Behaviours:**
- Asks one question at a time in the `summary` field — never a list
- Prioritises `startDate` → `numDays` → `destinations` → `pax` → `travelStyle` → `activities`
- Infers traveller count from natural language ("couple" → 2 adults, "family of four" → 2 adults + 2 children)
- Consolidates multiple cities into state or country when appropriate
- Honours date refusals permanently — does not re-ask for refused fields
- Auto-populates `pois` from mentioned landmarks or activities

---

### Agent 2 — Customer Service Agent

| Property | Value |
|----------|-------|
| **Model** | `gpt-4.1-nano` |
| **Output** | `Output_Format` |
| **Tool** | `customer_rag_n8n` (n8n webhook → FAQ RAG) |

**Purpose:** Handles platform-level customer support queries (account issues, billing, feature questions). Retrieves pre-built answers via RAG — never paraphrases or rewrites the retrieved answer.

---

### Agent 3 — Guardrail Agent (Validation)

| Property | Value |
|----------|-------|
| **Model** | `gpt-5.4` |
| **Output** | `global_input_guardrail` |
| **Stage** | First — runs on every request before routing |

**Purpose:** Safety and relevance gate. Every user message is screened before reaching any downstream agent.

**Output Schema:**

```json
{
  "isValid": true,
  "reason": "CLEAN",
  "isTravelRelated": true,
  "isMemoryQuery": false,
  "solution": ""
}
```

**Reason Codes:**

| Code | Meaning |
|------|---------|
| `CLEAN` | Message is safe and travel-relevant |
| `HATE_SPEECH_THREAT` | Violent or hateful content |
| `SEXUAL_CONTENT` | Explicit or inappropriate content |
| `PROMPT_INJECTION` | Attempt to manipulate the AI |
| `PII_DETECTED` | Personal identifiable information included |
| `TOXICITY` | Abusive language |
| `LINK_SPAM` | Contains unsolicited URLs or spam |
| `OFF_TOPIC` | Unrelated to travel (with a travel redirect suggested) |

**Four-Step Internal Process:**
1. Detect memory queries ("What do I usually prefer?")
2. Safety screening (blocks blocked categories above)
3. Travel relevance classification (considering full conversation context)
4. Generate a friendly redirect `solution` for off-topic or blocked queries

---

### Agent 4 — Explore Travel Agent

| Property | Value |
|----------|-------|
| **Model** | `gpt-4.1-mini` |
| **Output** | `global_travel_guardrail` |
| **Mode** | `param = "explore"` |

**Purpose:** Classifies an explore-mode query to determine which retrieval path to use.

**Output Schema:**

```json
{
  "isValid": true,
  "reason": "Travel query about restaurants",
  "isTravelRelated": true,
  "isPlanRelated": false,
  "travel_type": "specific-search-query"
}
```

| `travel_type` | Meaning |
|---------------|---------|
| `general-travel-query` | Broad travel topic or destination overview |
| `specific-search-query` | Concrete venue/experience search (dine, stay, play) |
| `none` | Not travel-related |

---

### Agent 5 — RAG Format Agent

| Property | Value |
|----------|-------|
| **Model** | `gpt-4o` |
| **Output** | `Output_Format` |
| **Input** | Results from RAG knowledge base |

**Purpose:** Takes raw RAG results and formats them into clean, curated travel recommendations. Blends retrieved data with base travel knowledge, groups results by city, and ranks by relevance.

**Key Output Rules:**
- No raw URLs, tables, or metadata blocks
- Groups destinations by country → city
- Closes with a planning steering question: *"Want me to build a trip itinerary around these in [city]?"*
- Destination integrity — never invents locations not in the retrieved data

---

### Agent 6 — Web Search Agent

| Property | Value |
|----------|-------|
| **Model** | `gpt-4o` |
| **Output** | `Output_Format` |
| **Tool** | `WebSearchTool` (low context) |

**Purpose:** Live web search fallback when the RAG knowledge base returns no results, or when the query is flagged as real-time (safety, weather, visa, current events, pricing).

**Real-time Query Triggers:**
- Current events or news at a destination
- Weather and natural disasters
- Entry requirements and visa rules
- Live pricing or availability
- Health and safety advisories

**Source Priority:** TripAdvisor → Yelp → other third-party sources

---

### Agent Routing Summary

```
User Message
    │
    ▼
[Guardrail Agent] ──── isMemoryQuery ──────────────────► Stream user preferences
    │
    ├── isValid = false ──────────────────────────────► Stream error + redirect
    │
    ▼
[Explore Travel Agent]
    │
    ├── isTravelRelated = false ─────────────────────► Loading starter + general response
    │
    ├── param = "plan" ──────────────────────────────► [Trip Planning Agent] → TripPlan JSON
    │
    └── param = "explore"
            │
            ├── is_realtime_query() = true ──────────► [Web Search Agent]
            │
            └── is_realtime_query() = false
                    │
                    ├── RAG returns results ─────────► [RAG Format Agent]
                    │
                    └── RAG empty ───────────────────► [Web Search Agent]
```

---

## 7. Data Flow Diagrams

### 7.1 Full Chat Request Lifecycle

```
┌──────────┐         ┌───────────┐         ┌───────────┐         ┌──────────┐
│  Client  │         │  FastAPI  │         │   Redis   │         │   Zep    │
└────┬─────┘         └─────┬─────┘         └─────┬─────┘         └────┬─────┘
     │  POST /chat         │                     │                     │
     │────────────────────►│                     │                     │
     │                     │  get_or_create_id   │                     │
     │                     │────────────────────►│                     │
     │                     │◄────────────────────│                     │
     │                     │  fetch_interactions │                     │
     │                     │────────────────────►│                     │
     │                     │◄────────────────────│                     │
     │                     │                     │   setup_session     │
     │                     │─────────────────────────────────────────►│
     │                     │◄─────────────────────────────────────────│
     │  SSE stream begins  │                     │                     │
     │◄────────────────────│                     │                     │
     │  token by token...  │                     │                     │
     │◄ ─ ─ ─ ─ ─ ─ ─ ─ ─│                     │                     │
     │                     │  save_interaction   │                     │
     │                     │────────────────────►│                     │
     │                     │   add_message       │                     │
     │                     │─────────────────────────────────────────►│
     │  SSE done           │                     │                     │
     │◄────────────────────│                     │                     │
```

### 7.2 Explore Mode — RAG Flow

```
User Query (explore)
        │
        ▼
  summarize_for_rag()         ← gpt-4.1-nano cleans & resolves pronouns
        │
        ▼
  rag(query, reference)       ← POST https://rag.hiptraveler.com/chat
        │
        ├── results found ──► [RAG Format Agent] ──► SSE stream
        │
        └── empty ──────────► [Web Search Agent] ──► SSE stream
```

### 7.3 Validator Research Pipeline

```
POST /validator/process
        │
        ▼
  Extract question
        │
        ▼
  Generate embedding (OpenAI)
        │
        ▼
  Supabase: similarity search
        │
        ├── duplicate found ──► return { duplicate: true }
        │
        └── unique
                │
                ▼
          Insert into Supabase
                │
                ▼
          Background task starts
                │
                ▼
          [Classify query]
          generic  ──► 5 sub-queries via OpenAI
          specific ──► 1 rewritten query
                │
                ▼
          Research each query (Perplexity / web)
                │
                ▼
          Score results (0/3 to 3/3)
                │
                ▼
          Store in Supabase insights table
```

---

## 8. Supporting Services

### 8.1 Redis — Short-Term Conversation Cache

Manages session-scoped conversation history with automatic expiry.

| Config | Default | Description |
|--------|---------|-------------|
| `REDIS_TTL_SECONDS` | 600 | Key expiry time |
| `REDIS_IDLE_CUTOFF_SECONDS` | 600 | Inactivity threshold before new conversation |
| `REDIS_OLD_INTERACTIONS_LIMIT` | 20 | Max history items per session |

When idle cutoff is exceeded, a new `conversation_id` is created — effectively starting a fresh session without deleting any data.

### 8.2 Zep Cloud — Long-Term Memory

Stores user-level memory across sessions. Activated only when `plan=true` in the request.

| Function | Description |
|----------|-------------|
| `setup_user_session()` | Creates or retrieves Zep user + thread |
| `add_message()` | Saves a Q&A exchange to the thread |
| `get_user_preferences()` | Returns user's travel style, activities, interests |
| `get_user_memory_for_engage()` | Returns a summary for re-engagement messaging |

### 8.3 Portal Geofencing (`region_metadata.py`)

Each portal `reference` code maps to a geographic scope. The system enforces that explore results and plan destinations are within the portal's allowed locations.

```python
REGION_METADATA = {
    "<portal_id>": {
        "country": "The Bahamas",
        "countryCd": "BS",
        "locations": ["Nassau", "Freeport", "Exuma", ...],
        "experiences": ["beach", "diving", "culture", ...]
    }
}
```

If a user asks about a destination outside the portal's scope, the system redirects them gracefully.

### 8.4 RAG Service

External service hosted at `https://rag.hiptraveler.com`. Called via HTTP by the `rag()` helper.

- **Endpoint:** `POST /chat`
- **Input:** `{ "query": "...", "reference": "..." }`
- **Output:** `{ "success": true, "data": [...] }`

Content is upserted into the RAG index via `POST /validator/rag/upsert`.

### 8.5 n8n Customer Service Webhook

A no-code automation webhook at `https://cs-automation.hiptraveler.com/webhook/customer_service` handles FAQ retrieval for the Customer Service Agent. The agent sends the user query; n8n returns a pre-built answer from the FAQ knowledge base.

---

## 9. API Reference Summary

### Main API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Welcome message and docs link |
| `GET /health` | GET | Health check — returns `{"ok": true}` |
| `POST /chat` | POST | Main streaming chat — returns SSE |
| `POST /create_user` | POST | Create a new user record in Zep |
| `GET /delete_user` | GET | Delete user and all memory (`?user_id=`) |
| `GET /memory/engage` | GET | Stream personalised re-engagement message (`?user_id=`) |

### Validator API (`/validator/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /validator/` | GET | API info |
| `GET /validator/health` | GET | Health check |
| `POST /validator/process` | POST | Queue async research on a query |
| `POST /validator/deep_validation` | POST | Synchronous full research pipeline |
| `POST /validator/rag/upsert` | POST | Upsert document to RAG index |
| `GET /validator/whitelist` | GET | List whitelisted domains |
| `POST /validator/whitelist` | POST | Add trusted domain |
| `DELETE /validator/whitelist` | DELETE | Remove domain |
| `GET /validator/whitelist/all` | GET | List all whitelist entries |
| `GET /validator/whitelist/check` | GET | Check if domain is whitelisted |
| `DELETE /validator/whitelist/all` | DELETE | Clear all whitelist entries |
| `GET /validator/saved-queries` | GET | List active research queries |
| `DELETE /validator/saved-queries` | DELETE | Clear all saved queries |
| `POST /validator/insights/sample` | POST | Insert sample insight |
| `DELETE /validator/insights/all` | DELETE | Clear all insights |

---


### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `ZEP_API_KEY` | Yes | Zep Cloud API key |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_KEY` | Yes | Supabase anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service role key |
| `REDIS_URL` | Yes | Redis connection string (Upstash) |
| `REDIS_ENABLED` | Yes | Enable Redis (`true`/`false`) |
| `GOOGLE_MAPS_API_KEY` | Optional | Google Maps integration |
| `PERPLEXITY_API_KEY` | Optional | Perplexity search API |
| `TAVILY_API_KEY` | Optional | Tavily web search API |


---

*Documentation generated for HipTraveler Experience Finder AI — v1.0*
