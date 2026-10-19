# app/services.py
import json
import re
import time
import asyncio
import random
from openai import AsyncOpenAI
from agents import Runner
from openai.types.responses import ResponseTextDeltaEvent
from .agents_ import trip_planning_agent, rag_format_agent, web_search_agent, customer_service_agent
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
                        "You are a travel query processor. The input may contain prior conversation turns; "
                        "the user's LATEST message is the one to act on.\n"
                        "Follow these steps in order:\n"
                        "1. If the message is not in English, translate it to English first.\n"
                        "2. Build a concise standalone search query for what the user wants in their LATEST message. "
                        "Use earlier turns ONLY to resolve references (pronouns like 'there'/'it', or a destination just named). "
                        "Do NOT search for an older topic the user has already moved past (e.g. a past comparison). "
                        "If the latest message just narrows to one place from a list the assistant offered (e.g. 'california' → its cities), "
                        "make the query about THAT place.\n"
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


# ─── RAG Payload Builder (full structured payload) ───────────────

# Valid retrieval categories for the /chat endpoint.
_RAG_CATEGORIES = {"places", "place", "tour", "activity", "restaurant", "hotel"}
_RAG_TOPK_DEFAULT = 8
_RAG_TOPK_MAX = 50


async def build_rag_payload(
    message: str,
    fallback_location: str = "",
    fallback_filters: str = "",
) -> dict:
    """
    Ultra-fast gpt-4.1-nano call that parses the user's LATEST message (which may
    contain conversation history / pronouns) into a complete, structured RAG payload.

    Returns a dict with: query, category, top_k, location, filters.

    - query    : clean standalone English search query.
    - category : one of place/tour/activity/restaurant/hotel; 'places' for a
                 general "things to do" request.
    - top_k    : how many results to retrieve. Default 10; raised only when the
                 user explicitly asks for more (e.g. "give me 20").
    - location : destination the search should be scoped to (extracted from the
                 message), else the caller-provided fallback.
    - filters  : audience/style keywords (e.g. family, vegan, budget, luxury),
                 else the caller-provided fallback.

    This REPLACES the older summarize_for_rag nano call — it is the same single
    nano round-trip, just returning a richer object, so it adds zero latency.
    On any failure it degrades gracefully to a query-only payload.
    """
    fallback = {
        "query": message.strip(),
        "category": "places",
        "top_k": _RAG_TOPK_DEFAULT,
        "location": fallback_location or "",
        "filters": fallback_filters or "",
    }
    try:
        response = await _openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You build a search payload for a travel RAG engine. The input may contain prior "
                        "conversation turns; act ONLY on the user's LATEST message, using earlier turns just "
                        "to resolve references (pronouns like 'there'/'it', or a place just named/selected).\n\n"
                        "Return ONLY valid JSON with EXACTLY these keys:\n"
                        '{"query": str, "category": str, "top_k": int, "location": str, "filters": str}\n\n'
                        "Rules:\n"
                        "- query: a concise standalone English search query for what the user wants NOW. "
                        "Translate to English if needed. Silently fix misspelled place names (e.g. 'Karahic'→'Karachi'). "
                        "Do NOT search an older topic the user has moved past. No trailing punctuation.\n"
                        "- category: the single best retrieval type. Use 'restaurant' for food/dining, 'hotel' for "
                        "stays/accommodation, 'tour' for guided tours/day trips, 'activity' for things to do/experiences "
                        "explicitly framed as activities, and 'places' for a general 'best things to do / what to see' "
                        "request. When unsure, use 'places'.\n"
                        f"- top_k: how many results to fetch. Default {_RAG_TOPK_DEFAULT}. ONLY change it if the user "
                        "explicitly states a count (e.g. 'show me 20 spots' → 20, 'just give me 3' → 3). "
                        f"Never exceed {_RAG_TOPK_MAX}.\n"
                        "- location: the destination the search should be scoped to, taken from the message "
                        "(e.g. 'best things to do in Paris' → 'Paris'). If the message names several destinations, "
                        "list ALL of them comma-separated, never their shared country or region "
                        "(e.g. 'a trip to Barcelona and Madrid' → 'Barcelona, Madrid', NOT 'Spain'). "
                        "Empty string if no destination is mentioned.\n"
                        "- filters: short audience/style keywords the user expressed (e.g. 'family', 'vegan', "
                        "'budget', 'luxury', 'romantic'), comma-separated. Empty string if none.\n"
                    ),
                },
                {"role": "user", "content": message},
            ],
            max_tokens=120,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)

        # ── sanitize each field, falling back where the model misbehaved ──
        query = str(data.get("query") or "").strip() or fallback["query"]

        category = str(data.get("category") or "").strip().lower()
        if category not in _RAG_CATEGORIES:
            category = "places"

        try:
            top_k = int(data.get("top_k") or _RAG_TOPK_DEFAULT)
        except (TypeError, ValueError):
            top_k = _RAG_TOPK_DEFAULT
        top_k = max(1, min(top_k, _RAG_TOPK_MAX))

        location = str(data.get("location") or "").strip() or (fallback_location or "")
        filters = str(data.get("filters") or "").strip() or (fallback_filters or "")

        return {
            "query": query,
            "category": category,
            "top_k": top_k,
            "location": location,
            "filters": filters,
        }
    except Exception:
        return fallback


# ─── LLM-Generated Loading Statements ───────────────────────────

# ─── Support answer leak guard ───────────────────────────────────
# customer_service_agent answers from file_search, and despite its prompt it
# sometimes says "the FAQ doesn't specify…" or pastes a citation that names an
# internal file. Streamed text is held one sentence at a time; a sentence that
# refers to its source is rewritten by nano before the user sees it.

_CITATION_RE = re.compile(r'\s*(?:【[^】]*】|[^]*|\\ue200.*?\\ue201)')
_OPEN_CITATION_RE = re.compile(r'\s*(?:【[^】]*|[^]*|\\ue200.*)$')
_SOURCE_LEAK_RE = re.compile(
    r"\bFAQs?\b|\bthe (?:provided |available |uploaded )?(?:documents?|files?|materials?)\b|\bdocumentation\b"
    r"|knowledge base|search results?"
    r"|\b(?:information|info|details|context)\s+(?:provided|available|given|here|"
    r"I (?:have|can see|can access|have access to|found))"
    r"|\b(?:provided|available)\s+(?:\w+\s+)?(?:information|info|details|resources)"
    r"|\b(?:described|defined)\s+(?:for|in|as|here)\b|\b(?:is|are)(?:n'?t| not)\s+(?:\w+\s+)?defined\b"
    r"|\bno (?:specific |explicit |direct )?(?:mention|information|info|details)\s+(?:of|about|on|regarding|available)"
    r"|\b(?:is|are|was|were)(?:n'?t| not)\s+(?:specifically |explicitly |directly |clearly |currently )?"
    r"(?:mentioned|covered|specified|outlined|detailed|listed|stated|documented|addressed)"
    r"|\bdo(?:es)?(?:n'?t| not)\s+(?:specifically |explicitly |directly |clearly |currently |really )?"
    r"(?:mention|specify|state|say|detail|outline|address|reference)\b"
    # "…has no referral program mentioned." / "not listed anywhere" / "no specific mention that…"
    r"|\b(?:mentioned|specified|listed|outlined|documented)(?=\s*[.,;]|\s+(?:here|anywhere)\b)"
    r"|\bno (?:\w+ ){0,2}(?:mention|indication)\b|\b(?:option|program|feature|policy)\s+mentioned\b"
    r"|\bwithout (?:any )?(?:specific |explicit )?mention\b|\b(?:is|are) described\b"
    r"|\bdo(?:es)?(?:n'?t| not)\s+list\b"
    # "details … aren't provided / described here" — the subject is the missing text itself
    r"|\b(?:details?|information|info|specifics|instructions)\b[^.]{0,60}?\b(?:not|aren'?t|isn'?t)\s+"
    r"(?:\w+\s+)?(?:provided|described|given|explained|available)\b"
    r"|\b(?:described|explained|stated|outlined|covered)\s+here\b"
    r"|\b(?:explicit|specific)\s+(?:instructions|details|information|mention)\s+(?:confirming|regarding)\b"
    r"|\b(?:information|anything|nothing)\s+(?:indicating|suggesting|showing|that (?:says|shows|indicates))"
    r"|\bfrom what I (?:have|can see|found)|\bbased on what I (?:have|found|can see)"
    r"|\bI (?:don'?t|do not|can'?t|cannot) see\b|\bI'?m not seeing\b|\bI (?:couldn'?t|could not) find\b"
    r"|\baccording to (?:the|my|our) (?:info|information|records|resources|help)",
    re.IGNORECASE,
)
# Sentence ends inside JSON-escaped stream text: ". " / "? " / "! " or an escaped newline
_SENTENCE_END_RE = re.compile(r'((?<=[.!?])\s+|\\n)')


_REWRITE_PROMPT = (
    "Rewrite this one sentence from HipTraveler's customer support chat. It must not refer to any source: "
    "no FAQ, documents, files, materials, 'information provided/available', nothing being "
    "'mentioned / listed / specified / covered / detailed', and no 'I don't see / couldn't find'.\n"
    "Speak in first person as HipTraveler support, about the thing itself.\n"
    "IMPORTANT: 'X isn't mentioned / specified' means support does NOT KNOW about X. Keep that uncertainty — "
    "say 'I don't have details on X' or 'I'm not aware of X'. Never turn it into a definite "
    "'HipTraveler doesn't offer X'.\n"
    "Examples:\n"
    "- 'Refund policies after six months are not specified.' → 'I don't have details on refunds after six months.'\n"
    "- 'HipTraveler does not currently mention a student discount.' → 'I'm not aware of a student discount right now.'\n"
    "- 'The FAQ doesn't say whether the eSIM supports 5G.' → 'I don't have details on 5G support for the eSIM.'\n"
    "Keep every other fact, any markdown, and roughly the length. If the sentence states a real fact or "
    "policy and does not talk about what some text says (e.g. 'Refunds are not provided for used eSIMs.', "
    "'Mention your booking ID when you contact support.'), return it EXACTLY unchanged.\n"
    "Return ONLY the sentence.\n\n"
    "Sentence: "
)


async def _rewrite_leaky_sentence(sentence: str) -> str:
    """Rewrite one JSON-escaped sentence so it doesn't mention its source."""
    try:
        plain = json.loads(f'"{sentence}"')
    except ValueError:
        plain = sentence
    try:
        rewritten = ""
        # nano first; mini only if nano's version still leaks (rare, so cost stays tiny)
        for model in ("gpt-4.1-nano", "gpt-4.1-mini"):
            response = await asyncio.wait_for(_openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _REWRITE_PROMPT + plain}],
                max_tokens=200,
                temperature=0,
            ), timeout=4)
            rewritten = response.choices[0].message.content.strip()
            if rewritten and not _SOURCE_LEAK_RE.search(rewritten):
                break
        if not rewritten:
            return sentence
        # back into the stream's JSON-escaped form, keeping the original edge whitespace
        lead = sentence[:len(sentence) - len(sentence.lstrip())]
        return lead + json.dumps(rewritten, ensure_ascii=False)[1:-1]
    except Exception:
        return sentence  # never break the answer over the guard


class _SupportLeakGuard:
    """Holds streamed support text until a sentence ends, cleans it, then queues it."""

    def __init__(self, queue: asyncio.Queue):
        self._queue = queue
        self._buf = ""

    async def feed(self, text: str) -> None:
        self._buf += text
        last = None
        for last in _SENTENCE_END_RE.finditer(self._buf):
            pass
        # nothing complete yet — but never hold a runaway sentence forever
        if last is None and len(self._buf) < 600:
            return
        cut = last.end() if last is not None else len(self._buf)
        # an unclosed citation stays buffered until it closes
        open_cite = _OPEN_CITATION_RE.search(self._buf[:cut])
        if open_cite:
            cut = open_cite.start()
        done, self._buf = self._buf[:cut], self._buf[cut:]
        await self._emit(done)

    async def flush(self) -> None:
        done, self._buf = _OPEN_CITATION_RE.sub('', self._buf), ""
        await self._emit(done)

    async def _emit(self, text: str) -> None:
        text = _CITATION_RE.sub('', text)
        if not text:
            return
        parts = _SENTENCE_END_RE.split(text)  # sentences interleaved with their separators
        for i in range(0, len(parts), 2):
            if _SOURCE_LEAK_RE.search(parts[i]):
                parts[i] = await _rewrite_leaky_sentence(parts[i])
        await self._queue.put(''.join(parts))

_ROBOTIC_OPENER_RE = re.compile(r"^\W*(acknowledging|processing|computing|scanning|searching)\b", re.I)
# "Checking why your subscription renewed…", "verifying if your eSIM works…"
_ACCOUNT_CLAIM_RE = re.compile(
    r"\b(check|verif|review|confirm|access|inspect|locat|retriev|fetch)\w*\s+(?:\w+\s+){0,2}your\b", re.I
)

_NEUTRAL_LOADERS = [
    "Getting things ready for you…",
    "One sec, I'm right here…",
    "Almost ready…",
]


async def generate_loading_statements(message: str, param: str) -> list:
    """
    Call gpt-4.1-nano to produce 5 personalised loading statements for this specific query.
    Runs as a concurrent asyncio.Task — adds zero wall-clock latency to the pipeline.

    Returns a list of up to 5 strings.
    Falls back to [] on any failure; caller uses the static instant message + a
    deep-research fallback when these run out.
    """
    if param == "neutral":
        # Greetings — nano invents "top picks for your destination" given any room
        return list(_NEUTRAL_LOADERS)
    if param == "support":
        return await _generate_support_loading_statements(message)
    if param == "plan":
        return await _generate_travel_loading_statements(message, "trip planner")

    # /chat: the router only decides support vs travel after validation, and no
    # keyword list catches every support question ("can I pay with paypal").
    # Classify and write both loader sets in parallel — same wall-clock as one call.
    kind, travel, support = await asyncio.gather(
        _classify_loader_topic(message),
        _generate_travel_loading_statements(message, "explore"),
        _generate_support_loading_statements(message, maybe_travel=True),
        return_exceptions=True,
    )
    if isinstance(kind, Exception) or kind is None:
        # classifier failed — fall back to the keyword hint from the caller
        kind = "support" if param == "unsure" else "travel"
    picked = support if kind == "support" else travel
    return [] if isinstance(picked, Exception) else picked


async def _classify_loader_topic(message: str) -> str | None:
    """'support' if the newest message is about HipTraveler itself, else 'travel'."""
    prompt = (
        f"Conversation with HipTraveler's AI (newest message is last):\n{message}\n\n"
        "Is the NEWEST message about HipTraveler itself — its app or website, the user's account, login, "
        "Premium/subscription, pricing of HipTraveler, payments or bookings made through HipTraveler, refunds, "
        "eSIMs sold by HipTraveler, loyalty points/rewards, saving/sharing/exporting trips, travelogues, "
        "notifications, emails, bugs, or contacting support? Also any question about what a HipTraveler "
        "feature is or does — Dream Trips, Travelogues, My Trips, AI Assist, Concierge, travel blogs. "
        "Answer 'support'.\n"
        "Examples: 'what is a dream trip?' → support; 'plan my dream trip to Bali' → travel; "
        "'how does premium work' → support; 'premium hotels in Dubai' → travel; "
        "'does the esim work in Japan' → support; 'esim or local sim in Vietnam?' → travel.\n"
        "Is it about travel itself — places, things to do, food, hotels to pick, flights, weather, safety, "
        "packing, money abroad, or anything else? Answer 'travel'.\n"
        "Reply with exactly one word: support or travel."
    )
    response = await _openai_client.chat.completions.create(
        model="gpt-4.1-mini",  # nano misreads "what is a dream trip?" as travel
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3,
        temperature=0,
    )
    answer = response.choices[0].message.content.strip().lower()
    return "support" if answer.startswith("support") else "travel" if answer.startswith("travel") else None


async def _generate_travel_loading_statements(message: str, mode: str) -> list:

    prompt = (
        f"Travel AI conversation context (may include prior turns; newest message is last):\n{message}\n\n"
        f"Mode: {mode}\n\n"
        "Write exactly 5 short loading messages (6–10 words each) to display while the AI finds the answer.\n"
        "Rules:\n"
        "- FIRST infer what the user actually wants RIGHT NOW from the conversation. If earlier turns "
        "established a destination or focus (e.g. they were exploring Pakistan and just said 'islamabad' "
        "or 'yes'), the loaders MUST be about THAT destination/focus — never a different place.\n"
        "- ⚠️ NEVER invent or assume a destination. Only name a place if it ACTUALLY appears in the conversation above. "
        "If the message is a greeting ('hi', 'hello', 'hey'), small talk, a vague question, or has NO destination "
        "anywhere in the context, the loaders MUST be completely generic with NO place names at all "
        "(e.g. 'Getting things ready for you…', 'One sec while I pull this together…'). "
        "Mentioning a destination that isn't in the conversation is a critical failure.\n"
        "- For a greeting or vague message there is nothing to recommend yet, so also no 'options', 'picks', "
        "'choices', 'recommendations', 'destination' or 'travel info' — just warm, topic-free waiting lines.\n"
        "- Only when a real destination/topic is present, reference it by name (e.g. 'Pulling up the best of Islamabad…').\n"
        "- Sound like a knowledgeable travel friend thinking out loud, not a chatbot.\n"
        "- Each message should feel like progress: looking → finding → comparing → narrowing → almost ready.\n"
        "- NEVER use: 'acknowledging', 'processing', 'computing', 'scanning', 'searching', or stage labels.\n"
        "- NEVER imply you are building/creating an itinerary or trip plan. Do NOT say 'building your itinerary', "
        "'putting your trip together', 'planning your trip', 'mapping out your trip', or anything similar. "
        "These loaders are for FINDING and GATHERING recommendations only — frame them as pulling up / rounding up / "
        "narrowing down the best options.\n"
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
            # nano ignores the banned-opener rule now and then — drop those lines
            return [str(s) for s in stmts[:5] if not _ROBOTIC_OPENER_RE.match(str(s))]
    except Exception:
        pass

    return []


async def _generate_support_loading_statements(message: str, maybe_travel: bool = False) -> list:
    """Loaders for HipTraveler support questions — about the help topic, never travel picks.

    maybe_travel=True is /chat, where a support keyword ("cancellation", "booking")
    can still belong to a travel question; the model falls back to generic loaders then.
    """
    travel_rule = (
        "- If the newest message is actually a TRAVEL question (places, hotels to stay at, things to do, "
        "e.g. 'hotels in Rome with free cancellation'), ignore the support framing and write short generic "
        "loaders with no place names (e.g. 'Pulling the good options together…').\n"
        if maybe_travel else ""
    )
    prompt = (
        f"HipTraveler customer support conversation (may include prior turns; newest message is last):\n{message}\n\n"
        "Write exactly 5 short loading messages (5–9 words each) to display while the support assistant "
        "looks up the answer in HipTraveler's help center.\n"
        "Rules:\n"
        "- Name the specific topic of the NEWEST question (e.g. Premium cancellation, hotel refunds, "
        "loyalty points, eSIM setup, login problems — these are only examples, never mention one the user "
        "didn't ask about). If the question is vague or just a greeting, keep them topic-free "
        "(e.g. 'Getting the right answer for you…', 'One sec, I'm here to help…').\n"
        f"{travel_rule}"
        "- The assistant only reads help articles. It can NOT see the user's account, bookings, payments, "
        "transactions, device, or eSIM/activation status — so never say it is checking, verifying, reviewing, "
        "or confirming anything of theirs (bad: 'Checking your booking details…', 'Verifying recent transactions…', "
        "'Confirming device compatibility…', 'Locating details on your past orders…'). Frame every loader around "
        "the help topic or 'how to' — good: 'Looking up how double charges get refunded…', "
        "'Finding how to view past orders…'.\n"
        "- Stick to exactly what the user described — don't add details (they said 'crashing', not 'lost').\n"
        "- Never invent policies, timelines, or outcomes — the loaders only say what is being looked up.\n"
        "- No destinations, places, travel recommendations, or 'best options/picks' language.\n"
        "- Each message should feel like progress: finding the help topic → reading the details → "
        "pulling the steps together → almost ready. The last one is just 'Almost ready with your answer…' "
        "or similar — never 'finalizing' a refund, cancellation, transfer or anything else.\n"
        "- Write naturally; never repeat a word back to back.\n"
        "- NEVER use: 'processing', 'computing', 'scanning', 'searching', or stage labels.\n"
        "- Tone: warm, calm, helpful.\n"
        'Return ONLY valid JSON: {"messages": ["…", "…", "…", "…", "…"]}'
    )
    try:
        response = await _openai_client.chat.completions.create(
            model="gpt-4.1-mini",  # nano slips into "checking your account…" wording
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        stmts = json.loads(response.choices[0].message.content.strip()).get("messages", [])
        if isinstance(stmts, list) and len(stmts) >= 1:
            # The agent can't see their account — drop the odd "Checking your…" nano still writes
            return [str(s) for s in stmts[:5]
                    if not _ROBOTIC_OPENER_RE.match(str(s)) and not _ACCOUNT_CLAIM_RE.search(str(s))]
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
    'support': [
        "Checking our help center…",
        "Looking that up for you…",
        "Getting the right answer for you…",
        "Pulling up the details on this…",
    ],
    # /chat message that isn't clearly travel — may be support, so fits both
    'neutral': [
        "Looking into this for you…",
        "One sec — getting you the right answer…",
        "Getting the details on this…",
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
    if param == 'unsure':
        param = 'neutral'
    mode = param if param in ('plan', 'support', 'neutral') else 'explore'
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
        elif agent_name == 'customer_service_agent':
            # Support answers name no places — nothing for the validator to research.
            result = Runner.run_streamed(customer_service_agent, final_message_with_ref)
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
        # Support answers come from file_search — never let the user see that
        leak_guard = _SupportLeakGuard(queue) if agent_name == 'customer_service_agent' else None

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
                        leaked = prefix_buf.lstrip("﻿")
                        prefix_buf = ""
                        # The exact wrapper wasn't found. If the buffer still
                        # opens with a JSON envelope, strip the opener so a stray
                        # "{" / '"answer":' can't leak into the rendered text.
                        if leaked.lstrip().startswith("{"):
                            leaked = re.sub(
                                r'^\s*\{\s*("answer"\s*:\s*"?)?', '', leaked, count=1
                            )
                        chunk = leaked
                        if not chunk:
                            continue
                    else:
                        continue

                # ── Rolling suffix buffer to strip trailing "} or "\n} ──
                pending = suffix_buf + chunk
                if len(pending) > _SUFFIX_LEN:
                    emit = pending[:-_SUFFIX_LEN]
                    emit = re.sub(r'\s*Source:\s*\S+', '', emit)
                    if emit:
                        if leak_guard:
                            await leak_guard.feed(emit)
                        else:
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
                if leak_guard:
                    await leak_guard.feed(cleaned)
                else:
                    await queue.put(cleaned)
                if answer_parts is not None:
                    answer_parts.append(cleaned)
        if leak_guard:
            await leak_guard.flush()

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
    
    
    
