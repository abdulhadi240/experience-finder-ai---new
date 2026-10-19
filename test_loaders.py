"""
Loader relevance test — 2 batches of 50 questions across /chat and /chat/support.

For each question it runs the real loader pipeline (mode pick → instant loader →
LLM loaders → fallback) without the server, then checks every loader the user
could see:
  1. hard rules (regex): no travel-pick language on support questions, no claims
     of checking the user's own account/booking/payment/device, no banned words
  2. an LLM judge (gpt-4.1) that the loader is relevant to what was asked

Run:  python test_loaders.py            (all)
      python test_loaders.py -k esim    (questions containing "esim")
      python test_loaders.py --runs 3   (repeat each question, loaders are sampled)
      python test_loaders.py --batch 2  (second question set; "all" for both)
"""

import argparse
import asyncio
import json
import re
import sys

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI  # noqa: E402

from app.helpers import _pick_loader_mode  # noqa: E402
from app.services import generate_loading_statements, get_instant_loading_message  # noqa: E402

# (path, expected kind, message, prior context or None)
# kind: support = question about HipTraveler itself; travel = places/trips;
#       generic = greeting / vague, loaders must not name places or topics
CASES = [
    # ── Support questions on /chat (most real traffic) ──
    ("chat", "support", "How do loyalty points work?", None),
    ("chat", "support", "How do I cancel my Premium subscription?", None),
    ("chat", "support", "I was charged twice for my hotel booking", None),
    ("chat", "support", "my esim isn't working", None),
    ("chat", "support", "how do I reset my password", None),
    ("chat", "support", "the map won't load on my phone", None),
    ("chat", "support", "how do I delete my profile", None),
    ("chat", "support", "can I use the app offline?", None),
    ("chat", "support", "how much does premium cost", None),
    ("chat", "support", "how do I export my trip to google maps", None),
    ("chat", "support", "who do I email if something goes wrong with my reservation", None),
    ("chat", "support", "where can I see my past orders", None),
    ("chat", "support", "is there an android version?", None),
    ("chat", "support", "how do I share my itinerary with my friends", None),
    ("chat", "support", "my card got declined when paying", None),
    ("chat", "support", "how do I write a travelogue", None),
    ("chat", "support", "can I get a refund on a non-refundable hotel", None),
    ("chat", "support", "I didn't get a confirmation email", None),
    ("chat", "support", "how do I change the email on my profile", None),
    ("chat", "support", "the app keeps crashing when I open saved trips", None),
    ("chat", "support", "what is a dream trip?", None),
    ("chat", "support", "can I post my itinerary to wordpress", None),
    # ── Same kinds of questions on /chat/support ──
    ("support", "support", "How do loyalty points work?", None),
    ("support", "support", "I was charged twice for my hotel booking", None),
    ("support", "support", "my esim isn't working", None),
    ("support", "support", "how do I reset my password", None),
    ("support", "support", "the map won't load on my phone", None),
    ("support", "support", "do points expire?", None),
    ("support", "support", "can I transfer my booking to someone else", None),
    ("support", "generic", "hi", None),
    # ── Travel questions on /chat ──
    ("chat", "travel", "things to do in Lisbon", None),
    ("chat", "travel", "best street food in Bangkok", None),
    ("chat", "travel", "where should I go in Italy for a honeymoon", None),
    ("chat", "travel", "hidden beaches in Bali", None),
    ("chat", "travel", "is Kyoto worth visiting in winter?", None),
    ("chat", "travel", "family friendly hikes near Denver", None),
    ("chat", "travel", "rooftop bars in New York", None),
    ("chat", "travel", "weekend in Paris on a budget", None),
    # ── Travel questions that contain support keywords ──
    ("chat", "travel", "hotels in Rome with free cancellation", None),
    ("chat", "travel", "best booking sites for Japan ryokans", None),
    ("chat", "travel", "where can I use my points for cheap flights to Hawaii", None),
    ("chat", "travel", "do I need an esim or a local sim card in Vietnam?", None),
    ("chat", "travel", "premium hotels in Dubai", None),
    # ── Greetings / vague ──
    ("chat", "generic", "hi", None),
    ("chat", "generic", "hello there", None),
    ("chat", "generic", "help", None),
    # ── Multi-turn: support question after travel talk, and vice versa ──
    ("chat", "support", "how do I save this to my trips?",
     "User: things to do in Lisbon\nAssistant: Lisbon is great for Alfama, Belém and LX Factory."),
    ("chat", "support", "wait, how do I cancel my premium?",
     "User: best beaches in Bali\nAssistant: Try Padang Padang, Bingin and Nusa Dua."),
    ("chat", "travel", "ok what about food there?",
     "User: things to do in Lisbon\nAssistant: Lisbon is great for Alfama, Belém and LX Factory."),
    ("support", "support", "and how long does that take?",
     "User: how do I get a refund?\nAssistant: Refunds go back to your original payment method."),
]

# Second batch — different wording, typos, longer messages, trickier overlaps
CASES_2 = [
    # ── Support on /chat, varied phrasing / typos / no obvious keywords ──
    ("chat", "support", "i cant log into my acount", None),
    ("chat", "support", "why was my subscribtion renewed i didnt want that", None),
    ("chat", "support", "is hiptraveler free?", None),
    ("chat", "support", "what do I get if I upgrade", None),
    ("chat", "support", "how do I turn off notifications", None),
    ("chat", "support", "does the esim work in multiple countries", None),
    ("chat", "support", "I booked a hotel through you and the hotel says they have no record of it", None),
    ("chat", "support", "where do I find the trips I saved last week", None),
    ("chat", "support", "can I pay with paypal", None),
    ("chat", "support", "how do i add a friend to my trip", None),
    ("chat", "support", "my photos aren't uploading to my travelogue", None),
    ("chat", "support", "how do I contact a real person", None),
    ("chat", "support", "is my payment info safe with you", None),
    ("chat", "support", "the website is really slow today", None),
    ("chat", "support", "how do I download my itinerary as a pdf", None),
    ("chat", "support", "can I change the dates on a hotel I already booked", None),
    ("chat", "support", "what's the difference between free and premium", None),
    ("chat", "support", "do you have a referral program", None),
    ("chat", "support", "I want to unsubscribe from your emails", None),
    ("chat", "support", "how do i redeem my rewards", None),
    # ── /chat/support ──
    ("support", "support", "how long do refunds take", None),
    ("support", "support", "my phone number changed how do I update it", None),
    ("support", "support", "can I get a receipt for my premium payment", None),
    ("support", "support", "the esim QR code won't scan", None),
    ("support", "support", "is there a student discount", None),
    ("support", "support", "can I use HipTraveler on my ipad", None),
    ("support", "generic", "hey there!", None),
    ("support", "generic", "thanks", None),
    # ── Travel on /chat, varied ──
    ("chat", "travel", "cheapest time to fly to tokyo", None),
    ("chat", "travel", "what should i pack for iceland in march", None),
    ("chat", "travel", "is it safe to walk around mexico city at night", None),
    ("chat", "travel", "best coffee shops in melbourne", None),
    ("chat", "travel", "3 day road trip ideas from los angeles", None),
    ("chat", "travel", "where can I see the northern lights", None),
    ("chat", "travel", "vegan restaurants in berlin", None),
    ("chat", "travel", "i want somewhere warm in december with good diving", None),
    # ── Travel with support-ish words ──
    ("chat", "travel", "which airlines let you cancel flights for free", None),
    ("chat", "travel", "is the Paris museum pass worth the money", None),
    ("chat", "travel", "best hotels in Tokyo I can book last minute", None),
    ("chat", "travel", "does Uber work in Lisbon or do I need a local app", None),
    ("chat", "travel", "where's good for a premium beach resort in the Maldives", None),
    ("chat", "travel", "should I use my credit card or cash in Japan", None),
    # ── Greetings / vague ──
    ("chat", "generic", "good morning", None),
    ("chat", "generic", "hey!", None),
    ("chat", "generic", "thank you", None),
    # ── Multi-turn ──
    ("chat", "travel", "what about the weather then?",
     "User: is Kyoto worth visiting in winter?\nAssistant: Yes — fewer crowds and snow on the temples."),
    ("chat", "support", "ok and can I share it with my wife?",
     "User: how do I save a trip?\nAssistant: Tap the heart on any trip to save it to My Trips."),
    ("chat", "travel", "any good day trips from there?",
     "User: things to do in Barcelona\nAssistant: Sagrada Família, the Gothic Quarter and Barceloneta beach."),
    ("support", "support", "it still doesn't work",
     "User: my esim isn't working\nAssistant: Try restarting your phone and turning on data roaming."),
    ("support", "support", "what if I already used some of the data?",
     "User: can I get a refund on my esim?\nAssistant: Unused eSIMs can usually be refunded."),
]
BATCHES = {"1": CASES, "2": CASES_2, "all": CASES + CASES_2}

FALLBACK = "Still pulling this together — almost there…"

# Robotic status verbs as the opener ("Processing…") — "refund processing times" is fine
BANNED_RE = re.compile(r"^\W*(acknowledging|processing|computing|scanning|searching)\b", re.I)
# "Looking up how to view your orders" is a how-to, not a claim of access
HOW_TO_RE = re.compile(r"(how to|help you|steps to|ways to|guide you|on|for)\s+$", re.I)
# Claims to look at the user's own data — the support agent only reads help articles
ACCOUNT_CLAIM_RE = re.compile(
    r"\b(check\w*|verif\w*|review\w*|confirm\w*|look\w* (?:at|into)|access\w*|inspect\w*)\s+"
    r"(?:\w+\s+){0,2}your\s+(?:\w+\s+)?(account|booking|reservation|payment|transaction|charge|card|"
    r"device|phone|e-?sim|activation|order|profile|membership|subscription|points|balance|status|details|history)",
    re.I,
)
TRAVEL_PICK_RE = re.compile(
    r"\b(best picks|top picks|top options|best options|worth your time|stands out|good stuff|"
    r"hidden gems|tourist traps|must-see)\b",
    re.I,
)

JUDGE_PROMPT = """You review the loading messages a travel app shows while its AI prepares an answer.
User's newest message: {message!r}
Earlier conversation (may be empty): {context!r}
What the message is: {kind_desc}

Loading messages shown, in order:
{loaders}

A loader FAILS if any of these hold:
- it is about a different subject than the user's message (e.g. travel picks / destinations for an app or account question, or account/help-center talk for a travel question)
- it names a place or topic that is not in the conversation
- it claims to look at the user's own account, booking, payment, transaction, device or status
- it promises or implies an outcome (e.g. "your refund is on its way")
- it is so mismatched a user would find it confusing
Generic but harmless loaders ("Looking into this for you…") PASS. For a travel question, generic
travel loaders ("Pulling up the best picks for you…") also PASS.

Return ONLY JSON: {{"results": [{{"i": <index>, "pass": true|false, "why": "<short reason if fail>"}}]}}"""

KIND_DESC = {
    "support": "a question about the HipTraveler app/service itself (account, Premium, payments, refunds, eSIM, app features)",
    "travel": "a travel question about places, trips or things to do",
    "generic": "a greeting or vague message with no specific topic",
}

_client = AsyncOpenAI()


async def loaders_for(path: str, message: str, context: str | None) -> tuple[str, list[str]]:
    mode = _pick_loader_mode(message, support_only=path == "support")
    final = f"{context}\n\nUser asked: {message}" if context else message
    llm = await generate_loading_statements(final, mode)
    # mirrors _main_stream: greetings get no fallback line
    fallback = [] if mode == "neutral" else [FALLBACK]
    return mode, [get_instant_loading_message(mode), *llm, *fallback]


def hard_fails(kind: str, message: str, loaders: list[str]) -> list[str]:
    fails = []
    for i, l in enumerate(loaders):
        if BANNED_RE.search(l):
            fails.append(f"[{i}] banned word: {l}")
        claim = ACCOUNT_CLAIM_RE.search(l)
        if claim and not HOW_TO_RE.search(l[:claim.start()]):
            fails.append(f"[{i}] claims to check user's data: {l}")
        pick = TRAVEL_PICK_RE.search(l)
        # "itinerary" is fine when the user said it ("share my itinerary")
        if kind != "travel" and pick and pick.group(0).lower() not in message.lower():
            fails.append(f"[{i}] travel-pick language: {l}")
    return fails


async def judge(kind: str, message: str, context: str | None, loaders: list[str]) -> list[str]:
    listing = "\n".join(f"{i}. {l}" for i, l in enumerate(loaders))
    prompt = JUDGE_PROMPT.format(message=message, context=context or "", kind_desc=KIND_DESC[kind], loaders=listing)
    for _ in range(3):
        try:
            r = await _client.chat.completions.create(
                model="gpt-4.1", temperature=0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            res = json.loads(r.choices[0].message.content)["results"]
            return [f"[{x['i']}] judge: {loaders[x['i']]} — {x.get('why', '')}"
                    for x in res if not x.get("pass") and 0 <= x.get("i", -1) < len(loaders)]
        except Exception as e:  # retry transient API / parse errors
            err = e
    return [f"judge error: {err}"]


async def run_case(sem, case, run_no):
    path, kind, message, context = case
    async with sem:
        mode, loaders = await loaders_for(path, message, context)
        fails = hard_fails(kind, message, loaders) + await judge(kind, message, context, loaders)
    return case, run_no, mode, loaders, fails


async def main(args) -> int:
    cases = [c for c in BATCHES[args.batch] if not args.k or args.k.lower() in c[2].lower()]
    sem = asyncio.Semaphore(10)
    results = await asyncio.gather(*(run_case(sem, c, n) for c in cases for n in range(args.runs)))
    failed = 0
    for (path, kind, message, _), n, mode, loaders, fails in results:
        status = "FAIL" if fails else "PASS"
        failed += bool(fails)
        if fails or args.verbose:
            print(f"\n[{status}] /{path} {kind} [{mode}] run {n}: {message}")
            for i, l in enumerate(loaders):
                print(f"    {i}. {l}")
            for f in fails:
                print(f"  ✗ {f}")
    total = len(results)
    print(f"\n{total - failed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", help="only questions containing this text")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--batch", choices=list(BATCHES), default="1")
    ap.add_argument("-v", "--verbose", action="store_true")
    sys.exit(asyncio.run(main(ap.parse_args())))
