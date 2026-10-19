"""
Source-leak test for customer_service_agent — the user must never learn the answer
came from a document search ("the information provided does not specify…").

Runs the real streaming path (agent + file_search + leak guard) on 40 support questions,
weighted toward ones the FAQ probably doesn't cover, since that's where it leaks.
Each answer is checked by regex and by an LLM judge (gpt-4.1).

Run:  python test_support_leaks.py            (all)
      python test_support_leaks.py --runs 2   (answers vary run to run)
      python test_support_leaks.py -k paypal
"""

import argparse
import asyncio
import json
import re
import sys

from dotenv import load_dotenv

load_dotenv(".env")

from openai import AsyncOpenAI  # noqa: E402

from app import services  # noqa: E402
from app.agents_ import customer_service_agent  # noqa: E402

# (message, prior context or None)
CASES = [
    # Probably not in the FAQ — where "the documents don't say" shows up
    ("can I pay with paypal", None),
    ("is there a student discount", None),
    ("what is a dream trip?", None),
    ("do you have a referral program", None),
    ("can I pay in bitcoin", None),
    ("do you offer gift cards", None),
    ("is there a family plan for premium", None),
    ("how many people work at HipTraveler", None),
    ("can I use HipTraveler on my apple watch", None),
    ("do you have a desktop app for mac", None),
    ("is there a dark mode", None),
    ("can I book trains through you", None),
    ("do you sell travel insurance", None),
    ("can I change the app language to spanish", None),
    ("what's your phone number", None),
    ("do points transfer to airline miles", None),
    ("is HipTraveler available in China", None),
    ("can I get a refund on premium after 6 months", None),
    ("does the esim support 5g", None),
    ("can two people share one premium account", None),
    # Likely covered — should answer plainly without naming the source
    ("How do I cancel my Premium subscription?", None),
    ("How do loyalty points work?", None),
    ("I was charged twice for my hotel booking", None),
    ("my esim isn't working", None),
    ("how do I reset my password", None),
    ("how do I export my trip to google maps", None),
    ("how do I write a travelogue", None),
    ("can I post my itinerary to wordpress", None),
    ("how long do refunds take", None),
    ("is my payment info safe with you", None),
    ("what's the difference between free and premium", None),
    ("how do I delete my account", None),
    ("I didn't get a confirmation email", None),
    ("can I change the dates on a hotel I already booked", None),
    ("who is HipTraveler?", None),
    ("the app keeps crashing when I open saved trips", None),
    # Multi-turn
    ("and how long does that take?", "User: how do I get a refund?\nAssistant: Refunds go back to your original payment method."),
    ("what if I already used some of the data?", "User: can I get a refund on my esim?\nAssistant: Unused eSIMs can usually be refunded."),
    ("it still doesn't work", "User: my esim isn't working\nAssistant: Try restarting your phone and turning on data roaming."),
    ("ok so is there any way at all?", "User: can I pay with paypal\nAssistant: I don't have details on PayPal as a payment option."),
]

# Any wording that tells the user an answer came from a lookup/source
LEAK_RE = re.compile(
    r"search results?|my search|the (?:uploaded |provided |available )?(?:documents?|files?|sources?|texts?|materials?)\b"
    r"|knowledge base|\bthe faq\b|faqs? (?:provided|available)|【"
    r"|(?:information|info|details|context)\s+(?:provided|available|given|i (?:have|was given|found))"
    r"|(?:provided|available|given)\s+(?:information|info|details|context|resources)"
    r"|according to (?:the|our|my) (?:information|info|records|documents?|resources|materials|help)"
    r"|based on (?:the|what i|my) (?:information|info|available|provided|documents?|resources|materials|records)"
    r"|(?:isn'?t|is not|aren'?t|are not|wasn'?t|not)\s+(?:mentioned|covered|specified|listed|found|stated)\s+(?:in|within)\b"
    # "the Free plan does not include exports" is an answer, not a leak — only text-ish verbs
    r"|(?:do(?:es)?n'?t|do(?:es)? not)\s+(?:mention|specify|state|say)\b",
    re.I,
)

JUDGE_PROMPT = """A support assistant for the travel app HipTraveler answered a user. It looked the answer up
in an internal help-center database, and the user must NEVER be able to tell that. It should speak as
HipTraveler support, in first person ("I don't have details on that", "HipTraveler doesn't offer…").

User question: {message!r}
Answer:
<<<
{answer}
>>>

LEAK if the answer refers in any way to where its knowledge came from or to a body of text it read: e.g.
"the information provided", "the available information", "the documents/files/FAQ", "according to the info",
"based on what I have", "isn't mentioned/covered in…", "the details don't specify", citation markers.
NOT a leak: telling the user to contact support@hiptraveler.com, check the in-app Support section/help center,
or look in their own app account/settings;
saying "I don't have details on that" or "HipTraveler doesn't currently offer…".
"I don't have details on …" and "I'm not aware of …" are the CORRECT way to say it — never a leak,
even when followed by advice to contact support.

Return ONLY JSON: {{"leak": true|false, "quote": "<the leaking phrase, if any>"}}"""

_client = AsyncOpenAI()


async def answer_for(message: str, context: str | None) -> str:
    """The answer exactly as streamed to the user — through the leak guard."""
    final = (
        f"Last conversation (this is the continuation of the conversation):\n{context}\n\nNew question asked: {message}"
        if context else message
    )
    queue: asyncio.Queue = asyncio.Queue()
    await services.stream_agent_to_queue(
        "customer_service_agent", final + "\n\nReference : hiptraveler", message, "test-leaks", queue
    )
    parts = []
    while (item := queue.get_nowait()) is not None:
        if isinstance(item, Exception):
            raise item
        parts.append(item)
    text = "".join(parts)
    try:
        return json.loads(f'"{text}"')  # the stream is the JSON-escaped "answer" value
    except ValueError:
        return text


async def judge(message: str, answer: str) -> str | None:
    err = None
    for _ in range(3):
        try:
            r = await _client.chat.completions.create(
                model="gpt-4.1", temperature=0,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(message=message, answer=answer)}],
                response_format={"type": "json_object"},
            )
            v = json.loads(r.choices[0].message.content)
            return f"judge: {v.get('quote', '')!r}" if v.get("leak") else None
        except Exception as e:  # retry transient API / parse errors
            err = e
    return f"judge error: {err}"


async def run_case(sem, case, run_no):
    message, context = case
    async with sem:
        try:
            answer = await answer_for(message, context)
        except Exception as e:
            return case, run_no, "", [f"agent error: {e}"]
        fails = []
        m = LEAK_RE.search(answer)
        if m:
            fails.append(f"regex: {m.group(0)!r}")
        j = await judge(message, answer)
        if j:
            fails.append(j)
    return case, run_no, answer, fails


async def main(args) -> int:
    cases = [c for c in CASES if not args.k or args.k.lower() in c[0].lower()]
    sem = asyncio.Semaphore(8)
    results = await asyncio.gather(*(run_case(sem, c, n) for c in cases for n in range(args.runs)))
    failed = 0
    for (message, _), n, answer, fails in results:
        failed += bool(fails)
        if fails or args.verbose:
            print(f"\n[{'FAIL' if fails else 'PASS'}] run {n}: {message}")
            print("    " + answer.replace("\n", "\n    "))
            for f in fails:
                print(f"  ✗ {f}")
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", help="only questions containing this text")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--model", help="try the agent on another model, e.g. gpt-4.1")
    ap.add_argument("--no-guard", action="store_true", help="skip the stream leak guard (agent alone)")
    a = ap.parse_args()
    if a.model:
        services.customer_service_agent = customer_service_agent.clone(model=a.model)
    if a.no_guard:
        services._SupportLeakGuard = lambda queue: None  # stream_agent_to_queue then queues directly
    sys.exit(asyncio.run(main(a)))
