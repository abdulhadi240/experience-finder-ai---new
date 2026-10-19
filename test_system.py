"""
End-to-end system tests against a running server — 5 cases per flow:

  Explore           POST /chat          place recommendations, safety, off-topic block, follow-up turn
  Travelling        POST /chat          trip-planning JSON (validation route + plan=True fast path)
  Customer care     POST /chat/support  FAQ answers from the vector store; never travel agents
                    POST /chat          support questions still route to the customer service agent

Run the server first, then:
  python test_system.py                 # all flows
  python test_system.py support         # one flow: explore | travel | support
  BASE_URL=http://localhost:8765 python test_system.py

These hit the real agents, so assertions check structure and routing, not exact wording.
Run with CREDITS_ENFORCE=false on the server, or every case burns a real credit.
"""

import json
import os
import re
import sys
import time

import requests

BASE = os.getenv("BASE_URL", "http://localhost:8080").rstrip("/")
HDRS = {"Content-Type": "application/json", "Accept": "text/event-stream"}

SUPPORT_EMAIL_RE = re.compile(r"support@hiptraveler\.com", re.I)
POIS_RE          = re.compile(r"<POIS>", re.I)
# The FAQ comes from a vector store — the user must never hear about it.
RETRIEVAL_LEAK_RE = re.compile(
    r"search results|the (?:uploaded )?(?:documents?|files?)\b|knowledge base|the FAQ|【", re.I
)


def send(message: str, path: str = "/chat", thread_id: str | None = None,
         plan: bool = False, reference: str = "hiptraveler") -> dict:
    payload = {
        "message":   message,
        "user_id":   "test_user_system",
        "user_type": "logged-in",
        "reference": reference,
        "param":     "explore",
        "threadId":  thread_id,
        "plan":      plan,
    }
    r = {"thread": thread_id, "text": "", "answer": "", "travel": None, "loaders": [],
         "blocked": False, "reason": None, "error": None, "elapsed": 0.0}
    t0 = time.time()
    with requests.post(BASE + path, json=payload, headers=HDRS, stream=True, timeout=240) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            ls = line.decode("utf-8")
            if not ls.startswith("data: "):
                continue
            try:
                d = json.loads(ls[6:])
            except json.JSONDecodeError:
                continue
            if d.get("status") == "started":
                r["thread"] = d.get("threadId", thread_id)
            elif "loading" in d:
                r["loaders"].append(d["loading"])
            elif "content" in d:
                r["text"] += d["content"]
            elif "travel" in d:
                r["travel"] = d["travel"][0]
            elif "error" in d:
                r["error"] = d["error"]
            if d.get("done"):
                r["blocked"] = bool(d.get("blocked"))
                r["reason"]  = d.get("reason")
    r["elapsed"] = time.time() - t0

    raw = r["text"].strip()
    answer = raw
    if raw.startswith("{"):
        try:
            answer = json.loads(raw).get("answer", raw)
        except json.JSONDecodeError:
            m = re.match(r'^\{"answer"\s*:\s*"(.*?)"?\}?$', raw, re.S)
            if m:
                answer = m.group(1)
    r["answer"] = answer.replace("\\n", "\n").replace('\\"', '"')
    return r


def prose(answer: str) -> str:
    """Answer without the POI block and links, so '?' in URLs can't fake a closing question."""
    a = re.sub(r"<POIS>.*?</POIS>", " ", answer, flags=re.S | re.I)
    a = re.sub(r"<link>.*?</link>|https?://\S+|www\.\S+", " ", a, flags=re.S | re.I)
    return a.strip().rstrip('"}').strip()


def ends_with_question(answer: str) -> bool:
    return prose(answer).endswith("?")


# ─── Check helpers — each returns a list of failure strings ───────

def expect(cond: bool, msg: str) -> list[str]:
    return [] if cond else [msg]


def common(r: dict) -> list[str]:
    return expect(r["error"] is None, f"stream error: {r['error']}")


def is_explore_answer(r: dict) -> list[str]:
    return (common(r)
            + expect(r["travel"] is None, "routed to trip planning, expected an explore answer")
            + expect(not r["blocked"], f"blocked ({r['reason']})")
            + expect(len(r["answer"]) > 80, "answer is empty or too short"))


def is_trip_plan(r: dict) -> list[str]:
    return (common(r)
            + expect(r["travel"] is not None, "no trip-planning payload ('travel' event)")
            + expect(bool(r["travel"] and r["travel"].get("destinations")), "trip plan has no destinations"))


def is_support_answer(r: dict) -> list[str]:
    return (common(r)
            + expect(r["travel"] is None, "support question went to trip planning")
            + expect(not r["blocked"], f"blocked ({r['reason']})")
            + expect(not POIS_RE.search(r["answer"]), "support answer contains a <POIS> block")
            + expect(not RETRIEVAL_LEAK_RE.search(r["answer"]),
                     f"answer mentions the retrieval: {RETRIEVAL_LEAK_RE.search(r['answer'])}")
            + expect(len(r["answer"]) > 40, "answer is empty or too short"))


def mentions(r: dict, *words: str) -> list[str]:
    hit = any(re.search(w, r["answer"], re.I) for w in words)
    return expect(hit, f"answer mentions none of {words}")


def dests(r: dict) -> list[str]:
    return (r["travel"] or {}).get("destinations") or []


# ─── Explore ──────────────────────────────────────────────────────

def explore_1():
    """City recommendations → POI block + closing question."""
    r = send("best things to do in Barcelona")
    return r, (is_explore_answer(r)
               + expect(bool(POIS_RE.search(r["answer"])), "no <POIS> block")
               + expect(ends_with_question(r["answer"]), "does not close with a question"))


def explore_2():
    """Food query → POIs about the asked-for place."""
    r = send("where can I find the best ramen in Tokyo?")
    return r, (is_explore_answer(r)
               + expect(bool(POIS_RE.search(r["answer"])), "no <POIS> block")
               + mentions(r, "ramen"))


def explore_3():
    """Destination safety is travel, not customer service."""
    r = send("is it safe to travel to Colombia right now?")
    return r, (is_explore_answer(r)
               + mentions(r, "Colombia")
               + expect(not SUPPORT_EMAIL_RE.search(r["answer"]), "answered like customer support"))


def explore_4():
    """Off-topic request is blocked by validation."""
    r = send("write me a poem about my cat")
    return r, (common(r)
               + expect(r["blocked"], "off-topic request was not blocked")
               + expect(r["travel"] is None, "off-topic request went to trip planning"))


def explore_5():
    """Follow-up turn: answering the mindset question → advice, no new POI list."""
    t1 = send("what should I see in Kyoto?")
    fails = is_explore_answer(t1)
    if fails:
        return t1, ["T1: " + f for f in fails]
    t2 = send("I love quiet temples, gardens and slow mornings — not big crowds",
              thread_id=t1["thread"])
    return t2, (common(t2)
                + expect(t2["travel"] is None, "T2 jumped to trip planning")
                + expect(not POIS_RE.search(t2["answer"]), "T2 repeated a <POIS> block")
                + expect(ends_with_question(t2["answer"]), "T2 does not close with the planning question"))


# ─── Travelling (trip planning) ───────────────────────────────────

def travel_1():
    """Full trip request routes to the planner with dates and travellers."""
    r = send("Plan a trip to Rome from June 10 to June 14 2027 for 2 adults")
    t = r["travel"] or {}
    return r, (is_trip_plan(r)
               + expect(any("Rome" in d for d in dests(r)), f"Rome not in destinations {dests(r)}")
               + expect(bool(t.get("startDate")), "startDate not extracted")
               + expect(bool(t.get("pax")), "pax not extracted"))


def travel_2():
    """plan=True fast path skips validation and returns the plan directly."""
    r = send("3 days in Paris in July", plan=True)
    return r, (is_trip_plan(r)
               + expect(any("Paris" in d for d in dests(r)), f"Paris not in destinations {dests(r)}"))


def travel_3():
    """Multi-city trip keeps one destination entry per city — never consolidated to the country."""
    r = send("Plan a 7-day trip to Tokyo and Kyoto")
    d = dests(r)
    return r, (is_trip_plan(r)
               + expect(any("Tokyo" in x for x in d) and any("Kyoto" in x for x in d),
                        f"expected Tokyo and Kyoto as separate entries, got {d}")
               + expect(d != ["Japan"], "consolidated cities into the country"))


def travel_4():
    """US city comes back in address form: 'City, ST, United States'."""
    r = send("Plan a weekend trip to Austin, Texas")
    d = dests(r)
    return r, (is_trip_plan(r)
               + expect(any(re.match(r"Austin, TX, United States", x) for x in d),
                        f"expected 'Austin, TX, United States', got {d}"))


def travel_5():
    """Statement of interest explores first and offers an itinerary; 'yes' then builds it.

    By design a first-message "I want to plan a trip to X" is NOT a build request
    (validation_agent rules in app/agents_.py — "FIRST-MESSAGE statement of interest").
    """
    t1 = send("I want to plan a trip to Lisbon")
    fails = (is_explore_answer(t1)
             + expect(ends_with_question(t1["answer"]), "T1 does not close with the planning offer"))
    if fails:
        return t1, ["T1: " + f for f in fails]
    t2 = send("yes, build the itinerary for Lisbon", thread_id=t1["thread"])
    t = t2["travel"] or {}
    return t2, (is_trip_plan(t2)
                + expect(any("Lisbon" in d for d in dests(t2)), f"Lisbon not in destinations {dests(t2)}")
                + expect(bool(t.get("feedback")), "no feedback fields requested for a trip with no dates"))


# ─── Customer care ────────────────────────────────────────────────

def support_1():
    """Premium cancellation answered from the FAQ vector store."""
    r = send("How do I cancel my Premium membership?", path="/chat/support")
    return r, (is_support_answer(r)
               + mentions(r, "subscription", "Premium", "membership"))


def support_2():
    """Hotel refund — grounded in the booking/cancellation FAQ, no invented promises."""
    r = send("Can I get a refund if I cancel my hotel booking?", path="/chat/support")
    return r, (is_support_answer(r)
               + mentions(r, "cancellation polic", "refund"))


def support_3():
    """A trip request on /chat/support never reaches the travel agents."""
    r = send("Plan a 5-day trip to Rome for me", path="/chat/support")
    return r, (common(r)
               + expect(r["travel"] is None, "/chat/support produced a trip plan")
               + expect(not POIS_RE.search(r["answer"]), "/chat/support returned a <POIS> block"))


def support_4():
    """Off-topic on /chat/support gets the support block message, not the travel one."""
    r = send("write me a poem about my cat", path="/chat/support")
    return r, (common(r)
               + expect(r["blocked"], "off-topic request was not blocked")
               + mentions(r, "support assistant", "HipTraveler")
               + expect(not re.search(r"travel-focused|I specialize in travel", r["answer"], re.I),
                        "got the travel block message"))


def support_5():
    """Support question on the main /chat still routes to the customer service agent."""
    r = send("How do loyalty points work?")
    return r, (is_support_answer(r)
               + mentions(r, "point"))


SUITES = {
    "explore": [explore_1, explore_2, explore_3, explore_4, explore_5],
    "travel":  [travel_1, travel_2, travel_3, travel_4, travel_5],
    "support": [support_1, support_2, support_3, support_4, support_5],
}


def run(names: list[str]) -> int:
    results = []
    for name in names:
        print(f"\n{'=' * 70}\n  {name.upper()}\n{'=' * 70}")
        for case in SUITES[name]:
            label = case.__name__
            try:
                r, fails = case()
            except Exception as e:
                r, fails = None, [f"request failed: {e}"]
            status = "PASS" if not fails else "FAIL"
            results.append((label, status))
            print(f"\n[{status}] {label} — {case.__doc__}")
            if r is not None:
                print(f"  {r['elapsed']:.1f}s | loaders={r['loaders']}")
                if r["travel"] is not None:
                    t = r["travel"]
                    print(f"  plan: destinations={t.get('destinations')} start={t.get('startDate')} "
                          f"end={t.get('endDate')} pax={t.get('pax')} feedback={t.get('feedback')}")
                else:
                    print(f"  answer: {r['answer'][:300].replace(chr(10), ' ')}")
            for f in fails:
                print(f"  ✗ {f}")

    passed = sum(1 for _, s in results if s == "PASS")
    print(f"\n{'=' * 70}\n  {passed}/{len(results)} passed")
    for label, s in results:
        if s == "FAIL":
            print(f"  FAIL  {label}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    picked = sys.argv[1:] or list(SUITES)
    unknown = [p for p in picked if p not in SUITES]
    if unknown:
        sys.exit(f"unknown suite(s) {unknown}; choose from {list(SUITES)}")
    sys.exit(run(picked))
