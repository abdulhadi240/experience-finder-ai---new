"""
Multi-turn flow tests:
  Flow A — destination known from T1 (Barcelona):
    T1: "best things to do in Barcelona"  → POIs + Q1
    T2: answer Q1                          → advice + planning Q  (no Q2!)
    T3: "yes, build it"                    → trip planning JSON

  Flow B — no destination, keep asking until one is chosen:
    T1: "I want to travel somewhere in Europe"  → no POIs (or multi), ask Q1 mindset
    T2: user answers mindset Q                   → advice + planning Q
    T3: user picks destination                   → trip planning
"""

import requests
import json
import time

URL  = "http://localhost:8080/chat"
HDRS = {"Content-Type": "application/json", "Accept": "text/event-stream"}


def send(message: str, thread_id: str = "", reference: str = "hiptraveler") -> dict:
    payload = {
        "message":   message,
        "user_id":   "test_user_flows",
        "reference": reference,
        "param":     "explore",
        "threadId":  thread_id,
    }
    full_text   = ""
    out_thread  = thread_id
    has_poi     = False
    ends_q      = False
    has_bullets = False
    is_trip     = False
    ttfb        = None
    t0          = time.time()

    with requests.post(URL, json=payload, headers=HDRS, stream=True, timeout=180) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            ls = line.decode("utf-8")
            if not ls.startswith("data: "):
                continue
            try:
                data = json.loads(ls[6:])
            except json.JSONDecodeError:
                continue

            if "status" in data and data["status"] == "started":
                out_thread = data.get("threadId", thread_id)
            elif "time_to_first_byte" in data:
                ttfb = data["time_to_first_byte"]
            elif "content" in data:
                full_text += data["content"]
            elif "travel" in data:
                is_trip = True
                print("  [TRIP PLAN destinations]:", data["travel"][0].get("destinations"))

    elapsed = time.time() - t0

    # Content chunks join to a JSON string like {"answer":"..."}
    # Extract the actual answer text before analysing it
    answer_text = full_text
    raw = full_text.strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            parsed = json.loads(raw)
            answer_text = parsed.get("answer", full_text)
        except json.JSONDecodeError:
            # Partial / streaming JSON — strip trailing }" manually
            if raw.endswith('"}'):
                answer_text = raw[len('{"answer":"'):-2] if raw.startswith('{"answer":"') else raw[:-2]

    stripped = answer_text.strip()
    has_poi     = "<poi" in full_text.lower() or "<pois>" in full_text.lower()
    ends_q      = stripped.endswith("?")
    has_bullets = "- " in answer_text or "• " in answer_text or "* " in answer_text

    return {
        "thread_id":  out_thread,
        "text":       full_text,
        "answer":     answer_text,
        "has_poi":    has_poi,
        "ends_q":     ends_q,
        "has_bullets":has_bullets,
        "is_trip":    is_trip,
        "elapsed":    elapsed,
        "ttfb":       ttfb,
    }


def divider(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def report(label: str, r: dict):
    snippet = r["answer"][:500].replace("\n", " ")
    print(f"\n  [{label}]")
    print(f"  elapsed={r['elapsed']:.1f}s | ttfb={r['ttfb']}s")
    print(f"  has_poi={r['has_poi']} | ends_q={r['ends_q']} | has_bullets={r['has_bullets']} | is_trip={r['is_trip']}")
    print(f"  thread={r['thread_id']}")
    print(f"  --- text snippet ---")
    print(f"  {snippet}...")
    print()


# ─────────────────────────────────────────────
# FLOW A — Destination known from T1
# ─────────────────────────────────────────────
divider("FLOW A — Destination known from T1 (Barcelona)")

print("\n[A-T1] Sending: 'best things to do in Barcelona'")
a1 = send("best things to do in Barcelona")
report("A-T1 result", a1)
assert a1["has_poi"],  "A-T1 FAIL: expected POIs in T1 response"
assert a1["ends_q"],   "A-T1 FAIL: expected T1 to end with a question"
print("  A-T1 PASS: got POIs + closing question")

print("\n[A-T2] Sending follow-up answer to Q1...")
a2 = send(
    "I love local food markets, hidden neighbourhood cafes, and architecture — "
    "not too rushed, prefer depth over ticking boxes",
    thread_id=a1["thread_id"],
)
report("A-T2 result", a2)
assert not a2["has_poi"], "A-T2 FAIL: T2 should NOT have POIs"
assert a2["ends_q"],      "A-T2 FAIL: T2 should end with planning question"
print("  A-T2 PASS: no POIs, ends with planning question (single follow-up done)")

print("\n[A-T3] Accepting trip planning...")
a3 = send(
    "Yes, sounds perfect — let's build it around Barcelona!",
    thread_id=a2["thread_id"],
)
report("A-T3 result", a3)
assert a3["is_trip"], "A-T3 FAIL: T3 should trigger trip planning JSON"
print("  A-T3 PASS: trip planning triggered")


# ─────────────────────────────────────────────
# FLOW B — No destination at first; keep Q&A until one is chosen
# ─────────────────────────────────────────────
divider("FLOW B — No destination first; multiple Q&A until destination picked")

print("\n[B-T1] Sending: 'I want to explore somewhere in Europe but not sure where'")
b1 = send("I want to explore somewhere in Europe but not sure where yet")
report("B-T1 result", b1)
assert b1["ends_q"], "B-T1 FAIL: expected a question"
print("  B-T1 PASS: got response with question")

print("\n[B-T2] Answering mindset Q without naming a destination...")
b2 = send(
    "I love history, local food, and walking cities — not a beach person",
    thread_id=b1["thread_id"],
)
report("B-T2 result", b2)
assert b2["ends_q"], "B-T2 FAIL: expected a question (planning or destination selection)"
print("  B-T2 PASS: got question — either planning Q or destination selector")

print("\n[B-T3] User picks destination...")
b3 = send(
    "Let's go with Lisbon, Portugal",
    thread_id=b2["thread_id"],
)
report("B-T3 result", b3)
print(f"  B-T3: is_trip={b3['is_trip']} | ends_q={b3['ends_q']}")
if b3["is_trip"]:
    print("  B-T3 PASS: trip planning triggered after destination chosen")
elif b3["ends_q"]:
    print("  B-T3 INFO: got planning Q (agent needs one more confirm) — sending T4")
    b4 = send("Yes, build it!", thread_id=b3["thread_id"])
    report("B-T4 result", b4)
    assert b4["is_trip"], "B-T4 FAIL: expected trip planning after confirm"
    print("  B-T4 PASS: trip planning triggered")


divider("ALL TESTS COMPLETE")
