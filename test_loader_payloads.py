"""
Payload matrix for the chat loaders — run against a live server.

Varies every request field (user_type, threadId new/existing/unknown, is_pro,
plan, param, reference, explicit nulls, long / non-English / PII messages, invalid
bodies) on /chat and /chat/support and checks each stream:
  - support requests show exactly the fixed lines, in order, on time:
      "One moment…" (<0.5s) → "Let me check that for you." (2s) → "Thanks for waiting." (5s)
  - no loader after the answer starts, no loader sent twice
  - the stream finishes with an answer (or the expected 422 for bad bodies)

Run the server first (CREDITS_ENFORCE=false), then:
  python test_loader_payloads.py
  BASE_URL=http://127.0.0.1:8765 python test_loader_payloads.py
Use 127.0.0.1, not localhost, on Windows — localhost adds ~2s per connection.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8080").rstrip("/")
SEQ = ["One moment…", "Let me check that for you.", "Thanks for waiting."]
DUE = [0.0, 2.0, 5.0]
LATE_BY = 0.8  # seconds of slack after a line's due time


def body(message="how do I cancel my premium subscription", **over) -> dict:
    b = {"message": message, "user_id": "test_payloads", "user_type": "logged-in",
         "reference": "hiptraveler", "param": "explore", "threadId": None, "plan": False}
    b.update(over)
    return b


def stream(path: str, payload: dict) -> dict:
    """POST and record every SSE event with its arrival time."""
    t0 = time.time()
    out = {"status": None, "events": [], "loaders": [], "first_content": None,
           "answer": "", "travel": None, "done": False, "error": None, "thread": None}
    with requests.post(BASE + path, json=payload, headers={"Accept": "text/event-stream"},
                       stream=True, timeout=240) as r:
        out["status"] = r.status_code
        if r.status_code != 200:
            return out
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            try:
                d = json.loads(line[6:])
            except ValueError:
                continue
            t = time.time() - t0
            if d.get("status") == "started":
                out["thread"] = d.get("threadId")
            if "loading" in d:
                out["loaders"].append((t, d["loading"]))
            if "content" in d:
                out["first_content"] = out["first_content"] or t
                out["answer"] += d["content"]
            if "travel" in d:
                out["first_content"] = out["first_content"] or t
                out["travel"] = d["travel"]
            if "error" in d:
                out["error"] = d["error"]
            if d.get("done"):
                out["done"] = True
    return out


def check(r: dict, expect: str) -> list[str]:
    """expect: support | travel | no_loaders | invalid"""
    if expect == "invalid":
        return [] if r["status"] == 422 else [f"expected 422, got {r['status']}"]
    fails = []
    if r["status"] != 200:
        return [f"HTTP {r['status']}"]
    if r["error"]:
        fails.append(f"stream error: {r['error']}")
    if not r["done"]:
        fails.append("stream never sent done")
    if not (r["answer"].strip() or r["travel"]):
        fails.append("no answer")
    texts = [txt for _, txt in r["loaders"]]
    if len(texts) != len(set(texts)):
        fails.append(f"loader sent twice: {texts}")
    if r["first_content"] is not None:
        after = [(round(t, 2), txt) for t, txt in r["loaders"] if t > r["first_content"]]
        if after:
            fails.append(f"loader after answer started: {after}")
    if expect == "no_loaders" and texts:
        fails.append(f"expected no loaders, got {texts}")
    if expect == "support":
        if texts != SEQ[:len(texts)] or not texts:
            fails.append(f"not the fixed support sequence: {texts}")
        for t, txt in r["loaders"]:
            if txt in SEQ:
                i = SEQ.index(txt)
                if t < DUE[i] - 0.05:
                    fails.append(f"{txt!r} early at {t:.2f}s")
                if t > DUE[i] + (0.5 if i == 0 else LATE_BY):
                    fails.append(f"{txt!r} late at {t:.2f}s")
        # every line that fell due before the answer started must have been shown
        if r["first_content"] is not None:
            owed = [s for s, due in zip(SEQ, DUE) if due + LATE_BY < r["first_content"]]
            missing = [s for s in owed if s not in texts]
            if missing:
                fails.append(f"missing lines that were due: {missing}")
    if expect == "travel" and any(txt in SEQ[1:] for txt in texts):
        fails.append(f"travel question got support lines: {texts}")
    return fails


def seed_thread(path: str, message: str) -> str:
    """Make a real conversation and return its threadId (for history cases)."""
    return stream(path, body(message)).get("thread")


def cases() -> list[tuple[str, str, dict, str]]:
    # Existing threads so the history fetch actually returns something
    support_thread = seed_thread("/chat/support", "how do I get a refund?")
    travel_thread = seed_thread("/chat", "things to do in Lisbon")
    long_msg = ("I bought Premium last month and I'm trying to understand how cancellation works. " * 25).strip()
    return [
        # ── /chat/support ──
        ("support baseline", "/chat/support", body(), "support"),
        ("support visitor", "/chat/support", body(user_type="visitor", user_id="v_123"), "support"),
        ("support user_type null", "/chat/support", body(user_type=None), "support"),
        ("support is_pro", "/chat/support", body(is_pro=True), "support"),
        ("support plan=True (ignored)", "/chat/support", body(plan=True), "support"),
        ("support param=plan", "/chat/support", body(param="plan"), "support"),
        ("support param empty", "/chat/support", body(param=""), "support"),
        ("support explicit null flags", "/chat/support", body(is_pro=None, plan=None), "support"),
        ("support other reference", "/chat/support", body(reference="otherportal"), "support"),
        ("support existing thread", "/chat/support",
         body("and how long does that take?", threadId=support_thread), "support"),
        ("support unknown threadId", "/chat/support", body(threadId="does-not-exist-123"), "support"),
        ("support greeting", "/chat/support", body("hi"), "support"),
        ("support long message", "/chat/support", body(long_msg), "support"),
        ("support Spanish", "/chat/support", body("¿Cómo cancelo mi suscripción Premium?"), "support"),
        ("support emoji", "/chat/support", body("my esim isn't working 😩📱"), "support"),
        ("support PII email", "/chat/support",
         body("refund to jane.doe@example.com please"), "support"),
        # ── /chat ──
        ("chat support baseline", "/chat", body("can I pay with paypal"), "support"),
        ("chat support visitor", "/chat", body("how do I reset my password", user_type="visitor", user_id="v_9"), "support"),
        ("chat support is_pro", "/chat", body("how do loyalty points work?", is_pro=True), "support"),
        ("chat support param=plan", "/chat", body("is there a student discount", param="plan"), "support"),
        ("chat support after travel thread", "/chat",
         body("how do I save this to my trips?", threadId=travel_thread), "support"),
        ("chat travel", "/chat", body("best street food in Bangkok"), "travel"),
        ("chat travel visitor", "/chat", body("rooftop bars in New York", user_type="visitor", user_id="v_7"), "travel"),
        ("chat travel existing thread", "/chat", body("what about museums there?", threadId=travel_thread), "travel"),
        ("chat plan=True fast path", "/chat",
         body("plan 3 days in Rome for 2 adults in May", plan=True), "no_loaders"),
        ("chat greeting", "/chat", body("hey!"), "travel"),
        # ── invalid bodies → 422, never a stream ──
        ("invalid missing threadId key", "/chat/support",
         {k: v for k, v in body().items() if k != "threadId"}, "invalid"),
        ("invalid blank user_id", "/chat/support", body(user_id="  "), "invalid"),
        ("invalid missing message", "/chat", {k: v for k, v in body().items() if k != "message"}, "invalid"),
        ("invalid bad user_type", "/chat", body(user_type="admin"), "invalid"),
    ]


def main() -> int:
    matrix = cases()

    def run(c):
        name, path, payload, expect = c
        try:
            r = stream(path, payload)
        except Exception as e:
            return name, path, expect, None, [f"request failed: {e}"]
        return name, path, expect, r, check(r, expect)

    with ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "4"))) as pool:
        results = list(pool.map(run, matrix))

    failed = 0
    for name, path, expect, r, fails in results:
        failed += bool(fails)
        status = "FAIL" if fails else "PASS"
        if r is None or r["status"] != 200:
            print(f"[{status}] {name} ({path}) → HTTP {r and r['status']}")
        else:
            lines = ", ".join(f"{t:.2f}s {txt}" for t, txt in r["loaders"]) or "—"
            start = f"{r['first_content']:.2f}s" if r["first_content"] else "never"
            print(f"[{status}] {name} ({path}) | answer at {start} | loaders: {lines}")
        for f in fails:
            print(f"    ✗ {f}")
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
