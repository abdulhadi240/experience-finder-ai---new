"""
Checks RULE ZERO: turn 1 must invite the user to heart/favorite their picks (Shape C);
the "start planning" offer only appears from turn 2 onward.

Shape A (one destination, turn 2+): "...share more..., or shall we start planning your trip to X?"
Shape B (many destinations, turn 2+): "which of these would you like to plan a trip to - or keep exploring?"
Shape C (turn 1): "...heart the ones you like, and we'll use your favorites to shape your itinerary."
"""
import json, re, sys, time, uuid
import requests

URL = "http://localhost:8080/chat"
HDRS = {"Content-Type": "application/json", "Accept": "text/event-stream"}

# An offer to start planning: some invitation verb near a planning noun.
OFFER_RE = re.compile(
    r"(start|begin) (planning|building|mapping)"
    r"|"
    r"(shall (we|i)|want me to|would you like (me )?to|should i|i can|ready to|like me to)"
    r"[^.?!]{0,70}?"
    r"(plan|planning|build|building|map out|put together)"
    r"|"
    r"(plan|planning|build|building|map out|put together|shape|shaping)[^.?!]{0,45}?(trip|itinerary)"
    r"|"
    r"(plan|planning) around",
    re.I,
)
EXPLORE_RE = re.compile(r"keep exploring|keep looking|explore (more|further)|narrow (it|things) down", re.I)

# Turn 1: an invitation to heart/favorite picks so they shape the itinerary.
FAVORITE_RE = re.compile(
    r"(heart|favorite|favourite)[^.?!]{0,60}?(like|love|pick|option|one)"
    r"|"
    r"(favorite|favourite)s?[^.?!]{0,45}?(shape|build|create|inform|guide)[^.?!]{0,20}?(itinerary|trip|plan)",
    re.I,
)

# A continent/region is never one plannable destination.
REGIONS = (r"Europe|Asia|Southeast Asia|South America|Central America|North America|Africa|"
           r"the Caribbean|Caribbean|Scandinavia|the Balkans|Balkans|the Middle East|Middle East")
REGION_COLLAPSE_RE = re.compile(
    r"(plan|planning|build|building)[^.?!]{0,30}?\b(" + REGIONS + r")\b[^.?!]{0,20}?(trip|itinerary)"
    r"|(trip|itinerary) to (" + REGIONS + r")\b",
    re.I,
)


def send(message, thread_id="", param="explore", user_id=None):
    payload = {
        "message": message,
        "user_id": user_id or ("po_" + uuid.uuid4().hex[:8]),
        "user_type": "logged-in",
        "reference": "hiptraveler",
        "param": param,
        "threadId": thread_id,
    }
    text, out_thread, is_trip = "", thread_id, False
    with requests.post(URL, json=payload, headers=HDRS, stream=True, timeout=240) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            ls = line.decode("utf-8", "replace")
            if not ls.startswith("data: "):
                continue
            try:
                d = json.loads(ls[6:])
            except json.JSONDecodeError:
                continue
            if d.get("status") == "started":
                out_thread = d.get("threadId", thread_id)
            elif "content" in d:
                text += d["content"]
            elif "travel" in d:
                is_trip = True

    answer = text
    raw = text.strip()
    if raw.startswith("{"):
        try:
            answer = json.loads(raw).get("answer", text)
        except json.JSONDecodeError:
            m = re.match(r'^\{"answer"\s*:\s*"(.*?)"?\}?$', raw, re.S)
            if m:
                answer = m.group(1)
    answer = answer.replace("\\n", "\n").replace('\\"', '"')
    return {"thread": out_thread, "answer": answer, "is_trip": is_trip}


def closing_of(answer):
    """Last sentence that ends with '?', else the last non-empty line."""
    a = answer.strip().rstrip('"').rstrip("}").strip()
    # strip any POI block so we look at prose only
    a = re.sub(r"<POIS>.*?</POIS>", " ", a, flags=re.S | re.I)
    # URLs contain '?' and would masquerade as the closing sentence
    a = re.sub(r"<link>.*?</link>", " ", a, flags=re.S | re.I)
    a = re.sub(r"https?://\S+|www\.\S+", " ", a)
    qs = re.findall(r"[^.?!\n]*\?", a)
    if qs:
        return qs[-1].strip()
    lines = [l.strip() for l in a.split("\n") if l.strip()]
    return lines[-1] if lines else ""


CASES = [
    # (label, query, expected_turn2_shape, destination that must be named for shape A, turn2 follow-up)
    ("A1 city",        "best things to do in Barcelona",                     "A", "Barcelona", "I like museums and food"),
    ("A2 city",        "what should I see in Kyoto",                         "A", "Kyoto", "I love temples and quiet spots"),
    ("A3 country",     "top things to do in Portugal",                       "either", "Portugal", "I'm into beaches"),
    ("B1 open",        "I want to travel somewhere in Europe but not sure where yet", "B", None, "I like warm weather and nightlife"),
    ("B2 compare",     "Tokyo or Seoul - which should I visit?",             "B", None, "I care most about food"),
    ("A4 safety",      "is it safe to travel to Colombia right now?",        "either", "Colombia", "I'm visiting family there"),
    ("A5 visa",        "do I need a visa for Vietnam?",                      "either", "Vietnam", "traveling for two weeks"),
    ("B3 list",        "best surf destinations in Central America",          "B", None, "I'm a beginner surfer"),
]


def run():
    results, failures = [], []
    for label, q, shape, dest, followup in CASES:
        t0 = time.time()
        try:
            r1 = send(q)
        except Exception as e:
            failures.append(f"{label}: REQUEST ERROR (turn1) {e}")
            print(f"\n[{label}] ERROR {e}")
            continue
        c1 = closing_of(r1["answer"])
        has_favorite = bool(FAVORITE_RE.search(c1))
        has_offer_early = bool(OFFER_RE.search(c1))

        print(f"\n[{label}-turn1] ({time.time()-t0:.0f}s) {q}")
        print(f"  closing: {c1}")
        print(f"  favorite_invite={has_favorite} premature_offer={has_offer_early}")

        if not has_favorite:
            failures.append(f"{label} turn1: NO FAVORITE/HEART INVITE in closing (Shape C) -> {c1!r}")
        if has_offer_early:
            failures.append(f"{label} turn1: premature 'start planning' offer before favoriting -> {c1!r}")

        # Turn 2: user shares preferences -> Shape A/B "start planning" offer should now appear.
        t1 = time.time()
        try:
            r2 = send(followup, thread_id=r1["thread"])
        except Exception as e:
            failures.append(f"{label}: REQUEST ERROR (turn2) {e}")
            print(f"  [turn2] ERROR {e}")
            results.append((label, c1))
            continue
        c2 = closing_of(r2["answer"])
        has_offer = bool(OFFER_RE.search(c2))
        has_explore = bool(EXPLORE_RE.search(c2))
        names = bool(dest and re.search(re.escape(dest), c2, re.I))
        ends_q = c2.endswith("?")
        collapsed = bool(REGION_COLLAPSE_RE.search(c2))

        print(f"[{label}-turn2] ({time.time()-t1:.0f}s) {followup}")
        print(f"  closing: {c2}")
        print(f"  offer={has_offer} ends_q={ends_q} names_dest={names} explore_branch={has_explore} region_collapse={collapsed}")

        if collapsed:
            failures.append(f"{label} turn2: collapsed a continent/region into ONE destination -> {c2!r}")
        if not has_offer:
            failures.append(f"{label} turn2: NO PLANNING OFFER in closing -> {c2!r}")
        if not ends_q:
            failures.append(f"{label} turn2: closing does not end with '?' -> {c2!r}")
        if shape == "A" and dest and not names:
            failures.append(f"{label} turn2: shape A must name '{dest}' -> {c2!r}")
        # Shape B is satisfied by "which of these", a keep-exploring branch, or
        # an explicit choice between named destinations ("...to Lisbon, Porto, or Split?").
        multi_choice = bool(re.search(r"[A-Z][\w'\-]+(,\s*[A-Z][\w'\-]+)+,?\s+or\s+[A-Z][\w'\-]+", c2))
        if shape == "B" and not (has_explore or multi_choice or re.search(r"which", c2, re.I)):
            failures.append(f"{label} turn2: shape B must ask which / offer to keep exploring -> {c2!r}")
        results.append((label, c1, c2))

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print("  -", f)
    else:
        print(f"ALL {len(results)} CASES PASS - turn1 invites favoriting, turn2 offers to start planning.")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
