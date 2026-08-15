"""
Accepting the planning offer must TRIGGER trip planning, not re-ask the question.

Regression test for the loop where "yeah lets plan it" was classified as an
answer to the preference question (FOLLOW_UP_MODE) and therefore never reached
the trip planner — the agent just offered to plan again.

Run the server first:  CREDITS_ENFORCE=false python main.py
"""
import sys
import uuid

from test_planning_offer import send, closing_of

SCENARIOS = [
    # (name, messages, expect trip planning to fire)
    ("Rome / accept immediately",
     ["best things to do in rome", "yeah lets plan it"], True),

    ("Kyoto / bare yes",
     ["what should I see in Kyoto", "yes"], True),

    ("Barcelona / share prefs, then accept",
     ["best things to do in Barcelona",
      "I love food markets and wandering, not big museums",
      "yes lets plan it"], True),

    # Multiple destinations on the table: a vague "yes" must NOT blind-plan a
    # region — the agent has to ask which one first.
    ("Central America / vague yes stays a question",
     ["best surf destinations in Central America", "yes"], False),
]


def run():
    fails = []
    for name, steps, want_trip in SCENARIOS:
        uid = "acc_" + uuid.uuid4().hex[:8]
        thread, got = "", False
        print(f"\n=== {name} ===")
        for i, msg in enumerate(steps, 1):
            r = send(msg, thread_id=thread, user_id=uid)
            thread = r["thread"]
            got = r["is_trip"]
            print(f"  T{i} {msg!r} -> is_trip={got}")
            if not got:
                print(f"     closing: {closing_of(r['answer'])[:160]}")
            else:
                break
        if got != want_trip:
            fails.append(f"{name}: expected is_trip={want_trip}, got {got}")

    print("\n" + "=" * 60)
    if fails:
        print(f"FAILURES ({len(fails)}):")
        for f in fails:
            print("  -", f)
    else:
        print(f"ALL {len(SCENARIOS)} SCENARIOS PASS")
    print("=" * 60)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(run())
