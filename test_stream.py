import requests
import json
import sys


def stream_chat_response():
    url = "http://localhost:8080/chat"

    payload = {
        "message": "best things to do in bali",
        "user_id": "user12113",
        "reference": "hiptraveler",
        "param": "explore",
        "threadId": ""
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    print("=" * 80)
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("=" * 80)

    try:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=120) as response:
            response.raise_for_status()

            full_answer = ""
            thread_id   = None

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8")
                if not line_str.startswith("data: "):
                    continue

                try:
                    data = json.loads(line_str[6:])
                except json.JSONDecodeError:
                    continue

                # ── Started ──────────────────────────────────────────
                if "status" in data and data["status"] == "started":
                    thread_id = data.get("threadId", "")
                    print(f"[STARTED] thread={thread_id}")

                # ── TTFB (fires on starter's first token) ────────────
                elif "time_to_first_byte" in data:
                    print(f"[TTFB]    {data['time_to_first_byte']:.2f}s")
                    print("-" * 80)

                # ── Streaming content (starter + agent, same format) ──
                elif "content" in data:
                    chunk = data["content"]
                    full_answer += chunk
                    print(chunk, end="", flush=True)

                # ── Non-streaming (trip planning) JSON response ───────
                elif "travel" in data:
                    trip_plan   = data["travel"][0]
                    timing_info = data["travel"][1]
                    print(f"\n[TRIP PLAN — FULL RAW RESPONSE]")
                    print(json.dumps(trip_plan, indent=2, ensure_ascii=False))
                    print(f"[TIMING] {json.dumps(timing_info, indent=2)}")

                # ── Dynamic loading indicator ─────────────────────────
                elif "loading" in data:
                    print(f"\n[LOADING] {data['loading']}", flush=True)

                # ── Error ─────────────────────────────────────────────
                elif "error" in data:
                    print(f"\n[ERROR] {data['error']}")

                # ── Done ──────────────────────────────────────────────
                elif data.get("done") and "travel" not in data:
                    total = data.get("total_time", 0)
                    blocked = " [BLOCKED]" if data.get("blocked") else ""
                    print(f"\n{'=' * 80}")
                    print(f"[DONE]{blocked} total={total:.2f}s  thread={thread_id}  chars={len(full_answer)}")
                    print("=" * 80)

    except requests.exceptions.HTTPError as e:
        print(f"[HTTP ERROR]    {e}")
        print(f"[STATUS CODE]   {e.response.status_code}")
        print(f"[RESPONSE BODY] {repr(e.response.text)}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"[FAILED] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")


if __name__ == "__main__":
    stream_chat_response()


