import requests
import json
import sys

def stream_chat_response():
    url = "http://localhost:8080/chat"

    payload = {
        "message": "can i camp in deosai , skardu",
        "user_id": "user12113",
        "reference": "hiptraveler",
        "param": "plan",
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
            in_answer   = False

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8")

                if not line_str.startswith("data: "):
                    continue

                data_str = line_str[6:]

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # ── Started ──────────────────────────────────────
                if "status" in data and data["status"] == "started":
                    thread_id = data.get("threadId", "")
                    print(f"[STARTED] thread={thread_id}")

                # ── Loading stage message ─────────────────────────
                elif data.get("type") == "loading":
                    print(f"[LOADING] {data['message']}")

                # ── TTFB ─────────────────────────────────────────
                elif "time_to_first_byte" in data:
                    print(f"[TTFB]    {data['time_to_first_byte']:.2f}s")
                    print("-" * 80)
                    print("[ANSWER]")
                    in_answer = True

                # ── Streaming content chunks ──────────────────────
                elif "content" in data:
                    chunk = data["content"]

                    # Strip JSON wrapper fragments {"answer":" and "}
                    # that the backend emits for some response types
                    if not in_answer:
                        in_answer = True
                        print("[ANSWER]")

                    # Try to unwrap {"answer":"..."} if the entire
                    # chunk is parseable JSON with an answer key
                    try:
                        parsed = json.loads(chunk)
                        if isinstance(parsed, dict) and "answer" in parsed:
                            chunk = parsed["answer"]
                    except (json.JSONDecodeError, ValueError):
                        pass

                    full_answer += chunk
                    print(chunk, end="", flush=True)

                # ── Non-streaming (trip planning) response ────────
                elif "response" in data:
                    print(f"\n[TRIP PLAN]\n{json.dumps(data['response'], indent=2)}")

                # ── Error ─────────────────────────────────────────
                elif "error" in data:
                    print(f"\n[ERROR] {data['error']}")

                # ── Done ──────────────────────────────────────────
                elif data.get("done"):
                    total = data.get("total_time", 0)
                    print(f"\n{'=' * 80}")
                    print(f"[DONE] total={total:.2f}s  thread={thread_id}  chars={len(full_answer)}")
                    print("=" * 80)

    except requests.exceptions.HTTPError as e:
        print(f"[HTTP ERROR]    {e}")
        print(f"[STATUS CODE]   {e.response.status_code}")
        print(f"[HEADERS]       {dict(e.response.headers)}")
        print(f"[RESPONSE BODY] {repr(e.response.text)}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"[FAILED] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")


if __name__ == "__main__":
    stream_chat_response()
