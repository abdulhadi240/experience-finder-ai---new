from .config import settings
from zep_cloud.client import Zep # type: ignore
import uuid
from zep_cloud.types import Message # type: ignore


client = Zep(api_key=settings.zep_api_key)


def check_user(user_id: str):
    if not user_id:
        return None

    try:
        # Try to get existing user first
        try:
            existing_user = client.user.get(user_id)
        except:  # User doesn't exist
            # Create new user with proper data
            client.user.add(
                user_id=user_id
            )

        # Create thread for all users
        thread_id = uuid.uuid4().hex
        client.thread.create(
            thread_id=thread_id,
            user_id=user_id,
        )

        return thread_id

    except Exception as e:
        print(f"Error in check_user: {e}")
        return None


def setup_user_session(user_id: str, thread_id: str) -> None:
    """Create/verify user and thread in Zep using a pre-generated thread_id.
    Designed to run in asyncio.to_thread so it never blocks the event loop."""
    if not user_id:
        return
    try:
        try:
            client.user.get(user_id)
        except:
            client.user.add(user_id=user_id)
        client.thread.create(thread_id=thread_id, user_id=user_id)
    except Exception as e:
        print(f"setup_user_session error: {e}")

    

def add_message(message: str, thread_id: str, role: str):
    messages = [
        Message(
            role=f"{role}",
            content=f"{message}",
        )
    ]
    client.thread.add_messages(thread_id, messages=messages)
    print(f"\n[ZEP SAVE] role={role} | thread={thread_id}")
    print(f"[ZEP SAVE] content={message[:120]}")

    # Verify by reading back the last 2 messages from the thread
    try:
        verify = client.thread.get(thread_id, lastn=2)
        saved = getattr(verify, 'messages', []) or []
        print(f"[ZEP VERIFY] Messages in thread: {len(saved)}")
        for m in saved:
            r = getattr(m, 'role', '?')
            c = str(getattr(m, 'content', ''))[:80]
            print(f"  [{r}] {c}")
    except Exception as e:
        print(f"[ZEP VERIFY] Error: {e}")
    
    
    
def get_message(thread_id: str) -> str :
   memory = client.thread.get_user_context(thread_id=thread_id)
   context_block = memory.context
   return context_block



def get_user_memory_for_engage(user_id: str) -> str:
    """
    Retrieve user's Zep node summary + last user question for re-engagement.
    Returns combined string, or None on failure/not found.
    """
    parts = []

    # 1. User node summary
    try:
        node_response = client.user.get_node(user_id=user_id)
        summary = None
        if node_response and hasattr(node_response, 'node') and node_response.node:
            summary = getattr(node_response.node, 'summary', None)
        if summary:
            parts.append(summary)
    except Exception as e:
        print(f"[ZEP ENGAGE] Error fetching node summary: {e}")
        return None

    # 2. Last user message from most recent thread
    try:
        threads_response = client.user.get_threads(user_id=user_id)
        threads = getattr(threads_response, 'threads', None) or threads_response or []

        if threads:
            # Sort by created_at descending to get the most recent thread
            threads_sorted = sorted(
                threads,
                key=lambda t: getattr(t, 'created_at', '') or '',
                reverse=True,
            )
            last_thread = threads_sorted[0]
            thread_id = getattr(last_thread, 'thread_id', None) or getattr(last_thread, 'id', None)
            print(f"[ZEP ENGAGE] Most recent thread ID: {thread_id}")

            if thread_id:
                msg_response = client.thread.get(thread_id, lastn=2)
                messages = getattr(msg_response, 'messages', None) or []

                for msg in reversed(messages):
                    role = getattr(msg, 'role', '') or getattr(msg, 'role_type', '') or ''
                    if str(role).lower() == 'user':
                        last_user_msg = getattr(msg, 'content', None)
                        if last_user_msg:
                            print(f"[ZEP ENGAGE] Last user message: {last_user_msg}")
                            # Insert at front so nano sees it first
                            parts.insert(0, f"[LAST_TOPIC] {last_user_msg}")
                        break
    except Exception as e:
        print(f"[ZEP ENGAGE] Error fetching last message: {e}")

    return "\n\n".join(parts) if parts else None


def delete_user(user_id: str):
    client.user.delete(f"{user_id}")


def create_new_user(email: str , first_name: str , last_name: str , user_id: str):
    client.user.add(
                user_id=user_id,
                email=f"{email}",
                first_name=f"{first_name}",
                last_name=f"{last_name}",
            )