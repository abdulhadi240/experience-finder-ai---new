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

    

def add_message(message: str , thread_id: str , role: str):
    messages = [
        Message(
            role=f"{role}",
            content=f"{message}",
        )
    ]
    client.thread.add_messages(thread_id, messages=messages)
    
    
    
def get_message(thread_id: str) -> str :
   memory = client.thread.get_user_context(thread_id=thread_id)
   context_block = memory.context
   return context_block



def delete_user(user_id: str):
    client.user.delete(f"{user_id}")


def create_new_user(email: str , first_name: str , last_name: str , user_id: str):
    client.user.add(
                user_id=user_id,
                email=f"{email}",
                first_name=f"{first_name}",
                last_name=f"{last_name}",
            )