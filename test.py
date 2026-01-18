import requests

def get_answer(message, priority="high", url="http://localhost:8000/chat",user_id="hadi123"):
    response = requests.post(
        url,
        json={"message": message, "priority": priority, "user_id": user_id},
        stream=True
    )
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            print(chunk.decode(), end="")

# Example usage
get_answer("What is Travelogging? and Can I share my Travelogues with others?")
