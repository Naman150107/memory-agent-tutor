from mem0 import MemoryClient
from dotenv import load_dotenv
import os

load_dotenv()

mem0_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))


def get_all_memories(user_id: str) -> list:
    """Fetch all memories for the live memory panel in React."""
    try:
        response = mem0_client.get_all(
            filters={"AND": [{"user_id": user_id}]}
        )
        results = response if isinstance(response, list) else response.get("results", [])
        return [
            {
                "id": m.get("id", ""),
                "memory": m.get("memory", ""),
                "created_at": m.get("created_at", ""),
            }
            for m in results
        ]
    except Exception as e:
        print(f"[Memory] Error fetching all memories: {e}")
        return []


def delete_memory(memory_id: str) -> bool:
    """Allow students to delete a specific memory."""
    try:
        mem0_client.delete(memory_id=memory_id)
        return True
    except Exception as e:
        print(f"[Memory] Error deleting memory {memory_id}: {e}")
        return False


def clear_all_memories(user_id: str) -> bool:
    """Reset all memories — useful for demo/testing."""
    try:
        mem0_client.delete_all(user_id=user_id)
        return True
    except Exception as e:
        print(f"[Memory] Error clearing memories for {user_id}: {e}")
        return False
