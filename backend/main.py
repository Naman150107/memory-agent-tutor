from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from agent import agent
from memory import get_all_memories, clear_all_memories, delete_memory
from reflection import save_reflection, get_reflections
import uuid

app = FastAPI(
    title="Memory Agent Tutor API",
    description="Persistent memory AI tutor with reflection loop",
    version="1.0.0",
)

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None


@app.get("/")
async def root():
    return {"status": "ok", "message": "Memory Agent Tutor API is running"}


@app.post("/chat")
async def chat(req: ChatRequest):
    """Main chat endpoint — runs the full LangGraph agent."""
    if not req.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())

    try:
        result = agent.invoke({
            "user_id": req.user_id,
            "message": req.message,
            "memories": [],
            "response": "",
            "reflection": "",
            "session_id": session_id,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Persist reflection to DB
    save_reflection(
        user_id=req.user_id,
        session_id=session_id,
        message=req.message,
        response=result.get("response", ""),
        reflection=result.get("reflection", ""),
    )

    return {
        "response": result.get("response", ""),
        "reflection": result.get("reflection", ""),
        "session_id": session_id,
        "memories_used": len(result.get("memories", [])),
    }


@app.get("/memories/{user_id}")
async def memories(user_id: str):
    """Returns all memories — used by live memory panel in React."""
    return {"memories": get_all_memories(user_id)}


@app.delete("/memories/{user_id}/all")
async def reset_memories(user_id: str):
    """Reset all memories for a user — useful for demo."""
    success = clear_all_memories(user_id)
    return {"status": "cleared" if success else "error"}


@app.delete("/memories/single/{memory_id}")
async def remove_memory(memory_id: str):
    """Delete a single memory by ID."""
    success = delete_memory(memory_id)
    return {"status": "deleted" if success else "error"}


@app.get("/reflections/{user_id}")
async def reflections(user_id: str):
    """Returns reflection history — shows how agent improved over time."""
    return {"reflections": get_reflections(user_id)}


# Run with: uvicorn main:app --reload --port 8000
