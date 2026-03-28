from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from mem0 import MemoryClient
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize clients
mem0_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Shared state across all nodes
class AgentState(TypedDict):
    user_id: str
    message: str
    memories: List[str]
    response: str
    reflection: str
    session_id: Optional[str]


def retrieve_memory(state: AgentState) -> dict:
    try:
        results = mem0_client.search(
            query=state["message"],
            filters={"AND": [{"user_id": state["user_id"]}]},
            limit=5
        )
        memories = [r["memory"] for r in results if "memory" in r]
        print(f"[Memory] Retrieved {len(memories)} memories for {state['user_id']}")
        return {"memories": memories}
    except Exception as e:
        print(f"[Memory] Error retrieving memories: {e}")
        return {"memories": []}


def generate_response(state: AgentState) -> dict:
    memory_context = "\n".join([f"- {m}" for m in state["memories"]])

    system_prompt = """You are an expert CS tutor with perfect memory of your student.
Use the student profile below to personalize every response.
If the student struggles with a topic, go slower and use analogies.
If they are advanced, go deeper. Always reference what you know about them.
Keep responses concise, warm, and educational."""

    if memory_context:
        user_prompt = f"""Student profile from memory:
{memory_context}

Student asks: {state["message"]}

Give a tailored, helpful response based on everything you know about this student."""
    else:
        user_prompt = f"""This is the student's first session — no previous interactions on record.

Student asks: {state["message"]}

Give a warm, helpful response and try to understand their current level."""

    try:
        response = llm.invoke([
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
        ])
        return {"response": response.content}
    except Exception as e:
        print(f"[Agent] Error generating response: {e}")
        return {"response": "I encountered an error. Please try again."}


def write_memory(state: AgentState) -> dict:
    try:
        mem0_client.add(
            messages=[
                {"role": "user", "content": state["message"]},
                {"role": "assistant", "content": state["response"]}
            ],
            user_id=state["user_id"]
        )
        print(f"[Memory] Stored interaction for {state['user_id']}")
    except Exception as e:
        print(f"[Memory] Error writing memory: {e}")
    return {}


def reflect(state: AgentState) -> dict:
    memory_summary = "\n".join(state["memories"]) if state["memories"] else "No prior memories."

    reflection_prompt = f"""You are reviewing your own tutoring session. Be honest and specific.

Student question: {state['message']}
Your response: {state['response']}
Student memory profile:
{memory_summary}

Provide a structured critique (1-2 sentences per point):
1. TAILORING: Was the response appropriately tailored to this student's level? (score /10)
2. ACCURACY: Was the explanation correct and clear?
3. GAPS: What misconception might the student still have?
4. NEXT_TIME: What will you do differently in the next session?
5. NEW_FACTS: What new facts did you learn about this student?"""

    try:
        reflection = llm.invoke([
            {"role": "user", "content": reflection_prompt}
        ])
        return {"reflection": reflection.content}
    except Exception as e:
        print(f"[Reflect] Error generating reflection: {e}")
        return {"reflection": "Reflection unavailable for this session."}


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_memory", retrieve_memory)
    graph.add_node("generate_response", generate_response)
    graph.add_node("write_memory", write_memory)
    graph.add_node("reflect", reflect)

    graph.add_edge(START, "retrieve_memory")
    graph.add_edge("retrieve_memory", "generate_response")
    graph.add_edge("generate_response", "write_memory")
    graph.add_edge("write_memory", "reflect")
    graph.add_edge("reflect", END)

    return graph.compile()


agent = build_agent()
