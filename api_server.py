from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any
from graph import graph  # Assuming 'graph' is the main entry point for AI/ML logic

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: List[Any]  # List of dicts with 'role' and 'content', e.g. [{"role": "user", "content": "..."}]

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Prepare the state for the graph (AI agent)
    # The 'messages' format may need to be adapted to match your backend's expectations
    state = {"messages": request.messages}
    result = graph.invoke(state)
    # Extract the AI's response (assuming result['messages'][-1] is the latest AI message)
    ai_message = result["messages"][-1].content if result["messages"] else "No response."
    return ChatResponse(response=ai_message) 