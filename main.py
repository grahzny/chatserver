# main.py - A clean Python backend for LLM interactions
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import httpx
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
LLM_SERVER_URL = "http://localhost:8001/v1/chat/completions"
DEFAULT_MODEL = "gemma3-27b"

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    conversationId: str
    userId: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    toolsUsed: Optional[List[str]] = None
    debug: Optional[Dict[str, Any]] = None

# Tool definitions
TOOLS = [
    {
        "name": "get_current_datetime",
        "description": "Returns the current date and time in ISO 8601 format",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "calculator",
        "description": "Calculate mathematical expressions safely",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }
]

# Tool implementations
def get_current_datetime():
    return {"datetime": datetime.now().isoformat()}

def calculator(expression: str):
    try:
        # Safe evaluation
        allowed_chars = set('0123456789+-*/()%. ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return {"result": result}
        else:
            return {"error": "Invalid expression"}
    except Exception as e:
        return {"error": str(e)}

TOOL_FUNCTIONS = {
    "get_current_datetime": get_current_datetime,
    "calculator": calculator
}

async def call_llm_with_tools(messages: List[dict], include_tools: bool = True):
    """Call the LLM server with optional tool support"""
    print(messages)
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        if include_tools:
            payload["tools"] = TOOLS
            payload["tool_choice"] = "auto"
        
        logger.info(f"Calling LLM with payload: {json.dumps(payload, indent=2)}")
        
        response = await client.post(LLM_SERVER_URL, json=payload)
        response.raise_for_status()
        
        return response.json()

@app.post("/api/chat", response_model=ChatResponse)

async def chat(request: ChatRequest):
    """Main chat endpoint that handles tool calling seamlessly"""
    try:
        # Convert messages to the format expected by LLM
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        print(messages)
        # Call LLM with tools - this will handle tool execution automatically
        # since your FastAPI server (on port 8001) already handles tool calling
        llm_response = await call_llm_with_tools(messages)
        
        # Extract the response - your FastAPI server returns the final response
        # after executing any tools, so we just pass it through
        assistant_message = llm_response["choices"][0]["message"]["content"]
        
        logger.info(f"Assistant response: {assistant_message[:100]}...")
        
        return ChatResponse(
            message=assistant_message,
            toolsUsed=None,  # Your FastAPI server handles this internally
            debug={"model": DEFAULT_MODEL, "tools_available": len(TOOLS)} if request.userId == "debug" else None
        )
        
    except httpx.HTTPError as e:
        logger.error(f"HTTP error calling LLM: {e}")
        raise HTTPException(status_code=500, detail=f"LLM server error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model": DEFAULT_MODEL, "tools": len(TOOLS)}

# Additional endpoints for specific use cases
@app.post("/api/tools/execute")
async def execute_tool(name: str, parameters: dict):
    """Direct tool execution endpoint for testing"""
    if name not in TOOL_FUNCTIONS:
        raise HTTPException(status_code=404, detail=f"Tool {name} not found")
    
    try:
        result = TOOL_FUNCTIONS[name](**parameters)
        return {"tool": name, "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)