from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import logging

from copilot_state import CopilotState
from copilot_graph import app as graph_app

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Collections ROI Copilot API")

# Apply CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and Response Models
class ChatRequest(BaseModel):
    message: str
    calculator_state: Dict[str, Any]
    session_id: str
    user_id: str
    # Add history to the request model to receive previous messages from the frontend
    history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    message: str
    calculator_updates: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []

# Manager for handling WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/copilot/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message and state using LangGraph's ainvoke,
    returning the model's reply and any calculator actions/artifacts.
    """
    try:
        # Construct the full message list including history
        all_messages = request.history if request.history else []
        all_messages.append({"role": "user", "content": request.message})

        initial_state: CopilotState = {
            "messages": all_messages,  # Now includes the full history for interactivity
            "calculator_state": request.calculator_state,
            "session_id": request.session_id,
            "user_id": request.user_id,
            "current_metrics": {},
            "active_warnings": [],
            "recent_changes": [],
            "agent_history": [],
            "pending_actions": [],
            "user_profile": {}
        }

        final_state = await graph_app.ainvoke(initial_state)

        # Retrieve the assistant's latest message
        assistant_message = final_state["messages"][-1]["content"]

        return ChatResponse(
            message=str(assistant_message),
            calculator_updates=final_state.get("pending_actions", []),
            artifacts=final_state.get("pending_artifacts", [])
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/copilot/sync")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handle real-time synchronization and send proactive suggestions
    based on incoming calculator state.
    """
    await manager.connect(websocket)
    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                await manager.send_personal_message(json.dumps({"type": "ERROR", "error": "Invalid JSON"}), websocket)
                continue

            if data.get("type") == "STATE_UPDATE":
                calc_state = data.get("state", {})
                peak_util = calc_state.get("peakUtilization", 0)

                if isinstance(peak_util, (int, float)) and peak_util > 120:
                    suggestion = {
                        "type": "PROACTIVE_SUGGESTION",
                        "content": f"I notice your peak utilization is at {peak_util:.0f}%. You are missing collection opportunities. Want me to model an AI augmentation scenario?"
                    }
                    await manager.send_personal_message(json.dumps(suggestion), websocket)
                else:
                    await manager.send_personal_message(json.dumps({"type": "ACK"}), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
