# sebsocket_backend
Project Overview: Collections ROI Copilot
The Collections ROI Copilot is an AI-powered conversational assistant designed to integrate seamlessly with a Collections ROI Calculator. It serves as a specialized data analyst and sales consultant, allowing users to interact with complex financial models through natural language instead of manual data entry.

Key Features
Real-Time Screen Awareness: The Copilot "sees" the calculator state in real-time via WebSocket synchronization, allowing it to provide context-aware insights based on current metrics like Peak Utilization and Recovery Rates.

Conversational Calculator Control: Users can modify calculator inputs through chat (e.g., "Reduce human salary by 15%"), and the assistant performs the math and updates the UI silently.

Multi-Agent Intelligence: Powered by a LangGraph multi-agent system, the brain coordinates specialized agents for intent routing, mathematical analysis, scenario running, and document generation.

Proactive Operational Insights: The system monitors live data and triggers "Proactive Suggestions" when it detects operational bottlenecks, such as peak utilization exceeding 120%.

Automated Document Generation: Instantly creates professional, board-ready executive summaries in PDF format based on real-time calculator data.

Contextual Interactivity: Maintains a "Golden Thread" of conversation history, enabling follow-up questions like "Why is this lower than Bucket X?" by referencing previous messages.

System Architecture
The architecture is built on three main pillars:

Frontend (React/Tailwind): A polished chat interface that manages local message history and listens for WebSocket state triggers.

API (FastAPI): A high-performance bridge that handles RESTful chat requests and coordinates real-time WebSocket communication for proactive alerts.

Brain (LangGraph/Gemini): A stateful graph that routes user intents to specialized AI nodes, ensuring precise data manipulation and analytical reasoning.

The "Golden Thread" Integration
To maintain a seamless user experience, the project implements a strict data flow across the stack:

Stateful Memory: Every interaction is stored in the CopilotState, allowing agents to recall previous context.

Dual-Output Logic: The calculator_controller generates a technical JSON update for the system and a human-readable summary for the user, preventing raw technical code from appearing in the chat.

Artifact Mapping: Generated documents are stored in pending_artifacts and mapped to the API response, allowing the frontend to render download buttons automatically.

Getting Started
Backend Setup
Install dependencies: pip install fastapi uvicorn langgraph langchain-google-genai fpdf

Set your API Key: export GOOGLE_API_KEY='your_key_here'

Run the API: python copilot_api.py

Frontend Setup
Install dependencies: npm install lucide-react

Ensure the wsUrl and apiUrl in CopilotChat.jsx match your backend configuration.

Launch the application: npm start

Example Test Cases
Modify: "Increase agent count to 50 and reduce rent by 10%."

Analyze: "Why is my peak utilization so high?"

Scenario: "What happens if I switch to AI augmentation for Bucket 1?"

Generate: "Prepare an executive summary for my CFO."
