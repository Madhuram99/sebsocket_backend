import json
import os

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from copilot_state import CopilotState

# Initialize LLM with optimized settings for logic
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=os.environ.get("GOOGLE_API_KEY")  # Ensure this line is exactly like this
)

# --- Enhanced Agent Logic ---

async def intent_router(state: CopilotState) -> CopilotState:
    """
    Uses LLM to classify user intent accurately based on conversation context
    """
    messages = state.get("messages", [])
    last_message = messages[-1]["content"] if messages else ""

    router_prompt = (
        "Classify the user's intent into one of these categories: "
        "'greeting' (saying hi), 'modify' (changing numbers/sliders), "
        "'explain' (asking why or how), 'scenario' (what-if analysis), "
        "or 'generate' (creating reports). Return ONLY the category name."
    )

    response = await llm.ainvoke([
        SystemMessage(content=router_prompt),
        HumanMessage(content=last_message)
    ])

    state["current_intent"] = response.content.strip().lower()
    state["agent_history"].append("router")
    return state

async def greeting_handler(state: CopilotState) -> CopilotState:
    """
    Handles greetings with contextual awareness
    """
    response = await llm.ainvoke([
        SystemMessage(content="You are a helpful Collections ROI Copilot. Greet the user and briefly mention you can see their current calculator state."),
        HumanMessage(content=state["messages"][-1]["content"])
    ])
    state["messages"].append({"role": "assistant", "content": response.content})
    state["agent_history"].append("greeting")
    return state

async def analyst_agent(state: CopilotState) -> CopilotState:
    """
    Explains calculations using real-time calculator data
    """
    calc_state = json.dumps(state.get("calculator_state", {}))

    analyst_system_prompt = (
        f"You are a Collections Data Analyst. Use the following real-time calculator state to answer: {calc_state}. "
        "Focus on metrics like Recovery Rate, Profit, and Peak Utilization. "
        "If peak utilization is > 100%, explain that they are missing revenue due to capacity bottlenecks."
    )

    chat_messages = []
    for m in state.get("messages", []):
        if m["role"] == "user":
            chat_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            chat_messages.append(AIMessage(content=m["content"]))

    response = await llm.ainvoke([
        SystemMessage(content=analyst_system_prompt),
        *chat_messages
    ])

    state["messages"].append({"role": "assistant", "content": response.content})
    state["agent_history"].append("analyst")
    return state

async def calculator_controller(state: CopilotState) -> CopilotState:
    """
    Parses user requests to modify the calculator and generates JSON patches
    """
    calc_state = json.dumps(state.get("calculator_state", {}))

    controller_prompt = (
        "You are a system controller. The user wants to change a value in the calculator. "
        f"Current state: {calc_state}. "
        "Output ONLY a raw JSON object of the fields to change. "
        "Example: {'agentCount': 45}. Do not include any text, only the JSON."
    )

    response = await llm.ainvoke([
        SystemMessage(content=controller_prompt),
        HumanMessage(content=state["messages"][-1]["content"])
    ])

    try:
        updates = json.loads(response.content.strip())
        state["pending_actions"].append(updates)
        state["messages"].append({"role": "assistant", "content": "I've updated those values in the calculator for you."})
    except json.JSONDecodeError:
        state["messages"].append({
            "role": "assistant",
            "content": "I understood you wanted to make a change, but I couldn't format the update correctly."
        })

    state["agent_history"].append("controller")
    return state

async def scenario_runner(state: CopilotState) -> CopilotState:
    """
    Handles 'What-if' scenarios by comparing current state with proposed changes
    """
    calc_state = json.dumps(state.get("calculator_state", {}))
    scenario_prompt = (
        f"Analyze this 'what-if' scenario using the current data: {calc_state}. "
        "Compare the 'Current' vs 'Proposed' impact on Profit and ROI."
    )

    response = await llm.ainvoke([
        SystemMessage(content=scenario_prompt),
        HumanMessage(content=state["messages"][-1]["content"])
    ])

    state["messages"].append({"role": "assistant", "content": response.content})
    state["agent_history"].append("scenario_runner")
    return state

# --- Graph Definition ---

workflow = StateGraph(CopilotState)
workflow.add_node("intent_router", intent_router)
workflow.add_node("greeting_handler", greeting_handler)
workflow.add_node("analyst_agent", analyst_agent)
workflow.add_node("calculator_controller", calculator_controller)
workflow.add_node("scenario_runner", scenario_runner)

def route_next(state: CopilotState):
    intent = state.get("current_intent", "")
    if intent == "greeting":
        return "greeting_handler"
    elif intent == "modify":
        return "calculator_controller"
    elif intent == "scenario":
        return "scenario_runner"
    else:
        return "analyst_agent"

workflow.set_entry_point("intent_router")
workflow.add_conditional_edges(
    "intent_router",
    route_next,
    {
        "greeting_handler": "greeting_handler",
        "calculator_controller": "calculator_controller",
        "scenario_runner": "scenario_runner",
        "analyst_agent": "analyst_agent"
    }
)
workflow.add_edge("greeting_handler", END)
workflow.add_edge("analyst_agent", END)
workflow.add_edge("calculator_controller", END)
workflow.add_edge("scenario_runner", END)

app = workflow.compile()
