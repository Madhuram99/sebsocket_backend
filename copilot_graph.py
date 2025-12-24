import json
import os
import re

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from copilot_state import CopilotState

# Initialize LLM with optimized settings for logic
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,  # Lower temperature for more reliable JSON output
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

def clean_json_response(content: str) -> str:
    """Removes markdown code blocks like ```json ... ``` often added by LLMs"""
    return re.sub(r"```json|```", "", content).strip()

# --- Enhanced Agent Logic ---

async def intent_router(state: CopilotState) -> CopilotState:
    """
    Uses full conversation history to classify user intent accurately
    """
    messages = state.get("messages", [])
    
    router_prompt = (
        "Classify the user's latest intent based on the conversation history. "
        "Categories: 'greeting', 'modify' (changing values/sliders), "
        "'explain' (asking why or how), 'scenario' (what-if analysis). "
        "Return ONLY the category name."
    )

    # Pass history to ensure the router understands follow-up context
    history = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    response = await llm.ainvoke([SystemMessage(content=router_prompt)] + history)

    state["current_intent"] = response.content.strip().lower()
    state["agent_history"].append("router")
    return state

async def calculator_controller(state: CopilotState) -> CopilotState:
    """
    Parses requests to modify the calculator (e.g., 'Reduce human fixed costs by 15%')
    and generates JSON patches.
    """
    calc_state = state.get("calculator_state", {})
    last_user_msg = state["messages"][-1]["content"]

    # Provide the LLM with the actual keys available in the calculator to prevent hallucinations
    available_keys = list(calc_state.keys())
    
    controller_prompt = (
        f"You are a system controller. Current state keys: {available_keys}. "
        "The user wants to change a value. If they use a percentage, calculate the new absolute value "
        "based on the current state provided. "
        "Output ONLY a raw JSON object of the fields to change. "
        "Example: {'hSalary': 15000}. Do not include markdown or text."
    )

    response = await llm.ainvoke([
        SystemMessage(content=controller_prompt),
        HumanMessage(content=f"Current State: {json.dumps(calc_state)}\nRequest: {last_user_msg}")
    ])

    try:
        # Clean and parse the JSON response
        clean_content = clean_json_response(response.content)
        updates = json.loads(clean_content)
        
        # In copilot_state.py, pending_actions is a List[Dict]
        state["pending_actions"] = [updates] 
        state["messages"].append({"role": "assistant", "content": f"I've updated the calculator for you with these changes: {updates}"})
    except (json.JSONDecodeError, Exception):
        state["messages"].append({
            "role": "assistant",
            "content": "I understood you wanted to make a change, but I had trouble calculating the exact update."
        })

    state["agent_history"].append("controller")
    return state

async def analyst_agent(state: CopilotState) -> CopilotState:
    """
    Explains calculations using the full context of the chat history
    """
    calc_state = json.dumps(state.get("calculator_state", {}))
    analyst_system_prompt = (
        f"You are a Collections Data Analyst. Current calculator state: {calc_state}. "
        "Focus on metrics like Recovery Rate, Profit, and Peak Utilization. "
        "Explain changes or current status based on the data provided."
    )

    history = []
    for m in state.get("messages", []):
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))

    response = await llm.ainvoke([SystemMessage(content=analyst_system_prompt)] + history)

    state["messages"].append({"role": "assistant", "content": response.content})
    state["agent_history"].append("analyst")
    return state

# --- Keep existing scenario_runner and greeting_handler logic, ensuring they append to messages ---

async def greeting_handler(state: CopilotState) -> CopilotState:
    response = await llm.ainvoke([
        SystemMessage(content="You are a helpful Collections ROI Copilot. Greet the user and briefly mention you can see their current calculator state."),
        HumanMessage(content=state["messages"][-1]["content"])
    ])
    state["messages"].append({"role": "assistant", "content": response.content})
    state["agent_history"].append("greeting")
    return state

async def scenario_runner(state: CopilotState) -> CopilotState:
    calc_state = json.dumps(state.get("calculator_state", {}))
    scenario_prompt = f"Analyze this 'what-if' scenario using the current data: {calc_state}."
    response = await llm.ainvoke([SystemMessage(content=scenario_prompt), HumanMessage(content=state["messages"][-1]["content"])])
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
    elif intent in ["modify", "action"]:
        return "calculator_controller"
    elif intent == "scenario":
        return "scenario_runner"
    else:
        return "analyst_agent"

workflow.set_entry_point("intent_router")
workflow.add_conditional_edges("intent_router", route_next)
workflow.add_edge("greeting_handler", END)
workflow.add_edge("analyst_agent", END)
workflow.add_edge("calculator_controller", END)
workflow.add_edge("scenario_runner", END)

app = workflow.compile()
