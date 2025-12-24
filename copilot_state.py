from typing import TypedDict, List, Dict, Optional, Annotated
import operator

class CopilotState(TypedDict):
    """Central state object for the Copilot system"""
    
    # Conversation
    messages: Annotated[List[Dict], operator.add]  # Chat history
    current_intent: Optional[str]  # What user wants: 'explain' | 'modify' | 'scenario' | 'generate'
    conversation_stage: str  # 'greeting' | 'discovery' | 'analysis' | 'decision' | 'action'
    
    # Calculator State (Real-time sync)
    calculator_state: Dict  # Full calculator config + results
    last_calculator_update: str  # ISO timestamp
    active_bucket: str  # 'b_x' | 'b_1' | 'b_2' | 'npa'
    strategy_mode: str  # 'displacement' | 'augmentation'
    
    # Analysis Context
    current_metrics: Dict  # Latest calculated metrics
    active_warnings: List[Dict]  # Current warnings/alerts
    recent_changes: List[Dict]  # Last 5 user actions
    
    # User Context
    user_id: str
    session_id: str
    user_profile: Dict  # Preferences, history, expertise level
    business_context: Dict  # Company, industry, challenges
    
    # Agent Routing
    active_agent: str  # 'router' | 'controller' | 'analyst' | 'sales' | 'generator'
    agent_history: List[str]  # Sequence of agents called
    pending_actions: List[Dict]  # Queued calculator updates
    
    # Memory & Learning
    relevant_memories: List[Dict]  # Retrieved from vector store
    learned_preferences: Dict  # Session-learned user preferences
    
    # Output Generation
    pending_artifacts: List[Dict]  # Reports/docs to generate
    generated_artifacts: List[str]  # IDs of created artifacts
    
    # Tool Results
    tool_outputs: Dict  # Results from tool calls
    
    # Error Handling
    errors: List[str]
    retry_count: int
