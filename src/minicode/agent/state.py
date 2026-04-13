"""Agent state definitions for MiniCode."""
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Main state schema for the MiniCode agent.

    Contains all runtime state that flows through the LangGraph.
    """

    # Core message history
    messages: Annotated[list, add_messages]

    # Todo management
    todo_items: list[dict]
    rounds_since_todo_update: int

    # Execution state
    execution_steps: list[str]
    evaluation_score: float
    tool_messages: list[Any]

    # Task system
    task_items: list[dict]
    pending_tasks: list[dict]

    # Memory and learning
    has_compacted: bool
    last_summary: str
    recent_files: list[str]
    compact_requested: bool
    compact_focus: Optional[str]

    # Permission
    mode: str
    permission_rules: list
    consecutive_denials: int

    # Team collaboration
    teammates: dict[str, dict]
    completed_results: list[dict]
    inbox_notifications: list[dict]
    pending_requests: list[dict]

    # Background tasks
    pending_background_tasks: list[dict]
    completed_notifications: list[dict]

    # Worktree
    worktree_events: list[dict]
    active_worktrees: list[str]

    # Self-improvement
    task_type: str
    matched_skill: Optional[dict]
    should_create_skill: bool
    should_update_memory: bool
    task_count: int

    # Error recovery
    max_output_recovery_count: int
    error_recovery_count: int

    # Scheduled tasks
    scheduled_notifications: list[str]
    active_schedules: list[dict]


class TeammateState(TypedDict):
    """State for teammate subgraphs."""
    messages: Annotated[list, add_messages]
    name: str
    role: str
    task: str
    result: str


class TodoItem(TypedDict):
    """Todo item schema."""
    content: str
    status: str
    activeForm: Optional[str] = None
