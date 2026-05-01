"""Agent state definitions."""
from __future__ import annotations

from typing import Annotated, Any, Optional, Union
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class MessageState(TypedDict):
    """Messages and tool results."""
    messages: Annotated[list, add_messages]
    tool_messages: list[Any]
    last_summary: str


class TaskState(TypedDict):
    """Task and todo management."""
    task_items: list[dict]
    pending_tasks: list[dict]
    todo_items: list[dict]
    rounds_since_todo_update: int
    task_type: str
    matched_skill: Optional[dict]
    should_create_skill: bool


class MemoryState(TypedDict):
    """Memory layer injected into system prompt."""
    static_memory: str
    session_context: str
    episodic_memory: str
    has_compacted: bool
    recent_files: list[str]
    should_update_memory: bool


class TeamState(TypedDict):
    """Team collaboration."""
    teammates: dict[str, dict]
    completed_results: list[dict]
    inbox_notifications: list[dict]
    pending_requests: list[dict]
    pending_background_tasks: list[dict]
    completed_notifications: list[dict]


class ExecutionState(TypedDict):
    """Execution context and control."""
    evaluation_score: float
    execution_steps: list[str]
    error_recovery_count: int
    max_output_recovery_count: int
    compact_requested: bool
    compact_focus: Optional[str]
    worktree_events: list[dict]
    active_worktrees: list[str]


class ConfigState(TypedDict):
    """Runtime configuration."""
    mode: str
    permission_rules: list
    consecutive_denials: int
    task_count: int
    scheduled_notifications: list[str]
    active_schedules: list[dict]


class AgentState(
    MessageState,
    TaskState,
    MemoryState,
    TeamState,
    ExecutionState,
    ConfigState,
):
    """Complete agent state."""
    pass


class TeammateState(TypedDict):
    """Sub-agent state."""
    messages: Annotated[list, add_messages]
    name: str
    role: str
    task: str
    result: str


class TodoItem(TypedDict):
    """Todo item."""
    content: str
    status: str
    activeForm: Optional[str] = None


def create_initial_state(
    messages: list = None,
    mode: str = "default",
    task_type: str = "",
) -> AgentState:
    """Create initial state with defaults."""
    return {
        "messages": messages or [],
        "tool_messages": [],
        "last_summary": "",
        "task_items": [],
        "pending_tasks": [],
        "todo_items": [],
        "rounds_since_todo_update": 0,
        "task_type": task_type,
        "matched_skill": None,
        "should_create_skill": False,
        "static_memory": "",
        "session_context": "",
        "episodic_memory": "",
        "has_compacted": False,
        "recent_files": [],
        "should_update_memory": False,
        "teammates": {},
        "completed_results": [],
        "inbox_notifications": [],
        "pending_requests": [],
        "pending_background_tasks": [],
        "completed_notifications": [],
        "evaluation_score": 0.0,
        "execution_steps": [],
        "error_recovery_count": 0,
        "max_output_recovery_count": 3,
        "compact_requested": False,
        "compact_focus": None,
        "worktree_events": [],
        "active_worktrees": [],
        "mode": mode,
        "permission_rules": [],
        "consecutive_denials": 0,
        "task_count": 0,
        "scheduled_notifications": [],
        "active_schedules": [],
    }


def get_message_state(state: AgentState) -> MessageState:
    return {
        "messages": state["messages"],
        "tool_messages": state["tool_messages"],
        "last_summary": state["last_summary"],
    }


def get_task_state(state: AgentState) -> TaskState:
    return {
        "task_items": state["task_items"],
        "pending_tasks": state["pending_tasks"],
        "todo_items": state["todo_items"],
        "rounds_since_todo_update": state["rounds_since_todo_update"],
        "task_type": state["task_type"],
        "matched_skill": state["matched_skill"],
        "should_create_skill": state["should_create_skill"],
    }


def get_memory_state(state: AgentState) -> MemoryState:
    return {
        "static_memory": state["static_memory"],
        "session_context": state["session_context"],
        "episodic_memory": state["episodic_memory"],
        "has_compacted": state["has_compacted"],
        "recent_files": state["recent_files"],
        "should_update_memory": state["should_update_memory"],
    }


# Type aliases
MinimalState = Union[MessageState, TaskState]
CheckpointerState = MessageState
