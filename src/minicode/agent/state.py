"""Agent state definitions with Composition pattern."""
from __future__ import annotations

from typing import Annotated, Any, Optional, Union
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# =============================================================================
# Core State (必需) - 消息和基础配置
# =============================================================================

class MessageState(TypedDict):
    """Messages and tool results - Graph 核心"""
    messages: Annotated[list, add_messages]
    tool_messages: list
    last_summary: str


class ConfigState(TypedDict):
    """Runtime configuration - 必需配置项"""
    mode: str
    task_count: int
    permission_rules: list
    consecutive_denials: int


# =============================================================================
# Optional Modules - 按需启用
# =============================================================================

class TaskState(TypedDict):
    """Task and todo management - 任务分解时使用"""
    task_items: list[dict]
    pending_tasks: list[dict]
    todo_items: list[dict]
    rounds_since_todo_update: int
    task_type: str
    matched_skill: Optional[dict]
    should_create_skill: bool


class MemoryState(TypedDict):
    """Memory layer injected into system prompt - 记忆系统使用"""
    static_memory: str
    session_context: str
    episodic_memory: str
    has_compacted: bool
    recent_files: list[str]
    should_update_memory: bool


class TeamState(TypedDict):
    """Team collaboration - 多 Agent 协作时使用"""
    teammates: dict[str, dict]
    completed_results: list[dict]
    inbox_notifications: list[dict]
    pending_requests: list[dict]
    pending_background_tasks: list[dict]
    completed_notifications: list[dict]


class ExecutionState(TypedDict):
    """Execution context and control - 错误恢复时使用"""
    evaluation_score: float
    execution_steps: list[str]
    error_recovery_count: int
    max_output_recovery_count: int
    compact_requested: bool
    compact_focus: Optional[str]
    worktree_events: list[dict]
    active_worktrees: list[str]


# =============================================================================
# Combined Agent State (使用组合模式)
# =============================================================================

class AgentState(TypedDict):
    """Complete agent state using Composition pattern.

    结构:
        AgentState.core: CoreState (必需)
        AgentState.tasks: TaskState (可选, 默认 None)
        AgentState.memory: MemoryState (可选, 默认 None)
        AgentState.team: TeamState (可选, 默认 None)
        AgentState.execution: ExecutionState (可选, 默认 None)
    """
    # Core (必需)
    messages: Annotated[list, add_messages]
    tool_messages: list
    last_summary: str
    mode: str
    task_count: int
    permission_rules: list
    consecutive_denials: int

    # Optional modules
    tasks: Optional[TaskState] = None
    memory: Optional[MemoryState] = None
    team: Optional[TeamState] = None
    execution: Optional[ExecutionState] = None


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


# =============================================================================
# Helper Functions - 便捷访问
# =============================================================================

def get_core(state: AgentState) -> dict:
    """获取核心状态 (messages, config)"""
    return {
        "messages": state.get("messages", []),
        "tool_messages": state.get("tool_messages", []),
        "last_summary": state.get("last_summary", ""),
        "mode": state.get("mode", "default"),
        "task_count": state.get("task_count", 0),
        "permission_rules": state.get("permission_rules", []),
        "consecutive_denials": state.get("consecutive_denials", 0),
    }


def get_tasks(state: AgentState) -> TaskState:
    """获取任务状态，懒初始化"""
    if state.get("tasks") is None:
        state["tasks"] = {
            "task_items": [],
            "pending_tasks": [],
            "todo_items": [],
            "rounds_since_todo_update": 0,
            "task_type": "",
            "matched_skill": None,
            "should_create_skill": False,
        }
    return state["tasks"]


def get_memory(state: AgentState) -> MemoryState:
    """获取记忆状态，懒初始化"""
    if state.get("memory") is None:
        state["memory"] = {
            "static_memory": "",
            "session_context": "",
            "episodic_memory": "",
            "has_compacted": False,
            "recent_files": [],
            "should_update_memory": False,
        }
    return state["memory"]


def get_team(state: AgentState) -> TeamState:
    """获取团队状态，懒初始化"""
    if state.get("team") is None:
        state["team"] = {
            "teammates": {},
            "completed_results": [],
            "inbox_notifications": [],
            "pending_requests": [],
            "pending_background_tasks": [],
            "completed_notifications": [],
        }
    return state["team"]


def get_execution(state: AgentState) -> ExecutionState:
    """获取执行状态，懒初始化"""
    if state.get("execution") is None:
        state["execution"] = {
            "evaluation_score": 0.0,
            "execution_steps": [],
            "error_recovery_count": 0,
            "max_output_recovery_count": 3,
            "compact_requested": False,
            "compact_focus": None,
            "worktree_events": [],
            "active_worktrees": [],
        }
    return state["execution"]


# =============================================================================
# Legacy Functions - 保持向后兼容
# =============================================================================

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
        "mode": mode,
        "task_count": 0,
        "permission_rules": [],
        "consecutive_denials": 0,
        # Optional modules 默认 None (懒初始化)
        "tasks": None,
        "memory": None,
        "team": None,
        "execution": None,
    }


def get_message_state(state: AgentState) -> MessageState:
    return {
        "messages": state["messages"],
        "tool_messages": state["tool_messages"],
        "last_summary": state["last_summary"],
    }


def get_task_state(state: AgentState) -> TaskState:
    return get_tasks(state)


def get_memory_state(state: AgentState) -> MemoryState:
    return get_memory(state)


# Type aliases
MinimalState = Union[MessageState, TaskState]
CheckpointerState = MessageState