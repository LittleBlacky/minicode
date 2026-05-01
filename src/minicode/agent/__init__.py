"""Agent module - Core agent implementation"""
from minicode.agent.state import (
    AgentState,
    TeammateState,
    TodoItem,
    create_initial_state,
    MessageState,
    TaskState,
    MemoryState,
    TeamState,
    ExecutionState,
    ConfigState,
    # Helper functions
    get_core,
    get_memory,
    get_tasks,
    get_team,
    get_execution,
)
from minicode.agent.graph import create_agent_graph
from minicode.agent.runner import AgentRunner, run_interactive
from minicode.agent.session import (
    SessionManager,
    SessionConfig,
    SessionMetrics,
    ContextOverflowError,
    get_session_manager,
    reset_session_manager,
)
from minicode.agent.subagent import SubAgent, SubAgentPool
from minicode.agent.error_recovery import ErrorRecovery, RecoveryManager, ErrorType, RecoveryResult
from minicode.agent.self_improve import (
    SelfImprovementEngine,
    TaskRecord,
    ImprovementTrigger,
    get_self_improvement,
    reset_self_improvement,
)
from minicode.agent.autonomous import AutonomousAgent, TeammateManager, IdleConfig
from minicode.agent.memory import MemoryLayer, MemoryIndex, MemoryEntry, get_memory_layer

__all__ = [
    # Core
    "AgentState",
    "AgentRunner",
    "run_interactive",
    # State types
    "MessageState",
    "TaskState",
    "MemoryState",
    "TeamState",
    "ExecutionState",
    "ConfigState",
    "create_initial_state",
    # Helper functions
    "get_core",
    "get_memory",
    "get_tasks",
    "get_team",
    "get_execution",
    # Session management
    "SessionManager",
    "SessionConfig",
    "SessionMetrics",
    "ContextOverflowError",
    "get_session_manager",
    "reset_session_manager",
    # SubAgent
    "SubAgent",
    "SubAgentPool",
    # Error recovery
    "ErrorRecovery",
    "RecoveryManager",
    "ErrorType",
    "RecoveryResult",
    # Self improvement
    "SelfImprovementEngine",
    "TaskRecord",
    "ImprovementTrigger",
    "get_self_improvement",
    "reset_self_improvement",
    # Autonomous
    "AutonomousAgent",
    "TeammateManager",
    "IdleConfig",
    # State
    "TeammateState",
    "TodoItem",
    # Memory layer
    "MemoryLayer",
    "MemoryIndex",
    "MemoryEntry",
    "get_memory_layer",
]