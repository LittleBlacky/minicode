"""Agent module - Core agent implementation"""
from minicode.agent.state import AgentState, TeammateState, TodoItem
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
from minicode.agent.self_improve import DreamConsolidator, SelfImprovingAgent
from minicode.agent.autonomous import AutonomousAgent, TeammateManager, IdleConfig

__all__ = [
    # Core
    "AgentState",
    "AgentRunner",
    "run_interactive",
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
    "DreamConsolidator",
    "SelfImprovingAgent",
    # Autonomous
    "AutonomousAgent",
    "TeammateManager",
    "IdleConfig",
    # State
    "TeammateState",
    "TodoItem",
]