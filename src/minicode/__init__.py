"""MiniCode - A Claude-style coding agent."""

__version__ = "0.1.0"

from minicode.agent.runner import AgentRunner
from minicode.agent.state import AgentState, create_initial_state
from minicode.services.config import get_config_manager

__all__ = [
    "AgentRunner",
    "AgentState",
    "create_initial_state",
    "get_config_manager",
    "__version__",
]