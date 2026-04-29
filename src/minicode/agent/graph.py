"""Main agent implementation using LangGraph - Fixed version."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool, BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from minicode.agent.state import AgentState
from minicode.tools.registry import ALL_TOOLS
from minicode.services.model_provider import create_provider
from minicode.utils.system_prompt import get_system_prompt
from minicode.tools.hook_tools import get_hook_manager
from minicode.tools.permission_tools import bash_validator, check_permission


WORKDIR = Path.cwd()


class BashSecurityValidator:
    """Validate bash commands for obviously dangerous patterns."""

    VALIDATORS = [
        ("shell_metachar", r"[;&|`$]"),
        ("sudo", r"\bsudo\b"),
        ("rm_rf", r"\brm\s+(-[a-zA-Z]*)?r"),
        ("cmd_substitution", r"\$\("),
        ("ifs_injection", r"\bIFS\s*="),
    ]

    def validate(self, command: str) -> list[tuple[str, str]]:
        failures = []
        for name, pattern in self.VALIDATORS:
            if re.search(pattern, command):
                failures.append((name, pattern))
        return failures

    def describe_failures(self, command: str) -> str:
        failures = self.validate(command)
        if not failures:
            return "No issues detected"
        parts = [name for name, _ in failures]
        return "Security flags: " + ", ".join(parts)


# Build tool map for fast lookup
TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}
TOOL_NODE = ToolNode(ALL_TOOLS, handle_tool_errors=True)


def _safe_path(path_str: str) -> Path:
    """Resolve path relative to workspace."""
    path = (WORKDIR / path_str).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return path


class AgentGraphBuilder:
    """Builder for the agent graph with model and tools bound."""

    def __init__(
        self,
        model_provider: str = "anthropic",
        model_name: str = "claude-sonnet-4-7",
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self._model = None
        self._model_with_tools = None

    @property
    def model(self):
        """Get or create the model (lazy initialization)."""
        if self._model is None:
            self._model = create_provider(
                provider=self.model_provider,
                model=self.model_name,
            ).client
        return self._model

    @property
    def model_with_tools(self):
        """Get model with tools bound (lazy initialization)."""
        if self._model_with_tools is None:
            self._model_with_tools = self.model.bind_tools(ALL_TOOLS)
        return self._model_with_tools


_graph_builder: Optional[AgentGraphBuilder] = None


def get_graph_builder(
    model_provider: str = "anthropic",
    model_name: str = "claude-sonnet-4-7",
) -> AgentGraphBuilder:
    """Get or create global graph builder instance."""
    global _graph_builder
    if _graph_builder is None:
        _graph_builder = AgentGraphBuilder(model_provider, model_name)
    return _graph_builder


def call_model(state: AgentState) -> dict:
    """Call the LLM with current messages and tools bound."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}

    system_prompt = get_system_prompt(WORKDIR)
    messages_with_system = [SystemMessage(content=system_prompt)] + list(messages)

    builder = get_graph_builder(
        model_provider=os.environ.get("MODEL_PROVIDER", "anthropic"),
        model_name=os.environ.get("MODEL_NAME", "claude-sonnet-4-7"),
    )

    response = builder.model_with_tools.invoke(messages_with_system)
    return {"messages": [response]}


def execute_tools(state: AgentState) -> dict:
    """Execute tools from the last AI message."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [], "tool_messages": []}

    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": [], "tool_messages": []}

    tool_calls = last_message.tool_calls
    tool_messages = []
    hook_manager = get_hook_manager()

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc.get("args", {}) or {}

        hook_context = {
            "tool_name": tool_name,
            "tool_input": tool_args,
        }

        # Run pre-tool hooks
        pre_hook_result = hook_manager.run_hooks("PreToolUse", hook_context)
        if pre_hook_result["blocked"]:
            content = f"[Blocked]: {pre_hook_result['block_reason']}"
            tool_messages.append(
                ToolMessage(content=content, tool_call_id=tc["id"])
            )
            continue

        # Check bash permissions
        if tool_name == "bash_tool":
            command = tool_args.get("command", "")
            allowed, reason = check_permission(command, "bash_tool")
            if not allowed:
                content = f"[Permission Denied]: {reason}"
                tool_messages.append(
                    ToolMessage(content=content, tool_call_id=tc["id"])
                )
                continue

            # Update args if hook modified them
            if pre_hook_result.get("updated_input"):
                tool_args = pre_hook_result["updated_input"]

        # Execute the tool
        if tool_name in TOOL_MAP:
            try:
                result = TOOL_NODE.invoke({"messages": [last_message]})
                if "messages" in result:
                    for msg in result["messages"]:
                        if isinstance(msg, ToolMessage):
                            tool_messages.append(msg)
            except Exception as e:
                tool_messages.append(
                    ToolMessage(content=f"[Error] {e}", tool_call_id=tc["id"])
                )
        else:
            tool_messages.append(
                ToolMessage(content=f"[Error] Tool not found: {tool_name}", tool_call_id=tc["id"])
            )

        # Run post-tool hooks
        hook_context["tool_output"] = result if 'result' in dir() else None
        post_hook_result = hook_manager.run_hooks("PostToolUse", hook_context)

    return {"messages": tool_messages, "tool_messages": tool_messages}


def should_continue(state: AgentState) -> Literal["tools", END]:
    """Determine if agent should continue or end."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


def create_agent_graph(
    model_provider: str = "anthropic",
    model_name: str = "claude-sonnet-4-7",
    use_checkpoint: bool = False,
):
    """Create the main agent graph (non-streaming)."""
    global _graph_builder
    _graph_builder = AgentGraphBuilder(model_provider, model_name)

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver() if use_checkpoint else None

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=None,  # Allow tools to run automatically
    )


def create_agent_graph_stream(
    model_provider: str = "anthropic",
    model_name: str = "claude-sonnet-4-7",
    use_checkpoint: bool = False,
):
    """Create the streaming agent graph with proper streaming."""
    global _graph_builder
    _graph_builder = AgentGraphBuilder(model_provider, model_name)

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver() if use_checkpoint else None

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=None,
    )


# Async streaming support
async def call_model_stream_async(state: AgentState) -> dict:
    """Async stream model responses and yield chunks."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}

    system_prompt = get_system_prompt(WORKDIR)
    messages_with_system = [SystemMessage(content=system_prompt)] + list(messages)

    builder = get_graph_builder(
        model_provider=os.environ.get("MODEL_PROVIDER", "anthropic"),
        model_name=os.environ.get("MODEL_NAME", "claude-sonnet-4-7"),
    )

    full_response = None
    async for chunk in builder.model_with_tools.astream(messages_with_system):
        if isinstance(chunk, AIMessageChunk):
            if full_response is None:
                full_response = chunk
            else:
                full_response = full_response + chunk

    if full_response is None:
        full_response = AIMessage(content="")

    return {"messages": [full_response]}


def create_agent_graph_async(
    model_provider: str = "anthropic",
    model_name: str = "claude-sonnet-4-7",
    use_checkpoint: bool = False,
):
    """Create the async streaming agent graph."""
    global _graph_builder
    _graph_builder = AgentGraphBuilder(model_provider, model_name)

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model_stream_async)
    workflow.add_node("tools", execute_tools)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver() if use_checkpoint else None

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=None,
    )
