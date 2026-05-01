"""LangGraph-based agent implementation."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from minicode.agent.state import AgentState, create_initial_state
from minicode.tools.registry import ALL_TOOLS
from minicode.services.model_provider import create_provider
from minicode.utils.system_prompt import get_system_prompt
from minicode.tools.permission_tools import check_permission

try:
    from minicode.agent.memory import get_memory_layer
    HAS_MEMORY_LAYER = True
except ImportError:
    HAS_MEMORY_LAYER = False


WORKDIR = Path.cwd()

TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}
TOOL_NODE = ToolNode(ALL_TOOLS, handle_tool_errors=True)


def refresh_mcp_tools() -> int:
    """Refresh dynamic MCP tools."""
    global TOOL_MAP, TOOL_NODE

    try:
        from minicode.tools.mcp_tools import get_mcp_client
        from langchain_core.tools import StructuredTool

        client = get_mcp_client()

        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(client.refresh())
        except RuntimeError:
            pass

        new_tools = client.get_tools()

        if new_tools:
            local_tool_names = {t.name for t in ALL_TOOLS}

            processed_mcp_tools = []
            for t in new_tools:
                if t.name in local_tool_names:
                    new_tool = StructuredTool(
                        name=f"mcp_{t.name}",
                        description=t.description,
                        args_schema=t.args_schema,
                        coroutine=t.coroutine,
                    )
                    processed_mcp_tools.append(new_tool)
                else:
                    processed_mcp_tools.append(t)

            renamed_count = len([t for t in processed_mcp_tools if t.name.startswith("mcp_")])
            print(f"[MCP] Added {len(processed_mcp_tools)} tools ({renamed_count} renamed)")

            TOOL_MAP = {t.name: t for t in ALL_TOOLS}
            for t in processed_mcp_tools:
                TOOL_MAP[t.name] = t

            TOOL_NODE = ToolNode(ALL_TOOLS + processed_mcp_tools, handle_tool_errors=True)
            reset_for_mcp_refresh()

            return len(processed_mcp_tools)
    except Exception as e:
        print(f"[MCP] Failed to refresh tools: {e}")
        import traceback
        traceback.print_exc()

    return 0


def get_all_tools() -> list[BaseTool]:
    """Get all available tools including MCP dynamic tools."""
    try:
        from minicode.tools.mcp_tools import get_mcp_client
        client = get_mcp_client()
        mcp_tools = client.get_tools()
        return ALL_TOOLS + mcp_tools
    except Exception:
        return ALL_TOOLS


def get_tool_map() -> dict[str, BaseTool]:
    return TOOL_MAP


def reset_for_mcp_refresh():
    """Reset AgentGraphBuilder after MCP tool refresh."""
    builder = AgentGraphBuilder.get_instance()
    if builder:
        builder.reset()


class BashSecurityValidator:
    """Bash command security validation."""
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


bash_validator = BashSecurityValidator()


class AgentGraphBuilder:
    """Agent Graph Builder."""
    _instance: Optional["AgentGraphBuilder"] = None

    def __init__(self):
        self._model = None
        self._model_with_tools = None
        AgentGraphBuilder._instance = self

    @classmethod
    def get_instance(cls) -> "AgentGraphBuilder":
        return cls._instance

    @property
    def model(self):
        if self._model is None:
            self._model = create_provider()
        return self._model

    @property
    def model_with_tools(self):
        if self._model_with_tools is None:
            all_tools = get_all_tools()
            self._model_with_tools = self.model.bind_tools(all_tools)
        return self._model_with_tools

    def reset(self):
        """Reset model instance."""
        self._model = None
        self._model_with_tools = None


def _build_system_message(state: dict) -> str:
    """Build system prompt with memory injection."""
    base_prompt = get_system_prompt(WORKDIR)

    if HAS_MEMORY_LAYER:
        parts = [base_prompt]

        static = state.get("static_memory", "")
        if static:
            parts.append(static)

        session = state.get("session_context", "")
        if session:
            parts.append(session)

        episodic = state.get("episodic_memory", "")
        if episodic:
            parts.append(episodic)

        return "\n\n".join(parts)

    return base_prompt


def call_model(state: AgentState) -> dict:
    """LLM inference node."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}

    system_msg = SystemMessage(content=_build_system_message(state))
    messages_with_system = [system_msg] + list(messages)

    builder = AgentGraphBuilder.get_instance()
    if not builder:
        builder = AgentGraphBuilder()
        AgentGraphBuilder._instance = builder

    response = builder.model_with_tools.invoke(messages_with_system)
    return {"messages": [response]}


def execute_tools(state: AgentState) -> dict:
    """Tool execution node using LangGraph ToolNode."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [], "tool_messages": []}

    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": [], "tool_messages": []}

    tool_calls = last_message.tool_calls

    # Check permissions for bash_tool
    for tc in tool_calls:
        if tc["name"] == "bash_tool":
            command = tc.get("args", {}).get("command", "")
            allowed, reason = check_permission(command, "bash_tool")
            if not allowed:
                tc["args"]["_permission_denied"] = reason

    # Use ToolNode to handle all tools (async/sync)
    result = TOOL_NODE.invoke({"messages": [last_message]})

    # Post-process results to add permission denied messages
    tool_messages = result.get("tool_messages", [])

    for tc in tool_calls:
        if tc["name"] == "bash_tool" and "_permission_denied" in tc.get("args", {}):
            # Find corresponding tool message or create one
            tc_id = tc.get("id", "")
            reason = tc["args"]["_permission_denied"]

            # Check if we already got a result for this call
            existing = [tm for tm in tool_messages if tm.tool_call_id == tc_id]
            if not existing:
                tool_messages.append(
                    ToolMessage(content=f"[Permission Denied]: {reason}", tool_call_id=tc_id)
                )

    return {"messages": tool_messages, "tool_messages": tool_messages}


def should_continue(state: AgentState) -> Literal["tools", END]:
    """Check if should continue with tool calls or end."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


def create_agent_graph(
    use_checkpoint: bool = False,
):
    """Create agent graph."""
    builder = AgentGraphBuilder()
    AgentGraphBuilder._instance = builder

    refresh_mcp_tools()

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver() if use_checkpoint else None

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=None,
    )


create_agent_graph_stream = create_agent_graph
create_agent_graph_async = create_agent_graph