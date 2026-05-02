"""LangGraph-based agent implementation."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal, Optional

from langchain_core.messages import (
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from minicode.agent.state import AgentState
from minicode.tools.hook_tools import get_hook_manager
from minicode.tools.permission_hook import register_permission_hooks
from minicode.tools.registry import ALL_TOOLS
from minicode.services.model_provider import create_provider
from minicode.utils.system_prompt import get_system_prompt

try:
    from minicode.agent.memory import get_memory_layer
    HAS_MEMORY_LAYER = True
except ImportError:
    HAS_MEMORY_LAYER = False


WORKDIR = Path.cwd()

TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}
TOOL_NODE = ToolNode(ALL_TOOLS, handle_tool_errors=True)


def _init_hooks() -> None:
    """Initialize hooks system."""
    register_permission_hooks()


def _on_mcp_tools_changed(tools: list) -> None:
    """MCP tools changed callback - update TOOL_NODE."""
    global TOOL_NODE, TOOL_MAP

    from minicode.tools.registry import ALL_TOOLS

    if tools:
        # 合并 MCP 工具 (冲突时重命名)
        local_names = {t.name for t in ALL_TOOLS}
        processed = []
        for t in tools:
            if t.name in local_names:
                from langchain_core.tools import StructuredTool
                new_t = StructuredTool(
                    name=f"mcp_{t.name}",
                    description=t.description,
                    args_schema=t.args_schema,
                    coroutine=t.coroutine,
                )
                processed.append(new_t)
            else:
                processed.append(t)

        renamed = len([t for t in processed if t.name.startswith("mcp_")])
        print(f"[MCP] Tools updated: {len(processed)} tools ({renamed} renamed)")

        all_tools = ALL_TOOLS + processed
        TOOL_MAP = {t.name: t for t in all_tools}
        TOOL_NODE = ToolNode(all_tools, handle_tool_errors=True)
    else:
        TOOL_MAP = {t.name: t for t in ALL_TOOLS}
        TOOL_NODE = ToolNode(ALL_TOOLS, handle_tool_errors=True)
        print("[MCP] Tools cleared")

    reset_for_mcp_refresh()


def refresh_mcp_tools() -> int:
    """Refresh dynamic MCP tools (手动刷新接口)."""
    global TOOL_NODE, TOOL_MAP

    try:
        from minicode.tools.mcp_tools import get_mcp_provider

        provider = get_mcp_provider()

        # 订阅变更事件 (如果尚未订阅)
        if _on_mcp_tools_changed not in provider._subscribers:
            provider.subscribe(_on_mcp_tools_changed)

        # 触发刷新
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(provider.refresh())
        except RuntimeError:
            pass

        tools = provider.tools
        count = len(tools)
        if count > 0:
            _on_mcp_tools_changed(tools)
        return count

    except Exception as e:
        print(f"[MCP] Failed to refresh tools: {e}")
        import traceback
        traceback.print_exc()

    return 0


def get_all_tools() -> list[BaseTool]:
    """Get all available tools including MCP dynamic tools."""
    try:
        from minicode.tools.mcp_tools import get_mcp_provider
        provider = get_mcp_provider()
        return ALL_TOOLS + provider.tools
    except Exception:
        return ALL_TOOLS


def get_tool_map() -> dict[str, BaseTool]:
    return TOOL_MAP


def reset_for_mcp_refresh():
    """Reset AgentGraphBuilder after MCP tool refresh."""
    builder = AgentGraphBuilder.get_instance()
    if builder:
        builder.reset()


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


def _build_system_message(state: AgentState) -> str:
    """Build system prompt with memory injection."""
    base_prompt = get_system_prompt(WORKDIR)

    if HAS_MEMORY_LAYER:
        memory = state.get("memory")
        if memory:
            parts = [base_prompt]
            if memory.get("static_memory"):
                parts.append(memory["static_memory"])
            if memory.get("session_context"):
                parts.append(memory["session_context"])
            if memory.get("episodic_memory"):
                parts.append(memory["episodic_memory"])
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
    hook_manager = get_hook_manager()

    # PreToolUse: run hooks before execution
    blocked_tool_calls = []
    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_input = tc.get("args", {}) or {}
        context = {"tool_name": tool_name, "tool_input": tool_input}

        # Run Python hooks first
        hook_result = hook_manager.run_python_hooks("PreToolUse", context)

        if hook_result.get("blocked"):
            blocked_tool_calls.append({
                "id": tc.get("id", ""),
                "name": tool_name,
                "reason": hook_result.get("block_reason", "Blocked by hook"),
            })
        elif hook_result.get("updated_input"):
            # Update tool input if hook modified it
            tc["args"] = hook_result["updated_input"]

    # Build result
    tool_messages = []

    # Add blocked messages
    for blocked in blocked_tool_calls:
        tool_messages.append(
            ToolMessage(
                content=f"[Permission Denied]: {blocked['reason']}",
                tool_call_id=blocked["id"]
            )
        )

    # Execute allowed tools
    allowed_tool_calls = [
        tc for tc in tool_calls
        if tc.get("id", "") not in [b["id"] for b in blocked_tool_calls]
    ]

    if allowed_tool_calls:
        try:
            from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME
            from langgraph.runtime import DEFAULT_RUNTIME

            runtime_config = {
                CONF: {
                    CONFIG_KEY_RUNTIME: DEFAULT_RUNTIME
                }
            }

            # Create a modified message with only allowed tool calls
            allowed_message = last_message.model_copy()
            allowed_message.tool_calls = allowed_tool_calls

            result = TOOL_NODE.invoke({"messages": [allowed_message]}, runtime_config)
            all_messages = result.get("messages", [])
            tool_messages.extend([m for m in all_messages if hasattr(m, "tool_call_id")])
        except Exception as e:
            error_msg = f"[Tool Execution Error] {type(e).__name__}: {e}"
            tool_messages.extend([
                ToolMessage(content=error_msg, tool_call_id=tc.get("id", ""))
                for tc in allowed_tool_calls
            ])

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


def _init_mcp_subscription() -> None:
    """Initialize MCP tools subscription."""
    try:
        from minicode.tools.mcp_tools import get_mcp_provider
        provider = get_mcp_provider()
        if _on_mcp_tools_changed not in provider._subscribers:
            provider.subscribe(_on_mcp_tools_changed)
    except Exception as e:
        print(f"[MCP] Subscription init failed: {e}")


def create_agent_graph(
    use_checkpoint: bool = False,
):
    """Create agent graph."""
    builder = AgentGraphBuilder()
    AgentGraphBuilder._instance = builder

    _init_hooks()  # Initialize hooks (permission checks, etc.)
    _init_mcp_subscription()  # Subscribe to MCP tool changes
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