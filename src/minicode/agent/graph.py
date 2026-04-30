"""Main agent implementation using LangGraph - 轻量级核心循环"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

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
from minicode.tools.permission_tools import bash_validator, check_permission

# Import Memory Layer for system prompt injection
try:
    from minicode.agent.memory import get_memory_layer
    HAS_MEMORY_LAYER = True
except ImportError:
    HAS_MEMORY_LAYER = False


WORKDIR = Path.cwd()

# 工具映射用于快速查找
TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}
TOOL_NODE = ToolNode(ALL_TOOLS, handle_tool_errors=True)

# ============ MCP 动态工具支持 ============
_MCP_DYNAMIC_TOOLS: list[BaseTool] = []
_MCP_TOOL_NODE: Optional[ToolNode] = None


def refresh_mcp_tools() -> int:
    """刷新 MCP 动态工具（从 langchain-mcp-adapters）

    调用此函数会重新从已连接的 MCP 服务器获取工具，
    并更新 TOOL_MAP 和重新创建 TOOL_NODE。

    Returns:
        获取到的工具数量
    """
    global _MCP_TOOL_NODE, TOOL_MAP, TOOL_NODE

    try:
        from minicode.tools.mcp_tools import get_mcp_client
        from langchain_core.tools import StructuredTool

        client = get_mcp_client()

        # 刷新 MCP 客户端的工具列表
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(client.refresh())
        except RuntimeError:
            # No running loop, use sync fallback
            pass

        # 直接从客户端获取 MCP 工具
        new_tools = client.get_tools()

        if new_tools:
            # 获取本地工具名称列表
            local_tool_names = {t.name for t in ALL_TOOLS}

            # 处理冲突的 MCP 工具，重命名为 mcp_xxx
            processed_mcp_tools = []
            for t in new_tools:
                if t.name in local_tool_names:
                    # 创建重命名版本
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

            # 更新全局工具映射
            TOOL_MAP = {t.name: t for t in ALL_TOOLS}
            for t in processed_mcp_tools:
                TOOL_MAP[t.name] = t

            # 重新创建工具节点（包含 MCP 工具）
            TOOL_NODE = ToolNode(ALL_TOOLS + processed_mcp_tools, handle_tool_errors=True)

            # 重置模型以包含新工具
            reset_for_mcp_refresh()

            return len(processed_mcp_tools)
    except Exception as e:
        print(f"[MCP] Failed to refresh tools: {e}")
        import traceback
        traceback.print_exc()

    return 0


def get_all_tools() -> list[BaseTool]:
    """获取所有可用工具（包括 MCP 动态工具）"""
    # 直接从 MCP 客户端获取工具列表，避免全局变量导入问题
    try:
        from minicode.tools.mcp_tools import get_mcp_client
        client = get_mcp_client()
        mcp_tools = client.get_tools()
        return ALL_TOOLS + mcp_tools
    except Exception:
        return ALL_TOOLS


def get_tool_map() -> dict[str, BaseTool]:
    """获取工具映射"""
    return TOOL_MAP


def reset_for_mcp_refresh():
    """在刷新 MCP 工具后重置 AgentGraphBuilder"""
    builder = AgentGraphBuilder.get_instance()
    if builder:
        builder.reset()


class BashSecurityValidator:
    """简单的 Bash 命令安全验证"""
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
    """构建 Agent Graph 的 Builder"""
    _instance: Optional["AgentGraphBuilder"] = None

    def __init__(
        self,
        model_provider: str = "anthropic",
        model_name: str = "claude-sonnet-4-7",
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self._model = None
        self._model_with_tools = None

    @classmethod
    def get_instance(cls) -> "AgentGraphBuilder":
        return cls._instance

    @property
    def model(self):
        if self._model is None:
            self._model = create_provider(
                provider=self.model_provider,
                model=self.model_name,
            ).client
        return self._model

    @property
    def model_with_tools(self):
        if self._model_with_tools is None:
            # 绑定所有工具（包括 MCP 动态工具）
            all_tools = get_all_tools()
            self._model_with_tools = self.model.bind_tools(all_tools)
        return self._model_with_tools

    def reset(self):
        """重置模型实例（用于切换模型或刷新工具时）"""
        self._model = None
        self._model_with_tools = None


def _build_system_message(state: Optional[AgentState] = None) -> str:
    """构建系统提示 - 支持记忆层注入"""
    base_prompt = get_system_prompt(WORKDIR)

    # 如果有 AgentState，尝试注入记忆层
    if state and HAS_MEMORY_LAYER:
        parts = [base_prompt]

        # 静态记忆: 用户偏好、项目配置
        if "static_memory" in state and state["static_memory"]:
            parts.append(state["static_memory"])

        # 动态记忆: 当前会话状态
        if "session_context" in state and state["session_context"]:
            parts.append(state["session_context"])

        # 事件记忆: 相关经验
        if "episodic_memory" in state and state["episodic_memory"]:
            parts.append(state["episodic_memory"])

        return "\n\n".join(parts)

    return base_prompt


def call_model(state: AgentState) -> dict:
    """调用 LLM 进行推理 - 核心节点"""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}

    # 构建系统提示 - 传入 state 以便注入记忆
    system_msg = SystemMessage(content=_build_system_message(state))
    messages_with_system = [system_msg] + list(messages)

    # 获取环境变量中的模型配置
    provider = os.environ.get("MODEL_PROVIDER", "anthropic")
    name = os.environ.get("MODEL_NAME", "claude-sonnet-4-7")

    builder = AgentGraphBuilder.get_instance()
    if not builder or builder.model_provider != provider:
        builder = AgentGraphBuilder(provider, name)

    response = builder.model_with_tools.invoke(messages_with_system)
    return {"messages": [response]}


def _run_tool_async(tool, args: dict) -> Any:
    """在线程池中运行异步工具"""
    async def _ainvoke():
        return await tool.ainvoke(args)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_ainvoke())
    finally:
        loop.close()


def _is_async_tool(tool: Any) -> bool:
    """检查工具是否为异步工具（通过 coroutine 属性判断）"""
    return hasattr(tool, 'coroutine') and tool.coroutine is not None


def _execute_tool(tool: Any, tool_args: dict) -> Any:
    """执行单个工具（同步或异步）"""
    try:
        # 检查是否有 coroutine（异步工具）
        if _is_async_tool(tool):
            # 异步工具通过 _run_tool_async 执行
            return _run_tool_async(tool, tool_args)
        else:
            # 同步工具直接调用
            if hasattr(tool, 'invoke'):
                return tool.invoke(tool_args)
            elif hasattr(tool, 'func'):
                return tool.func(**tool_args)
            else:
                return f"[Error] Tool has no invoke or func method"
    except Exception as e:
        return f"[Error] {e}"


def execute_tools(state: AgentState) -> dict:
    """执行工具 - 核心节点"""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [], "tool_messages": []}

    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": [], "tool_messages": []}

    tool_calls = last_message.tool_calls
    tool_messages = []

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc.get("args", {}) or {}
        tool_call_id = tc.get("id", "")

        # Bash 权限检查
        if tool_name == "bash_tool":
            command = tool_args.get("command", "")
            allowed, reason = check_permission(command, "bash_tool")
            if not allowed:
                tool_messages.append(
                    ToolMessage(content=f"[Permission Denied]: {reason}", tool_call_id=tool_call_id)
                )
                continue

        # 查找工具
        tool = TOOL_MAP.get(tool_name)
        if not tool:
            tool_messages.append(
                ToolMessage(content=f"[Error] Unknown tool: {tool_name}", tool_call_id=tool_call_id)
            )
            continue

        try:
            # 根据工具类型选择执行方式
            result = _execute_tool(tool, tool_args)

            # 处理结果
            if isinstance(result, ToolMessage):
                result.tool_call_id = tool_call_id
                tool_messages.append(result)
            else:
                # 转换为 ToolMessage
                content = str(result) if not isinstance(result, str) else result
                tool_messages.append(
                    ToolMessage(content=content, tool_call_id=tool_call_id)
                )
        except Exception as e:
            tool_messages.append(
                ToolMessage(content=f"[Error] {e}", tool_call_id=tool_call_id)
            )

    return {"messages": tool_messages, "tool_messages": tool_messages}


def should_continue(state: AgentState) -> Literal["tools", END]:
    """判断是否继续（工具调用）还是结束"""
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
    """创建轻量级 Agent Graph"""
    builder = AgentGraphBuilder(model_provider, model_name)
    AgentGraphBuilder._instance = builder

    # 初始化时刷新 MCP 工具
    refresh_mcp_tools()

    workflow = StateGraph(AgentState)

    # 核心节点：只保留 agent 和 tools
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)

    # 边：agent → tools(如果需要) → agent → ...
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
        interrupt_before=None,  # 工具自动执行
    )


# 别名用于向后兼容
create_agent_graph_stream = create_agent_graph
create_agent_graph_async = create_agent_graph