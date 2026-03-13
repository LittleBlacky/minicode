#!/usr/bin/env python3
"""
s04_subagent_langgraph.py - Subagents (LangGraph 重写版)

用 LangGraph 实现父子 Agent 的上下文隔离模式：
- 父 Agent：拥有完整对话历史，通过 task 工具派发子任务
- 子 Agent：独立图实例，fresh 消息列表，完成后只返回摘要

架构示意：
    Parent Graph                     Subagent Graph
    +------------------+             +------------------+
    | StateGraph       |             | StateGraph       |  <-- fresh state
    | messages=[...]   |  dispatch   | messages=[]      |
    |                  | ----------> |                  |
    | node: agent      |             | node: agent      |
    | node: tools      |             | node: tools      |
    |                  |  summary    |                  |
    |   result = "..." | <---------- | return last AIMsg|
    +------------------+             +------------------+

Key insight: 子图用独立的 StateGraph 实例 + 空 messages 实现上下文隔离。
"""

import os
import subprocess
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# 环境 & 配置
# ---------------------------------------------------------------------------
load_dotenv(override=True)

WORKDIR = Path.cwd()

os.environ["NO_PROXY"] = "*"
MODEL_ID = os.environ["AGENCY_LLM_MODEL"]
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER")

SYSTEM_PARENT = (
    f"You are a coding agent at {WORKDIR}. "
    "Use the task tool to delegate exploration or subtasks."
)
SYSTEM_SUBAGENT = (
    f"You are a coding subagent at {WORKDIR}. "
    "Complete the given task, then summarize your findings."
)


# ---------------------------------------------------------------------------
# 状态定义
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# 工具实现（父子共享文件系统操作）
# ---------------------------------------------------------------------------
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@tool
def bash(command: str) -> str:
    """Run a shell command in the workspace directory."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


@tool
def read_file(path: str, limit: int = None) -> str:
    """Read file contents, optionally limiting to first N lines."""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace the first occurrence of old_text with new_text in a file."""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# 子 Agent 可用的工具（不含 task，防止递归派发）
CHILD_TOOLS = [bash, read_file, write_file, edit_file]


# ---------------------------------------------------------------------------
# 子 Agent 图构建（上下文隔离：每次调用创建新实例）
# ---------------------------------------------------------------------------
def build_subagent_graph():
    """构建子 Agent 图。每次派发任务时调用，保证 fresh 状态。"""
    llm = init_chat_model(
        MODEL_ID,
        model_provider=PROVIDER,  # explicitly set provider
        temperature=0,
        max_tokens=8000,
        base_url=BASE_URL,
        api_key=API_KEY,
    ).bind_tools(CHILD_TOOLS)

    def subagent_node(state: AgentState):
        response = llm.invoke(
            [{"role": "system", "content": SYSTEM_SUBAGENT}] + state["messages"]
        )
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", END]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", subagent_node)
    graph.add_node("tools", ToolNode(CHILD_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_subagent(prompt: str) -> str:
    """
    派发子任务：在独立图实例中以空消息列表启动子 Agent。
    只将最终 AIMessage 的文本返回给父 Agent，子上下文随即丢弃。
    """
    subgraph = build_subagent_graph()
    # fresh messages=[] 实现上下文隔离
    final_state = subgraph.invoke({"messages": [HumanMessage(content=prompt)]})
    messages = final_state["messages"]
    # 提取最后一条 AI 消息的文本作为摘要
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return "(no summary)"


# ---------------------------------------------------------------------------
# task 工具：父 Agent 专属，触发子 Agent 派发
# ---------------------------------------------------------------------------
@tool
def task(prompt: str, description: str = "subtask") -> str:
    """
    Spawn a subagent with fresh context to handle a subtask.
    The subagent shares the filesystem but has no conversation history.
    Returns only the subagent's final summary.
    """
    print(f"\033[33m> task ({description}): {prompt[:80]}\033[0m")
    result = run_subagent(prompt)
    print(f"\033[90m  {result[:200]}\033[0m")
    return result


# 父 Agent 工具 = 子 Agent 工具 + task
PARENT_TOOLS = CHILD_TOOLS + [task]


# ---------------------------------------------------------------------------
# 父 Agent 图构建
# ---------------------------------------------------------------------------
def build_parent_graph():
    llm = init_chat_model(
        MODEL_ID,
        model_provider=PROVIDER,  # explicitly set provider
        temperature=0,
        max_tokens=8000,
        base_url=BASE_URL,
        api_key=API_KEY,
    ).bind_tools(PARENT_TOOLS)

    def parent_node(state: AgentState):
        response = llm.invoke(
            [{"role": "system", "content": SYSTEM_PARENT}] + state["messages"]
        )
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", END]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", parent_node)
    graph.add_node("tools", ToolNode(PARENT_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# 主循环：保持父 Agent 跨轮次的对话历史
# ---------------------------------------------------------------------------
def main():
    parent_graph = build_parent_graph()
    history: list = []  # 跨轮次持久化父 Agent 消息历史

    print("\033[36m父子 Agent (LangGraph) — 输入 q/exit 退出\033[0m\n")

    while True:
        try:
            query = input("\033[36ms04 >> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in ("q", "exit", ""):
            break

        # 将用户消息追加到历史，传入图
        history.append(HumanMessage(content=query))
        final_state = parent_graph.invoke({"messages": history})

        # 用图返回的完整消息列表替换历史（包含 AI 回复 & 工具结果）
        history = final_state["messages"]

        # 打印最后一条 AI 消息
        for msg in reversed(history):
            if isinstance(msg, AIMessage) and msg.content:
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                print(f"\n\033[32m{text}\033[0m")
                break

        print()


if __name__ == "__main__":
    main()
