#!/usr/bin/env python3
import os
import subprocess
from typing import TypedDict, List
from pathlib import Path
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately


# =====================
# ENV
# =====================
load_dotenv()
os.environ["NO_PROXY"] = "*"

MODEL_ID = os.environ["AGENCY_LLM_MODEL"]
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER", "openai")

WORKDIR = Path.cwd()

MAX_TOKEN_BEFORE_SUMMARY = 4000  # 总 Token 超过此值触发摘要
KEEP_RECENT_MESSAGES = 15  # 摘要后保留的最近原始消息数


# =====================
# LLM
# =====================
llm = init_chat_model(
    model=MODEL_ID,
    model_provider=PROVIDER,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0,
)


# =====================
# TOOLS
# =====================
def safe_path(path_str: str) -> Path:
    path = (WORKDIR / path_str).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError("Path escapes workspace")
    return path


@tool
def read_file(path: str) -> str:
    """Read file content from workspace path."""
    try:
        return safe_path(path).read_text()
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in workspace."""
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


@tool
def bash(command: str) -> str:
    """Run bash command inside workspace."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return (result.stdout + result.stderr).strip() or "(no output)"
    except Exception as e:
        return f"Error: {e}"


TOOLS = [read_file, write_file, bash]
llm_with_tools = llm.bind_tools(TOOLS)


# =====================
# STATE
# =====================
class AgentState(TypedDict):
    messages: List


# =====================
# NODES
# =====================


def pre_process(state: AgentState):
    """入口节点：不做修改，仅用于路由判断"""
    return state


def agent_node(state: AgentState):
    """主推理节点，支持工具调用和流式输出"""
    response = None
    for chunk in llm_with_tools.stream(state["messages"]):
        if isinstance(chunk, AIMessageChunk):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if response is None:
                response = chunk
            else:
                response = response + chunk
    print()
    if response is None:
        response = AIMessage(content="")
    return {"messages": [response]}


def summarize_node(state: AgentState):
    """
    智能摘要节点：
    - 将早期消息交给LLM总结
    - 保留最近若干条原始消息，避免丢失细节
    """
    messages = state["messages"]
    # 分离旧消息和需保留的最近消息
    if len(messages) <= KEEP_RECENT_MESSAGES:
        return {"messages": messages}

    old_messages = messages[:-KEEP_RECENT_MESSAGES]
    recent_messages = messages[-KEEP_RECENT_MESSAGES:]

    print("🧠 上下文过长，正在生成摘要...")
    summary_prompt = [
        SystemMessage(
            content="将以下对话历史总结成一段简洁的摘要，保留关键目标、决策和未完成事项。"
        ),
        HumanMessage(content=str(old_messages)),
    ]
    summary_response = llm.invoke(summary_prompt)

    # 摘要作为系统消息注入，保留最近原始消息
    summary_msg = SystemMessage(content=f"[对话历史摘要]\n{summary_response.content}")
    new_messages = [summary_msg] + recent_messages

    print(f"✅ 摘要完成，消息从 {len(messages)} 条压缩为 {len(new_messages)} 条")
    return {"messages": new_messages}


# =====================
# ROUTERS
# =====================


def should_compact(state: AgentState):
    """使用近似Token计数判断是否需要摘要"""
    total_tokens = count_tokens_approximately(
        state["messages"],
    )
    # 调试时可打开下面的打印
    # print(f"当前Token数: {total_tokens}")
    return "summarize" if total_tokens > MAX_TOKEN_BEFORE_SUMMARY else "agent"


def should_use_tool(state: AgentState):
    """判断最后一条消息是否包含工具调用"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool"
    return "end"


# =====================
# GRAPH
# =====================
builder = StateGraph(AgentState)

# 注册节点
builder.add_node("pre_process", pre_process)
builder.add_node("summarize", summarize_node)
builder.add_node("agent", agent_node)
builder.add_node("tool", ToolNode(TOOLS))

# 入口为 pre_process
builder.set_entry_point("pre_process")

# 上下文控制：根据Token数量决定是否先摘要
builder.add_conditional_edges(
    "pre_process",
    should_compact,
    {"summarize": "summarize", "agent": "agent"},
)

# 工具路由
builder.add_conditional_edges(
    "agent",
    should_use_tool,
    {"tool": "tool", "end": END},
)

# 摘要后和工具执行后都进入agent
builder.add_edge("summarize", "agent")
builder.add_edge("tool", "summarize")

graph = builder.compile()


# =====================
# CLI
# =====================
def run():
    state: AgentState = {
        "messages": [
            SystemMessage(content="You are a coding agent. Use tools when needed.")
        ]
    }

    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip() in ("q", "exit", ""):
            break

        state["messages"].append(HumanMessage(content=query))
        # 每次输入后调用 graph，自动进行上下文检查
        state = graph.invoke(state)
        print()


if __name__ == "__main__":
    run()
