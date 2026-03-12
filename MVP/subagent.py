import os
import operator
from typing import Annotated, List, TypedDict, Union

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessageChunk,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# 1. 环境准备
load_dotenv(override=True)
os.environ["NO_PROXY"] = "*"
MODEL_ID = os.environ["AGENCY_LLM_MODEL"]
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER")
llm = init_chat_model(
    MODEL_ID,
    model_provider=PROVIDER,  # explicitly set provider
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
)

WORKDIR = os.getcwd()

# --- 2. 定义基础工具 (Shared Tools) ---


@tool
def bash(command: str) -> str:
    """Run a shell command in the local filesystem."""
    import subprocess

    dangerous = ["rm -rf /", "sudo", "shutdown"]
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
        return (r.stdout + r.stderr).strip() or "(no output)"
    except Exception as e:
        return f"Error: {e}"


@tool
def read_file(path: str) -> str:
    """Read file contents."""
    try:
        with open(os.path.join(WORKDIR, path), "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to file."""
    try:
        with open(os.path.join(WORKDIR, path), "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


# 基础工具集
BASE_TOOLS = [bash, read_file, write_file]

# --- 3. 定义子代理 (Subagent Graph) ---


class AgentState(TypedDict):
    # 使用 Annotated[..., operator.add] 实现消息列表的追加模式
    messages: Annotated[List[BaseMessage], operator.add]


def create_subagent_graph():
    # 子代理的系统提示词
    SUBAGENT_SYSTEM = (
        f"You are a coding subagent at {WORKDIR}. Complete the task and summarize."
    )

    subagent_llm = llm.bind_tools(BASE_TOOLS)

    def call_model(state: AgentState):
        messages = [SystemMessage(content=SUBAGENT_SYSTEM)] + state["messages"]
        response = subagent_llm.invoke(messages)
        return {"messages": [response]}

    # 构建子代理图
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(BASE_TOOLS))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# 编译子代理图
subagent_app = create_subagent_graph()

# --- 4. 定义“任务分发”工具 (Task Tool) ---


@tool
def task(prompt: str) -> str:
    """
    Spawn a subagent with fresh context to handle a subtask.
    It shares the filesystem but not conversation history.
    """
    # 关键点：这里传入全新的 messages 列表，实现上下文隔离
    inputs = {"messages": [HumanMessage(content=prompt)]}
    result = subagent_app.invoke(inputs)

    # 只返回子代理最后的一条文本消息作为总结
    final_msg = result["messages"][-1].content
    return final_msg


# --- 5. 定义主代理 (Parent Agent Graph) ---


def create_parent_graph():
    PARENT_SYSTEM = f"You are a lead coding agent. Use the 'task' tool to delegate complex subtasks."

    # 父代理拥有基础工具 + task 调度工具
    PARENT_TOOLS = BASE_TOOLS + [task]
    parent_llm = llm.bind_tools(PARENT_TOOLS)

    def call_model(state: AgentState):
        messages = state["messages"]
        response = None
        for chunk in parent_llm.stream(messages):
            if isinstance(chunk, AIMessageChunk):
                print(chunk.content, end="", flush=True)
            if response is None:
                response = chunk
            else:
                response = response + chunk
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(PARENT_TOOLS))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# 编译主代理
parent_app = create_parent_graph()

# --- 6. 运行入口 ---

if __name__ == "__main__":
    print("\033[32m--- LangGraph Parent-Subagent System Started ---\033[0m")
    state = {"messages": []}
    while True:
        user_input = input("> ")
        state["messages"].append(HumanMessage(content=user_input))
        state = parent_app.invoke(state)
        print()
