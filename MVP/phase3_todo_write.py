import os
import subprocess
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessageChunk,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv(override=True)
os.environ["NO_PROXY"] = "*"
MODEL_ID = os.environ["AGENCY_LLM_MODEL"]
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER")
model = init_chat_model(
    MODEL_ID,
    model_provider=PROVIDER,  # explicitly set provider
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
)


WORKDIR = Path.cwd()
PLAN_REMINDER_INTERVAL = 3
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool for multi-step work.
Keep exactly one step in_progress when a task has multiple steps.
Refresh the plan as work advances. Prefer tools over prose."""


def safe_path(path: str) -> Path:
    """Resolve path relative to workspace and ensure it stays inside."""
    full_path = (WORKDIR / path).resolve()
    if not full_path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return full_path


@tool
def read_file(path: str, limit: Optional[int] = None) -> str:
    """Read file contents. Optionally limit the number of lines returned."""
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"[Error]: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"[Error]: {e}"


@tool
def edit_file(path: str, old_text: str, new_str: str) -> str:
    """Replace the first occurrence of old_text with new_text in a file."""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"[Error]: Text not found in {path}"
        new_content = content.replace(old_text, new_str, 1)
        fp.write_text(new_content)
        return f"Replaced 1 occurrence in {path}"
    except Exception as e:
        return f"[Error]: {e}"


@tool
def bash_tool(command: str) -> str:
    """Run a shell command in the workspace."""
    # 更严格（但仍非完全安全）的危险命令检测
    dangerous_patterns = [
        "rm -rf /",
        "rm -rf /*",
        "sudo",
        "shutdown",
        "reboot",
        "> /dev/",
        "dd if=",
        "mkfs",
        ":(){ :|:& };:",  # fork bomb
    ]
    if any(pattern in command for pattern in dangerous_patterns):
        return "[Error]: Dangerous command blocked. If you believe this is safe, please re-evaluate."

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
        return "[Error]: Timeout (120s)"


class TodoItem(BaseModel):
    content: str = Field(description="任务内容")
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="任务状态"
    )
    activeForm: Optional[str] = Field(
        default=None, description="当前进行中的活动形式描述"
    )


class TodoUpdate(BaseModel):
    items: list[TodoItem] = Field(description="计划项列表")


def _render_todo(items) -> str:
    if not items:
        return "No session plan yet"
    lines = []
    todo_marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
    for idx, item in enumerate(items):
        marker = todo_marker[item["status"]]
        line = f"{marker} {item['content']}"
        if item["status"] == "in_progress" and item.get("active_form"):
            line += f" ( {item['active_form']} )"
        lines.append(line)
    completed = sum(1 for item in items if item["status"] == "completed")
    lines.append(f"\n({completed} / {len(items)}) completed")
    return "\n".join(lines)


@tool(args_schema=TodoUpdate)
def update_todo(items: list[TodoItem]) -> str:
    """Rewrite the current session plan for multi-step work. Keep exactly one item in_progress."""
    if len(items) > 12:
        raise ValueError("Keep the session plan short (max 12 items)")

    normalized = []
    in_progress_count = 0
    for idx, item in enumerate(items):
        content = item.content.strip()
        status = item.status.lower()
        if not content:
            raise ValueError(f"Item {idx}: content required")
        if status not in {"pending", "in_progress", "completed"}:
            raise ValueError(f"Item {idx}: invalid status {status}")
        if status == "in_progress":
            in_progress_count += 1
        normalized.append({"content": content, "status": status})

    if in_progress_count > 1:
        raise ValueError("Only one plan item can be in_progress")
    return _render_todo(normalized)


tools = [bash_tool, read_file, write_file, edit_file, update_todo]
tool_map = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    todo_items: list[dict]
    rounds_since_todo_update: int


PLAN_REMINDER_INTERVAL = 3


def call_model_stream(state: AgentState) -> dict:
    """流式输出模型的响应，并可能插入todo刷新提醒。"""
    # 如果需要提醒，则在消息副本中修改最后一条HumanMessage
    messages_to_use = state["messages"]
    if state["rounds_since_todo_update"] >= PLAN_REMINDER_INTERVAL:
        # 找到最后一条HumanMessage
        for i in range(len(messages_to_use) - 1, -1, -1):
            if isinstance(messages_to_use[i], HumanMessage):
                original_msg = messages_to_use[i]
                reminder = (
                    "<reminder>Refresh your current plan before continuing.</reminder>"
                )
                # 修改副本：构造新HumanMessage，但因为我们不修改状态，只用于本次调用
                # 为简单，直接修改列表的副本（列表内容不变，只是创建一个新的列表引用？消息对象不可变？HumanMessage可哈希但不可变，需新建）
                # 这里我们构建一个新的消息列表，替换最后一条HumanMessage
                new_msg = HumanMessage(
                    content=f"{reminder}\n{original_msg.content}",
                    id=original_msg.id,  # 保持相同ID，若传入模型可能会忽略ID，保留无妨
                )
                messages_to_use = (
                    messages_to_use[:i] + [new_msg] + messages_to_use[i + 1 :]
                )
                break

    full_response = None
    for chunk in model_with_tools.stream(messages_to_use):
        if isinstance(chunk, AIMessageChunk):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if full_response is None:
                full_response = chunk
            else:
                full_response = full_response + chunk
    print()  # 换行
    if full_response is None:
        full_response = AIMessage(content="")
    return {"messages": [full_response]}


def execute_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    tool_calls = last_message.tool_calls
    todo_updated = False
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_handler = tool_map.get(tool_name)
        if tool_handler:
            print(f"> {tool_name}: ")
            try:
                output = tool_handler.invoke(tool_args)
            except Exception as e:
                output = f"[Error]: {e}"
        else:
            output = f"Unknown tool: {tool_name}"
        print(str(output[:200]))
        tool_messages.append(
            ToolMessage(content=str(output), tool_call_id=tool_call["id"])
        )
        if tool_name == "update_todo":
            todo_updated = True
    rounds_since_todo_update = state["rounds_since_todo_update"] + 1
    if todo_updated:
        rounds_since_todo_update = 0
    return {
        "messages": tool_messages,
        "rounds_since_todo_update": rounds_since_todo_update,
    }


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"


workflow = StateGraph(AgentState)
# 移除了 reminder_todo 节点，直接从 agent 开始
workflow.add_node("agent", call_model_stream)
workflow.add_node("tools", execute_tools)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

graph = workflow.compile()

if __name__ == "__main__":
    print(f"\033[32mCoding Agent started in {WORKDIR}\033[0m")
    print("Type 'q', 'exit' or Ctrl+C to quit.\n")

    messages = [SystemMessage(content=SYSTEM)]
    state = {
        "messages": messages,
        "todo_items": [],
        "rounds_since_todo_update": 0,
    }

    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        state["messages"].append(HumanMessage(content=query))
        state = graph.invoke(state)
