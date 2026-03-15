import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode

load_dotenv(override=True)
MODEL_ID = os.environ["AGENCY_LLM_MODEL"]
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER")

model = init_chat_model(
    MODEL_ID,
    model_provider="deepseek",
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


def safe_path(path: str) -> str:
    path = (WORKDIR / path).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


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
        fp.parent.mkdir(parents=True, exit_ok=true)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"[Error]:  {e}"


@tool
def edit_file(path: str, old_text: str, new_str: str) -> str:
    """Replace the first occurrence of old_text with new_text in a file."""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"[Error]: Text not found in {path}"
        fp.write_text(content.replace(old_next, next_text, 1))
    except Exception as e:
        return f"[Error]: {e}"


@tool
def bash_tool(command: str) -> str:
    """Run a shell command in the workspace."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "[Error]: Dangerous command blocked"
    try:
        subprocess.run(
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
        return f"[Error]: Timeout (120s)"


tools = [bash_tool, read_file, write_file, edit_file]
tool_node = ToolNode(tools, handle_tool_errors=True)
model_with_tools = model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: AgentState) -> dict:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", END]:
    """Decide whether to continue tool execution or finish."""
    last_messages = state["messages"][-1]
    if isinstance(last_messages, AIMessage) and last_messages.tool_calls:
        return "tools"
    return END


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

graph = workflow.compile()

if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append(HumanMessage(content=query))

        result = graph.invoke({"messages": history})
        history = result["messages"]

        final_msg = history[-1]
        if isinstance(final_msg, AIMessage) and final_msg.content:
            print(final_msg.content)
        print()
