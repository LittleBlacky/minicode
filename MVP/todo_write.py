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

load_dotenv(override=True)

WORKDIR = Path.cwd()
MODEL_ID = os.environ["MODEL_ID"]
BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
API_KEY = os.getenv("ANTHROPIC_API_KEY")

model = init_chat_model(
    MODEL_ID,
    model_provider="anthropic",
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
)

PLAN_REMINDER_INTERVAL = 3
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool for multi-step work.
Keep exactly one step in_progress when a task has multiple steps.
Refresh the plan as work advances. Prefer tools over prose."""


@dataclass
class PlanItem:
    content: str
    status: str = "pending"
    active_form: str = ""


@dataclass
class PlanningState:
    items: list[PlanItem] = field(default_factory=list)
    rounds_since_update: int = 0


class TodoManager:
    def __init__(self):
        self.state = PlanningState()

    def update(self, items: list) -> str:
        if len(items) > 12:
            raise ValueError("Keep the session plan short (max 12 items)")
        normalized = []
        in_progress_count = 0
        for index, raw_item in enumerate(items):
            content = str(raw_item.get("content", "")).strip()
            status = str(raw_item.get("status", "pending")).lower()
            active_form = str(raw_item.get("activeForm", "")).strip()
            if not content:
                raise ValueError(f"Item {index}: content required")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Item {index}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            normalized.append(
                PlanItem(
                    content=content,
                    status=status,
                    active_form=active_form,
                )
            )
        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")
        self.state.items = normalized
        self.state.rounds_since_update = 0
        return self.render()

    def note_round_without_update(self) -> None:
        self.state.rounds_since_update += 1

    def reminder(self) -> Optional[str]:
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>Refresh your current plan before continuing.</reminder>"

    def render(self) -> str:
        if not self.state.items:
            return "No session plan yet."
        lines = []
        for item in self.state.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item.status]
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)
        completed = sum(1 for item in self.state.items if item.status == "completed")
        lines.append(f"\n({completed}/{len(self.state.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


def safe_path(path_str: str) -> Path:
    path = (WORKDIR / path_str).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    output = (result.stdout + result.stderr).strip()
    return output[:50000] if output else "(no output)"


def run_read(path: str, limit: Optional[int] = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as exc:
        return f"Error: {exc}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"Error: {exc}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        content = file_path.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        file_path.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as exc:
        return f"Error: {exc}"


def run_todo(items: list) -> str:
    """Wrapper for the todo manager update."""
    return TODO.update(items)


@tool
def bash_tool(command: str) -> str:
    """Run a shell command in the workspace."""
    return run_bash(command)


@tool
def read_file(path: str, limit: Optional[int] = None) -> str:
    """Read file contents. Optionally limit the number of lines returned."""
    return run_read(path, limit)


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    return run_write(path, content)


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace the first occurrence of old_text with new_text in a file."""
    return run_edit(path, old_text, new_text)


@tool
def todo_tool(items: list) -> str:
    """Rewrite the current session plan for multi-step work.
    Provide a list of plan items with content, status (pending/in_progress/completed),
    and optional activeForm string.
    """
    return run_todo(items)


tools = [bash_tool, read_file, write_file, edit_file, todo_tool]
tool_map = {t.name: t for t in tools}

# Bind tools to model
model_with_tools = model.bind_tools(tools)


# ----------------------------------------------------------------------
# State definition (extended with reminder tracking)
# ----------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # Track whether a todo update occurred in the last turn
    todo_updated_this_turn: bool


# ----------------------------------------------------------------------
# Graph nodes
# ----------------------------------------------------------------------
def call_model(state: AgentState) -> dict:
    """Invoke LLM with system prompt and current messages."""
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM)] + messages

    # Inject reminder if needed (original logic: insert into user message content)
    # We'll handle reminder by adding a text block to the last user message before model call.
    # But LangGraph's state is immutable; we can create a modified copy for this turn.
    reminder = TODO.reminder()
    if reminder and messages and isinstance(messages[-1], HumanMessage):
        # Append reminder as additional text to the last user message
        original_content = messages[-1].content
        if isinstance(original_content, str):
            new_content = f"{reminder}\n{original_content}"
        else:
            new_content = f"{reminder}\n{str(original_content)}"
        messages = list(messages)
        messages[-1] = HumanMessage(content=new_content)

    response = model_with_tools.invoke(messages)
    return {"messages": [response], "todo_updated_this_turn": False}


def execute_tools(state: AgentState) -> dict:
    """Execute tool calls from the last assistant message."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

    tool_messages = []
    todo_updated = False
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_obj = tool_map.get(tool_name)
        if tool_obj:
            print(f"> {tool_name}:")
            try:
                output = tool_obj.invoke(tool_args)
            except Exception as e:
                output = f"Error: {e}"
        else:
            output = f"Unknown tool: {tool_name}"

        print(str(output)[:200])
        tool_messages.append(
            ToolMessage(content=str(output), tool_call_id=tool_call["id"])
        )

        if tool_name == "todo_tool":
            todo_updated = True

    # Update rounds_since_update if todo was not used
    if not todo_updated:
        TODO.note_round_without_update()
    else:
        TODO.state.rounds_since_update = 0

    return {
        "messages": tool_messages,
        "todo_updated_this_turn": todo_updated,
    }


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Decide whether to continue tool execution or finish."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"


# ----------------------------------------------------------------------
# Build graph
# ----------------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

graph = workflow.compile()

# ----------------------------------------------------------------------
# CLI loop (preserved original style)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize state with empty messages list
    state = {"messages": [], "todo_updated_this_turn": False}
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        # Add user message to state
        state["messages"].append(HumanMessage(content=query))
        state["todo_updated_this_turn"] = False

        # Run the graph
        result = graph.invoke(state)
        state = result

        # Print final assistant response (text part)
        final_msg = state["messages"][-1]
        if isinstance(final_msg, AIMessage) and final_msg.content:
            print(final_msg.content)
        print()
