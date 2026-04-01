#!/usr/bin/env python3
from __future__ import annotations
"""
phase13_task_system.py - Tasks (Persistent Task Graph)

Tasks persist as JSON files in .tasks/ so they survive context compression.
Each task carries a small dependency graph:
- blockedBy: what must finish first
- blocks: what this task unlocks later

Key insight: Task state survives compression because it lives on disk,
not only inside the conversation.

    .tasks/
      task_1.json  {"id":1, "subject":"...", "status":"completed", ...}
      task_2.json  {"id":2, "blockedBy":[1], "status":"pending", ...}
      task_3.json  {"id":3, "blockedBy":[2], "blocks":[], ...}

These are durable work-graph tasks, not transient runtime execution slots.

Key LangGraph concepts:
- use StateGraph with task state for persistent work tracking
- tool nodes for task operations
- memory persistence via JSON files
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

load_dotenv(override=True)
os.environ["NO_PROXY"] = "*"

MODEL_ID = os.environ.get("AGENCY_LLM_MODEL", os.environ.get("MODEL_ID", "claude-sonnet-4-7"))
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER", "openai")

WORKDIR = Path.cwd()
TASKS_DIR = WORKDIR / ".tasks"

model = init_chat_model(
    MODEL_ID,
    model_provider=PROVIDER,
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
)


# ========== TaskManager: Persistent Task Store ==========

class TaskManager:
    """
    Persistent TaskRecord store on disk.
    Think "work graph on disk", not "currently running worker".
    """

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> dict:
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],
            "blocks": [],
            "owner": "",
        }
        self._save(task)
        self._next_id += 1
        return task

    def get(self, task_id: int) -> dict:
        return self._load(task_id)

    def update(
        self,
        task_id: int,
        status: Optional[str] = None,
        owner: Optional[str] = None,
        add_blocked_by: Optional[list] = None,
        add_blocks: Optional[list] = None,
    ) -> dict:
        task = self._load(task_id)
        if owner is not None:
            task["owner"] = owner
        if status:
            if status not in ("pending", "in_progress", "completed", "deleted"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            # When a task is completed, remove it from all other tasks' blockedBy
            if status == "completed":
                self._clear_dependency(task_id)
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
            # Bidirectional: also update the blocked tasks' blockedBy lists
            for blocked_id in add_blocks:
                try:
                    blocked = self._load(blocked_id)
                    if task_id not in blocked["blockedBy"]:
                        blocked["blockedBy"].append(task_id)
                        self._save(blocked)
                except ValueError:
                    pass
        self._save(task)
        return task

    def _clear_dependency(self, completed_id: int):
        """Remove completed_id from all other tasks' blockedBy lists."""
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> list[dict]:
        """List all tasks."""
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        return tasks


# Global task manager instance
TASKS = TaskManager(TASKS_DIR)


# ========== Agent State ==========

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    last_tool_result: Optional[str]
    pending_tasks: list[dict]


# ========== Tool Functions ==========

def safe_path(path: str) -> Path:
    """Resolve path relative to workspace."""
    p = (WORKDIR / path).resolve()
    if not p.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return p


@tool
def bash_tool(command: str) -> str:
    """Run a shell command in the workspace."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "[Error]: Dangerous command blocked"
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


@tool
def read_file(path: str, limit: Optional[int] = None) -> str:
    """Read file contents."""
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"[Error]: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"[Error]: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in a file."""
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"[Error]: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"[Error]: {e}"


@tool
def task_create(subject: str, description: str = "") -> str:
    """Create a new task. Returns task details as JSON."""
    task = TASKS.create(subject, description)
    return json.dumps(task, indent=2)


@tool
def task_update(
    task_id: int,
    status: Optional[str] = None,
    owner: Optional[str] = None,
    add_blocked_by: Optional[list] = None,
    add_blocks: Optional[list] = None,
) -> str:
    """Update a task's status, owner, or dependencies."""
    try:
        task = TASKS.update(task_id, status, owner, add_blocked_by, add_blocks)
        return json.dumps(task, indent=2)
    except ValueError as e:
        return f"[Error]: {e}"


@tool
def task_list() -> str:
    """List all tasks with status summary."""
    tasks = TASKS.list_all()
    if not tasks:
        return "No tasks."
    lines = []
    for t in tasks:
        marker = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
            "deleted": "[-]",
        }.get(t["status"], "[?]")
        blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
        owner = f" owner={t['owner']}" if t.get("owner") else ""
        lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{blocked}")
    return "\n".join(lines)


@tool
def task_get(task_id: int) -> str:
    """Get full details of a task by ID."""
    try:
        task = TASKS.get(task_id)
        return json.dumps(task, indent=2)
    except ValueError as e:
        return f"[Error]: {e}"


# Define tool list and tool node
agent_tools = [bash_tool, read_file, write_file, edit_file, task_create, task_update, task_list, task_get]
tool_node = ToolNode(agent_tools, handle_tool_errors=True)
model_with_tools = model.bind_tools(agent_tools)


# ========== Graph Nodes ==========

SYSTEM_PROMPT = f"""You are a coding agent at {WORKDIR}. Use task tools to plan and track work.

Available task operations:
- task_create(subject, description): Create a new task
- task_update(task_id, status, owner, add_blocked_by, add_blocks): Update task
- task_list(): List all tasks with status
- task_get(task_id): Get task details

Task workflow:
1. Break complex work into tasks
2. Track progress with task_update
3. Set dependencies with add_blocked_by
4. Tasks survive context compression (stored on disk)
"""


def call_model(state: AgentState) -> dict:
    """Call the model with current messages."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response], "last_tool_result": None}


def should_continue(state: AgentState) -> Literal["tools", END]:
    """Check if there are tool calls to execute."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)

graph = workflow.compile()


def run_agent(query: str):
    """Run the agent with a query."""
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "last_tool_result": None,
        "pending_tasks": [],
    }

    for event in graph.stream(initial_state):
        node_name = list(event.keys())[0]
        if node_name == "agent":
            response = event[node_name]["messages"][-1]
            if hasattr(response, 'content') and response.content:
                print(f"\nAssistant: {response.content}")
        elif node_name == "tools":
            pass  # Tools handled internally


if __name__ == "__main__":
    print("Task System Agent (phase13)")
    print("Type 'exit' or 'q' to quit\n")

    while True:
        try:
            query = input("\033[36mphase13 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        run_agent(query)