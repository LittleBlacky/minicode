#!/usr/bin/env python3
from __future__ import annotations
"""
phase19_worktree_task_isolation.py - Worktree + Task Isolation with LangGraph Native Patterns

Directory-level isolation for parallel task execution.
Tasks are the control plane and worktrees are the execution plane.

    .mini-agent-cli/tasks/task_12.json
      {
        "id": 12,
        "subject": "Implement auth refactor",
        "status": "in_progress",
        "worktree": "auth-refactor"
      }
    .mini-agent-cli/worktrees/index.json
      {
        "worktrees": [
          {
            "name": "auth-refactor",
            "path": ".../.mini-agent-cli/worktrees/auth-refactor",
            "branch": "wt/auth-refactor",
            "task_id": 12,
            "status": "active"
          }
        ]
      }

Key insight: "Isolate by directory, coordinate by task ID."

LangGraph native patterns:
- MemorySaver checkpointer for session persistence
- State updates for worktree lifecycle tracking
- Event state for lifecycle observability
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
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
STORAGE_DIR = WORKDIR / ".mini-agent-cli"
WORKTREES_DIR = STORAGE_DIR / "worktrees"
EVENTS_PATH = WORKTREES_DIR / "events.jsonl"

model = init_chat_model(
    MODEL_ID,
    model_provider=PROVIDER,
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
)


def detect_repo_root(cwd: Path) -> Optional[Path]:
    """Detect git repository root."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, timeout=10
        )
        root = Path(r.stdout.strip())
        return root if r.returncode == 0 and root.exists() else None
    except Exception:
        return None


REPO_ROOT = detect_repo_root(WORKDIR)


class GitAvailable:
    """Check git availability and provide worktree commands."""
    @staticmethod
    def is_available() -> bool:
        return REPO_ROOT is not None

    @staticmethod
    def worktree_add(name: str, path: Path, branch: str) -> str:
        if not REPO_ROOT:
            return "[Error]: Not in a git repository"
        try:
            r = subprocess.run(
                ["git", "worktree", "add", str(path), branch],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
            )
            if r.returncode == 0:
                return f"Created worktree '{name}' at {path}"
            return f"[Error]: {r.stderr.strip()}"
        except Exception as e:
            return f"[Error]: {e}"

    @staticmethod
    def worktree_remove(name: str, path: Path, force: bool = False) -> str:
        if not REPO_ROOT:
            return "[Error]: Not in a git repository"
        try:
            args = ["git", "worktree", "remove", str(path)]
            if force:
                args.append("--force")
            r = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return f"Removed worktree '{name}'"
            return f"[Error]: {r.stderr.strip()}"
        except Exception as e:
            return f"[Error]: {e}"

    @staticmethod
    def status(path: Path) -> str:
        try:
            r = subprocess.run(["git", "status", "--porcelain"],
                             cwd=path, capture_output=True, text=True, timeout=10)
            return r.stdout.strip() or "(clean)"
        except Exception as e:
            return f"[Error]: {e}"


WORKTREES = GitAvailable()


# ========== EventBus ==========

class EventBus:
    """Append-only lifecycle events for worktree observability."""

    def __init__(self, event_log_path: Path):
        self.path = event_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def emit(self, event: str, task_id: Optional[int] = None, wt_name: Optional[str] = None,
             error: Optional[str] = None, **extra):
        payload = {"event": event, "ts": time.time()}
        if task_id is not None:
            payload["task_id"] = task_id
        if wt_name:
            payload["worktree"] = wt_name
        if error:
            payload["error"] = error
        payload.update(extra)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def list_recent(self, limit: int = 20) -> str:
        n = max(1, min(int(limit or 20), 200))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        items = []
        for line in lines[-n:]:
            try:
                items.append(json.loads(line))
            except Exception:
                items.append({"event": "parse_error", "raw": line})
        return json.dumps(items, indent=2)


EVENTS = EventBus(EVENTS_PATH)


# ========== WorktreeRegistry ==========

class WorktreeRegistry:
    """Manage worktree directory isolation."""

    def __init__(self, worktrees_dir: Path):
        self.dir = worktrees_dir
        self.index_path = self.dir / "index.json"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.worktrees = self._load()

    def _load(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {"worktrees": []}

    def _save(self):
        self.index_path.write_text(json.dumps(self.worktrees, indent=2))

    def create(self, name: str, task_id: Optional[int] = None, base_ref: str = "HEAD") -> dict:
        """Create a new worktree."""
        if any(w["name"] == name for w in self.worktrees["worktrees"]):
            raise ValueError(f"Worktree '{name}' already exists")

        path = self.dir / name
        branch = f"wt/{name}"

        # Create the worktree directory and init git
        if WORKTREES.is_available():
            result = WORKTREES.worktree_add(name, path, branch)
            if "[Error]" in result:
                raise ValueError(result)
        else:
            path.mkdir(parents=True, exist_ok=True)
            result = f"Created directory {path} (git not available)"

        wt = {
            "name": name,
            "path": str(path),
            "branch": branch,
            "task_id": task_id,
            "status": "active",
            "created_at": time.time(),
            "last_used": time.time(),
        }
        self.worktrees["worktrees"].append(wt)
        self._save()
        EVENTS.emit("worktree_created", wt_name=name, task_id=task_id)
        return wt

    def get(self, name: str) -> Optional[dict]:
        for w in self.worktrees["worktrees"]:
            if w["name"] == name:
                return w
        return None

    def list_all(self) -> list:
        return self.worktrees["worktrees"]

    def update(self, name: str, updates: dict) -> dict:
        for w in self.worktrees["worktrees"]:
            if w["name"] == name:
                w.update(updates)
                self._save()
                return w
        raise ValueError(f"Worktree '{name}' not found")

    def closeout(self, name: str, action: str, reason: str = "", force: bool = False,
                 complete_task: bool = False) -> dict:
        """Close out a worktree lane."""
        wt = self.get(name)
        if not wt:
            raise ValueError(f"Worktree '{name}' not found")

        task_id = wt.get("task_id")
        path = Path(wt["path"])

        if action == "remove":
            if path.exists():
                result = WORKTREES.worktree_remove(name, path, force)
                EVENTS.emit("worktree_removed", wt_name=name, task_id=task_id, **({"error": result} if "[Error]" in result else {}))
            self.worktrees["worktrees"] = [w for w in self.worktrees["worktrees"] if w["name"] != name]
            self._save()
            if complete_task and task_id:
                TASKS.update(task_id, status="completed")
                EVENTS.emit("task_completed", task_id=task_id)
            return {"action": "removed", "result": result}

        elif action == "keep":
            self.update(name, {"status": "kept", "closeout_reason": reason})
            EVENTS.emit("worktree_kept", wt_name=name, task_id=task_id, reason=reason)
            return {"action": "kept", "reason": reason}

        raise ValueError(f"Unknown action: {action}")


REGISTRY = WorktreeRegistry(WORKTREES_DIR)


# ========== TaskManager ==========

class TaskManager:
    """Persistent task board with optional worktree binding."""

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = []
        for f in self.dir.glob("task_*.json"):
            try:
                parts = f.stem.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    ids.append(int(parts[1]))
            except Exception:
                pass
        return max(ids) if ids else 0

    def _path(self, task_id: int) -> Path:
        return self.dir / f"task_{task_id}.json"

    def _load(self, task_id: int) -> dict:
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        self._path(task["id"]).write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> dict:
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "owner": "",
            "worktree": "",
            "worktree_state": "unbound",
            "last_worktree": "",
            "closeout": None,
            "blockedBy": [],
            "blocks": [],
        }
        self._save(task)
        self._next_id += 1
        EVENTS.emit("task_created", task_id=task["id"])
        return task

    def get(self, task_id: int) -> dict:
        return self._load(task_id)

    def update(self, task_id: int, status: Optional[str] = None, owner: Optional[str] = None) -> dict:
        task = self._load(task_id)
        if owner is not None:
            task["owner"] = owner
        if status:
            if status not in ("pending", "in_progress", "completed", "deleted"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                for f in self.dir.glob("task_*.json"):
                    t = json.loads(f.read_text())
                    if task_id in t.get("blockedBy", []):
                        t["blockedBy"].remove(task_id)
                        self._save(t)
        self._save(task)
        EVENTS.emit("task_updated", task_id=task_id, status=status)
        return task

    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> dict:
        task = self._load(task_id)
        if task.get("worktree"):
            task["last_worktree"] = task["worktree"]
        task["worktree"] = worktree
        task["worktree_state"] = "active"
        if owner:
            task["owner"] = owner
        self._save(task)
        EVENTS.emit("task_bound_worktree", task_id=task_id, worktree=worktree)
        return task

    def list_all(self) -> list:
        return [json.loads(f.read_text()) for f in sorted(self.dir.glob("task_*.json"))]


TASKS = TaskManager(TASKS_DIR)


# ========== Base Tool Implementations ==========

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "[Error]: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()[:50000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "[Error]: Timeout (120s)"

def run_read(path: str) -> str:
    try:
        return safe_path(path).read_text()[:50000]
    except Exception as e:
        return f"[Error]: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"[Error]: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"[Error]: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"[Error]: {e}"


# ========== Agent State (LangGraph Native) ==========

class AgentState(TypedDict):
    """Agent state with langgraph native checkpoint support."""
    messages: Annotated[list, add_messages]
    # LangGraph native: state-based tracking for worktree lifecycle
    worktree_events: list[dict]  # Recent worktree lifecycle events
    active_worktrees: list[str]  # Track active worktree names


# ========== Tool Functions ==========

@tool
def bash_tool(command: str) -> str:
    """Run a shell command."""
    return run_bash(command)

@tool
def read_file(path: str) -> str:
    """Read file contents."""
    return run_read(path)

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return run_write(path, content)

@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in a file."""
    return run_edit(path, old_text, new_text)

@tool
def task_create(subject: str, description: str = "") -> str:
    """Create a new task."""
    return json.dumps(TASKS.create(subject, description), indent=2)

@tool
def task_list() -> str:
    """List all tasks with status, owner, and worktree binding."""
    tasks = TASKS.list_all()
    if not tasks:
        return "No tasks."
    lines = []
    for t in tasks:
        marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]", "deleted": "[-]"}.get(t["status"], "[?]")
        owner = f" owner={t['owner']}" if t.get("owner") else ""
        wt = f" @ {t['worktree']}" if t.get("worktree") else ""
        lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{wt}")
    return "\n".join(lines)

@tool
def task_get(task_id: int) -> str:
    """Get task details by ID."""
    try:
        return json.dumps(TASKS.get(task_id), indent=2)
    except ValueError as e:
        return f"[Error]: {e}"

@tool
def task_update(task_id: int, status: Optional[str] = None, owner: Optional[str] = None) -> str:
    """Update task status or owner."""
    try:
        return json.dumps(TASKS.update(task_id, status, owner), indent=2)
    except ValueError as e:
        return f"[Error]: {e}"

@tool
def task_bind_worktree(task_id: int, worktree: str, owner: str = "") -> str:
    """Bind a task to a worktree name."""
    try:
        return json.dumps(TASKS.bind_worktree(task_id, worktree, owner), indent=2)
    except ValueError as e:
        return f"[Error]: {e}"

@tool
def worktree_create(name: str, task_id: Optional[int] = None, base_ref: str = "HEAD") -> str:
    """Create a git worktree and optionally bind it to a task."""
    try:
        wt = REGISTRY.create(name, task_id, base_ref)
        if task_id:
            TASKS.bind_worktree(task_id, name)
        return json.dumps(wt, indent=2)
    except ValueError as e:
        return f"[Error]: {e}"

@tool
def worktree_list() -> str:
    """List worktrees."""
    wts = REGISTRY.list_all()
    if not wts:
        return "No worktrees."
    lines = []
    for w in wts:
        lines.append(f"  {w['name']}: {w['path']} ({w['status']})")
    return "\n".join(lines)

@tool
def worktree_status(name: str) -> str:
    """Show git status for a worktree."""
    wt = REGISTRY.get(name)
    if not wt:
        return f"[Error]: Worktree '{name}' not found"
    return WORKTREES.status(Path(wt["path"]))

@tool
def worktree_run(name: str, command: str) -> str:
    """Run a shell command in a named worktree directory."""
    wt = REGISTRY.get(name)
    if not wt:
        return f"[Error]: Worktree '{name}' not found"
    path = Path(wt["path"])
    try:
        r = subprocess.run(command, shell=True, cwd=path, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()[:50000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "[Error]: Timeout (120s)"

@tool
def worktree_closeout(name: str, action: str, reason: str = "", force: bool = False,
                      complete_task: bool = False) -> str:
    """Close out a lane by keeping it for follow-up or removing it."""
    try:
        return json.dumps(REGISTRY.closeout(name, action, reason, force, complete_task), indent=2)
    except ValueError as e:
        return f"[Error]: {e}"

@tool
def worktree_events(limit: int = 20) -> str:
    """List recent lifecycle events."""
    return EVENTS.list_recent(limit)


# Define tool list and tool node
agent_tools = [
    bash_tool, read_file, write_file, edit_file,
    task_create, task_list, task_get, task_update, task_bind_worktree,
    worktree_create, worktree_list, worktree_status, worktree_run, worktree_closeout, worktree_events
]
tool_node = ToolNode(agent_tools, handle_tool_errors=True)
model_with_tools = model.bind_tools(agent_tools)


# ========== Graph Nodes ==========

SYSTEM_PROMPT = f"""You are a coding agent at {WORKDIR}. Use task + worktree tools for multi-task work.

For parallel or risky changes: create tasks, allocate worktree lanes, run commands in those lanes, then choose keep/remove for closeout.

Key insight: "Isolate by directory, coordinate by task ID."

LangGraph native features:
- Session persistence via checkpoint
- State-based worktree lifecycle tracking
- Event injection for observability
"""


def call_model(state: AgentState) -> dict:
    """Call the model with current messages.
    LangGraph native: injects worktree events into context."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # Inject recent worktree events (LangGraph native: state-based event tracking)
    if state.get("worktree_events"):
        events_text = "\n".join(
            f"[{e.get('event', 'unknown')}] at {e.get('ts', 0)}" for e in state["worktree_events"][-5:]
        )
        messages.append(HumanMessage(content=f"<recent-worktree-events>\n{events_text}\n</recent-worktree-events>"))

    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", END]:
    """Check if there are tool calls to execute."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# Build the graph with checkpoint (LangGraph native)
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

# Compile with checkpoint for session persistence (LangGraph native)
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


def get_session_config(thread_id: str) -> dict:
    """LangGraph native: Get session config for checkpointing."""
    return {"configurable": {"thread_id": thread_id}}


def run_agent(query: str, thread_id: str = "wt_session_1") -> dict:
    """Run the agent with checkpoint support for session persistence."""
    config = get_session_config(thread_id)

    # Check for existing state
    existing = graph.get_state(config)
    existing_events = existing.values.get("worktree_events", []) if existing else []
    existing_wts = existing.values.get("active_worktrees", []) if existing else []

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "worktree_events": existing_events,
        "active_worktrees": existing_wts,
    }

    for event in graph.stream(initial_state, config):
        node_name = list(event.keys())[0]
        if node_name == "agent":
            response = event[node_name]["messages"][-1]
            if hasattr(response, 'content') and response.content:
                print(f"\nAssistant: {response.content}")


if __name__ == "__main__":
    print("Worktree Task Isolation (phase19) - LangGraph Native Patterns")
    print("Features: Checkpoint persistence, state-based worktree tracking")
    if not WORKTREES.is_available():
        print("Note: Not in a git repo. Worktree tools will create directories only.")
    print("Type 'exit' or 'q' to quit\n")

    thread_id = "wt_session_1"
    config = get_session_config(thread_id)

    # Resume from checkpoint
    existing = graph.get_state(config)
    if existing and existing.values.get("messages"):
        print(f"[Resuming session with {len(existing.values['messages'])} messages]\n")

    while True:
        try:
            query = input(f"\033[36m{thread_id} >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        run_agent(query, thread_id)