#!/usr/bin/env python3
from __future__ import annotations
"""
phase14_background_tasks.py - Background Tasks with LangGraph Native Patterns

Run slow commands in background threads with langgraph integration.
Uses:
- Checkpoint for session persistence
- State updates for notification handling
- Interrupt for async result polling

Key insight: "Background execution allows the agent to think while waiting."

    Main thread                Background thread
    +-----------------+        +-----------------+
    | agent loop      |        | task executes   |
    | ...             |        | ...             |
    | [LLM call] <---+------- | enqueue(result) |
    |  ^drain queue   |        +-----------------+
    +-----------------+

LangGraph native patterns used:
- MemorySaver checkpointer for session persistence
- State updates for notification injection
- Interrupt-based result polling
- Custom node for background task coordination
"""
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

load_dotenv(override=True)
os.environ["NO_PROXY"] = "*"

MODEL_ID = os.environ.get("AGENCY_LLM_MODEL", os.environ.get("MODEL_ID", "claude-sonnet-4-7"))
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER", "openai")

WORKDIR = Path.cwd()
STORAGE_DIR = WORKDIR / ".mini-agent-cli"
RUNTIME_DIR = STORAGE_DIR / "runtime-tasks"
RUNTIME_DIR.mkdir(exist_ok=True)

STALL_THRESHOLD_S = 45  # seconds before a task is considered stalled

model = init_chat_model(
    MODEL_ID,
    model_provider=PROVIDER,
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
)


# ========== NotificationQueue ==========

class NotificationQueue:
    """
    Priority-based notification queue with same-key folding.
    Folding means a newer message can replace an older message with the
    same key, so the context is not flooded with stale updates.
    """

    PRIORITIES = {"immediate": 0, "high": 1, "medium": 2, "low": 3}

    def __init__(self):
        self._queue = []  # list of (priority, key, message)
        self._lock = threading.Lock()

    def push(self, message: str, priority: str = "medium", key: str = None):
        """Add a message to the queue, folding if key matches an existing entry."""
        with self._lock:
            if key:
                # Fold: replace existing message with same key
                self._queue = [(p, k, m) for p, k, m in self._queue if k != key]
            self._queue.append((self.PRIORITIES.get(priority, 2), key, message))
            self._queue.sort(key=lambda x: x[0])

    def drain(self) -> list[str]:
        """Return all pending messages in priority order and clear the queue."""
        with self._lock:
            messages = [m for _, _, m in self._queue]
            self._queue.clear()
            return messages


# ========== BackgroundManager ==========

class BackgroundManager:
    """
    Threaded execution + notification queue for background tasks.
    Background tasks are runtime execution slots, not the durable task-board
    records from phase13.
    """

    def __init__(self):
        self.dir = RUNTIME_DIR
        self.tasks = {}  # task_id -> {status, result, command, started_at}
        self._notification_queue = NotificationQueue()
        self._lock = threading.Lock()

    def _record_path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.json"

    def _output_path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.log"

    def _persist_task(self, task_id: str):
        record = dict(self.tasks[task_id])
        self._record_path(task_id).write_text(
            json.dumps(record, indent=2, ensure_ascii=False)
        )

    def _preview(self, output: str, limit: int = 500) -> str:
        compact = " ".join((output or "(no output)").split())
        return compact[:limit]

    def run(self, command: str) -> str:
        """Start a background thread, return task_id immediately."""
        task_id = str(uuid.uuid4())[:8]
        output_file = self._output_path(task_id)
        self.tasks[task_id] = {
            "id": task_id,
            "status": "running",
            "result": None,
            "command": command,
            "started_at": time.time(),
            "finished_at": None,
            "result_preview": "",
            "output_file": str(output_file.relative_to(WORKDIR)),
        }
        self._persist_task(task_id)
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True
        )
        thread.start()
        return (
            f"Background task {task_id} started: {command[:80]} "
            f"(output_file={output_file.relative_to(WORKDIR)})"
        )

    def _execute(self, task_id: str, command: str):
        """Thread target: run subprocess, capture output, push to queue."""
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=WORKDIR,
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "[Error]: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"[Error]: {e}"
            status = "error"
        final_output = output or "(no output)"
        preview = self._preview(final_output)
        output_path = self._output_path(task_id)
        output_path.write_text(final_output)
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = final_output
        self.tasks[task_id]["finished_at"] = time.time()
        self.tasks[task_id]["result_preview"] = preview
        self._persist_task(task_id)
        with self._lock:
            self._notification_queue.push(
                {
                    "task_id": task_id,
                    "status": status,
                    "command": command[:80],
                    "preview": preview,
                    "output_file": str(output_path.relative_to(WORKDIR)),
                },
                priority="medium",
                key=task_id,
            )

    def check(self, task_id: str = None) -> str:
        """Check status of one task or list all."""
        if task_id:
            t = self.tasks.get(task_id)
            if not t:
                return f"[Error]: Unknown task {task_id}"
            visible = {
                "id": t["id"],
                "status": t["status"],
                "command": t["command"],
                "result_preview": t.get("result_preview", ""),
                "output_file": t.get("output_file", ""),
            }
            return json.dumps(visible, indent=2, ensure_ascii=False)
        lines = []
        for tid, t in self.tasks.items():
            lines.append(
                f"{tid}: [{t['status']}] {t['command'][:60]} "
                f"-> {t.get('result_preview') or '(running)'}"
            )
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        """Return and clear all pending completion notifications."""
        return self._notification_queue.drain()

    def detect_stalled(self) -> list[str]:
        """
        Return task IDs that have been running longer than STALL_THRESHOLD_S.
        """
        now = time.time()
        stalled = []
        for task_id, info in self.tasks.items():
            if info["status"] != "running":
                continue
            elapsed = now - info.get("started_at", now)
            if elapsed > STALL_THRESHOLD_S:
                stalled.append(task_id)
        return stalled


# Global background manager instance
BG = BackgroundManager()


# ========== Agent State ==========

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_background_tasks: list[dict]  # LangGraph native: track running background tasks
    completed_notifications: list[dict]   # LangGraph native: completed task results
    interrupted_for_polling: bool         # LangGraph native: interrupt flag for polling


# ========== Tool Functions ==========

def safe_path(path: str) -> Path:
    """Resolve path relative to workspace."""
    p = (WORKDIR / path).resolve()
    if not p.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return p


@tool
def bash_tool(command: str) -> str:
    """Run a shell command in the workspace (blocking)."""
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
def background_run(command: str) -> str:
    """Run command in background thread. Returns task_id immediately.
    LangGraph: tracks task in state for later polling."""
    result = BG.run(command)
    # Return will be included in tool result, state update handles tracking
    return result


@tool
def check_background(task_id: Optional[str] = None) -> str:
    """Check background task status. Omit task_id to list all.
    LangGraph: uses state-based tracking."""
    return BG.check(task_id)


@tool
def poll_background_results(task_ids: list[str]) -> str:
    """Poll for results from specific background tasks.
    Returns results for completed tasks."""
    results = []
    for task_id in task_ids:
        info = BG.tasks.get(task_id)
        if info:
            if info["status"] == "completed":
                results.append(f"[{task_id}] completed: {info.get('result_preview', '')}")
            elif info["status"] == "running":
                results.append(f"[{task_id}] still running...")
            elif info["status"] in ("error", "timeout"):
                results.append(f"[{task_id}] {info['status']}: {info.get('result_preview', '')}")
        else:
            results.append(f"[{task_id}] unknown task")
    return "\n".join(results) if results else "No tasks to poll"


# Define tool list and tool node
agent_tools = [bash_tool, read_file, write_file, edit_file, background_run, check_background, poll_background_results]
tool_node = ToolNode(agent_tools, handle_tool_errors=True)
model_with_tools = model.bind_tools(agent_tools)


# ========== Graph Nodes ==========

SYSTEM_PROMPT = f"""You are a coding agent at {WORKDIR}. Use background_run for long-running commands.

Available background operations:
- background_run(command): Run command in background, returns task_id immediately
- check_background(task_id): Check task status (omit task_id to list all)

Background tasks run in separate threads and notify when complete.
"""


def call_model(state: AgentState) -> dict:
    """Call the model with current messages, draining background notifications.
    LangGraph native: uses state for notification injection."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # Inject completed notifications from state if any
    completed = state.get("completed_notifications", [])
    if completed:
        notif_text = "\n".join(
            f"[bg:{n['task_id']}] {n['status']}: {n['preview']} "
            f"(output_file={n['output_file']})"
            for n in completed
        )
        messages.append(
            HumanMessage(content=f"<background-results>\n{notif_text}\n</background-results>")
        )

    response = model_with_tools.invoke(messages)
    return {
        "messages": [response],
        "completed_notifications": [],  # Clear after injection
    }


def background_coordinator(state: AgentState) -> dict:
    """
    LangGraph native: Check for newly completed background tasks and update state.
    This node runs after tools to detect completed background tasks.
    """
    # Drain notifications from BackgroundManager
    notifs = BG.drain_notifications()

    if notifs:
        # Merge with existing notifications
        existing = state.get("completed_notifications", [])
        return {"completed_notifications": existing + notifs}

    return {}


def should_continue(state: AgentState) -> Literal["tools", "check_notifications", END]:
    """Check if there are tool calls or pending notifications."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # Check for pending notifications
    if state.get("pending_background_tasks"):
        # Could route to notification handling
        return "check_notifications"

    return END


def check_notifications(state: AgentState) -> dict:
    """Check for newly completed background tasks."""
    return background_coordinator(state)


# Build the graph with checkpoint
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("check_notifications", check_notifications)

workflow.add_edge(START, "agent")
workflow.add_edge("tools", "agent")
workflow.add_edge("check_notifications", "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "check_notifications": "check_notifications", END: END}
)

# Compile with checkpoint for session persistence (LangGraph native)
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


def get_session_config(thread_id: str) -> dict:
    """LangGraph native: Get session config for checkpointing."""
    return {"configurable": {"thread_id": thread_id}}


def run_agent(query: str, thread_id: str = "bg_session_1") -> dict:
    """Run the agent with checkpoint support for session persistence."""
    config = get_session_config(thread_id)

    # Check existing state for pending notifications
    existing = graph.get_state(config)
    pending = existing.values.get("pending_background_tasks", []) if existing else []

    # Inject notifications from pending tasks
    notifications = []
    for task_id in list(pending):
        info = BG.tasks.get(task_id)
        if info and info["status"] != "running":
            notifications.append({
                "task_id": task_id,
                "status": info["status"],
                "preview": info.get("result_preview", ""),
                "output_file": info.get("output_file", ""),
            })
            pending.remove(task_id)

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "pending_background_tasks": pending,
        "completed_notifications": notifications,
        "interrupted_for_polling": False,
    }

    # Stream with event handling (LangGraph native pattern)
    for event in graph.stream(initial_state, config):
        node_name = list(event.keys())[0]
        if node_name == "agent":
            response = event[node_name]["messages"][-1]
            if hasattr(response, 'content') and response.content:
                print(f"\nAssistant: {response.content}")
        elif node_name == "tools":
            pass  # Tools handled internally


if __name__ == "__main__":
    print("Background Tasks Agent (phase14) - LangGraph Native Patterns")
    print("Features: Checkpoint persistence, state-based notifications, interrupt polling")
    print("Type 'exit' or 'q' to quit\n")

    thread_id = "bg_session_1"
    config = get_session_config(thread_id)

    # Resume from checkpoint if exists
    existing = graph.get_state(config)
    if existing and existing.values.get("messages"):
        print(f"[Resuming session {thread_id} with {len(existing.values['messages'])} messages]\n")

    while True:
        try:
            query = input(f"\033[36m{thread_id} >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        run_agent(query, thread_id)