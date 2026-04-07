#!/usr/bin/env python3
from __future__ import annotations
"""
phase15_cron_scheduler.py - Cron / Scheduled Tasks with LangGraph Native Patterns

The agent can schedule prompts for future execution using standard cron
expressions. When a schedule matches the current time, it pushes a
notification back into the main conversation loop.

    Cron expression: 5 fields
    +-------+-------+-------+-------+-------+
    | min   | hour  | dom   | month | dow   |
    | 0-59  | 0-23  | 1-31  | 1-12  | 0-6   |
    +-------+-------+-------+-------+-------+

    Examples:
      "*/5 * * * *"   -> every 5 minutes
      "0 9 * * 1"     -> Monday 9:00 AM
      "30 14 * * *"   -> daily 2:30 PM

Key insight: Scheduling remembers future work, then hands it back to
the same main loop when the time arrives.

LangGraph native patterns:
- MemorySaver checkpointer for session persistence
- State updates for scheduled task injection
- Interrupt-based task cancellation
"""
import json
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue, Empty
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
SCHEDULED_TASKS_FILE = STORAGE_DIR / "scheduled_tasks.json"
AUTO_EXPIRY_DAYS = 7
JITTER_MINUTES = [0, 30]
JITTER_OFFSET_MAX = 4

model = init_chat_model(
    MODEL_ID,
    model_provider=PROVIDER,
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
)


# ========== Cron Utilities ==========

def cron_matches(expr: str, dt: datetime) -> bool:
    """
    Check if a 5-field cron expression matches a given datetime.
    Fields: minute hour day-of-month month day-of-week
    Supports: * (any), */N (every N), N (exact), N-M (range), N,M (list)
    """
    fields = expr.strip().split()
    if len(fields) != 5:
        return False
    values = [dt.minute, dt.hour, dt.day, dt.month, dt.weekday()]
    cron_dow = (dt.weekday() + 1) % 7
    values[4] = cron_dow
    ranges = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    for field, value, (lo, hi) in zip(fields, values, ranges):
        if not _field_matches(field, value, lo, hi):
            return False
    return True


def _field_matches(field: str, value: int, lo: int, hi: int) -> bool:
    """Match a single cron field against a value."""
    if field == "*":
        return True
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, step_str = part.split("/", 1)
            step = int(step_str)
        if part == "*":
            if (value - lo) % step == 0:
                return True
        elif "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            if start <= value <= end and (value - start) % step == 0:
                return True
        else:
            if int(part) == value:
                return True
    return False


# ========== CronScheduler ==========

class CronScheduler:
    """
    Manage scheduled tasks with background checking.
    Two persistence modes:
    +--------------------+-------------------------------+
    | session-only       | In-memory list, lost on exit  |
    | durable            | .mini-agent-cli/scheduled_tasks.json  |
    +--------------------+-------------------------------+

    Two trigger modes:
    +--------------------+-------------------------------+
    | recurring          | Repeats until deleted or      |
    |                    | 7-day auto-expiry             |
    | one-shot           | Fires once, then auto-deleted |
    +--------------------+-------------------------------+
    """

    def __init__(self):
        self.tasks = []
        self.queue = Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self._last_check_minute = -1

    def start(self):
        """Load durable tasks and start the background check thread."""
        self._load_durable()
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()
        count = len(self.tasks)
        if count:
            print(f"[Cron] Loaded {count} scheduled tasks")

    def stop(self):
        """Stop the background thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def create(
        self, cron_expr: str, prompt: str, recurring: bool = True, durable: bool = False
    ) -> str:
        """Create a new scheduled task. Returns the task ID."""
        task_id = str(uuid.uuid4())[:8]
        now = time.time()
        task = {
            "id": task_id,
            "cron": cron_expr,
            "prompt": prompt,
            "recurring": recurring,
            "durable": durable,
            "createdAt": now,
        }
        if recurring:
            task["jitter_offset"] = self._compute_jitter(cron_expr)
        self.tasks.append(task)
        if durable:
            self._save_durable()
        mode = "recurring" if recurring else "one-shot"
        store = "durable" if durable else "session-only"
        return f"Created task {task_id} ({mode}, {store}): cron={cron_expr}"

    def delete(self, task_id: str) -> str:
        """Delete a scheduled task by ID."""
        before = len(self.tasks)
        self.tasks = [t for t in self.tasks if t["id"] != task_id]
        if len(self.tasks) < before:
            self._save_durable()
            return f"Deleted task {task_id}"
        return f"Task {task_id} not found"

    def list_tasks(self) -> str:
        """List all scheduled tasks."""
        if not self.tasks:
            return "No scheduled tasks."
        lines = []
        for t in self.tasks:
            mode = "recurring" if t["recurring"] else "one-shot"
            store = "durable" if t["durable"] else "session"
            age_hours = (time.time() - t["createdAt"]) / 3600
            lines.append(
                f"  {t['id']}  {t['cron']}  [{mode}/{store}] "
                f"({age_hours:.1f}h old): {t['prompt'][:60]}"
            )
        return "\n".join(lines)

    def drain_notifications(self) -> list[str]:
        """Drain all pending notifications from the queue."""
        notifications = []
        while True:
            try:
                notifications.append(self.queue.get_nowait())
            except Empty:
                break
        return notifications

    def _compute_jitter(self, cron_expr: str) -> int:
        """If cron targets :00 or :30, return a small offset (1-4 minutes)."""
        fields = cron_expr.strip().split()
        if len(fields) < 1:
            return 0
        minute_field = fields[0]
        try:
            minute_val = int(minute_field)
            if minute_val in JITTER_MINUTES:
                return (hash(cron_expr) % JITTER_OFFSET_MAX) + 1
        except ValueError:
            pass
        return 0

    def _check_loop(self):
        """Background thread: check every second if any task is due."""
        while not self._stop_event.is_set():
            now = datetime.now()
            current_minute = now.hour * 60 + now.minute
            if current_minute != self._last_check_minute:
                self._last_check_minute = current_minute
                self._check_tasks(now)
            self._stop_event.wait(timeout=1)

    def _check_tasks(self, now: datetime):
        """Check all tasks against current time, fire matches."""
        expired = []
        fired_oneshots = []
        for task in self.tasks:
            age_days = (time.time() - task["createdAt"]) / 86400
            if task["recurring"] and age_days > AUTO_EXPIRY_DAYS:
                expired.append(task["id"])
                continue
            check_time = now
            jitter = task.get("jitter_offset", 0)
            if jitter:
                check_time = now - timedelta(minutes=jitter)
            if cron_matches(task["cron"], check_time):
                notification = f"[Scheduled task {task['id']}]: {task['prompt']}"
                self.queue.put(notification)
                task["last_fired"] = time.time()
                print(f"[Cron] Fired: {task['id']}")
                if not task["recurring"]:
                    fired_oneshots.append(task["id"])
        if expired or fired_oneshots:
            remove_ids = set(expired) | set(fired_oneshots)
            self.tasks = [t for t in self.tasks if t["id"] not in remove_ids]
            for tid in expired:
                print(f"[Cron] Auto-expired: {tid} (older than {AUTO_EXPIRY_DAYS} days)")
            for tid in fired_oneshots:
                print(f"[Cron] One-shot completed and removed: {tid}")
            self._save_durable()

    def _load_durable(self):
        """Load durable tasks from .mini-agent-cli/scheduled_tasks.json."""
        if not SCHEDULED_TASKS_FILE.exists():
            return
        try:
            data = json.loads(SCHEDULED_TASKS_FILE.read_text())
            self.tasks = [t for t in data if t.get("durable")]
        except Exception as e:
            print(f"[Cron] Error loading tasks: {e}")

    def _save_durable(self):
        """Save durable tasks to disk."""
        durable = [t for t in self.tasks if t.get("durable")]
        SCHEDULED_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCHEDULED_TASKS_FILE.write_text(json.dumps(durable, indent=2) + "\n")


# Global scheduler
scheduler = CronScheduler()


# ========== Agent State (LangGraph Native) ==========

class AgentState(TypedDict):
    """Agent state with langgraph native checkpoint support."""
    messages: Annotated[list, add_messages]
    scheduled_notifications: list[str]  # LangGraph native: state-based notification tracking
    active_schedules: list[dict]  # LangGraph native: track active scheduled tasks in state


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
def cron_create(
    cron: str,
    prompt: str,
    recurring: bool = True,
    durable: bool = False,
) -> str:
    """
    Schedule a task with a cron expression.

    Args:
        cron: 5-field cron expression (min hour dom month dow)
        prompt: The prompt to inject when the task fires
        recurring: True=repeat, False=fire once then delete
        durable: True=persist to disk, False=session-only
    """
    return scheduler.create(cron, prompt, recurring, durable)


@tool
def cron_delete(id: str) -> str:
    """Delete a scheduled task by ID."""
    return scheduler.delete(id)


@tool
def cron_list() -> str:
    """List all scheduled tasks."""
    return scheduler.list_tasks()


# Define tool list and tool node
agent_tools = [bash_tool, read_file, write_file, edit_file, cron_create, cron_delete, cron_list]
tool_node = ToolNode(agent_tools, handle_tool_errors=True)
model_with_tools = model.bind_tools(agent_tools)


# ========== Graph Nodes ==========

SYSTEM_PROMPT = f"""You are a coding agent at {WORKDIR}. You can schedule future work with cron expressions.

Available cron operations:
- cron_create(cron, prompt, recurring, durable): Schedule a task
  Example: cron_create("*/5 * * * *", "Check server status", recurring=True, durable=False)
- cron_delete(id): Delete a scheduled task
- cron_list(): List all scheduled tasks

Cron format: 'min hour dom month dow'
  */5 * * * * = every 5 minutes
  0 9 * * 1 = Monday 9:00 AM
  30 14 * * * = daily 2:30 PM

Scheduled tasks fire automatically and their prompts are injected into the conversation.
"""


def call_model(state: AgentState) -> dict:
    """Call the model with current messages, draining scheduled notifications.
    LangGraph native: uses state for notification injection."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # Drain scheduled task notifications
    notifications = scheduler.drain_notifications()
    if notifications:
        for note in notifications:
            print(f"[Cron notification] {note[:100]}")
            messages.append(HumanMessage(content=note))

    response = model_with_tools.invoke(messages)
    return {"messages": [response], "scheduled_notifications": []}


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


def run_agent(query: str, thread_id: str = "cron_session_1") -> dict:
    """Run the agent with checkpoint support for session persistence."""
    config = get_session_config(thread_id)

    # Check for existing state
    existing = graph.get_state(config)
    existing_msgs = existing.values.get("messages", []) if existing else []
    existing_schedules = existing.values.get("active_schedules", []) if existing else []

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "scheduled_notifications": [],
        "active_schedules": existing_schedules,
    }

    for event in graph.stream(initial_state, config):
        node_name = list(event.keys())[0]
        if node_name == "agent":
            response = event[node_name]["messages"][-1]
            if hasattr(response, 'content') and response.content:
                print(f"\nAssistant: {response.content}")
        elif node_name == "tools":
            pass  # Tools handled internally


if __name__ == "__main__":
    scheduler.start()
    print("Cron Scheduler Agent (phase15) - LangGraph Native Patterns")
    print("Features: Checkpoint persistence, state-based scheduling")
    print("Type 'exit' or 'q' to quit, '/cron' to list scheduled tasks\n")

    thread_id = "cron_session_1"
    config = get_session_config(thread_id)

    # Resume from checkpoint
    existing = graph.get_state(config)
    if existing and existing.values.get("messages"):
        print(f"[Resuming session with {len(existing.values['messages'])} messages]\n")

    while True:
        try:
            query = input(f"\033[36m{thread_id} >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            scheduler.stop()
            break
        if query.strip().lower() in ("q", "exit", ""):
            scheduler.stop()
            break
        if query.strip() == "/cron":
            print(scheduler.list_tasks())
            continue
        run_agent(query, thread_id)