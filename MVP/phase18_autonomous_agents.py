#!/usr/bin/env python3
"""
phase18_autonomous_agents.py - Autonomous Agents

Idle cycle with task board polling, auto-claiming unclaimed tasks, and
identity re-injection after context compression.

Teammate lifecycle:
  spawn -> work -> idle (poll for inbox/unclaimed tasks) -> work -> ... -> shutdown

Key idea: an idle teammate can safely claim ready work instead of waiting
for every assignment from the lead.
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
LLM_DIR = WORKDIR / ".mini-agent-cli"
TEAM_DIR = LLM_DIR / "team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = LLM_DIR / ".tasks"
REQUESTS_DIR = TEAM_DIR / "requests"
CLAIM_EVENTS_PATH = TASKS_DIR / "claim_events.jsonl"

POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

model = init_chat_model(
    MODEL_ID,
    model_provider=PROVIDER,
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
)

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval",
    "plan_approval_response",
}

_claim_lock = threading.Lock()


class MessageBus:
    """JSONL inbox per teammate."""

    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        sender: str,
        to: str,
        content: str,
        msg_type: str = "message",
        extra: dict = None,
    ) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


class RequestStore:
    """Durable protocol request records."""

    def __init__(self, base_dir: Path):
        self.dir = base_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, request_id: str) -> Path:
        return self.dir / f"{request_id}.json"

    def create(self, record: dict) -> dict:
        request_id = record["request_id"]
        with self._lock:
            self._path(request_id).write_text(json.dumps(record, indent=2))
        return record

    def get(self, request_id: str) -> dict | None:
        path = self._path(request_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def update(self, request_id: str, **changes) -> dict | None:
        with self._lock:
            record = self.get(request_id)
            if not record:
                return None
            record.update(changes)
            record["updated_at"] = time.time()
            self._path(request_id).write_text(json.dumps(record, indent=2))
        return record


REQUEST_STORE = RequestStore(REQUESTS_DIR)


def _append_claim_event(payload: dict):
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    with CLAIM_EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _task_allows_role(task: dict, role: str | None) -> bool:
    required_role = task.get("claim_role") or task.get("required_role") or ""
    if not required_role:
        return True
    return bool(role) and role == required_role


def is_claimable_task(task: dict, role: str | None = None) -> bool:
    return (
        task.get("status") == "pending"
        and not task.get("owner")
        and not task.get("blockedBy")
        and _task_allows_role(task, role)
    )


def scan_unclaimed_tasks(role: str | None = None) -> list:
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if is_claimable_task(task, role):
            unclaimed.append(task)
    return unclaimed


def claim_task(
    task_id: int,
    owner: str,
    role: str | None = None,
    source: str = "manual",
) -> str:
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        if not is_claimable_task(task, role):
            return f"Error: Task {task_id} is not claimable for role={role or '(any)'}"
        task["owner"] = owner
        task["status"] = "in_progress"
        task["claimed_at"] = time.time()
        task["claim_source"] = source
        path.write_text(json.dumps(task, indent=2))
    _append_claim_event(
        {
            "event": "task.claimed",
            "task_id": task_id,
            "owner": owner,
            "role": role,
            "source": source,
            "ts": time.time(),
        }
    )
    return f"Claimed task #{task_id} for {owner} via {source}"


def make_identity_block(name: str, role: str, team_name: str) -> dict:
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }


def ensure_identity_context(messages: list, name: str, role: str, team_name: str):
    if messages and "<identity>" in str(messages[0].get("content", "")):
        return
    messages.insert(0, make_identity_block(name, role, team_name))
    messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})


class TeammateManager:
    """Autonomous teammate registry with idle polling."""

    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def _set_status(self, name: str, status: str):
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()
        thread = threading.Thread(
            target=self._loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        teammate_model = init_chat_model(
            MODEL_ID,
            model_provider=PROVIDER,
            temperature=0,
            max_tokens=8000,
            base_url=BASE_URL,
            api_key=API_KEY,
        )
        sys_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            f"Use idle tool when you have no more work. You will auto-claim new tasks."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        while True:
            for _ in range(50):
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})

                try:
                    response = teammate_model.bind_tools(tools).invoke(messages)
                except Exception:
                    self._set_status(name, "idle")
                    return

                messages.append({"role": "assistant", "content": response.content})
                if not (hasattr(response, "tool_calls") and response.tool_calls):
                    break

                results = []
                idle_requested = False
                for tc in response.tool_calls:
                    if tc["name"] == "idle":
                        idle_requested = True
                        output = "Entering idle phase. Will poll for new tasks."
                    else:
                        output = self._exec(name, tc["name"], tc["args"])
                    print(f"  [{name}] {tc['name']}: {str(output)[:120]}")
                    results.append({"role": "tool", "content": output})

                messages.append({"role": "user", "content": results})
                if idle_requested:
                    break

            self._set_status(name, "idle")
            resume = False
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)
            for _ in range(polls):
                time.sleep(POLL_INTERVAL)
                inbox = BUS.read_inbox(name)
                if inbox:
                    ensure_identity_context(messages, name, role, team_name)
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break

                unclaimed = scan_unclaimed_tasks(role)
                if unclaimed:
                    task = unclaimed[0]
                    claim_result = claim_task(
                        task["id"], name, role=role, source="auto"
                    )
                    if claim_result.startswith("Error:"):
                        continue
                    task_prompt = (
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                        f"{task.get('description', '')}</auto-claimed>"
                    )
                    ensure_identity_context(messages, name, role, team_name)
                    messages.append({"role": "user", "content": task_prompt})
                    messages.append(
                        {"role": "assistant", "content": f"{claim_result}. Working on it."}
                    )
                    resume = True
                    break

            if not resume:
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"])
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(
                sender, args["to"], args["content"], args.get("msg_type", "message")
            )
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            REQUEST_STORE.update(
                req_id,
                status="approved" if args["approve"] else "rejected",
                resolved_by=sender,
                resolved_at=time.time(),
                response={"approve": args["approve"], "reason": args.get("reason", "")},
            )
            BUS.send(
                sender, "lead", args.get("reason", ""), "shutdown_response",
                {"request_id": req_id, "approve": args["approve"]},
            )
            return f"Shutdown {'approved' if args['approve'] else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            REQUEST_STORE.create(
                {
                    "request_id": req_id,
                    "kind": "plan_approval",
                    "from": sender,
                    "to": "lead",
                    "status": "pending",
                    "plan": plan_text,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }
            )
            BUS.send(
                sender, "lead", plan_text, "plan_approval",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for approval."
        if tool_name == "claim_task":
            return claim_task(
                args["task_id"],
                sender,
                role=self._find_member(sender).get("role") if self._find_member(sender) else None,
                source="manual",
            )
        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        return [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write content to a file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Replace exact text in a file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
            {"name": "send_message", "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string"}}, "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "shutdown_response", "description": "Respond to a shutdown request.",
             "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["request_id", "approve"]}},
            {"name": "plan_approval", "description": "Submit a plan for lead approval.",
             "input_schema": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}},
            {"name": "idle", "description": "Signal that you have no more work.",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "claim_task", "description": "Claim a task from the task board.",
             "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
        ]

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


def _safe(p: str) -> Path:
    p = (WORKDIR / p).resolve()
    if not p.is_relative_to(WORKDIR):
        raise ValueError(p)
    return p


def _run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "[Error]: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        return "[Error]: Timeout (120s)"
    return (r.stdout + r.stderr).strip() or "(no output)"


def _run_read(path: str, limit: int = None) -> str:
    try:
        lines = _safe(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)
    except Exception as e:
        return f"[Error]: {e}"


def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"[Error]: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = _safe(path)
        c = fp.read_text()
        if old_text not in c:
            return f"[Error]: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"[Error]: {e}"


def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    REQUEST_STORE.create({
        "request_id": req_id, "kind": "shutdown", "from": "lead", "to": teammate,
        "status": "pending", "created_at": time.time(), "updated_at": time.time(),
    })
    BUS.send("lead", teammate, "Please shut down gracefully.", "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent to '{teammate}'"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    req = REQUEST_STORE.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    REQUEST_STORE.update(request_id, status="approved" if approve else "rejected",
                         reviewed_by="lead", resolved_at=time.time(), feedback=feedback)
    BUS.send("lead", req["from"], feedback, "plan_approval_response",
             {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"Plan {'approved' if approve else 'rejected'} for '{req['from']}'"


@tool
def bash(command: str) -> str:
    return _run_bash(command)

@tool
def read_file(path: str, limit: Optional[int] = None) -> str:
    return _run_read(path, limit)

@tool
def write_file(path: str, content: str) -> str:
    return _run_write(path, content)

@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    return _run_edit(path, old_text, new_text)

@tool
def spawn_teammate(name: str, role: str, prompt: str) -> str:
    return TEAM.spawn(name, role, prompt)

@tool
def list_teammates() -> str:
    return TEAM.list_all()

@tool
def send_message(to: str, content: str, msg_type: str = "message") -> str:
    return BUS.send("lead", to, content, msg_type)

@tool
def read_inbox() -> str:
    return json.dumps(BUS.read_inbox("lead"), indent=2)

@tool
def broadcast(content: str) -> str:
    return BUS.broadcast("lead", content, TEAM.member_names())

@tool
def shutdown_request(teammate: str) -> str:
    return handle_shutdown_request(teammate)

@tool
def shutdown_response(request_id: str) -> str:
    return json.dumps(REQUEST_STORE.get(request_id) or {"error": "not found"})

@tool
def plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    return handle_plan_review(request_id, approve, feedback)

@tool
def idle() -> str:
    return "Lead does not idle."

@tool
def claim_task_tool(task_id: int) -> str:
    return claim_task(task_id, "lead")


ALL_TOOLS = [
    bash, read_file, write_file, edit_file,
    spawn_teammate, list_teammates, send_message, read_inbox, broadcast,
    shutdown_request, shutdown_response, plan_review, idle, claim_task_tool
]
tool_node = ToolNode(ALL_TOOLS, handle_tool_errors=True)
model_with_tools = model.bind_tools(ALL_TOOLS)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM = f"You are a team lead at {WORKDIR}. Teammates are autonomous -- they find work themselves."


def call_model(state: AgentState) -> dict:
    inbox = BUS.read_inbox("lead")
    messages_with_system = [SystemMessage(content=SYSTEM)] + state["messages"]
    if inbox:
        messages_with_system.append(HumanMessage(content=f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"))
        messages_with_system.append(HumanMessage(content="Noted inbox messages."))
    response = model_with_tools.invoke(messages_with_system)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
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
    print("[Autonomous agents enabled: teammates auto-claim tasks]")
    state: AgentState = {"messages": []}

    while True:
        try:
            q = input("\033[36mphase18 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in ("q", "exit", ""):
            break
        if q.strip() == "/team":
            print(TEAM.list_all())
            continue
        if q.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                t = json.loads(f.read_text())
                marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
                owner = f" @{t['owner']}" if t.get("owner") else ""
                print(f"  {marker} #{t['id']}: {t['subject']}{owner}")
            continue

        state["messages"].append(HumanMessage(content=q))
        state = graph.invoke(state)
        print()
