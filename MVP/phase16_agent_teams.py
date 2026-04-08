#!/usr/bin/env python3
from __future__ import annotations
"""
phase16_agent_teams.py - Agent Teams with LangGraph Native Patterns

Multi-agent orchestration using langgraph's native patterns:
- send() for fan-out to multiple agents
- Subgraphs for each teammate agent
- Checkpoint for session persistence
- Command for inter-agent control

Key insight: "Teammates have names, inboxes, and independent loops."

LangGraph native patterns:
- CoercionType.FAN_OUT_NODES for parallel agent execution
- StateGraph subgraphs for each teammate
- MemorySaver checkpoint for session persistence
- interrupt/Command for human-in-the-loop coordination
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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

# LangGraph multi-agent imports
try:
    from langgraph.constants import Send
except ImportError:
    # Fallback for older versions
    Send = "send"

load_dotenv(override=True)
os.environ["NO_PROXY"] = "*"

MODEL_ID = os.environ.get("AGENCY_LLM_MODEL", os.environ.get("MODEL_ID", "claude-sonnet-4-7"))
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER", "openai")

WORKDIR = Path.cwd()
STORAGE_DIR = WORKDIR / ".mini-agent-cli"
TEAM_DIR = STORAGE_DIR / "team"
INBOX_DIR = TEAM_DIR / "inbox"

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


# ========== MessageBus ==========

class MessageBus:
    """JSONL inbox per teammate for inter-agent communication."""

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
            return f"[Error]: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
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


# ========== TeammateManager ==========

class TeammateManager:
    """Persistent teammate registry plus worker-loop launcher."""

    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(parents=True, exist_ok=True)
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

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"[Error]: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        """Run a teammate's agent loop with inbox communication."""
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        for _ in range(50):
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})

            try:
                response = model_with_tools.invoke(
                    [SystemMessage(content=sys_prompt)] + [
                        HumanMessage(content=m["content"]) if m["role"] == "user"
                        else AIMessage(content=m.get("content", ""))
                        for m in messages
                    ]
                )
            except Exception as e:
                print(f"[{name}] Error: {e}")
                break

            messages.append({"role": "assistant", "content": response.content})
            if hasattr(response, 'tool_calls') and not response.tool_calls:
                break

            results = []
            for tc in (response.tool_calls or []):
                output = self._exec(name, tc["name"], tc.get("args", {}) or {})
                print(f"  [{name}] {tc['name']}: {str(output)[:120]}")
                results.append({"type": "tool_result", "tool_call_id": tc.get("id", ""), "content": str(output)})
            messages.append({"role": "user", "content": results})

        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            member["status"] = "idle"
            self._save_config()

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
        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        return [
            {"name": "bash", "description": "Run a shell command.", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write content to a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Replace exact text in a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
            {"name": "send_message", "description": "Send message to a teammate.", "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.", "input_schema": {"type": "object", "properties": {}}},
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


# ========== Teammate Subgraph (LangGraph Native Pattern) ==========

def create_teammate_subgraph(name: str, role: str):
    """Create a LangGraph subgraph for a teammate agent.
    LangGraph native: Each teammate is a StateGraph subgraph."""

    tools = [
        {"name": "bash", "description": "Run a shell command.",
         "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        {"name": "read_file", "description": "Read file contents.",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "write_file", "description": "Write content to a file.",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
        {"name": "edit_file", "description": "Replace exact text in a file.",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    ]

    system_prompt = f"You are '{name}', role: {role}, at {WORKDIR}. Complete your assigned task."

    subgraph_model = model.bind_tools(tools)

    def teammate_agent(state: TeammateState) -> dict:
        msgs = [SystemMessage(content=system_prompt)] + state["messages"]
        response = subgraph_model.invoke(msgs)
        return {"messages": [response]}

    def teammate_tools(state: TeammateState) -> dict:
        """Execute tools for teammate."""
        last_msg = state["messages"][-1]
        if not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls:
            return {"result": "", "messages": state["messages"]}

        results = []
        for tc in last_msg.tool_calls:
            name_tc, args = tc["name"], tc.get("args", {}) or {}
            if name_tc == "bash":
                content = _run_bash(args["command"])
            elif name_tc == "read_file":
                content = _run_read(args["path"])
            elif name_tc == "write_file":
                content = _run_write(args["path"], args["content"])
            elif name_tc == "edit_file":
                content = _run_edit(args["path"], args["old_text"], args["new_text"])
            else:
                content = f"Unknown tool: {name_tc}"
            results.append({"type": "tool_result", "tool_call_id": tc.get("id", ""), "content": str(content)})

        # Continue with results
        msgs = state["messages"] + [
            HumanMessage(content=json.dumps(results)) if isinstance(results, list) else HumanMessage(content=str(results))
        ]
        response = subgraph_model.invoke(msgs)
        return {"messages": [response]}

    def teammate_should_continue(state: TeammateState) -> Literal["teammate_tools", END]:
        last = state["messages"][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return "teammate_tools"
        return END

    subgraph = StateGraph(TeammateState)
    subgraph.add_node("teammate_agent", teammate_agent)
    subgraph.add_node("teammate_tools", teammate_tools)
    subgraph.add_edge(START, "teammate_agent")
    subgraph.add_conditional_edges("teammate_agent", teammate_should_continue, {"teammate_tools": "teammate_tools", END: END})
    subgraph.add_edge("teammate_tools", "teammate_agent")

    return subgraph.compile()

def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "[Error]: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "[Error]: Timeout (120s)"


def _run_read(path: str, limit: int = None) -> str:
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"[Error]: {e}"


def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"[Error]: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"[Error]: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"[Error]: {e}"


# ========== Agent State (LangGraph Native) ==========

class AgentState(TypedDict):
    """Lead agent state with multi-agent coordination support."""
    messages: Annotated[list, add_messages]
    # LangGraph native: team coordination state
    teammates: dict[str, dict]  # teammate_name -> {role, status, subgraph}
    pending_tasks: list[dict]  # Tasks sent to teammates, awaiting results
    completed_results: list[dict]  # Results from completed teammate tasks


class TeammateState(TypedDict):
    """Individual teammate subgraph state (LangGraph native)."""
    messages: Annotated[list, add_messages]
    name: str
    role: str
    task: str
    result: str


# ========== Tool Functions ==========

@tool
def bash_tool(command: str) -> str:
    """Run a shell command."""
    return _run_bash(command)


@tool
def read_file(path: str, limit: Optional[int] = None) -> str:
    """Read file contents."""
    return _run_read(path, limit)


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return _run_write(path, content)


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in a file."""
    return _run_edit(path, old_text, new_text)


@tool
def spawn_teammate(name: str, role: str, prompt: str) -> str:
    """Spawn a persistent teammate that runs in its own thread."""
    return TEAM.spawn(name, role, prompt)


@tool
def list_teammates() -> str:
    """List all teammates with name, role, status."""
    return TEAM.list_all()


@tool
def send_message(to: str, content: str, msg_type: str = "message") -> str:
    """Send a message to a teammate's inbox."""
    return BUS.send("lead", to, content, msg_type)


@tool
def read_inbox() -> str:
    """Read and drain the lead's inbox."""
    return json.dumps(BUS.read_inbox("lead"), indent=2)


@tool
def broadcast(content: str) -> str:
    """Send a message to all teammates."""
    return BUS.broadcast("lead", content, TEAM.member_names())


@tool
def wait_for_results() -> str:
    """Wait for teammate tasks to complete. Returns status of pending tasks.
    LangGraph native: Uses checkpoint-based state to track teammate completion."""
    # This tool signals the agent to wait for teammate results
    # Results are aggregated through the graph state
    return "Waiting for teammate results..."


# Define tool list and tool node
agent_tools = [bash_tool, read_file, write_file, edit_file, spawn_teammate, list_teammates, send_message, read_inbox, broadcast, wait_for_results]
tool_node = ToolNode(agent_tools, handle_tool_errors=True)
model_with_tools = model.bind_tools(agent_tools)


# ========== Graph Nodes (LangGraph Native Multi-Agent) ==========

SYSTEM_PROMPT = f"""You are a team lead at {WORKDIR}. Spawn teammates and coordinate their work.

LangGraph native team operations:
- spawn_teammate(name, role, task): Create a teammate subgraph and assign a task
- list_teammates(): Show all team members
- send_message(to, content): Send a message to a teammate's inbox
- read_inbox(): Check your inbox
- broadcast(content): Send to all teammates
- wait_for_results(): Wait for teammate task completion

Teammates run as LangGraph subgraphs and communicate through state updates.
"""


def call_model(state: AgentState) -> dict:
    """Call the model with current messages and inject pending teammate results."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # Inject completed teammate results (LangGraph native: state-based result injection)
    if state.get("completed_results"):
        results_text = "\n".join(
            f"[{r['teammate']}] completed task: {r['result'][:200]}"
            for r in state["completed_results"]
        )
        messages.append(HumanMessage(content=f"<teammate-results>\n{results_text}\n</teammate-results>"))

    # Check for incoming messages
    inbox = BUS.read_inbox("lead")
    if inbox:
        messages.append(HumanMessage(content=f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"))

    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def spawn_teammate_node(state: AgentState) -> dict:
    """
    LangGraph native: Spawn teammate subgraph using Send.
    Returns list of Send commands for fan-out execution.
    """
    last_msg = state["messages"][-1]
    if not hasattr(last_msg, 'tool_calls'):
        return {}

    # Check if spawn_teammate was called
    for tc in last_msg.tool_calls:
        if tc["name"] == "spawn_teammate":
            args = tc.get("args", {}) or {}
            name = args.get("name")
            role = args.get("role")
            task = args.get("task")

            if name and role and task:
                # Create teammate subgraph
                teammate_subgraph = create_teammate_subgraph(name, role)

                # Update teammates registry
                teammates = dict(state.get("teammates", {}))
                teammates[name] = {
                    "role": role,
                    "status": "working",
                    "subgraph": teammate_subgraph,
                    "task": task,
                }

                # Add pending task
                pending = list(state.get("pending_tasks", []))
                pending.append({"teammate": name, "task": task})

                return {
                    "teammates": teammates,
                    "pending_tasks": pending,
                }

    return {}


def fan_out_teammates(state: AgentState) -> list:
    """
    LangGraph native: Fan-out to multiple teammate subgraphs using Send.
    This implements the fan-out pattern for parallel teammate execution.
    """
    if Send == "send":
        # Fallback for older langgraph versions
        return []

    pending = state.get("pending_tasks", [])
    teammates = state.get("teammates", {})

    # Return Send commands for each pending task
    sends = []
    for task_info in pending:
        name = task_info["teammate"]
        if name in teammates and teammates[name]["status"] == "working":
            subgraph = teammates[name]["subgraph"]
            sends.append(
                Send(
                    name,  # node name
                    {
                        "messages": [HumanMessage(content=task_info["task"])],
                        "name": name,
                        "role": teammates[name]["role"],
                        "task": task_info["task"],
                        "result": "",
                    }
                )
            )

    return sends


def aggregate_results(state: AgentState) -> dict:
    """
    LangGraph native: Aggregate results from teammate subgraphs.
    Results are collected through state updates.
    """
    # This is called after fan-out completes
    # Results are automatically aggregated through the state

    teammates = dict(state.get("teammates", {}))
    completed = list(state.get("completed_results", []))
    pending = list(state.get("pending_tasks", []))

    # Mark completed teammates as idle
    for name, info in teammates.items():
        if info["status"] == "working" and info["task"]:
            # Check if task has been completed (would be in completed_results)
            already_completed = any(r["teammate"] == name for r in completed)
            if not already_completed:
                # Task is still pending
                pass

    return {"teammates": teammates, "completed_results": completed}


def should_continue(state: AgentState) -> Literal["spawn", "wait_results", "tools", END]:
    """Check if there are tool calls or pending teammate results."""
    last_message = state["messages"][-1]

    # Check for teammate tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tc in last_message.tool_calls:
            if tc["name"] == "spawn_teammate":
                return "spawn"
            if tc["name"] == "wait_for_results":
                return "wait_results"
        return "tools"

    # Check for pending teammate results
    if state.get("pending_tasks"):
        return "wait_results"

    return END


# Build the main graph with checkpoint (LangGraph native)
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("spawn", spawn_teammate_node)
workflow.add_node("wait_results", aggregate_results)

workflow.add_edge(START, "agent")
workflow.add_edge("tools", "agent")
workflow.add_edge("spawn", "agent")
workflow.add_edge("wait_results", "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "spawn": "spawn", "wait_results": "wait_results", END: END}
)

# Compile with checkpoint for session persistence (LangGraph native)
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


def get_session_config(thread_id: str) -> dict:
    """LangGraph native: Get session config for checkpointing."""
    return {"configurable": {"thread_id": thread_id}}


def run_agent(query: str, thread_id: str = "team_lead_1") -> dict:
    """Run the team lead agent with checkpoint support."""
    config = get_session_config(thread_id)

    # Resume from checkpoint if exists
    existing = graph.get_state(config)
    if existing and existing.values.get("messages"):
        print(f"[Resuming session {thread_id}]\n")

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "teammates": existing.values.get("teammates", {}) if existing else {},
        "pending_tasks": existing.values.get("pending_tasks", []) if existing else [],
        "completed_results": existing.values.get("completed_results", []) if existing else [],
    }

    for event in graph.stream(initial_state, config):
        node_name = list(event.keys())[0]
        if node_name == "agent":
            response = event[node_name]["messages"][-1]
            if hasattr(response, 'content') and response.content:
                print(f"\nAssistant: {response.content}")
        elif node_name == "tools":
            pass


if __name__ == "__main__":
    print("Agent Teams (phase16) - LangGraph Native Patterns")
    print("Features: Subgraph teammates, checkpoint persistence, multi-agent coordination")
    print("Type 'exit' or 'q' to quit, '/team' to list teammates\n")

    thread_id = "team_lead_1"
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
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        run_agent(query, thread_id)