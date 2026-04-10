#!/usr/bin/env python3
from __future__ import annotations
"""
phase20_mcp_plugin.py - MCP & Plugin System with LangGraph Native Patterns

External processes can expose tools, and your agent can treat them like
normal tools after a small amount of normalization.

Minimal path:
  1. start an MCP server process
  2. ask it which tools it has
  3. prefix and register those tools
  4. route matching calls to that server

Plugins add one more layer: discovery. A tiny manifest tells the agent which
external server to start.

Key insight: "External tools should enter the same tool pipeline, not form a
completely separate world."

LangGraph native patterns:
- MemorySaver checkpointer for session persistence
- State updates for plugin/MCP tool tracking
- Unified permission gate integration with langgraph state
"""
import json
import os
import subprocess
import threading
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
PERMISSION_MODES = ("default", "auto")

model = init_chat_model(
    MODEL_ID,
    model_provider=PROVIDER,
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
)


# ========== CapabilityPermissionGate ==========

class CapabilityPermissionGate:
    """
    Shared permission gate for native tools and external capabilities.
    MCP does not bypass the control plane.
    Native tools and MCP tools both become normalized capability intents first,
    then pass through the same allow / ask policy.
    """

    READ_PREFIXES = ("read", "list", "get", "show", "search", "query", "inspect")
    HIGH_RISK_PREFIXES = ("delete", "remove", "drop", "shutdown")

    def __init__(self, mode: str = "default"):
        self.mode = mode if mode in PERMISSION_MODES else "default"

    def normalize(self, tool_name: str, tool_input: dict) -> dict:
        if tool_name.startswith("mcp__"):
            _, server_name, actual_tool = tool_name.split("__", 2)
            source = "mcp"
        else:
            server_name = None
            actual_tool = tool_name
            source = "native"
        lowered = actual_tool.lower()
        if actual_tool == "read_file" or lowered.startswith(self.READ_PREFIXES):
            risk = "read"
        elif actual_tool == "bash":
            command = tool_input.get("command", "")
            risk = "high" if any(token in command for token in ("rm -rf", "sudo", "shutdown", "reboot")) else "write"
        elif lowered.startswith(self.HIGH_RISK_PREFIXES):
            risk = "high"
        else:
            risk = "write"
        return {"source": source, "server": server_name, "tool": actual_tool, "risk": risk}

    def check(self, tool_name: str, tool_input: dict) -> dict:
        intent = self.normalize(tool_name, tool_input)
        if intent["risk"] == "read":
            return {"behavior": "allow", "reason": "Read capability", "intent": intent}
        if self.mode == "auto" and intent["risk"] != "high":
            return {"behavior": "allow", "reason": "Auto mode for non-high-risk capability", "intent": intent}
        if intent["risk"] == "high":
            return {"behavior": "ask", "reason": "High-risk capability requires confirmation", "intent": intent}
        return {"behavior": "ask", "reason": "State-changing capability requires confirmation", "intent": intent}

    def ask_user(self, intent: dict, tool_input: dict) -> bool:
        preview = json.dumps(tool_input, ensure_ascii=False)[:200]
        source = f"{intent['source']}:{intent['server']}/{intent['tool']}" if intent.get("server") else f"{intent['source']}:{intent['tool']}"
        print(f"\n  [Permission] {source} risk={intent['risk']}: {preview}")
        try:
            answer = input("  Allow? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")


permission_gate = CapabilityPermissionGate()


# ========== MCPClient ==========

class MCPClient:
    """
    Minimal MCP client over stdio.
    Enough to teach the core architecture without dragging through
    every transport, auth flow, or marketplace detail.
    """

    def __init__(self, server_name: str, command: str, args: list = None, env: dict = None):
        self.server_name = server_name
        self.command = command
        self.args = args or []
        self.env = {**os.environ, **(env or {})}
        self.process = None
        self._request_id = 0

    def connect(self) -> bool:
        try:
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=self.env, text=True
            )
            return True
        except Exception as e:
            print(f"[MCP] Failed to start {self.server_name}: {e}")
            return False

    def _send(self, msg: dict) -> dict:
        if not self.process or self.process.poll() is not None:
            return {"error": "Server not running"}
        request_id = str(self._request_id)
        self._request_id += 1
        msg["jsonrpc"] = "2.0"
        msg["id"] = request_id
        self.process.stdin.write(json.dumps(msg) + "\n")
        self.process.stdin.flush()
        try:
            response = self.process.stdout.readline()
            if response:
                return json.loads(response)
        except Exception:
            pass
        return {"error": "No response"}

    def list_tools(self) -> list:
        result = self._send({"method": "tools/list", "params": {}})
        if "result" in result:
            return result["result"].get("tools", [])
        return []

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        result = self._send({"method": "tools/call", "params": {"name": tool_name, "arguments": arguments}})
        if "result" in result:
            return json.dumps(result["result"], ensure_ascii=False)
        if "error" in result:
            return f"[MCP Error]: {result['error']}"
        return "[MCP Error]: Unknown response"

    def disconnect(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)


# ========== MCPToolRouter ==========

class MCPToolRouter:
    """Route MCP tool calls to the appropriate server."""

    def __init__(self):
        self.clients = {}

    def register_client(self, client: MCPClient):
        self.clients[client.server_name] = client

    def is_mcp_tool(self, tool_name: str) -> bool:
        return tool_name.startswith("mcp__")

    def call(self, tool_name: str, tool_input: dict) -> str:
        if not self.is_mcp_tool(tool_name):
            return "[Error]: Not an MCP tool"
        _, server_name, actual_tool = tool_name.split("__", 2)
        client = self.clients.get(server_name)
        if not client:
            return f"[Error]: Unknown MCP server: {server_name}"
        return client.call_tool(actual_tool, tool_input)

    def get_all_tools(self) -> list:
        """Get all tools from all MCP servers."""
        tools = []
        for server_name, client in self.clients.items():
            for tool_spec in client.list_tools():
                tools.append({
                    "name": f"mcp__{server_name}__{tool_spec.get('name', 'unknown')}",
                    "description": tool_spec.get("description", ""),
                    "input_schema": tool_spec.get("inputSchema", {"type": "object", "properties": {}}),
                    "source": "mcp",
                    "server": server_name,
                })
        return tools


# ========== PluginLoader ==========

class PluginLoader:
    """Load plugins from .mini-agent-cli/plugins/ directory."""

    def __init__(self, plugins_dir: Path = None):
        self.plugins_dir = plugins_dir or (STORAGE_DIR / "plugins")
        self.manifests = {}

    def scan(self) -> list:
        """Scan for plugin manifests."""
        if not self.plugins_dir.exists():
            return []
        found = []
        for manifest_path in self.plugins_dir.glob("**/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
                self.manifests[manifest.get("name", "unknown")] = manifest
                found.append(manifest.get("name", "unknown"))
            except Exception:
                pass
        return found

    def get_mcp_servers(self) -> dict:
        """Get all MCP server configurations."""
        servers = {}
        for name, manifest in self.manifests.items():
            if manifest.get("type") == "mcp":
                servers[name] = manifest.get("config", {})
        return servers


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


NATIVE_TOOLS = [
    {"name": "bash", "description": "Run a shell command.", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]

NATIVE_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"]),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}


# ========== MCP Setup ==========

mcp_router = MCPToolRouter()
plugin_loader = PluginLoader()


def build_tool_pool() -> list:
    """Assemble the complete tool pool: native + MCP tools."""
    all_tools = list(NATIVE_TOOLS)
    mcp_tools = mcp_router.get_all_tools()
    native_names = {t["name"] for t in all_tools}
    for tool_spec in mcp_tools:
        if tool_spec["name"] not in native_names:
            all_tools.append(tool_spec)
    return all_tools


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """Dispatch to native handler or MCP router."""
    if mcp_router.is_mcp_tool(tool_name):
        return mcp_router.call(tool_name, tool_input)
    handler = NATIVE_HANDLERS.get(tool_name)
    if handler:
        return handler(**tool_input)
    return f"[Error]: Unknown tool: {tool_name}"


def normalize_tool_result(tool_name: str, output: str, intent: dict = None) -> str:
    intent = intent or permission_gate.normalize(tool_name, {})
    status = "error" if "Error:" in output or "MCP Error:" in output else "ok"
    payload = {
        "source": intent["source"],
        "server": intent.get("server"),
        "tool": intent["tool"],
        "risk": intent["risk"],
        "status": status,
        "preview": output[:500],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ========== Agent State ==========

class AgentState(TypedDict):
    """Agent state with langgraph native checkpoint support."""
    messages: Annotated[list, add_messages]
    permission_decisions: list[dict]  # LangGraph native: track permission decisions
    tool_pool_info: dict  # LangGraph native: cache tool pool metadata


# ========== Tool Functions (for LangGraph) ==========

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


# Use native + MCP tools combined
all_tools_for_langgraph = NATIVE_TOOLS + mcp_router.get_all_tools()
tool_node = ToolNode([bash_tool, read_file, write_file, edit_file], handle_tool_errors=True)
model_with_tools = model.bind_tools(all_tools_for_langgraph)


# ========== Graph Nodes ==========

SYSTEM_PROMPT = f"""You are a coding agent at {WORKDIR}. Use tools to solve tasks.

You have both native tools and MCP tools available.
MCP tools are prefixed with mcp__{{server}}__{{tool}}.
All capabilities pass through the same permission gate before execution.
LangGraph native: Checkpoint persistence, state-based permission tracking.
"""


def call_model(state: AgentState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", END]:
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
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

# Compile with checkpoint for session persistence (LangGraph native)
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


def get_session_config(thread_id: str) -> dict:
    """LangGraph native: Get session config for checkpointing."""
    return {"configurable": {"thread_id": thread_id}}


def run_agent_with_permission(query: str, thread_id: str = "mcp_session_1"):
    """Run agent with permission checking and checkpoint support."""
    config = get_session_config(thread_id)

    # Resume from checkpoint
    existing = graph.get_state(config)
    if existing and existing.values.get("messages"):
        print(f"[Resuming session {thread_id} with {len(existing.values['messages'])} messages]\n")

    messages = [HumanMessage(content=query)]

    while True:
        response = model_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + messages
        )
        messages.append(response)

        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            break

        results = []
        for tc in response.tool_calls:
            decision = permission_gate.check(tc["name"], tc.get("args", {}))
            try:
                if decision["behavior"] == "deny":
                    output = f"[Permission denied]: {decision['reason']}"
                elif decision["behavior"] == "ask" and not permission_gate.ask_user(
                    decision["intent"], tc.get("args", {})
                ):
                    output = f"[Permission denied by user]: {decision['reason']}"
                else:
                    output = handle_tool_call(tc["name"], tc.get("args", {}))
            except Exception as e:
                output = f"[Error]: {e}"
            print(f"> {tc['name']}: {str(output)[:200]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": tc.get("id", ""),
                "content": normalize_tool_result(tc["name"], str(output), decision.get("intent")),
            })
        messages.append(HumanMessage(content=json.dumps(results)))

    response_content = messages[-1]
    if hasattr(response_content, 'content') and response_content.content:
        print(f"\nAssistant: {response_content.content}")


if __name__ == "__main__":
    # Scan for plugins
    found = plugin_loader.scan()
    if found:
        print(f"[Plugins loaded: {', '.join(found)}]")
        for server_name, config in plugin_loader.get_mcp_servers().items():
            mcp_client = MCPClient(
                server_name, config.get("command", ""), config.get("args", [])
            )
            if mcp_client.connect():
                mcp_client.list_tools()
                mcp_router.register_client(mcp_client)
                print(f"[MCP] Connected to {server_name}")

    tool_count = len(build_tool_pool())
    mcp_count = len(mcp_router.get_all_tools())
    print(f"[Tool pool: {tool_count} tools ({mcp_count} from MCP)]")

    thread_id = "mcp_session_1"
    config = get_session_config(thread_id)

    # Resume from checkpoint if exists
    existing = graph.get_state(config)
    if existing and existing.values.get("messages"):
        print(f"[Resuming session {thread_id} with {len(existing.values['messages'])} messages]\n")

    print("MCP Plugin System (phase20) - LangGraph Native Patterns")
    print("Features: Checkpoint persistence, unified permission gate")
    print("Use /tools to list all available tools")
    print("Type 'exit' or 'q' to quit\n")

    while True:
        try:
            query = input(f"\033[36m{thread_id} >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            # Cleanup MCP connections
            for c in mcp_router.clients.values():
                c.disconnect()
            break
        if query.strip().lower() in ("q", "exit", ""):
            for c in mcp_router.clients.values():
                c.disconnect()
            break
        if query.strip() == "/tools":
            for tool_spec in build_tool_pool():
                prefix = "[MCP] " if tool_spec["name"].startswith("mcp__") else "       "
                print(f"  {prefix}{tool_spec['name']}: {tool_spec.get('description', '')[:60]}")
            continue
        if query.strip() == "/mcp":
            if mcp_router.clients:
                for name, client in mcp_router.clients.items():
                    tools = client.list_tools()
                    print(f"  {name}: {len(tools)} tools")
            else:
                print("  (no MCP servers connected)")
            continue
        run_agent_with_permission(query, thread_id)