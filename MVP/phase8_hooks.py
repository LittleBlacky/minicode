#!/usr/bin/env python3
# Harness: extensibility -- injecting behavior without touching the loop.
"""
phase8_hooks.py - Hook System
Hooks are extension points around the main loop.
They let you add behavior without rewriting the graph nodes themselves.
Events: SessionStart, PreToolUse, PostToolUse
Exit-code contract: 0 → continue, 1 → block, 2 → inject message
Key insight: "Extend the agent without touching the loop."
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Annotated

from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

# ---------- env ----------
load_dotenv(override=True)
os.environ["NO_PROXY"] = "*"
MODEL_ID = os.environ["AGENCY_LLM_MODEL"]
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER", "openai")

WORKDIR = Path.cwd()
HOOK_EVENTS = ("PreToolUse", "PostToolUse", "SessionStart")
HOOK_TIMEOUT = 30
STORAGE_DIR = WORKDIR / ".mini-agent-cli"
TRUST_MARKER = STORAGE_DIR / ".trusted"
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."


# ---------- Hook system ----------
class HookManager:
    """Load and execute hooks from .hooks.json configuration."""

    def __init__(self, config_path: Path = None, sdk_mode: bool = False):
        self.hooks = {"PreToolUse": [], "PostToolUse": [], "SessionStart": []}
        self._sdk_mode = sdk_mode
        config_path = config_path or (STORAGE_DIR / ".hooks.json")
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                for event in HOOK_EVENTS:
                    self.hooks[event] = config.get("hooks", {}).get(event, [])
                print(f"[Hooks loaded from {config_path}]")
            except Exception as e:
                print(f"[Hook config error: {e}]")

    def _check_workspace_trust(self) -> bool:
        if self._sdk_mode:
            return True
        return TRUST_MARKER.exists()

    def run_hooks(self, event: str, context: dict = None) -> dict:
        """Execute all hooks for an event.
        Returns: {"blocked": bool, "block_reason": str, "messages": list[str],
                  "updated_input": dict|None}
        """
        result = {
            "blocked": False,
            "block_reason": "",
            "messages": [],
            "updated_input": None,
        }
        if not self._check_workspace_trust():
            return result
        hooks = self.hooks.get(event, [])
        for hook_def in hooks:
            matcher = hook_def.get("matcher")
            if matcher and context and matcher != "*":
                if matcher != context.get("tool_name", ""):
                    continue
            command = hook_def.get("command", "")
            if not command:
                continue
            env = dict(os.environ)
            if context:
                env["HOOK_EVENT"] = event
                env["HOOK_TOOL_NAME"] = context.get("tool_name", "")
                env["HOOK_TOOL_INPUT"] = json.dumps(
                    context.get("tool_input", {}), ensure_ascii=False
                )[:10000]
                if "tool_output" in context:
                    env["HOOK_TOOL_OUTPUT"] = str(context["tool_output"])[:10000]
            try:
                r = subprocess.run(
                    command,
                    shell=True,
                    cwd=WORKDIR,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=HOOK_TIMEOUT,
                )
                if r.returncode == 0:
                    if r.stdout.strip():
                        print(f"  [hook:{event}] {r.stdout.strip()[:100]}")
                    try:
                        out = json.loads(r.stdout)
                        if "updatedInput" in out and context:
                            context["tool_input"] = out["updatedInput"]
                            result["updated_input"] = out["updatedInput"]
                        if "additionalContext" in out:
                            result["messages"].append(out["additionalContext"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif r.returncode == 1:
                    result["blocked"] = True
                    reason = r.stderr.strip() or "Blocked by hook"
                    result["block_reason"] = reason
                    print(f"  [hook:{event}] BLOCKED: {reason[:200]}")
                elif r.returncode == 2:
                    msg = r.stderr.strip()
                    if msg:
                        result["messages"].append(msg)
                        print(f"  [hook:{event}] INJECT: {msg[:200]}")
            except subprocess.TimeoutExpired:
                print(f"  [hook:{event}] Timeout ({HOOK_TIMEOUT}s)")
            except Exception as e:
                print(f"  [hook:{event}] Error: {e}")
        return result


HOOKS = HookManager()


# ---------- helpers ----------
def _safe(p: str) -> Path:
    p = (WORKDIR / p).resolve()
    if not p.is_relative_to(WORKDIR):
        raise ValueError(p)
    return p


# ---------- tools ----------
@tool
def bash(command: str) -> str:
    """Run a shell command."""
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout"
    return (r.stdout + r.stderr).strip() or "(no output)"


@tool
def read_file(path: str, limit: int = None) -> str:
    """Read file contents."""
    try:
        p = _safe(path)
        lines = p.read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines)-limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        f = _safe(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in a file once."""
    try:
        f = _safe(path)
        content = f.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        f.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


ALL_TOOLS = [bash, read_file, write_file, edit_file]
TOOL_BY_NAME = {t.name: t for t in ALL_TOOLS}


# ---------- state schema ----------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------- LangGraph nodes ----------
def _model():
    return init_chat_model(
        model=MODEL_ID, model_provider=PROVIDER, base_url=BASE_URL, api_key=API_KEY
    )


model = _model().bind_tools(ALL_TOOLS)


def agent(state: AgentState) -> dict:
    msgs = [SystemMessage(content=SYSTEM)] + state["messages"]
    full_response = None
    for chunk in model.stream(msgs):
        if isinstance(chunk, AIMessageChunk):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if full_response is None:
                full_response = chunk
            else:
                full_response = full_response + chunk
    print()
    if full_response is None:
        return {"messages": [AIMessage(content="")]}
    return {"messages": [full_response]}


def tools_wrapper(state: AgentState) -> dict:
    """Execute tools with PreToolUse / PostToolUse hooks."""
    last_ai = state["messages"][-1]

    final_msgs = []
    for tc in last_ai.tool_calls:
        name, tid, args = tc["name"], tc["id"], tc.get("args", {}) or {}
        ctx = {"tool_name": name, "tool_input": dict(args)}

        # -- PreToolUse hooks --
        pre = HOOKS.run_hooks("PreToolUse", ctx)

        # Inject hook messages as separate ToolMessages
        for msg in pre.get("messages", []):
            final_msgs.append(
                ToolMessage(
                    content=f"[Hook message]: {msg}",
                    tool_call_id=tid,
                )
            )

        if pre.get("blocked"):
            reason = pre.get("block_reason", "Blocked by hook")
            print(f"  [BLOCKED] {name}: {reason}")
            final_msgs.append(
                ToolMessage(
                    content=f"Tool blocked by PreToolUse hook: {reason}",
                    tool_call_id=tid,
                )
            )
            continue

        # Use updated input from hook
        tool_input = pre.get("updated_input") or args

        # -- Execute tool --
        tool_fn = TOOL_BY_NAME.get(name)
        if tool_fn:
            try:
                content = tool_fn.invoke(tool_input)
            except Exception as e:
                content = f"Error: {e}"
        else:
            content = f"Unknown tool: {name}"

        print(f"> {name}: {str(content)[:200]}")

        # -- PostToolUse hooks --
        ctx["tool_output"] = content
        post = HOOKS.run_hooks("PostToolUse", ctx)
        for msg in post.get("messages", []):
            content += f"\n[Hook note]: {msg}"

        final_msgs.append(ToolMessage(content=content, tool_call_id=tid))

    return {"messages": final_msgs}


def route_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# ---------- build graph ----------
graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", tools_wrapper)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route_agent, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

# Compile with checkpoint for session persistence (LangGraph native)
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


def get_session_config(thread_id: str) -> dict:
    """LangGraph native: Get session config for checkpointing."""
    return {"configurable": {"thread_id": thread_id}}


if __name__ == "__main__":
    # Fire SessionStart hooks
    HOOKS.run_hooks("SessionStart", {"tool_name": "", "tool_input": {}})

    thread_id = "hooks_session_1"
    config = get_session_config(thread_id)

    # Resume from checkpoint if exists
    existing = app.get_state(config)
    existing_msgs = existing.values.get("messages", []) if existing and existing.values else []
    if existing_msgs:
        print(f"[Resuming session {thread_id} with {len(existing_msgs)} messages]\n")

    print("Hooks Agent (phase8) - LangGraph Native Patterns")
    print("Features: Checkpoint persistence, extensible hooks")
    print("Type 'exit' or 'q' to quit\n")

    while (q := input(f"\033[36m{thread_id} >> \033[0m")) not in ("q", "exit", ""):
        # Get existing messages from checkpoint
        existing = app.get_state(config)
        existing_msgs = existing.values.get("messages", []) if existing and existing.values else []
        existing_msgs.append(HumanMessage(content=q))

        state = app.invoke({"messages": existing_msgs}, config)
        print()
