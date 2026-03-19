#!/usr/bin/env python3
# Harness: safety -- the pipeline between intent and execution.
"""
phase7_permission.py - Permission System
Every tool call passes through a permission pipeline before execution.
Pipeline: deny_rules → mode_check → allow_rules → ask_user
Modes: default (ask), plan (read-only), auto (allow reads, ask writes)
Key insight: "Safety is a pipeline, not a boolean."
"""
import json
import os
import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated

from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
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
MODES = ("default", "plan", "auto")
READ_ONLY_TOOLS = {"read_file"}
WRITE_TOOLS = {"write_file", "edit_file", "bash"}
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Use tools to solve tasks.\n"
    f"Permission modes: default (user approves each write), "
    f"plan (read-only, no writes), auto (reads allowed, writes ask).\n"
    f"Use set_mode to switch modes."
)


# ---------- Bash security validator ----------
class BashSecurityValidator:
    """Validate bash commands for obviously dangerous patterns."""

    VALIDATORS = [
        ("shell_metachar", r"[;&|`$]"),
        ("sudo", r"\bsudo\b"),
        ("rm_rf", r"\brm\s+(-[a-zA-Z]*)?r"),
        ("cmd_substitution", r"\$\("),
        ("ifs_injection", r"\bIFS\s*="),
    ]

    def validate(self, command: str) -> list:
        failures = []
        for name, pattern in self.VALIDATORS:
            if re.search(pattern, command):
                failures.append((name, pattern))
        return failures

    def describe_failures(self, command: str) -> str:
        failures = self.validate(command)
        if not failures:
            return "No issues detected"
        parts = [f"{name}" for name, _ in failures]
        return "Security flags: " + ", ".join(parts)


bash_validator = BashSecurityValidator()

# ---------- Permission rules ----------
DEFAULT_RULES = [
    {"tool": "bash", "content": "rm -rf /", "behavior": "deny"},
    {"tool": "bash", "content": "sudo *", "behavior": "deny"},
    {"tool": "read_file", "path": "*", "behavior": "allow"},
]


class PermissionManager:
    """Pipeline: deny_rules → mode_check → allow_rules → ask (via interrupt)"""

    def __init__(self, mode: str = "default", rules: list = None):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}. Choose from {MODES}")
        self.mode = mode
        self.rules = rules or list(DEFAULT_RULES)
        self.consecutive_denials = 0
        self.max_consecutive_denials = 3

    def check(self, tool_name: str, tool_input: dict) -> dict:
        """Returns {"behavior": "allow"|"deny"|"ask", "reason": str}"""
        # Step 0: Bash security validation
        if tool_name == "bash":
            command = tool_input.get("command", "")
            failures = bash_validator.validate(command)
            if failures:
                severe = {"sudo", "rm_rf"}
                severe_hits = [f for f in failures if f[0] in severe]
                desc = bash_validator.describe_failures(command)
                if severe_hits:
                    return {"behavior": "deny", "reason": f"Bash: {desc}"}
                return {"behavior": "ask", "reason": f"Bash flagged: {desc}"}
        # Step 1: Deny rules (always checked first)
        for rule in self.rules:
            if rule["behavior"] != "deny":
                continue
            if self._matches(rule, tool_name, tool_input):
                return {"behavior": "deny", "reason": f"Blocked by rule: {rule}"}
        # Step 2: Mode-based decisions
        if self.mode == "plan":
            if tool_name in WRITE_TOOLS:
                return {"behavior": "deny", "reason": "Plan mode: writes blocked"}
            return {"behavior": "allow", "reason": "Plan mode: read allowed"}
        if self.mode == "auto":
            if tool_name in READ_ONLY_TOOLS:
                return {"behavior": "allow", "reason": "Auto mode: read approved"}
        # Step 3: Allow rules
        for rule in self.rules:
            if rule["behavior"] != "allow":
                continue
            if self._matches(rule, tool_name, tool_input):
                self.consecutive_denials = 0
                return {"behavior": "allow", "reason": f"Matched rule: {rule}"}
        # Step 4: Ask user (via LangGraph interrupt)
        return {"behavior": "ask", "reason": f"No rule for {tool_name}, asking user"}

    def add_rule(self, tool_name: str):
        """Add a permanent allow rule for a tool."""
        self.rules.append({"tool": tool_name, "path": "*", "behavior": "allow"})
        self.consecutive_denials = 0

    def record_denial(self):
        self.consecutive_denials += 1
        if self.consecutive_denials >= self.max_consecutive_denials:
            print(f"  [{self.consecutive_denials} denials — consider /mode plan]")

    def _matches(self, rule: dict, tool_name: str, tool_input: dict) -> bool:
        if rule.get("tool") and rule["tool"] != "*" and rule["tool"] != tool_name:
            return False
        if "path" in rule and rule["path"] != "*":
            path = tool_input.get("path", "")
            if not fnmatch(path, rule["path"]):
                return False
        if "content" in rule:
            command = tool_input.get("command", "")
            if not fnmatch(command, rule["content"]):
                return False
        return True


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


@tool
def set_mode(mode: str) -> str:
    """Switch permission mode. Valid modes: default, plan, auto."""
    if mode not in MODES:
        return f"Unknown mode: {mode}. Choose: {', '.join(MODES)}"
    return f"Switched to {mode} mode"


ALL_TOOLS = [bash, read_file, write_file, edit_file, set_mode]
TOOL_BY_NAME = {t.name: t for t in ALL_TOOLS}


# ---------- state schema ----------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str
    permission_rules: list
    consecutive_denials: int


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
    """
    Permission-aware tool execution with LangGraph interrupt().
    Pipeline: deny_rules → mode_check → allow_rules → interrupt (was ask_user)
    """
    last_ai = state["messages"][-1]
    perms = PermissionManager(
        mode=state.get("mode", "default"),
        rules=list(state.get("permission_rules", DEFAULT_RULES)),
    )

    final_msgs = []
    for tc in last_ai.tool_calls:
        name = tc["name"]
        tid = tc["id"]
        args = tc.get("args", {}) or {}

        # -- Permission check --
        decision = perms.check(name, args)

        if decision["behavior"] == "deny":
            content = f"Permission denied: {decision['reason']}"
            print(f"  [DENIED] {name}: {decision['reason']}")
            perms.record_denial()
            final_msgs.append(ToolMessage(content=content, tool_call_id=tid))
            continue

        if decision["behavior"] == "ask":
            # Use LangGraph interrupt() instead of blocking input()
            preview = json.dumps(args, ensure_ascii=False)[:200]
            user_choice = interrupt(
                f"[Permission] {name}: {preview}\n  Allow? (y/n/always): "
            )
            # user_choice is a dict: {"approved": bool, "add_rule": bool}
            if not user_choice.get("approved"):
                content = f"Permission denied by user for {name}"
                print(f"  [USER DENIED] {name}")
                perms.record_denial()
                final_msgs.append(ToolMessage(content=content, tool_call_id=tid))
                continue
            if user_choice.get("add_rule"):
                perms.add_rule(name)
                print(f"  [Rule added: always allow {name}]")
            perms.consecutive_denials = 0

        # -- Execute tool --
        tool_fn = TOOL_BY_NAME.get(name)
        if tool_fn:
            try:
                content = tool_fn.invoke(args)
            except Exception as e:
                content = f"Error: {e}"
        else:
            content = f"Unknown tool: {name}"

        print(f"> {name}: {str(content)[:200]}")

        if name == "set_mode":
            new_mode = args.get("mode", "")
            if new_mode in MODES:
                state["mode"] = new_mode

        final_msgs.append(ToolMessage(content=content, tool_call_id=tid))

    return {
        "messages": final_msgs,
        "mode": perms.mode,
        "permission_rules": perms.rules,
        "consecutive_denials": perms.consecutive_denials,
    }


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
app = graph.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    print(f"Permission modes: {', '.join(MODES)}")
    mode_input = input("Mode (default): ").strip().lower() or "default"
    if mode_input not in MODES:
        mode_input = "default"
    print(f"[Permission mode: {mode_input}]")

    state: AgentState = {
        "messages": [],
        "mode": mode_input,
        "permission_rules": list(DEFAULT_RULES),
        "consecutive_denials": 0,
    }
    config = {"configurable": {"thread_id": "session_1"}}

    while (q := input("\033[36ms07 >> \033[0m")) not in ("q", "exit", ""):
        if q.startswith("/mode"):
            parts = q.split()
            if len(parts) == 2 and parts[1] in MODES:
                state["mode"] = parts[1]
                print(f"[Switched to {parts[1]} mode]")
            else:
                print(f"Usage: /mode <{'|'.join(MODES)}>")
            continue
        if q.strip() == "/rules":
            for i, rule in enumerate(state["permission_rules"]):
                print(f"  {i}: {rule}")
            continue

        state["messages"].append(HumanMessage(content=q))

        # Stream with interrupt handling
        stream_input = state
        while True:
            interrupted = False
            for event in app.stream(stream_input, config):
                if "__interrupt__" in event:
                    interrupted = True
                    for info in event["__interrupt__"]:
                        print(f"\n  {info.value}", end="")
                        try:
                            answer = input().strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            answer = "n"
                        stream_input = Command(
                            resume={
                                "approved": answer in ("y", "yes", "always"),
                                "add_rule": answer == "always",
                            }
                        )
                        break  # handle one interrupt at a time
                    break  # re-enter stream loop with resume
            if not interrupted:
                break  # graph completed

        # Sync state back from checkpointer
        snapshot = app.get_state(config)
        if snapshot and snapshot.values:
            state.update(snapshot.values)

        print()
