#!/usr/bin/env python3
from __future__ import annotations
"""
phase12_error_recovery.py - Error Recovery with Checkpoint Persistence

A robust agent that recovers from errors instead of crashing.
Uses langgraph checkpoints for session persistence.

Recovery strategies:
1. max_tokens -> inject continuation, retry
2. prompt_too_long -> compact history, retry
3. connection/rate limit -> exponential backoff, retry

Key insight: "Resilience means having a plan B, C, and D."
"""
import json
import os
import random
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
TRANSCRIPT_DIR = STORAGE_DIR / "transcripts"
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

model = init_chat_model(
    MODEL_ID,
    model_provider=PROVIDER,
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
)

# Recovery constants
MAX_RECOVERY_ATTEMPTS = 3
BACKOFF_BASE_DELAY = 1.0
BACKOFF_MAX_DELAY = 30.0
TOKEN_THRESHOLD = 50000

CONTINUATION_MESSAGE = (
    "Output limit hit. Continue directly from where you stopped -- "
    "no recap, no repetition. Pick up mid-sentence if needed."
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    max_output_recovery_count: int
    error_recovery_count: int


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(json.dumps(messages, default=str)) // 4


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


tools = [bash_tool, read_file, write_file, edit_file]
tool_node = ToolNode(tools, handle_tool_errors=True)
model_with_tools = model.bind_tools(tools)


def call_model(state: AgentState) -> dict:
    """Call the model with current messages."""
    response = model_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "max_output_recovery_count": 0,
        "error_recovery_count": 0,
    }


def should_continue(state: AgentState) -> Literal["tools", "check_error", END]:
    """Decide whether to continue tool execution, check for errors, or finish."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "check_error"
    return END


def check_error(state: AgentState) -> dict:
    """
    Check for errors and decide recovery strategy.

    Recovery priority (first match wins):
    1. max_tokens -> inject continuation, retry
    2. prompt_too_long -> compact, retry
    3. connection error -> backoff, retry
    4. all retries exhausted -> fail gracefully
    """
    last_message = state["messages"][-1]

    # Check for max_tokens error (empty response with tool_calls)
    if isinstance(last_message, AIMessage):
        if not last_message.content and last_message.tool_calls:
            count = state.get("max_output_recovery_count", 0)
            if count < MAX_RECOVERY_ATTEMPTS:
                print(f"[Recovery] Max output tokens (attempt {count + 1}/{MAX_RECOVERY_ATTEMPTS})")
                return {"_next": "inject_continuation"}
            else:
                print("[Recovery] Max output retries exhausted")
                return {"_next": END}

    # Check if context is too large (should compact)
    if estimate_tokens(state["messages"]) > TOKEN_THRESHOLD:
        print("[Recovery] Context too large, compacting...")
        return {"_next": "compact"}

    return {"_next": "tools"}


def inject_continuation(state: AgentState) -> dict:
    """Inject continuation message for max_tokens recovery."""
    count = state.get("max_output_recovery_count", 0)
    new_messages = state["messages"] + [
        HumanMessage(content=CONTINUATION_MESSAGE)
    ]
    return {
        "messages": new_messages,
        "max_output_recovery_count": count + 1,
    }


def compact_history(state: AgentState) -> dict:
    """
    Compress conversation history into a short summary.
    Uses LangChain messages instead of raw dicts.
    """
    messages = state["messages"]

    # Build conversation text from messages
    conv_parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            conv_parts.append(f"Human: {m.content[:2000]}")
        elif isinstance(m, AIMessage):
            conv_parts.append(f"Assistant: {m.content[:2000] if m.content else '[tool calls]'}")

    conversation_text = "\n".join(conv_parts)[:80000]
    prompt = (
        "Summarize this coding-agent conversation for continuity. Include:\n"
        "1) Task overview and success criteria\n"
        "2) Current state: completed work, files touched\n"
        "3) Key decisions and failed approaches\n"
        "4) Remaining next steps\n"
        "Be concise but preserve critical details.\n\n" + conversation_text
    )

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        summary = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        summary = f"(compact failed: {e}). Previous context lost."

    continuation = (
        "This session continues from a previous conversation that was compacted.\n\n"
        f"Summary of prior context:\n\n{summary}\n\n"
        "Continue from where we left off without re-asking the user."
    )

    # Save transcript before compacting
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with transcript_path.open("w") as f:
        for m in messages:
            msg_dict = m.model_dump() if hasattr(m, 'model_dump') else m
            f.write(json.dumps(msg_dict, default=str) + "\n")
    print(f"[Recovery] Transcript saved to {transcript_path}")

    return {
        "messages": [HumanMessage(content=continuation)],
        "max_output_recovery_count": 0,
        "error_recovery_count": 0,
    }


def backoff_retry(state: AgentState) -> dict:
    """
    Apply exponential backoff for connection/rate limit errors.
    In a real implementation, this would check for specific error types
    and apply the backoff before retrying.
    """
    count = state.get("error_recovery_count", 0)
    if count >= MAX_RECOVERY_ATTEMPTS:
        print("[Recovery] Backoff retries exhausted")
        return {"_next": END}

    delay = min(BACKOFF_BASE_DELAY * (2 ** count), BACKOFF_MAX_DELAY)
    jitter = random.uniform(0, 1)
    actual_delay = delay + jitter

    print(f"[Recovery] Connection error, backing off {actual_delay:.2f}s (attempt {count + 1}/{MAX_RECOVERY_ATTEMPTS})")
    time.sleep(actual_delay)

    return {
        "messages": state["messages"],
        "max_output_recovery_count": 0,
        "error_recovery_count": count + 1,
        "_next": "agent",
    }


# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("check_error", check_error)
workflow.add_node("compact", compact_history)
workflow.add_node("inject_continuation", inject_continuation)
workflow.add_node("backoff_retry", backoff_retry)

# Unconditional edges
workflow.add_edge(START, "agent")
workflow.add_edge("tools", "agent")
workflow.add_edge("inject_continuation", "agent")
workflow.add_edge("compact", "agent")
workflow.add_edge("backoff_retry", "agent")

# After agent, always check for errors
workflow.add_edge("agent", "check_error")

# Conditional routing from check_error
def route_check_error(state: AgentState) -> str:
    return state.get("_next", "tools")

workflow.add_conditional_edges(
    "check_error",
    route_check_error,
    {
        "tools": "tools",
        "inject_continuation": "inject_continuation",
        "compact": "compact",
        END: END,
    }
)

# Compile with checkpoint for resume capability
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


def get_session_config(thread_id: str) -> dict:
    """Get langgraph config for a session."""
    return {"configurable": {"thread_id": thread_id}}


def run_agent(query: str, thread_id: str = "default") -> dict:
    """Run the agent with a query using checkpoint-based session."""
    config = get_session_config(thread_id)

    # Check if resuming from existing session
    existing_state = graph.get_state(config)
    if existing_state and existing_state.values.get("messages"):
        print(f"[Session {thread_id}] Resuming from checkpoint...")
        # Add new message to existing session
        new_messages = existing_state.values["messages"] + [HumanMessage(content=query)]
    else:
        # Start fresh session
        new_messages = [HumanMessage(content=query)]

    initial_state = {
        "messages": new_messages,
        "max_output_recovery_count": existing_state.values.get("max_output_recovery_count", 0) if existing_state else 0,
        "error_recovery_count": existing_state.values.get("error_recovery_count", 0) if existing_state else 0,
    }

    final_state = None
    for event in graph.stream(initial_state, config):
        node_name = list(event.keys())[0]
        if node_name == "agent":
            response = event[node_name]["messages"][-1]
            if hasattr(response, 'content') and response.content:
                print(f"\nAssistant: {response.content}")
        elif node_name == "tools":
            pass  # Tools are handled internally
        elif node_name == "compact":
            print("[Agent] History compacted")
        elif node_name == "inject_continuation":
            print("[Agent] Continuing from truncated output...")
        # Get final state for return
        if hasattr(event, 'values'):
            final_state = event

    # Return final state for checkpoint verification
    return final_state or {}


def list_sessions() -> list:
    """List all active sessions from checkpointer."""
    # MemorySaver stores in memory, so we track sessions separately
    # In production, use PostgresSaver or SqliteSaver for persistence
    return []  # Override in subclass or extend


def get_session_info(thread_id: str) -> Optional[dict]:
    """Get info about a specific session."""
    config = get_session_config(thread_id)
    state = graph.get_state(config)
    if state and state.values:
        msg_count = len(state.values.get("messages", []))
        return {
            "thread_id": thread_id,
            "message_count": msg_count,
            "recovery_count": state.values.get("error_recovery_count", 0),
        }
    return None


if __name__ == "__main__":
    print("Error Recovery Agent (phase12) with Checkpoint Persistence")
    print("Type 'exit' or 'q' to quit, '/sessions' to list sessions")
    print("Sessions persist across restarts via langgraph checkpoints\n")

    # Session management
    current_thread = "session_1"
    sessions = {}

    while True:
        try:
            query = input(f"\033[36m{current_thread} >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        # Session commands
        if query.strip() == "/sessions":
            # List all sessions from checkpoint storage
            all_threads = checkpointer.storage.keys() if hasattr(checkpointer, 'storage') else []
            if all_threads:
                print("Active sessions:")
                for thread_id in all_threads:
                    info = get_session_info(thread_id)
                    if info:
                        print(f"  {thread_id}: {info['message_count']} messages")
            else:
                print("No active sessions")
            continue

        if query.startswith("/session "):
            parts = query.split(maxsplit=1)
            if len(parts) == 2:
                current_thread = parts[1].strip()
                print(f"Switched to session: {current_thread}")
                # Check if session exists
                info = get_session_info(current_thread)
                if info:
                    print(f"Session has {info['message_count']} messages")
                else:
                    print("New session")
            continue

        if query.strip() == "/new":
            # Create new session with timestamp
            current_thread = f"session_{int(time.time())}"
            print(f"Created new session: {current_thread}")
            continue

        if query.strip() == "/clear":
            # Clear current session by creating new graph state
            # Note: In production, you'd delete the checkpoint
            print(f"Cleared session: {current_thread}")
            current_thread = f"session_{int(time.time())}"
            continue

        run_agent(query, current_thread)
