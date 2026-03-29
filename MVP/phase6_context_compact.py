# !/usr/bin/env python3
import json, os, subprocess, time
from pathlib import Path
from typing import Annotated, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    SystemMessage,
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
LLM_DIR = WORKDIR / ".mini-agent-cli"
SYSTEM = f"You are a coding agent at {WORKDIR}. Keep working step by step, and use compact if the conversation gets too long."
CONTEXT_LIMIT, KEEP_RECENT, PERSIST_THRESHOLD, PREVIEW_CHARS = 50000, 3, 30000, 2000
TRANSCRIPT_DIR = LLM_DIR / "transcripts"
TOOL_RESULTS_DIR = LLM_DIR / "task_outputs" / "tool-results"


# ---------- helpers ----------
def _size(messages):
    return len(str(messages))


def _safe(p: str) -> Path:
    p = (WORKDIR / p).resolve()
    if not p.is_relative_to(WORKDIR):
        raise ValueError(p)
    return p


def _persist(tool_id: str, out: str) -> str:
    if len(out) <= PERSIST_THRESHOLD:
        return out
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    f = TOOL_RESULTS_DIR / f"{tool_id}.txt"
    if not f.exists():
        f.write_text(out)
    return f"<persisted-output>\nFull output: {f.relative_to(WORKDIR)}\nPreview:\n{out[:PREVIEW_CHARS]}\n</persisted-output>"


# ---------- tools defined with @tool ----------
@tool
def bash(command: str) -> str:
    """Run a shell command."""
    if any(x in command for x in ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]):
        return "Error: Dangerous"
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
def compact(focus: str = None) -> str:
    """Summarize earlier conversation so work can continue in a smaller context."""
    return "Compacting conversation..."


# Build the ToolNode from these tools
ALL_TOOLS = [bash, read_file, write_file, edit_file, compact]
tool_node = ToolNode(ALL_TOOLS)


# ---------- state schema ----------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    has_compacted: bool
    last_summary: str
    recent_files: list[str]
    compact_requested: bool
    compact_focus: Optional[str]


# ---------- compact logic ----------
def _model():
    return init_chat_model(
        model=MODEL_ID, model_provider=PROVIDER, base_url=BASE_URL, api_key=API_KEY
    )


def _compact_summary(messages):
    conv = json.dumps([m.model_dump() for m in messages])[:80000]
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve: 1. Goal 2. Findings/decisions 3. Files read/changed 4. Remaining work 5. User constraints.\n\n"
        f"{conv}"
    )
    accumulated = None
    for chunk in _model().stream([HumanMessage(content=prompt)]):
        if accumulated is None:
            accumulated = chunk
        else:
            accumulated += chunk
    return (accumulated.content if accumulated else "").strip()


def _compact_history(messages: list, state: AgentState, focus: str = None) -> list:
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    with path.open("w") as h:
        for m in messages:
            h.write(json.dumps(m.model_dump(), default=str) + "\n")
    print(f"[transcript saved: {path}]")
    summary = _compact_summary(messages)
    if focus:
        summary += f"\nFocus: {focus}"
    if state.get("recent_files"):
        summary += "\nRecent files:\n" + "\n".join(
            f"- {p}" for p in state["recent_files"]
        )
    state["has_compacted"] = True
    state["last_summary"] = summary
    return [HumanMessage(content="Compacted conversation:\n" + summary)]


def _brief(content: str) -> str:
    """Extract a short hint from tool output so the agent doesn't need to re-call."""
    first_line = content.strip().split("\n", 1)[0][:80]
    return f"[compacted] {first_line}"


def _micro_compact(messages):
    tools = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if len(tools) <= KEEP_RECENT:
        return messages
    for i in tools[:-KEEP_RECENT]:
        msg = messages[i]
        if isinstance(msg.content, str) and len(msg.content) > 120:
            messages[i] = ToolMessage(
                content=_brief(msg.content), tool_call_id=msg.tool_call_id
            )
    return messages


# ---------- LangGraph nodes ----------
def pre_model(state: AgentState) -> dict:
    msgs = state["messages"]
    _micro_compact(msgs)  # modifies in-place
    if _size(msgs) > CONTEXT_LIMIT:
        print("[auto compact]")
        new_msgs = _compact_history(msgs, state)
        msgs[:] = new_msgs  # replace in-place
    return {}


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
    print()  # newline after streaming
    if full_response is None:
        return {"messages": [AIMessage(content="")]}
    return {"messages": [full_response]}


def tools_wrapper(state: AgentState) -> dict:
    """
    Execute tools via ToolNode, then:
    - persist large outputs
    - track recently read files
    - detect manual compact request
    """
    last_ai = state["messages"][-1]
    # Use ToolNode to produce ToolMessages
    result = tool_node.invoke(state)
    new_tool_msgs = result["messages"]

    # Post-process each tool message
    recent_files = list(state.get("recent_files", []))
    compact_requested = False
    compact_focus = None

    final_msgs = []
    for i, tc in enumerate(last_ai.tool_calls):
        msg = new_tool_msgs[i]
        name = tc["name"]
        tid = tc["id"]
        content = msg.content

        # persist large outputs for bash/read/write/edit
        if name in {"bash", "read_file"}:
            content = _persist(tid, content)
            msg = ToolMessage(content=content, tool_call_id=tid)

        # track read files
        if name == "read_file" and "Error:" not in content:
            path = tc["args"].get("path", "")
            if path:
                if path in recent_files:
                    recent_files.remove(path)
                recent_files.append(path)
                if len(recent_files) > 5:
                    recent_files = recent_files[-5:]

        # compact signal
        if name == "compact":
            compact_requested = True
            compact_focus = tc["args"].get("focus")

        final_msgs.append(msg)
        print(f"> {name}: {str(content)[:200]}")

    return {
        "messages": final_msgs,
        "recent_files": recent_files,
        "compact_requested": compact_requested,
        "compact_focus": compact_focus,
    }


def compact_node(state: AgentState) -> dict:
    print("[manual compact]")
    focus = state.get("compact_focus")
    new_msgs = _compact_history(state["messages"], state, focus)
    state["messages"][:] = new_msgs  # replace in-place
    return {"compact_requested": False}


def route_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def route_tools(state: AgentState) -> str:
    return "compact" if state.get("compact_requested") else "pre_model"


# ---------- build graph ----------
graph = StateGraph(AgentState)
graph.add_node("pre_model", pre_model)
graph.add_node("agent", agent)
graph.add_node("tools", tools_wrapper)
graph.add_node("compact", compact_node)

graph.set_entry_point("pre_model")
graph.add_edge("pre_model", "agent")
graph.add_conditional_edges("agent", route_agent, {"tools": "tools", END: END})
graph.add_conditional_edges(
    "tools", route_tools, {"compact": "compact", "pre_model": "pre_model"}
)
graph.add_edge("compact", "agent")
app = graph.compile()

if __name__ == "__main__":
    state: AgentState = {
        "messages": [],
        "has_compacted": False,
        "last_summary": "",
        "recent_files": [],
        "compact_requested": False,
        "compact_focus": None,
    }
    while (q := input("\033[36ms06 >> \033[0m")) not in ("q", "exit", ""):
        state["messages"].append(HumanMessage(content=q))
        state = app.invoke(state)
        print()
