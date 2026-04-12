#!/usr/bin/env python3
"""
phase11_system_prompt.py - System Prompt Construction
The system prompt is assembled from independent sections, not one giant
hardcoded blob.

Pipeline:
  1. core instructions
  2. tool listing
  3. skill metadata (from skills/*.md, progressive disclosure)
  4. memory section (from .mini-agent-cli/.memory/*.md)
  5. MINI_AGENT.md chain (user -> project -> subdir)
  6. dynamic context (date, workdir, model)

Key insight: "Prompt construction is a pipeline with boundaries, not one
big string."
"""
import datetime
import os
import re
import subprocess
from pathlib import Path
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

load_dotenv(override=True)
os.environ["NO_PROXY"] = "*"

MODEL_ID = os.environ["AGENCY_LLM_MODEL"]
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER", "openai")

WORKDIR = Path.cwd()
STORAGE_DIR = WORKDIR / ".mini-agent-cli"
SKILLS_DIR = STORAGE_DIR / "skills"
MEMORY_DIR = STORAGE_DIR / ".memory"


model = init_chat_model(
    MODEL_ID,
    model_provider=PROVIDER,
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
)


# ---------- System prompt builder ----------
class SystemPromptBuilder:
    """Assemble the system prompt from independent sections."""

    def __init__(self, workdir: Path = None):
        self.workdir = workdir or WORKDIR

    def _build_core(self) -> str:
        return (
            f"You are a coding agent operating in {self.workdir}.\n"
            "Use the provided tools to explore, read, write, and edit files.\n"
            "Always verify before assuming. Prefer reading files over guessing."
        )

    def _build_tool_listing(self) -> str:
        lines = ["# Available tools"]
        for t in ALL_TOOLS:
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def _build_skill_listing(self) -> str:
        if not SKILLS_DIR.exists():
            return ""
        skills = []
        for skill_dir in sorted(SKILLS_DIR.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            text = skill_md.read_text()
            match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
            if not match:
                continue
            meta = {}
            for line in match.group(1).splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            name = meta.get("name", skill_dir.name)
            desc = meta.get("description", "")
            skills.append(f"- {name}: {desc}")
        if not skills:
            return ""
        return "# Available skills\n" + "\n".join(skills)

    def _build_memory_section(self) -> str:
        if not MEMORY_DIR.exists():
            return ""
        memories = []
        for md_file in sorted(MEMORY_DIR.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            text = md_file.read_text()
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
            if not match:
                continue
            header, body = match.group(1), match.group(2).strip()
            meta = {}
            for line in header.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            name = meta.get("name", md_file.stem)
            mem_type = meta.get("type", "project")
            desc = meta.get("description", "")
            memories.append(f"[{mem_type}] {name}: {desc}\n{body}")
        if not memories:
            return ""
        return "# Memories (persistent)\n\n" + "\n\n".join(memories)

    def _build_agent_md(self) -> str:
        sources = []
        user_agent = Path.home() / ".mini-agent-cli" / "MINI_AGENT.md"
        if user_agent.exists():
            sources.append(("user global", user_agent.read_text()))
        project_agent = self.workdir / "MINI_AGENT.md"
        if project_agent.exists():
            sources.append(("project root", project_agent.read_text()))
        cwd = Path.cwd()
        if cwd != self.workdir:
            subdir_agent = cwd / "MINI_AGENT.md"
            if subdir_agent.exists():
                sources.append(("subdir", subdir_agent.read_text()))
        if not sources:
            return ""
        parts = ["# MINI_AGENT.md instructions"]
        for label, content in sources:
            parts.append(f"## From {label}")
            parts.append(content.strip())
        return "\n\n".join(parts)

    def _build_dynamic_context(self) -> str:
        lines = [
            f"Current date: {datetime.date.today().isoformat()}",
            f"Working directory: {self.workdir}",
            f"Model: {MODEL_ID}",
        ]
        return "# Dynamic context\n" + "\n".join(lines)

    def build(self) -> str:
        sections = []
        for builder in [
            self._build_core,
            self._build_tool_listing,
            self._build_skill_listing,
            self._build_memory_section,
            self._build_agent_md,
            self._build_dynamic_context,
        ]:
            text = builder()
            if text:
                sections.append(text)
        return "\n\n".join(sections)


prompt_builder = SystemPromptBuilder()
SYSTEM = prompt_builder.build()


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
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
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
    except subprocess.TimeoutExpired:
        return "[Error]: Timeout (120s)"
    return (r.stdout + r.stderr).strip() or "(no output)"


@tool
def read_file(path: str, limit: Optional[int] = None) -> str:
    """Read file contents."""
    try:
        p = _safe(path)
        lines = p.read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines)-limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"[Error]: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        f = _safe(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"[Error]: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in a file once."""
    try:
        f = _safe(path)
        content = f.read_text()
        if old_text not in content:
            return f"[Error]: Text not found in {path}"
        f.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"[Error]: {e}"


ALL_TOOLS = [bash, read_file, write_file, edit_file]
tool_node = ToolNode(ALL_TOOLS, handle_tool_errors=True)
model_with_tools = model.bind_tools(ALL_TOOLS)


# ---------- state schema ----------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------- LangGraph nodes ----------
def call_model(state: AgentState) -> dict:
    """Stream model responses."""
    messages_with_system = [HumanMessage(content=SYSTEM)] + state["messages"]
    full_response = None
    for chunk in model_with_tools.stream(messages_with_system):
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


def should_continue(state: AgentState) -> Literal["tools", END]:
    """Decide whether to continue tool execution or finish."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# ---------- build graph ----------
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Compile with checkpoint for session persistence (LangGraph native)
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


def get_session_config(thread_id: str) -> dict:
    """LangGraph native: Get session config for checkpointing."""
    return {"configurable": {"thread_id": thread_id}}


# ---------- CLI ----------
if __name__ == "__main__":
    section_count = SYSTEM.count("\n# ")
    print(f"[System prompt assembled: {len(SYSTEM)} chars, ~{section_count} sections]")

    thread_id = "system_prompt_session_1"
    config = get_session_config(thread_id)

    # Resume from checkpoint if exists
    existing = graph.get_state(config)
    existing_msgs = existing.values.get("messages", []) if existing and existing.values else []
    if existing_msgs:
        print(f"[Resuming session {thread_id} with {len(existing_msgs)} messages]\n")

    print("System Prompt Agent (phase11) - LangGraph Native Patterns")
    print("Features: Checkpoint persistence, dynamic system prompts")
    print("Type 'exit' or 'q' to quit\n")

    while True:
        try:
            q = input(f"\033[36m{thread_id} >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in ("q", "exit", ""):
            break
        if q.strip() == "/prompt":
            print("--- System Prompt ---")
            print(SYSTEM)
            print("--- End ---")
            continue
        if q.strip() == "/sections":
            for line in SYSTEM.splitlines():
                if line.startswith("# "):
                    print(f"  {line}")
            continue

        # Get existing messages from checkpoint
        existing = graph.get_state(config)
        existing_msgs = existing.values.get("messages", []) if existing and existing.values else []
        existing_msgs.append(HumanMessage(content=q))

        state = graph.invoke({"messages": existing_msgs}, config)
        print()
