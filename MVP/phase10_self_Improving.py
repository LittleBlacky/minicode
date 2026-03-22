#!/usr/bin/env python3
# Harness: persistence -- remembering across the session boundary.
"""
phase9_memory.py - Memory System
Some information should survive the current conversation, but not everything
belongs in memory.

Use memory for:
  - user preferences
  - repeated user feedback
  - project facts that are NOT obvious from the current code
  - pointers to external resources

Do NOT use memory for:
  - code structure that can be re-read from the repo
  - temporary task state
  - secrets

Storage layout:
  .memory/
    MEMORY.md
    prefer_tabs.md
    review_style.md

Key insight: "Memory only stores cross-session information that is still
worth recalling later and is not easy to re-derive from the current repo."
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Annotated

from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
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
LLM_DIR = WORKDIR / ".mini-agent-cli"
MEMORY_DIR = LLM_DIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
MEMORY_TYPES = ("user", "feedback", "project", "reference")
MAX_INDEX_LINES = 200


# ---------- Memory manager ----------
class MemoryManager:
    """Load, build, and save persistent memories across sessions."""

    def __init__(self, memory_dir: Path = None):
        self.memory_dir = memory_dir or MEMORY_DIR
        self.memories: dict = {}

    def load_all(self):
        self.memories = {}
        if not self.memory_dir.exists():
            return
        for md_file in sorted(self.memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            parsed = self._parse_frontmatter(md_file.read_text())
            if parsed:
                name = parsed.get("name", md_file.stem)
                self.memories[name] = {
                    "description": parsed.get("description", ""),
                    "type": parsed.get("type", "project"),
                    "content": parsed.get("content", ""),
                    "file": md_file.name,
                }
        count = len(self.memories)
        if count > 0:
            print(f"[Memory loaded: {count} memories from {self.memory_dir}]")

    def load_memory_prompt(self) -> str:
        """Build a memory section for injection into the system prompt."""
        if not self.memories:
            return ""
        sections = ["# Memories (persistent across sessions)", ""]
        for mem_type in MEMORY_TYPES:
            typed = {k: v for k, v in self.memories.items() if v["type"] == mem_type}
            if not typed:
                continue
            sections.append(f"## [{mem_type}]")
            for name, mem in typed.items():
                sections.append(f"### {name}: {mem['description']}")
                if mem["content"].strip():
                    sections.append(mem["content"].strip())
                sections.append("")
        return "\n".join(sections)

    def save_memory(
        self, name: str, description: str, mem_type: str, content: str
    ) -> str:
        if mem_type not in MEMORY_TYPES:
            return f"Error: type must be one of {MEMORY_TYPES}"
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())
        if not safe_name:
            return "Error: invalid memory name"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        frontmatter = f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n{content}\n"
        file_path = self.memory_dir / f"{safe_name}.md"
        file_path.write_text(frontmatter)
        self.memories[name] = {
            "description": description,
            "type": mem_type,
            "content": content,
            "file": file_path.name,
        }
        self._rebuild_index()
        return f"Saved memory '{name}' [{mem_type}]"

    def _rebuild_index(self):
        lines = ["# Memory Index", ""]
        for name, mem in self.memories.items():
            lines.append(f"- {name}: {mem['description']} [{mem['type']}]")
            if len(lines) >= MAX_INDEX_LINES:
                lines.append(f"... (truncated at {MAX_INDEX_LINES} lines)")
                break
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        MEMORY_INDEX.write_text("\n".join(lines) + "\n")

    def _parse_frontmatter(self, text: str) -> dict | None:
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if not match:
            return None
        header, body = match.group(1), match.group(2)
        result = {"content": body.strip()}
        for line in header.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip()
        return result


memory_mgr = MemoryManager()
memory_mgr.load_all()

MEMORY_GUIDANCE = """
When to save memories:
- User states a preference ("I like tabs", "always use pytest") -> type: user
- User corrects you ("don't do X", "that was wrong because...") -> type: feedback
- You learn a project fact that is not easy to infer from current code alone
  (for example: a rule exists because of compliance, or a legacy module must
  stay untouched for business reasons) -> type: project
- You learn where an external resource lives (ticket board, dashboard, docs URL)
  -> type: reference
When NOT to save:
- Anything easily derivable from code (function signatures, file structure, directory layout)
- Temporary task state (current branch, open PR numbers, current TODOs)
- Secrets or credentials (API keys, passwords)
"""

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Use tools to solve tasks.\n"
    f"Use save_memory to persist important information across sessions.\n"
    f"{MEMORY_GUIDANCE}\n"
    f"{memory_mgr.load_memory_prompt()}"
)


# ---------- Dream consolidator ----------
class DreamConsolidator:
    """Auto-consolidation of memories between sessions ("Dream")."""

    COOLDOWN_SECONDS = 86400
    SCAN_THROTTLE_SECONDS = 600
    MIN_SESSION_COUNT = 5
    LOCK_STALE_SECONDS = 3600
    PHASES = [
        "Orient: scan MEMORY.md index for structure and categories",
        "Gather: read individual memory files for full content",
        "Consolidate: merge related memories, remove stale entries",
        "Prune: enforce 200-line limit on MEMORY.md index",
    ]

    def __init__(self, memory_dir: Path = None):
        self.memory_dir = memory_dir or MEMORY_DIR
        self.lock_file = self.memory_dir / ".dream_lock"
        self.enabled = True
        self.mode = "default"
        self.last_consolidation_time = 0.0
        self.last_scan_time = 0.0
        self.session_count = 1

    def should_consolidate(self) -> tuple[bool, str]:
        now = time.time()
        if not self.enabled:
            return False, "consolidation is disabled"
        if not self.memory_dir.exists():
            return False, "memory directory does not exist"
        memory_files = [
            f for f in self.memory_dir.glob("*.md") if f.name != "MEMORY.md"
        ]
        if not memory_files:
            return False, "no memory files found"
        if self.mode == "plan":
            return False, "plan mode does not allow consolidation"
        time_since_last = now - self.last_consolidation_time
        if time_since_last < self.COOLDOWN_SECONDS:
            return (
                False,
                f"cooldown active, {int(self.COOLDOWN_SECONDS - time_since_last)}s remaining",
            )
        time_since_scan = now - self.last_scan_time
        if time_since_scan < self.SCAN_THROTTLE_SECONDS:
            return (
                False,
                f"scan throttle active, {int(self.SCAN_THROTTLE_SECONDS - time_since_scan)}s remaining",
            )
        if self.session_count < self.MIN_SESSION_COUNT:
            return (
                False,
                f"only {self.session_count} sessions, need {self.MIN_SESSION_COUNT}",
            )
        if not self._acquire_lock():
            return False, "lock held by another process"
        return True, "all gates passed"

    def consolidate(self) -> dict:
        """Run the 4-phase consolidation. Returns summary dict."""
        can_run, reason = self.should_consolidate()
        if not can_run:
            print(f"[Dream] Cannot consolidate: {reason}")
            return {"ran": False, "reason": reason}

        print("[Dream] Starting consolidation...")
        self.last_scan_time = time.time()
        summary = {"ran": True, "merged": 0, "deleted": 0, "kept": 0}

        # Phase 1-2: Gather all memories
        memory_files = [
            f for f in self.memory_dir.glob("*.md") if f.name != "MEMORY.md"
        ]
        all_memories = {}
        for mf in memory_files:
            parsed = memory_mgr._parse_frontmatter(mf.read_text())
            if parsed:
                all_memories[mf.name] = parsed
        print(f"[Dream] Phase 1-2: gathered {len(all_memories)} memories")

        if not all_memories:
            self._release_lock()
            return summary

        # Phase 3: LLM consolidation
        print("[Dream] Phase 3: analyzing for duplicates/contradictions...")
        memories_text = json.dumps(
            {
                k: {
                    "description": v.get("description", ""),
                    "type": v.get("type", ""),
                    "content": v.get("content", "")[:500],
                }
                for k, v in all_memories.items()
            },
            ensure_ascii=False,
            indent=2,
        )

        prompt = (
            "You are maintaining a memory store. Analyze these memories and "
            "identify:\n"
            "1. Duplicates to merge (same concept, different names)\n"
            "2. Contradictions to resolve (conflicting advice)\n"
            "3. Obsolete entries to delete (no longer relevant)\n\n"
            f"Memories:\n{memories_text}\n\n"
            'Return JSON: {"actions": [{"action": "merge|delete|keep", '
            '"files": ["file1.md", ...], "new_name": "...", '
            '"new_description": "...", "new_type": "user|feedback|project|reference", '
            '"new_content": "..."}]}\n'
            "Rules: max 2000 chars total for new_content. "
            "Only flag real problems, don't over-merge."
        )

        try:
            llm = _model()
            response = llm.invoke([HumanMessage(content=prompt)])
            plan = json.loads(
                response.content.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
        except Exception as e:
            print(f"[Dream] LLM analysis failed: {e}")
            self._release_lock()
            return summary

        # Phase 4: Apply changes
        print("[Dream] Phase 4: applying changes...")
        for action in plan.get("actions", []):
            act = action.get("action", "keep")
            files = action.get("files", [])

            if act == "merge":
                for f in files:
                    fp = self.memory_dir / f
                    if fp.exists():
                        fp.unlink()
                        summary["deleted"] += 1
                memory_mgr.save_memory(
                    action.get("new_name", "merged"),
                    action.get("new_description", ""),
                    action.get("new_type", "project"),
                    action.get("new_content", ""),
                )
                summary["merged"] += 1
                print(f"  [Dream] merged {files} → {action.get('new_name')}")

            elif act == "delete":
                for f in files:
                    fp = self.memory_dir / f
                    if fp.exists():
                        fp.unlink()
                        summary["deleted"] += 1
                print(f"  [Dream] deleted {files}")

            elif act == "keep":
                summary["kept"] += len(files)

        memory_mgr.load_all()  # reload after changes
        self.last_consolidation_time = time.time()
        self._release_lock()
        print(
            f"[Dream] Done: {summary['merged']} merged, "
            f"{summary['deleted']} deleted, {summary['kept']} kept"
        )
        return summary

    def _acquire_lock(self) -> bool:
        if self.lock_file.exists():
            try:
                pid_str, ts_str = self.lock_file.read_text().strip().split(":", 1)
                pid, lock_time = int(pid_str), float(ts_str)
                if (time.time() - lock_time) > self.LOCK_STALE_SECONDS:
                    print(f"[Dream] Removing stale lock from PID {pid}")
                    self.lock_file.unlink()
                else:
                    try:
                        os.kill(pid, 0)
                        return False
                    except OSError:
                        print(f"[Dream] Removing lock from dead PID {pid}")
                        self.lock_file.unlink()
            except (ValueError, OSError):
                self.lock_file.unlink(missing_ok=True)
        try:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self.lock_file.write_text(f"{os.getpid()}:{time.time()}")
            return True
        except OSError:
            return False

    def _release_lock(self):
        try:
            if self.lock_file.exists():
                pid_str = self.lock_file.read_text().strip().split(":")[0]
                if int(pid_str) == os.getpid():
                    self.lock_file.unlink()
        except (ValueError, OSError):
            pass


dream = DreamConsolidator()


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
def save_memory(name: str, description: str, type: str, content: str) -> str:
    """Save a persistent memory that survives across sessions.
    Types: user (preferences), feedback (corrections), project (non-obvious conventions), reference (external resources).
    """
    return memory_mgr.save_memory(name, description, type, content)


ALL_TOOLS = [bash, read_file, write_file, edit_file, save_memory]
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
    last_ai = state["messages"][-1]
    final_msgs = []
    for tc in last_ai.tool_calls:
        name, tid, args = tc["name"], tc["id"], tc.get("args", {}) or {}
        tool_fn = TOOL_BY_NAME.get(name)
        if tool_fn:
            try:
                content = tool_fn.invoke(args)
            except Exception as e:
                content = f"Error: {e}"
        else:
            content = f"Unknown tool: {name}"
        print(f"> {name}: {str(content)[:200]}")
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
app = graph.compile()

if __name__ == "__main__":
    state: AgentState = {"messages": []}

    while (q := input("\033[36ms09 >> \033[0m")) not in ("q", "exit", ""):
        state["messages"].append(HumanMessage(content=q))
        state = app.invoke(state)
        dream.consolidate()  # 每次结束后检查，门控防止频繁执行
        print()
