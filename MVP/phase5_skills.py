#!/usr/bin/env python3
# Harness: on-demand knowledge -- discover skills cheaply, load them only when needed.
"""
skills.py - Skills

核心设计:
1. 系统提示中只放廉价的技能目录
2. 仅在模型请求时才加载完整技能内容

LangGraph 内置复用:
- create_react_agent  → 替代整个 agent_loop + while True 工具调用循环
- ToolNode            → 替代手写的 TOOL_HANDLERS 分发
- MessagesState       → 替代手写的 messages list 管理
- tools_condition     → 替代手写的 stop_reason 判断
"""

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from typing_extensions import TypedDict
from typing import Annotated, Any, Literal, Optional

load_dotenv(override=True)

os.environ["NO_PROXY"] = "*"
MODEL_ID = os.environ["AGENCY_LLM_MODEL"]
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER")
WORKDIR = Path.cwd()
AGENCY_DIR = WORKDIR / ".mini-agent-cli"
SKILLS_DIR = AGENCY_DIR / "skills"


@dataclass
class SkillManifest:
    name: str
    description: str
    path: Path


@dataclass
class SkillDocument:
    manifest: SkillManifest
    body: str


class SkillRegistry:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.documents: dict[str, SkillDocument] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return
        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            meta, body = self._parse_frontmatter(path.read_text())
            name = meta.get("name", path.parent.name)
            description = meta.get("description", "No description")
            manifest = SkillManifest(name=name, description=description, path=path)
            self.documents[name] = SkillDocument(manifest=manifest, body=body.strip())

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta: dict = {}
        for line in match.group(1).strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
        return meta, match.group(2)

    def describe_available(self) -> str:
        if not self.documents:
            return "(no skills available)"
        lines = []
        for name in sorted(self.documents):
            manifest = self.documents[name].manifest
            lines.append(f"- {manifest.name}: {manifest.description}")
        return "\n".join(lines)

    def load_full_text(self, name: str) -> str:
        document = self.documents.get(name)
        if not document:
            known = ", ".join(sorted(self.documents)) or "(none)"
            return f"Error: Unknown skill '{name}'. Available skills: {known}"
        return (
            f'<skill name="{document.manifest.name}">\n' f"{document.body}\n" "</skill>"
        )


SKILL_REGISTRY = SkillRegistry(SKILLS_DIR)


def _safe_path(path_str: str) -> Path:
    path = (WORKDIR / path_str).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return path


@tool
def bash(command: str) -> str:
    """Run a shell command."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "[Error]: Timeout (120s)"
    output = (result.stdout + result.stderr).strip()
    return output[:50000] if output else "(no output)"


@tool
def read_file(path: str, limit: int | None = None) -> str:
    """Read file contents."""
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as exc:
        return f"[Error]: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        file_path = _safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"[Error]: {exc}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in a file once."""
    try:
        file_path = _safe_path(path)
        content = file_path.read_text()
        if old_text not in content:
            return f"[Error]: Text not found in {path}"
        file_path.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as exc:
        return f"[Error]: {exc}"


@tool
def load_skill(name: str) -> str:
    """Load the full body of a named skill into the current context."""
    return SKILL_REGISTRY.load_full_text(name)


tools = [bash, read_file, write_file, edit_file, load_skill]
tool_map = {t.name: t for t in tools}
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill when a task needs specialized instructions before you act.
Skills available:
{SKILL_REGISTRY.describe_available()}
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


model_with_tools = init_chat_model(
    MODEL_ID,
    model_provider=PROVIDER,  # explicitly set provider
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
).bind_tools(tools)


def execute_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []
    tool_calls = last_message.tool_calls
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_handler = tool_map.get(tool_name)
        if tool_handler:
            print(f"> {tool_name}: ")
            try:
                output = tool_handler.invoke(tool_args)
            except Exception as e:
                output = f"[Error]: {e}"
        else:
            output = f"Unknown tool: {tool_name}"
        print(str(output[:200]))
        tool_messages.append(
            ToolMessage(content=str(output), tool_call_id=tool_call["id"])
        )
        return {"messages": tool_messages}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"


def call_model(state: AgentState) -> dict:
    """流式输出模型的响应"""
    # 如果需要提醒，则在消息副本中修改最后一条HumanMessage
    messages_to_use = state["messages"]
    full_response = None
    for chunk in model_with_tools.stream(messages_to_use):
        if isinstance(chunk, AIMessageChunk):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if full_response is None:
                full_response = chunk
            else:
                full_response = full_response + chunk
    print()  # 换行
    if full_response is None:
        full_response = AIMessage(content="")
    return {"messages": [full_response]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.add_edge(START, "agent")
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("agent", should_continue)

graph = workflow.compile()

if __name__ == "__main__":
    state = {"messages": []}
    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        messages = state["messages"]
        messages.append(HumanMessage(content=query))
        state = graph.invoke(state)
        print()
