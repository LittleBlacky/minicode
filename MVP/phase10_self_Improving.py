#!/usr/bin/env python3
"""
self_improving_agent.py — Self-Improving Agent with Typed Memory + Dream Consolidation
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ───────────────────────────── 环境配置 ──────────────────────────────────────

load_dotenv(override=True)
os.environ.setdefault("NO_PROXY", "*")

MODEL_ID = os.environ.get("AGENCY_LLM_MODEL", "gpt-4o")
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER", "openai")

WORKDIR = Path.cwd()
STORAGE_DIR = WORKDIR / ".mini-agent-cli"
MEMORY_DIR = STORAGE_DIR / ".memory"
SKILLS_DIR = STORAGE_DIR / "skills"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"

MEMORY_TYPES = ("user", "feedback", "project", "reference")
MAX_INDEX_LINES = 200
MEMORY_CHAR_CAP = 3000  # 热记忆注入上限（字符）


# ═══════════════════════════════════════════════════════════════════════════════
# ①  MemoryManager —— 分类型 Markdown 持久记忆
# ═══════════════════════════════════════════════════════════════════════════════


class MemoryManager:
    """
    将记忆按类型存储为独立 Markdown 文件，并维护 MEMORY.md 索引。

    类型说明：
      user      — 用户偏好（"我喜欢用 pytest"）
      feedback  — 用户纠正（"别这样做，因为…"）
      project   — 非显而易见的项目事实（合规要求、遗留限制）
      reference — 外部资源指针（文档 URL、看板地址）
    """

    def __init__(self, memory_dir: Path = MEMORY_DIR):
        self.memory_dir = memory_dir
        self.memories: dict[str, dict] = {}

    # ── 加载 ────────────────────────────────────────────────────────────────

    def load_all(self) -> None:
        self.memories = {}
        if not self.memory_dir.exists():
            return
        for md_file in sorted(self.memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            parsed = self._parse_frontmatter(md_file.read_text(encoding="utf-8"))
            if parsed:
                name = parsed.get("name", md_file.stem)
                self.memories[name] = {
                    "description": parsed.get("description", ""),
                    "type": parsed.get("type", "project"),
                    "content": parsed.get("content", ""),
                    "file": md_file.name,
                }
        if self.memories:
            print(f"[Memory] 已加载 {len(self.memories)} 条记忆")

    # ── Prompt 注入 ─────────────────────────────────────────────────────────

    def build_prompt_section(self) -> str:
        """构建注入 system prompt 的记忆段落（按类型分组）。"""
        if not self.memories:
            return "[No persistent memory yet]"
        lines = ["# 持久记忆（跨会话）", ""]
        for mem_type in MEMORY_TYPES:
            typed = {k: v for k, v in self.memories.items() if v["type"] == mem_type}
            if not typed:
                continue
            lines.append(f"## [{mem_type}]")
            for name, mem in typed.items():
                lines.append(f"### {name}: {mem['description']}")
                if mem["content"].strip():
                    lines.append(mem["content"].strip())
                lines.append("")
        result = "\n".join(lines)
        # 防止热记忆超出上限
        return result[:MEMORY_CHAR_CAP] if len(result) > MEMORY_CHAR_CAP else result

    # ── 保存 ────────────────────────────────────────────────────────────────

    def save(self, name: str, description: str, mem_type: str, content: str) -> str:
        if mem_type not in MEMORY_TYPES:
            return f"Error: type must be one of {MEMORY_TYPES}"
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.lower()) or "unnamed"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        text = f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n{content}\n"
        path = self.memory_dir / f"{safe}.md"
        path.write_text(text, encoding="utf-8")
        self.memories[name] = {
            "description": description,
            "type": mem_type,
            "content": content,
            "file": path.name,
        }
        self._rebuild_index()
        return f"[Memory] 已保存 '{name}' [{mem_type}]"

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        lines = ["# Memory Index", ""]
        for i, (name, mem) in enumerate(self.memories.items()):
            lines.append(f"- {name}: {mem['description']} [{mem['type']}]")
            if i >= MAX_INDEX_LINES:
                lines.append(f"... (超过 {MAX_INDEX_LINES} 条，已截断)")
                break
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "MEMORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _parse_frontmatter(text: str) -> Optional[dict]:
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if not m:
            return None
        header, body = m.group(1), m.group(2)
        result: dict = {"content": body.strip()}
        for line in header.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                result[k.strip()] = v.strip()
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# ②  SkillStore —— 冷存储技能库
# ═══════════════════════════════════════════════════════════════════════════════


class SkillStore:
    """
    存储从成功任务中提炼的可复用技能。

    只有技能索引（name + description）常驻内存；
    完整 procedure 在路由器命中时才按需加载。
    """

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def get_index(self) -> list[dict]:
        """返回轻量索引列表（仅含 name、description、use_count、avg_score）。"""
        index = []
        for f in sorted(self.skills_dir.glob("*.json")):
            skill = json.loads(f.read_text(encoding="utf-8"))
            index.append(
                {
                    "name": skill["name"],
                    "description": skill["description"],
                    "task_type": skill.get("task_type", ""),
                    "use_count": skill.get("use_count", 0),
                    "avg_score": skill.get("avg_score", 0.0),
                }
            )
        return index

    def get(self, skill_name: str) -> Optional[dict]:
        path = self.skills_dir / f"{skill_name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def save(self, skill: dict) -> None:
        skill["updated_at"] = datetime.now().isoformat()
        path = self.skills_dir / f"{skill['name']}.json"
        path.write_text(
            json.dumps(skill, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def update_stats(self, skill_name: str, score: float) -> None:
        skill = self.get(skill_name)
        if skill is None:
            return
        skill["use_count"] = skill.get("use_count", 0) + 1
        prior = skill.get("avg_score", 0.0) * (skill["use_count"] - 1)
        skill["avg_score"] = round((prior + score) / skill["use_count"], 2)
        self.save(skill)


# ═══════════════════════════════════════════════════════════════════════════════
# ③  DreamConsolidator —— 跨会话记忆整合
# ═══════════════════════════════════════════════════════════════════════════════


class DreamConsolidator:
    """
    多门控防抖的 "梦境整合" 系统：
      - 每 N 次会话触发一次
      - 24 小时冷却
      - 进程锁防并发
      - 四阶段：扫描 → 收集 → LLM 合并 → 写回
    """

    COOLDOWN_SECONDS = 86_400  # 24 h
    SCAN_THROTTLE_SECS = 600  # 10 min
    MIN_SESSION_COUNT = 5
    LOCK_STALE_SECONDS = 3_600

    def __init__(self, memory_mgr: MemoryManager):
        self.memory_mgr = memory_mgr
        self.lock_file = memory_mgr.memory_dir / ".dream_lock"
        self.last_consolidation = 0.0
        self.last_scan = 0.0
        self.session_count = 0

    def tick(self) -> None:
        """每次任务完成后调用，内部门控决定是否实际整合。"""
        self.session_count += 1
        ok, reason = self._should_run()
        if not ok:
            print(f"[Dream] 跳过整合: {reason}")
            return
        self._run()

    def _should_run(self) -> tuple[bool, str]:
        now = time.time()
        if not self.memory_mgr.memory_dir.exists():
            return False, "memory_dir 不存在"
        files = [
            f for f in self.memory_mgr.memory_dir.glob("*.md") if f.name != "MEMORY.md"
        ]
        if not files:
            return False, "没有记忆文件"
        if self.session_count < self.MIN_SESSION_COUNT:
            return False, f"会话数 {self.session_count} < {self.MIN_SESSION_COUNT}"
        if now - self.last_consolidation < self.COOLDOWN_SECONDS:
            remaining = int(self.COOLDOWN_SECONDS - (now - self.last_consolidation))
            return False, f"冷却中，剩余 {remaining}s"
        if now - self.last_scan < self.SCAN_THROTTLE_SECS:
            remaining = int(self.SCAN_THROTTLE_SECS - (now - self.last_scan))
            return False, f"扫描节流，剩余 {remaining}s"
        if not self._acquire_lock():
            return False, "锁被其他进程持有"
        return True, "全部门控通过"

    def _run(self) -> None:
        self.last_scan = time.time()
        print("[Dream] 开始记忆整合...")

        # 收集所有记忆内容
        files = [
            f for f in self.memory_mgr.memory_dir.glob("*.md") if f.name != "MEMORY.md"
        ]
        all_mem = {}
        for f in files:
            parsed = self.memory_mgr._parse_frontmatter(f.read_text(encoding="utf-8"))
            if parsed:
                all_mem[f.name] = parsed

        if not all_mem:
            self._release_lock()
            return

        mem_text = json.dumps(
            {
                k: {
                    "description": v.get("description", ""),
                    "type": v.get("type", ""),
                    "content": v.get("content", "")[:400],
                }
                for k, v in all_mem.items()
            },
            ensure_ascii=False,
            indent=2,
        )

        prompt = (
            "你是记忆管理员。分析以下记忆条目，找出：\n"
            "1. 重复条目（相同概念，不同名称）→ 合并\n"
            "2. 矛盾条目（相互冲突的建议）→ 解决\n"
            "3. 过时条目（不再相关）→ 删除\n\n"
            f"记忆：\n{mem_text}\n\n"
            '仅返回 JSON：{"actions": [{"action": "merge|delete|keep", '
            '"files": ["file1.md"], "new_name": "...", "new_description": "...", '
            '"new_type": "user|feedback|project|reference", "new_content": "..."}]}\n'
            "规则：new_content 总长不超过 2000 字；只标记真实问题，不过度合并。"
        )

        try:
            llm = _build_llm()
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content.strip().replace("```json", "").replace("```", "").strip()
            plan = json.loads(raw)
        except Exception as e:
            print(f"[Dream] LLM 分析失败: {e}")
            self._release_lock()
            return

        merged = deleted = kept = 0
        for act in plan.get("actions", []):
            action = act.get("action", "keep")
            files_list = act.get("files", [])
            if action == "merge":
                for fn in files_list:
                    fp = self.memory_mgr.memory_dir / fn
                    if fp.exists():
                        fp.unlink()
                        deleted += 1
                self.memory_mgr.save(
                    act.get("new_name", "merged"),
                    act.get("new_description", ""),
                    act.get("new_type", "project"),
                    act.get("new_content", ""),
                )
                merged += 1
                print(f"  [Dream] 合并 {files_list} → {act.get('new_name')}")
            elif action == "delete":
                for fn in files_list:
                    fp = self.memory_mgr.memory_dir / fn
                    if fp.exists():
                        fp.unlink()
                        deleted += 1
                print(f"  [Dream] 删除 {files_list}")
            else:
                kept += len(files_list)

        self.memory_mgr.load_all()
        self.last_consolidation = time.time()
        self._release_lock()
        print(f"[Dream] 完成：合并 {merged}，删除 {deleted}，保留 {kept}")

    def _acquire_lock(self) -> bool:
        if self.lock_file.exists():
            try:
                pid_s, ts_s = self.lock_file.read_text().strip().split(":", 1)
                pid, lock_ts = int(pid_s), float(ts_s)
                if time.time() - lock_ts > self.LOCK_STALE_SECONDS:
                    self.lock_file.unlink()
                else:
                    try:
                        os.kill(pid, 0)
                        return False
                    except OSError:
                        self.lock_file.unlink()
            except (ValueError, OSError):
                self.lock_file.unlink(missing_ok=True)
        try:
            self.memory_mgr.memory_dir.mkdir(parents=True, exist_ok=True)
            self.lock_file.write_text(f"{os.getpid()}:{time.time()}")
            return True
        except OSError:
            return False

    def _release_lock(self) -> None:
        try:
            if self.lock_file.exists():
                pid_s = self.lock_file.read_text().strip().split(":")[0]
                if int(pid_s) == os.getpid():
                    self.lock_file.unlink()
        except (ValueError, OSError):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# ④  工具链
# ═══════════════════════════════════════════════════════════════════════════════


def _safe_path(p: str) -> Path:
    resolved = (WORKDIR / p).resolve()
    if not resolved.is_relative_to(WORKDIR):
        raise ValueError(f"路径越界: {p}")
    return resolved


# 全局记忆管理器（工具需引用）
_memory_mgr_global: Optional[MemoryManager] = None


@tool
def bash(command: str) -> str:
    """在工作目录执行 shell 命令（超时 120 秒）。"""
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
        return "Error: 超时"
    return (r.stdout + r.stderr).strip() or "(无输出)"


@tool
def read_file(path: str, limit: int = None) -> str:
    """读取文件内容（可限制行数）。"""
    try:
        p = _safe_path(path)
        lines = p.read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... (还有 {len(lines)-limit} 行)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """将内容写入文件（自动创建父目录）。"""
    try:
        f = _safe_path(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        return f"已写入 {len(content)} 字节到 {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """在文件中精确替换一处文本。"""
    try:
        f = _safe_path(path)
        content = f.read_text(encoding="utf-8")
        if old_text not in content:
            return f"Error: 未在 {path} 中找到目标文本"
        f.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"已编辑 {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def save_memory_tool(name: str, description: str, mem_type: str, content: str) -> str:
    """
    保存跨会话持久记忆。
    mem_type: user（偏好）| feedback（纠正）| project（项目事实）| reference（外部资源）
    """
    if _memory_mgr_global is None:
        return "Error: MemoryManager 未初始化"
    return _memory_mgr_global.save(name, description, mem_type, content)


ALL_TOOLS = [bash, read_file, write_file, edit_file, save_memory_tool]
TOOL_BY_NAME = {t.name: t for t in ALL_TOOLS}


# ═══════════════════════════════════════════════════════════════════════════════
# ⑤  LangGraph 状态
# ═══════════════════════════════════════════════════════════════════════════════


class AgentState(TypedDict):
    """流经整个图的状态对象。"""

    # ── 任务信息 ──────────────────────────────
    task: str
    task_type: str
    matched_skill: Optional[dict]
    # ── 执行结果 ──────────────────────────────
    execution_steps: list[str]
    result: str
    tool_messages: Annotated[list, add_messages]  # 工具调用记录
    # ── 评估与学习 ────────────────────────────
    evaluation_score: float
    should_create_skill: bool
    should_update_memory: bool
    task_count: int


# ═══════════════════════════════════════════════════════════════════════════════
# ⑥  SelfImprovingAgent —— 五节点图
# ═══════════════════════════════════════════════════════════════════════════════


def _build_llm(with_tools: bool = False):
    llm = init_chat_model(
        model=MODEL_ID, model_provider=PROVIDER, base_url=BASE_URL, api_key=API_KEY
    )
    return llm.bind_tools(ALL_TOOLS) if with_tools else llm


def _parse_json(text: str) -> Optional[object]:
    """从 LLM 输出中鲁棒地提取第一个 JSON 对象/数组。"""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, candidate, re.DOTALL)
        if not m:
            continue
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


MEMORY_GUIDANCE = """
何时保存记忆：
- 用户表达偏好 → type: user
- 用户纠正你的做法 → type: feedback
- 发现无法从代码直接推断的项目约定 → type: project
- 发现外部资源地址（文档、看板、URL）→ type: reference
不要保存：
- 可从代码结构直接推断的内容
- 临时任务状态（当前分支、PR 编号）
- 密钥或凭据
"""


class SelfImprovingAgent:
    """
    自改进编码智能体，整合了两套系统的全部能力。

    参数
    ----
    skill_threshold   : 触发技能提取的最低评分（满分 10）
    min_steps         : 触发技能提取的最少执行步数
    """

    def __init__(
        self,
        skill_threshold: float = 7.0,
        min_steps: int = 3,
    ):
        global _memory_mgr_global

        # 初始化三大子系统
        self.memory = MemoryManager(MEMORY_DIR)
        self.skills = SkillStore(SKILLS_DIR)
        self.dream = DreamConsolidator(self.memory)

        _memory_mgr_global = self.memory  # 让工具可访问
        self.memory.load_all()

        self.skill_threshold = skill_threshold
        self.min_steps = min_steps

        # 构建 LLM（带工具版用于 executor，不带工具版用于其他节点）
        self._llm = _build_llm(with_tools=False)
        self._llm_tools = _build_llm(with_tools=True)

    # ── Node 1: 路由器 ──────────────────────────────────────────────────────

    def _router(self, state: AgentState) -> dict:
        skill_index = self.skills.get_index()
        skill_list = (
            "\n".join(
                f"  - {s['name']}: {s['description']} (用了{s['use_count']}次, 均分={s['avg_score']})"
                for s in skill_index
            )
            or "  [暂无技能]"
        )

        prompt = (
            "你是任务路由器。分析用户任务：\n"
            "1. 归类 task_type（如 code_review / data_analysis / debugging / writing / shell_task）\n"
            "2. 检查现有技能是否匹配，若匹配则返回技能名\n\n"
            f"可用技能：\n{skill_list}\n\n"
            f"持久记忆：\n{self.memory.build_prompt_section()}\n\n"
            f"用户任务：{state['task']}\n\n"
            '仅返回 JSON：{{"task_type": "...", "matched_skill": "技能名或null"}}'
        )

        resp = self._llm.invoke([SystemMessage(content=prompt)])
        parsed = _parse_json(resp.content) or {}

        task_type = (
            parsed.get("task_type", "general")
            if isinstance(parsed, dict)
            else "general"
        )
        matched_name = parsed.get("matched_skill") if isinstance(parsed, dict) else None

        if isinstance(matched_name, str) and matched_name.strip().lower() in (
            "null",
            "none",
            "",
        ):
            matched_name = None

        matched = self.skills.get(matched_name) if matched_name else None
        print(f"[Router] task_type={task_type}, matched_skill={matched_name or '无'}")
        return {"task_type": task_type, "matched_skill": matched}

    # ── Node 2: 执行器（含真实工具）───────────────────────────────────────

    def _executor(self, state: AgentState) -> dict:
        skill_guidance = ""
        if state.get("matched_skill"):
            sk = state["matched_skill"]
            skill_guidance = (
                f"\n【匹配技能】{sk['name']}\n"
                f"步骤：{sk.get('procedure','N/A')}\n"
                f"避坑：{sk.get('pitfalls','N/A')}\n"
                f"验证：{sk.get('verification','N/A')}\n"
            )

        system = (
            f"你是一个编码智能体，工作目录 {WORKDIR}。\n"
            f"使用工具完成任务，逐步思考并记录每一步。\n"
            f"{MEMORY_GUIDANCE}\n"
            f"{skill_guidance}\n"
            f"持久记忆：\n{self.memory.build_prompt_section()}"
        )

        messages: list = [
            SystemMessage(content=system),
            HumanMessage(content=state["task"]),
        ]

        steps: list[str] = []
        result: str = ""

        # 工具调用循环（ReAct 风格）
        for _ in range(10):  # 最多 10 轮工具调用
            resp = self._llm_tools.invoke(messages)
            messages.append(resp)

            # 收集纯文本内容
            if isinstance(resp.content, str) and resp.content.strip():
                result = resp.content

            tool_calls = getattr(resp, "tool_calls", []) or []
            if not tool_calls:
                break  # 无工具调用 → 任务完成

            tool_msgs = []
            for tc in tool_calls:
                name, tid, args = tc["name"], tc["id"], tc.get("args", {}) or {}
                fn = TOOL_BY_NAME.get(name)
                try:
                    content = fn.invoke(args) if fn else f"未知工具: {name}"
                except Exception as e:
                    content = f"Error: {e}"
                steps.append(f"{name}({json.dumps(args, ensure_ascii=False)[:80]})")
                print(f"  [Tool] {name}: {str(content)[:160]}")
                tool_msgs.append(ToolMessage(content=str(content), tool_call_id=tid))

            messages.extend(tool_msgs)

        if not result and steps:
            result = f"已完成 {len(steps)} 步工具调用"
        if not result:
            result = "任务已完成"

        return {
            "execution_steps": steps,
            "result": result,
            "tool_messages": messages[2:],
        }

    # ── Node 3: 评估器 ──────────────────────────────────────────────────────

    def _evaluator(self, state: AgentState) -> dict:
        prompt = (
            "评估此次任务执行质量（1-10 分）。\n\n"
            f"任务：{state['task']}\n"
            f"执行步骤：{json.dumps(state['execution_steps'], ensure_ascii=False)}\n"
            f"结果：{state['result']}\n\n"
            "评分维度：正确性、完整性、效率。\n"
            '仅返回 JSON：{{"score": N, "reasoning": "..."}}'
        )
        resp = self._llm.invoke([SystemMessage(content=prompt)])
        parsed = _parse_json(resp.content)

        try:
            score = float(parsed["score"]) if isinstance(parsed, dict) else 5.0
        except (KeyError, TypeError, ValueError):
            score = 5.0

        task_count = state.get("task_count", 0) + 1
        num_steps = len(state.get("execution_steps", []))

        should_create = (
            score >= self.skill_threshold
            and num_steps >= self.min_steps
            and not state.get("matched_skill")
        )

        if state.get("matched_skill"):
            self.skills.update_stats(state["matched_skill"]["name"], score)

        print(f"[Evaluator] 得分={score:.1f}, 创建技能={should_create}")
        return {
            "evaluation_score": score,
            "should_create_skill": should_create,
            "task_count": task_count,
        }

    # ── Node 4: 技能提取器 ─────────────────────────────────────────────────

    def _skill_extractor(self, state: AgentState) -> dict:
        if not state.get("should_create_skill"):
            return {}

        prompt = (
            "从这次成功的任务执行中提炼一个可复用技能。\n\n"
            f"任务类型：{state['task_type']}\n"
            f"任务：{state['task']}\n"
            f"步骤：{json.dumps(state['execution_steps'], ensure_ascii=False)}\n"
            f"结果：{state['result']}\n"
            f"得分：{state['evaluation_score']}\n\n"
            "返回 JSON 技能文档：\n"
            "{{\n"
            '  "name": "short_snake_case",\n'
            '  "description": "一句话说明何时使用此技能",\n'
            f'  "task_type": "{state["task_type"]}",\n'
            '  "procedure": "可复用的分步流程",\n'
            '  "pitfalls": "常见错误",\n'
            '  "verification": "如何验证结果正确",\n'
            '  "use_count": 1,\n'
            f'  "avg_score": {state["evaluation_score"]}\n'
            "}}"
        )
        resp = self._llm.invoke([SystemMessage(content=prompt)])
        skill = _parse_json(resp.content)

        if isinstance(skill, dict) and "name" in skill:
            self.skills.save(skill)
            print(f"[SkillExtractor] 新技能已保存: {skill['name']}")
        else:
            print("[SkillExtractor] 技能 JSON 解析失败，跳过")
        return {}

    # ── Node 5: 记忆更新（Dream 触发入口）───────────────────────────────────

    def _memory_updater(self, state: AgentState) -> dict:
        """
        用 DreamConsolidator 替代 Doc1 的简单 LLM 整合，
        内部门控决定是否真正执行。
        """
        self.dream.tick()
        return {}

    # ── 条件路由 ────────────────────────────────────────────────────────────

    def _after_evaluator(self, state: AgentState) -> str:
        if state.get("should_create_skill"):
            return "skill_extractor"
        return "memory_updater"  # 无论是否整合，都过一遍 Dream 门控

    def _after_skill_extractor(self, state: AgentState) -> str:
        return "memory_updater"

    # ── 图构建 ──────────────────────────────────────────────────────────────

    def build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("router", self._router)
        graph.add_node("executor", self._executor)
        graph.add_node("evaluator", self._evaluator)
        graph.add_node("skill_extractor", self._skill_extractor)
        graph.add_node("memory_updater", self._memory_updater)

        graph.add_edge(START, "router")
        graph.add_edge("router", "executor")
        graph.add_edge("executor", "evaluator")
        graph.add_conditional_edges(
            "evaluator",
            self._after_evaluator,
            {"skill_extractor": "skill_extractor", "memory_updater": "memory_updater"},
        )
        graph.add_conditional_edges(
            "skill_extractor",
            self._after_skill_extractor,
            {"memory_updater": "memory_updater"},
        )
        graph.add_edge("memory_updater", END)
        # Compile with checkpoint for session persistence (LangGraph native)
        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)

    def get_session_config(self, thread_id: str = "self_improving_session_1") -> dict:
        """LangGraph native: Get session config for checkpointing."""
        return {"configurable": {"thread_id": thread_id}}

    def run(self, task: str, task_count: int = 0, config: dict = None) -> dict:
        """Run the agent with optional checkpoint config."""
        # Cache the compiled graph for efficiency
        if not hasattr(self, '_compiled_graph'):
            self._compiled_graph = self.build_graph()

        graph = self._compiled_graph
        initial: AgentState = {
            "task": task,
            "task_type": "",
            "matched_skill": None,
            "execution_steps": [],
            "result": "",
            "tool_messages": [],
            "evaluation_score": 0.0,
            "should_create_skill": False,
            "should_update_memory": False,
            "task_count": task_count,
        }
        if config:
            return graph.invoke(initial, config)
        return graph.invoke(initial)


# ═══════════════════════════════════════════════════════════════════════════════
# ⑦  交互式 REPL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Self-Improving Agent  (输入 q 或 exit 退出)")
    print("  LangGraph Native Patterns - Checkpoint Persistence")
    print("=" * 60)
    agent = SelfImprovingAgent()
    graph = agent.build_graph()
    config = agent.get_session_config()
    task_count = 0

    # Resume from checkpoint if exists
    existing = graph.get_state(config)
    if existing and existing.values:
        print(f"[Resuming session with {existing.values.get('task_count', 0)} previous tasks]\n")

    print("Features: Checkpoint persistence, skill extraction, self-improvement")
    print("Type 'exit' or 'q' to quit\n")

    while True:
        try:
            q = input("\n\033[36m>> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if q.lower() in ("q", "exit", ""):
            break

        result = agent.run(q, task_count=task_count, config=config)
        task_count = result.get("task_count", task_count + 1)

        print("\n" + "─" * 50)
        print(f"结果：{result.get('result', '')}")
        print(f"得分：{result.get('evaluation_score', 0):.1f} / 10")
        print(f"步骤：{result.get('execution_steps', [])}")
        print("─" * 50)
