"""Memory Layer - 三层架构

记忆的三种类型:
1. 静态记忆 (Static) - 用户偏好、项目配置 → 注入 System Prompt
2. 动态记忆 (Session) - 当前会话上下文 → 每次运行追加
3. 事件记忆 (Episodic) - 过去发生的事 → 按需检索
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional, TypedDict
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


WORKDIR = Path.cwd()
MEMORY_DIR = WORKDIR / ".minicode" / "memory"
STATIC_DIR = MEMORY_DIR / "static"
SESSION_DIR = MEMORY_DIR / "session"


@dataclass
class MemoryEntry:
    """记忆条目"""
    name: str
    description: str
    content: str
    memory_type: str  # user, feedback, project, reference
    created_at: float
    access_count: int = 0
    last_accessed: float = 0


@dataclass
class SessionContext:
    """会话上下文 - 动态记忆"""
    thread_id: str
    created_at: float = field(default_factory=time.time)
    task_id: str = ""
    task_description: str = ""
    recent_decisions: list[str] = field(default_factory=list)
    pending_items: list[str] = field(default_factory=list)
    completed_items: list[str] = field(default_factory=list)
    session_summary: str = ""


class MemoryIndex:
    """记忆索引 - 快速检索 (用于事件记忆)"""

    def __init__(self, memory_dir: Path = MEMORY_DIR):
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, MemoryEntry] = {}
        self._load_index()

    def _load_index(self) -> None:
        """加载索引"""
        self._index.clear()
        for md_file in self.memory_dir.glob("*.md"):
            if md_file.name in ("MEMORY.md", "STATIC.md"):
                continue
            entry = self._parse_memory_file(md_file)
            if entry:
                self._index[entry.name] = entry

    def _parse_memory_file(self, file_path: Path) -> Optional[MemoryEntry]:
        """解析记忆文件"""
        try:
            text = file_path.read_text(encoding="utf-8")
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
            if not match:
                return None

            header, content = match.groups()
            meta = {}
            for line in header.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()

            created_match = re.search(r"created_at:\s*(.+)", header)
            created_at = float(created_match.group(1)) if created_match else time.time()

            return MemoryEntry(
                name=meta.get("name", file_path.stem),
                description=meta.get("description", ""),
                content=content.strip(),
                memory_type=meta.get("type", "unknown"),
                created_at=created_at,
            )
        except Exception:
            return None

    def save_entry(self, entry: MemoryEntry) -> None:
        """保存记忆条目"""
        safe_name = entry.name.replace(" ", "-").lower()
        file_path = self.memory_dir / f"{safe_name}.md"

        frontmatter = f"""---
name: {entry.name}
description: {entry.description}
type: {entry.memory_type}
created_at: {entry.created_at}
access_count: {entry.access_count}
---

{entry.content}
"""
        file_path.write_text(frontmatter, encoding="utf-8")
        self._index[entry.name] = entry

    def search(self, query: str, memory_type: Optional[str] = None, limit: int = 3) -> list[MemoryEntry]:
        """搜索相关记忆"""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for name, entry in self._index.items():
            if memory_type and entry.memory_type != memory_type:
                continue

            score = 0
            if query_lower in entry.name.lower():
                score += 10
            if query_lower in entry.description.lower():
                score += 5
            for word in query_words:
                if word in entry.content.lower():
                    score += 1
            score += min(entry.access_count * 0.1, 2)

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored[:limit]]

        for entry in results:
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.save_entry(entry)

        return results

    def list_all(self) -> list[dict]:
        return [
            {
                "name": e.name,
                "description": e.description,
                "type": e.memory_type,
                "created_at": e.created_at,
            }
            for e in self._index.values()
        ]


class StaticMemory:
    """静态记忆 - 用户偏好、项目配置

    这部分记忆会注入到 System Prompt 中
    """

    def __init__(self):
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        self.preferences_file = STATIC_DIR / "preferences.md"
        self.project_file = STATIC_DIR / "project.md"
        self.skills_file = STATIC_DIR / "skills.md"

    def save_preference(self, key: str, value: str) -> None:
        """保存用户偏好"""
        content = self.preferences_file.read_text() if self.preferences_file.exists() else ""
        lines = content.split('\n') if content else []

        # 更新或添加
        new_lines = []
        found = False
        for line in lines:
            if line.startswith(f"{key}:"):
                new_lines.append(f"{key}: {value}")
                found = True
            else:
                new_lines.append(line)

        if not found:
            new_lines.append(f"{key}: {value}")

        self.preferences_file.write_text('\n'.join(new_lines), encoding="utf-8")

    def get_preferences(self) -> str:
        """获取用户偏好"""
        if not self.preferences_file.exists():
            return ""

        content = self.preferences_file.read_text(encoding="utf-8")
        if not content.strip():
            return ""

        return f"\n\n# 用户偏好\n{content}\n"

    def save_project_knowledge(self, key: str, value: str) -> None:
        """保存项目知识"""
        content = self.project_file.read_text() if self.project_file.exists() else ""
        lines = content.split('\n') if content else []

        new_lines = []
        found = False
        for line in lines:
            if line.startswith(f"{key}:"):
                new_lines.append(f"{key}: {value}")
                found = True
            else:
                new_lines.append(line)

        if not found:
            new_lines.append(f"{key}: {value}")

        self.project_file.write_text('\n'.join(new_lines), encoding="utf-8")

    def get_project_knowledge(self) -> str:
        """获取项目知识"""
        if not self.project_file.exists():
            return ""

        content = self.project_file.read_text(encoding="utf-8")
        if not content.strip():
            return ""

        return f"\n\n# 项目知识\n{content}\n"

    def save_skill(self, name: str, description: str, code: str) -> None:
        """保存技能定义"""
        file_path = STATIC_DIR / f"skill_{name}.md"
        content = f"""---
name: {name}
description: {description}
type: skill
created_at: {time.time()}
---

# {name}
{description}

## 代码
```
{code}
```

## 使用场景
自动推断使用此技能
"""
        file_path.write_text(content, encoding="utf-8")

    def get_skills(self) -> list[dict]:
        """获取所有技能"""
        skills = []
        for f in STATIC_DIR.glob("skill_*.md"):
            try:
                text = f.read_text(encoding="utf-8")
                match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
                if match:
                    header = match.group(1)
                    name_match = re.search(r"name:\s*(.+)", header)
                    desc_match = re.search(r"description:\s*(.+)", header)
                    if name_match:
                        skills.append({
                            "name": name_match.group(1),
                            "description": desc_match.group(1) if desc_match else "",
                            "file": str(f),
                        })
            except Exception:
                continue
        return skills


class SessionMemory:
    """会话记忆 - 动态记忆

    管理当前会话的上下文，追踪任务、决策、待办事项
    """

    def __init__(self, thread_id: str = "default"):
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        self.thread_id = thread_id
        self.context_file = SESSION_DIR / f"session_{thread_id}.json"
        self.context: SessionContext = self._load()

    def _load(self) -> SessionContext:
        """加载会话上下文"""
        if self.context_file.exists():
            try:
                data = json.loads(self.context_file.read_text())
                return SessionContext(**data)
            except Exception:
                pass
        return SessionContext(thread_id=self.thread_id)

    def _save(self) -> None:
        """保存会话上下文"""
        data = {
            "thread_id": self.context.thread_id,
            "created_at": self.context.created_at,
            "task_id": self.context.task_id,
            "task_description": self.context.task_description,
            "recent_decisions": self.context.recent_decisions,
            "pending_items": self.context.pending_items,
            "completed_items": self.context.completed_items,
            "session_summary": self.context.session_summary,
        }
        self.context_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def set_task(self, task_id: str, description: str) -> None:
        """设置当前任务"""
        self.context.task_id = task_id
        self.context.task_description = description
        self._save()

    def add_decision(self, decision: str) -> None:
        """记录决策"""
        self.context.recent_decisions.append(f"[{time.strftime('%H:%M')}] {decision}")
        # 只保留最近10条
        if len(self.context.recent_decisions) > 10:
            self.context.recent_decisions = self.context.recent_decisions[-10:]
        self._save()

    def add_pending(self, item: str) -> None:
        """添加待办"""
        if item not in self.context.pending_items:
            self.context.pending_items.append(item)
        self._save()

    def complete_pending(self, item: str) -> None:
        """完成待办"""
        if item in self.context.pending_items:
            self.context.pending_items.remove(item)
            self.context.completed_items.append(item)
        self._save()

    def update_summary(self, summary: str) -> None:
        """更新会话摘要"""
        self.context.session_summary = summary
        self._save()

    def get_current_context(self) -> str:
        """获取当前会话上下文 - 动态记忆"""
        parts = []

        # 当前任务
        if self.context.task_description:
            parts.append(f"## 当前任务\n{self.context.task_description}")

        # 待办事项
        if self.context.pending_items:
            parts.append(f"## 待办事项\n" + "\n".join(f"- {p}" for p in self.context.pending_items))

        # 最近决策
        if self.context.recent_decisions:
            parts.append(f"## 最近决策\n" + "\n".join(self.context.recent_decisions[-3:]))

        if not parts:
            return ""

        return "\n\n## 当前会话状态\n" + "\n\n".join(parts)

    def clear(self) -> None:
        """清除会话"""
        self.context = SessionContext(thread_id=self.thread_id)
        if self.context_file.exists():
            self.context_file.unlink()


class MemoryLayer:
    """记忆层 - 三层架构

    1. 静态记忆 (Static) - 用户偏好、项目配置 → 注入 System Prompt
    2. 动态记忆 (Session) - 当前会话上下文 → 每次运行追加
    3. 事件记忆 (Episodic) - 过去发生的事 → 按需检索
    """

    def __init__(self, thread_id: str = "default"):
        self.thread_id = thread_id

        # 静态记忆
        self.static = StaticMemory()

        # 动态记忆
        self.session = SessionMemory(thread_id)

        # 事件记忆
        self.episodic = MemoryIndex()

        # 检索缓存
        self._cache: dict[str, str] = {}
        self._cache_time: float = 0
        self._cache_ttl = 180  # 3分钟

    def should_retrieve_episodic(self, query: str) -> bool:
        """判断是否需要检索事件记忆

        触发条件:
        - 查询长度 > 15
        - 不是命令
        - 可能是复杂任务或引用过去
        """
        if not query or len(query) < 15:
            return False
        if query.startswith("/"):
            return False

        # 引用过去
        past_keywords = ["之前", "上次", "记得", "之前做", "以前"]
        if any(kw in query for kw in past_keywords):
            return True

        # 复杂任务
        if len(query) > 100:
            return True

        return False

    def _get_cached(self, query: str) -> Optional[str]:
        """获取缓存的记忆"""
        cache_key = hash(query)
        if cache_key in self._cache:
            if time.time() - self._cache_time < self._cache_ttl:
                return self._cache[cache_key]
        return None

    def _set_cached(self, query: str, result: str) -> None:
        """设置缓存"""
        self._cache[hash(query)] = result
        self._cache_time = time.time()

    def retrieve_episodic(self, query: str) -> str:
        """检索事件记忆（按需）"""
        # 检查缓存
        cached = self._get_cached(query)
        if cached is not None:
            return cached

        # 执行检索
        entries = self.episodic.search(query, limit=3)

        if not entries:
            result = ""
        else:
            parts = ["\n\n## 相关经验\n"]
            for e in entries:
                parts.append(f"### [{e.memory_type}] {e.name}")
                parts.append(e.content[:300])
                parts.append("")
            result = "\n".join(parts)

        # 更新缓存
        self._set_cached(query, result)

        return result

    def build_static_prompt(self) -> str:
        """构建静态记忆部分 - 注入 System Prompt"""
        parts = []

        preferences = self.static.get_preferences()
        if preferences:
            parts.append(preferences)

        project = self.static.get_project_knowledge()
        if project:
            parts.append(project)

        return "\n".join(parts)

    def build_session_context(self) -> str:
        """构建动态记忆部分 - 当前会话状态"""
        return self.session.get_current_context()

    def save_episodic(self, name: str, content: str, memory_type: str, description: str = "") -> str:
        """保存事件记忆"""
        entry = MemoryEntry(
            name=name,
            description=description,
            content=content,
            memory_type=memory_type,
            created_at=time.time(),
        )
        self.episodic.save_entry(entry)
        # 清除缓存
        self._cache.clear()
        return f"Saved episodic memory: {name}"

    def on_task_complete(self, task_id: str, result: str) -> None:
        """任务完成时 - 保存到事件记忆"""
        self.save_episodic(
            name=f"task_{task_id}_{int(time.time())}",
            content=result,
            memory_type="task_completion",
            description=f"Task {task_id} completed",
        )
        # 更新会话
        self.session.complete_pending(f"task:{task_id}")

    def on_decision(self, decision: str) -> None:
        """记录决策"""
        self.session.add_decision(decision)

    def save_preference(self, key: str, value: str) -> None:
        """保存用户偏好到静态记忆"""
        self.static.save_preference(key, value)

    def save_project_info(self, key: str, value: str) -> None:
        """保存项目知识到静态记忆"""
        self.static.save_project_knowledge(key, value)

    def list_all(self) -> dict:
        """列出所有记忆"""
        return {
            "static": {
                "preferences": self.static.get_preferences(),
                "project": self.static.get_project_knowledge(),
                "skills_count": len(self.static.get_skills()),
            },
            "session": {
                "task": self.session.context.task_description,
                "pending": self.session.context.pending_items,
                "decisions": len(self.session.context.recent_decisions),
            },
            "episodic": self.episodic.list_all(),
        }

    def consolidate(self) -> dict:
        """整合事件记忆"""
        # 合并相似记忆
        deleted = self.episodic._index  # 只保留最新20条
        by_type: dict[str, list] = {}
        for name, entry in self.episodic._index.items():
            if entry.memory_type not in by_type:
                by_type[entry.memory_type] = []
            by_type[entry.memory_type].append(entry)

        for mem_type, entries in by_type.items():
            if len(entries) > 20:
                entries.sort(key=lambda e: e.last_accessed, reverse=True)
                for entry in entries[20:]:
                    safe_name = entry.name.replace(" ", "-").lower()
                    file_path = MEMORY_DIR / f"{safe_name}.md"
                    if file_path.exists():
                        file_path.unlink()
                        del self.episodic._index[entry.name]

        return {
            "action": "consolidate",
            "remaining": len(self.episodic._index),
        }


# 全局实例
_memory_layer: Optional[MemoryLayer] = None


def get_memory_layer(thread_id: str = "default") -> MemoryLayer:
    """获取全局记忆层"""
    global _memory_layer
    if _memory_layer is None:
        _memory_layer = MemoryLayer(thread_id)
    return _memory_layer


def reset_memory_layer() -> None:
    """重置记忆层"""
    global _memory_layer
    _memory_layer = None