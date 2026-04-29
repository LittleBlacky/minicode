"""Session Manager - 处理上下文压缩、记忆和反思"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from minicode.tools.memory_tools import MemoryManager
from minicode.tools.compact_tools import compact_messages, should_compact, get_context_size


WORKDIR = Path.cwd()
STORAGE_DIR = WORKDIR / ".mini-agent-cli"

# 上下文限制配置
DEFAULT_CONTEXT_LIMIT = 50000  # 字符数限制
LLM_MAX_TOKENS = 200000       # LLM 最大上下文（Claude 200K）


@dataclass
class SessionConfig:
    """会话配置"""
    compact_threshold: int = 50      # 消息数量阈值，超过则压缩
    compact_keep_recent: int = 5     # 压缩时保留最近N条消息
    memory_on_task_complete: bool = True  # 任务完成时保存记忆
    reflect_on_idle: bool = True     # 空闲时运行反思
    reflect_interval: int = 10      # 每N轮对话运行一次反思
    context_limit: int = DEFAULT_CONTEXT_LIMIT  # 上下文大小限制
    max_retry_on_compact: int = 2   # 压缩后最大重试次数


@dataclass
class SessionMetrics:
    """会话指标"""
    total_turns: int = 0
    total_tools_called: int = 0
    tasks_completed: int = 0
    context_size: int = 0
    last_compact_turn: int = 0
    last_reflect_turn: int = 0
    session_start: float = field(default_factory=time.time)
    compact_count: int = 0  # 压缩次数统计


class ContextOverflowError(Exception):
    """上下文超限异常"""
    def __init__(self, current_size: int, limit: int):
        self.current_size = current_size
        self.limit = limit
        super().__init__(f"Context size {current_size} exceeds limit {limit}")


class SessionManager:
    """会话管理器 - 负责压缩、记忆、反思的后台处理"""

    def __init__(self, config: Optional[SessionConfig] = None):
        self.config = config or SessionConfig()
        self.metrics = SessionMetrics()
        self.memory_manager = MemoryManager()
        self._compact_retry_count = 0

        # 任务历史用于反思
        self.task_history: list[dict] = []
        self.completed_tasks: list[dict] = []

    # ========== 上下文大小检查 ==========

    def check_context_size(self, messages: list) -> tuple[bool, int]:
        """检查上下文大小是否超限

        Returns:
            (is_safe, current_size): 是否安全，当前大小
        """
        current_size = get_context_size(messages)
        is_safe = current_size < self.config.context_limit
        return is_safe, current_size

    def preflight_check(self, messages: list) -> list:
        """运行前检查，返回可以安全发送的消息列表

        如果上下文超限，自动压缩后返回
        如果压缩后仍然超限，抛出异常
        """
        is_safe, current_size = self.check_context_size(messages)

        if is_safe:
            return messages

        # 需要压缩
        compacted = self.compact(messages)

        # 再次检查
        is_safe_after, size_after = self.check_context_size(compacted)

        if not is_safe_after:
            # 压缩后仍然超限，尝试更激进的压缩
            compacted = self.compact(messages, aggressive=True)

            is_safe_final, size_final = self.check_context_size(compacted)
            if not is_safe_final:
                raise ContextOverflowError(size_final, self.config.context_limit)

        return compacted

    # ========== 上下文压缩 ==========

    def check_should_compact(self, messages: list) -> bool:
        """检查是否需要压缩"""
        turn_number = self.metrics.total_turns
        since_last = turn_number - self.metrics.last_compact_turn

        # 条件1：超过压缩阈值
        if len(messages) >= self.config.compact_threshold:
            return True

        # 条件2：超过15轮没压缩
        if since_last >= 15:
            return True

        # 条件3：上下文接近限制
        is_safe, size = self.check_context_size(messages)
        if not is_safe:
            return True

        return False

    def compact(self, messages: list, aggressive: bool = False) -> list:
        """执行上下文压缩

        Args:
            messages: 消息列表
            aggressive: 是否激进压缩（保留更少消息）
        """
        keep = 3 if aggressive else self.config.compact_keep_recent
        compacted = compact_messages(messages, keep_recent=keep)

        # 记录指标
        self.metrics.last_compact_turn = self.metrics.total_turns
        self.metrics.context_size = len(compacted)
        self.metrics.compact_count += 1

        return compacted

    def get_context_size(self, messages: list) -> int:
        """获取当前上下文大小"""
        return get_context_size(messages)

    # ========== 运行后处理 ==========

    def after_run(self, messages: list, had_error: bool = False) -> dict:
        """运行后处理 - 检查是否需要压缩、反思等"""
        self.increment_turn()

        result = {
            "actions": [],
            "context_size": self.get_context_size(messages),
        }

        # 检查是否需要压缩
        if self.check_should_compact(messages):
            result["actions"].append("compact")
            result["messages"] = self.compact(messages)
        else:
            result["messages"] = messages

        # 检查是否需要反思
        reflect_result = self.run_reflection()
        if reflect_result["action"] != "skip":
            result["actions"].append("reflect")
            result["reflection"] = reflect_result

        # 如果有错误，进行错误记录
        if had_error:
            self.record_error()

        return result

    # ========== 记忆系统 ==========

    def record_tool_call(self, tool_name: str, success: bool = True) -> None:
        """记录工具调用"""
        self.metrics.total_tools_called += 1

    def record_task(self, task: dict) -> None:
        """记录任务"""
        self.task_history.append({
            **task,
            "timestamp": time.time(),
        })

        if task.get("status") == "completed":
            self.completed_tasks.append({**task, "timestamp": time.time()})
            self.metrics.tasks_completed += 1

            # 任务完成时自动保存记忆
            if self.config.memory_on_task_complete:
                self._auto_save_memory(task)

    def _auto_save_memory(self, task: dict) -> None:
        """自动保存任务记忆"""
        subject = task.get("subject", "Unknown task")
        description = task.get("description", "")
        status = task.get("status", "")

        content = f"""## 任务完成记录

- **任务**: {subject}
- **状态**: {status}
- **完成时间**: {time.strftime('%Y-%m-%d %H:%M')}

### 描述
{description}
"""

        self.memory_manager.save(
            name=f"task_{int(time.time())}",
            description=subject,
            mem_type="project",
            content=content,
        )

    def save_memory(self, name: str, description: str, mem_type: str, content: str) -> str:
        """保存记忆"""
        return self.memory_manager.save(name, description, mem_type, content)

    def list_memory(self) -> list:
        """列出所有记忆"""
        return self.memory_manager.list_all()

    # ========== 自我反思 ==========

    def check_should_reflect(self) -> bool:
        """检查是否应该运行反思"""
        if not self.config.reflect_on_idle:
            return False

        turn_number = self.metrics.total_turns
        since_last = turn_number - self.metrics.last_reflect_turn

        return since_last >= self.config.reflect_interval

    def run_reflection(self) -> dict:
        """运行自我反思"""
        if not self.check_should_reflect():
            return {"action": "skip", "reason": "not_due"}

        self.metrics.last_reflect_turn = self.metrics.total_turns

        # 分析任务历史
        analysis = self._analyze_patterns()

        # 检查是否应该创建技能
        should_create_skill, pattern = self._should_create_skill()

        return {
            "action": "reflect",
            "patterns": analysis,
            "should_create_skill": should_create_skill,
            "skill_pattern": pattern,
            "metrics": self.get_summary(),
        }

    def _analyze_patterns(self) -> list[str]:
        """分析任务模式"""
        patterns = []

        if not self.task_history:
            return ["暂无任务历史"]

        patterns.append(f"完成 {len(self.task_history)} 个任务")

        return patterns

    def _should_create_skill(self) -> tuple[bool, Optional[str]]:
        """检查是否应该创建技能"""
        task_types = {}
        for task in self.task_history:
            task_type = task.get("type", "unknown")
            task_types[task_type] = task_types.get(task_type, 0) + 1

        for task_type, count in task_types.items():
            if count >= 3:
                return True, task_type

        return False, None

    def record_error(self) -> None:
        """记录错误（用于反思）"""
        pass

    # ========== 会话统计 ==========

    def increment_turn(self) -> None:
        """增加对话轮次"""
        self.metrics.total_turns += 1

    def get_summary(self) -> dict:
        """获取会话摘要"""
        return {
            "total_turns": self.metrics.total_turns,
            "tasks_completed": self.metrics.tasks_completed,
            "tools_called": self.metrics.total_tools_called,
            "context_size": self.metrics.context_size,
            "compact_count": self.metrics.compact_count,
            "session_duration": int(time.time() - self.metrics.session_start),
        }

    def reset(self) -> None:
        """重置会话"""
        self.metrics = SessionMetrics()
        self.task_history.clear()
        self.completed_tasks.clear()


# 全局实例
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """获取全局会话管理器"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def reset_session_manager() -> None:
    """重置全局会话管理器"""
    global _session_manager
    _session_manager = None