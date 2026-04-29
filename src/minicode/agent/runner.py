"""Agent Runner - 使用 SessionManager 处理压缩、记忆、反思"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import AsyncIterator, Optional

from langchain_core.messages import HumanMessage

from minicode.agent.graph import create_agent_graph
from minicode.agent.state import AgentState
from minicode.agent.session import (
    SessionManager,
    SessionConfig,
    ContextOverflowError,
    get_session_manager,
    reset_session_manager,
)
from minicode.services.checkpoint import CheckpointManager
from minicode.tools.hook_tools import get_hook_manager


class AgentRunner:
    """Agent 运行器 - 使用 SessionManager 处理上下文管理"""

    def __init__(
        self,
        model_provider: str = "anthropic",
        model_name: str = "claude-sonnet-4-7",
        use_checkpoint: bool = False,
        db_path: Optional[str] = None,
        workdir: Optional[Path] = None,
        session_config: Optional[SessionConfig] = None,
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self.use_checkpoint = use_checkpoint
        self.db_path = db_path
        self.workdir = workdir or Path.cwd()

        # 会话管理器
        self.session = SessionManager(session_config)

        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(
            use_sqlite=bool(db_path),
            db_path=db_path,
        )

        # Agent Graph
        self.graph = create_agent_graph(
            model_provider=model_provider,
            model_name=model_name,
            use_checkpoint=use_checkpoint,
        )

        # SessionStart hooks
        hook_manager = get_hook_manager()
        hook_result = hook_manager.run_hooks("SessionStart", {
            "model_provider": model_provider,
            "model_name": model_name,
            "thread_id": "default",
        })
        if hook_result["messages"]:
            for msg in hook_result["messages"]:
                print(f"[SessionStart] {msg}")

    def _get_initial_state(self, messages: list) -> AgentState:
        """获取初始状态"""
        return {
            "messages": messages,
            "todo_items": [],
            "rounds_since_todo_update": 0,
            "execution_steps": [],
            "evaluation_score": 0.0,
            "tool_messages": [],
            "task_items": [],
            "pending_tasks": [],
            "has_compacted": False,
            "last_summary": "",
            "recent_files": [],
            "compact_requested": False,
            "compact_focus": None,
            "mode": "default",
            "permission_rules": [],
            "consecutive_denials": 0,
            "teammates": {},
            "completed_results": [],
            "inbox_notifications": [],
            "pending_requests": [],
            "pending_background_tasks": [],
            "completed_notifications": [],
            "worktree_events": [],
            "active_worktrees": [],
            "task_type": "",
            "matched_skill": None,
            "should_create_skill": False,
            "should_update_memory": False,
            "task_count": 0,
            "max_output_recovery_count": 3,
            "error_recovery_count": 0,
            "scheduled_notifications": [],
            "active_schedules": [],
        }

    async def run(self, messages: list, thread_id: str = "default") -> dict:
        """运行 Agent

        关键流程:
        1. preflight_check - 运行前检查上下文大小
        2. graph.invoke - 执行 graph
        3. after_run - 运行后处理（可能触发压缩/反思）
        """
        config = self.checkpoint_manager.get_session_config(thread_id)

        try:
            # Step 1: 运行前检查 - 确保上下文安全
            safe_messages = self.session.preflight_check(messages)

            # Step 2: 执行 Graph
            initial_state = self._get_initial_state(safe_messages)
            result = await self.graph.ainvoke(initial_state, config)

            # Step 3: 运行后处理
            post_result = self.session.after_run(
                result.get("messages", []),
                had_error=False
            )

            # 如果发生了压缩，更新结果
            if "compact" in post_result.get("actions", []):
                result["messages"] = post_result.get("messages", result.get("messages", []))

            return result

        except ContextOverflowError as e:
            # 上下文超限，尝试压缩后重试
            print(f"[Warning] Context overflow: {e}")
            compacted = self.session.compact(messages, aggressive=True)

            # 重试一次
            initial_state = self._get_initial_state(compacted)
            result = await self.graph.ainvoke(initial_state, config)

            post_result = self.session.after_run(result.get("messages", []), had_error=True)
            result["messages"] = post_result.get("messages", result.get("messages", []))

            return result

    async def stream(self, messages: list, thread_id: str = "default") -> AsyncIterator[str]:
        """流式运行 Agent"""
        config = self.checkpoint_manager.get_session_config(thread_id)

        # 运行前检查
        safe_messages = self.session.preflight_check(messages)
        initial_state = self._get_initial_state(safe_messages)

        async for event in self.graph.astream(initial_state, config):
            if isinstance(event, dict):
                if "messages" in event:
                    for msg in event["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            yield msg.content
                elif "tool_messages" in event:
                    for msg in event["tool_messages"]:
                        if hasattr(msg, "content"):
                            yield f"\n> {msg.content[:200]}"
            else:
                yield str(event)

    def get_session_summary(self) -> dict:
        """获取会话摘要"""
        return self.session.get_summary()

    def get_memory(self) -> list:
        """获取记忆列表"""
        return self.session.list_memory()

    def save_memory(self, name: str, description: str, mem_type: str, content: str) -> str:
        """保存记忆"""
        return self.session.save_memory(name, description, mem_type, content)

    def compact_now(self) -> dict:
        """手动触发压缩"""
        # 获取当前状态并压缩
        state = self.get_session_state()
        if state and "messages" in state.values:
            messages = state.values["messages"]
            compacted = self.session.compact(list(messages))
            return {
                "original_count": len(messages),
                "compacted_count": len(compacted),
                "compacted": compacted,
            }
        return {"error": "No session state"}

    def get_session_state(self, thread_id: str = "default") -> Optional[dict]:
        """获取会话状态"""
        config = self.checkpoint_manager.get_session_config(thread_id)
        return self.graph.get_state(config)

    def clear_session(self, thread_id: str = "default") -> None:
        """清除会话"""
        self.checkpoint_manager.clear_session(thread_id)
        self.session.reset()


async def run_interactive(
    model_provider: str = "anthropic",
    model_name: str = "claude-sonnet-4-7",
    thread_id: str = "default",
) -> None:
    """交互式运行"""
    # 重置全局会话管理器
    reset_session_manager()

    runner = AgentRunner(
        model_provider=model_provider,
        model_name=model_name,
        use_checkpoint=True,
    )

    print(f"MiniCode Interactive Agent")
    print(f"Model: {model_name}")
    print("Commands: /clear, /history, /state, /memory, /compact")
    print("-" * 50)

    messages = []

    while True:
        try:
            user_input = input(f"\033[36m{thread_id} >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        # 处理命令
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd in ("/q", "/quit", "/exit"):
                break
            elif cmd == "/clear":
                messages = []
                print("History cleared")
                continue
            elif cmd == "/history":
                summary = runner.get_session_summary()
                print(f"Turns: {summary['total_turns']}, "
                      f"Tasks: {summary['tasks_completed']}, "
                      f"Tools: {summary['tools_called']}")
                continue
            elif cmd == "/state":
                state = runner.get_session_state(thread_id)
                if state:
                    msgs = state.values.get("messages", [])
                    print(f"Session active: {len(msgs)} messages")
                else:
                    print("No active session")
                continue
            elif cmd == "/memory":
                mems = runner.get_memory()
                print(f"Memory entries: {len(mems)}")
                for m in mems[:5]:
                    print(f"  - {m.get('name', 'unnamed')}")
                continue
            elif cmd == "/compact":
                result = runner.compact_now()
                print(f"Compacted: {result.get('original_count', 0)} -> {result.get('compacted_count', 0)} messages")
                continue
            elif cmd == "/help":
                print("Commands: /clear, /history, /state, /memory, /compact, /q")
                continue

        messages.append(HumanMessage(content=user_input))

        print("\n[Processing...]\n")

        try:
            result = await runner.run(messages, thread_id)
            response_msgs = result.get("messages", [])

            for msg in response_msgs:
                if hasattr(msg, "content") and msg.content:
                    print(f"\033[32m{msg.content}\033[0m")
        except Exception as e:
            print(f"[Error]: {e}")

        print()


if __name__ == "__main__":
    asyncio.run(run_interactive())