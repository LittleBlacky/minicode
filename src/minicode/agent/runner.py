"""Agent Runner - 五层架构 (Input + Memory + SelfImprove + Graph + Output)"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage

from minicode.agent.graph import create_agent_graph
from minicode.agent.state import AgentState
from minicode.agent.session import (
    SessionManager,
    SessionConfig,
    get_session_manager,
    reset_session_manager,
)
from minicode.agent.memory import (
    MemoryLayer,
    get_memory_layer,
    reset_memory_layer,
)
from minicode.agent.self_improve import (
    SelfImprovementEngine,
    TaskRecord,
    get_self_improvement,
    reset_self_improvement,
)
from minicode.services.checkpoint import CheckpointManager
from minicode.tools.hook_tools import get_hook_manager


def _is_rate_limit_error(error: Exception) -> bool:
    """检查是否是速率限制错误"""
    error_str = str(error).lower()
    return "429" in error_str or "rate_limit" in error_str or "rate limit" in error_str


def _is_overflow_error(error: Exception) -> bool:
    """检查是否是上下文溢出错误"""
    error_str = str(error).lower()
    indicators = ["context", "token", "maximum", "too long", "length", "exceeds"]
    return any(i in error_str for i in indicators)


class AgentRunner:
    """Agent 运行器 - 五层架构

    Layer 1: Input Safety      - 运行前预检+压缩
    Layer 2: Memory Layer      - 三层记忆 (Static/Session/Episodic)
    Layer 3: Self-Improve      - 自我提升引擎 (自动触发)
    Layer 4: Graph Execution   - 轻量循环
    Layer 5: Output Protection - 保护过长输出

    自我提升触发时机:
    - 周期性: 每完成 N 个任务
    - 失败时: 任务失败立即分析
    - 模式检测: 重复模式出现 3+ 次
    - 空闲时: 超过 N 秒无活动
    - 退出时: 会话结束总结
    """

    def __init__(
        self,
        use_checkpoint: bool = False,
        db_path: Optional[str] = None,
        workdir: Optional[Path] = None,
        session_config: Optional[SessionConfig] = None,
        thread_id: str = "default",
    ):
        import os
        self.use_checkpoint = use_checkpoint
        self.db_path = db_path
        self.workdir = workdir or Path.cwd()
        self.thread_id = thread_id

        # 模型配置属性（供其他模块读取）
        self.model_provider = os.environ.get("MINICODE_PROVIDER") or os.environ.get("MINICODE_MODEL_PROVIDER") or "anthropic"
        self.model_name = os.environ.get("MINICODE_MODEL") or "claude-sonnet-4-7"

        # Layer 2: 记忆层
        self.memory = get_memory_layer(thread_id)

        # Layer 3: 自我提升引擎
        self.self_improve = get_self_improvement()

        # Layer 1: 会话管理器
        self.session = SessionManager(session_config)

        # 检查点
        self.checkpoint_manager = CheckpointManager(
            use_sqlite=bool(db_path),
            db_path=db_path,
        )

        # Layer 4: Agent Graph - 所有配置从环境变量读取
        self.graph = create_agent_graph(use_checkpoint=use_checkpoint)

        # Hooks
        hook_manager = get_hook_manager()
        hook_result = hook_manager.run_hooks("SessionStart", {
            "thread_id": thread_id,
        })
        if hook_result["messages"]:
            for msg in hook_result["messages"]:
                print(f"[SessionStart] {msg}")

    def _get_initial_state(
        self,
        messages: list,
        static_memory: str = "",
        session_context: str = "",
        episodic_memory: str = "",
    ) -> AgentState:
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
            # 记忆层
            "static_memory": static_memory,
            "session_context": session_context,
            "episodic_memory": episodic_memory,
        }

    def _check_self_improve_trigger(self, task_result: Optional[dict] = None) -> None:
        """检查并触发自我提升"""
        # 如果有任务结果，记录并检查触发
        if task_result:
            task_record = TaskRecord(
                task_id=task_result.get("id", "unknown"),
                description=task_result.get("description", ""),
                success=task_result.get("success", False),
                duration=task_result.get("duration", 0),
                error=task_result.get("error", ""),
                task_type=task_result.get("type", "general"),
            )
            trigger = self.self_improve.record_task(task_record)
            if trigger:
                analysis = self.self_improve.analyze(trigger)
                self._apply_self_improvement(analysis)

        # 检查空闲触发
        if self.self_improve.should_trigger_idle():
            trigger = self.self_improve.trigger_manual()
            analysis = self.self_improve.analyze(trigger)
            self._apply_self_improvement(analysis)

    def _apply_self_improvement(self, analysis: dict) -> None:
        """应用自我提升结果"""
        # 创建技能
        for skill in analysis.get("created_skills", []):
            print(f"[Self-Improve] {skill}")

        # 保存记忆
        for memory in analysis.get("saved_memories", []):
            print(f"[Self-Improve] {memory}")

        # 更新建议
        for suggestion in analysis.get("suggestions", []):
            print(f"[Self-Improve] 建议: {suggestion}")

    async def run(self, messages: list, thread_id: str = "default") -> dict:
        """运行 Agent - 五层架构"""
        config = self.checkpoint_manager.get_session_config(thread_id)

        # Layer 1: Input Safety
        safe_messages = self.session.preflight_check(messages)

        # Layer 2: Memory Layer
        static_memory = self.memory.build_static_prompt()
        session_context = self.memory.build_session_context()

        episodic_memory = ""
        if messages:
            query = messages[-1].content[:200] if hasattr(messages[-1], "content") else ""
            if self.memory.should_retrieve_episodic(query):
                episodic_memory = self.memory.retrieve_episodic(query)

        initial_state = self._get_initial_state(
            safe_messages,
            static_memory=static_memory,
            session_context=session_context,
            episodic_memory=episodic_memory,
        )

        # Layer 4: Graph Execution (with retry for rate limits)
        max_retries = 3
        retry_delay = 5  # seconds
        for attempt in range(max_retries):
            try:
                result = await self.graph.ainvoke(initial_state, config)

                # Layer 5: Output Protection
                all_messages = self.session.protect_output(list(result.get("messages", [])))
                result["messages"] = all_messages

                # Layer 3: 检查自我提升触发 (成功时)
                self._check_self_improve_trigger({
                    "id": f"task_{len(messages)}",
                    "description": messages[-1].content[:50] if messages else "",
                    "success": True,
                    "duration": 0,
                    "type": "general",
                })

                # 周期性检查
                post_result = self.session.after_run(all_messages)
                if "compact" in post_result.get("actions", []):
                    result["messages"] = post_result.get("messages", all_messages)

                return result

            except Exception as e:
                # Layer 3: 失败时触发自我提升
                self._check_self_improve_trigger({
                    "id": f"task_{len(messages)}",
                    "description": messages[-1].content[:50] if messages else "",
                    "success": False,
                    "error": str(e),
                    "duration": 0,
                    "type": "general",
                })

                # Rate limit handling: retry with backoff
                if _is_rate_limit_error(e):
                    if attempt < max_retries - 1:
                        print(f"[Warning] Rate limit hit, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        print("[Error] Rate limit exceeded, all retries failed")
                        raise

                if _is_overflow_error(e):
                    print(f"[Warning] Context overflow, attempting recovery...")
                    state = self.graph.get_state(config)
                    current = list(state.values["messages"]) if state and "messages" in state.values else safe_messages
                    compacted = self.session.compact(current, aggressive=True)
                    retry_state = self._get_initial_state(compacted, static_memory, session_context)
                    result = await self.graph.ainvoke(retry_state, config)
                    result["messages"] = self.session.protect_output(list(result.get("messages", [])))
                    return result

                raise

    async def run_with_task(self, messages: list, task_id: str, task_type: str = "general") -> dict:
        """运行任务 (带追踪)"""
        import time
        start = time.time()

        result = await self.run(messages)

        duration = time.time() - start

        # 记录任务结果
        task_record = TaskRecord(
            task_id=task_id,
            description=result.get("summary", "")[:100],
            success=len(result.get("messages", [])) > 0,
            duration=duration,
            task_type=task_type,
        )
        trigger = self.self_improve.record_task(task_record)
        if trigger:
            analysis = self.self_improve.analyze(trigger)
            self._apply_self_improvement(analysis)

        return result

    def trigger_dream(self) -> dict:
        """手动触发梦境整合"""
        trigger = self.self_improve.trigger_manual()
        return self.self_improve.analyze(trigger)

    def on_exit(self) -> dict:
        """退出时触发"""
        trigger = self.self_improve.trigger_exit()
        return self.self_improve.analyze(trigger)

    # ========== API ==========

    def get_stats(self) -> dict:
        """获取统计"""
        return {
            "session": self.session.get_summary(),
            "memory": self.memory.list_all(),
            "self_improve": self.self_improve.get_stats(),
        }

    def get_memory(self) -> dict:
        return self.memory.list_all()

    def save_preference(self, key: str, value: str) -> None:
        """保存用户偏好到静态记忆"""
        self.memory.save_preference(key, value)

    def save_project_knowledge(self, key: str, value: str) -> None:
        """保存项目知识到静态记忆"""
        self.memory.save_project_info(key, value)

    def clear_session(self, thread_id: str = "default") -> None:
        self.checkpoint_manager.clear_session(thread_id)
        self.session.reset()
        self.memory.session.clear()

    def reload_config(self) -> None:
        """热重载配置 - 清除模型缓存，下次访问时会从 config.json 重新加载"""
        from minicode.agent.graph import AgentGraphBuilder
        builder = AgentGraphBuilder.get_instance()
        if builder:
            builder.reset()


async def run_interactive(thread_id: str = "default") -> None:
    """交互式运行 - 配置从环境变量读取"""
    import os
    reset_session_manager()
    reset_memory_layer()
    reset_self_improvement()

    runner = AgentRunner(
        use_checkpoint=True,
        thread_id=thread_id,
    )

    model_name = os.environ.get("MINICODE_MODEL") or "claude-sonnet-4-7"
    print(f"MiniCode Agent (5 Layers)")
    print(f"Model: {model_name}")
    print("Commands: /clear, /stats, /dream, /memory, /quit")
    print("-" * 50)

    messages = []

    try:
        while True:
            user_input = input(f"\033[36m{thread_id} >> \033[0m")

            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()

                if cmd in ("/q", "/quit", "/exit"):
                    # 退出时触发自我提升
                    result = runner.on_exit()
                    if result.get("patterns") or result.get("suggestions"):
                        print("\n[退出总结]")
                        for p in result.get("patterns", []):
                            print(f"  - {p}")
                        for s in result.get("suggestions", []):
                            print(f"  建议: {s}")
                    break

                elif cmd == "/clear":
                    messages = []
                    print("History cleared")

                elif cmd == "/stats":
                    stats = runner.get_stats()
                    print(f"Tasks: {stats['self_improve']['total_tasks']}, "
                          f"Success: {stats['self_improve']['success_count']}, "
                          f"Triggers: {stats['self_improve']['improvements_triggered']}")

                elif cmd == "/dream":
                    result = runner.trigger_dream()
                    print(f"Dream triggered: {len(result.get('patterns', []))} patterns, "
                          f"{len(result.get('created_skills', []))} skills created")

                elif cmd == "/memory":
                    mem = runner.get_memory()
                    print(f"Static skills: {mem['static']['skills_count']}")
                    print(f"Session: {len(mem['session']['pending'])} pending items")
                    print(f"Episodic: {len(mem['episodic'])} entries")

                elif cmd == "/help":
                    print("Commands: /clear, /stats, /dream, /memory, /quit")

                continue

            messages.append(HumanMessage(content=user_input))
            print("\n[Processing...]\n")

            try:
                result = await runner.run(messages, thread_id)
                for msg in result.get("messages", []):
                    if hasattr(msg, "content") and msg.content:
                        print(f"\033[32m{msg.content}\033[0m\n")
            except Exception as e:
                print(f"[Error]: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted. Running exit triggers...")
        runner.on_exit()


if __name__ == "__main__":
    asyncio.run(run_interactive())