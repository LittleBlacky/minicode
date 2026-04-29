"""REPL interface for interactive mode - Enhanced version."""
import asyncio
import sys
from typing import Optional

from langchain_core.messages import HumanMessage


class REPL:
    """Interactive REPL for the agent."""

    COMMANDS = {
        # 基础命令
        "/help": "显示帮助信息",
        "/quit": "退出程序",
        "/exit": "退出程序",
        "/clear": "清屏",
        "/status": "查看Agent状态",

        # 工具与权限
        "/tools": "列出所有可用工具",
        "/permission": "管理权限 (list/allow/deny)",

        # 任务与待办
        "/tasks": "查看任务列表",
        "/todos": "查看待办事项",

        # 记忆与知识
        "/memory": "查看记忆系统",
        "/dream": "触发梦境整合",
        "/skills": "查看技能列表",
        "/preference": "保存偏好 (用法: /preference <key> <value>)",
        "/project": "保存项目知识 (用法: /project <key> <value>)",

        # 团队协作
        "/team": "查看团队状态",

        # 系统功能
        "/cron": "查看定时任务",
        "/hooks": "查看钩子列表",
        "/compact": "压缩对话历史",
        "/stats": "查看统计信息",
    }

    def __init__(self, runner):
        self.runner = runner
        self.history: list[str] = []
        self.running = True

    def print_welcome(self) -> None:
        """Print welcome message."""
        print("=" * 60)
        print("MiniCode - Claude-style coding agent")
        print("=" * 60)
        print("命令: /help 查看所有命令")
        print("=" * 60)

    def print_help(self) -> None:
        """Print help message."""
        print("\n可用命令:")
        for cmd, desc in self.COMMANDS.items():
            print(f"  {cmd:<15} {desc}")
        print()

    def print_status(self) -> None:
        """Print agent status."""
        stats = self.runner.get_stats()
        print("\n[状态]")
        print(f"  会话ID: {self.runner.thread_id}")
        print(f"  模型: {self.runner.model_name}")
        print(f"  提供商: {self.runner.model_provider}")
        print(f"  总Turns: {stats['session']['total_turns']}")
        print(f"  任务统计: {stats['self_improve']['total_tasks']} 个")
        print()

    def print_tools(self) -> None:
        """Print available tools."""
        from minicode.tools.registry import ALL_TOOLS
        print(f"\n[可用工具] {len(ALL_TOOLS)} 个")
        for i, tool in enumerate(ALL_TOOLS, 1):
            print(f"  {i:>2}. {tool.name}")
        print()

    def print_permission(self, cmd: str) -> None:
        """Print or modify permissions."""
        from minicode.tools.permission_tools import get_permission_mode
        mode = get_permission_mode()
        print("\n[权限模式]")
        print(f"  当前: {mode}")
        print("  用法: /permission list|allow|deny")
        print()

    def print_todos(self) -> None:
        """Print todo list."""
        mem = self.runner.get_memory()
        todos = mem['session'].get('pending', [])
        if not todos:
            print("\n[待办] 暂无待办事项")
        else:
            print(f"\n[待办] {len(todos)} 项")
            for todo in todos:
                print(f"  • {todo}")
        print()

    def print_skills(self) -> None:
        """Print skill list."""
        from minicode.tools.skill_tools import SkillManager
        sm = SkillManager()
        skills = sm.list()
        if not skills:
            print("\n[技能] 暂无注册技能")
        else:
            print(f"\n[技能] {len(skills)} 个")
            for skill in skills:
                print(f"  • {skill['name']}: {skill['description']}")
        print()

    def print_cron(self) -> None:
        """Print cron jobs."""
        from minicode.tools.cron_tools import CronScheduler
        cs = CronScheduler()
        jobs = cs.list()
        if not jobs:
            print("\n[Cron] 暂无定时任务")
        else:
            print(f"\n[Cron] {len(jobs)} 个任务")
            for job in jobs:
                print(f"  • {job['id']}: {job.get('task', 'N/A')}")
        print()

    def print_hooks(self) -> None:
        """Print hook list."""
        from minicode.tools.hook_tools import HookManager
        hm = HookManager()
        hooks = hm.list_hooks()
        if not hooks or all(not v for v in hooks.values()):
            print("\n[钩子] 暂无活跃钩子")
        else:
            print(f"\n[钩子]")
            for hook_type, hook_list in hooks.items():
                if hook_list:
                    print(f"  {hook_type}: {len(hook_list)} 个")
                    for h in hook_list:
                        print(f"    • {h}")
        print()

    def do_compact(self) -> None:
        """Compact conversation history."""
        from minicode.tools.compact_tools import compact_history
        # 保留最近3条消息
        result = compact_history.invoke({"keep_recent": 3})
        print(f"\n[压缩] {result}")
        print()

    def print_response(self, messages: list) -> None:
        """Print agent response."""
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                print(f"\n[Agent]\n{msg.content}\n")

    def print_stats(self) -> None:
        """Print agent stats."""
        stats = self.runner.get_stats()
        print("\n[统计]")
        print(f"  总任务: {stats['self_improve']['total_tasks']}")
        print(f"  成功: {stats['self_improve']['success_count']}")
        print(f"  失败: {stats['self_improve']['failure_count']}")
        print(f"  自我提升触发: {stats['self_improve']['improvements_triggered']}")
        print()

    def print_memory(self) -> None:
        """Print memory status."""
        mem = self.runner.get_memory()
        print("\n[记忆状态]")
        print(f"  静态技能: {mem['static']['skills_count']}")
        print(f"  动态待办: {len(mem['session']['pending'])}")
        print(f"  事件记忆: {len(mem['episodic'])}")
        print()

    async def handle_command(self, cmd: str) -> bool:
        """Handle special command. Returns True if handled."""
        cmd = cmd.strip()

        if cmd in ("/quit", "/exit"):
            # 退出时触发自我提升总结
            result = self.runner.on_exit()
            if result.get("patterns") or result.get("suggestions"):
                print("\n[退出总结]")
                for p in result.get("patterns", []):
                    print(f"  - {p}")
                for s in result.get("suggestions", []):
                    print(f"  建议: {s}")
            print("再见!")
            self.running = False
            return True

        if cmd == "/help":
            self.print_help()
            return True

        if cmd == "/clear":
            print("\033[2J\033[H")  # ANSI clear screen
            self.print_welcome()
            return True

        if cmd == "/stats":
            self.print_stats()
            return True

        if cmd == "/memory":
            self.print_memory()
            return True

        if cmd == "/dream":
            result = self.runner.trigger_dream()
            print("\n[梦境整合]")
            print(f"  模式: {len(result.get('patterns', []))}")
            print(f"  建议: {len(result.get('suggestions', []))}")
            print(f"  创建技能: {len(result.get('created_skills', []))}")
            return True

        if cmd == "/team":
            from minicode.tools.team_tools import TeamManager
            tm = TeamManager()
            members = tm.list()
            if members:
                print("\n[团队]")
                for m in members:
                    print(f"  {m.get('name', 'anonymous')} ({m.get('role', 'member')})")
            else:
                print("\n[团队] 暂无成员")
            return True

        if cmd == "/tasks":
            from minicode.tools.task_tools import TaskManager
            tm = TaskManager()
            tasks = tm.list_all()
            if not tasks:
                print("\n[任务] 暂无任务")
            else:
                print(f"\n[任务] {len(tasks)} 个")
                for t in tasks:
                    status = t.get('status', 'pending')
                    subject = t.get('subject', 'N/A')
                    print(f"  [{status}] {subject}")
            return True

        if cmd == "/todos":
            self.print_todos()
            return True

        if cmd == "/status":
            self.print_status()
            return True

        if cmd == "/tools":
            self.print_tools()
            return True

        if cmd == "/permission":
            self.print_permission(cmd)
            return True

        if cmd == "/skills":
            self.print_skills()
            return True

        if cmd == "/cron":
            self.print_cron()
            return True

        if cmd == "/hooks":
            self.print_hooks()
            return True

        if cmd == "/compact":
            self.do_compact()
            return True

        if cmd.startswith("/preference "):
            parts = cmd.split(maxsplit=2)
            if len(parts) >= 3:
                self.runner.save_preference(parts[1], parts[2])
                print(f"[偏好已保存] {parts[1]}")
            else:
                print("[用法] /preference <key> <value>")
            return True

        if cmd.startswith("/project "):
            parts = cmd.split(maxsplit=2)
            if len(parts) >= 3:
                self.runner.save_project_knowledge(parts[1], parts[2])
                print(f"[项目知识已保存] {parts[1]}")
            else:
                print("[用法] /project <key> <value>")
            return True

        return False

    async def run(self) -> None:
        """Run the REPL loop."""
        self.print_welcome()

        while self.running:
            try:
                user_input = input("\033[36m>>> \033[0m").strip()

                if not user_input:
                    continue

                # 处理命令
                if user_input.startswith("/"):
                    await self.handle_command(user_input)
                    continue

                self.history.append(user_input)

                # 执行任务
                print("\n[思考中...]\n")
                messages = [HumanMessage(content=user_input)]

                try:
                    result = await self.runner.run(messages)
                    self.print_response(result.get("messages", []))
                except Exception as e:
                    print(f"\n[错误] {e}")

            except KeyboardInterrupt:
                print("\n使用 /quit 退出")
            except EOFError:
                break
            except Exception as e:
                print(f"[错误] {e}")

    def stop(self) -> None:
        """Stop the REPL."""
        self.running = False


async def start_repl(runner) -> None:
    """Start the REPL."""
    repl = REPL(runner)
    await repl.run()
