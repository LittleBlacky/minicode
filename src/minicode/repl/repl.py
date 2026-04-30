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
        "/teammates": "列出所有队友",
        "/spawn": "召唤新队友 (用法: /spawn <name> <role> <task>)",
        "/send": "发送消息给队友 (用法: /send <name> <message>)",
        "/inbox": "查看收件箱",
        "/pool": "管理 Agent 池 (list/run/stop)",

        # 系统功能
        "/cron": "查看定时任务",
        "/hooks": "查看钩子列表",
        "/compact": "压缩对话历史",
        "/stats": "查看统计信息",
        "/mcp": "管理 MCP 服务器 (list/add/remove)",
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

    def do_team(self) -> None:
        """Handle team commands."""
        from minicode.tools.team_tools import get_teammate_manager, get_message_bus

        mgr = get_teammate_manager()
        bus = get_message_bus()

        teammates = mgr.list_teammates()
        print("\n[团队]")

        if not teammates:
            print("  暂无队友")
            print("\n  使用 /spawn <name> <role> <task> 召唤新队友")
        else:
            print(f"  队友数量: {len(teammates)}")
            for tm in teammates:
                status = "空闲" if tm.get("status") == "idle" else "工作中"
                print(f"  - {tm['name']} ({tm['role']}) [{status}]")

        # 显示收件箱未读消息
        inbox = bus._load_inbox()
        unread = sum(len(msgs) for msgs in inbox.values())
        if unread > 0:
            print(f"\n  未读消息: {unread} 条 (使用 /inbox 查看)")

        print()

    def do_teammates(self) -> None:
        """List all teammates."""
        from minicode.tools.team_tools import get_teammate_manager

        mgr = get_teammate_manager()
        teammates = mgr.list_teammates()

        if not teammates:
            print("\n[队友] 暂无队友")
            print("  使用 /spawn <name> <role> <task> 召唤新队友")
        else:
            print(f"\n[队友] {len(teammates)} 个")
            for tm in teammates:
                print(f"  - {tm['name']}: {tm['role']} (状态: {tm.get('status', 'unknown')})")
        print()

    def do_spawn(self, args: str = "") -> None:
        """Spawn a new teammate."""
        parts = args.split(maxsplit=2)

        if len(parts) < 3:
            print("\n[召唤] 用法: /spawn <name> <role> <task>")
            print("  示例: /spawn coder 前端开发 实现用户登录页面")
            print("  示例: /spawn reviewer 代码审查 审查登录模块代码")
            print()
            return

        name, role, task = parts[0], parts[1], parts[2]

        from minicode.tools.team_tools import get_teammate_manager
        mgr = get_teammate_manager()

        tm = mgr.spawn(name, role, task)
        print(f"\n[召唤] 成功创建队友 {name}")
        print(f"  角色: {role}")
        print(f"  任务: {task}")
        print(f"  状态: {tm['status']}")
        print("\n使用 /send <name> <message> 发送消息给队友")
        print()

    def do_send(self, args: str = "") -> None:
        """Send message to a teammate."""
        parts = args.split(maxsplit=1)

        if len(parts) < 2:
            print("\n[发送] 用法: /send <name> <message>")
            print("  示例: /send coder 开始实现登录功能")
            print()
            return

        name, message = parts[0], parts[1]

        from minicode.tools.team_tools import get_message_bus
        bus = get_message_bus()

        result = bus.send(name, message)
        print(f"\n{result}")
        print()

    def do_inbox(self) -> None:
        """Read inbox messages."""
        from minicode.tools.team_tools import get_message_bus

        bus = get_message_bus()
        messages = bus.read_inbox("main")

        print(f"\n{messages}")
        print()

    def do_pool(self, args: str = "") -> None:
        """Manage agent pool."""
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else ""

        from minicode.agent.subagent import SubAgentPool

        # 获取或创建全局池
        global _subagent_pool
        if '_subagent_pool' not in globals():
            globals()['_subagent_pool'] = SubAgentPool(max_agents=5)

        pool = globals()['_subagent_pool']

        if action == "" or action == "list":
            print(f"\n[Agent 池]")
            print(f"  最大代理数: {pool.max_agents}")
            print(f"  当前代理数: {len(pool.agents)}")
            if pool.agents:
                for agent in pool.agents:
                    print(f"  - {agent.name}: {agent.role}")
            print("\n  用法:")
            print("    /pool list      - 列出代理")
            print("    /pool run <name> <role> <task>")
            print("                  - 创建并运行子代理")
            print("    /pool clear     - 清空代理池")
            print()
            return

        if action == "run":
            if len(parts) < 2:
                print("\n[池] 用法: /pool run <name> <role> <task>")
                print("  示例: /pool run coder 前端 实现登录页面")
                print()
                return

            sub_parts = parts[1].split(maxsplit=2)
            if len(sub_parts) < 3:
                print("\n[池] 参数不足: /pool run <name> <role> <task>")
                print()
                return

            name, role, task = sub_parts[0], sub_parts[1], sub_parts[2]

            print(f"\n[池] 创建子代理 {name}...")
            agent = pool.create(name, role, task)

            # 异步运行
            async def run_agent():
                result = await agent.run()
                print(f"\n[池] {name} 完成:")
                print(f"  {result[:200]}..." if len(result) > 200 else f"  {result}")

            import asyncio
            try:
                asyncio.get_event_loop().run_until_complete(run_agent())
            except RuntimeError:
                asyncio.new_event_loop().run_until_complete(run_agent())
            print()
            return

        if action == "clear":
            pool.clear()
            print("\n[池] 已清空")
            print()
            return

        print(f"\n[池] 未知命令: {action}")
        print("  用法: /pool [list|run|clear]")
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

    def do_mcp(self, args: str = "") -> None:
        """Handle MCP commands."""
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        from minicode.tools.mcp_tools import get_mcp_client, mcp_connect, mcp_disconnect

        client = get_mcp_client()

        if action == "" or action == "list":
            # 列出所有服务器
            servers = client.list_servers()
            if not servers:
                print("\n[MCP] 暂无已配置的服务器")
                print("  用法: /mcp add <name> <transport> <command> [args]")
                print("  示例: /mcp add filesystem stdio npx -y @modelcontextprotocol/server-filesystem /path")
            else:
                print(f"\n[MCP] 已配置服务器 ({len(servers)}):")
                for srv in servers:
                    print(f"  - {srv['name']} ({srv['config'].get('transport', 'unknown')})")
                tools = client.get_tools()
                if tools:
                    print(f"\n[MCP] 可用工具 ({len(tools)}):")
                    for t in tools[:10]:
                        print(f"  - {t.name}")
                    if len(tools) > 10:
                        print(f"  ... 还有 {len(tools) - 10} 个")
            print()
            return

        if action == "add":
            # 添加服务器
            # 格式: /mcp add <name> <transport> <command> [args] [env]
            # 示例: /mcp add filesystem stdio npx -y @modelcontextprotocol/server-filesystem C:/Temp
            if not rest:
                print("\n[MCP] 用法: /mcp add <name> <transport> <command> [cmd_args]")
                print("  示例: /mcp add filesystem stdio npx -y @modelcontextprotocol/server-filesystem C:/Temp")
                print("  示例: /mcp add github stdio npx -y @modelcontextprotocol/server-github")
                print()
                return

            # 解析参数
            sub_parts = rest.split()
            if len(sub_parts) < 3:
                print("\n[MCP] 参数不足，需要: <name> <transport> <command> [cmd_args]")
                print("  示例: /mcp add filesystem stdio npx -y @modelcontextprotocol/server-filesystem C:/Temp")
                print()
                return

            name = sub_parts[0]
            transport = sub_parts[1]
            command = sub_parts[2]
            cmd_args = " ".join(sub_parts[3:]) if len(sub_parts) > 3 else ""

            print(f"\n[MCP] 连接 {name}...")
            result = mcp_connect.invoke({
                "server_name": name,
                "transport": transport,
                "command": command,
                "cmd_args": cmd_args,
                "env": "{}",
                "url": "",
            })
            print(f"  {result}")

            # 刷新并显示工具
            client.refresh()
            tools = client.get_tools()
            if tools:
                print(f"\n[MCP] 已连接！可用工具 ({len(tools)}):")
                for t in tools[:10]:
                    print(f"  - {t.name}")
                if len(tools) > 10:
                    print(f"  ... 还有 {len(tools) - 10} 个")
            print()
            return

        if action == "remove" or action == "disconnect":
            if not rest:
                print("\n[MCP] 用法: /mcp remove <name>")
                print("  示例: /mcp remove github")
                print()
                return

            print(f"\n[MCP] 断开 {rest}...")
            result = mcp_disconnect.invoke({"server_name": rest})
            print(f"  {result}")
            print()
            return

        if action == "refresh":
            print("\n[MCP] 刷新工具列表...")
            count = client.refresh()
            print(f"  已刷新，{count} 个工具可用")
            print()
            return

        if action == "help":
            print("\n[MCP] 可用命令:")
            print("  /mcp list        - 列出已配置的服务器")
            print("  /mcp add <name> <transport> <command> [cmd_args]")
            print("                  - 添加并连接 MCP 服务器")
            print("  /mcp remove <name> - 断开并移除服务器")
            print("  /mcp refresh     - 刷新工具列表")
            print("\n示例:")
            print("  /mcp add filesystem stdio npx -y @modelcontextprotocol/server-filesystem C:/Temp")
            print("  /mcp add github stdio npx -y @modelcontextprotocol/server-github")
            print()
            return

        print(f"\n[MCP] 未知命令: {action}")
        print("  用法: /mcp [list|add|remove|refresh|help]")
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
            from minicode.tools.team_tools import get_teammate_manager
            tm = get_teammate_manager()
            members = tm.list_teammates()
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

        if cmd.startswith("/mcp"):
            # 提取参数
            args = cmd[4:].strip()
            self.do_mcp(args)
            return True

        if cmd == "/team":
            self.do_team()
            return True

        if cmd == "/teammates":
            self.do_teammates()
            return True

        if cmd.startswith("/spawn"):
            args = cmd[7:].strip()
            self.do_spawn(args)
            return True

        if cmd.startswith("/send"):
            args = cmd[6:].strip()
            self.do_send(args)
            return True

        if cmd == "/inbox":
            self.do_inbox()
            return True

        if cmd.startswith("/pool"):
            args = cmd[5:].strip()
            self.do_pool(args)
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
