"""Textual TUI interface for MiniCode."""
import asyncio
from datetime import datetime
from typing import Optional

from textual.app import App, ComposeResult
from textual.command import Command, CommandHelp, CommandList
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Header, Footer, Log, Input, RichLog, Static, Button, Label
from textual.binding import Binding
from textual import work

from langchain_core.messages import HumanMessage, AIMessage

from minicode.agent.runner import AgentRunner
from minicode.tui.ascii_art import ASCIIArt, CatAnimator


class MiniCodeTUI(App):
    """MiniCode Textual TUI Application."""

    CSS = """
    Screen {
        background: $surface;
    }

    #header {
        height: auto;
        background: $primary;
        color: $text;
        dock: top;
    }

    #ascii-cat {
        height: auto;
        width: 100%;
        background: $surface-darken-1;
        color: $accent;
        content-align: center middle;
        padding: 0;
        margin: 0;
        text-style: bold;
    }

    #message-area {
        height: 1fr;
        border: solid $border;
        margin: 1 2;
        padding: 1 2;
    }

    #input-container {
        height: auto;
        border-top: solid $border;
        margin: 0 2;
        padding: 1 2;
    }

    #input-row {
        height: auto;
        layout: horizontal;
    }

    #prompt-indicator {
        width: auto;
        padding: 1 1;
        color: $primary;
        text-style: bold;
    }

    Input {
        margin: 1 0;
    }

    #status-bar {
        height: auto;
        background: $surface-darken-1;
        dock: bottom;
        padding: 0 2;
    }

    #spinner {
        color: $accent;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
        Binding("ctrl+r", "recall", "Recall", show=True),
        Binding("up", "history_up", "History", show=False),
        Binding("down", "history_down", "History", show=False),
        Binding("ctrl+k", "toggle_command_palette", "Commands", show=True),
        Binding("ctrl+z", "suspend", "Suspend", show=True),
        Binding("ctrl+t", "new_tab", "New Tab", show=True),
        Binding("ctrl+a", "toggle_tool_panel", "Tools", show=True),
        Binding("ctrl+e", "toggle_mode", "Mode", show=True),
        Binding("f1", "help", "Help", show=True),
        Binding("f2", "status", "Status", show=True),
        Binding("f3", "history", "History", show=True),
        Binding("f4", "session", "Session", show=True),
    ]

    def __init__(self, runner: AgentRunner, **kwargs):
        super().__init__(**kwargs)
        self.runner = runner
        self.messages: list = []
        self.history: list[str] = []
        self.history_index: int = -1
        self.mode: str = "default"
        self.tool_panel_visible: bool = False
        self.start_time: datetime = datetime.now()
        self.command_count: int = 0
        self.cat_animator = CatAnimator()
        self.is_thinking: bool = False

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header(id="header")
        yield Static(ASCIIArt.get_cat_frame(0), id="ascii-cat")
        yield RichLog(id="message-area", highlight=True, markup=True)
        with Container(id="input-container"):
            with Horizontal(id="input-row"):
                yield Static(f"[{self.mode}]", id="prompt-indicator")
                yield Input(placeholder="Enter command... (type /help for commands)", id="command-input")
        yield Static("Ready. Press Ctrl+K for commands.", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the application."""
        self.title = "MiniCode"
        self.sub_title = "Claude-style coding agent"

        # Focus input
        self.query_one("#command-input", Input).focus()

        # Start cat animation
        self.animate_cat()

        # Show welcome message
        log = self.query_one("#message-area", RichLog)
        log.write_line("[bold green]Welcome to MiniCode![/bold green]")
        log.write_line("[dim]Type /help for available commands[/dim]")
        log.write_line("[dim]Press Ctrl+K for command palette[/dim]")
        log.write_line("")

    @work(exclusive=False)
    async def animate_cat(self) -> None:
        """Animate the cat ASCII art."""
        cat_widget = self.query_one("#ascii-cat", Static)
        while True:
            if not self.is_thinking:
                self.cat_animator.next_frame()
                cat_widget.update(self.cat_animator.get_art())
            await asyncio.sleep(0.5)

    async def run_command(self, command: str) -> None:
        """Execute a command."""
        log = self.query_one("#message-area", RichLog)
        status = self.query_one("#status-bar", Static)
        self.command_count += 1

        # Handle built-in commands
        if command.startswith("/"):
            await self.handle_command(command)
            status.update(f"Ready. {len(self.messages)} messages. Mode: {self.mode}")
            return

        # Add user message
        self.messages.append(HumanMessage(content=command))
        self.history.append(command)
        self.history_index = len(self.history)

        timestamp = datetime.now().strftime("%H:%M")
        log.write_line(f"[dim]{timestamp}[/dim] [bold blue]You:[/bold blue] {command}")
        status.update("[accent]Thinking...[/accent] [/]🐱")

        # Update cat to thinking state
        self.is_thinking = True
        self.cat_animator.set_state("thinking")
        cat_widget = self.query_one("#ascii-cat", Static)
        cat_widget.update(self.cat_animator.get_art())

        # Run agent
        try:
            result = await self.runner.run(self.messages)

            # Extract assistant response
            if "messages" in result:
                response = result["messages"][-1]
                content = response.content if hasattr(response, "content") else str(response)
                self.messages.append(AIMessage(content=content))
                timestamp = datetime.now().strftime("%H:%M")
                log.write_line(f"[dim]{timestamp}[/dim] [bold green]MiniCode:[/bold green] {content}")
            else:
                log.write_line(f"[bold red]Error:[/bold red] No response")

        except Exception as e:
            log.write_line(f"[bold red]Error:[/bold red] {e}")

        # Reset cat state
        self.is_thinking = False
        self.cat_animator.set_state("idle")
        cat_widget.update(self.cat_animator.get_art())
        status.update(f"Ready. {len(self.messages)} messages. Mode: {self.mode}")

    async def handle_command(self, command: str) -> None:
        """Handle built-in commands."""
        log = self.query_one("#message-area", RichLog)
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        commands = {
            "/help": self._cmd_help,
            "/clear": self._cmd_clear,
            "/history": self._cmd_history,
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
            "/status": self._cmd_status,
            "/config": self._cmd_config,
            "/session": self._cmd_session,
            "/memory": self._cmd_memory,
            "/skills": self._cmd_skills,
            "/context": self._cmd_context,
            "/stat": self._cmd_stat,
            "/compact": self._cmd_compact,
            "/retry": self._cmd_retry,
            "/mode": self._cmd_mode,
            "/model": self._cmd_model,
            "/provider": self._cmd_provider,
            "/permission": self._cmd_permission,
            "/theme": self._cmd_theme,
            "/log": self._cmd_log,
            "/export": self._cmd_export,
            "/import": self._cmd_import,
            "/tools": self._cmd_tools,
            "/env": self._cmd_env,
            "/variables": self._cmd_variables,
            "/keys": self._cmd_keys,
            "/permission-mode": self._cmd_permission_mode,
            "/set": self._cmd_set,
            "/get": self._cmd_get,
            "/echo": self._cmd_echo,
            "/sleep": self._cmd_sleep,
            "/time": self._cmd_time,
            "/uptime": self._cmd_uptime,
            "/cat": self._cmd_cat,
            "/purr": self._cmd_purr,
        }

        if cmd in commands:
            await commands[cmd](log, args)
        else:
            log.write_line(f"[yellow]Unknown command:[/yellow] {cmd}")
            log.write_line("[dim]Type /help for available commands[/dim]")

    async def _cmd_help(self, log: RichLog, args: str) -> None:
        """Show help."""
        help_text = """
[bold]Available Commands:[/bold]
[cyan]/help[/cyan]           - Show this help
[cyan]/clear[/cyan]          - Clear the screen
[cyan]/history[/cyan]        - Show command history
[cyan]/quit, /exit[/cyan]    - Exit the application
[cyan]/status[/cyan]         - Show status
[cyan]/config[/cyan]         - Show configuration
[cyan]/session[/cyan]        - Show session info
[cyan]/memory[/cyan]         - Show saved memories
[cyan]/skills[/cyan]         - List available skills
[cyan]/context[/cyan]        - Show context info
[cyan]/stat[/cyan]           - Show statistics
[cyan]/compact[/cyan]        - Compact context
[cyan]/retry[/cyan]          - Retry last command
[cyan]/mode[/cyan] <mode>    - Change mode (default/auto/plan)
[cyan]/model[/cyan] <model>  - Change model
[cyan]/provider[/cyan] <p>  - Change provider
[cyan]/tools[/cyan]          - List all tools
[cyan]/env[/cyan]            - Show environment
[cyan]/variables[/cyan]      - Show variables
[cyan]/keys[/cyan]           - Manage API keys
[cyan]/export[/cyan] <file>  - Export session
[cyan]/import[/cyan] <file>  - Import session
[cyan]/time[/cyan]           - Show current time
[cyan]/uptime[/cyan]         - Show uptime
[cyan]/cat[/cyan]            - Show cat animation
[cyan]/purr[/cyan]           - Make the cat purr

[bold]Keyboard Shortcuts:[/bold]
[cyan]Ctrl+K[/cyan]          - Command palette
[cyan]Ctrl+L[/cyan]          - Clear screen
[cyan]Ctrl+R[/cyan]          - Recall last command
[cyan]Ctrl+A[/cyan]          - Toggle tool panel
[cyan]Ctrl+E[/cyan]          - Toggle mode
[cyan]Up/Down[/cyan]         - Navigate history
[cyan]F1[/cyan]              - Help
[cyan]F2[/cyan]              - Status
[cyan]F3[/cyan]              - History
[cyan]F4[/cyan]              - Session
"""
        log.write_line(help_text)

    async def _cmd_clear(self, log: RichLog, args: str) -> None:
        """Clear screen."""
        log.clear()
        log.write_line("[dim]Screen cleared.[/dim]")

    async def _cmd_history(self, log: RichLog, args: str) -> None:
        """Show history."""
        if self.history:
            limit = int(args) if args.isdigit() else 10
            history_text = f"[bold]Command History (last {limit}):[/bold]\n"
            start = max(0, len(self.history) - limit)
            for i, h in enumerate(self.history[start:], start + 1):
                truncated = h[:50] + "..." if len(h) > 50 else h
                history_text += f"  {i}. {truncated}\n"
            log.write_line(history_text)
        else:
            log.write_line("[dim]No history yet.[/dim]")

    async def _cmd_quit(self, log: RichLog, args: str) -> None:
        """Quit."""
        log.write_line("[dim]Goodbye! 🐱[/dim]")
        self.exit()

    async def _cmd_status(self, log: RichLog, args: str) -> None:
        """Show status."""
        from minicode.services.config import get_config_manager
        config = get_config_manager()

        uptime = datetime.now() - self.start_time
        log.write_line("[bold]Status:[/bold]")
        log.write_line(f"  Status: [green]Ready[/green]")
        log.write_line(f"  Mode: {self.mode}")
        log.write_line(f"  Model: {config.get('model.provider')}/{config.get('model.model')}")
        log.write_line(f"  Messages: {len(self.messages)}")
        log.write_line(f"  Commands: {len(self.history)}")
        log.write_line(f"  Uptime: {uptime}")

    async def _cmd_config(self, log: RichLog, args: str) -> None:
        """Show config."""
        from minicode.services.config import get_config_manager
        config = get_config_manager()

        log.write_line("[bold]Configuration:[/bold]")
        log.write_line(f"  Model Provider: {config.get('model.provider')}")
        log.write_line(f"  Model: {config.get('model.model')}")
        log.write_line(f"  Permission Mode: {config.get('permissions.mode')}")
        log.write_line(f"  Auto Compact: {config.get('features.auto_compact')}")
        log.write_line(f"  Team Enabled: {config.get('features.team_enabled')}")
        log.write_line(f"  Skills Enabled: {config.get('features.skills_enabled')}")

    async def _cmd_session(self, log: RichLog, args: str) -> None:
        """Show session."""
        log.write_line("[bold]Session Info:[/bold]")
        log.write_line(f"  Messages: {len(self.messages)}")
        log.write_line(f"  History: {len(self.history)}")
        log.write_line(f"  Mode: {self.mode}")
        log.write_line(f"  Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    async def _cmd_memory(self, log: RichLog, args: str) -> None:
        """Show memories."""
        try:
            from minicode.tools.memory_tools import get_memories
            memories = get_memories()
            if memories:
                log.write_line("[bold]Saved Memories:[/bold]")
                for m in memories[:10]:
                    name = m.get('name', 'unnamed')
                    created = m.get('created', 'unknown')
                    log.write_line(f"  - {name} [dim]({created})[/dim]")
            else:
                log.write_line("[dim]No memories saved.[/dim]")
        except ImportError:
            log.write_line("[dim]Memory system not available.[/dim]")

    async def _cmd_skills(self, log: RichLog, args: str) -> None:
        """Show skills."""
        try:
            from minicode.tools.skill_tools import get_skills
            skills = get_skills()
            if skills:
                log.write_line("[bold]Available Skills:[/bold]")
                for s in skills:
                    log.write_line(f"  - {s}")
            else:
                log.write_line("[dim]No skills available.[/dim]")
        except ImportError:
            log.write_line("[dim]Skill system not available.[/dim]")

    async def _cmd_context(self, log: RichLog, args: str) -> None:
        """Show context."""
        log.write_line("[bold]Context Info:[/bold]")
        log.write_line(f"  Messages: {len(self.messages)}")
        log.write_line(f"  History: {len(self.history)}")
        log.write_line(f"  Mode: {self.mode}")

    async def _cmd_stat(self, log: RichLog, args: str) -> None:
        """Show statistics."""
        uptime = datetime.now() - self.start_time
        log.write_line("[bold]Statistics:[/bold]")
        log.write_line(f"  Total Messages: {len(self.messages)}")
        log.write_line(f"  Total Commands: {len(self.history)}")
        log.write_line(f"  Commands This Session: {self.command_count}")
        log.write_line(f"  Uptime: {uptime}")

    async def _cmd_compact(self, log: RichLog, args: str) -> None:
        """Compact context."""
        log.write_line("[dim]Context will be compacted before next response.[/dim]")

    async def _cmd_retry(self, log: RichLog, args: str) -> None:
        """Retry last command."""
        if self.history:
            last_cmd = self.history[-1]
            log.write_line(f"[dim]Retrying: {last_cmd}[/dim]")
            await self.run_command(last_cmd)
        else:
            log.write_line("[dim]No command to retry.[/dim]")

    async def _cmd_mode(self, log: RichLog, args: str) -> None:
        """Change mode."""
        if args in ("default", "auto", "plan"):
            self.mode = args
            prompt = self.query_one("#prompt-indicator", Static)
            prompt.update(f"[{self.mode}]")
            log.write_line(f"[green]Mode changed to: {self.mode}[/green]")
        else:
            log.write_line(f"[yellow]Invalid mode: {args}[/yellow]")
            log.write_line("[dim]Available modes: default, auto, plan[/dim]")

    async def _cmd_model(self, log: RichLog, args: str) -> None:
        """Change model."""
        if args:
            from minicode.services.config import get_config_manager
            config = get_config_manager()
            config.set("model.model", args)
            log.write_line(f"[green]Model changed to: {args}[/green]")
        else:
            log.write_line("[dim]Usage: /model <model-name>[/dim]")

    async def _cmd_provider(self, log: RichLog, args: str) -> None:
        """Change provider."""
        if args in ("anthropic", "openai"):
            from minicode.services.config import get_config_manager
            config = get_config_manager()
            config.set("model.provider", args)
            log.write_line(f"[green]Provider changed to: {args}[/green]")
        else:
            log.write_line("[yellow]Invalid provider[/yellow]")
            log.write_line("[dim]Available providers: anthropic, openai[/dim]")

    async def _cmd_tools(self, log: RichLog, args: str) -> None:
        """List tools."""
        try:
            from minicode.tools.registry import ALL_TOOLS
            log.write_line("[bold]Available Tools:[/bold]")
            for tool_name in sorted(ALL_TOOLS.keys()):
                log.write_line(f"  - {tool_name}")
        except ImportError:
            log.write_line("[dim]Tool registry not available.[/dim]")

    async def _cmd_env(self, log: RichLog, args: str) -> None:
        """Show environment."""
        import os
        log.write_line("[bold]Environment:[/bold]")
        log.write_line(f"  Python: {os.sys.version.split()[0]}")
        log.write_line(f"  CWD: {os.getcwd()}")
        log.write_line(f"  HOME: {os.path.expanduser('~')}")

    async def _cmd_variables(self, log: RichLog, args: str) -> None:
        """Show variables."""
        log.write_line("[bold]Session Variables:[/bold]")
        log.write_line(f"  mode: {self.mode}")
        log.write_line(f"  messages: {len(self.messages)}")
        log.write_line(f"  history: {len(self.history)}")
        log.write_line(f"  command_count: {self.command_count}")

    async def _cmd_keys(self, log: RichLog, args: str) -> None:
        """Manage API keys."""
        import os
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        log.write_line("[bold]API Keys:[/bold]")
        log.write_line(f"  ANTHROPIC_API_KEY: {'[green]Set[/green]' if has_anthropic else '[red]Not set[/red]'}")
        log.write_line(f"  OPENAI_API_KEY: {'[green]Set[/green]' if has_openai else '[red]Not set[/red]'}")

    async def _cmd_permission_mode(self, log: RichLog, args: str) -> None:
        """Set permission mode."""
        if args in ("default", "dangerously-unrestricted"):
            from minicode.services.config import get_config_manager
            config = get_config_manager()
            config.set("permissions.mode", args)
            log.write_line(f"[green]Permission mode changed to: {args}[/green]")
        else:
            log.write_line("[yellow]Invalid permission mode[/yellow]")
            log.write_line("[dim]Available modes: default, dangerously-unrestricted[/dim]")

    async def _cmd_set(self, log: RichLog, args: str) -> None:
        """Set config."""
        if "=" in args:
            key, value = args.split("=", 1)
            from minicode.services.config import get_config_manager
            config = get_config_manager()
            config.set(key.strip(), value.strip())
            log.write_line(f"[green]Set {key.strip()} = {value.strip()}[/green]")
        else:
            log.write_line("[yellow]Usage: /set <key>=<value>[/yellow]")

    async def _cmd_get(self, log: RichLog, args: str) -> None:
        """Get config."""
        if args:
            from minicode.services.config import get_config_manager
            config = get_config_manager()
            value = config.get(args)
            log.write_line(f"{args}: {value}")
        else:
            log.write_line("[yellow]Usage: /get <key>[/yellow]")

    async def _cmd_echo(self, log: RichLog, args: str) -> None:
        """Echo message."""
        log.write_line(args or "")

    async def _cmd_sleep(self, log: RichLog, args: str) -> None:
        """Sleep."""
        import asyncio
        seconds = int(args) if args.isdigit() else 1
        log.write_line(f"[dim]Sleeping for {seconds} seconds...[/dim]")
        await asyncio.sleep(seconds)
        log.write_line("[dim]Done.[/dim]")

    async def _cmd_time(self, log: RichLog, args: str) -> None:
        """Show current time."""
        now = datetime.now()
        log.write_line(f"[bold]Current Time:[/bold] {now.strftime('%Y-%m-%d %H:%M:%S')}")

    async def _cmd_uptime(self, log: RichLog, args: str) -> None:
        """Show uptime."""
        uptime = datetime.now() - self.start_time
        log.write_line(f"[bold]Uptime:[/bold] {uptime}")

    async def _cmd_export(self, log: RichLog, args: str) -> None:
        """Export session."""
        if args:
            import json
            data = {
                "messages": [str(m.content) for m in self.messages],
                "history": self.history,
                "mode": self.mode,
            }
            with open(args, "w") as f:
                json.dump(data, f, indent=2)
            log.write_line(f"[green]Session exported to: {args}[/green]")
        else:
            log.write_line("[yellow]Usage: /export <filepath>[/yellow]")

    async def _cmd_import(self, log: RichLog, args: str) -> None:
        """Import session."""
        if args:
            import json
            try:
                with open(args) as f:
                    data = json.load(f)
                self.messages = [HumanMessage(content=m) for m in data.get("messages", [])]
                self.history = data.get("history", [])
                self.mode = data.get("mode", "default")
                log.write_line(f"[green]Session imported from: {args}[/green]")
            except Exception as e:
                log.write_line(f"[red]Error: {e}[/red]")
        else:
            log.write_line("[yellow]Usage: /import <filepath>[/yellow]")

    async def _cmd_theme(self, log: RichLog, args: str) -> None:
        """Change theme."""
        log.write_line("[dim]Theme change not yet implemented[/dim]")

    async def _cmd_log(self, log: RichLog, args: str) -> None:
        """Show logs."""
        log.write_line("[dim]Log viewer not yet implemented[/dim]")

    async def _cmd_cat(self, log: RichLog, args: str) -> None:
        """Show cat animation."""
        cat_widget = self.query_one("#ascii-cat", Static)
        states = ["idle", "happy", "sleeping"]
        for state in states:
            self.cat_animator.set_state(state)
            cat_widget.update(self.cat_animator.get_art())
            await asyncio.sleep(1.0)
        self.cat_animator.set_state("idle")
        cat_widget.update(self.cat_animator.get_art())
        log.write_line("[accent]🐱 Meow![/accent]")

    async def _cmd_purr(self, log: RichLog, args: str) -> None:
        """Make the cat purr."""
        cat_widget = self.query_one("#ascii-cat", Static)
        self.cat_animator.set_state("happy")
        cat_widget.update(self.cat_animator.get_art())
        log.write_line("[accent]🐱 Purrrrrr...[/accent]")
        await asyncio.sleep(2.0)
        self.cat_animator.set_state("idle")
        cat_widget.update(self.cat_animator.get_art())

    def action_history_up(self) -> None:
        """Navigate history up."""
        if self.history and self.history_index > 0:
            self.history_index -= 1
            inp = self.query_one("#command-input", Input)
            inp.value = self.history[self.history_index]

    def action_history_down(self) -> None:
        """Navigate history down."""
        if self.history and self.history_index < len(self.history) - 1:
            self.history_index += 1
            inp = self.query_one("#command-input", Input)
            inp.value = self.history[self.history_index]
        else:
            self.history_index = len(self.history)
            inp = self.query_one("#command-input", Input)
            inp.value = ""

    def action_clear_screen(self) -> None:
        """Clear the message area."""
        log = self.query_one("#message-area", RichLog)
        log.clear()

    def action_recall(self) -> None:
        """Recall last command."""
        if self.history:
            inp = self.query_one("#command-input", Input)
            inp.value = self.history[-1]
            inp.focus()

    def action_toggle_tool_panel(self) -> None:
        """Toggle tool panel."""
        self.tool_panel_visible = not self.tool_panel_visible

    def action_toggle_mode(self) -> None:
        """Toggle mode."""
        modes = ["default", "auto", "plan"]
        current = modes.index(self.mode) if self.mode in modes else 0
        self.mode = modes[(current + 1) % len(modes)]
        prompt = self.query_one("#prompt-indicator", Static)
        prompt.update(f"[{self.mode}]")

    def action_help(self) -> None:
        """Show help."""
        asyncio.create_task(self.handle_command("/help"))

    def action_status(self) -> None:
        """Show status."""
        asyncio.create_task(self.handle_command("/status"))

    def action_session(self) -> None:
        """Show session."""
        asyncio.create_task(self.handle_command("/session"))

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        command = event.value.strip()
        if command:
            await self.run_command(command)
            event.input.value = ""


async def run_tui(runner: AgentRunner) -> None:
    """Run the Textual TUI."""
    app = MiniCodeTUI(runner)
    await app.run_async()


def main():
    """Main entry point for TUI mode."""
    from minicode.agent.runner import AgentRunner

    runner = AgentRunner()
    asyncio.run(run_tui(runner))
