"""MiniCode Textual TUI - Professional dark theme with streaming and tool panel."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import Header, Footer, Input, RichLog, Static
from textual.message import Message

from langchain_core.messages import HumanMessage, AIMessage

from minicode.agent.runner import AgentRunner
from minicode.tui.ascii_art import ASCIIArt, CatAnimator
from minicode.tui.themes.dark import dark_theme as theme
from minicode.tui.themes.styles import get_theme_css


class MiniCodeTUI(App):
    """MiniCode Textual TUI Application - Professional dark theme."""

    # Use dark theme CSS
    CSS = get_theme_css("dark")

    BINDINGS = [
        # Core bindings
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
        Binding("ctrl+r", "recall", "Recall", show=True),
        Binding("ctrl+k", "toggle_command_palette", "Commands", show=True),
        Binding("ctrl+z", "suspend", "Suspend", show=True),

        # Navigation
        Binding("up", "history_up", "History", show=False),
        Binding("down", "history_down", "History", show=False),

        # History navigation
        Binding("ctrl+shift+up", "history_up", "", show=False),
        Binding("ctrl+shift+down", "history_down", "", show=False),

        # Tool panel
        Binding("ctrl+a", "toggle_sidebar", "Tools", show=True),
        Binding("ctrl+t", "toggle_sidebar", "Tools", show=True),

        # Mode toggle
        Binding("ctrl+e", "toggle_mode", "Mode", show=True),

        # Help and info
        Binding("f1", "show_help", "Help", show=True),
        Binding("f2", "show_status", "Status", show=True),
        Binding("f3", "show_history", "History", show=True),
        Binding("f4", "show_session", "Session", show=True),
    ]

    def __init__(self, runner: AgentRunner, **kwargs):
        super().__init__(**kwargs)
        self.runner = runner
        self.messages: list = []
        self.history: list[str] = []
        self.history_index: int = -1
        self.mode: str = "default"
        self.start_time: datetime = datetime.now()
        self.command_count: int = 0
        self.cat_animator = CatAnimator()
        self.is_thinking: bool = False
        self.sidebar_visible: bool = True

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        # Header
        yield Header(id="header")

        # Main container (messages + sidebar)
        with Container(id="main-container"):
            # Message area
            with Container(id="message-area"):
                yield Static(ASCIIArt.get_cat_frame(0), id="ascii-art")
                yield RichLog(id="message-log", highlight=True, markup=True)

            # Sidebar (tools panel)
            if self.sidebar_visible:
                from minicode.tui.widgets.sidebar import ToolSidebar
                yield ToolSidebar()

        # Input area
        with Container(id="input-container"):
            with Horizontal(id="input-row"):
                yield Static("[default]", id="prompt-indicator")
                yield Input(
                    placeholder="Type your message... (@file /command)",
                    id="command-input",
                )

        # Status bar
        with Container(id="status-bar"):
            yield Static("MiniCode", id="status-left")
            yield Static("", id="status-center")
            yield Static(
                "Ctrl+K: 命令  |  Ctrl+L: 清屏  |  Ctrl+A: 工具",
                id="status-right",
            )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize on mount."""
        self.title = "MiniCode"
        self.sub_title = "Claude-style coding agent"

        # Focus input
        self.query_one("#command-input", Input).focus()

        # Show welcome message
        log = self.query_one("#message-log", RichLog)
        log.write("[bold green]Welcome to MiniCode![/bold green]\n")
        log.write("[dim]Type /help for available commands[/dim]\n")
        log.write("[dim]Press Ctrl+K for command palette[/dim]\n")
        log.write("[dim]Press Ctrl+A to toggle tools panel[/dim]\n")
        log.write("\n")

        # Update status
        self._update_status()

    def _update_status(self, extra: str = "") -> None:
        """Update status bar."""
        center = self.query_one("#status-center", Static)
        status_text = f"Messages: {len(self.messages)} | Mode: {self.mode}"
        if extra:
            status_text += f" | {extra}"
        center.update(status_text)

    # ============ Actions ============

    def action_clear_screen(self) -> None:
        """Clear the message log."""
        log = self.query_one("#message-log", RichLog)
        log.clear()
        log.write("[dim]Screen cleared.[/dim]\n")

    def action_recall(self) -> None:
        """Recall last command."""
        if self.history:
            inp = self.query_one("#command-input", Input)
            inp.value = self.history[-1]
            inp.focus()

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

    def action_toggle_sidebar(self) -> None:
        """Toggle the sidebar panel."""
        self.sidebar_visible = not self.sidebar_visible
        if self.sidebar_visible:
            self.mount(ToolSidebar())
        else:
            sidebar = self.query_one("#sidebar")
            if sidebar:
                sidebar.remove()

    def action_toggle_mode(self) -> None:
        """Toggle mode."""
        modes = ["default", "auto", "plan"]
        current = modes.index(self.mode) if self.mode in modes else 0
        self.mode = modes[(current + 1) % len(modes)]
        self._update_prompt_indicator()
        self._update_status()

    def action_toggle_command_palette(self) -> None:
        """Toggle command palette."""
        # TODO: Implement command palette overlay
        pass

    def action_show_help(self) -> None:
        """Show help."""
        log = self.query_one("#message-log", RichLog)
        self._show_help(log)

    def action_show_status(self) -> None:
        """Show status."""
        log = self.query_one("#message-log", RichLog)
        self._cmd_status(log, "")

    def action_show_history(self) -> None:
        """Show history."""
        log = self.query_one("#message-log", RichLog)
        self._cmd_history(log, "")

    def action_show_session(self) -> None:
        """Show session info."""
        log = self.query_one("#message-log", RichLog)
        self._cmd_session(log, "")

    def action_suspend(self) -> None:
        """Suspend execution."""
        pass  # TODO: Implement

    # ============ Event Handlers ============

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        command = event.value.strip()
        if command:
            event.input.value = ""
            await self.run_command(command)

    def _update_prompt_indicator(self) -> None:
        """Update the prompt indicator based on mode."""
        indicator = self.query_one("#prompt-indicator", Static)
        mode_colors = {
            "default": "green",
            "auto": "cyan",
            "plan": "yellow",
        }
        color = mode_colors.get(self.mode, "green")
        mode_text = self.mode.upper() if self.mode != "default" else "MINI"
        indicator.update(f"[{color}]{mode_text}[/{color}]")

    # ============ Command Execution ============

    async def run_command(self, command: str) -> None:
        """Execute a command."""
        log = self.query_one("#message-log", RichLog)
        status = self.query_one("#status-center", Static)
        self.command_count += 1

        # Handle built-in commands
        if command.startswith("/"):
            await self.handle_command(command)
            self._update_status()
            return

        # Add to history
        self.history.append(command)
        self.history_index = len(self.history)

        # Add to sidebar history
        sidebar = self.query_one("#sidebar", Static)
        if sidebar and hasattr(sidebar, "add_history"):
            sidebar.add_history(command)

        # Add user message
        self.messages.append(HumanMessage(content=command))

        # Display user message
        timestamp = datetime.now().strftime("%H:%M")
        log.write(f"[dim]{timestamp}[/dim] [bold cyan]You:[/bold cyan]\n")
        log.write(f"  {command}\n")
        log.write("\n")

        # Update thinking state
        self.is_thinking = True
        self.cat_animator.set_state("thinking")
        self._update_ascii_art("thinking")
        status.update("[yellow]Thinking...[/yellow]")

        # Run agent
        try:
            result = await self.runner.run(self.messages)

            # Extract assistant response
            if "messages" in result:
                response = result["messages"][-1]
                content = response.content if hasattr(response, "content") else str(response)
                self.messages.append(AIMessage(content=content))

                # Display response
                timestamp = datetime.now().strftime("%H:%M")
                log.write(f"[dim]{timestamp}[/dim] [bold green]MiniCode:[/bold green]\n")
                # Parse and display markdown
                for line in content.split("\n"):
                    log.write(f"  {line}\n")
                log.write("\n")
            else:
                log.write("[bold red]Error:[/bold red] No response")

        except Exception as e:
            log.write(f"[bold red]Error:[/bold red] {e}")

        # Reset thinking state
        self.is_thinking = False
        self.cat_animator.set_state("idle")
        self._update_ascii_art("idle")
        self._update_status()

    def _update_ascii_art(self, state: str) -> None:
        """Update ASCII art based on state."""
        self.cat_animator.set_state(state)
        ascii_widget = self.query_one("#ascii-art", Static)
        if ascii_widget:
            ascii_widget.update(self.cat_animator.get_art())

    async def handle_command(self, command: str) -> None:
        """Handle built-in slash commands."""
        log = self.query_one("#message-log", RichLog)
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
            "/theme": self._cmd_theme,
            "/tools": self._cmd_tools,
            "/env": self._cmd_env,
            "/keys": self._cmd_keys,
            "/export": self._cmd_export,
            "/import": self._cmd_import,
            "/time": self._cmd_time,
            "/uptime": self._cmd_uptime,
            "/cat": self._cmd_cat,
        }

        if cmd in commands:
            await commands[cmd](log, args)
        else:
            log.write(f"[yellow]Unknown command:[/yellow] {cmd}\n")
            log.write("[dim]Type /help for available commands[/dim]\n")

    # ============ Command Handlers ============

    async def _show_help(self, log: RichLog) -> None:
        """Show help."""
        help_text = """
[bold cyan]MiniCode Commands[/bold cyan]

[bold]Navigation:[/bold]
[cyan]/help[/cyan]           Show this help
[cyan]/clear[/cyan]          Clear the screen
[cyan]/history[/cyan]        Show command history
[cyan]/status[/cyan]         Show status
[cyan]/config[/cyan]         Show configuration
[cyan]/session[/cyan]        Session info
[cyan]/exit[/cyan]           Exit the application

[bold]Context:[/bold]
[cyan]/memory[/cyan]         Show saved memories
[cyan]/skills[/cyan]         List skills
[cyan]/context[/cyan]        Context info
[cyan]/compact[/cyan]        Compact context
[cyan]/stat[/cyan]           Statistics

[bold]Model:[/bold]
[cyan]/model[/cyan] <name>   Change model
[cyan]/provider[/cyan] <p>   Change provider
[cyan]/theme[/cyan]          Change theme

[bold]Tools:[/bold]
[cyan]/tools[/cyan]          List all tools
[cyan]/keys[/cyan]           API key status
[cyan]/env[/cyan]            Environment

[bold]Session:[/bold]
[cyan]/export[/cyan] <file>  Export session
[cyan]/import[/cyan] <file>  Import session
[cyan]/retry[/cyan]          Retry last command
[cyan]/time[/cyan]           Current time
[cyan]/uptime[/cyan]         Show uptime

[bold]Keyboard Shortcuts:[/bold]
[cyan]Ctrl+K[/cyan]          Command palette
[cyan]Ctrl+L[/cyan]          Clear screen
[cyan]Ctrl+R[/cyan]          Recall last command
[cyan]Ctrl+A[/cyan]          Toggle tools panel
[cyan]Ctrl+E[/cyan]          Toggle mode
[cyan]Up/Down[/cyan]         Navigate history
[cyan]F1[/cyan]              Help
[cyan]F2[/cyan]              Status
[cyan]F3[/cyan]              History
[cyan]F4[/cyan]              Session
"""
        log.write(help_text + "\n")

    async def _cmd_help(self, log: RichLog, args: str) -> None:
        """Show help."""
        await self._show_help(log)

    async def _cmd_clear(self, log: RichLog, args: str) -> None:
        """Clear screen."""
        log.clear()
        log.write("[dim]Screen cleared.[/dim]")

    async def _cmd_history(self, log: RichLog, args: str) -> None:
        """Show history."""
        if self.history:
            limit = int(args) if args.isdigit() else 10
            history_text = f"[bold]Command History (last {limit}):[/bold]\n"
            start = max(0, len(self.history) - limit)
            for i, h in enumerate(self.history[start:], start + 1):
                truncated = h[:50] + "..." if len(h) > 50 else h
                history_text += f"  {i}. {truncated}\n"
            log.write(history_text)
        else:
            log.write("[dim]No history yet.[/dim]")

    async def _cmd_quit(self, log: RichLog, args: str) -> None:
        """Quit."""
        log.write("[dim]Goodbye! 🐱[/dim]")
        self.exit()

    async def _cmd_status(self, log: RichLog, args: str) -> None:
        """Show status."""
        try:
            from minicode.services.config import get_config_manager
            config = get_config_manager()
            model_provider = config.get("model.provider", "unknown")
            model_name = config.get("model.model", "unknown")
        except Exception:
            model_provider = "unknown"
            model_name = "unknown"

        uptime = datetime.now() - self.start_time
        log.write("[bold cyan]Status:[/bold cyan]")
        log.write(f"  Status: [green]Ready[/green]")
        log.write(f"  Mode: {self.mode}")
        log.write(f"  Model: {model_provider}/{model_name}")
        log.write(f"  Messages: {len(self.messages)}")
        log.write(f"  Commands: {len(self.history)}")
        log.write(f"  Uptime: {uptime}")

    async def _cmd_config(self, log: RichLog, args: str) -> None:
        """Show config."""
        try:
            from minicode.services.config import get_config_manager
            config = get_config_manager()
            log.write("[bold cyan]Configuration:[/bold cyan]")
            log.write(f"  Model Provider: {config.get('model.provider')}")
            log.write(f"  Model: {config.get('model.model')}")
            log.write(f"  Permission Mode: {config.get('permissions.mode')}")
        except Exception as e:
            log.write(f"[dim]Config not available: {e}[/dim]")

    async def _cmd_session(self, log: RichLog, args: str) -> None:
        """Show session."""
        log.write("[bold cyan]Session:[/bold cyan]")
        log.write(f"  Messages: {len(self.messages)}")
        log.write(f"  History: {len(self.history)}")
        log.write(f"  Mode: {self.mode}")
        log.write(f"  Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    async def _cmd_memory(self, log: RichLog, args: str) -> None:
        """Show memories."""
        try:
            from minicode.tools.memory_tools import get_memory_manager
            mgr = get_memory_manager()
            memories = list(mgr.memory_dir.glob("*.md"))
            memories = [m for m in memories if m.name != "MEMORY.md"]
            if memories:
                log.write("[bold cyan]Saved Memories:[/bold cyan]")
                for m in memories[:10]:
                    log.write(f"  - {m.stem}")
            else:
                log.write("[dim]No memories saved.[/dim]")
        except Exception as e:
            log.write(f"[dim]Memory not available: {e}[/dim]")

    async def _cmd_skills(self, log: RichLog, args: str) -> None:
        """Show skills."""
        try:
            from minicode.tools.skill_tools import get_skill_manager
            mgr = get_skill_manager()
            skills = mgr.list()
            if skills:
                log.write("[bold cyan]Available Skills:[/bold cyan]")
                for s in skills:
                    log.write(f"  - {s['name']}: {s['description']}")
            else:
                log.write("[dim]No skills available.[/dim]")
        except Exception as e:
            log.write(f"[dim]Skills not available: {e}[/dim]")

    async def _cmd_context(self, log: RichLog, args: str) -> None:
        """Show context."""
        log.write("[bold cyan]Context:[/bold cyan]")
        log.write(f"  Messages: {len(self.messages)}")
        log.write(f"  History: {len(self.history)}")
        log.write(f"  Mode: {self.mode}")

    async def _cmd_stat(self, log: RichLog, args: str) -> None:
        """Show statistics."""
        uptime = datetime.now() - self.start_time
        log.write("[bold cyan]Statistics:[/bold cyan]")
        log.write(f"  Total Messages: {len(self.messages)}")
        log.write(f"  Total Commands: {len(self.history)}")
        log.write(f"  Commands This Session: {self.command_count}")
        log.write(f"  Uptime: {uptime}")

    async def _cmd_compact(self, log: RichLog, args: str) -> None:
        """Compact context."""
        log.write("[dim]Context will be compacted before next response.[/dim]")

    async def _cmd_retry(self, log: RichLog, args: str) -> None:
        """Retry last command."""
        if self.history:
            last_cmd = self.history[-1]
            log.write(f"[dim]Retrying: {last_cmd}[/dim]")
            await self.run_command(last_cmd)
        else:
            log.write("[dim]No command to retry.[/dim]")

    async def _cmd_mode(self, log: RichLog, args: str) -> None:
        """Change mode."""
        if args in ("default", "auto", "plan"):
            self.mode = args
            self._update_prompt_indicator()
            log.write(f"[green]Mode changed to: {self.mode}[/green]")
        else:
            log.write(f"[yellow]Invalid mode: {args}[/yellow]")
            log.write("[dim]Available modes: default, auto, plan[/dim]")

    async def _cmd_model(self, log: RichLog, args: str) -> None:
        """Change model."""
        if args:
            try:
                from minicode.services.config import get_config_manager
                config = get_config_manager()
                config.set("model.model", args)
                log.write(f"[green]Model changed to: {args}[/green]")
            except Exception as e:
                log.write(f"[yellow]Failed: {e}[/yellow]")
        else:
            log.write("[dim]Usage: /model <model-name>[/dim]")

    async def _cmd_provider(self, log: RichLog, args: str) -> None:
        """Change provider."""
        if args in ("anthropic", "openai"):
            try:
                from minicode.services.config import get_config_manager
                config = get_config_manager()
                config.set("model.provider", args)
                log.write(f"[green]Provider changed to: {args}[/green]")
            except Exception as e:
                log.write(f"[yellow]Failed: {e}[/yellow]")
        else:
            log.write("[yellow]Invalid provider[/yellow]")
            log.write("[dim]Available providers: anthropic, openai[/dim]")

    async def _cmd_theme(self, log: RichLog, args: str) -> None:
        """Change theme."""
        log.write("[dim]Theme change not yet implemented[/dim]")

    async def _cmd_tools(self, log: RichLog, args: str) -> None:
        """List tools."""
        try:
            from minicode.tools.registry import ALL_TOOLS
            log.write(f"[bold cyan]Available Tools ({len(ALL_TOOLS)}):[/bold cyan]")
            for tool in ALL_TOOLS[:30]:
                log.write(f"  - {tool.name}")
            if len(ALL_TOOLS) > 30:
                log.write(f"  [dim]... and {len(ALL_TOOLS) - 30} more[/dim]")
        except Exception as e:
            log.write(f"[dim]Tools not available: {e}[/dim]")

    async def _cmd_env(self, log: RichLog, args: str) -> None:
        """Show environment."""
        import os
        log.write("[bold cyan]Environment:[/bold cyan]")
        log.write(f"  Python: {os.sys.version.split()[0]}")
        log.write(f"  CWD: {os.getcwd()}")

    async def _cmd_keys(self, log: RichLog, args: str) -> None:
        """Show API keys status."""
        import os
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        log.write("[bold cyan]API Keys:[/bold cyan]")
        log.write(f"  ANTHROPIC_API_KEY: {'[green]Set[/green]' if has_anthropic else '[red]Not set[/red]'}")
        log.write(f"  OPENAI_API_KEY: {'[green]Set[/green]' if has_openai else '[red]Not set[/red]'}")

    async def _cmd_export(self, log: RichLog, args: str) -> None:
        """Export session."""
        if args:
            import json
            data = {
                "messages": [str(m.content) for m in self.messages],
                "history": self.history,
                "mode": self.mode,
            }
            try:
                with open(args, "w") as f:
                    json.dump(data, f, indent=2)
                log.write(f"[green]Session exported to: {args}[/green]")
            except Exception as e:
                log.write(f"[red]Error: {e}[/red]")
        else:
            log.write("[yellow]Usage: /export <filepath>[/yellow]")

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
                log.write(f"[green]Session imported from: {args}[/green]")
            except Exception as e:
                log.write(f"[red]Error: {e}[/red]")
        else:
            log.write("[yellow]Usage: /import <filepath>[/yellow]")

    async def _cmd_time(self, log: RichLog, args: str) -> None:
        """Show current time."""
        now = datetime.now()
        log.write(f"[bold cyan]Current Time:[/bold cyan] {now.strftime('%Y-%m-%d %H:%M:%S')}")

    async def _cmd_uptime(self, log: RichLog, args: str) -> None:
        """Show uptime."""
        uptime = datetime.now() - self.start_time
        log.write(f"[bold cyan]Uptime:[/bold cyan] {uptime}")

    async def _cmd_cat(self, log: RichLog, args: str) -> None:
        """Show cat animation."""
        self.cat_animator.set_state("happy")
        self._update_ascii_art("happy")
        await asyncio.sleep(1.5)
        self.cat_animator.set_state("idle")
        self._update_ascii_art("idle")
        log.write("[accent]🐱 Meow![/accent]")


async def run_tui(runner: AgentRunner) -> None:
    """Run the Textual TUI."""
    app = MiniCodeTUI(runner)
    await app.run_async()


def main():
    """Main entry point for TUI mode."""
    from minicode.agent.runner import AgentRunner

    runner = AgentRunner()
    asyncio.run(run_tui(runner))
