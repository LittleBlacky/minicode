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
        Binding("f5", "show_config", "Config", show=True),
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

        # Permission system state
        self._pending_command: Optional[str] = None
        self._permission_callback_result: Optional[tuple[str, str]] = None
        self._permission_lock: Optional[asyncio.Lock] = None

        # Setup permission callback for bash tools
        self._setup_permission_system()

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
                "F5:配置  |  Ctrl+K: 命令  |  Ctrl+L: 清屏",
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

        # Show permission status
        from minicode.tools.permission_config import get_permission_config
        config = get_permission_config()
        summary = config.get_config_summary()
        log.write(f"[dim]Permission system: {summary['prompt_threshold']} threshold[/dim]\n")

        # Update status
        self._update_status()

    def _update_status(self, extra: str = "") -> None:
        """Update status bar."""
        center = self.query_one("#status-center", Static)
        status_text = f"Messages: {len(self.messages)} | Mode: {self.mode}"
        if extra:
            status_text += f" | {extra}"
        center.update(status_text)

    # ============ Permission System ============

    def _setup_permission_system(self) -> None:
        """Setup permission callback for bash tool."""
        from minicode.tools.bash_tools import set_permission_callback
        set_permission_callback(self._permission_ask)

    def _permission_ask(self, command: str) -> tuple[str, str]:
        """Ask for permission - called from bash tool.

        Returns:
            tuple of (status, message)
            - ("allow", "") if allowed
            - ("prompt", message) if needs user confirmation
            - ("block", reason) if blocked
        """
        from minicode.tools.permission_config import get_permission_config

        config = get_permission_config()
        allowed, reason, risk, _ = config.check(command)

        if not allowed:
            return ("block", reason)

        if not config.needs_prompt(command):
            return ("allow", "")

        # Need to prompt - prepare UI
        pattern = config.extract_command_type(command)
        message = f"Command requires permission: {command}\nRisk: {risk}\nPattern: {pattern}"

        return ("prompt", message)

    def _show_permission_dialog(self, command: str) -> None:
        """Show interactive permission prompt dialog."""
        from minicode.tools.permission_config import get_permission_config
        from minicode.tui.dialogs import PermissionPromptDialog

        config = get_permission_config()
        allowed, reason, risk, _ = config.check(command)
        pattern = config.extract_command_type(command)

        # Store pending command
        self._pending_command = command

        # Show dialog
        dialog = PermissionPromptDialog(
            command=command,
            reason=reason or "Requires confirmation",
            risk=risk,
            pattern=pattern,
        )
        self.mount(dialog)

    def _handle_permission_response(self, action: str, pattern: str) -> None:
        """Handle permission response from dialog."""
        from minicode.tools.permission_config import get_permission_config

        config = get_permission_config()
        command = self._pending_command

        if action == "session_allow" and command:
            # Add session pattern (选项 a)
            config.add_session_pattern(command)
            # Post to log
            log = self.query_one("#message-log", RichLog)
            log.write(f"[green]✓[/green] Session allowed: {pattern}")

        elif action == "deny" and command:
            # Add to deny list (选项 d)
            # TODO: Implement permanent deny
            log = self.query_one("#message-log", RichLog)
            log.write(f"[red]Added to deny list: {command}[/red]")

        # Clear pending
        self._pending_command = None

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

    def action_show_config(self) -> None:
        """Show interactive config dialog."""
        from minicode.tui.dialogs import ConfigDialog
        dialog = ConfigDialog()
        self.mount(dialog)

    def action_suspend(self) -> None:
        """Suspend execution."""
        pass  # TODO: Implement

    # ============ Event Handlers ============

    def on_config_saved(self, event) -> None:
        """Handle config saved event - trigger hot reload."""
        from minicode.tui.dialogs import ConfigSaved
        if isinstance(event, ConfigSaved):
            # 显示重载提示
            log = self.query_one("#message-log", RichLog)
            log.write(f"[green]Config saved! Reloading model...[/green]")
            # 触发热重载
            self.runner.reload_config()
            log.write(f"[green]Model reloaded: {event.config['provider']}/{event.config['model']}[/green]")

    def on_permission_response(self, event) -> None:
        """Handle permission response from dialog."""
        from minicode.tui.dialogs import PermissionResponse
        if isinstance(event, PermissionResponse):
            self._handle_permission_response(event.action, event.pattern)

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
        try:
            sidebar = self.query_one("#sidebar")
            if hasattr(sidebar, "add_history"):
                sidebar.add_history(command)
        except Exception:
            pass

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
            "/permission": self._cmd_permission,
            "/permissions": self._cmd_permission,
            "/perms": self._cmd_permission,
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
[cyan]/config[/cyan]          Unified config (show|provider|model|apikey|baseurl)
[cyan]/theme[/cyan]          Change theme

[bold]Security:[/bold]
[cyan]/permission[/cyan]     Show permission configuration
[cyan]/permission reload[/cyan]  Reload permission rules

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

    async def _cmd_config(self, log: RichLog, args: str) -> None:
        """Unified config command - opens dialog or shows text output."""
        import os
        from minicode.services.config import get_config_manager

        parts = args.split()
        action = parts[0].lower() if parts else "interactive"

        config = get_config_manager()

        if action == "interactive" or action == "show" or not args:
            # Launch interactive dialog
            from minicode.tui.dialogs import ConfigDialog
            dialog = ConfigDialog()
            self.mount(dialog)
            return

        if action == "provider":
            if len(parts) < 2:
                log.write("[yellow]Usage: /config provider <anthropic|openai>[/yellow]")
                return
            provider = parts[1].lower()
            if provider not in ("anthropic", "openai"):
                log.write("[yellow]Invalid provider. Use: anthropic, openai[/yellow]")
                return
            config.set("model.provider", provider)
            log.write(f"[green]Provider changed to: {provider}[/green]")
            return

        if action == "model":
            if len(parts) < 2:
                log.write("[yellow]Usage: /config model <model-name>[/yellow]")
                return
            model = parts[1]
            config.set("model.model", model)
            log.write(f"[green]Model changed to: {model}[/green]")
            return

        if action == "apikey":
            if len(parts) < 2:
                log.write("[yellow]Usage: /config apikey <key>[/yellow]")
                log.write("[dim]Example: /config apikey sk-xxxx[/dim]")
                return
            key = parts[1]
            os.environ["MINICODE_API_KEY"] = key
            log.write(f"[green]MINICODE_API_KEY set (in memory)[/green]")
            log.write("[dim]Note: This only affects current session. Restart to persist.[/dim]")
            return

        if action == "baseurl":
            if len(parts) < 2:
                log.write("[yellow]Usage: /config baseurl <url>[/yellow]")
                log.write("[dim]Example: /config baseurl https://api.anthropic.com[/dim]")
                return
            url = parts[1]
            os.environ["MINICODE_BASE_URL"] = url
            log.write(f"[green]MINICODE_BASE_URL set to: {url}[/green]")
            log.write("[dim]Note: This only affects current session. Restart to persist.[/dim]")
            return

        log.write(f"[yellow]Unknown action: {action}[/yellow]")
        log.write("[dim]Usage: /config show|provider|model|apikey|baseurl[/dim]")

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
        from minicode.services.config import get_config_manager
        config = get_config_manager()
        model_cfg = config.get_model_config()

        log.write("[bold cyan]MiniCode API Settings:[/bold cyan]")
        log.write(f"  Provider: {model_cfg.get('provider', 'unknown')}")
        log.write(f"  Model: {model_cfg.get('model', 'unknown')}")
        log.write(f"  API Key: {'[green]Set[/green]' if model_cfg.get('api_key') else '[red]Not set[/red]'}")
        log.write(f"  Base URL: {'[green]Set[/green]' if model_cfg.get('base_url') else '[red]Not set[/red]'}")

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

    async def _cmd_permission(self, log: RichLog, args: str) -> None:
        """Show permission configuration and rules."""
        from minicode.tools.permission_tools import get_permission_rules
        from minicode.tools.permission_config import get_permission_config

        parts = args.split()
        action = parts[0].lower() if parts else "show"

        rules = get_permission_rules()
        config_info = rules.get("config", {})
        config = get_permission_config()

        if action == "show" or action == "":
            log.write("[bold cyan]Permission Configuration:[/bold cyan]")
            log.write(f"  Config file: {config_info.get('config_path', 'N/A')}")
            log.write(f"  Loaded: {'[green]Yes[/green]' if config_info.get('loaded') else '[yellow]No[/yellow]'}")
            log.write(f"  Allow patterns: {config_info.get('allow_patterns', 0)}")
            log.write(f"  Deny patterns: {config_info.get('deny_patterns', 0)}")
            log.write(f"  Permanent deny: {config_info.get('permanent_deny_patterns', 0)}")
            log.write(f"  Session patterns: {config_info.get('session_patterns', 0)} (选项 a)")
            log.write(f"  Prompt threshold: {config_info.get('prompt_threshold', 'medium')}")
            log.write("")

            # Show permanent deny patterns
            perm_deny = config.get_permanent_deny_patterns()
            if perm_deny:
                log.write("[bold]Permanent Deny (选项 d):[/bold]")
                for p in perm_deny:
                    log.write(f"  [red]- {p}[/red]")
                log.write("")

            # Show session patterns
            sess_patterns = config.get_session_patterns()
            if sess_patterns:
                log.write("[bold]Session Patterns (选项 a):[/bold]")
                for p in sess_patterns:
                    log.write(f"  [cyan]+ {p}[/cyan]")
                log.write("")

            log.write("[bold]Built-in Dangerous Patterns:[/bold]")
            for p in rules.get("builtin_patterns", []):
                risk_color = {
                    "critical": "[red]",
                    "high": "[orange]",
                    "medium": "[yellow]",
                    "low": "[green]",
                }.get(p["risk"], "")
                log.write(f"  {risk_color}[{p['risk']}][/{risk_color}] {p['name']}: {p['description']}")
            log.write("")
            log.write("[dim]Use /permission reload to reload config[/dim]")
            log.write("[dim]Use /permission clear-session to clear session patterns[/dim]")
            return

        if action == "reload":
            from minicode.tools.permission_tools import reset_permission_config
            reset_permission_config()
            log.write("[green]Permission configuration reloaded[/green]")
            return

        if action == "clear-session":
            config.clear_session_patterns()
            log.write("[green]Session patterns cleared[/green]")
            return

        log.write(f"[yellow]Unknown action: {action}[/yellow]")
        log.write("[dim]Usage: /permission show|reload|clear-session[/dim]")


async def run_tui(runner: AgentRunner) -> None:
    """Run the Textual TUI."""
    app = MiniCodeTUI(runner)
    await app.run_async()


def main():
    """Main entry point for TUI mode."""
    from minicode.agent.runner import AgentRunner

    runner = AgentRunner()
    asyncio.run(run_tui(runner))
