"""Tool Sidebar widget for MiniCode TUI - displays tool calls and history."""
from __future__ import annotations

import time
from typing import Optional
from dataclasses import dataclass, field

from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Header, Static, Label

from minicode.tui.themes.dark import dark_theme as theme


@dataclass
class ToolItem:
    """Represents a tool call item."""
    name: str
    status: str = "pending"  # pending, running, success, error
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    args: dict = field(default_factory=dict)
    result: str = ""
    error: str = ""

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def duration_str(self) -> str:
        """Get formatted duration string."""
        d = self.duration
        if d < 1:
            return f"{int(d * 1000)}ms"
        elif d < 60:
            return f"{d:.1f}s"
        else:
            mins = int(d / 60)
            secs = int(d % 60)
            return f"{mins}m {secs}s"


@dataclass
class HistoryItem:
    """Represents a command history item."""
    command: str
    timestamp: float = field(default_factory=time.time)


class ToolSidebar(Widget):
    """Sidebar widget showing tool calls and command history.

    Has two tabs:
    - Tools: Shows currently running and recent tool calls
    - History: Shows command history
    """

    class ToolCallStarted(Message):
        """Posted when a tool call starts."""
        def __init__(self, tool_name: str) -> None:
            super().__init__()
            self.tool_name = tool_name

    class ToolCallCompleted(Message):
        """Posted when a tool call completes."""
        def __init__(self, tool_name: str, success: bool) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.success = success

    BINDINGS = [
        ("t", "switch_tab('tools')", "Tools"),
        ("h", "switch_tab('history')", "History"),
    ]

    def __init__(self) -> None:
        super().__init__(id="sidebar")
        self.tools: list[ToolItem] = []
        self.history: list[HistoryItem] = []
        self.current_tab: str = "tools"
        self.max_tools: int = 50
        self.max_history: int = 20

    def compose(self) -> ComposeResult:
        """Compose the sidebar widgets."""
        yield Header("Tools", id="sidebar-header")

        # Tab buttons
        with Container(id="sidebar-tabs"):
            yield Static(
                "[Tools]",
                id="tab-tools",
                classes="sidebar-tab active",
            )
            yield Static(
                "[History]",
                id="tab-history",
                classes="sidebar-tab",
            )

        # Content area
        with ScrollableContainer(id="sidebar-content"):
            yield Static("", id="sidebar-content-inner")

        # Status bar
        yield Static("", id="sidebar-status")

    def on_mount(self) -> None:
        """Initialize sidebar on mount."""
        self._update_content()

    def _update_content(self) -> None:
        """Update the content based on current tab."""
        content = self.query_one("#sidebar-content-inner", Static)
        status = self.query_one("#sidebar-status", Static)

        if self.current_tab == "tools":
            content.update(self._render_tools())
            running = sum(1 for t in self.tools if t.status == "running")
            status.update(f"{len(self.tools)} tools | {running} running")
        else:
            content.update(self._render_history())
            status.update(f"{len(self.history)} commands")

    def _render_tools(self) -> str:
        """Render tools list."""
        if not self.tools:
            return "[dim]No tool calls yet[/dim]"

        lines = []
        # Show running tools first
        running = [t for t in self.tools if t.status == "running"]
        completed = [t for t in self.tools if t.status != "running"][:self.max_tools]

        for tool in running + completed:
            status_icon = {
                "pending": "[dim]○[/dim]",
                "running": f"[yellow]◐[/yellow]",
                "success": "[green]●[/green]",
                "error": "[red]✗[/red]",
            }.get(tool.status, "[dim]○[/dim]")

            status_text = {
                "pending": "pending",
                "running": "running...",
                "success": f"done ({tool.duration_str})",
                "error": f"error ({tool.duration_str})",
            }.get(tool.status, "")

            lines.append(f"{status_icon} [cyan]{tool.name}[/cyan]")
            if tool.status != "running" and tool.status != "pending":
                lines.append(f"   [dim]{status_text}[/dim]")

        return "\n".join(lines)

    def _render_history(self) -> str:
        """Render history list."""
        if not self.history:
            return "[dim]No commands yet[/dim]"

        lines = []
        for i, item in enumerate(reversed(self.history[-self.max_history:])):
            # Format time
            elapsed = time.time() - item.timestamp
            if elapsed < 60:
                time_str = f"{int(elapsed)}s ago"
            elif elapsed < 3600:
                time_str = f"{int(elapsed / 60)}m ago"
            else:
                time_str = f"{int(elapsed / 3600)}h ago"

            # Truncate long commands
            cmd = item.command[:40] + "..." if len(item.command) > 40 else item.command
            lines.append(f"[dim]{time_str}[/dim]  {cmd}")

        return "\n".join(lines)

    def add_tool(self, name: str, args: dict = None) -> None:
        """Add a new tool call to the sidebar.

        Args:
            name: Tool name
            args: Tool arguments (optional)
        """
        tool = ToolItem(name=name, args=args or {}, start_time=time.time())
        tool.status = "running"
        self.tools.append(tool)

        # Trim old tools
        if len(self.tools) > self.max_tools:
            self.tools = self.tools[-self.max_tools:]

        self._update_content()
        self.post_message(self.ToolCallStarted(name))

    def complete_tool(self, name: str, success: bool = True, result: str = "", error: str = "") -> None:
        """Mark a tool call as complete.

        Args:
            name: Tool name
            success: Whether the tool succeeded
            result: Tool result (optional)
            error: Error message if failed (optional)
        """
        # Find the most recent running tool with this name
        for tool in reversed(self.tools):
            if tool.name == name and tool.status == "running":
                tool.status = "success" if success else "error"
                tool.end_time = time.time()
                tool.result = result
                tool.error = error
                break

        self._update_content()
        self.post_message(self.ToolCallCompleted(name, success))

    def add_history(self, command: str) -> None:
        """Add a command to history.

        Args:
            command: Command string
        """
        self.history.append(HistoryItem(command=command))

        # Trim old history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        if self.current_tab == "history":
            self._update_content()

    def clear_tools(self) -> None:
        """Clear all tool calls."""
        self.tools.clear()
        self._update_content()

    def clear_history(self) -> None:
        """Clear command history."""
        self.history.clear()
        self._update_content()

    def switch_tab(self, tab: str) -> None:
        """Switch to a different tab.

        Args:
            tab: Tab name ('tools' or 'history')
        """
        if tab not in ("tools", "history"):
            return

        self.current_tab = tab

        # Update tab styles
        tools_tab = self.query_one("#tab-tools", Static)
        history_tab = self.query_one("#tab-history", Static)

        if tab == "tools":
            tools_tab.update_classes("sidebar-tab active")
            history_tab.update_classes("sidebar-tab")
        else:
            tools_tab.update_classes("sidebar-tab")
            history_tab.update_classes("sidebar-tab active")

        self._update_content()

    def action_switch_tab(self, tab: str) -> None:
        """Action to switch tabs (bound to key)."""
        self.switch_tab(tab)


class ToolItemWidget(Widget):
    """Individual tool item widget."""

    def __init__(self, tool: ToolItem) -> None:
        super().__init__()
        self.tool = tool

    def compose(self) -> ComposeResult:
        """Compose the tool item."""
        status_icon = self._get_status_icon()
        yield Label(f"{status_icon} {self.tool.name}", id="tool-name")
        if self.tool.status != "running":
            yield Label(self._get_status_text(), id="tool-status")

    def _get_status_icon(self) -> str:
        """Get status icon."""
        return {
            "pending": "[dim]○[/dim]",
            "running": "[yellow]◐[/yellow]",
            "success": "[green]●[/green]",
            "error": "[red]✗[/red]",
        }.get(self.tool.status, "[dim]○[/dim]")

    def _get_status_text(self) -> str:
        """Get status text."""
        if self.tool.status == "success":
            return f"[dim]done in {self.tool.duration_str}[/dim]"
        elif self.tool.status == "error":
            return f"[red]error in {self.tool.duration_str}[/red]"
        return ""


class CommandPaletteWidget(Widget):
    """Command palette overlay widget."""

    class CommandSelected(Message):
        """Posted when a command is selected."""
        def __init__(self, command: str) -> None:
            super().__init__()
            self.command = command

    def __init__(self) -> None:
        super().__init__(id="command-palette")
        self.commands: list[tuple[str, str, str]] = []  # (name, description, key)
        self.selected_index: int = 0
        self.query: str = ""

    def compose(self) -> ComposeResult:
        """Compose the command palette."""
        yield Static("Search commands...", id="palette-search")
        with ScrollableContainer(id="command-list"):
            yield Static("", id="command-list-inner")

    def add_command(self, name: str, description: str, key: str = "") -> None:
        """Add a command to the palette."""
        self.commands.append((name, description, key))

    def filter_commands(self, query: str) -> list[tuple[str, str, str]]:
        """Filter commands by query."""
        if not query:
            return self.commands

        q = query.lower()
        return [
            (name, desc, key)
            for name, desc, key in self.commands
            if q in name.lower() or q in desc.lower()
        ]

    def select_next(self) -> None:
        """Select next command."""
        filtered = self.filter_commands(self.query)
        if filtered:
            self.selected_index = (self.selected_index + 1) % len(filtered)
            self._update_list()

    def select_previous(self) -> None:
        """Select previous command."""
        filtered = self.filter_commands(self.query)
        if filtered:
            self.selected_index = (self.selected_index - 1) % len(filtered)
            self._update_list()

    def get_selected(self) -> Optional[tuple[str, str, str]]:
        """Get currently selected command."""
        filtered = self.filter_commands(self.query)
        if filtered and 0 <= self.selected_index < len(filtered):
            return filtered[self.selected_index]
        return None

    def _update_list(self) -> None:
        """Update the command list display."""
        filtered = self.filter_commands(self.query)
        if not filtered:
            self.query_one("#command-list-inner", Static).update("[dim]No commands found[/dim]")
            return

        lines = []
        for i, (name, desc, key) in enumerate(filtered):
            prefix = "> " if i == self.selected_index else "  "
            key_str = f" [{key}]" if key else ""
            lines.append(f"{prefix}[cyan]{name}[/cyan] - {desc}{key_str}")

        self.query_one("#command-list-inner", Static).update("\n".join(lines))
