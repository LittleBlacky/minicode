"""Textual TUI widgets and components."""
from textual.widgets import Static, Log, RichLog
from textual.message import Message


class StatusBar(Static):
    """Status bar widget."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: $surface-darken-1;
        dock: bottom;
        padding: 0 2;
    }

    StatusBar > .status-text {
        color: $text-muted;
    }

    StatusBar > .status-indicator {
        color: $success;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._status = "Ready"
        self._indicator = ""

    def set_status(self, status: str, indicator: str = "") -> None:
        """Update status text."""
        self._status = status
        self._indicator = indicator
        self.refresh()

    def render(self) -> str:
        """Render the status bar."""
        indicator = f"[{self._indicator}] " if self._indicator else ""
        return f"{indicator}{self._status}"


class CommandPalette(Static):
    """Command palette widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commands: list[tuple[str, str]] = []

    def add_command(self, name: str, description: str) -> None:
        """Add a command to the palette."""
        self.commands.append((name, description))

    def render(self) -> str:
        """Render the command palette."""
        lines = ["[bold]Commands:[/bold]"]
        for name, desc in self.commands:
            lines.append(f"  {name} - {desc}")
        return "\n".join(lines)


class ToolCallLog(Static):
    """Log widget for tool calls."""

    DEFAULT_CSS = """
    ToolCallLog {
        background: $surface-darken-2;
        color: $text;
        padding: 1 2;
        height: auto;
        max-height: 10;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[dict] = []

    def add_call(self, tool_name: str, args: dict, result: str = "") -> None:
        """Log a tool call."""
        self.calls.append({
            "tool": tool_name,
            "args": args,
            "result": result,
        })
        self.refresh()

    def clear(self) -> None:
        """Clear all logs."""
        self.calls = []
        self.refresh()

    def render(self) -> str:
        """Render tool call logs."""
        if not self.calls:
            return "[dim]No tool calls[/dim]"

        lines = ["[bold]Tool Calls:[/bold]"]
        for call in self.calls[-5:]:  # Show last 5
            lines.append(f"  [cyan]{call['tool']}[/cyan]")
        return "\n".join(lines)


class MessageBubble(Static):
    """Message bubble widget."""

    DEFAULT_CSS = """
    MessageBubble {
        width: 100%;
        height: auto;
        margin: 1 0;
        padding: 1 2;
    }

    MessageBubble.user {
        background: $primary;
        color: $text;
        text-style: bold;
    }

    MessageBubble.assistant {
        background: $surface;
        color: $text;
        border-left: solid $primary;
    }

    MessageBubble.system {
        background: $warning;
        color: $text;
    }
    """

    def __init__(self, role: str, content: str, *args, **kwargs):
        super().__init__(content, *args, **kwargs)
        self.role = role
        self.update_classes({"user", "assistant", "system"})

    def render(self) -> str:
        """Render the message."""
        prefix = {
            "user": "You",
            "assistant": "MiniCode",
            "system": "System",
        }.get(self.role, self.role)
        return f"[bold]{prefix}:[/bold] {self.content}"
