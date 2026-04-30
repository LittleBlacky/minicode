"""Textual TUI widgets and components - Dark theme compatible."""
from textual.widgets import Static, Log, RichLog
from textual.message import Message

from minicode.tui.themes.dark import dark_theme as theme


class StatusBar(Static):
    """Status bar widget with dark theme."""

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
        lines = ["[bold cyan]Commands:[/bold cyan]"]
        for name, desc in self.commands:
            lines.append(f"  [cyan]{name}[/cyan] - {desc}")
        return "\n".join(lines)


class ToolCallLog(Static):
    """Log widget for tool calls with dark theme."""

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

        lines = ["[bold cyan]Tool Calls:[/bold cyan]"]
        for call in self.calls[-5:]:  # Show last 5
            lines.append(f"  [cyan]{call['tool']}[/cyan]")
            if call.get("result"):
                result = call["result"]
                if len(result) > 50:
                    result = result[:50] + "..."
                lines.append(f"    [dim]{result}[/dim]")
        return "\n".join(lines)


class MessageBubble(Static):
    """Message bubble widget with dark theme."""

    def __init__(self, role: str, content: str, *args, **kwargs):
        super().__init__(content, *args, **kwargs)
        self.role = role
        self.update_classes({"user", "assistant", "system"})

    def render(self) -> str:
        """Render the message."""
        prefix_colors = {
            "user": "[cyan bold]You[/cyan bold]",
            "assistant": "[green bold]MiniCode[/green bold]",
            "system": "[dim bold]System[/dim bold]",
        }
        prefix = prefix_colors.get(self.role, self.role)
        return f"{prefix}\n{self.content}"


class ToolItem(Static):
    """Individual tool item widget."""

    def __init__(self, name: str, status: str = "running", *args, **kwargs):
        self.tool_name = name
        self.status = status
        super().__init__(*args, **kwargs)

    def render(self) -> str:
        """Render the tool item."""
        status_icon = {
            "running": "[yellow]◐[/yellow]",
            "success": "[green]●[/green]",
            "error": "[red]✗[/red]",
        }.get(self.status, "[dim]○[/dim]")

        return f"{status_icon} [cyan]{self.tool_name}[/cyan]"


class Notification(Static):
    """Notification toast widget."""

    def __init__(self, message: str, type: str = "info", *args, **kwargs):
        """
        Args:
            message: Notification message
            type: Type of notification (info, success, warning, error)
        """
        self.message = message
        self.notification_type = type
        super().__init__(*args, **kwargs)

    def render(self) -> str:
        """Render the notification."""
        border_colors = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }
        color = border_colors.get(self.notification_type, "blue")
        icon = {
            "info": "ℹ",
            "success": "✓",
            "warning": "⚠",
            "error": "✗",
        }.get(self.notification_type, "ℹ")

        return f"[{color}]{icon}[/{color}] {self.message}"
