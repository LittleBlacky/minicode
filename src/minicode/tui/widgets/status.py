"""Status bar widget for MiniCode TUI - Shows status and hints."""
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static
from textual.widget import Widget

from minicode.tui.themes.dark import dark_theme as theme


class TUIStatusBar(Widget):
    """Custom status bar for MiniCode.

    Shows:
    - Left: MiniCode branding
    - Center: Session info
    - Right: Keyboard hints
    """

    def __init__(
        self,
        session_id: str = "default",
        message_count: int = 0,
    ) -> None:
        super().__init__(id="status-bar")
        self.session_id = session_id
        self.message_count = message_count

    def compose(self) -> ComposeResult:
        """Compose the status bar."""
        with Horizontal(classes="status-row"):
            # Left: Branding
            yield Static(
                f"[dim]MiniCode[/dim]",
                id="status-left",
            )

            # Center: Session info
            yield Static(
                f"[dim]Session: {self.session_id} | Messages: {self.message_count}[/dim]",
                id="status-center",
                classes="status-center",
            )

            # Right: Hints
            yield Static(
                "[cyan]Ctrl+K[/cyan]: 命令  |  [cyan]Ctrl+L[/cyan]: 清屏  |  [cyan]Ctrl+A[/cyan]: 工具",
                id="status-right",
                classes="status-right",
            )


class StatusIndicator(Widget):
    """Shows connection and status indicator."""

    def __init__(self, connected: bool = True) -> None:
        super().__init__(id="status-indicator")
        self._connected = connected

    def set_connected(self, connected: bool) -> None:
        """Set connection status."""
        self._connected = connected
        self.refresh()

    def render(self) -> str:
        """Render the indicator."""
        if self._connected:
            return "[green]●[/green] Connected"
        return "[yellow]●[/yellow] Connecting..."


class ModeIndicator(Widget):
    """Shows current mode indicator."""

    def __init__(self, mode: str = "default") -> None:
        super().__init__(id="mode-indicator")
        self._mode = mode

    def set_mode(self, mode: str) -> None:
        """Set current mode."""
        self._mode = mode
        self.refresh()

    def render(self) -> str:
        """Render the mode indicator."""
        mode_colors = {
            "default": "green",
            "auto": "cyan",
            "plan": "yellow",
        }
        color = mode_colors.get(self._mode, "green")
        mode_text = self._mode.upper() if self._mode != "default" else "MINI"
        return f"[{color}]{mode_text}[/{color}]"


class ToolCounter(Widget):
    """Shows active tool count."""

    def __init__(self, running: int = 0, total: int = 0) -> None:
        super().__init__(id="tool-counter")
        self._running = running
        self._total = total

    def update(self, running: int, total: int) -> None:
        """Update tool counts."""
        self._running = running
        self._total = total
        self.refresh()

    def render(self) -> str:
        """Render the counter."""
        if self._running > 0:
            return f"[yellow]◐[/yellow] {self._running}/{self._total}"
        elif self._total > 0:
            return f"[green]●[/green] {self._total} tools"
        return ""


class HintBar(Widget):
    """Bottom hint bar with keyboard shortcuts."""

    def __init__(self) -> None:
        super().__init__(id="hint-bar")

    def compose(self) -> ComposeResult:
        """Compose the hint bar."""
        yield Static(
            "[dim]@file /command[/dim]  |  "
            "[cyan]Ctrl+K[/cyan]: 命令面板  |  "
            "[cyan]Ctrl+L[/cyan]: 清屏  |  "
            "[cyan]Ctrl+A[/cyan]: 工具  |  "
            "[cyan]Ctrl+E[/cyan]: 模式",
            markup=True,
        )