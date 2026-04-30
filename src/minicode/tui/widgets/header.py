"""Header widget for MiniCode TUI - Custom header with branding."""
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Header, Static
from textual.widget import Widget

from minicode.tui.themes.dark import dark_theme as theme


class TUIHeader(Widget):
    """Custom header with MiniCode branding.

    Shows:
    - MiniCode logo
    - Model name
    - Status indicator
    - Window controls (optional)
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-7",
        show_window_controls: bool = True,
    ) -> None:
        super().__init__(id="header")
        self.model_name = model_name
        self.show_window_controls = show_window_controls

    def compose(self) -> ComposeResult:
        """Compose the header."""
        with Horizontal(id="title-bar"):
            # Logo
            yield Static(
                f"[bold green]●[/bold green] [bold]MiniCode[/bold]",
                id="logo",
            )

            # Spacer
            yield Static("", classes="spacer")

            # Model name
            yield Static(
                f"[dim]{self.model_name}[/dim]",
                id="model-name",
            )

            # Status indicator
            yield Static(
                "[green]●[/green]",
                id="status-indicator",
            )

            # Window controls (if enabled)
            if self.show_window_controls:
                with Horizontal(id="window-controls"):
                    yield Static("─", classes="window-btn", id="btn-minimize")
                    yield Static("□", classes="window-btn", id="btn-maximize")
                    yield Static("×", classes="window-btn close", id="btn-close")

    def set_model_name(self, name: str) -> None:
        """Update the model name display."""
        model = self.query_one("#model-name", Static)
        model.update(f"[dim]{name}[/dim]")

    def set_status(self, status: str, color: str = "green") -> None:
        """Update the status indicator.

        Args:
            status: Status text
            color: Color for the indicator
        """
        indicator = self.query_one("#status-indicator", Static)
        indicator.update(f"[{color}]●[/{color}] {status}")


class TitleBar(Widget):
    """Alternative title bar component."""

    def __init__(self) -> None:
        super().__init__(id="title-bar")

    def compose(self) -> ComposeResult:
        """Compose the title bar."""
        yield Static(
            "[bold green]●[/bold green] [bold]MiniCode[/bold]",
            id="title",
        )
        yield Static("", classes="spacer", id="title-spacer")
        yield Static("[dim]Professional Coding Agent[/dim]", id="subtitle")


# Default header for easy import
default_header = TUIHeader