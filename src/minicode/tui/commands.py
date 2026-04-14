"""Textual TUI commands and keybindings."""
from textual.app import App, ComposeResult
from textual.command import Command, CommandHelp, Hit, Hits, Provider
from textual.keymap import Keymap
from textual.widgets import Header, Footer

from minicode.agent.runner import AgentRunner


class MiniCodeCommands(Provider):
    """Command provider for command palette."""

    def __init__(self, runner: AgentRunner, app: App):
        super().__init__()
        self.runner = runner
        self.app = app

    async def search(self, query: str) -> Hits:
        """Search for commands matching query."""
        commands = [
            ("help", "Show help information", "Show all available commands"),
            ("clear", "Clear the screen", "Clear message area"),
            ("history", "Show command history", "Show recent commands"),
            ("quit", "Exit the application", "Quit MiniCode"),
            ("exit", "Exit the application", "Quit MiniCode"),
            ("status", "Show status", "Show current status"),
            ("config", "Show configuration", "Show config settings"),
            ("session", "Show session info", "Show current session"),
            ("memory", "Show saved memories", "List saved memories"),
            ("skills", "List available skills", "Show available skills"),
            ("context", "Show context info", "Show context details"),
            ("stat", "Show statistics", "Show usage statistics"),
            ("compact", "Compact context", "Request context compaction"),
            ("retry", "Retry last command", "Retry the last command"),
            ("model", "Change model", "Change the AI model"),
            ("provider", "Change provider", "Change the model provider"),
            ("permission", "Manage permissions", "Manage permission settings"),
            ("theme", "Change theme", "Change color theme"),
            ("log", "Show logs", "Show application logs"),
            ("export", "Export session", "Export current session"),
            ("import", "Import session", "Import a session"),
        ]

        for name, description, detail in commands:
            if query.lower() in name.lower() or query.lower() in description.lower():
                yield Hit(
                    0,
                    name,
                    f"{description}\n{detail}",
                    self.app.action_run_command,
                )

    def action_run_command(self, command: str) -> None:
        """Run a command."""
        # This will be called when command is selected
        pass


def get_keybindings() -> dict:
    """Get keybindings configuration."""
    return {
        "ctrl+c": "quit",
        "ctrl+l": "clear_screen",
        "ctrl+r": "recall",
        "ctrl+k": "command_palette",
        "ctrl+z": "suspend",
        "ctrl+t": "new_tab",
        "up": "history_up",
        "down": "history_down",
        "tab": "complete",
        "escape": "cancel",
    }
