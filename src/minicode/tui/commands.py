"""Textual TUI commands and keybindings for MiniCode."""
from typing import Optional
from textual.app import App, ComposeResult
from textual.command import Command, Hit, Hits, Provider
from textual.widgets import Header, Footer
from textual.message import Message

from minicode.agent.runner import AgentRunner


# All available commands with descriptions and shortcuts
COMMANDS = {
    # Navigation
    "/help": ("Show help information", "F1"),
    "/clear": ("Clear the screen", "Ctrl+L"),
    "/status": ("Show status", "F2"),
    "/config": ("Show configuration", ""),
    "/session": ("Show session info", "F4"),
    "/exit": ("Exit the application", "Ctrl+C"),
    "/quit": ("Exit the application", "Ctrl+C"),

    # Context
    "/memory": ("Show saved memories", ""),
    "/skills": ("List available skills", ""),
    "/context": ("Show context info", ""),
    "/compact": ("Compact context", ""),
    "/history": ("Show command history", "Ctrl+R"),

    # Statistics
    "/stat": ("Show statistics", ""),
    "/time": ("Show current time", ""),
    "/uptime": ("Show uptime", ""),

    # Model
    "/model": ("Change model", ""),
    "/provider": ("Change provider", ""),
    "/theme": ("Change theme", ""),
    "/keys": ("Manage API keys", ""),

    # Tools
    "/tools": ("List all tools", ""),
    "/env": ("Show environment", ""),

    # Session
    "/retry": ("Retry last command", ""),
    "/export": ("Export session", ""),
    "/import": ("Import session", ""),

    # Mode
    "/mode": ("Change mode (default/auto/plan)", ""),

    # MCP
    "/mcp": ("MCP server management", ""),
    "/mcp-connect": ("Connect to MCP server", ""),
    "/mcp-list": ("List MCP servers", ""),
    "/mcp-refresh": ("Refresh MCP tools", ""),

    # Agent
    "/team": ("Team collaboration", ""),
    "/task": ("Task management", ""),
    "/cron": ("Scheduled tasks", ""),
    "/hooks": ("Manage hooks", ""),

    # Permission
    "/permission": ("Permission settings", ""),
    "/permission-mode": ("Set permission mode", ""),

    # Fun
    "/cat": ("Show cat animation", ""),
    "/purr": ("Make the cat purr", ""),
}


class MiniCodeCommands(Provider):
    """Command provider for command palette.

    Provides fuzzy search through all available commands.
    """

    def __init__(self, runner: AgentRunner, app: App):
        super().__init__()
        self.runner = runner
        self.app = app

    async def search(self, query: str) -> Hits:
        """Search for commands matching query.

        Args:
            query: Search query

        Yields:
            Hits matching the query
        """
        if not query:
            # Show all commands when no query
            for name, (desc, key) in COMMANDS.items():
                key_str = f" [{key}]" if key else ""
                yield Hit(
                    0,
                    Command(f"{name} - {desc}{key_str}", desc),
                    self._run_command,
                )
            return

        # Fuzzy search
        query_lower = query.lower()
        matches = []

        for name, (desc, key) in COMMANDS.items():
            name_lower = name.lower()
            desc_lower = desc.lower()

            # Calculate match score
            score = 0
            if name_lower == query_lower:
                score = 100  # Exact match
            elif name_lower.startswith(query_lower):
                score = 80  # Prefix match
            elif query_lower in name_lower:
                score = 60  # Contains match
            elif query_lower in desc_lower:
                score = 40  # Description contains match
            elif self._fuzzy_match(query_lower, name_lower):
                score = 20  # Fuzzy match

            if score > 0:
                matches.append((score, name, desc, key))

        # Sort by score (descending)
        matches.sort(key=lambda x: x[0], reverse=True)

        for score, name, desc, key in matches:
            key_str = f" [{key}]" if key else ""
            yield Hit(
                score,
                Command(f"{name} - {desc}{key_str}", desc),
                self._run_command,
            )

    def _fuzzy_match(self, query: str, text: str) -> bool:
        """Simple fuzzy matching.

        Args:
            query: Query string
            text: Text to match against

        Returns:
            True if query matches text fuzzy
        """
        qi = 0
        for char in text:
            if qi < len(query) and char == query[qi]:
                qi += 1
        return qi == len(query)

    def _run_command(self, command: str) -> None:
        """Run a command.

        Args:
            command: Command string (with description)
        """
        # Extract command name from full string
        cmd_name = command.split(" - ")[0] if " - " in command else command

        # Create and post the command event
        self.app.post_message(CommandExecuted(cmd_name))


class CommandExecuted(Message):
    """Message posted when a command is executed from the palette."""

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command


def get_keybindings() -> dict:
    """Get keybindings configuration.

    Returns:
        Dict of key -> action mappings
    """
    return {
        # Core
        "ctrl+c": "quit",
        "ctrl+l": "clear_screen",
        "ctrl+r": "recall",
        "ctrl+k": "toggle_command_palette",
        "ctrl+z": "suspend",

        # Navigation
        "up": "history_up",
        "down": "history_down",

        # Tool panel
        "ctrl+a": "toggle_sidebar",
        "ctrl+t": "toggle_sidebar",

        # Mode
        "ctrl+e": "toggle_mode",

        # Help
        "f1": "show_help",
        "f2": "show_status",
        "f3": "show_history",
        "f4": "show_session",
    }


def get_command_list() -> list[tuple[str, str, str]]:
    """Get list of all commands.

    Returns:
        List of (name, description, shortcut) tuples
    """
    return [(name, desc, key) for name, (desc, key) in COMMANDS.items()]


def get_command_categories() -> dict:
    """Get commands organized by category.

    Returns:
        Dict of category -> list of commands
    """
    return {
        "Navigation": ["/help", "/clear", "/status", "/config", "/session", "/exit", "/quit"],
        "Context": ["/memory", "/skills", "/context", "/compact", "/history"],
        "Statistics": ["/stat", "/time", "/uptime"],
        "Model": ["/model", "/provider", "/theme", "/keys"],
        "Tools": ["/tools", "/env"],
        "Session": ["/retry", "/export", "/import"],
        "Mode": ["/mode"],
        "MCP": ["/mcp", "/mcp-connect", "/mcp-list", "/mcp-refresh"],
        "Agent": ["/team", "/task", "/cron", "/hooks"],
        "Permission": ["/permission", "/permission-mode"],
        "Fun": ["/cat", "/purr"],
    }


def format_command_help() -> str:
    """Format all commands as help text.

    Returns:
        Formatted help string
    """
    lines = ["[bold cyan]MiniCode Commands[/bold cyan]\n"]
    categories = get_command_categories()

    for category, commands in categories.items():
        lines.append(f"[bold]{category}:[/bold]")
        for cmd in commands:
            if cmd in COMMANDS:
                desc, key = COMMANDS[cmd]
                key_str = f" [dim]({key})[/dim]" if key else ""
                lines.append(f"  [cyan]{cmd}[/cyan] - {desc}{key_str}")
        lines.append("")

    return "\n".join(lines)
