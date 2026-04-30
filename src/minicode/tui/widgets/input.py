"""Smart input widget with @file and /command completion for MiniCode TUI."""
from __future__ import annotations

import os
import glob as glob_module
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

from textual.app import ComposeResult
from textual.containers import Container
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, Static
from textual.keys import Keys

from minicode.tui.themes.dark import dark_theme as theme


@dataclass
class Completion:
    """Represents a completion suggestion."""
    type: str  # "file", "command", "skill"
    value: str
    display: str
    description: str = ""


@dataclass
class CompletionResult:
    """Result of completion selection."""
    completion: Completion
    start_pos: int
    end_pos: int


class InputArea(Widget):
    """Smart input area with @file and /command completion.

    Features:
    - @ prefix triggers file path completion
    - / prefix triggers command completion
    - Tab to cycle through completions
    - Enter to submit, Shift+Enter for new line
    - Up/Down for history navigation
    """

    class Submit(Message):
        """Posted when the user submits text."""
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(
        self,
        placeholder: str = "Type your message... (@file /command)",
        max_history: int = 100,
    ) -> None:
        super().__init__(id="input-container")
        self.placeholder = placeholder
        self.max_history = max_history

        # History
        self._history: list[str] = []
        self._history_index: int = -1

        # Completion
        self._completions: list[Completion] = []
        self._completion_index: int = -1
        self._completion_start: int = 0
        self._current_word_start: int = 0
        self._showing_completions: bool = False

        # Current text
        self._text: str = ""

        # Callbacks
        self._on_file_complete: Optional[Callable[[str], list[str]]] = None
        self._on_command_complete: Optional[Callable[[str], list[str]]] = None

        # Prompt mode
        self._mode: str = "default"  # default, user, tool

    @property
    def mode(self) -> str:
        """Get current mode."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        """Set current mode."""
        self._mode = value
        self._update_prompt_indicator()

    def set_file_completer(self, callback: Callable[[str], list[str]]) -> None:
        """Set file path completer callback.

        Args:
            callback: Function that takes a path prefix and returns list of matches
        """
        self._on_file_complete = callback

    def set_command_completer(self, callback: Callable[[str], list[str]]) -> None:
        """Set command completer callback.

        Args:
            callback: Function that takes a command prefix and returns list of commands
        """
        self._on_command_complete = callback

    def compose(self) -> ComposeResult:
        """Compose the input area."""
        with Container(id="input-row"):
            yield Static("[default]", id="prompt-indicator")
            yield Input(
                self.placeholder,
                id="command-input",
                classes="command-input",
            )
        # Completion popup (hidden by default)
        yield Static("", id="completion-popup", classes="completion-popup hidden")

    def on_mount(self) -> None:
        """Initialize input on mount."""
        self._update_prompt_indicator()
        input_widget = self.query_one("#command-input", Input)
        input_widget.focus()

    def _update_prompt_indicator(self) -> None:
        """Update the prompt indicator based on mode."""
        indicator = self.query_one("#prompt-indicator", Static)
        mode_colors = {
            "default": "default",
            "user": "user",
            "tool": "tool",
        }
        color = mode_colors.get(self._mode, "default")
        mode_text = self._mode.upper() if self._mode != "default" else "MINI"
        indicator.update(f"[{color}]{mode_text}[/{color}]")

    def _get_current_word(self) -> tuple[str, int, int]:
        """Get the current word at cursor.

        Returns:
            Tuple of (word, start_pos, end_pos)
        """
        input_widget = self.query_one("#command-input", Input)
        text = input_widget.value
        cursor = input_widget.cursor_position

        # Find word start
        start = cursor
        while start > 0 and text[start - 1] not in (" ", "\t", "@", "/"):
            start -= 1

        # Handle @ and / prefixes
        if start > 0 and text[start - 1] in ("@", "/"):
            prefix_char = text[start - 1]
        else:
            prefix_char = ""

        # Find word end
        end = cursor
        while end < len(text) and text[end] not in (" ", "\t"):
            end += 1

        word = text[start:end]
        return word, start, end

    def _get_completions(self, prefix: str) -> list[Completion]:
        """Get completions for the current prefix.

        Args:
            prefix: Current word prefix (may include @ or /)

        Returns:
            List of Completion objects
        """
        if not prefix:
            return []

        completions = []

        if prefix.startswith("@"):
            # File path completion
            path_prefix = prefix[1:]  # Remove @
            matches = self._get_file_completions(path_prefix)
            completions.extend(matches)
        elif prefix.startswith("/"):
            # Command completion
            cmd_prefix = prefix[1:]  # Remove /
            matches = self._get_command_completions(cmd_prefix)
            completions.extend(matches)

        return completions

    def _get_file_completions(self, path_prefix: str) -> list[Completion]:
        """Get file path completions.

        Args:
            path_prefix: Path prefix to match

        Returns:
            List of file Completion objects
        """
        completions = []

        # Use callback if provided
        if self._on_file_complete:
            matches = self._on_file_complete(path_prefix)
            for match in matches:
                completions.append(Completion(
                    type="file",
                    value=f"@{match}",
                    display=match,
                    description="File",
                ))
            return completions

        # Default: use glob
        try:
            # Determine search pattern
            if path_prefix:
                pattern = f"**/{path_prefix}*"
            else:
                pattern = "*"

            matches = []
            for match in glob_module.glob(pattern, recursive=True):
                if os.path.isfile(match):
                    matches.append(match)
                elif os.path.isdir(match):
                    matches.append(match + "/")

            for match in matches[:10]:
                # Truncate if too long
                display = match if len(match) <= 40 else "..." + match[-37:]
                completions.append(Completion(
                    type="file",
                    value=f"@{match}",
                    display=display,
                    description="File" if os.path.isfile(match) else "Directory",
                ))
        except Exception:
            pass

        return completions

    def _get_command_completions(self, cmd_prefix: str) -> list[Completion]:
        """Get command completions.

        Args:
            cmd_prefix: Command prefix to match

        Returns:
            List of command Completion objects
        """
        completions = []

        # Use callback if provided
        if self._on_command_complete:
            matches = self._on_command_complete(cmd_prefix)
            for match in matches:
                completions.append(Completion(
                    type="command",
                    value=match,
                    display=match,
                    description="Command",
                ))
            return completions

        # Default commands
        commands = [
            ("/help", "Show help information", "F1"),
            ("/clear", "Clear the screen", "Ctrl+L"),
            ("/status", "Show status", "F2"),
            ("/config", "Show configuration", ""),
            ("/session", "Show session info", ""),
            ("/memory", "Show saved memories", ""),
            ("/skills", "List available skills", ""),
            ("/context", "Show context info", ""),
            ("/stat", "Show statistics", ""),
            ("/compact", "Compact context", ""),
            ("/retry", "Retry last command", ""),
            ("/model", "Change model", ""),
            ("/provider", "Change provider", ""),
            ("/theme", "Change theme", ""),
            ("/export", "Export session", ""),
            ("/import", "Import session", ""),
            ("/tools", "List all tools", ""),
            ("/env", "Show environment", ""),
            ("/keys", "Manage API keys", ""),
            ("/permission", "Permission settings", ""),
            ("/mcp", "MCP server management", ""),
            ("/team", "Team collaboration", ""),
            ("/task", "Task management", ""),
            ("/cron", "Scheduled tasks", ""),
            ("/exit", "Exit the application", "Ctrl+C"),
            ("/quit", "Exit the application", "Ctrl+C"),
            ("/history", "Show command history", "Ctrl+R"),
        ]

        for cmd, desc, key in commands:
            if cmd.startswith(f"/{cmd_prefix}"):
                key_str = f" [{key}]" if key else ""
                completions.append(Completion(
                    type="command",
                    value=cmd,
                    display=cmd,
                    description=f"{desc}{key_str}",
                ))

        return completions

    def _show_completions(self) -> None:
        """Show completion popup with current completions."""
        if not self._completions:
            self._hide_completions()
            return

        self._showing_completions = True
        self._completion_index = 0

        popup = self.query_one("#completion-popup", Static)
        lines = []
        for i, comp in enumerate(self._completions[:10]):
            prefix = "> " if i == 0 else "  "
            type_icon = {
                "file": "[blue]@[/blue]",
                "command": "[cyan]/[/cyan]",
                "skill": "[purple]*[/purple]",
            }.get(comp.type, "")
            lines.append(f"{prefix}{type_icon} {comp.display} [dim]{comp.description}[/dim]")

        popup.update("\n".join(lines))
        popup.remove_class("hidden")

    def _hide_completions(self) -> None:
        """Hide the completion popup."""
        self._showing_completions = False
        popup = self.query_one("#completion-popup", Static)
        popup.add_class("hidden")

    def _select_completion(self, index: int) -> None:
        """Select a completion by index and apply it.

        Args:
            index: Index of completion to select
        """
        if not self._completions or index < 0 or index >= len(self._completions):
            return

        completion = self._completions[index]
        input_widget = self.query_one("#command-input", Input)
        text = input_widget.value

        # Replace current word with completion
        new_text = text[:self._completion_start] + completion.value + text[len(self._text):]
        input_widget.value = new_text

        # Move cursor to end of completion
        cursor_pos = self._completion_start + len(completion.value)
        input_widget.cursor_position = cursor_pos

        self._hide_completions()
        self._completions = []

    def _cycle_completion(self, forward: bool = True) -> None:
        """Cycle through completions.

        Args:
            forward: True to go forward, False to go backward
        """
        if not self._completions:
            return

        if forward:
            self._completion_index = (self._completion_index + 1) % len(self._completions)
        else:
            self._completion_index = (self._completion_index - 1) % len(self._completions)

        self._show_completions()

    def _on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        self._text = event.value

        # Check for completion triggers
        word, start, end = self._get_current_word()

        # Check for @ or / at the start
        if word and (word.startswith("@") or word.startswith("/")):
            self._current_word_start = start
            self._text = word
            self._completion_start = start
            self._completions = self._get_completions(word)
            if self._completions:
                self._show_completions()
            else:
                self._hide_completions()
        else:
            self._hide_completions()
            self._completions = []

    def _on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        text = event.value.strip()
        if not text:
            return

        # Add to history
        self._history.append(text)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
        self._history_index = len(self._history)

        # Hide completions
        self._hide_completions()

        # Post submit message
        self.post_message(self.Submit(text))

        # Clear input
        event.input.value = ""
        self._text = ""

    def _navigate_history(self, direction: int) -> None:
        """Navigate through command history.

        Args:
            direction: -1 for up (older), 1 for down (newer)
        """
        input_widget = self.query_one("#command-input", Input)

        if not self._history:
            return

        # Adjust index
        new_index = self._history_index + direction

        if new_index < 0:
            # At the beginning, can't go further
            return
        elif new_index >= len(self._history):
            # Past the end, clear input
            self._history_index = len(self._history)
            input_widget.value = ""
        else:
            # Show history item
            self._history_index = new_index
            input_widget.value = self._history[new_index]
            input_widget.cursor_position = len(input_widget.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changed event."""
        self._on_input_changed(event)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submitted event."""
        self._on_input_submitted(event)

    def on_key(self, event) -> None:
        """Handle key events for completion navigation."""
        input_widget = self.query_one("#command-input", Input)

        if self._showing_completions:
            if event.key == Keys.tab:
                event.prevent_default()
                self._cycle_completion(forward=True)
            elif event.key == Keys.shift_tab:
                event.prevent_default()
                self._cycle_completion(forward=False)
            elif event.key == Keys.enter and self._completions:
                event.prevent_default()
                self._select_completion(self._completion_index)
            elif event.key == Keys.escape:
                event.prevent_default()
                self._hide_completions()
            elif event.key == Keys.up:
                event.prevent_default()
                self._cycle_completion(forward=False)
            elif event.key == Keys.down:
                event.prevent_default()
                self._cycle_completion(forward=True)
        else:
            # History navigation when not showing completions
            if event.key == Keys.up and event.ctrl:
                event.prevent_default()
                self._navigate_history(direction=-1)
            elif event.key == Keys.down and event.ctrl:
                event.prevent_default()
                self._navigate_history(direction=1)


class InputFooter(Widget):
    """Footer with input hints and status."""

    def __init__(self) -> None:
        super().__init__(id="input-footer")

    def compose(self) -> ComposeResult:
        """Compose the footer."""
        yield Static(
            f"[dim]@file /command[/dim]  |  "
            f"[cyan]Ctrl+K[/cyan]: 命令面板  |  "
            f"[cyan]Ctrl+L[/cyan]: 清屏  |  "
            f"[cyan]Ctrl+R[/cyan]: 历史",
            markup=True,
            id="footer-hints",
        )
