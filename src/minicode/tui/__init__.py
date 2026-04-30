"""Textual TUI module for MiniCode."""
from minicode.tui.app import MiniCodeTUI, run_tui, main
from minicode.tui.ascii_art import ASCIIArt, CatAnimator, StatusAnimation
from minicode.tui.components import StatusBar, CommandPalette, ToolCallLog, MessageBubble
from minicode.tui.commands import (
    MiniCodeCommands,
    get_keybindings,
    get_command_list,
    get_command_categories,
    format_command_help,
)
from minicode.tui.render import (
    extract_code_blocks,
    highlight_code,
    render_markdown,
    render_message,
    render_content,
    render_tool_call,
    render_error,
    render_file_preview,
    to_ansi,
)
from minicode.tui.themes import DarkTheme, dark_theme, DARK_CSS, get_theme_css

__all__ = [
    # App
    "MiniCodeTUI",
    "run_tui",
    "main",
    # ASCII Art
    "ASCIIArt",
    "CatAnimator",
    "StatusAnimation",
    # Components
    "StatusBar",
    "CommandPalette",
    "ToolCallLog",
    "MessageBubble",
    # Commands
    "MiniCodeCommands",
    "get_keybindings",
    "get_command_list",
    "get_command_categories",
    "format_command_help",
    # Render
    "extract_code_blocks",
    "highlight_code",
    "render_markdown",
    "render_message",
    "render_content",
    "render_tool_call",
    "render_error",
    "render_file_preview",
    "to_ansi",
    # Theme
    "DarkTheme",
    "dark_theme",
    "DARK_CSS",
    "get_theme_css",
]