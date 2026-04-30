"""Dark professional theme for MiniCode TUI - GitHub Dark style."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class DarkTheme:
    """Dark professional theme colors - GitHub Dark inspired."""

    # Background colors
    BACKGROUND: ClassVar[str] = "#0d1117"
    SURFACE: ClassVar[str] = "#161b22"
    SURFACE_HOVER: ClassVar[str] = "#21262d"
    SURFACE_ACTIVE: ClassVar[str] = "#30363d"
    BORDER: ClassVar[str] = "#30363d"
    BORDER_MUTED: ClassVar[str] = "#21262d"

    # Text colors
    TEXT: ClassVar[str] = "#e6edf3"
    TEXT_MUTED: ClassVar[str] = "#8b949e"
    TEXT_DIM: ClassVar[str] = "#484f58"
    TEXT_INVERSE: ClassVar[str] = "#0d1117"

    # Accent colors
    PRIMARY: ClassVar[str] = "#58a6ff"
    PRIMARY_HOVER: ClassVar[str] = "#79c0ff"
    SUCCESS: ClassVar[str] = "#3fb950"
    SUCCESS_HOVER: ClassVar[str] = "#56d364"
    WARNING: ClassVar[str] = "#d29922"
    WARNING_HOVER: ClassVar[str] = "#e3b341"
    ERROR: ClassVar[str] = "#f85149"
    ERROR_HOVER: ClassVar[str] = "#ff7b72"
    ACCENT: ClassVar[str] = "#bc8cff"
    ACCENT_HOVER: ClassVar[str] = "#d2a8ff"

    # Message colors
    USER_BORDER: ClassVar[str] = "#388bfd"
    USER_BG: ClassVar[str] = "#161b22"
    AGENT_BORDER: ClassVar[str] = "#3fb950"
    AGENT_BG: ClassVar[str] = "#161b22"
    TOOL_BORDER: ClassVar[str] = "#d29922"
    TOOL_BG: ClassVar[str] = "#21262d"
    ERROR_BG: ClassVar[str] = "#3d1f1f"

    # Code block colors
    CODE_BG: ClassVar[str] = "#0d1117"
    CODE_BORDER: ClassVar[str] = "#30363d"
    CODE_TEXT: ClassVar[str] = "#e6edf3"

    # Scrollbar colors
    SCROLLBAR_BG: ClassVar[str] = "#21262d"
    SCROLLBAR_COLOR: ClassVar[str] = "#484f58"

    # Selection colors
    SELECTION_BG: ClassVar[str] = "#388bfd40"
    SELECTION_TEXT: ClassVar[str] = "#e6edf3"

    # Focus colors
    FOCUS: ClassVar[str] = "#58a6ff"
    FOCUS_RING: ClassVar[str] = "#58a6ff40"

    # Cursor colors
    CURSOR: ClassVar[str] = "#58a6ff"
    CURSOR_BG: ClassVar[str] = "#58a6ff20"

    # Prompt colors
    PROMPT_TEXT: ClassVar[str] = "#3fb950"
    PROMPT_USER: ClassVar[str] = "#388bfd"
    PROMPT_TOOL: ClassVar[str] = "#d29922"

    # Status indicator colors
    STATUS_ONLINE: ClassVar[str] = "#3fb950"
    STATUS_OFFLINE: ClassVar[str] = "#f85149"
    STATUS_BUSY: ClassVar[str] = "#d29922"

    # Header colors
    HEADER_BG: ClassVar[str] = "#161b22"
    HEADER_TEXT: ClassVar[str] = "#e6edf3"
    HEADER_BORDER: ClassVar[str] = "#30363d"

    # Footer colors
    FOOTER_BG: ClassVar[str] = "#161b22"
    FOOTER_TEXT: ClassVar[str] = "#8b949e"

    # Sidebar colors
    SIDEBAR_BG: ClassVar[str] = "#0d1117"
    SIDEBAR_BORDER: ClassVar[str] = "#30363d"

    # Markdown heading colors
    H1: ClassVar[str] = "#e6edf3"
    H2: ClassVar[str] = "#e6edf3"
    H3: ClassVar[str] = "#e6edf3"
    H4: ClassVar[str] = "#e6edf3"
    H5: ClassVar[str] = "#e6edf3"
    H6: ClassVar[str] = "#8b949e"

    # Markdown link colors
    LINK: ClassVar[str] = "#58a6ff"
    LINK_HOVER: ClassVar[str] = "#79c0ff"

    # Markdown code colors
    INLINE_CODE_BG: ClassVar[str] = "#388bfd20"
    INLINE_CODE_TEXT: ClassVar[str] = "#ff7b72"

    # Markdown blockquote
    BLOCKQUOTE_BORDER: ClassVar[str] = "#30363d"
    BLOCKQUOTE_TEXT: ClassVar[str] = "#8b949e"

    # Markdown list markers
    LIST_MARKER: ClassVar[str] = "#8b949e"

    # Markdown horizontal rule
    HR: ClassVar[str] = "#30363d"

    # Markdown table
    TABLE_BORDER: ClassVar[str] = "#30363d"
    TABLE_HEADER_BG: ClassVar[str] = "#161b22"
    TABLE_ALT_ROW_BG: ClassVar[str] = "#21262d"


# Global dark theme instance
dark_theme = DarkTheme()


# Helper function to get color with optional transparency
def with_alpha(color: str, alpha: float) -> str:
    """Add alpha transparency to a hex color."""
    if color.startswith("#") and len(color) == 7:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        return f"#{r:02x}{g:02x}{b:02x}{int(alpha * 255):02x}"
    return color


# Aliases for common use cases
THEME = dark_theme
