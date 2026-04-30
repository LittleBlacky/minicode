"""MiniCode TUI widgets."""
# Existing widgets
from minicode.tui.components import StatusBar, CommandPalette, ToolCallLog, MessageBubble

# New widgets
from .header import TUIHeader
from .message import MessageList, MessageBubble as RichMessageBubble, StreamMessage
from .input import InputArea, InputFooter
from .status import TUIStatusBar
from .sidebar import ToolSidebar, ToolItemWidget, CommandPaletteWidget

__all__ = [
    # Existing
    "StatusBar",
    "CommandPalette",
    "ToolCallLog",
    "MessageBubble",
    # New
    "TUIHeader",
    "MessageList",
    "RichMessageBubble",
    "StreamMessage",
    "InputArea",
    "InputFooter",
    "TUIStatusBar",
    "ToolSidebar",
    "ToolItemWidget",
    "CommandPaletteWidget",
]