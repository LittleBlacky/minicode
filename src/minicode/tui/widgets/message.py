"""Message bubble and list widgets for MiniCode TUI - with streaming support."""
from __future__ import annotations

import time
from typing import Optional
from dataclasses import dataclass, field

from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Label
from rich.text import Text
from rich.panel import Panel
from rich.box import ROUNDED

from minicode.tui.themes.dark import dark_theme as theme
from minicode.tui.render import render_markdown, highlight_code, render_content


@dataclass
class MessageData:
    """Represents a message."""
    content: str
    sender: str  # "user", "agent", "tool", "system"
    timestamp: float = field(default_factory=time.time)
    tool_name: Optional[str] = None
    tool_status: Optional[str] = None  # "running", "success", "error"
    streaming: bool = False


class MessageBubble(Widget):
    """A message bubble widget with rich text support and streaming.

    Features:
    - User messages: cyan border, right-aligned
    - Agent messages: green border, left-aligned
    - Tool messages: yellow border, dim background
    - Error messages: red border
    - Streaming animation support
    """

    class Clicked(Message):
        """Called when the message bubble is clicked."""
        def __init__(self, message_id: str) -> None:
            super().__init__()
            self.message_id = message_id

    def __init__(
        self,
        content: str,
        sender: str = "user",
        message_id: str = "",
        show_sender: bool = True,
        streaming: bool = False,
    ) -> None:
        super().__init__(classes=f"message-bubble {sender}")
        self.content = content
        self.sender = sender
        self.message_id = message_id or f"msg_{int(time.time() * 1000)}"
        self.show_sender = show_sender
        self.streaming = streaming
        self._full_content = content

    def compose(self) -> ComposeResult:
        """Compose the message bubble."""
        # Render content
        try:
            rendered = self._render_content()
        except Exception:
            rendered = Text(self.content)

        # Title based on sender
        title = self._get_title()

        yield Static(
            rendered,
            markup=True,
            id=f"bubble-content-{self.message_id}",
        )

        # Streaming indicator
        if self.streaming:
            yield Static("[yellow]...[/yellow]", id="streaming-indicator")

    def _get_title(self) -> str:
        """Get the title based on sender."""
        titles = {
            "user": "[cyan bold]You[/cyan bold]",
            "agent": "[green bold]MiniCode[/green bold]",
            "tool": f"[yellow bold]Tool[/yellow bold]",
            "system": "[dim bold]System[/dim bold]",
            "error": "[red bold]Error[/red bold]",
        }
        return titles.get(self.sender, "")

    def _render_content(self) -> Text:
        """Render message content with syntax highlighting."""
        content = self._full_content

        # Handle streaming - show partial content
        if self.streaming and len(content) > 0:
            # Show last 500 chars for streaming
            display = content[-500:] if len(content) > 500 else content
            if len(content) > 500:
                display = "...\n" + display
        else:
            display = content

        # Check for code blocks
        if "```" in display:
            return render_content(display)

        # Check for inline code
        if "`" in display:
            # Simple inline code handling
            parts = display.split("`")
            if len(parts) > 2:
                return render_markdown(display)

        # Pure markdown
        return render_markdown(display)

    def update_content(self, new_content: str) -> None:
        """Update the message content (for streaming).

        Args:
            new_content: New content to display
        """
        self._full_content = new_content
        content_widget = self.query_one(f"#bubble-content-{self.message_id}", Static)
        content_widget.update(self._render_content())

    def append_content(self, chunk: str) -> None:
        """Append content to the message (for streaming).

        Args:
            chunk: Text chunk to append
        """
        self._full_content += chunk
        self.update_content(self._full_content)

    def complete_streaming(self) -> None:
        """Mark streaming as complete."""
        self.streaming = False
        self.update_content(self._full_content)
        # Remove streaming indicator
        indicator = self.query_one("#streaming-indicator", Static)
        indicator.update("")

    def on_click(self) -> None:
        """Handle click event."""
        self.post_message(self.Clicked(self.message_id))


class MessageList(Widget):
    """A scrollable list of message bubbles with auto-scroll.

    Features:
    - Auto-scroll to bottom on new messages
    - Efficient rendering of messages
    - Message count tracking
    """

    def __init__(self, name: str = "messages") -> None:
        super().__init__(name=name, id="message-area")
        self._messages: list[MessageData] = []
        self._message_widgets: dict[str, MessageBubble] = {}
        self._auto_scroll: bool = True

    def compose(self) -> ComposeResult:
        """Compose the message list."""
        yield ScrollableContainer(id="message-scroll")
        yield Static("", id="message-list-inner")

    def on_mount(self) -> None:
        """Initialize on mount."""
        pass

    def add_message(
        self,
        content: str,
        sender: str = "user",
        message_id: str = "",
        streaming: bool = False,
    ) -> MessageBubble:
        """Add a message to the list.

        Args:
            content: Message content
            sender: Sender type ("user", "agent", "tool", "system")
            message_id: Unique message ID
            streaming: Whether this message is streaming

        Returns:
            The created MessageBubble widget
        """
        msg_data = MessageData(
            content=content,
            sender=sender,
            streaming=streaming,
        )
        self._messages.append(msg_data)

        # Create widget
        bubble = MessageBubble(
            content=content,
            sender=sender,
            message_id=msg_data.message_id if not message_id else message_id,
            streaming=streaming,
        )
        self._message_widgets[msg_data.message_id] = bubble

        # Update display
        self._update_display()

        # Auto-scroll
        if self._auto_scroll:
            self._scroll_to_bottom()

        return bubble

    def add_tool_message(
        self,
        tool_name: str,
        args: dict,
        result: str = "",
        status: str = "running",
    ) -> MessageBubble:
        """Add a tool call message.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool result (optional)
            status: Tool status ("running", "success", "error")

        Returns:
            The created MessageBubble widget
        """
        content = f"[cyan]{tool_name}[/cyan]"
        if args:
            content += f"\n[dim]{args}[/dim]"
        if result:
            content += f"\n\n{result}"

        bubble = self.add_message(
            content=content,
            sender="tool",
        )
        return bubble

    def update_message(self, message_id: str, content: str) -> None:
        """Update an existing message.

        Args:
            message_id: Message ID
            content: New content
        """
        # Find message data
        for msg in self._messages:
            if msg.message_id == message_id:
                msg.content = content
                break

        # Update widget
        if message_id in self._message_widgets:
            self._message_widgets[message_id].update_content(content)

    def append_to_message(self, message_id: str, chunk: str) -> None:
        """Append content to an existing message.

        Args:
            message_id: Message ID
            chunk: Text chunk to append
        """
        # Find message data
        for msg in self._messages:
            if msg.message_id == message_id:
                msg.content += chunk
                break

        # Update widget
        if message_id in self._message_widgets:
            self._message_widgets[message_id].append_content(chunk)

        # Auto-scroll
        if self._auto_scroll:
            self._scroll_to_bottom()

    def complete_message(self, message_id: str) -> None:
        """Mark a streaming message as complete.

        Args:
            message_id: Message ID
        """
        # Find message data
        for msg in self._messages:
            if msg.message_id == message_id:
                msg.streaming = False
                break

        # Update widget
        if message_id in self._message_widgets:
            self._message_widgets[message_id].complete_streaming()

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._message_widgets.clear()
        self._update_display()

    def _update_display(self) -> None:
        """Update the message list display."""
        content = self.query_one("#message-list-inner", Static)
        lines = []

        for msg in self._messages:
            # Create styled message block
            border_color = {
                "user": "cyan",
                "agent": "green",
                "tool": "yellow",
                "system": "dim",
                "error": "red",
            }.get(msg.sender, "white")

            title = {
                "user": "You",
                "agent": "MiniCode",
                "tool": f"Tool: {msg.tool_name or 'unknown'}",
                "system": "System",
                "error": "Error",
            }.get(msg.sender, msg.sender)

            # Format content
            display_content = msg.content
            if msg.streaming:
                # Add streaming indicator
                display_content += " [yellow]...[/yellow]"

            # Create panel
            lines.append(f"[{border_color} bold]{title}[/{border_color} bold]")
            lines.append(display_content)
            lines.append("")  # Empty line between messages

        content.update("\n".join(lines) if lines else "[dim]No messages yet[/dim]")

    def _scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the message list."""
        scroll = self.query_one("#message-scroll", ScrollableContainer)
        scroll.scroll_end(animate=True)

    def set_auto_scroll(self, enabled: bool) -> None:
        """Set auto-scroll behavior.

        Args:
            enabled: Whether to auto-scroll
        """
        self._auto_scroll = enabled


class ToolCallWidget(Static):
    """Display a tool call with its result.

    Features:
    - Tool name and arguments display
    - Status indicator (running/success/error)
    - Duration display
    - Expandable result view
    """

    def __init__(
        self,
        tool_name: str,
        args: str = "",
        result: str = "",
        status: str = "running",
        duration: float = 0,
    ) -> None:
        self.tool_name = tool_name
        self.args = args
        self.result = result
        self.status = status
        self.duration = duration

        super().__init__("", markup=True)

    def compose(self) -> ComposeResult:
        """Compose the tool call widget."""
        # Status icon
        status_icon = {
            "running": "[yellow]◐[/yellow]",
            "success": "[green]●[/green]",
            "error": "[red]✗[/red]",
            "pending": "[dim]○[/dim]",
        }.get(self.status, "[dim]○[/dim]")

        # Duration
        duration_str = ""
        if self.duration > 0:
            if self.duration < 1:
                duration_str = f"[dim]({int(self.duration * 1000)}ms)[/dim]"
            elif self.duration < 60:
                duration_str = f"[dim]({self.duration:.1f}s)[/dim]"
            else:
                mins = int(self.duration / 60)
                secs = int(self.duration % 60)
                duration_str = f"[dim]({mins}m {secs}s)[/dim]"

        # Content
        content = f"{status_icon} [cyan bold]{self.tool_name}[/cyan bold] {duration_str}\n"
        if self.args:
            content += f"[dim]{self.args}[/dim]\n"
        if self.result:
            content += f"\n{self.result}"

        yield Static(content, markup=True)


class StreamMessage(Widget):
    """A streaming message widget that updates in real-time."""

    def __init__(
        self,
        sender: str = "agent",
        message_id: str = "",
    ) -> None:
        super().__init__()
        self.sender = sender
        self.message_id = message_id or f"stream_{int(time.time() * 1000)}"
        self._content = ""
        self._chunks: list[str] = []

    def compose(self) -> ComposeResult:
        """Compose the streaming message."""
        yield Static("", id=f"stream-content-{self.message_id}", markup=True)
        yield Static("[yellow]...[/yellow]", id=f"stream-indicator-{self.message_id}")

    def append(self, chunk: str) -> None:
        """Append a chunk to the message.

        Args:
            chunk: Text chunk to append
        """
        self._chunks.append(chunk)
        self._content += chunk

        # Update display (show last 500 chars)
        display = self._content[-500:] if len(self._content) > 500 else self._content
        if len(self._content) > 500:
            display = "...\n" + display

        content_widget = self.query_one(f"#stream-content-{self.message_id}", Static)
        content_widget.update(render_content(display))

    def complete(self) -> None:
        """Mark the stream as complete."""
        indicator = self.query_one(f"#stream-indicator-{self.message_id}", Static)
        indicator.update("")

        # Update content to full content
        content_widget = self.query_one(f"#stream-content-{self.message_id}", Static)
        content_widget.update(render_content(self._content))

    def get_content(self) -> str:
        """Get the full message content."""
        return self._content
