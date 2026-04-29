"""Context compression tools - Reduce conversation history size."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

WORKDIR = Path.cwd()
STORAGE_DIR = WORKDIR / ".mini-agent-cli"
TOOL_RESULTS_DIR = STORAGE_DIR / "task_outputs" / "tool-results"

# Limits
CONTEXT_LIMIT = 50000  # Max characters before compacting
KEEP_RECENT = 3  # Keep last N messages
PERSIST_THRESHOLD = 30000  # Save tool output if larger than this
PREVIEW_CHARS = 2000  # Preview length when persisting


def _count_messages(messages: list) -> int:
    """Count total message characters."""
    return sum(len(str(m.content)) for msg in messages if hasattr(msg, "content") for content in [msg.content] if content)


def _persist_tool_output(tool_id: str, output: str) -> str:
    """Persist large tool output to file."""
    if len(output) <= PERSIST_THRESHOLD:
        return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    f = TOOL_RESULTS_DIR / f"{tool_id}.txt"
    if not f.exists():
        f.write_text(output)
    return f"<persisted-output>\nFull output: {f.relative_to(WORKDIR)}\nPreview:\n{output[:PREVIEW_CHARS]}\n</persisted-output>"


def compact_messages(messages: list, keep_recent: int = KEEP_RECENT) -> list:
    """Compact conversation history by keeping recent messages and summarizing old ones.

    Args:
        messages: List of messages to compact
        keep_recent: Number of recent messages to keep intact

    Returns:
        Compacted message list with summary
    """
    if len(messages) <= keep_recent:
        return messages

    # Keep system message
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]

    # Keep recent user/AI messages
    recent = []
    recent_count = 0
    for msg in reversed(messages):
        if not isinstance(msg, (SystemMessage, ToolMessage)):
            recent.insert(0, msg)
            recent_count += 1
            if recent_count >= keep_recent:
                break

    # Summarize tool messages
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    tool_summary = f"[Previous tool calls: {len(tool_messages)} operations]"

    # Create summary of old messages
    old_count = len(messages) - len(system_messages) - len(recent) - len(tool_messages)
    if old_count > 0:
        summary_msg = AIMessage(
            content=f"[Context compressed: {old_count} previous messages summarized. {len(tool_messages)} tool operations were executed.]"
        )
    else:
        summary_msg = None

    # Build compacted list
    result = system_messages.copy() if system_messages else []
    if summary_msg:
        result.append(summary_msg)
    result.extend(tool_messages[:10])  # Keep last 10 tool messages
    result.extend(recent)

    return result


def get_context_size(messages: list) -> int:
    """Get current context size in characters."""
    return _count_messages(messages)


def should_compact(messages: list, limit: int = CONTEXT_LIMIT) -> bool:
    """Check if context should be compacted."""
    return _count_messages(messages) > limit


# LangChain tool for explicit compact command
from langchain_core.tools import tool


@tool
def compact_history(keep_recent: int = KEEP_RECENT) -> str:
    """Compress conversation history to save context space.

    Args:
        keep_recent: Number of recent messages to keep (default: 3)
    """
    # This is a placeholder - actual compaction happens in agent state
    return f"[Compact] Would compact history keeping {keep_recent} recent messages"


COMPACT_TOOLS = [compact_history]

__all__ = ["COMPACT_TOOLS", "compact_messages", "get_context_size", "should_compact"]