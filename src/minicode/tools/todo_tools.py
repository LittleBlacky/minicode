"""Todo management tools."""
from typing import Optional

from langchain_core.tools import tool


class TodoTools:
    """Todo management utilities."""

    def __init__(self):
        self._items: list[dict] = []

    def update_todo(
        self,
        content: Optional[str] = None,
        status: Optional[str] = None,
        activeForm: Optional[str] = None,
    ) -> str:
        """Update todo list."""
        items = self._items
        has_progress = any(i.get("status") == "in_progress" for i in items)

        if content:
            if not has_progress:
                self._items.append({
                    "content": content,
                    "status": status or "in_progress",
                    "activeForm": activeForm or "",
                })
            else:
                return "[Error]: Already have an in-progress task"

        # Return current todo state
        if not items:
            return "No todo items"
        lines = ["# Todo"]
        for item in items:
            status_icon = "☐" if item.get("status") == "pending" else "◐" if item.get("status") == "in_progress" else "✓"
            lines.append(f"{status_icon} {item.get('content', '')}")
        return "\n".join(lines)

    def get_items(self) -> list[dict]:
        return self._items


# Global instance
_todo_tools: Optional[TodoTools] = None


def get_todo_tools() -> TodoTools:
    global _todo_tools
    if _todo_tools is None:
        _todo_tools = TodoTools()
    return _todo_tools


@tool
def update_todo(
    content: Optional[str] = None,
    status: Optional[str] = None,
    activeForm: Optional[str] = None,
) -> str:
    """Update todo list.

    Args:
        content: Todo content
        status: "pending" | "in_progress" | "completed"
        activeForm: Active form description
    """
    tools = get_todo_tools()
    return tools.update_todo(content, status, activeForm)


TODO_TOOLS = [update_todo]