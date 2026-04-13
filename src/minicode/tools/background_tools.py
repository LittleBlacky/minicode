"""Background task tools."""
import json
import uuid
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


class BackgroundManager:
    """Manage background tasks."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path.cwd() / ".mini-agent-cli"
        self.pending_file = self.storage_dir / "background_pending.json"
        self.completed_dir = self.storage_dir / "background_completed"
        self.completed_dir.mkdir(parents=True, exist_ok=True)

    def _load_pending(self) -> list[dict]:
        if self.pending_file.exists():
            return json.loads(self.pending_file.read_text(encoding="utf-8"))
        return []

    def _save_pending(self, tasks: list[dict]) -> None:
        self.pending_file.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

    def run(self, command: str, description: str = "") -> dict:
        """Start a background task."""
        task_id = str(uuid.uuid4())[:8]
        task = {
            "id": task_id,
            "command": command,
            "description": description,
            "status": "pending",
        }
        tasks = self._load_pending()
        tasks.append(task)
        self._save_pending(tasks)
        return task

    def list_pending(self) -> list[dict]:
        """List pending background tasks."""
        return self._load_pending()

    def get_result(self, task_id: str) -> Optional[str]:
        """Get completed task result."""
        result_file = self.completed_dir / f"{task_id}.json"
        if result_file.exists():
            data = json.loads(result_file.read_text(encoding="utf-8"))
            return data.get("result", "")
        return None

    def complete(self, task_id: str, result: str) -> None:
        """Mark task as completed."""
        tasks = self._load_pending()
        tasks = [t for t in tasks if t["id"] != task_id]
        self._save_pending(tasks)
        self.completed_dir / f"{task_id}.json"
        result_file = self.completed_dir / f"{task_id}.json"
        result_file.write_text(json.dumps({"id": task_id, "result": result}), encoding="utf-8")


# Global instance
_bg_manager: Optional[BackgroundManager] = None


def get_background_manager(storage_dir: Optional[Path] = None) -> BackgroundManager:
    global _bg_manager
    if _bg_manager is None:
        _bg_manager = BackgroundManager(storage_dir)
    return _bg_manager


@tool
def background_run(command: str, description: str = "") -> str:
    """Start a background task.

    Args:
        command: Shell command to run
        description: Task description
    """
    mgr = get_background_manager()
    task = mgr.run(command, description)
    return f"Started background task {task['id']}: {description or command}"


@tool
def check_background() -> str:
    """Check pending background tasks."""
    mgr = get_background_manager()
    tasks = mgr.list_pending()
    if not tasks:
        return "No pending background tasks"
    lines = ["# Background Tasks"]
    for t in tasks:
        lines.append(f"- {t['id']}: {t['description'] or t['command']} [{t['status']}]")
    return "\n".join(lines)


@tool
def poll_background_results(task_id: str) -> str:
    """Poll result of a background task.

    Args:
        task_id: Task ID to poll
    """
    mgr = get_background_manager()
    result = mgr.get_result(task_id)
    if result is None:
        return f"Task {task_id} not completed yet"
    return f"Result: {result}"


BACKGROUND_TOOLS = [background_run, check_background, poll_background_results]
