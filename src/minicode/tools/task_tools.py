"""Task management tools (persistent)."""
import json
import uuid
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


class TaskManager:
    """Persistent task management."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path.cwd() / ".mini-agent-cli" / "tasks"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _task_file(self, task_id: str) -> Path:
        return self.storage_dir / f"task_{task_id}.json"

    def create(
        self,
        subject: str,
        description: str = "",
        owner: str = "",
    ) -> dict:
        """Create a new task."""
        task_id = str(uuid.uuid4())[:8]
        task = {
            "id": task_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "owner": owner,
            "blockedBy": [],
            "blocks": [],
        }
        self._task_file(task_id).write_text(json.dumps(task, indent=2), encoding="utf-8")
        return task

    def get(self, task_id: str) -> Optional[dict]:
        """Get task by ID."""
        fp = self._task_file(task_id)
        if not fp.exists():
            return None
        return json.loads(fp.read_text(encoding="utf-8"))

    def list_all(self) -> list[dict]:
        """List all tasks."""
        tasks = []
        for fp in self.storage_dir.glob("task_*.json"):
            try:
                tasks.append(json.loads(fp.read_text(encoding="utf-8")))
            except Exception:
                continue
        return sorted(tasks, key=lambda t: t.get("id", ""))

    def update(self, task_id: str, **kwargs) -> Optional[dict]:
        """Update task fields."""
        task = self.get(task_id)
        if not task:
            return None
        task.update(kwargs)
        self._task_file(task_id).write_text(json.dumps(task, indent=2), encoding="utf-8")
        return task

    def delete(self, task_id: str) -> bool:
        """Delete a task."""
        fp = self._task_file(task_id)
        if fp.exists():
            fp.unlink()
            return True
        return False

    def find_unclaimed(self) -> list[dict]:
        """Find tasks with no owner."""
        return [t for t in self.list_all() if not t.get("owner")]

    def claim(self, task_id: str, owner: str) -> bool:
        """Atomically claim a task."""
        task = self.get(task_id)
        if not task or task.get("owner"):
            return False
        task["owner"] = owner
        task["status"] = "in_progress"
        self._task_file(task_id).write_text(json.dumps(task, indent=2), encoding="utf-8")
        return True


# Global instance
_task_manager: Optional[TaskManager] = None


def get_task_manager(storage_dir: Optional[Path] = None) -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(storage_dir)
    return _task_manager


@tool
def task_create(subject: str, description: str = "") -> str:
    """Create a new task.

    Args:
        subject: Task subject/title
        description: Task description
    """
    mgr = get_task_manager()
    task = mgr.create(subject, description)
    return f"Created task {task['id']}: {subject}"


@tool
def task_list() -> str:
    """List all tasks."""
    mgr = get_task_manager()
    tasks = mgr.list_all()
    if not tasks:
        return "No tasks"
    lines = ["# Tasks"]
    for t in tasks:
        lines.append(f"- [{t['status']}] {t['id']}: {t['subject']} (owner: {t.get('owner') or 'unclaimed'})")
    return "\n".join(lines)


@tool
def task_update(task_id: str, status: Optional[str] = None) -> str:
    """Update task status.

    Args:
        task_id: Task ID
        status: New status (pending/in_progress/completed)
    """
    mgr = get_task_manager()
    task = mgr.update(task_id, status=status)
    if not task:
        return f"[Error]: Task {task_id} not found"
    return f"Updated task {task_id} to {status}"


@tool
def task_get(task_id: str) -> str:
    """Get task details.

    Args:
        task_id: Task ID
    """
    mgr = get_task_manager()
    task = mgr.get(task_id)
    if not task:
        return f"[Error]: Task {task_id} not found"
    return json.dumps(task, indent=2)


TASK_TOOLS = [task_create, task_list, task_update, task_get]