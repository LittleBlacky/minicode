"""Scheduled task (cron) tools."""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


class CronScheduler:
    """Simple cron-like scheduler for tasks."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path.cwd() / ".mini-agent-cli"
        self.schedules_file = self.storage_dir / "cron_schedules.json"
        self.notifications_file = self.storage_dir / "cron_notifications.json"

    def _load_schedules(self) -> list[dict]:
        if self.schedules_file.exists():
            return json.loads(self.schedules_file.read_text(encoding="utf-8"))
        return []

    def _save_schedules(self, schedules: list[dict]) -> None:
        self.schedules_file.write_text(json.dumps(schedules, indent=2), encoding="utf-8")

    def create(self, cron_expr: str, prompt: str, recurring: bool = True) -> dict:
        """Create a scheduled task."""
        schedule_id = str(uuid.uuid4())[:8]
        schedule = {
            "id": schedule_id,
            "cron": cron_expr,
            "prompt": prompt,
            "recurring": recurring,
            "created": datetime.now().isoformat(),
            "last_fired": None,
        }
        schedules = self._load_schedules()
        schedules.append(schedule)
        self._save_schedules(schedules)
        return schedule

    def list(self) -> list[dict]:
        """List all schedules."""
        return self._load_schedules()

    def delete(self, schedule_id: str) -> bool:
        """Delete a schedule."""
        schedules = self._load_schedules()
        original = len(schedules)
        schedules = [s for s in schedules if s["id"] != schedule_id]
        if len(schedules) < original:
            self._save_schedules(schedules)
            return True
        return False

    def add_notification(self, schedule_id: str, message: str) -> None:
        """Add a notification for a schedule."""
        notif_file = self.notifications_file
        notifications = []
        if notif_file.exists():
            notifications = json.loads(notif_file.read_text(encoding="utf-8"))
        notifications.append({
            "schedule_id": schedule_id,
            "message": message,
            "time": datetime.now().isoformat(),
        })
        notif_file.write_text(json.dumps(notifications, indent=2), encoding="utf-8")

    def get_notifications(self) -> list[dict]:
        """Get pending notifications."""
        if self.notifications_file.exists():
            return json.loads(self.notifications_file.read_text(encoding="utf-8"))
        return []


# Global instance
_cron_scheduler: Optional[CronScheduler] = None


def get_cron_scheduler(storage_dir: Optional[Path] = None) -> CronScheduler:
    global _cron_scheduler
    if _cron_scheduler is None:
        _cron_scheduler = CronScheduler(storage_dir)
    return _cron_scheduler


@tool
def cron_create(cron_expr: str, prompt: str, recurring: bool = True) -> str:
    """Create a scheduled task.

    Args:
        cron_expr: Cron expression (e.g., "0 9 * * *")
        prompt: Prompt to execute
        recurring: True for recurring, False for one-shot
    """
    scheduler = get_cron_scheduler()
    schedule = scheduler.create(cron_expr, prompt, recurring)
    return f"Created schedule {schedule['id']}: {cron_expr}"


@tool
def cron_list() -> str:
    """List all scheduled tasks."""
    scheduler = get_cron_scheduler()
    schedules = scheduler.list()
    if not schedules:
        return "No scheduled tasks"
    lines = ["# Scheduled Tasks"]
    for s in schedules:
        lines.append(f"- {s['id']}: {s['cron']} ({'recurring' if s['recurring'] else 'one-shot'})")
    return "\n".join(lines)


@tool
def cron_delete(schedule_id: str) -> str:
    """Delete a scheduled task.

    Args:
        schedule_id: Schedule ID to delete
    """
    scheduler = get_cron_scheduler()
    if scheduler.delete(schedule_id):
        return f"Deleted schedule {schedule_id}"
    return f"[Error]: Schedule {schedule_id} not found"


CRON_TOOLS = [cron_create, cron_list, cron_delete]
