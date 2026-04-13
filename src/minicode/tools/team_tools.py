"""Team collaboration tools."""
import json
import uuid
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


class MessageBus:
    """Simple message bus for inter-agent communication."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path.cwd() / ".mini-agent-cli" / "team"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.inbox_file = self.storage_dir / "inbox.json"

    def _load_inbox(self) -> dict:
        if self.inbox_file.exists():
            return json.loads(self.inbox_file.read_text(encoding="utf-8"))
        return {}

    def _save_inbox(self, inbox: dict) -> None:
        self.inbox_file.write_text(json.dumps(inbox, indent=2), encoding="utf-8")

    def send(self, to: str, message: str, from_: str = "main") -> str:
        """Send message to an agent."""
        inbox = self._load_inbox()
        if to not in inbox:
            inbox[to] = []
        inbox[to].append({
            "id": str(uuid.uuid4()),
            "from": from_,
            "message": message,
        })
        self._save_inbox(inbox)
        return f"Message sent to {to}"

    def read_inbox(self, agent_name: str) -> str:
        """Read messages for an agent."""
        inbox = self._load_inbox()
        messages = inbox.get(agent_name, [])
        if not messages:
            return "No messages"
        lines = [f"# Inbox for {agent_name}", ""]
        for msg in messages:
            lines.append(f"- From {msg['from']}: {msg['message']}")
        return "\n".join(lines)

    def clear_inbox(self, agent_name: str) -> str:
        """Clear inbox for an agent."""
        inbox = self._load_inbox()
        inbox[agent_name] = []
        self._save_inbox(inbox)
        return f"Cleared inbox for {agent_name}"


class TeammateManager:
    """Manage teammate agents."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path.cwd() / ".mini-agent-cli" / "team"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.storage_dir / "teammates.json"
        self.bus = MessageBus(storage_dir)

    def _load_teammates(self) -> dict:
        if self.config_file.exists():
            return json.loads(self.config_file.read_text(encoding="utf-8"))
        return {}

    def _save_teammates(self, data: dict) -> None:
        self.config_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def spawn(self, name: str, role: str, task: str) -> dict:
        """Spawn a new teammate."""
        teammates = self._load_teammates()
        teammates[name] = {
            "name": name,
            "role": role,
            "task": task,
            "status": "idle",
        }
        self._save_teammates(teammates)
        return teammates[name]

    def list_teammates(self) -> list[dict]:
        """List all teammates."""
        return list(self._load_teammates().values())

    def get_teammate(self, name: str) -> Optional[dict]:
        """Get teammate by name."""
        return self._load_teammates().get(name)


# Global instances
_message_bus: Optional[MessageBus] = None
_teammate_manager: Optional[TeammateManager] = None


def get_message_bus(storage_dir: Optional[Path] = None) -> MessageBus:
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus(storage_dir)
    return _message_bus


def get_teammate_manager(storage_dir: Optional[Path] = None) -> TeammateManager:
    global _teammate_manager
    if _teammate_manager is None:
        _teammate_manager = TeammateManager(storage_dir)
    return _teammate_manager


@tool
def spawn_teammate(name: str, role: str, task: str) -> str:
    """Spawn a new teammate agent.

    Args:
        name: Teammate name
        role: Teammate role
        task: Initial task for teammate
    """
    mgr = get_teammate_manager()
    tm = mgr.spawn(name, role, task)
    return f"Spawned teammate {name} with role {role}"


@tool
def list_teammates() -> str:
    """List all teammate agents."""
    mgr = get_teammate_manager()
    teammates = mgr.list_teammates()
    if not teammates:
        return "No teammates"
    lines = ["# Teammates"]
    for tm in teammates:
        lines.append(f"- {tm['name']}: {tm['role']} ({tm['status']})")
    return "\n".join(lines)


@tool
def send_message(to: str, message: str) -> str:
    """Send message to a teammate.

    Args:
        to: Teammate name
        message: Message content
    """
    bus = get_message_bus()
    return bus.send(to, message)


@tool
def read_inbox(agent_name: str = "main") -> str:
    """Read messages in inbox.

    Args:
        agent_name: Agent name (default: main)
    """
    bus = get_message_bus()
    return bus.read_inbox(agent_name)


TEAM_TOOLS = [spawn_teammate, list_teammates, send_message, read_inbox]
