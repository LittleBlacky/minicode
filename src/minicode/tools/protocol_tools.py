"""Permission and protocol tools."""
from typing import Optional

from langchain_core.tools import tool


class PermissionManager:
    """Manage command permissions."""

    def __init__(self):
        self._rules: list[dict] = []
        self._mode = "default"

    def set_mode(self, mode: str) -> None:
        """Set permission mode: default, bypass, ask."""
        self._mode = mode

    def add_rule(self, pattern: str, action: str) -> None:
        """Add a permission rule."""
        self._rules.append({"pattern": pattern, "action": action})

    def check(self, command: str) -> tuple[bool, str]:
        """Check if command is allowed."""
        if self._mode == "bypass":
            return True, "bypass"

        for rule in self._rules:
            if rule["pattern"] in command:
                return rule["action"] == "allow", rule["action"]

        return True, "default"

    def get_mode(self) -> str:
        return self._mode


# Global instance
_permission_manager: Optional[PermissionManager] = None


def get_permission_manager() -> PermissionManager:
    global _permission_manager
    if _permission_manager is None:
        _permission_manager = PermissionManager()
    return _permission_manager


@tool
def set_permission_mode(mode: str) -> str:
    """Set permission mode.

    Args:
        mode: "default", "bypass", or "ask"
    """
    mgr = get_permission_manager()
    mgr.set_mode(mode)
    return f"Permission mode set to {mode}"


@tool
def check_permission(command: str) -> str:
    """Check if a command is allowed.

    Args:
        command: Command to check
    """
    mgr = get_permission_manager()
    allowed, reason = mgr.check(command)
    if allowed:
        return f"Allowed ({reason})"
    return f"[Blocked] {reason}"


PERMISSION_TOOLS = [set_permission_mode, check_permission]


# Protocol tools - for shutdown and plan approval
@tool
def shutdown_request(reason: str = "") -> str:
    """Request session shutdown.

    Args:
        reason: Reason for shutdown (optional)
    """
    return f"SHUTDOWN_REQUEST:{reason}"


@tool
def plan_approval(approved: bool, notes: str = "") -> str:
    """Submit plan approval.

    Args:
        approved: True to approve, False to reject
        notes: Optional notes
    """
    return f"PLAN_APPROVAL:{approved}:{notes}"


PROTOCOL_TOOLS = [shutdown_request, plan_approval]
