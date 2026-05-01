"""Permission management for bash commands - YAML-based configuration."""
from typing import Optional

from minicode.tools.permission_config import (
    get_permission_config,
    reset_permission_config,
    BUILTIN_DANGEROUS_PATTERNS,
    PermissionConfig,
)


# Re-export for backward compatibility
BASH_DANGEROUS_PATTERNS = BUILTIN_DANGEROUS_PATTERNS


class BashSecurityValidator:
    """Validate bash commands for security risks.

    Uses PermissionConfig for pattern matching.
    """

    def __init__(self, config: Optional[PermissionConfig] = None):
        self._config = config

    @property
    def config(self) -> PermissionConfig:
        """Lazy load config."""
        if self._config is None:
            self._config = get_permission_config()
        return self._config

    def validate(self, command: str) -> list[tuple[str, str]]:
        """Validate a command and return list of violations."""
        violations = []
        for name, pattern, risk, desc in BUILTIN_DANGEROUS_PATTERNS:
            import re
            if re.search(pattern, command):
                violations.append((name, desc))
        return violations

    def is_safe(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute."""
        allowed, reason, _, _ = self.config.check(command)
        return allowed, reason

    def get_risk_level(self, command: str) -> str:
        """Get risk level: none, low, medium, high, critical."""
        _, _, risk, _ = self.config.check(command)
        return risk

    def describe_failures(self, command: str) -> str:
        """Get description of all failures."""
        violations = self.validate(command)
        if not violations:
            return "No issues detected"
        return "; ".join(v[1] for v in violations)


# Global validator instance
bash_validator = BashSecurityValidator()


# Permission mode (for backward compatibility)
_permission_mode = "deny"  # Default to deny mode for security


def set_permission_mode(mode: str) -> None:
    """Set permission mode: 'allow', 'deny', 'prompt'."""
    global _permission_mode
    if mode in ("allow", "deny", "prompt"):
        _permission_mode = mode


def get_permission_mode() -> str:
    """Get current permission mode."""
    return _permission_mode


def check_permission(command: str, tool_name: str = "bash_tool") -> tuple[bool, str]:
    """Check if a command is allowed to run.

    Args:
        command: The command to check
        tool_name: The tool name (default: bash_tool)

    Returns:
        tuple of (allowed, reason)
    """
    if tool_name != "bash_tool":
        return True, ""

    # Check using PermissionConfig
    config = get_permission_config()
    allowed, reason, risk, _ = config.check(command)

    if not allowed:
        return False, reason

    # Additional check based on permission mode
    if _permission_mode == "deny":
        return False, "Permission mode is 'deny'"

    if _permission_mode == "prompt" and risk in ("high", "critical"):
        return False, f"Requires confirmation: {reason}"

    return True, ""


def get_permission_rules() -> dict:
    """Get current permission rules summary."""
    config = get_permission_config()
    return {
        "mode": _permission_mode,
        "config": config.get_config_summary(),
        "builtin_patterns": config.get_builtin_patterns(),
    }


# LangChain tools for registry
from langchain_core.tools import tool


@tool
def set_mode(mode: str) -> str:
    """Set permission mode: 'allow', 'deny', 'prompt'."""
    set_permission_mode(mode)
    return f"Permission mode set to: {mode}"


@tool
def check_bash_permission(command: str) -> str:
    """Check if a bash command is safe to run.

    Uses the YAML permission configuration to determine if the command
    is allowed or blocked.
    """
    allowed, reason = check_permission(command)
    if allowed:
        return f"[OK] Command is safe to execute"
    return f"[BLOCKED] {reason}"


@tool
def reload_permissions() -> str:
    """Reload permission configuration from .minicode/permissions.yaml."""
    reset_permission_config()
    return "Permission configuration reloaded"


@tool
def show_permission_rules() -> str:
    """Show current permission rules and configuration."""
    config = get_permission_config()
    summary = config.get_config_summary()

    lines = [
        "# Permission Configuration",
        f"Config file: {summary['config_path']}",
        f"Loaded: {summary['loaded']}",
        "",
        f"User allow patterns: {summary['allow_patterns']}",
        f"User deny patterns: {summary['deny_patterns']}",
        f"Session patterns: {summary['session_patterns']} (选项 a, 会话结束失效)",
        f"Prompt unknown: {summary['prompt_unknown']}",
        f"Prompt threshold: {summary['prompt_threshold']}",
        "",
        "# Built-in Dangerous Patterns",
    ]

    for pattern in config.get_builtin_patterns():
        lines.append(f"- [{pattern['risk']}] {pattern['name']}: {pattern['description']}")

    return "\n".join(lines)


@tool
def add_session_allow(command: str) -> str:
    """Allow all variants of current command type (选项 a).

    Extracts the command type from the input and adds it to session patterns.
    All commands matching this pattern will be allowed without prompting.

    Example:
        add_session_allow("rm -rf /tmp/test") -> adds "rm -rf" pattern
        After this, "rm -rf /home/cleanup" will also be allowed
    """
    config = get_permission_config()
    pattern = config.add_session_pattern(command)
    return f"Added session pattern: {pattern}"


@tool
def list_session_patterns() -> str:
    """List all current session allow patterns (选项 a)."""
    config = get_permission_config()
    patterns = config.get_session_patterns()
    if not patterns:
        return "No session patterns (use 选项 a to add)"
    return "Session patterns:\n" + "\n".join(f"- {p}" for p in patterns)


@tool
def clear_session_patterns() -> str:
    """Clear all session allow patterns."""
    config = get_permission_config()
    count = len(config.get_session_patterns())
    config.clear_session_patterns()
    return f"Cleared {count} session pattern(s)"


@tool
def add_permanent_deny(command: str) -> str:
    """Add command to permanent deny list (选项 d).

    Extracts the command type and permanently blocks all variants.
    Persists to .minicode/permissions.yaml.

    Example:
        add_permanent_deny("rm -rf /home/user") -> adds "rm -rf" to permanent deny
    """
    config = get_permission_config()
    pattern = config.add_permanent_deny(command)
    return f"Added to permanent deny: {pattern}"


@tool
def list_permanent_deny() -> str:
    """List all permanent deny patterns (选项 d)."""
    config = get_permission_config()
    patterns = config.get_permanent_deny_patterns()
    if not patterns:
        return "No permanent deny patterns"
    return "Permanent deny patterns:\n" + "\n".join(f"- {p}" for p in patterns)


@tool
def remove_permanent_deny(pattern: str) -> str:
    """Remove a pattern from permanent deny list."""
    config = get_permission_config()
    if config.remove_permanent_deny(pattern):
        return f"Removed from permanent deny: {pattern}"
    return f"Pattern not found: {pattern}"


def needs_prompt(command: str) -> bool:
    """Check if a command needs to prompt user for confirmation.

    Args:
        command: The command to check

    Returns:
        True if should prompt, False if already allowed/blocked
    """
    config = get_permission_config()
    return config.needs_prompt(command)


def ask_permission(command: str) -> tuple[str, str]:
    """Generate permission prompt message for user.

    Args:
        command: The command to check

    Returns:
        tuple of (prompt_message, suggested_pattern)
        The suggested_pattern is extracted from the command (e.g., "rm -rf")
    """
    config = get_permission_config()
    allowed, reason, risk, _ = config.check(command)
    pattern = config.extract_command_type(command)

    if not allowed:
        action = "被阻止"
    else:
        action = "自动允许"

    message = f"""
命令: {command}
原因: {reason or '未知命令'}
风险: {risk}

选项:
  [y] 仅允许这一次
  [a] 允许当前命令类型 ({pattern}) 的所有变体
  [n] 仅拒绝这一次
  [d] 永久加入 deny 列表

请输入选择 (y/a/n/d):"""

    return message, pattern


PERMISSION_TOOLS = [
    set_mode,
    check_bash_permission,
    reload_permissions,
    show_permission_rules,
    add_session_allow,
    list_session_patterns,
    clear_session_patterns,
    add_permanent_deny,
    list_permanent_deny,
    remove_permanent_deny,
]