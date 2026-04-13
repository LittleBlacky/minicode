"""Bash execution tools."""
import subprocess
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


# Dangerous commands that should be blocked
DANGEROUS_COMMANDS = [
    "rm -rf /",
    "rm -rf /*",
    "sudo shutdown",
    "sudo reboot",
    "init 0",
    "init 6",
    ":(){:|:&};:",  # Fork bomb
]


class BashSecurityValidator:
    """Validate bash commands for safety."""

    def __init__(self):
        self.dangerous_patterns = DANGEROUS_COMMANDS

    def is_safe(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute."""
        cmd_lower = command.lower().strip()

        # Check dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.lower() in cmd_lower:
                return False, f"Dangerous command blocked: {pattern}"

        # Check for destructive rm patterns
        if "rm -rf /" in cmd_lower or "rm -rf /*" in cmd_lower:
            return False, "Recursive delete of root blocked"

        # Check for piping to eval/sh
        if " | sh" in cmd_lower or " | bash" in cmd_lower:
            if any(d in cmd_lower for d in ["curl", "wget"]):
                return False, "Pipe to shell blocked for security"

        return True, ""


class BashTools:
    """Bash execution utilities."""

    def __init__(self, workdir: Optional[Path] = None, timeout: int = 120):
        self.workdir = workdir or Path.cwd()
        self.timeout = timeout
        self.validator = BashSecurityValidator()

    def run(self, command: str) -> str:
        """Run bash command with security check."""
        # Security validation
        safe, msg = self.validator.is_safe(command)
        if not safe:
            return f"[Error]: {msg}"

        # Execute
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = (result.stdout + result.stderr).strip()
            return output[:50000] if output else "(no output)"
        except subprocess.TimeoutExpired:
            return f"[Error]: Timeout ({self.timeout}s)"
        except Exception as e:
            return f"[Error]: {e}"


# Global instance
_bash_tools: Optional[BashTools] = None


def get_bash_tools(workdir: Optional[Path] = None, timeout: int = 120) -> BashTools:
    """Get or create global BashTools instance."""
    global _bash_tools
    if _bash_tools is None:
        _bash_tools = BashTools(workdir, timeout)
    return _bash_tools


def set_bash_tools(tools: BashTools) -> None:
    """Set global BashTools instance."""
    global _bash_tools
    _bash_tools = tools


# Tool functions

@tool
def bash_tool(command: str) -> str:
    """Run a shell command.

    Args:
        command: Shell command to execute
    """
    tools = get_bash_tools()
    return tools.run(command)


# Tool list for registration
BASH_TOOLS = [bash_tool]
