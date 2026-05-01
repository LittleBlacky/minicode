"""Permission configuration loader for YAML-based permission rules."""
import fnmatch
import re
from pathlib import Path
from typing import Optional

import yaml


# Built-in dangerous patterns (cannot be overridden)
BUILTIN_DANGEROUS_PATTERNS = [
    ("rm_rf_root", r"\brm\s+(-[rf]+)?\s*/\s*$", "critical", "Recursive delete of root"),
    ("sudo_shutdown", r"\bsudo\s+(shutdown|reboot|init\s+[06])", "high", "System shutdown/reboot"),
    ("fork_bomb", r":\(\)\s*\{[^}]+\}\s*;", "critical", "Fork bomb"),
    ("dd_zero", r"\bdd\s+.*of=/dev/", "critical", "Direct disk write"),
    ("mkfs", r"\bmkfs\b", "critical", "Filesystem format"),
    ("curl_pipe_sh", r"curl.*\|\s*(sh|bash|fish|zsh)", "medium", "Pipe to shell"),
    ("wget_pipe_sh", r"wget.*\|\s*(sh|bash|fish|zsh)", "medium", "Pipe to shell"),
    ("chmod_sensitive", r"chmod\s+[47]0[47]0", "medium", "Dangerous chmod"),
]

# Risk level to priority mapping
RISK_PRIORITY = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}


class PermissionConfig:
    """Load and manage YAML permission configuration.

    Configuration file: .minicode/permissions.yaml

    Check order:
    0. Session allowed patterns (from [a] option, in-memory only)
    1. Built-in dangerous patterns (always blocked, cannot be overridden)
    2. Permanent deny patterns (选项 d, persisted to YAML)
    3. User deny patterns (block even if allowed by allow rules)
    4. User allow patterns (explicitly permitted commands)
    5. Default: allow

    Pattern formats:
    - Glob: "rm -rf /tmp/*"
    - Regex: "re:^sudo\s+rm"

    Session patterns (选项 a):
    - Extract command type: "rm -rf /tmp" → "rm -rf"
    - "npm run build" → "npm run"
    - "git push --force" → "git push"
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self._config: dict = {}
        self._allow_patterns: list[tuple[str, str, re.Pattern]] = []
        self._deny_patterns: list[tuple[str, str, re.Pattern]] = []
        self._permanent_deny_patterns: list[tuple[str, str, re.Pattern]] = []
        self._session_allowed_patterns: list[tuple[str, str, re.Pattern]] = []
        self._prompt_unknown = False
        self._prompt_risk_threshold = "medium"  # Prompt for medium+ by default
        self._load()

    def _load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path or not self.config_path.exists():
            # Use defaults
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}

            self._parse_patterns()
            self._parse_risk_threshold()
            self._parse_prompt_config()

        except Exception as e:
            print(f"[Permission config error: {e}]")
            self._config = {}

    def _parse_prompt_config(self) -> None:
        """Parse prompt configuration."""
        self._prompt_unknown = self._config.get("prompt_unknown", False)

    def _parse_patterns(self) -> None:
        """Parse allow and deny patterns."""
        self._allow_patterns = []
        self._deny_patterns = []
        self._permanent_deny_patterns = []

        # Parse permanent deny patterns (选项 d)
        for pattern in self._config.get("permanent_deny") or []:
            ptype, compiled = self._compile_pattern(pattern)
            self._permanent_deny_patterns.append((pattern, ptype, compiled))

        # Parse allow patterns (handle None from empty/commented YAML lists)
        for pattern in self._config.get("allow") or []:
            ptype, compiled = self._compile_pattern(pattern)
            self._allow_patterns.append((pattern, ptype, compiled))

        # Parse deny patterns
        for pattern in self._config.get("deny") or []:
            ptype, compiled = self._compile_pattern(pattern)
            self._deny_patterns.append((pattern, ptype, compiled))

    def _compile_pattern(self, pattern: str) -> tuple[str, re.Pattern]:
        """Compile a pattern string to regex.

        Formats:
        - "re:..." -> treat as regex
        - "..." -> treat as glob
        """
        if pattern.startswith("re:"):
            return ("regex", re.compile(pattern[3:]))
        else:
            # Convert glob to regex
            return ("glob", self._glob_to_regex(pattern))

    def _glob_to_regex(self, pattern: str) -> re.Pattern:
        """Convert glob pattern to regex."""
        # Escape special chars except * and ?
        escaped = ""
        for c in pattern:
            if c == "*":
                escaped += ".*"
            elif c == "?":
                escaped += "."
            elif c in r"\+.[]{}()|^$":
                escaped += "\\" + c
            else:
                escaped += c
        return re.compile(escaped, re.IGNORECASE)

    def _parse_risk_threshold(self) -> None:
        """Parse risk threshold for prompts."""
        threshold = self._config.get("prompt_above_risk", "medium")
        if threshold in RISK_PRIORITY:
            self._prompt_risk_threshold = threshold

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load()

    def check(self, command: str) -> tuple[bool, str, str, list[str]]:
        """Check if a command is allowed.

        Returns:
            tuple of (allowed, reason, risk_level, matched_patterns)
        """
        matched_patterns = []

        # 1. Check built-in dangerous patterns FIRST (cannot be overridden)
        for name, pattern, risk, desc in BUILTIN_DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return False, f"[BUILT-IN] {desc}", risk, [f"builtin:{name}"]

        # 2. Check session allowed patterns (from [a] option)
        for pattern, ptype, compiled in self._session_allowed_patterns:
            if self._match_startswith(compiled, ptype, command):
                return True, f"[SESSION] Allowed: {pattern}", "none", [f"session:{pattern}"]

        # 3. Check permanent deny patterns (选项 d)
        for pattern, ptype, compiled in self._permanent_deny_patterns:
            if self._match(compiled, ptype, command):
                matched_patterns.append(f"permanent_deny:{pattern}")
                return False, f"[PERMANENT DENY] Blocked: {pattern}", "high", matched_patterns

        # 4. Check user deny patterns
        for pattern, ptype, compiled in self._deny_patterns:
            if self._match(compiled, ptype, command):
                matched_patterns.append(f"deny:{pattern}")
                return False, f"[DENY] Blocked by permissions.yaml", "high", matched_patterns

        # 5. Check user allow patterns
        for pattern, ptype, compiled in self._allow_patterns:
            if self._match(compiled, ptype, command):
                matched_patterns.append(f"allow:{pattern}")
                return True, f"[ALLOW] Matched: {pattern}", "none", matched_patterns

        # 6. Default: allow
        return True, "", "none", []

    def _match_startswith(self, compiled: re.Pattern, ptype: str, command: str) -> bool:
        """Match command that starts with the pattern.

        This ensures "rm -rf" pattern only matches "rm -rf /tmp" but not
        "echo rm -rf" (command embedded in larger command).
        """
        # Strip leading whitespace
        command_stripped = command.strip()

        # First, check direct startswith (handles simple cases like "rm -rf /tmp")
        if command_stripped.startswith(compiled.pattern):
            return True

        # Check if pattern starts after common command prefixes like sudo, doas
        # e.g., "sudo rm -rf /tmp" should match "rm -rf" pattern
        prefix_pattern = r'^(sudo|doas|pkexec)\s+'
        if re.match(prefix_pattern, command_stripped):
            remainder = re.sub(prefix_pattern, '', command_stripped, count=1)
            if remainder.startswith(compiled.pattern):
                return True

        return False

    def add_session_pattern(self, command: str) -> str:
        """Add a session pattern for this command type (选项 a).

        Extracts command type and adds to session patterns.
        Returns the extracted pattern.

        Example: "rm -rf /tmp/test" → adds "rm -rf" pattern
        """
        pattern = self.extract_command_type(command)
        compiled = self._glob_to_regex(pattern)
        self._session_allowed_patterns.append((pattern, "glob", compiled))
        return pattern

    @staticmethod
    def extract_command_type(command: str) -> str:
        """Extract command type from command string.

        Example: "rm -rf /tmp/test" → "rm -rf"
        Example: "npm run build --verbose" → "npm run"
        Example: "git push --force origin main" → "git push"

        Returns:
            First two words (command + first flag) or just the command.
        """
        parts = command.strip().split()
        if not parts:
            return command
        # Take first two parts to capture "rm -rf", "npm run", "git push"
        if len(parts) >= 2:
            return f"{parts[0]} {parts[1]}"
        return parts[0]

    def needs_prompt(self, command: str) -> bool:
        """Check if command needs to prompt user.

        Returns:
            True if should prompt user for confirmation.
            Includes session patterns, unknown commands, etc.
        """
        allowed, reason, risk, _ = self.check(command)

        # Already blocked, no need to prompt (will just deny)
        if not allowed:
            return False

        # Check if matched session pattern
        for pattern, _, _ in self._session_allowed_patterns:
            compiled = self._glob_to_regex(pattern)
            if compiled.search(command):
                return False  # Already allowed by session pattern

        # Check if matched user allow pattern
        for _, ptype, compiled in self._allow_patterns:
            if compiled.search(command):
                return False  # Already allowed by user config

        # Unknown command - check if should prompt based on config
        if self._prompt_unknown:
            return True

        # Check risk threshold
        return self.should_prompt(risk)

    def get_session_patterns(self) -> list[str]:
        """Get list of current session patterns."""
        return [p[0] for p in self._session_allowed_patterns]

    def get_permanent_deny_patterns(self) -> list[str]:
        """Get list of permanent deny patterns."""
        return [p[0] for p in self._permanent_deny_patterns]

    def add_permanent_deny(self, command: str) -> str:
        """Add command to permanent deny list (选项 d).

        Extracts command type and adds to permanent deny patterns.
        Persists to YAML file.

        Returns:
            The pattern that was added.
        """
        pattern = self.extract_command_type(command)
        ptype, compiled = self._compile_pattern(pattern)

        # Check if already exists
        if pattern not in [p[0] for p in self._permanent_deny_patterns]:
            self._permanent_deny_patterns.append((pattern, ptype, compiled))
            self._save_to_yaml()

        return pattern

    def remove_permanent_deny(self, pattern: str) -> bool:
        """Remove a pattern from permanent deny list."""
        original_len = len(self._permanent_deny_patterns)
        self._permanent_deny_patterns = [
            (p, t, c) for p, t, c in self._permanent_deny_patterns if p != pattern
        ]
        if len(self._permanent_deny_patterns) < original_len:
            self._save_to_yaml()
            return True
        return False

    def _save_to_yaml(self) -> None:
        """Save current patterns back to YAML file."""
        if not self.config_path:
            return

        try:
            # Read existing config
            existing = {}
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}

            # Update permanent_deny list
            existing["permanent_deny"] = [p[0] for p in self._permanent_deny_patterns]

            # Write back
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, allow_unicode=True, default_flow_style=False)

        except Exception as e:
            print(f"[Permission config save error: {e}]")

    def clear_session_patterns(self) -> None:
        """Clear all session patterns."""
        self._session_allowed_patterns = []

    def _match(self, compiled: re.Pattern, ptype: str, command: str) -> bool:
        """Match a command against a compiled pattern."""
        if ptype == "regex":
            return bool(compiled.search(command))
        else:
            return bool(compiled.search(command))

    def should_prompt(self, risk_level: str) -> bool:
        """Check if we should prompt for this risk level."""
        return RISK_PRIORITY.get(risk_level, 4) <= RISK_PRIORITY.get(
            self._prompt_risk_threshold, 2
        )

    def get_builtin_patterns(self) -> list[dict]:
        """Get list of built-in dangerous patterns."""
        return [
            {"name": name, "risk": risk, "description": desc}
            for name, _, risk, desc in BUILTIN_DANGEROUS_PATTERNS
        ]

    def get_config_summary(self) -> dict:
        """Get configuration summary."""
        return {
            "config_path": str(self.config_path) if self.config_path else None,
            "loaded": bool(self._config),
            "allow_patterns": len(self._allow_patterns),
            "deny_patterns": len(self._deny_patterns),
            "permanent_deny_patterns": len(self._permanent_deny_patterns),
            "session_patterns": len(self._session_allowed_patterns),
            "prompt_unknown": self._prompt_unknown,
            "prompt_threshold": self._prompt_risk_threshold,
            "builtin_patterns": len(BUILTIN_DANGEROUS_PATTERNS),
        }


# Global config instance
_global_config: Optional[PermissionConfig] = None


def get_permission_config(config_path: Optional[Path] = None) -> PermissionConfig:
    """Get or create global PermissionConfig instance."""
    global _global_config
    if _global_config is None:
        default_path = Path.cwd() / ".minicode" / "permissions.yaml"
        _global_config = PermissionConfig(config_path or default_path)
    return _global_config


def reset_permission_config() -> None:
    """Reset global config (for testing)."""
    global _global_config
    _global_config = None