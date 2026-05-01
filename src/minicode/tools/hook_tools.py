"""Hook system for extension points around the agent loop."""
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


HOOK_EVENTS = ("PreToolUse", "PostToolUse", "SessionStart")
HOOK_TIMEOUT = 30

STORAGE_DIR = Path.cwd() / ".minicode"
HOOK_CONFIG_PATH = STORAGE_DIR / "hooks.json"
TRUST_MARKER = STORAGE_DIR / "trusted"


class HookManager:
    """Load and execute hooks from .hooks.json configuration.

    Hooks are extension points around the main loop that let you add behavior
    without rewriting the graph nodes themselves.

    Supports two types of hooks:
    - subprocess hooks: execute external commands (loaded from .hooks.json)
    - python hooks: execute Python functions directly (registered at runtime)

    Exit-code contract (for subprocess hooks):
    - 0: Continue execution
    - 1: Block the operation
    - 2: Inject additional context

    Example .hooks.json:
    {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "bash_tool",
                    "command": "python verify_command.py"
                }
            ],
            "SessionStart": [
                {
                    "matcher": "*",
                    "command": "echo 'Session started'"
                }
            ]
        }
    }
    """

    def __init__(self, config_path: Optional[Path] = None, sdk_mode: bool = False):
        self.hooks = {event: [] for event in HOOK_EVENTS}
        self._python_hooks = {event: [] for event in HOOK_EVENTS}
        self._sdk_mode = sdk_mode
        self.config_path = config_path or HOOK_CONFIG_PATH
        self._load_hooks()

    def _load_hooks(self) -> None:
        """Load hooks from configuration file."""
        if not self.config_path.exists():
            return

        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            for event in HOOK_EVENTS:
                self.hooks[event] = config.get("hooks", {}).get(event, [])
            if any(self.hooks.values()):
                print(f"[Hooks loaded from {self.config_path}]")
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Hook config error: {e}]")

    def _check_workspace_trust(self) -> bool:
        """Check if workspace is trusted (for security)."""
        if self._sdk_mode:
            return True
        return TRUST_MARKER.exists()

    def run_hooks(
        self,
        event: str,
        context: Optional[dict] = None,
    ) -> dict:
        """Execute all hooks for an event.

        Args:
            event: One of PreToolUse, PostToolUse, SessionStart
            context: Context dict with tool_name, tool_input, tool_output, etc.

        Returns:
            dict with keys:
            - blocked: bool - whether to block the operation
            - block_reason: str - reason for blocking
            - messages: list[str] - injected context messages
            - updated_input: dict | None - modified tool input
        """
        result = {
            "blocked": False,
            "block_reason": "",
            "messages": [],
            "updated_input": None,
        }

        if not self._check_workspace_trust():
            return result

        if event not in self.hooks:
            return result

        hooks = self.hooks.get(event, [])
        if not hooks:
            return result

        for hook_def in hooks:
            hook_result = self._execute_hook(hook_def, event, context)
            if hook_result["blocked"]:
                result["blocked"] = True
                result["block_reason"] = hook_result["block_reason"]
                break
            if hook_result["messages"]:
                result["messages"].extend(hook_result["messages"])
            if hook_result.get("updated_input"):
                result["updated_input"] = hook_result["updated_input"]

        return result

    def _execute_hook(
        self,
        hook_def: dict,
        event: str,
        context: Optional[dict],
    ) -> dict:
        """Execute a single hook definition."""
        result = {
            "blocked": False,
            "block_reason": "",
            "messages": [],
            "updated_input": None,
        }

        matcher = hook_def.get("matcher")
        command = hook_def.get("command", "")

        if not command:
            return result

        if matcher and matcher != "*":
            if context and matcher != context.get("tool_name", ""):
                return result

        env = dict(os.environ)
        if context:
            env["HOOK_EVENT"] = event
            env["HOOK_TOOL_NAME"] = context.get("tool_name", "")
            env["HOOK_TOOL_INPUT"] = json.dumps(
                context.get("tool_input", {}),
                ensure_ascii=False,
            )[:10000]
            if "tool_output" in context:
                env["HOOK_TOOL_OUTPUT"] = str(context["tool_output"])[:10000]

        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=Path.cwd(),
                env=env,
                capture_output=True,
                text=True,
                timeout=HOOK_TIMEOUT,
            )

            if r.returncode == 0:
                if r.stdout.strip():
                    print(f"  [hook:{event}] {r.stdout.strip()[:100]}")

                try:
                    output = json.loads(r.stdout)
                    if "updatedInput" in output and context:
                        context["tool_input"] = output["updatedInput"]
                        result["updated_input"] = output["updatedInput"]
                    if "additionalContext" in output:
                        result["messages"].append(output["additionalContext"])
                except (json.JSONDecodeError, TypeError):
                    pass

            elif r.returncode == 1:
                result["blocked"] = True
                reason = r.stderr.strip() or "Blocked by hook"
                result["block_reason"] = reason
                print(f"  [hook:{event}] BLOCKED: {reason[:200]}")

            elif r.returncode == 2:
                msg = r.stderr.strip()
                if msg:
                    result["messages"].append(msg)
                    print(f"  [hook:{event}] INJECT: {msg[:200]}")

        except subprocess.TimeoutExpired:
            print(f"  [hook:{event}] Timeout ({HOOK_TIMEOUT}s)")
        except Exception as e:
            print(f"  [hook:{event}] Error: {e}")

        return result

    def register_python_hook(
        self,
        event: str,
        handler: callable,
        matcher: Optional[str] = None,
    ) -> None:
        """Register a Python function as a hook.

        Args:
            event: One of PreToolUse, PostToolUse, SessionStart
            handler: Callable that takes (context: dict) and returns dict:
                {"blocked": bool, "block_reason": str, "messages": list}
            matcher: Optional tool name filter (e.g., "bash_tool")
        """
        if event not in self._python_hooks:
            return
        self._python_hooks[event].append({
            "handler": handler,
            "matcher": matcher,
        })

    def run_python_hooks(
        self,
        event: str,
        context: Optional[dict] = None,
    ) -> dict:
        """Execute Python hooks for an event."""
        result = {
            "blocked": False,
            "block_reason": "",
            "messages": [],
            "updated_input": None,
        }

        if not self._check_workspace_trust():
            return result

        for hook_def in self._python_hooks.get(event, []):
            handler = hook_def["handler"]
            matcher = hook_def.get("matcher")

            # Match check
            if matcher and matcher != "*":
                if context and matcher != context.get("tool_name", ""):
                    continue

            try:
                hook_result = handler(context or {})
                if isinstance(hook_result, dict):
                    if hook_result.get("blocked"):
                        result["blocked"] = True
                        result["block_reason"] = hook_result.get("block_reason", "Blocked")
                        break
                    if hook_result.get("messages"):
                        result["messages"].extend(hook_result["messages"])
                    if hook_result.get("updated_input"):
                        result["updated_input"] = hook_result["updated_input"]
                elif isinstance(hook_result, bool):
                    # Simple bool return: True = blocked
                    if hook_result:
                        result["blocked"] = True
                        result["block_reason"] = "Blocked by hook"
                        break
            except Exception as e:
                print(f"  [python_hook:{event}] Error: {e}")

        return result

    def add_hook(self, event: str, hook_def: dict) -> None:
        """Add a hook at runtime."""
        if event in self.hooks:
            self.hooks[event].append(hook_def)

    def list_hooks(self) -> dict:
        """List all registered hooks (both subprocess and python)."""
        result = {}
        for event in HOOK_EVENTS:
            all_hooks = list(self.hooks.get(event, []))
            for ph in self._python_hooks.get(event, []):
                handler = ph["handler"]
                name = getattr(handler, "__name__", repr(handler))
                all_hooks.append({
                    "matcher": ph.get("matcher", "*"),
                    "type": "python",
                    "handler": name,
                })
            result[event] = all_hooks
        return result

    def reload(self) -> None:
        """Reload hooks from configuration file."""
        self.hooks = {event: [] for event in HOOK_EVENTS}
        self._python_hooks = {event: [] for event in HOOK_EVENTS}
        self._load_hooks()


_global_hook_manager: Optional[HookManager] = None


def get_hook_manager(sdk_mode: bool = False) -> HookManager:
    """Get or create global HookManager instance."""
    global _global_hook_manager
    if _global_hook_manager is None:
        _global_hook_manager = HookManager(sdk_mode=sdk_mode)
    return _global_hook_manager


@tool
def hook_list() -> str:
    """List all registered hooks by event type.

    Returns a summary of available hooks.
    """
    manager = get_hook_manager()
    hooks = manager.list_hooks()

    lines = ["# Registered Hooks"]
    for event, hook_list in hooks.items():
        if hook_list:
            lines.append(f"\n## {event}")
            for hook in hook_list:
                matcher = hook.get("matcher", "*")
                command = hook.get("command", "")[:50]
                lines.append(f"- matcher: {matcher}")
                lines.append(f"  command: {command}...")

    return "\n".join(lines) if lines else "No hooks registered"


@tool
def hook_reload() -> str:
    """Reload hooks from configuration file.

    Use this after editing .minicode/.hooks.json
    """
    manager = get_hook_manager()
    manager.reload()
    return "Hooks reloaded"


# Tool list for registration
HOOK_TOOLS = [hook_list, hook_reload]
