"""Git worktree tools."""
import subprocess
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


class WorktreeManager:
    """Manage git worktrees."""

    def __init__(self, repo_dir: Optional[Path] = None):
        self.repo_dir = repo_dir or Path.cwd()

    def list(self) -> list[dict]:
        """List all worktrees."""
        try:
            result = subprocess.run(
                ["git", "worktree", "list", "--porcelain"],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            worktrees = []
            current = {}
            for line in result.stdout.splitlines():
                if line.startswith("worktree "):
                    current["path"] = line[9:]
                elif line.startswith("HEAD "):
                    current["head"] = line[4:]
                elif line == "" and current:
                    worktrees.append(current)
                    current = {}
            return worktrees
        except Exception as e:
            return [{"error": str(e)}]

    def create(self, name: str, branch: Optional[str] = None) -> dict:
        """Create a new worktree."""
        cmd = ["git", "worktree", "add"]
        if branch:
            cmd.extend(["-b", branch])
        cmd.append(name)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return {"status": "created", "path": name}
            return {"status": "error", "message": result.stderr}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def remove(self, path: str) -> dict:
        """Remove a worktree."""
        try:
            result = subprocess.run(
                ["git", "worktree", "remove", path],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return {"status": "removed"}
            return {"status": "error", "message": result.stderr}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Global instance
_worktree_manager: Optional[WorktreeManager] = None


def get_worktree_manager(repo_dir: Optional[Path] = None) -> WorktreeManager:
    global _worktree_manager
    if _worktree_manager is None:
        _worktree_manager = WorktreeManager(repo_dir)
    return _worktree_manager


@tool
def worktree_list() -> str:
    """List all git worktrees."""
    mgr = get_worktree_manager()
    worktrees = mgr.list()
    if not worktrees or "error" in worktrees[0]:
        return "No worktrees found"
    lines = ["# Worktrees"]
    for wt in worktrees:
        lines.append(f"- {wt.get('path')}: HEAD={wt.get('head', 'unknown')}")
    return "\n".join(lines)


@tool
def worktree_create(name: str, branch: Optional[str] = None) -> str:
    """Create a new git worktree.

    Args:
        name: Worktree name/path
        branch: Branch name (optional, creates from HEAD)
    """
    mgr = get_worktree_manager()
    result = mgr.create(name, branch)
    if result.get("status") == "created":
        return f"Created worktree {name}" + (f" with branch {branch}" if branch else "")
    return f"[Error]: {result.get('message', 'Unknown error')}"


@tool
def worktree_remove(path: str) -> str:
    """Remove a git worktree.

    Args:
        path: Worktree path to remove
    """
    mgr = get_worktree_manager()
    result = mgr.remove(path)
    if result.get("status") == "removed":
        return f"Removed worktree {path}"
    return f"[Error]: {result.get('message', 'Unknown error')}"


WORKTREE_TOOLS = [worktree_list, worktree_create, worktree_remove]