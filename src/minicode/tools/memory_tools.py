"""Memory management tools."""
import json
import re
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


class MemoryManager:
    """Manage persistent memories."""

    def __init__(self, memory_dir: Optional[Path] = None):
        self.memory_dir = memory_dir or Path.cwd() / ".mini-agent-cli" / ".memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.memory_dir / "MEMORY.md"

    def save(
        self,
        name: str,
        content: str,
        memory_type: str,
        description: str = "",
    ) -> str:
        """Save a memory."""
        safe_name = name.replace(" ", "-").lower()
        file_path = self.memory_dir / f"{safe_name}.md"

        frontmatter = f"""---
name: {name}
description: {description}
type: {memory_type}
---

{content}
"""
        file_path.write_text(frontmatter, encoding="utf-8")
        return f"Saved memory: {name}"

    def get(self, name: str) -> Optional[str]:
        """Get a memory by name."""
        safe_name = name.replace(" ", "-").lower()
        file_path = self.memory_dir / f"{safe_name}.md"
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return None

    def list_by_type(self, memory_type: str) -> list[dict]:
        """List memories by type."""
        memories = []
        for md_file in self.memory_dir.glob("*.md"):
            if md_file.name == "MEMORY.md":
                continue
            try:
                text = md_file.read_text(encoding="utf-8")
                match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
                if not match:
                    continue
                header = match.group(1)
                meta = {}
                for line in header.splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        meta[k.strip()] = v.strip()
                if meta.get("type") == memory_type:
                    memories.append({
                        "name": meta.get("name", md_file.stem),
                        "description": meta.get("description", ""),
                        "file": str(md_file),
                    })
            except Exception:
                continue
        return memories

    def delete(self, name: str) -> bool:
        """Delete a memory."""
        safe_name = name.replace(" ", "-").lower()
        file_path = self.memory_dir / f"{safe_name}.md"
        if file_path.exists():
            file_path.unlink()
            return True
        return False


# Global instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(memory_dir: Optional[Path] = None) -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(memory_dir)
    return _memory_manager


@tool
def memory_save(name: str, content: str, memory_type: str, description: str = "") -> str:
    """Save a memory.

    Args:
        name: Memory name
        content: Memory content
        memory_type: Type: user, feedback, project, reference
        description: One-line description
    """
    mgr = get_memory_manager()
    return mgr.save(name, content, memory_type, description)


@tool
def memory_get(name: str) -> str:
    """Get a memory by name.

    Args:
        name: Memory name
    """
    mgr = get_memory_manager()
    content = mgr.get(name)
    if content:
        return content
    return f"[Error]: Memory {name} not found"


@tool
def memory_list(memory_type: Optional[str] = None) -> str:
    """List memories.

    Args:
        memory_type: Filter by type (optional)
    """
    mgr = get_memory_manager()
    if memory_type:
        memories = mgr.list_by_type(memory_type)
    else:
        memories = []
        for md_file in mgr.memory_dir.glob("*.md"):
            if md_file.name != "MEMORY.md":
                memories.append({"name": md_file.stem})

    if not memories:
        return "No memories"
    lines = ["# Memories"]
    for m in memories:
        lines.append(f"- {m['name']}: {m.get('description', '')}")
    return "\n".join(lines)


MEMORY_TOOLS = [memory_save, memory_get, memory_list]
