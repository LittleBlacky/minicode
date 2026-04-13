"""Skill management tools."""
import json
import re
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


class SkillManager:
    """Manage agent skills."""

    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = skills_dir or Path.cwd() / ".mini-agent-cli" / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict]:
        """List all skills."""
        skills = []
        for skill_path in self.skills_dir.rglob("SKILL.md"):
            try:
                text = skill_path.read_text(encoding="utf-8")
                match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
                if not match:
                    continue
                meta = {}
                for line in match.group(1).splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        meta[k.strip()] = v.strip()
                skills.append({
                    "name": meta.get("name", skill_path.parent.name),
                    "description": meta.get("description", ""),
                    "path": str(skill_path),
                })
            except Exception:
                continue
        return skills

    def get_content(self, name: str) -> Optional[str]:
        """Get skill content by name."""
        for skill_path in self.skills_dir.rglob("SKILL.md"):
            if skill_path.parent.name == name:
                return skill_path.read_text(encoding="utf-8")
        return None

    def create(self, name: str, description: str, trigger: str, content: str) -> str:
        """Create a new skill."""
        skill_dir = self.skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"

        skill_md = f"""---
name: {name}
description: {description}
trigger: {trigger}
---

# {name}

{content}
"""
        skill_file.write_text(skill_md, encoding="utf-8")
        return f"Created skill: {name}"

    def delete(self, name: str) -> bool:
        """Delete a skill."""
        skill_dir = self.skills_dir / name
        if skill_dir.exists():
            import shutil
            shutil.rmtree(skill_dir)
            return True
        return False


# Global instance
_skill_manager: Optional[SkillManager] = None


def get_skill_manager(skills_dir: Optional[Path] = None) -> SkillManager:
    global _skill_manager
    if _skill_manager is None:
        _skill_manager = SkillManager(skills_dir)
    return _skill_manager


@tool
def skill_list() -> str:
    """List all available skills."""
    mgr = get_skill_manager()
    skills = mgr.list()
    if not skills:
        return "No skills configured"
    lines = ["# Skills"]
    for s in skills:
        lines.append(f"- {s['name']}: {s['description']}")
    return "\n".join(lines)


@tool
def skill_get(name: str) -> str:
    """Get skill content.

    Args:
        name: Skill name
    """
    mgr = get_skill_manager()
    content = mgr.get_content(name)
    if content:
        return content
    return f"[Error]: Skill {name} not found"


@tool
def skill_create(name: str, description: str, trigger: str, content: str) -> str:
    """Create a new skill.

    Args:
        name: Skill name
        description: Skill description
        trigger: Trigger pattern (e.g., /skillname)
        content: Skill content/instructions
    """
    mgr = get_skill_manager()
    return mgr.create(name, description, trigger, content)


SKILL_TOOLS = [skill_list, skill_get, skill_create]
