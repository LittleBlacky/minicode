"""Dynamic system prompt builder."""
import datetime
import os
import re
from pathlib import Path
from typing import Optional

# Default memory guidance
MEMORY_GUIDANCE = """
When to save memories:
- User states a preference ("I like tabs", "always use pytest") -> type: user
- User corrects you ("don't do X", "that was wrong because...") -> type: feedback
- You learn a project fact that is NOT obvious from current code alone
  (for example: a rule exists because of compliance, or a legacy module must
  stay untouched for business reasons) -> type: project
- You learn where an external resource lives (ticket board, dashboard, docs URL)
  -> type: reference
When NOT to save:
- Anything easily derivable from code (function signatures, file structure)
- Temporary task state (current branch, open PR numbers, current TODOs)
- Secrets or credentials (API keys, passwords)
"""


class SystemPromptBuilder:
    """Build system prompts from independent sections.

    Pipeline:
      1. core instructions
      2. tool listing
      3. skill metadata
      4. memory section
      5. MINI_AGENT.md chain
      6. dynamic context
    """

    def __init__(
        self,
        workdir: Optional[Path] = None,
        skills_dir: Optional[Path] = None,
        memory_dir: Optional[Path] = None,
        storage_dir: Optional[Path] = None,
    ):
        self.workdir = workdir or Path.cwd()
        self.storage_dir = storage_dir or (self.workdir / ".mini-agent-cli")
        self.skills_dir = skills_dir or (self.storage_dir / "skills")
        self.memory_dir = memory_dir or (self.storage_dir / ".memory")

    def _build_core(self) -> str:
        """Build core instructions."""
        return (
            f"You are a coding agent operating in {self.workdir}.\n"
            "Use the provided tools to explore, read, write, and edit files.\n"
            "Always verify before assuming. Prefer reading files over guessing.\n"
            "Use the todo tool for multi-step work.\n"
            "Keep exactly one step in_progress when a task has multiple steps.\n"
            "Refresh the plan as work advances. Prefer tools over prose."
        )

    def _build_tool_listing(self, tools: list) -> str:
        """Build tool listing section."""
        lines = ["# Available tools"]
        for t in tools:
            name = getattr(t, 'name', str(t))
            desc = getattr(t, 'description', '')
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def _build_skill_listing(self) -> str:
        """Build skill listing from skills directory."""
        if not self.skills_dir.exists():
            return ""
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
                name = meta.get("name", skill_path.parent.name)
                desc = meta.get("description", "")
                skills.append(f"- {name}: {desc}")
            except Exception:
                continue
        if not skills:
            return ""
        return "# Available skills\n" + "\n".join(skills)

    def _build_memory_section(self) -> str:
        """Build memory section from .memory directory."""
        if not self.memory_dir.exists():
            return ""
        memories = []
        memory_types = ("user", "feedback", "project", "reference")

        for md_file in self.memory_dir.glob("*.md"):
            if md_file.name == "MEMORY.md":
                continue
            try:
                text = md_file.read_text(encoding="utf-8")
                match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
                if not match:
                    continue
                header, body = match.group(1), match.group(2).strip()
                meta = {}
                for line in header.splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        meta[k.strip()] = v.strip()
                name = meta.get("name", md_file.stem)
                mem_type = meta.get("type", "project")
                desc = meta.get("description", "")
                memories.append(f"[{mem_type}] {name}: {desc}\n{body}")
            except Exception:
                continue

        if not memories:
            return ""

        lines = ["# Memories (persistent across sessions)", ""]
        for mem_type in memory_types:
            typed = [m for m in memories if f"[{mem_type}]" in m]
            if not typed:
                continue
            lines.append(f"## [{mem_type}]")
            for m in typed:
                lines.append(m)
            lines.append("")

        return "\n".join(lines)

    def _build_agent_md(self) -> str:
        """Build MINI_AGENT.md instructions."""
        sources = []

        # User global
        user_agent = Path.home() / ".mini-agent-cli" / "MINI_AGENT.md"
        if user_agent.exists():
            try:
                sources.append(("user global", user_agent.read_text(encoding="utf-8")))
            except Exception:
                pass

        # Project root
        project_agent = self.workdir / "MINI_AGENT.md"
        if project_agent.exists():
            try:
                sources.append(("project root", project_agent.read_text(encoding="utf-8")))
            except Exception:
                pass

        # Current subdir
        cwd = Path.cwd()
        if cwd != self.workdir:
            subdir_agent = cwd / "MINI_AGENT.md"
            if subdir_agent.exists():
                try:
                    sources.append(("subdir", subdir_agent.read_text(encoding="utf-8")))
                except Exception:
                    pass

        if not sources:
            return ""

        parts = ["# MINI_AGENT.md instructions"]
        for label, content in sources:
            parts.append(f"## From {label}")
            parts.append(content.strip())
        return "\n\n".join(parts)

    def _build_dynamic_context(self, model_id: str) -> str:
        """Build dynamic context section."""
        lines = [
            f"Current date: {datetime.date.today().isoformat()}",
            f"Working directory: {self.workdir}",
            f"Model: {model_id}",
        ]
        return "# Dynamic context\n" + "\n".join(lines)

    def build(
        self,
        tools: Optional[list] = None,
        model_id: Optional[str] = None,
        memory_section: Optional[str] = None,
    ) -> str:
        """Build complete system prompt.

        Args:
            tools: List of tools to include in listing
            model_id: Model identifier for dynamic context
            memory_section: Pre-built memory section (optional)

        Returns:
            Complete system prompt string
        """
        sections = []

        # 1. Core instructions
        sections.append(self._build_core())

        # 2. Tool listing
        if tools:
            sections.append(self._build_tool_listing(tools))

        # 3. Skill listing
        skill_listing = self._build_skill_listing()
        if skill_listing:
            sections.append(skill_listing)

        # 4. Memory section
        if memory_section:
            sections.append(memory_section)
        else:
            memory_section = self._build_memory_section()
            if memory_section:
                sections.append(memory_section)

        # 5. MINI_AGENT.md
        agent_md = self._build_agent_md()
        if agent_md:
            sections.append(agent_md)

        # 6. Dynamic context
        sections.append(self._build_dynamic_context(model_id or "unknown"))

        return "\n\n".join(filter(None, sections))
