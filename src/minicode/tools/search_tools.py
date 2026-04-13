"""Search and glob tools."""
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


@tool
def glob_tool(pattern: str, path: Optional[str] = None) -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "**/*.py")
        path: Base path (default: current directory)
    """
    base = Path(path) if path else Path.cwd()
    try:
        files = list(base.glob(pattern))
        if not files:
            return f"No files matching {pattern}"
        return "\n".join(str(f.relative_to(base)) for f in files[:50])
    except Exception as e:
        return f"[Error]: {e}"


@tool
def grep_tool(pattern: str, path: Optional[str] = None, case_insensitive: bool = False) -> str:
    """Search for pattern in files.

    Args:
        pattern: Regex pattern
        path: Path to search in
        case_insensitive: Case insensitive search
    """
    import re

    base = Path(path) if path else Path.cwd()
    flags = re.IGNORECASE if case_insensitive else 0
    compiled = re.compile(pattern, flags)

    matches = []
    for fp in base.rglob("*"):
        if not fp.is_file():
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(content.splitlines(), 1):
                if compiled.search(line):
                    matches.append(f"{fp}:{i}: {line.rstrip()}")
                    if len(matches) >= 100:
                        matches.append("... (truncated)")
                        break
        except Exception:
            continue

    if not matches:
        return f"No matches for {pattern}"
    return "\n".join(matches)


SEARCH_TOOLS = [glob_tool, grep_tool]