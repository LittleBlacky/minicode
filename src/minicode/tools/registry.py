"""Tool registry - collects all tools for the agent."""
from minicode.tools.file_tools import FILE_TOOLS
from minicode.tools.bash_tools import BASH_TOOLS
from minicode.tools.todo_tools import TODO_TOOLS
from minicode.tools.task_tools import TASK_TOOLS
from minicode.tools.team_tools import TEAM_TOOLS
from minicode.tools.background_tools import BACKGROUND_TOOLS
from minicode.tools.cron_tools import CRON_TOOLS
from minicode.tools.worktree_tools import WORKTREE_TOOLS
from minicode.tools.search_tools import SEARCH_TOOLS
from minicode.tools.mcp_tools import MCP_TOOLS
from minicode.tools.memory_tools import MEMORY_TOOLS
from minicode.tools.skill_tools import SKILL_TOOLS
from minicode.tools.permission_tools import PERMISSION_TOOLS
from minicode.tools.protocol_tools import PROTOCOL_TOOLS


ALL_TOOLS = (
    FILE_TOOLS
    + BASH_TOOLS
    + TODO_TOOLS
    + TASK_TOOLS
    + TEAM_TOOLS
    + BACKGROUND_TOOLS
    + CRON_TOOLS
    + WORKTREE_TOOLS
    + SEARCH_TOOLS
    + MCP_TOOLS
    + MEMORY_TOOLS
    + SKILL_TOOLS
    + PERMISSION_TOOLS
    + PROTOCOL_TOOLS
)


__all__ = ["ALL_TOOLS"]
