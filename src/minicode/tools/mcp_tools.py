"""MCP (Model Context Protocol) client tools."""
import json
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


class MCPClient:
    """Client for MCP server communication."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.cwd() / ".mini-agent-cli" / "mcp_config.json"
        self.servers = {}
        self._load_config()

    def _load_config(self) -> None:
        if self.config_path.exists():
            try:
                config = json.loads(self.config_path.read_text(encoding="utf-8"))
                self.servers = config.get("servers", {})
            except Exception:
                self.servers = {}

    def connect(self, server_name: str) -> str:
        """Connect to an MCP server."""
        if server_name not in self.servers:
            return f"[Error]: Server {server_name} not configured"
        # Placeholder - actual implementation would use stdio or HTTP
        return f"Connected to {server_name}"

    def list_servers(self) -> list[str]:
        """List configured servers."""
        return list(self.servers.keys())

    def call_tool(self, server: str, tool: str, args: dict) -> str:
        """Call a tool on an MCP server."""
        if server not in self.servers:
            return f"[Error]: Server {server} not found"
        # Placeholder - actual implementation would use MCP protocol
        return f"Called {tool} on {server}"


# Global instance
_mcp_client: Optional[MCPClient] = None


def get_mcp_client(config_path: Optional[Path] = None) -> MCPClient:
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient(config_path)
    return _mcp_client


@tool
def mcp_connect(server_name: str) -> str:
    """Connect to an MCP server.

    Args:
        server_name: Server name from config
    """
    client = get_mcp_client()
    return client.connect(server_name)


@tool
def mcp_list() -> str:
    """List configured MCP servers."""
    client = get_mcp_client()
    servers = client.list_servers()
    if not servers:
        return "No MCP servers configured"
    return "Available servers: " + ", ".join(servers)


@tool
def mcp_call(server: str, tool: str, args: str = "{}") -> str:
    """Call a tool on an MCP server.

    Args:
        server: Server name
        tool: Tool name
        args: JSON string of arguments
    """
    client = get_mcp_client()
    try:
        args_dict = json.loads(args)
    except json.JSONDecodeError:
        return "[Error]: Invalid JSON for args"
    return client.call_tool(server, tool, args_dict)


MCP_TOOLS = [mcp_connect, mcp_list, mcp_call]