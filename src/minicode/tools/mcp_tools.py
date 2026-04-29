"""MCP (Model Context Protocol) client tools - Full implementation."""
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass


WORKDIR = Path.cwd()
STORAGE_DIR = WORKDIR / ".mini-agent-cli"


@dataclass
class MCPTool:
    """Represents a tool from MCP server."""
    name: str
    description: str
    input_schema: dict


class MCPConnection:
    """Manages a connection to an MCP server via stdio."""

    def __init__(self, name: str, command: list[str]):
        self.name = name
        self.command = command
        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._request_id = 0
        self._callbacks: dict[str, callable] = {}

    def connect(self) -> bool:
        """Start the MCP server process."""
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=WORKDIR,
            )
            # Send initialize request
            result = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "minicode", "version": "0.1.0"},
            })
            return result is not None
        except Exception as e:
            print(f"[MCP] Failed to connect to {self.name}: {e}")
            return False

    def _send_request(self, method: str, params: dict = None) -> Optional[dict]:
        """Send a JSON-RPC request and wait for response."""
        if not self.process or self.process.poll() is not None:
            return None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }

        with self._lock:
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()

                # Read response
                line = self.process.stdout.readline()
                if line:
                    return json.loads(line)
            except Exception as e:
                print(f"[MCP] Request failed: {e}")
        return None

    def list_tools(self) -> list[MCPTool]:
        """List available tools from the server."""
        result = self._send_request("tools/list")
        if result and "result" in result:
            tools = []
            for t in result["result"].get("tools", []):
                tools.append(MCPTool(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                ))
            return tools
        return []

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the MCP server."""
        result = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        if result and "result" in result:
            content = result["result"].get("content", [])
            if content:
                return content[0].get("text", "")
            return "Tool executed successfully"
        return "[Error] Tool call failed"

    def disconnect(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None


class MCPClient:
    """Client for managing multiple MCP server connections."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or STORAGE_DIR / "mcp_servers.json"
        self.servers: dict[str, MCPConnection] = {}
        self.available_tools: dict[str, list[MCPTool]] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load server configuration from file."""
        if self.config_path.exists():
            try:
                config = json.loads(self.config_path.read_text(encoding="utf-8"))
                servers_config = config.get("servers", {})
                for name, cfg in servers_config.items():
                    command = cfg.get("command")
                    if command:
                        self.servers[name] = MCPConnection(name, command)
            except Exception as e:
                print(f"[MCP] Failed to load config: {e}")

    def _save_config(self) -> None:
        """Save server configuration to file."""
        config = {
            "servers": {
                name: {"command": conn.command}
                for name, conn in self.servers.items()
            }
        }
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def add_server(self, name: str, command: list[str]) -> str:
        """Add and connect a new MCP server."""
        if name in self.servers:
            return f"[Error] Server {name} already exists"

        conn = MCPConnection(name, command)
        if not conn.connect():
            return f"[Error] Failed to connect to {name}"

        self.servers[name] = conn
        self.available_tools[name] = conn.list_tools()
        self._save_config()
        return f"[MCP] Connected to {name} with {len(self.available_tools[name])} tools"

    def remove_server(self, name: str) -> str:
        """Disconnect and remove an MCP server."""
        if name not in self.servers:
            return f"[Error] Server {name} not found"

        self.servers[name].disconnect()
        del self.servers[name]
        if name in self.available_tools:
            del self.available_tools[name]
        self._save_config()
        return f"[MCP] Disconnected from {name}"

    def list_servers(self) -> list[dict]:
        """List all configured servers."""
        result = []
        for name, conn in self.servers.items():
            tools = self.available_tools.get(name, [])
            result.append({
                "name": name,
                "connected": conn.process is not None,
                "tool_count": len(tools),
            })
        return result

    def list_all_tools(self) -> list[dict]:
        """List all available tools from all servers."""
        result = []
        for name, tools in self.available_tools.items():
            for t in tools:
                result.append({
                    "server": name,
                    "name": t.name,
                    "description": t.description,
                })
        return result

    def call_tool(self, server: str, tool: str, args: dict) -> str:
        """Call a tool on a specific server."""
        if server not in self.servers:
            return f"[Error] Server {server} not found"
        return self.servers[server].call_tool(tool, args)

    def disconnect_all(self):
        """Disconnect all servers."""
        for conn in self.servers.values():
            conn.disconnect()


# Global client instance
_mcp_client: Optional[MCPClient] = None


def get_mcp_client(config_path: Optional[Path] = None) -> MCPClient:
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient(config_path)
    return _mcp_client


# LangChain tools
from langchain_core.tools import tool


@tool
def mcp_connect(server_name: str, command: str) -> str:
    """Connect to an MCP server.

    Args:
        server_name: Unique name for this server
        command: Command to start the MCP server (e.g., ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path"])
    """
    import shlex
    client = get_mcp_client()
    try:
        cmd_list = shlex.split(command) if isinstance(command, str) else command
    except Exception:
        return "[Error] Invalid command format"

    return client.add_server(server_name, cmd_list)


@tool
def mcp_disconnect(server_name: str) -> str:
    """Disconnect from an MCP server.

    Args:
        server_name: Name of the server to disconnect
    """
    client = get_mcp_client()
    return client.remove_server(server_name)


@tool
def mcp_list() -> str:
    """List all connected MCP servers and their tools."""
    client = get_mcp_client()
    servers = client.list_servers()

    if not servers:
        return "No MCP servers configured. Use mcp_connect to add one."

    lines = ["# MCP Servers", ""]
    for srv in servers:
        status = "✓" if srv["connected"] else "✗"
        lines.append(f"- {status} {srv['name']}: {srv['tool_count']} tools")

    all_tools = client.list_all_tools()
    if all_tools:
        lines.append("\n# Available Tools")
        for t in all_tools[:20]:
            lines.append(f"  - [{t['server']}] {t['name']}: {t['description'][:50]}")

    return "\n".join(lines)


@tool
def mcp_call(server: str, tool: str, args: str = "{}") -> str:
    """Call a tool on an MCP server.

    Args:
        server: Server name
        tool: Tool name on that server
        args: JSON string of arguments
    """
    client = get_mcp_client()
    try:
        args_dict = json.loads(args)
    except json.JSONDecodeError:
        return "[Error] Invalid JSON for args"

    return client.call_tool(server, tool, args_dict)


MCP_TOOLS = [mcp_connect, mcp_disconnect, mcp_list, mcp_call]