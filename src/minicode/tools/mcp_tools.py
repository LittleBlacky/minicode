"""MCP (Model Context Protocol) client tools - 使用 langchain-mcp-adapters."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional, Any

from langchain_core.tools import tool

WORKDIR = Path.cwd()
STORAGE_DIR = WORKDIR / ".mini-agent-cli"


class MultiServerMCP:
    """基于 langchain-mcp-adapters 的 MCP 客户端."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or STORAGE_DIR / "mcp_servers.json"
        self._client = None
        self._servers: dict[str, dict] = {}
        self._tools: list[Any] = []
        self._load_config()

    def _load_config(self) -> None:
        """从文件加载服务器配置."""
        if self.config_path.exists():
            try:
                config = json.loads(self.config_path.read_text(encoding="utf-8"))
                self._servers = config.get("servers", {})
            except Exception as e:
                print(f"[MCP] Failed to load config: {e}")

    def _save_config(self) -> None:
        """保存服务器配置到文件."""
        config = {"servers": self._servers}
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    async def _ensure_client(self):
        """确保客户端已初始化."""
        if self._client is None:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langchain_mcp_adapters.sessions import StdioConnection, SSEConnection, WebsocketConnection, StreamableHttpConnection

            connections = {}
            for name, cfg in self._servers.items():
                transport = cfg.get("transport")
                if transport == "stdio":
                    connections[name] = StdioConnection(
                        command=cfg.get("command", ""),
                        args=cfg.get("args", []),
                        env=cfg.get("env", {}),
                        transport="stdio",
                    )
                elif transport == "sse":
                    connections[name] = SSEConnection(url=cfg.get("url", ""), transport="sse")
                elif transport == "websocket":
                    connections[name] = WebsocketConnection(url=cfg.get("url", ""), transport="websocket")
                elif transport == "http":
                    connections[name] = StreamableHttpConnection(url=cfg.get("url", ""), transport="http")

            self._client = MultiServerMCPClient(connections=connections or None)

    async def connect(self, name: str, config: dict) -> str:
        """连接 MCP 服务器."""
        if name in self._servers:
            return f"[Error] Server '{name}' already exists"

        transport = config.get("transport")
        if transport not in ("stdio", "sse", "websocket", "http"):
            return f"[Error] Unknown transport: {transport}"

        self._servers[name] = config
        self._save_config()

        try:
            await self._ensure_client()
            return f"[MCP] Connected to {name} (call mcp_refresh to load tools)"
        except Exception as e:
            if name in self._servers:
                del self._servers[name]
                self._save_config()
            return f"[Error] Failed to connect: {e}"

    def disconnect(self, name: str) -> str:
        """断开 MCP 服务器."""
        if name not in self._servers:
            return f"[Error] Server '{name}' not found"

        del self._servers[name]
        self._save_config()

        self._client = None
        self._tools = []

        return f"[MCP] Disconnected from {name}"

    def list_servers(self) -> list[dict]:
        """列出所有已配置的服务器."""
        return [
            {"name": name, "config": cfg}
            for name, cfg in self._servers.items()
        ]

    def get_tools(self) -> list[Any]:
        """获取所有可用的 LangChain 工具."""
        return self._tools

    async def _refresh_async(self) -> int:
        """刷新工具列表（异步实现）."""
        if self._client is None:
            await self._ensure_client()

        if self._client is not None:
            self._tools = await self._client.get_tools()

        return len(self._tools)

    async def refresh(self) -> int:
        """刷新工具列表."""
        return await self._refresh_async()


# 全局 MCP 客户端实例
_mcp_client: Optional["MultiServerMCP"] = None


def get_mcp_client() -> MultiServerMCP:
    """获取全局 MCP 客户端实例."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MultiServerMCP()
    return _mcp_client


# ============ LangChain Tools (异步) ============

@tool
async def mcp_connect(
    server_name: str,
    transport: str,
    command: str = "",
    cmd_args: str = "",
    env: str = "",
    url: str = "",
) -> str:
    """Connect to an MCP server.

    Args:
        server_name: Unique name for this server (e.g., "filesystem", "git")
        transport: Transport type - "stdio", "sse", "websocket", or "http"
        command: For stdio transport - the command to run (e.g., "npx")
        cmd_args: For stdio transport - command arguments as space-separated string
        env: For stdio transport - environment variables as JSON string
        url: For sse/websocket/http transport - the server URL
    """
    client = get_mcp_client()

    cfg = {"transport": transport}
    if transport == "stdio":
        cfg["command"] = command
        args = cmd_args
        if isinstance(args, str):
            args = args.split() if args else []
        cfg["args"] = args
        if isinstance(env, str):
            try:
                env = json.loads(env) if env else {}
            except Exception:
                env = {}
        cfg["env"] = env or {}
    elif transport in ("sse", "websocket", "http"):
        cfg["url"] = url
    else:
        return f"[Error] Unknown transport: {transport}"

    result = await client.connect(server_name, cfg)

    if "Connected" in result:
        try:
            from minicode.agent.graph import refresh_mcp_tools
            refresh_mcp_tools()
        except Exception:
            pass

    return result


@tool
async def mcp_disconnect(server_name: str) -> str:
    """Disconnect from an MCP server."""
    client = get_mcp_client()
    result = client.disconnect(server_name)

    try:
        from minicode.agent.graph import refresh_mcp_tools
        refresh_mcp_tools()
    except Exception:
        pass

    return result


@tool
def mcp_list() -> str:
    """List all configured MCP servers and available tools."""
    client = get_mcp_client()
    servers = client.list_servers()

    if not servers:
        return """# MCP Servers

No servers configured. Use `mcp_connect` to add one:

**stdio example:**
```python
mcp_connect(
    server_name="filesystem",
    transport="stdio",
    command="npx",
    cmd_args="-y @modelcontextprotocol/server-filesystem /path"
)
```

**HTTP/SSE example:**
```python
mcp_connect(
    server_name="remote",
    transport="sse",
    url="https://mcp-server.example.com/sse"
)
```
"""

    lines = ["# MCP Servers", ""]
    for srv in servers:
        lines.append(f"- **{srv['name']}** ({srv['config']['transport']})")

    tools = client.get_tools()
    if tools:
        lines.append(f"\n# Available Tools ({len(tools)})")
        for t in tools:
            lines.append(f"- `{t.name}`: {t.description[:60]}...")

    return "\n".join(lines)


@tool
def mcp_get_tools() -> str:
    """Get all MCP tools as a formatted list for agent use."""
    client = get_mcp_client()
    tools = client.get_tools()

    if not tools:
        return "No MCP tools available. Connect to a server first."

    lines = [f"# MCP Tools ({len(tools)})", ""]
    for t in tools:
        lines.append(f"## {t.name}")
        lines.append(f"{t.description}")
        lines.append("")

    return "\n".join(lines)


@tool
async def mcp_refresh() -> str:
    """Refresh MCP tools from all connected servers."""
    client = get_mcp_client()
    count = await client.refresh()

    try:
        from minicode.agent.graph import refresh_mcp_tools
        refresh_mcp_tools()
        return f"[MCP] Refreshed, {count} tools available and registered"
    except Exception as e:
        return f"[MCP] Refreshed client, {count} tools available. Warning: {e}"


# 导出工具列表
MCP_TOOLS = [mcp_connect, mcp_disconnect, mcp_list, mcp_get_tools, mcp_refresh]