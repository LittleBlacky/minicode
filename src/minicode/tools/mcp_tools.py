"""MCP (Model Context Protocol) client tools - 标准化接口，支持热更新."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable, Optional, Any

from langchain_core.tools import BaseTool

WORKDIR = Path.cwd()
STORAGE_DIR = WORKDIR / ".minicode"


class MCPProvider:
    """MCP 工具提供者 - 标准化接口，支持热更新。

    事件驱动的工具更新:
    - connect/disconnect 时自动触发更新
    - 通过 subscribe() 订阅工具变更事件
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or STORAGE_DIR / "mcp_servers.json"
        self._client = None
        self._servers: dict[str, dict] = {}
        self._tools: list[Any] = []
        self._subscribers: list[Callable[[list[BaseTool]], None]] = []
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

    # ============ 事件订阅 ============

    def subscribe(self, callback: Callable[[list[BaseTool]], None]) -> None:
        """订阅工具变更事件.

        Args:
            callback: 工具变更时的回调函数，参数为新的工具列表
        """
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[list[BaseTool]], None]) -> None:
        """取消订阅."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _notify_changed(self) -> None:
        """通知所有订阅者工具已更新."""
        for callback in self._subscribers:
            try:
                callback(self._tools)
            except Exception as e:
                print(f"[MCP] Subscriber error: {e}")

    # ============ 客户端管理 ============

    async def _ensure_client(self) -> bool:
        """确保客户端已初始化.

        Returns:
            True if client initialized, False otherwise
        """
        if self._client is None:
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient
                from langchain_mcp_adapters.sessions import (
                    StdioConnection,
                    SSEConnection,
                    WebsocketConnection,
                    StreamableHttpConnection,
                )

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
                        connections[name] = SSEConnection(
                            url=cfg.get("url", ""), transport="sse"
                        )
                    elif transport == "websocket":
                        connections[name] = WebsocketConnection(
                            url=cfg.get("url", ""), transport="websocket"
                        )
                    elif transport == "http":
                        connections[name] = StreamableHttpConnection(
                            url=cfg.get("url", ""), transport="http"
                        )

                self._client = MultiServerMCPClient(connections=connections or None)
                return True
            except ImportError as e:
                print(f"[MCP] langchain-mcp-adapters not installed: {e}")
                return False
            except Exception as e:
                print(f"[MCP] Failed to init client: {e}")
                return False
        return True

    # ============ 连接管理 ============

    async def connect(self, name: str, config: dict) -> str:
        """连接 MCP 服务器.

        Args:
            name: 服务器名称
            config: 服务器配置 {transport, command, args, env, url}

        Returns:
            连接结果消息
        """
        if name in self._servers:
            return f"[Error] Server '{name}' already exists"

        transport = config.get("transport")
        if transport not in ("stdio", "sse", "websocket", "http"):
            return f"[Error] Unknown transport: {transport}"

        self._servers[name] = config
        self._save_config()

        try:
            # 重新初始化客户端
            self._client = None
            if await self._ensure_client():
                await self.refresh()
                self._notify_changed()
                return f"[MCP] Connected to {name}"
            else:
                return f"[MCP] Config saved, but langchain-mcp-adapters not available"
        except Exception as e:
            if name in self._servers:
                del self._servers[name]
                self._save_config()
            return f"[Error] Failed to connect: {e}"

    def disconnect(self, name: str) -> str:
        """断开 MCP 服务器.

        Args:
            name: 服务器名称

        Returns:
            断开结果消息
        """
        if name not in self._servers:
            return f"[Error] Server '{name}' not found"

        del self._servers[name]
        self._save_config()

        self._client = None
        self._tools = []
        self._notify_changed()

        return f"[MCP] Disconnected from {name}"

    def list_servers(self) -> list[dict]:
        """列出所有已配置的服务器."""
        return [{"name": name, "config": cfg} for name, cfg in self._servers.items()]

    # ============ 工具访问 ============

    @property
    def tools(self) -> list[Any]:
        """获取所有可用的 LangChain 工具."""
        return self._tools

    async def refresh(self) -> int:
        """刷新工具列表.

        Returns:
            获取到的工具数量
        """
        if not await self._ensure_client():
            return 0

        try:
            self._tools = await self._client.get_tools()
            return len(self._tools)
        except Exception as e:
            print(f"[MCP] Refresh failed: {e}")
            return 0

    # ============ 便捷方法 ============

    def get_tools_info(self) -> str:
        """获取工具信息用于显示."""
        if not self._tools:
            return "No MCP tools available."

        lines = [f"# MCP Tools ({len(self._tools)})", ""]
        for t in self._tools:
            desc = t.description[:60] if t.description else "No description"
            lines.append(f"- **{t.name}**: {desc}...")
        return "\n".join(lines)


# ============ 全局实例 ============

_mcp_provider: Optional["MCPProvider"] = None


def get_mcp_provider() -> "MCPProvider":
    """获取全局 MCPProvider 实例."""
    global _mcp_provider
    if _mcp_provider is None:
        _mcp_provider = MCPProvider()
    return _mcp_provider


def reset_mcp_provider() -> None:
    """重置全局实例 (用于测试)."""
    global _mcp_provider
    _mcp_provider = None


# ============ LangChain Tools ============

from langchain_core.tools import tool


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
    provider = get_mcp_provider()

    config = {"transport": transport}
    if transport == "stdio":
        config["command"] = command
        args = cmd_args.split() if isinstance(cmd_args, str) and cmd_args else []
        config["args"] = args
        if isinstance(env, str) and env:
            try:
                config["env"] = json.loads(env)
            except Exception:
                config["env"] = {}
        else:
            config["env"] = {}
    elif transport in ("sse", "websocket", "http"):
        config["url"] = url
    else:
        return f"[Error] Unknown transport: {transport}"

    return await provider.connect(server_name, config)


@tool
def mcp_disconnect(server_name: str) -> str:
    """Disconnect from an MCP server."""
    provider = get_mcp_provider()
    return provider.disconnect(server_name)


@tool
def mcp_list() -> str:
    """List all configured MCP servers."""
    provider = get_mcp_provider()
    servers = provider.list_servers()

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

    if provider.tools:
        lines.append(f"\n# Available Tools ({len(provider.tools)})")
        for t in provider.tools:
            lines.append(f"- `{t.name}`: {t.description[:60]}...")

    return "\n".join(lines)


@tool
async def mcp_refresh() -> str:
    """Refresh MCP tools from all connected servers."""
    provider = get_mcp_provider()
    count = await provider.refresh()
    if count > 0:
        provider._notify_changed()
    return f"[MCP] Refreshed, {count} tools available"


# 导出工具列表
MCP_TOOLS = [mcp_connect, mcp_disconnect, mcp_list, mcp_refresh]