"""Model provider services - 优化的 LLM API 调用层

使用 LangChain 的 init_chat_model 统一管理多提供商
"""
from __future__ import annotations

import os
from typing import Any, AsyncIterator, Optional
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel


# 模型别名映射
MODEL_ALIASES: dict[str, str] = {
    # Claude
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-haiku-4": "claude-haiku-4-20250514",
    # OpenAI
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    # DeepSeek
    "deepseek-chat": "deepseek-chat",
    "deepseek-coder": "deepseek-coder",
}


@dataclass
class ProviderConfig:
    """模型提供商配置"""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra_body: Optional[dict] = None
    timeout: float = 60.0
    max_retries: int = 3
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    streaming: bool = True


@dataclass
class ProviderInfo:
    """提供商信息"""
    name: str
    default_model: str
    base_url: Optional[str] = None
    env_api_key: str = "API_KEY"
    features: list[str] = field(default_factory=list)


# 提供商注册表
PROVIDERS: dict[str, ProviderInfo] = {
    "anthropic": ProviderInfo(
        name="anthropic",
        default_model="claude-sonnet-4-7",
        base_url="https://api.anthropic.com",
        env_api_key="ANTHROPIC_API_KEY",
        features=["vision", "tool_use", "streaming", "thinking"],
    ),
    "openai": ProviderInfo(
        name="openai",
        default_model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        env_api_key="OPENAI_API_KEY",
        features=["vision", "tool_use", "streaming"],
    ),
    "deepseek": ProviderInfo(
        name="deepseek",
        default_model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        env_api_key="DEEPSEEK_API_KEY",
        features=["tool_use", "streaming"],
    ),
    "ollama": ProviderInfo(
        name="ollama",
        default_model="llama3",
        base_url="http://localhost:11434/v1",
        env_api_key="",
        features=["local", "streaming"],
    ),
    "groq": ProviderInfo(
        name="groq",
        default_model="llama-3.1-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        env_api_key="GROQ_API_KEY",
        features=["tool_use", "streaming", "fast"],
    ),
    "gemini": ProviderInfo(
        name="gemini",
        default_model="gemini-2.0-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        env_api_key="GOOGLE_API_KEY",
        features=["vision", "tool_use", "streaming"],
    ),
}


def _normalize_model(provider: str, model: str) -> str:
    """标准化模型名称"""
    # 如果模型已经包含冒号，说明格式是 provider:model
    if ":" in model:
        parts = model.split(":", 1)
        provider = parts[0]
        model = parts[1]

    # 应用别名
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]

    return model


def _get_api_key(provider: str, info: ProviderInfo) -> Optional[str]:
    """获取 API Key"""
    # 1. 环境变量 MINICODE_API_KEY (最高优先级)
    if key := os.environ.get("MINICODE_API_KEY"):
        return key

    # 2. 提供商专用环境变量
    if info.env_api_key and (key := os.environ.get(info.env_api_key)):
        return key

    # 3. 通用的环境变量 (兜底)
    if key := os.environ.get("OPENAI_API_KEY"):
        return key

    return None


def _get_base_url(provider: str, info: ProviderInfo) -> Optional[str]:
    """获取 Base URL"""
    # 1. 环境变量 MINICODE_BASE_URL (最高优先级)
    if url := os.environ.get("MINICODE_BASE_URL"):
        return url

    # 2. 提供商专用环境变量
    if url := os.environ.get(f"{provider.upper()}_BASE_URL"):
        return url

    # 3. 默认 base_url
    return info.base_url


def create_chat_model(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-7",
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    extra_body: Optional[dict] = None,
    **kwargs,
) -> BaseChatModel:
    """创建 Chat Model 实例 - 工厂函数

    Args:
        provider: 提供商名称 (anthropic, openai, deepseek, ollama, groq, gemini)
        model: 模型名称，支持 "provider:model" 格式
        api_key: API Key (可选，默认从环境变量读取)
        base_url: Base URL (可选)
        temperature: 温度参数
        max_tokens: 最大 token 数
        timeout: 超时时间(秒)
        max_retries: 最大重试次数
        extra_body: 额外参数
        **kwargs: 其他参数

    Returns:
        BaseChatModel 实例

    Examples:
        >>> # 基本用法
        >>> model = create_chat_model("anthropic", "claude-sonnet-4-7")
        >>>
        >>> # 使用 provider:model 格式
        >>> model = create_chat_model("openai:gpt-4o")
        >>>
        >>> # 自定义参数
        >>> model = create_chat_model(
        ...     "anthropic",
        ...     "claude-opus-4",
        ...     temperature=0.7,
        ...     max_tokens=4096,
        ... )
    """
    # 标准化模型名称
    model = _normalize_model(provider, model)

    # 获取提供商信息
    info = PROVIDERS.get(provider)
    if not info:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS.keys())}")

    # 获取配置
    api_key = api_key or _get_api_key(provider, info)
    base_url = base_url or _get_base_url(provider, info)

    # 构建模型 ID (provider/model 或直接 model)
    model_id = f"{provider}/{model}"

    # 根据提供商创建不同的客户端
    if provider == "anthropic":
        return _create_anthropic_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            extra_body=extra_body,
            **kwargs,
        )
    elif provider == "openai":
        return _create_openai_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
    else:
        # 使用 LangChain 的 init_chat_model (通用方式)
        return _create_generic_model(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            extra_body=extra_body,
            **kwargs,
        )


def _create_anthropic_model(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    timeout: float,
    max_retries: int,
    extra_body: Optional[dict],
    **kwargs,
) -> ChatAnthropic:
    """创建 Anthropic 模型"""
    # 构建 model_kwargs (ChatAnthropic 特定参数)
    model_kwargs = kwargs.pop("model_kwargs", {})
    model_kwargs.setdefault("extra_headers", {"anthropic-version": "2023-06-01"})

    return ChatAnthropic(
        model=model,
        anthropic_api_key=api_key,
        anthropic_api_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens or 8192,
        timeout=timeout,
        max_retries=max_retries,
        model_kwargs=model_kwargs,
        **kwargs,
    )


def _create_openai_model(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    timeout: float,
    max_retries: int,
    **kwargs,
) -> ChatOpenAI:
    """创建 OpenAI 兼容模型"""
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
    )


def _create_generic_model(
    provider: str,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    timeout: float,
    max_retries: int,
    extra_body: Optional[dict],
    **kwargs,
) -> BaseChatModel:
    """使用 init_chat_model 创建通用模型"""
    from langchain.chat_models import init_chat_model

    model_id = f"{provider}:{model}"

    params = {
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_retries": max_retries,
    }

    # 过滤 None 值
    params = {k: v for k, v in params.items() if v is not None}

    if extra_body:
        params["extra_body"] = extra_body

    return init_chat_model(model_id, **params)


# ============== 高级封装 ==============

class ChatProvider:
    """聊天提供商封装类 - 提供更简洁的 API"""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-7",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        self.provider = provider
        self.model = model
        self.kwargs = kwargs

        # 延迟初始化客户端
        self._client: Optional[BaseChatModel] = None
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def client(self) -> BaseChatModel:
        """延迟加载的客户端"""
        if self._client is None:
            self._client = create_chat_model(
                provider=self.provider,
                model=self.model,
                api_key=self._api_key,
                base_url=self._base_url,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                **self.kwargs,
            )
        return self._client

    def reset(self) -> None:
        """重置客户端（用于切换模型）"""
        self._client = None

    async def ainvoke(self, messages: list[BaseMessage], **kwargs) -> BaseMessage:
        """异步调用"""
        return await self.client.ainvoke(messages, **kwargs)

    async def astream(self, messages: list[BaseMessage], **kwargs) -> AsyncIterator[str]:
        """异步流式调用"""
        async for chunk in self.client.astream(messages, **kwargs):
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)

    def invoke(self, messages: list[BaseMessage], **kwargs) -> BaseMessage:
        """同步调用"""
        return self.client.invoke(messages, **kwargs)

    def stream(self, messages: list[BaseMessage], **kwargs) -> AsyncIterator[str]:
        """同步流式调用"""
        for chunk in self.client.stream(messages, **kwargs):
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)

    def bind_tools(self, tools: list) -> BaseChatModel:
        """绑定工具"""
        return self.client.bind_tools(tools)

    def with_structured_output(self, schema: type) -> BaseChatModel:
        """结构化输出"""
        return self.client.with_structured_output(schema)

    def get_config(self) -> ProviderConfig:
        """获取配置"""
        return ProviderConfig(
            provider=self.provider,
            model=self.model,
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self.kwargs.get("timeout", 60.0),
            max_retries=self.kwargs.get("max_retries", 3),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    def __repr__(self) -> str:
        return f"ChatProvider(provider={self.provider}, model={self.model})"


# ============== 便捷函数 ==============

@lru_cache(maxsize=4)
def get_default_provider() -> ChatProvider:
    """获取默认提供商 (带缓存)"""
    return ChatProvider()


def list_providers() -> list[ProviderInfo]:
    """列出所有支持的提供商"""
    return list(PROVIDERS.values())


def test_connection(provider: str, model: str) -> dict[str, Any]:
    """测试连接

    Returns:
        {"success": bool, "latency": float, "error": str or None, "model": str}
    """
    import time
    from langchain_core.messages import HumanMessage

    try:
        client = create_chat_model(provider, model)
        start = time.time()

        import asyncio
        result = asyncio.run(client.ainvoke([HumanMessage(content="Hi")]))

        latency = time.time() - start

        return {
            "success": True,
            "latency": round(latency, 3),
            "model": model,
            "response_length": len(result.content) if hasattr(result, "content") else 0,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": model,
        }


# ============== 向后兼容 ==============

def create_provider(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-7",
    cache: bool = False,
    **kwargs,
) -> ChatProvider:
    """工厂函数 - 向后兼容

    Args:
        provider: 提供商名称
        model: 模型名称
        cache: 是否启用缓存 (保留参数，向后兼容)
        **kwargs: 额外参数
    """
    return ChatProvider(provider=provider, model=model, **kwargs)


# 别名
ModelProvider = ChatProvider