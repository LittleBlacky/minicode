"""Model provider services - 基于 langchain init_chat_model

使用 langchain 的统一接口管理多提供商
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Optional
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """模型提供商配置"""

    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


def create_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    **kwargs,
) -> Any:
    """创建 Chat Model 实例 - 使用 langchain init_chat_model

    优先级: 外部传参 > config.json > 系统环境变量 > 硬编码默认值

    Args:
        provider: 提供商名称 (anthropic, openai, deepseek, ollama, groq, gemini 等)
        model: 模型名称
        ...
    """
    from langchain.chat_models import init_chat_model
    from minicode.services.config import get_config_manager

    config = get_config_manager()

    # 配置回退链: 显式参数 > config.json > 环境变量 > 硬编码默认值
    _cfg_provider = config.get("model.provider") or os.environ.get("MINICODE_PROVIDER") or "anthropic"
    _cfg_model = config.get("model.model") or os.environ.get("MINICODE_MODEL") or "claude-sonnet-4-7"
    _cfg_api_key = config.get("model.api_key") or os.environ.get("MINICODE_API_KEY")
    _cfg_base_url = config.get("model.base_url") or os.environ.get("MINICODE_BASE_URL")

    # 优先级: 外部传参 > 配置文件/环境变量
    provider = provider or _cfg_provider
    model = model or _cfg_model
    api_key = api_key or _cfg_api_key
    base_url = base_url or _cfg_base_url

    # 构建 model_id 和参数 - 参考官方示例
    # init_chat_model(model="deepseek-chat", model_provider="deepseek", api_key=...)
    if provider == "anthropic":
        model_id = model  # Anthropic 直接用模型名
    else:
        model_id = model  # 只需传模型名，不要 provider 前缀

    # 构建参数
    params = {"timeout": timeout, "max_retries": max_retries}

    if api_key:
        params["api_key"] = api_key
    else:
        raise ValueError("API Key is required! 请在 TUI 中按 F5 配置 API Key。")
    # 注意：不需要传 base_url，init_chat_model 会使用提供商默认地址
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    # 关键：使用 model_provider 参数指定提供商
    return init_chat_model(model_id, model_provider=provider, **params)


class ChatProvider:
    """聊天提供商封装类 - 所有配置从环境变量读取"""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        self._client: Optional[Any] = None
        # 所有参数由 create_chat_model 内部从环境变量读取
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = create_chat_model(
                provider=self._provider,
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                **self.kwargs,
            )
        return self._client

    def reset(self) -> None:
        self._client = None

    def invoke(self, messages: list, **kwargs):
        return self.client.invoke(messages, **kwargs)

    def stream(self, messages: list, **kwargs):
        return self.client.stream(messages, **kwargs)

    def bind_tools(self, tools: list) -> Any:
        return self.client.bind_tools(tools)

    def get_config(self) -> ProviderConfig:
        """获取当前配置 - 使用与 create_chat_model 相同的回退链"""
        from minicode.services.config import get_config_manager

        config = get_config_manager()
        _default_provider = config.get("model.provider") or os.environ.get("MINICODE_PROVIDER") or "anthropic"
        _default_model = config.get("model.model") or os.environ.get("MINICODE_MODEL") or "claude-sonnet-4-7"
        return ProviderConfig(
            provider=self._provider or _default_provider,
            model=self._model or _default_model,
            api_key=self._api_key or os.environ.get("MINICODE_API_KEY"),
            base_url=self._base_url or os.environ.get("MINICODE_BASE_URL"),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )


# 别名
create_provider = create_chat_model
ModelProvider = ChatProvider
