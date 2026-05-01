"""Model provider services using langchain init_chat_model."""
from __future__ import annotations

import os
from typing import Any, Optional
from dataclasses import dataclass

from minicode.services.config import get_config_manager


@dataclass
class ProviderConfig:
    """Model provider configuration."""
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
    """Create chat model using langchain init_chat_model.

    Priority: explicit params > config.json > env vars > defaults
    """
    from langchain.chat_models import init_chat_model

    config = get_config_manager()
    model_cfg = config.get_model_config()

    # Fallback chain: explicit > config > env > defaults
    provider = provider or model_cfg.get("provider") or "anthropic"
    model = model or model_cfg.get("model") or "claude-sonnet-4-7"
    api_key = api_key or model_cfg.get("api_key")
    base_url = base_url or model_cfg.get("base_url")

    # Build params
    params = {"timeout": timeout, "max_retries": max_retries}

    if api_key:
        params["api_key"] = api_key
    else:
        raise ValueError("API Key is required! Please configure via F5 in TUI.")

    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    return init_chat_model(model, model_provider=provider, **params)


class ChatProvider:
    """Chat provider wrapper."""

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
        """Get current configuration."""
        config = get_config_manager()
        model_cfg = config.get_model_config()
        return ProviderConfig(
            provider=self._provider or model_cfg.get("provider", "anthropic"),
            model=self._model or model_cfg.get("model", "claude-sonnet-4-7"),
            api_key=self._api_key or model_cfg.get("api_key"),
            base_url=self._base_url or model_cfg.get("base_url"),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )


# Aliases
create_provider = create_chat_model
ModelProvider = ChatProvider