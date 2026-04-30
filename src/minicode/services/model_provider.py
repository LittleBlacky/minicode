"""Model provider services using langchain's init_chat_model."""
import os
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from langchain.chat_models import init_chat_model


class ModelProvider(ABC):
    """Abstract model provider."""

    @abstractmethod
    async def ainvoke(self, messages: list, **kwargs) -> Any:
        """Async invoke."""
        pass

    @abstractmethod
    async def astream(self, messages: list, **kwargs) -> AsyncIterator[str]:
        """Async stream."""
        pass


class ChatModelProvider(ModelProvider):
    """Generic chat model provider using init_chat_model."""

    def __init__(self, model: str = "claude-sonnet-4-7", provider: str = "anthropic"):
        model_id = f"{provider}:{model}" if ":" not in model else model
        self.client = init_chat_model(model_id)

    async def ainvoke(self, messages: list, **kwargs) -> Any:
        return await self.client.ainvoke(messages, **kwargs)

    async def astream(self, messages: list, **kwargs) -> AsyncIterator[str]:
        async for chunk in self.client.astream(messages, **kwargs):
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)


def create_provider(provider: str = "anthropic", **kwargs) -> ModelProvider:
    """Factory to create model provider.

    Args:
        provider: Model provider name ("anthropic", "openai", "google", etc.)
        **kwargs: Additional arguments passed to init_chat_model

    Returns:
        ModelProvider instance
    """
    return ChatModelProvider(provider=provider, **kwargs)