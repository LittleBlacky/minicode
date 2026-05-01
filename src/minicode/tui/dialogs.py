"""Config dialog for MiniCode TUI - Single tab configuration interface."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Input,
    Label,
    Static,
)


class ConfigSaved(Message):
    """Message emitted when config is saved."""

    def __init__(self, config: dict) -> None:
        self.config = config
        super().__init__()


class ConfigDialog(Widget):
    """Interactive configuration dialog - 单界面设计"""

    CSS = """
    ConfigDialog {
        align: center middle;
        width: 70;
        height: auto;
        background: $surface;
        border: thick $primary;
        border-radius: 8px;
        padding: 1 2;
    }

    ConfigDialog #title-bar {
        width: 100%;
        height: 1;
        background: $primary;
        color: $text;
        content-align: center middle;
    }

    ConfigDialog #close-hint {
        color: $text-muted;
    }

    ConfigDialog Button {
        margin: 1 0;
    }

    ConfigDialog Input {
        margin-bottom: 1;
    }

    ConfigDialog #save-btn {
        background: $success;
    }
    """

    # Configuration values
    provider = reactive("anthropic")
    model = reactive("claude-sonnet-4-7")
    api_key = reactive("")
    base_url = reactive("")

    BINDINGS = [
        ("escape", "close", "Close"),
        ("ctrl+s", "save", "Save"),
    ]

    def __init__(self):
        super().__init__()
        self._closed = False
        self._result = None
        self._load_config()

    def _load_config(self) -> None:
        """Load current configuration."""
        from minicode.services.config import get_config_manager

        try:
            config = get_config_manager()
            model_cfg = config.get_model_config()
            self.provider = model_cfg.get("provider", "anthropic")
            self.model = model_cfg.get("model", "claude-sonnet-4-7")
            self.api_key = model_cfg.get("api_key") or ""
            self.base_url = model_cfg.get("base_url") or ""
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        """Create the dialog UI - 单一布局"""
        # Title bar
        with Horizontal(id="title-bar"):
            yield Static("[bold]Configuration[/bold]", id="title-text")
            yield Static("[dim]  |  Esc: Close  |  Ctrl+S: Save[/dim]", id="close-hint")

        # Provider (用户可输入)
        yield Label("[bold]Provider:[/bold]")
        yield Input(
            value=self.provider,
            id="provider-input",
            placeholder="anthropic, openai, ollama, deepseek, groq, gemini...",
        )

        # Model name
        yield Label("[bold]Model:[/bold]")
        yield Input(value=self.model, id="model-input", placeholder="e.g., claude-sonnet-4-7")

        # API Key
        yield Label("[bold]API Key:[/bold]")
        yield Input(
            value=self.api_key,
            id="api-key-input",
            placeholder="sk-xxxx (optional, uses env var if empty)",
            password=True,
        )

        # Base URL
        yield Label("[bold]Base URL:[/bold]")
        yield Input(
            value=self.base_url,
            id="base-url-input",
            placeholder="https://api.anthropic.com/v1 (optional)",
        )

        # Note
        yield Label("[yellow]Config auto-reloads on save.[/yellow]", id="note")

        # Action buttons
        with Horizontal(id="button-row"):
            yield Button("Cancel", id="cancel-btn", variant="error")
            yield Button("Save", id="save-btn", variant="success")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        input_map = {
            "provider-input": "provider",
            "model-input": "model",
            "api-key-input": "api_key",
            "base-url-input": "base_url",
        }
        field = input_map.get(event.control.id)
        if field:
            setattr(self, field, event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self.action_close()
        elif event.button.id == "save-btn":
            self.action_save()

    def get_config(self) -> dict:
        """Get current configuration values."""
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
        }

    def action_close(self) -> None:
        """Close the dialog."""
        self._closed = True
        self.remove()

    def action_save(self) -> None:
        """Save configuration and close."""
        import os
        from minicode.services.config import get_config_manager

        try:
            config = get_config_manager()
            config.set("model.provider", self.provider)
            config.set("model.model", self.model)

            # API Key 和 Base URL 通过环境变量传递（create_chat_model 读取 env）
            if self.api_key:
                os.environ["MINICODE_API_KEY"] = self.api_key
            if self.base_url:
                os.environ["MINICODE_BASE_URL"] = self.base_url

            self._result = self.get_config()

            # 发送配置保存消息，触发热重载
            self.post_message(ConfigSaved(self._result))
        except Exception as e:
            self._result = {"error": str(e)}

        self.action_close()

    @property
    def is_closed(self) -> bool:
        """Check if dialog is closed."""
        return self._closed

    @property
    def result(self) -> dict:
        """Get saved configuration result."""
        return self._result