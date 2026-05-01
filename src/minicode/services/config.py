"""Config service for managing settings."""
import json
from pathlib import Path
from typing import Any, Optional


class ConfigManager:
    """Manage application configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".minicode" / "config.json"
        self._config = self._load()

    def _load(self) -> dict:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return self._default_config()

    def _default_config(self) -> dict:
        return {
            "model": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-7",
            },
            "permissions": {
                "mode": "default",
            },
            "storage": {
                "dir": str(Path.home() / ".minicode"),
            },
            "features": {
                "auto_compact": True,
                "team_enabled": False,
                "skills_enabled": True,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-notation key.

        直接从 config.json 读取，不回退到环境变量。
        环境变量回退由调用方在需要时自行处理。
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = None
            if value is None:
                break

        return value if value is not None else default

    def _get_from_env(self, key: str) -> Optional[Any]:
        """已废弃 - 仅保留签名兼容，内部不再使用"""
        return None

    def set(self, key: str, value: Any) -> None:
        """Set config value by dot-notation key."""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save()

    def save(self) -> None:
        """Save config to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(self._config, indent=2), encoding="utf-8")


# Global instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager
