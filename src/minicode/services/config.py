"""Config service - unified configuration management."""
import json
import os
from pathlib import Path
from typing import Any, Optional


# 环境变量映射到配置路径
ENV_MAPPING = {
    "MINICODE_PROVIDER": "model.provider",
    "MINICODE_MODEL": "model.model",
    "MINICODE_API_KEY": "model.api_key",
    "MINICODE_BASE_URL": "model.base_url",
    "MINICODE_MODEL_PROVIDER": "model.provider",  # 别名
}


class ConfigManager:
    """Unified configuration management.

    All modules should read config through this class.
    Priority: config.json > environment variables > defaults
    """

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
        """Get config value with environment variable fallback.

        Priority: config.json value > environment variable > default
        """
        # Try config.json first
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = None
            if value is None:
                break

        # Fallback to environment variable
        if value is None:
            env_key = self._get_env_key(key)
            if env_key and env_key in os.environ:
                value = os.environ[env_key]

        return value if value is not None else default

    def _get_env_key(self, config_key: str) -> Optional[str]:
        """Map config key to environment variable."""
        for env_var, config_path in ENV_MAPPING.items():
            if config_path == config_key:
                return env_var
        return None

    def get_model_config(self) -> dict:
        """Get model configuration with all fallbacks applied."""
        return {
            "provider": self.get("model.provider", "anthropic"),
            "model": self.get("model.model", "claude-sonnet-4-7"),
            "api_key": self.get("model.api_key") or os.environ.get("MINICODE_API_KEY"),
            "base_url": self.get("model.base_url") or os.environ.get("MINICODE_BASE_URL"),
        }

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

    def reload(self) -> None:
        """Reload config from file."""
        self._config = self._load()


# Global instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def reset_config_manager() -> None:
    """Reset global config manager."""
    global _config_manager
    _config_manager = None