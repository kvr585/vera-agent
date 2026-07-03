"""Configuration loader using Pydantic Settings and PyYAML."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderSettings(BaseModel):
    """Configuration settings for LLM and Model backends."""

    name: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    timeout_seconds: float = 60.0


class Settings(BaseSettings):
    """VERA Engine system configuration schema."""

    database_url: str = "sqlite:///logs/vera.db"
    workspace_dir: str = "workspace"
    prompts_dir: str = "prompts"
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    debug: bool = False

    # Configuration binding: matches environment prefix VERA_ (e.g. VERA_DEBUG=true)
    model_config = SettingsConfigDict(env_prefix="VERA_", env_nested_delimiter="__")


def load_settings(config_path: Path | None = None) -> Settings:
    """Loads system settings, merging YAML file config and Env overrides.

    Args:
        config_path: Optional path to the config.yaml file.

    Returns:
        An initialized Settings model.
    """
    init_kwargs = {}
    if config_path:
        path = Path(config_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                init_kwargs = yaml.safe_load(f) or {}

    return Settings(**init_kwargs)
