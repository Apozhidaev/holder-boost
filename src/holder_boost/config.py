"""Application settings loaded from config.toml with ENV overrides.

Settings are loaded once at process start. ENV variables follow the nested
delimiter convention (e.g. ``LOGGING__LEVEL``) and override values from
``config.toml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["json", "pretty"]


class PolyHouseSettings(BaseModel):
    base_url: str = "http://127.0.0.1:8000"


class HttpSettings(BaseModel):
    timeout_seconds: float = 10.0


class LoggingSettings(BaseModel):
    level: LogLevel = "INFO"
    format: LogFormat = "pretty"


class TrainSettings(BaseModel):
    """``train``: CatBoost model on holder distribution rows from poly-house."""

    tag_id: int
    model_dir: str = ".models"
    dataset_dir: str = ".datasets"
    neg_risk: bool = False
    exclude_tag_ids: list[int] = Field(default_factory=list)
    min_duration_days: int = 0
    holders_limit: int | None = None
    min_wallet_count: int = 8
    n_trials: int = 70
    n_folds: int = 5
    random_state: int = 42


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        toml_file=CONFIG_PATH,
        extra="ignore",
    )

    poly_house: PolyHouseSettings = Field(default_factory=PolyHouseSettings)
    http: HttpSettings = Field(default_factory=HttpSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    train: TrainSettings
    api: ApiSettings = Field(default_factory=ApiSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # ENV must override TOML — env_settings goes first.
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return process-wide singleton Settings, loading on first access."""
    global _settings
    if _settings is None:
        _settings = Settings()  # pyright: ignore[reportCallIssue, reportArgumentType]
    return _settings
