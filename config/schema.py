"""Pydantic models for proxy-config.yaml validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ListenConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)


class AuthConfig(BaseModel):
    api_key: str | None = None


class ModelConfig(BaseModel):
    """One model entry — the `name` is the sole identifier.

    Fields:
        name: Unique model name (used as the OpenAI `model` identifier).
        url: Backend server URL.
        backend: Adapter type selector (e.g. "llama_cpp", "openai_compatible").
                 Determines how requests are forwarded and responses normalised.
        provider: Optional display hint for frontends like Open WebUI
                  (e.g. "llama.cpp"). Does NOT affect routing — use `backend`
                  for that.
        api_key: Optional per-model backend auth token (${VAR} expanded).
        default_params: Default sampling parameters merged into each request.
    """
    name: str
    url: str
    backend: str = "openai_compatible"  # adapter type selector
    provider: str | None = None  # OWUI provider hint (e.g. "llama.cpp")
    api_key: str | None = None  # per-model backend auth token (${VAR} expanded)
    default_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v


class ProxyConfig(BaseModel):
    listen: ListenConfig = Field(default_factory=ListenConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    models: list[ModelConfig] = Field(default_factory=list)
    plugins_dir: Path | None = None
    health_check_interval: float = Field(default=2.0, gt=0, description="Seconds between backend health probes")

    @field_validator("models")
    @classmethod
    def unique_model_names(cls, v: list[ModelConfig]) -> list[ModelConfig]:
        names = [m.name for m in v]
        if len(names) != len(set(names)):
            raise ValueError("models[].name must be globally unique")
        return v

    @field_validator("plugins_dir")
    @classmethod
    def validate_plugins_dir(cls, v: Path | None) -> Path | None:
        if v is not None and not v.is_dir():
            raise ValueError(f"plugins_dir does not exist or is not a directory: {v}")
        return v
