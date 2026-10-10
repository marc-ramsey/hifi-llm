"""Pydantic models for proxy-config.yaml validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ListenConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)


class CORSConfig(BaseModel):
    """CORS policy — allow all origins for development; lock down in production."""
    enabled: bool = True
    allow_origins: list[str] = Field(default=["*"], description="List of allowed origins. Use [\"*\"] for development.")
    allow_methods: list[str] = Field(default=["*"], description="Allowed HTTP methods.")
    allow_headers: list[str] = Field(default=["*"], description="Allowed request headers.")


class AuthConfig(BaseModel):
    api_key: str | None = None


class RateLimitConfig(BaseModel):
    enabled: bool = False
    requests_per_minute: int = Field(default=60, ge=1)
    endpoints: dict[str, int] = Field(
        default_factory=dict,
        description="Per-endpoint override (requests per minute). Keys are URL paths.",
    )


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
        verify_ssl: Whether to verify the backend's TLS certificate
                    (default True; set False for self-signed certs).
    """
    name: str
    url: str
    backend: str = "openai_compatible"  # adapter type selector
    provider: str | None = None  # OWUI provider hint (e.g. "llama.cpp")
    api_key: str | None = None  # per-model backend auth token (${VAR} expanded)
    default_params: dict[str, Any] = Field(default_factory=dict)
    verify_ssl: bool = True  # TLS certificate verification

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v


class StaticFileConfig(BaseModel):
    """Serve static files from one or more directories at a URL path."""
    path: str = "/"
    directories: list[Path]

    @field_validator("directories")
    @classmethod
    def validate_directories(cls, v: list[Path]) -> list[Path]:
        for d in v:
            if not d.is_dir():
                raise ValueError(f"static_files directory does not exist: {d}")
        return v


class ProxyConfig(BaseModel):
    listen: ListenConfig = Field(default_factory=ListenConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    models: list[ModelConfig] = Field(default_factory=list)
    static_files: list[StaticFileConfig] = Field(default_factory=list)
    health_check_interval: float = Field(default=2.0, gt=0, description="Seconds between backend health probes")

    @field_validator("models")
    @classmethod
    def unique_model_names(cls, v: list[ModelConfig]) -> list[ModelConfig]:
        names = [m.name for m in v]
        if len(names) != len(set(names)):
            raise ValueError("models[].name must be globally unique")
        return v
