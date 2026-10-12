"""Pydantic models for proxy-config.yaml validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


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
    """Auth policy — either a single global key or a per-model map."""
    api_key: str | dict[str, str] | None = None


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
        backend: Adapter type selector (e.g. "llama_cpp", "openai_compatible",
                 "managed_llama"). Determines how requests are forwarded and
                 responses normalised.
        provider: Optional display hint for frontends like Open WebUI
                  (e.g. "llama.cpp"). Does NOT affect routing — use `backend`
                  for that.
        api_key: Optional per-model backend auth token (${VAR} expanded).
        default_params: Parameters the proxy supplies when the request omits
                 them, applied to every request naming this model.  A value the
                 client sends always wins — including an explicit ``null``,
                 which means "no value chosen" and is also replaced.  Nested
                 dicts are merged key by key.  Every key the proxy adds is
                 reported in the ``X-Hifi-Defaulted`` response header and
                 counted in ``proxy_default_params_total``, so a default can
                 never take effect invisibly.
        server_args: Extra raw arguments appended to the ``llama-server``
                 command line for ``backend: managed_llama``.  Process
                 configuration, never part of a request body.
        verify_ssl: Whether to verify the backend's TLS certificate
                    (default True; set False for self-signed certs).
        llama_binary: Path or name of the llama-server binary. Required when
                      ``backend`` is ``managed_llama``.
        server_config: Server-level configuration in snake_case keys, mapped
                       to llama-server CLI flags at launch time
                       (e.g. {"ctx_size": 4096} → --ctx-size 4096).
    """
    name: str
    url: str
    backend: str = "openai_compatible"  # adapter type selector
    provider: str | None = None  # OWUI provider hint (e.g. "llama.cpp")
    api_key: str | None = None  # per-model backend auth token (${VAR} expanded)
    default_params: dict[str, Any] = Field(default_factory=dict)
    server_args: list[str] = Field(default_factory=list)  # raw llama-server args
    verify_ssl: bool = True  # TLS certificate verification
    llama_binary: str | None = None  # path to llama-server binary
    server_config: dict[str, Any] = Field(default_factory=dict)  # mapped to CLI flags

    @model_validator(mode="after")
    def _promote_legacy_server_args(self) -> "ModelConfig":
        """Move ``default_params.server_args`` to its own field.

        It used to be possible to hide launch arguments in ``default_params``,
        which the adapter read out before building the command line.  Now that
        ``default_params`` is genuinely applied to requests, that key would
        also be forwarded to the HTTP API on every call — a process flag
        leaking into request bodies.  Promote it and keep working.
        """
        if "server_args" in self.default_params:
            legacy = self.default_params.pop("server_args")
            if isinstance(legacy, str):
                legacy = [legacy]
            if not isinstance(legacy, list):
                raise ValueError(
                    f"model {self.name!r}: default_params.server_args must be a "
                    f"list of strings, got {type(legacy).__name__}; use the "
                    f"top-level server_args field instead"
                )
            logger.warning(
                "model %r: default_params.server_args is deprecated; moved to "
                "server_args=%s (launch arguments are never sent to the backend)",
                self.name,
                legacy,
            )
            self.server_args = [*self.server_args, *[str(a) for a in legacy]]
        return self

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v

    @field_validator("llama_binary")
    @classmethod
    def validate_llama_binary(cls, v: str | None, info) -> str | None:
        if info.data.get("backend") == "managed_llama" and not v:
            raise ValueError("llama_binary is required when backend is 'managed_llama'")
        return v


class StaticFileConfig(BaseModel):
    """Serve static files from one or more directories at a URL path.

    Directory existence is validated lazily at mount time (in create_app),
    so missing dirs are skipped with a warning instead of crashing startup.
    This also means static_files can be added via SIGHUP hot-reload.
    """
    path: str = "/"
    directories: list[str]  # validated at mount time, not at parse time


class ProxyConfig(BaseModel):
    listen: ListenConfig = Field(default_factory=ListenConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    models: list[ModelConfig] = Field(default_factory=list)
    static_files: list[StaticFileConfig] = Field(default_factory=list)
    health_check_interval: float = Field(default=2.0, gt=0, description="Seconds between backend health probes")
    max_body_size_mb: float = Field(
        default=10.0,
        gt=0,
        description="Largest request body accepted (MB). Larger bodies get HTTP 413. "
                    "Enforced on Content-Length and, for chunked uploads, while reading.",
    )

    @field_validator("models")
    @classmethod
    def unique_model_names(cls, v: list[ModelConfig]) -> list[ModelConfig]:
        names = [m.name for m in v]
        if len(names) != len(set(names)):
            raise ValueError("models[].name must be globally unique")
        return v
