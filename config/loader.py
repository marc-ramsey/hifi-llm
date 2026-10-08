"""Load and parse proxy-config.yaml with ${VAR} expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from .schema import ProxyConfig

_env_re = re.compile(r"\$\{([^}]+)\}")


def _expand_envs(value: str) -> str:
    """Replace ${VAR} with environment variable values."""
    return _env_re.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


def _expand_env_in_dict(d: dict) -> dict:
    """Recursively expand ${VAR} in string values of a dict."""
    result = {}
    for k, v in d.items():
        if isinstance(v, str):
            result[k] = _expand_envs(v)
        elif isinstance(v, dict):
            result[k] = _expand_env_in_dict(v)
        elif isinstance(v, list):
            result[k] = [_expand_envs(item) if isinstance(item, str) else item for item in v]
        else:
            result[k] = v
    return result


def load_config(path: str | Path | None = None) -> ProxyConfig:
    """Load, expand env vars, and validate config. Returns a ProxyConfig."""
    if path is None:
        path = "proxy-config.yaml"
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    expanded = _expand_env_in_dict(raw)
    return ProxyConfig(**expanded)
