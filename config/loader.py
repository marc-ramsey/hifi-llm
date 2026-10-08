"""Load and parse config with ${VAR} expansion.

Config resolution order (first found wins):
  1. --config CLI argument
  2. HIFI_CONFIG environment variable
  3. $HOME/.config/hifi/config.yaml
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from .schema import ProxyConfig

_DEFAULT_DIR = Path.home() / ".config" / "hifi"
_DEFAULT_FILE = _DEFAULT_DIR / "config.yaml"
_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def resolve_config_path(cli_path: str | None = None) -> Path:
    """Resolve the config file path using the priority order:
      1. --config CLI arg
      2. HIFI_CONFIG env var
      3. $HOME/.config/hifi/config.yaml
    """
    if cli_path:
        return Path(cli_path).resolve()
    env_path = os.environ.get("HIFI_CONFIG")
    if env_path:
        return Path(env_path).resolve()
    return _DEFAULT_FILE


def _expand_envs(value: str) -> str:
    """Replace ${VAR} with environment variable values."""
    return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


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


def load_config(cli_path: str | None = None) -> ProxyConfig:
    """Load, expand env vars, and validate config. Returns a ProxyConfig."""
    path = resolve_config_path(cli_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Override with --config <path> or HIFI_CONFIG=<path>"
        )
    raw = yaml.safe_load(path.read_text())
    expanded = _expand_env_in_dict(raw)
    return ProxyConfig(**expanded)
