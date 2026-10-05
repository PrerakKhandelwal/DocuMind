"""
config_loader.py
Loads and validates config/config.yaml. Provides a global `cfg` object.

Relative paths under `paths` are resolved against the project root (folder that
contains `config/`, `src/`, `scripts/`), not the shell's current working directory.
"""
import os
import yaml
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


def project_root() -> Path:
    """Absolute path to the AgenticModel application root."""
    return _PROJECT_ROOT


def _resolve_project_path(value: str) -> str:
    p = Path(value).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((_PROJECT_ROOT / p).resolve())


def _normalize_paths(data: dict) -> None:
    paths = data.get("paths")
    if not isinstance(paths, dict):
        return
    for key, val in list(paths.items()):
        if isinstance(val, str):
            paths[key] = _resolve_project_path(val)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


class Config:
    """Dot-access wrapper around a nested yaml dict."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            val = self._data[name]
            if isinstance(val, dict):
                return Config(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __repr__(self):
        return f"Config({self._data})"


def load_config(path: str | Path = _CONFIG_PATH) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Allow environment variable overrides: AGENT_LOGGING__LEVEL=DEBUG
    # (double underscore = nesting separator)
    for env_key, env_val in os.environ.items():
        if env_key.startswith("AGENT_"):
            keys = env_key[len("AGENT_"):].lower().split("__")
            node = data
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = env_val
    _normalize_paths(data)
    return Config(data)


# Global singleton — import this everywhere
cfg: Config = load_config()
