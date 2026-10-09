"""Plugin manager — scans a directory and loads modules that export register(app)."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


def load_plugins(plugins_dir: Path | None, app: Any) -> None:
    """
    Scan plugins_dir for Python files. Import each one and call
    register(app) if the function exists.

    Each plugin is loaded in isolation:
    - A fresh module namespace is created per plugin to avoid name collisions.
    - The plugin's module is temporarily added to sys.modules so imports work,
      but it is removed after loading (unless register() explicitly keeps state).
    - Errors in one plugin do not prevent others from loading.
    - If a plugin's register() raises, the error is logged and the proxy
      continues running with the remaining plugins.
    """
    if plugins_dir is None or not plugins_dir.is_dir():
        return

    loaded = 0
    errors = 0

    for mod_file in sorted(plugins_dir.glob("*.py")):
        # Skip __init__ and hidden files
        if mod_file.stem.startswith(("_", ".")):
            continue

        module_name = f"hifi_plugin_{mod_file.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, mod_file)
            if spec is None or spec.loader is None:
                logger.warning("Could not load plugin spec: %s", mod_file)
                errors += 1
                continue

            module: ModuleType = importlib.util.module_from_spec(spec)
            # Register in sys.modules so relative imports within the plugin work
            sys.modules[module_name] = module

            # Load the module (runs top-level code, but not register())
            spec.loader.exec_module(module)

            # Call register(app) if it exists
            if hasattr(module, "register"):
                module.register(app)
                loaded += 1
                logger.info("Loaded plugin: %s (%s)", mod_file.stem, mod_file)
            else:
                logger.debug("Plugin %s has no register() function — skipped", mod_file.stem)

        except Exception:
            errors += 1
            logger.exception("Failed to load plugin: %s", mod_file)
        finally:
            # Clean up sys.modules to avoid namespace pollution across reloads
            sys.modules.pop(module_name, None)

    if loaded:
        logger.info("Loaded %d plugin(s) from %s", loaded, plugins_dir)
    if errors:
        logger.warning("%d plugin(s) failed to load from %s", errors, plugins_dir)
