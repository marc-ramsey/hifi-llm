"""Plugin manager — scans a directory and loads modules that export register(app)."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_plugins(plugins_dir: Path | None, app: Any) -> None:
    """
    Scan plugins_dir for Python files. Import each one and call
    register(app) if the function exists.
    """
    if plugins_dir is None or not plugins_dir.is_dir():
        return

    loaded = 0
    for mod_file in sorted(plugins_dir.glob("*.py")):
        # Skip __init__ and hidden files
        if mod_file.stem.startswith(("_", ".")):
            continue

        try:
            spec = importlib.util.spec_from_file_location(mod_file.stem, mod_file)
            if spec is None or spec.loader is None:
                logger.warning("Could not load plugin spec: %s", mod_file)
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Call register(app) if it exists
            if hasattr(module, "register"):
                module.register(app)
                loaded += 1
                logger.info("Loaded plugin: %s (%s)", mod_file.stem, mod_file)
        except Exception:
            logger.exception("Failed to load plugin: %s", mod_file)

    if loaded:
        logger.info("Loaded %d plugin(s) from %s", loaded, plugins_dir)
