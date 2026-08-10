"""Shared data directory utilities for pytcl.

All modules that load external data files (terrain, magnetism, gravity)
use this common utility for locating the data directory.
"""

import os
from pathlib import Path

from pytcl.diagnostics import logger


def get_data_dir() -> Path:
    """Get the pytcl data directory for external data files.

    The data directory is located at ``~/.pytcl/data/`` by default.
    Can be overridden by setting the ``PYTCL_DATA_DIR`` environment variable.

    Returns
    -------
    Path
        Path to the data directory.
    """
    env_dir = os.environ.get("PYTCL_DATA_DIR")
    if env_dir:
        path = Path(env_dir)
        override = "override active"
    else:
        path = Path.home() / ".pytcl" / "data"
        override = "not set"
    logger.bind(site="data-files").debug(
        "data dir resolved to {} (PYTCL_DATA_DIR {})", path, override
    )
    return path


def ensure_data_dir() -> Path:
    """Ensure the data directory exists and return its path."""
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
