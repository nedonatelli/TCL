"""Pytest configuration for notebook validation.

This file configures pytest-nbval for validating Jupyter notebooks.
"""

import pytest


def pytest_configure(config):
    """Register custom markers for notebook tests."""
    config.addinivalue_line(
        "markers", "notebook: mark test as a notebook validation test"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_matplotlib_backend():
    """Configure matplotlib for non-interactive backend in CI."""
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture(scope="session")
def sample_data_dir():
    """Return path to sample data directory."""
    from pathlib import Path

    return Path(__file__).parent.parent.parent / "examples" / "data"
