# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to the path for autodoc
sys.path.insert(0, os.path.abspath(".."))
# Add docs dir to path for project_metadata
sys.path.insert(0, os.path.abspath("."))

from project_metadata import (
    count_public_functions,
    count_python_modules,
    count_test_functions,
    read_pyproject_metadata,
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Tracker Component Library"
copyright = (
    "2024-2026, nrl-tracker contributors; original MATLAB library by the "
    "U.S. Naval Research Laboratory (public domain)"
)
author = "nrl-tracker contributors"
release = read_pyproject_metadata().get("version", "0.0.0")

# Dynamic project stats for RST substitutions
_n_functions = f"{count_public_functions():,}"
_n_modules = f"{count_python_modules():,}"
_n_tests = f"{count_test_functions():,}"

# RST substitutions available in all .rst files
rst_epilog = f"""
.. |version_badge| replace:: **v{release}**
.. |n_functions| replace:: {_n_functions}
.. |n_modules| replace:: {_n_modules}
.. |n_tests| replace:: {_n_tests}
"""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.mermaid",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "nbsphinx",
]

# nbsphinx settings
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Continue build even if notebook has errors
nbsphinx_kernel_name = "python3"

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Custom CSS to override RTD theme with pyTCL dark theme
html_css_files = [
    "rtd-overrides.css",
]


# -- Custom landing page setup -----------------------------------------------
def setup(app):
    """Register the build-finished event to set up landing page."""
    app.connect("build-finished", copy_landing_page)


def copy_landing_page(app, exception):
    """Copy landing page as index.html and rename Sphinx index to docs.html.

    Also replaces placeholders with actual project metadata:
    - @VERSION@ → release version
    - @FUNCTIONS@ → function count
    - @TESTS@ → test case count
    - @MODULES@ → module count
    - @MATLAB_PARITY@ → MATLAB feature parity percentage
    - @PACKAGE_NAME@ → PyPI package name
    - @GITHUB_URL@ → GitHub repository URL
    """
    import shutil
    import sys
    from pathlib import Path

    # Add docs dir to path to import project_metadata
    docs_dir = Path(__file__).parent
    if str(docs_dir) not in sys.path:
        sys.path.insert(0, str(docs_dir))

    from project_metadata import get_project_metadata

    if exception is not None:
        return

    # Only process HTML builds
    if app.builder.name != "html":
        return

    build_dir = Path(app.outdir)

    # Rename Sphinx's index.html to docs.html
    sphinx_index = build_dir / "index.html"
    docs_page = build_dir / "docs.html"

    if sphinx_index.exists():
        shutil.copy(str(sphinx_index), str(docs_page))

    # Copy landing page as new index.html and inject metadata
    landing_src = build_dir / "_static" / "landing.html"
    if landing_src.exists():
        # Read landing page content
        with open(landing_src, "r", encoding="utf-8") as f:
            content = f.read()

        # Get all project metadata
        metadata = get_project_metadata()

        # Replace all placeholders with actual metadata
        replacements = {
            "@VERSION@": metadata["version"],
            "@FUNCTIONS@": metadata["functions"],
            "@TESTS@": metadata["tests"],
            "@MODULES@": metadata["modules"],
            "@COVERAGE@": metadata["coverage"],
            "@PACKAGE_NAME@": metadata["package_name"],
            "@GITHUB_URL@": metadata["github_url"],
        }

        for placeholder, value in replacements.items():
            content = content.replace(placeholder, str(value))

        # Write to destination
        with open(sphinx_index, "w", encoding="utf-8") as f:
            f.write(content)


# -- Extension configuration -------------------------------------------------

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
# True renders a NumPy "Attributes" section as :ivar: fields inside the class
# description. With False, Napoleon emits a separate py:attribute directive for
# each entry, which then collides with the attribute autodoc already documented
# from the class itself.
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# MathJax settings for rendering equations
mathjax3_config = {
    "tex": {
        "macros": {
            "RR": r"\mathbb{R}",
            "bold": [r"\mathbf{#1}", 1],
        }
    }
}
