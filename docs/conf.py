# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to the path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Tracker Component Library"
copyright = "2024-2026, U.S. Naval Research Laboratory (Python port)"
author = "U.S. Naval Research Laboratory"
release = "0.22.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

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
    """Copy landing page as index.html and rename Sphinx index to docs.html."""
    import shutil
    from pathlib import Path

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

    # Copy landing page as new index.html
    landing_src = build_dir / "_static" / "landing.html"
    if landing_src.exists():
        shutil.copy(str(landing_src), str(sphinx_index))


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
napoleon_use_ivar = False
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
