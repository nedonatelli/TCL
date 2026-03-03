"""
Project metadata and statistics for dynamic landing page generation.

This module provides project information that gets injected into the landing page
during the Sphinx build process. Statistics can be auto-calculated from the codebase
or manually maintained for release information.
"""

import re
from pathlib import Path
from typing import Dict, Any


# Manually maintained statistics that cannot be auto-calculated
RELEASE_STATS = {
    "matlab_parity": "100",
    "coverage": "80",
}

# Project URLs and identifiers
PROJECT_INFO = {
    "github_url": "https://github.com/nedonatelli/TCL",
    "package_name": "nrl-tracker",
    "organization": "U.S. Naval Research Laboratory",
}


def read_pyproject_metadata() -> Dict[str, Any]:
    """Read metadata from pyproject.toml.
    
    Returns
    -------
    dict
        Dictionary containing 'version', 'description', 'name', 'urls'
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    
    metadata = {}
    try:
        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract version
        version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if version_match:
            metadata["version"] = version_match.group(1)
        
        # Extract name
        name_match = re.search(r'name\s*=\s*"([^"]+)"', content)
        if name_match:
            metadata["name"] = name_match.group(1)
        
        # Extract description
        desc_match = re.search(r'description\s*=\s*"([^"]+)"', content)
        if desc_match:
            metadata["description"] = desc_match.group(1)
    except Exception as e:
        print(f"Warning: Could not read pyproject.toml: {e}")
    
    return metadata


def count_public_functions() -> int:
    """Count public functions and methods in pytcl package.

    Returns
    -------
    int
        Number of public (non-underscore-prefixed) function and method defs.
    """
    pytcl_path = Path(__file__).parent.parent / "pytcl"
    if not pytcl_path.exists():
        return 0

    count = 0
    for py_file in pytcl_path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            # Top-level public functions
            count += len(re.findall(r"^def [a-zA-Z]\w*", content, re.MULTILINE))
            # Public methods (single indent)
            count += len(re.findall(r"    def [a-zA-Z]\w*", content, re.MULTILINE))
        except Exception:
            pass
    return count


def count_python_modules() -> int:
    """Count Python modules in pytcl package.
    
    Returns
    -------
    int
        Number of .py files in pytcl directory (excluding __pycache__)
    """
    pytcl_path = Path(__file__).parent.parent / "pytcl"
    if not pytcl_path.exists():
        return 0
    
    # Count .py files recursively
    count = len(list(pytcl_path.rglob("*.py")))
    return count


def count_test_functions() -> int:
    """Count test functions in tests directory.
    
    Returns
    -------
    int
        Number of test functions (def test_*)
    """
    tests_path = Path(__file__).parent.parent / "tests"
    if not tests_path.exists():
        return 0
    
    count = 0
    for test_file in tests_path.rglob("test_*.py"):
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
            # Count function definitions starting with 'def test_'
            matches = re.findall(r"def test_\w+", content)
            count += len(matches)
        except Exception:
            pass
    
    return count


def get_project_metadata() -> Dict[str, str]:
    """Get all project metadata for landing page injection.
    
    Returns
    -------
    dict
        Dictionary with all metadata needed for landing page:
        - version: Release version
        - package_name: PyPI package name
        - github_url: GitHub repository URL
        - functions: Function count
        - modules: Module count
        - tests: Test case count
        - matlab_parity: MATLAB feature parity percentage
        - coverage: Code coverage percentage
    """
    pyproject = read_pyproject_metadata()
    
    metadata = {
        # From pyproject.toml
        "version": pyproject.get("version", "0.0.0"),
        "package_name": pyproject.get("name", PROJECT_INFO["package_name"]),
        "description": pyproject.get(
            "description",
            "Python port of the U.S. Naval Research Laboratory's Tracker Component Library"
        ),
        
        # From PROJECT_INFO
        "github_url": PROJECT_INFO["github_url"],
        "organization": PROJECT_INFO["organization"],
        
        # Auto-calculated statistics
        "functions": f"{count_public_functions():,}",
        "modules": f"{count_python_modules():,}",
        "tests": f"{count_test_functions():,}",

        # Manually maintained statistics
        "matlab_parity": RELEASE_STATS["matlab_parity"],
        "coverage": RELEASE_STATS["coverage"],
    }
    
    return metadata


if __name__ == "__main__":
    # For debugging/testing
    metadata = get_project_metadata()
    print("Project Metadata:")
    print("=" * 60)
    for key, value in metadata.items():
        print(f"  {key:20} {value}")
