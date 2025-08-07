# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../../src"))


def get_version_from_pyproject():
    """Read version from pyproject.toml file using regex parsing."""
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    try:
        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Use regex to find version line
        version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if version_match:
            return version_match.group(1)
        else:
            return "unknown"
    except (FileNotFoundError, Exception):
        return "unknown"


# -- Project information -----------------------------------------------------

project = "TabStruct"
copyright = "2025-present, Xiangjian Jiang"
author = "Xiangjian Jiang"

# The full version, including alpha/beta/rc tags
release = get_version_from_pyproject()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_logo = "_media/repo_logo.png"
html_theme = "sphinx_book_theme"
html_title = "TabStruct Documentation"

htmlhelp_basename = "mainDoc"

html_theme_options = {
    "repository_url": "https://github.com/SilenceX12138/TabStruct.git",
    "path_to_docs": "",
    "repository_branch": "master",
    "use_repository_button": True,
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com/",
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "_media"]

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension -------------------------------------------

# Automatically extract typehints
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Include __init__ docstrings
autoclass_content = "both"

# Sort members by source order
autodoc_member_order = "bysource"

# -- Options for autosummary extension ---------------------------------------

autosummary_generate = True

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
}

# -- Options for napoleon extension ------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- Options for todo extension ----------------------------------------------

todo_include_todos = True

# Source file suffixes
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}

# Master document
master_doc = "index"
