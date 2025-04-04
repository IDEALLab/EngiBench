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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
import os
import sys
from pathlib import Path

import engibench

sys.path.append(str(Path('_ext').resolve()))

project = "EngiBench"
author = "ETH Zurich's IDEAL Lab"

# The full version, including alpha/beta/rc tags
release = engibench.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",  # Google style docstrings
    "sphinx.ext.doctest",  # Test code snippets in the documentation
    "sphinx.ext.autodoc",  # Include documentation from docstrings
    "sphinx.ext.githubpages",  # Publish the documentation on GitHub pages
    "sphinx.ext.viewcode",  # Add links to the source code
    "myst_parser",  # Markdown support
    "sphinx_github_changelog",  # Generate changelog
    "sphinx_multiversion",  # Versioning
    "problem_doc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = []

# Napoleon settings
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"
html_title = "EngiBench Documentation"
html_baseurl = ""
html_logo = "_static/img/logo_2.png"
html_copy_source = False
html_favicon = "_static/img/logo_2.png"

# Theme options
html_theme_options = {
    "source_repository": "https://github.com/IDEALLab/EngiBench",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Add version information to the context
html_context = {
    "version_info": {
        "version": release,
        "versions": {
            "latest": "/",
            "stable": "/v" + release,
        },
        "current": "latest",
    }
}

# Add version switcher to the left sidebar
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
        "versions.html",
    ]
}

html_static_path = ["_static"]
html_css_files: list[str] = []

# -- Generate Changelog -------------------------------------------------

sphinx_github_changelog_token = os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN")

# -- Versioning configuration ------------------------------------------------

# Configure sphinx-multiversion
smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'  # Only include version tags
smv_branch_whitelist = r'^main$'  # Only include main branch
smv_remote_whitelist = r'^.*$'
smv_latest_version = 'main'  # Use main branch as latest version
