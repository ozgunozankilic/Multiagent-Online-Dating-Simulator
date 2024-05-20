# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import importlib.metadata


sys.path.insert(0, os.path.abspath(".."))

project = "catfish-sim"
copyright = "2024, catfish-sim developers"
author = "Oz Kilic"
release = importlib.metadata.version("catfish_sim")


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "myst_parser",
    # "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

# nbsphinx_allow_errors = True
# nbsphinx_execute = "always"
# myst_enable_extensions = ["colon_fence"]
myst_all_links_external = True
# source_suffix = [
#     ".rst",
#     ".md",
# ]

pygments_style = "sphinx"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "catfish_sim": (
        "https://catfish-sim.readthedocs.io/en/latest/",
        None,
    ),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
