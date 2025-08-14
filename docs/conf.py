"""Configuration file for the Sphinx documentarion builder"""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Dependencies Mock just in en Read the Docs
# if os.environ.get("READTHEDOCS") == "True":
autodoc_mock_imports = [
    "astropy",
    "photutils",
    "matplotlib",
    "mc3",
]

project = "profe"
copyright = "2025, S. Paez, Y. Gómez Maqueo Chew, L. H. Hebb"
author = "S. Paez, Y. Gómez Maqueo Chew, L. H. Hebb"
root_doc = "index"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
napoleon_google_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
