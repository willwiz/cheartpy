# ruff: noqa: INP001
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CheartPy"
copyright = "2026, Will Zhang"  # noqa: A001
author = "Will Zhang"
release = "1.3.0a"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx_autodoc_typehints"]
autodoc_typehints = "both"
autodoc_type_aliases = {
    "Path": "pathlib.Path",
}
templates_path = ["_templates"]
exclude_patterns = ["**/_*"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

sys.path.insert(0, str(Path("../../cheartpy-abaqus/src").resolve()))
sys.path.insert(0, str(Path("../../cheartpy-cl/src").resolve()))
sys.path.insert(0, str(Path("../../cheartpy-fe/src").resolve()))
sys.path.insert(0, str(Path("../../cheartpy-gmsh/src").resolve()))
sys.path.insert(0, str(Path("../../cheartpy-io/src").resolve()))
sys.path.insert(0, str(Path("../../cheartpy-mesh/src").resolve()))
sys.path.insert(0, str(Path("../../cheartpy-paraview/src").resolve()))
sys.path.insert(0, str(Path("../../cheartpy-vtk/src").resolve()))
