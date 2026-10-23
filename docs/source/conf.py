# ruff: noqa: INP001
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from sphinx.application import Sphinx

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
autodoc_typehints_ignore_names = ["Path", "pathlib.Path"]
templates_path = ["_templates"]
exclude_patterns = ["**/_*"]
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

for f in Path(__file__).resolve().parent.parent.parent.glob("cheartpy-*/src"):
    if not f.is_dir():
        continue
    sys.path.insert(0, str(f.resolve()))


# ruff: ignore[ARG001, ANN401, FBT001, PLR0913, PLR0917]
def skip_hyphenated_typeddict_keys(
    app: Sphinx,
    what: Literal["module", "class", "exception", "function", "method", "attribute", "data"],
    name: str,
    obj: Any,
    skip: bool,
    options: Any,
) -> bool | None:
    """Filter out hyphenated dictionary keys to prevent signature parsing errors."""
    if "-" in name:
        return True  # Instruct autodoc to skip generating an autoattribute signature
    return None  # Fall back to Sphinx default behavior for other items


def setup(app: Sphinx) -> None:
    """Connect custom event handlers to the Sphinx application instance."""
    app.connect("autodoc-skip-member", skip_hyphenated_typeddict_keys)
