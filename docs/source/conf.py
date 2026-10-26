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

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
]
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_type_aliases = {
    "Result": "pytools.result.Result",
    # Add any other ambiguous generic types here if needed
}
# autodoc_typehints_ignore_names = ["Path", "pathlib.Path"]
autodoc_default_options = {
    # "members": True,
    # "undoc-members": True,
    # "show-inheritance": True,
    # "imported-members": True,
}
templates_path = ["_templates"]
exclude_patterns = ["**/_*"]
always_document_param_types = True
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

print("Adding ...", str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
for f in Path(__file__).resolve().parent.parent.parent.glob("cheartpy-*/src"):
    if not f.is_dir():
        continue
    print("Adding ...", str(f.resolve()))
    sys.path.insert(0, str(f.resolve()))


# ruff: ignore[ARG001, ANN401, FBT001, PLR0913, PLR0917]
def custom_autodoc_skip_member(
    app: Sphinx,
    what: Literal["module", "class", "exception", "function", "method", "attribute", "data"],
    name: str,
    obj: Any,
    skip: bool,
    options: Any,
) -> bool:
    """Filter out hyphenated dictionary keys to prevent signature parsing errors."""
    if "-" in name:
        return True  # Instruct autodoc to skip generating an autoattribute signature
    return skip  # Fall back to default behavior for all other items


def setup(app: Sphinx) -> None:
    app.connect("autodoc-skip-member", custom_autodoc_skip_member)
