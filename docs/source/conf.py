import os
from datetime import datetime
import sys
from pathlib import Path


DOCS_DIR = Path(__file__).parent
sys.path.insert(0, os.path.abspath("../../anndata_fcs"))

# Project information
project = "anndata_fcs"
author = "harryhaller001"
copyright = f"{datetime.now():%Y}, {author}."

# Configuration
templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
default_role = "literal"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "autoapi.extension",
]

# Mapping for intersphinx extension
intersphinx_mapping = dict(
    python=("https://docs.python.org/3", None),
    numpy=("https://numpy.org/doc/stable/", None),
    matplotlib=("https://matplotlib.org/stable", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/latest/", None),
)

# don't run the notebooks
nbsphinx_execute = "never"

pygments_style = "sphinx"


exclude_trees = ["_build", "dist"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# Generate the API documentation when building
autoapi_type = "python"
autoapi_add_toctree_entry = False
autoapi_ignore: list[str] = ["_*.py"]
autoapi_dirs = ["../../anndata_fcs"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_member_order = "alphabetical"

autosummary_generate = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Configuration of sphinx.ext.coverage
coverage_show_missing_items = True


# Configurate sphinx rtd theme
html_theme = "furo"
html_static_path = ["_static"]
html_show_sphinx = False
html_context = dict(
    display_github=True,
    github_user="harryhaller001",
    github_repo="anndata_fcs",
    github_version="main",
    conf_py_path="/docs/",
    github_button=True,
    show_powered_by=False,
)
html_title = "anndata_fcs"
html_css_files = [
    "css/custom.css",
]
# html_favicon = "./_static/favicon.ico"
