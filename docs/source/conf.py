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
import sys

sys.path.insert(0, os.path.abspath('../../tests/'))

# -- Project information -----------------------------------------------------

project = "STREAM"
copyright = ""
author = "Aviv Barnea, Eshed Magali, et al."
version = "1.1.0"
# The full version, including alpha/beta/rc tags
release = "1.1.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

autoclass_content = "both"
autodoc_typehints = "description"
autodoc_member_order = "groupwise"
autodoc_inherit_docstrings = False

mathjax_path = "mathjax/es5/tex-chtml-full.js"

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.inheritance_diagram",
    "nbsphinx",
    "sphinx.ext.coverage",
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
# themes = ['alabaster',
#           'agogo',
#           'basic',
#           'bizstyle',
#           'classic',
#           'default',
#           'epub',
#           'haiku',
#           'nature',
#           'nonav',
#           'pyramid',
#           'scrolls',
#           'sphinxdoc',
#           'traditional']
html_theme = "sphinx_rtd_theme"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_sidebars = {"**": ["localtoc.html", "sourcelink.html", "searchbox.html"]}
html_logo = "_static/STREAM6.png"
html_favicon = "_static/volcano.png"
html_theme_options = dict(
    navigation_with_keys=True,
    # canonical_url=,
    # analytics_id=,
    collapse_navigation=True,
    sticky_navigation=True,
    navigation_depth=4,
    includehidden=True,
    # titles_only=,
    logo_only=True,
    display_version=True,
    prev_next_buttons_location="bottom",
    style_external_links=False,
    # style_nav_header_background=,
)
modindex_common_prefix = ["stream.", "stream.calculations."]
add_function_parentheses = False
add_module_names = False
trim_footnote_reference_space = True
todo_include_todos = True
todo_link_only = True
html_show_copyright = False
html_show_sphinx = False
html_last_updated_fmt = ""

# -- Options for LaTeX output -------------------------------------------------

latex_logo = "_static/STREAM6.png"
