# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import re
import sys
from pathlib import Path


# -- pyvista configuration ---------------------------------------------------
import pyvista
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.docs import linkcode_resolve  # noqa: F401
from pyvista.core.utilities.docs import pv_html_page_context
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

# Manage errors
pyvista.set_error_output_file("errors.txt")
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
# pyvista.set_plot_theme('DocumentTheme')
pyvista.set_jupyter_backend(None)
# Save figures in specified directory
pyvista.FIGURE_PATH = str(Path("./_static/").resolve() / "auto-generated/")


try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata
# __version__ = metadata.version("jsonschema")
# from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = metadata.version("lyceanem")
except metadata.PackageNotFoundError:
    # package is not installed
    pass
# for example take major/minor
version = ".".join(__version__.split(".")[:2])


# sys.path.insert(0, os.path.abspath("."))
# sys.path.insert(0, os.path.abspath("../../"))
# -- Project information -----------------------------------------------------

project = "LyceanEM"
copyright = "2025, Timothy Pelham"
author = "Timothy Pelham"

# The full version, including alpha/beta/rc tags
# release = version#'0.01'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.imgmath",
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
    "sphinx_design",
]
bibtex_bibfiles = ["_static/lyceanemrefs.bib"]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# autodoc options
# autodoc_mock_imports = ['numba',
#                        'cupy',
#                        'open3d',
#                        'solidpython']

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
}

# Sphinx Gallery Configuration
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": re.escape(os.sep),
    # Modules for which function level galleries are created.  In
    "doc_module": "pyvista",
    "image_scrapers": ("pyvista", "matplotlib"),
    "first_notebook_cell": "%matplotlib inline",
    "reset_modules_order": "both",
    "matplotlib_animations": True,
    "run_stale_examples": True,
    "junit": str(Path("sphinx-gallery") / "junit-results.xml"),
    "reference_url": {
        # The module you locally document uses None
        "sphinx_gallery": None,
        "pyvista": None,
    },
    "plot_gallery": False,  # documentation examples require cuda on build machine, so must be fully built before being passed to readthedocs
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'classic'
html_theme = "sphinx_rtd_theme"
html_logo = "_static/LY_logo_RGB_2000px.jpg"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
source_suffix = ".rst"
master_doc = "index"

# -- Options for LaTeX output ---------------------------------------------
latex_engine = "xelatex"
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "a4paper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "LyceanEM.tex", "LyceanEM Documentation", "Timothy Pelham", "manual"),
]
