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


os.environ['PYVISTA_DOCUMENTATION_BULKY_IMPORTS_ALLOWED'] = 'true'
# -- pyvista configuration ---------------------------------------------------
import pyvista
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.docs import linkcode_resolve  # noqa: F401
from pyvista.core.utilities.docs import pv_html_page_context
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

# Manage errors
pyvista.set_error_output_file('errors.txt')
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
#pyvista.set_plot_theme('DocumentTheme')
pyvista.set_jupyter_backend(None)
# Save figures in specified directory
pyvista.FIGURE_PATH = str(Path('./_static/').resolve() / 'auto-generated/')

# Configuration of pyvista.ext.coverage
coverage_additional_modules = [
    'pyvista',
    'pyvista.errors',
    'pyvista.report',
    # core
    'pyvista.core.cell',
    'pyvista.core.celltype',
    'pyvista.core.composite',
    'pyvista.core.dataobject',
    'pyvista.core.dataset',
    'pyvista.core.datasetattributes',
    'pyvista.core.errors',
    'pyvista.core.grid',
    'pyvista.core.objects',
    'pyvista.core.pointset',
    'pyvista.core.pyvista_ndarray',
    'pyvista.core.filters.composite',
    'pyvista.core.filters.data_set',
    'pyvista.core.filters.poly_data',
    'pyvista.core.filters.structured_grid',
    'pyvista.core.filters.image_data',
    'pyvista.core.filters.unstructured_grid',
    'pyvista.core.utilitis.arrays',
    'pyvista.core.utilitis.cell_type_helper',
    'pyvista.core.utilitis.cells',
    'pyvista.core.utilitis.features',
    'pyvista.core.utilitis.fileio',
    'pyvista.core.utilitis.geometric_objects',
    'pyvista.core.utilitis.helpers',
    'pyvista.core.utilitis.misc',
    'pyvista.core.utilitis.observers',
    'pyvista.core.utilitis.parametric_objects',
    'pyvista.core.utilitis.points',
    'pyvista.core.utilitis.reader',
    'pyvista.core.utilitis.transformations',
    # demos
    'pyvista.demos.demos',
    'pyvista.demos.logo',
    # examples
    'pyvista.examples.cells',
    'pyvista.examples.downloads',
    'pyvista.examples.examples',
    'pyvista.examples.gltf',
    'pyvista.examples.planets',
    'pyvista.examples.vrml',
    # plotting
    'pyvista.plotting.actor_properties',
    'pyvista.plotting.actor',
    'pyvista.plotting.axes_actor',
    'pyvista.plotting.axes',
    'pyvista.plotting.background_renderer',
    'pyvista.plotting.camera',
    'pyvista.plotting.charts',
    'pyvista.plotting.colors',
    'pyvista.plotting.composite_mapper',
    'pyvista.plotting.cube_axes_actor',
    'pyvista.plotting.errors',
    'pyvista.plotting.helpers',
    'pyvista.plotting.lights',
    'pyvista.plotting.lookup_table',
    'pyvista.plotting.mapper',
    'pyvista.plotting.opts',
    'pyvista.plotting.picking',
    'pyvista.plotting.plotter',
    'pyvista.plotting.prop3d',
    'pyvista.plotting.render_passes',
    'pyvista.plotting.render_window_interactor',
    'pyvista.plotting.renderer',
    'pyvista.plotting.renderers',
    'pyvista.plotting.scalar_bars',
    'pyvista.plotting.texture',
    'pyvista.plotting.themes',
    'pyvista.plotting.tools',
    'pyvista.plotting.volume_property',
    'pyvista.plotting.volume',
    'pyvista.plotting.widgets',
]
# -- Sphinx Gallery Options
from sphinx_gallery.sorting import FileNameSortKey


class ResetPyVista:
    """Reset pyvista module to default settings."""

    def __call__(self, gallery_conf, fname):
        """Reset pyvista module to default settings.

        If default documentation settings are modified in any example, reset here.
        """
        import pyvista

        pyvista._wrappers['vtkPolyData'] = pyvista.PolyData
        #pyvista.set_plot_theme('document_build')

    def __repr__(self):
        return 'ResetPyVista'


reset_pyvista = ResetPyVista()
if not Path(pyvista.FIGURE_PATH).exists():
    Path(pyvista.FIGURE_PATH).mkdir()

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
os.environ['PYVISTA_BUILDING_GALLERY'] = 'true'
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

# class PNGScraper(object):
#    def __init__(self):
#        self.seen = set()

#    def __repr__(self):
#        return 'PNGScraper'

#    def __call__(self, block, block_vars, gallery_conf):
# Find all PNG files in the directory of this example.
#        path_current_example = os.path.dirname(block_vars['src_file'])
#        pngs = sorted(glob(os.path.join(path_current_example, '*.png')))

# Iterate through PNGs, copy them to the sphinx-gallery output directory
#        image_names = list()
#        image_path_iterator = block_vars['image_path_iterator']
#        for png in pngs:
#            if png not in self.seen:
#                self.seen |= set(png)
#                this_image_path = image_path_iterator.next()
#                image_names.append(this_image_path)
#                shutil.move(png, this_image_path)
# Use the `figure_rst` helper function to generate rST for image files
#        return figure_rst(image_names, gallery_conf['src_dir'])

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../"))
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
    "myst_parser",
    "sphinx_c_autodoc",
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
    'doc_module': 'pyvista',
    'reference_url': {'pyvista': None},  # Add hyperlinks inside code blocks to pyvista methods
    'image_scrapers': (DynamicScraper(), 'matplotlib'),
    'first_notebook_cell': '%matplotlib inline',
    'reset_modules': (reset_pyvista,),
    'reset_modules_order': 'both',
    "matplotlib_animations": True,
    "run_stale_examples": True,
    'junit': str(Path('sphinx-gallery') / 'junit-results.xml'),
    "reference_url": {
        # The module you locally document uses None
        "sphinx_gallery": None,
    },
    "plot_gallery": True,  # documentation examples require cuda on build machine, so must be fully built before being passed to readthedocs
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
