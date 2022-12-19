# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sys
import os
import re
from glob import glob
import shutil
from sphinx_gallery.scrapers import figure_rst
try:
    from importlib import metadata
except ImportError: # for Python<3.8
    import importlib_metadata as metadata
#__version__ = metadata.version("jsonschema")
#from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = metadata.version("lyceanem")
except metadata.PackageNotFoundError:
    # package is not installed
    pass
# for example take major/minor
version = '.'.join(__version__.split('.')[:2])

class PNGScraper(object):
    def __init__(self):
        self.seen = set()

    def __repr__(self):
        return 'PNGScraper'

    def __call__(self, block, block_vars, gallery_conf):
        # Find all PNG files in the directory of this example.
        path_current_example = os.path.dirname(block_vars['src_file'])
        pngs = sorted(glob(os.path.join(path_current_example, '*.png')))

        # Iterate through PNGs, copy them to the sphinx-gallery output directory
        image_names = list()
        image_path_iterator = block_vars['image_path_iterator']
        for png in pngs:
            if png not in self.seen:
                self.seen |= set(png)
                this_image_path = image_path_iterator.next()
                image_names.append(this_image_path)
                shutil.move(png, this_image_path)
        # Use the `figure_rst` helper function to generate rST for image files
        return figure_rst(image_names, gallery_conf['src_dir'])

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))
# -- Project information -----------------------------------------------------

project = 'LyceanEM'
copyright = '2022, Timothy Pelham'
author = 'Timothy Pelham'

# The full version, including alpha/beta/rc tags
#release = version#'0.01'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinxcontrib.bibtex',
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode',
              'sphinx_gallery.gen_gallery',
              'sphinx.ext.imgmath']
bibtex_bibfiles = ['_static/lyceanemrefs.bib']
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

#autodoc options
#autodoc_mock_imports = ['numba',
#                        'cupy',
#                        'open3d',
#                        'solidpython']

intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'open3d': ('http://www.open3d.org/docs/0.9.0/', None),
    'numba': ('https://numba.readthedocs.io/en/stable/', None),
    'solidpython': ('https://solidpython.readthedocs.io/en/latest/', None)
}

#Sphinx Gallery Configuration
sphinx_gallery_conf = {
    'examples_dirs':['../examples'],
    'gallery_dirs': ['auto_examples'],
    'filename_pattern': re.escape(os.sep),
    'image_scrapers': ('matplotlib', PNGScraper()),
    'matplotlib_animations': True,
    'run_stale_examples': False,
    'first_notebook_cell': ("# This cell is added by sphinx-gallery\n"
                            "# It can be customized to whatever you like\n"
                            "%matplotlib inline"),
    'last_notebook_cell': "# This is the last cell",
    'notebook_images': f'https://stonesoup.rtfd.io/en/{os.environ.get("READTHEDOCS_VERSION", "latest")}/',
    'reference_url': {
         # The module you locally document uses None
        'sphinx_gallery': None,
    },
    'plot_gallery': False, #documentation examples require cuda on build machine, so much be fully built before being passed to readthedocs
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'classic'
html_theme = 'sphinx_rtd_theme'
html_logo = "_static/lynx.png"
html_theme_options = {
    'logo_only' : True,
    'display_version' : False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
source_suffix='.rst'
master_doc='index'

# -- Options for LaTeX output ---------------------------------------------
latex_engine = 'xelatex'
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'a4paper',

    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',

    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'LyceanEM.tex', 'LyceanEM Documentation',
     'Timothy Pelham', 'manual'),
]
