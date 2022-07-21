"""LyceanEM : Electromagnetics Modelling for Antenna and Antenna Array Development on Complex Platforms"""
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

__copyright__ = """\
© Timothy Pelham 2016-2022
"""
__license__ = "GNU Affero General Public License v3"
