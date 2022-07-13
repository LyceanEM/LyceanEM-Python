"""LyceanEM : Electromagnetics Modelling for Antenna and Antenna Array Development on Complex Platforms"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lyceanem")
except PackageNotFoundError:
    # package is not installed
    pass

__copyright__ = """\
© Timothy Pelham 2016-2022
"""
__license__ = "GNU Affero General Public License v3"
