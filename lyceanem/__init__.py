"""LyceanEM : Electromagnetics Modelling for Antenna and Antenna Array Development on Complex Platforms"""

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("lyceanem").version
except DistributionNotFound:
    #test for package installaiton
    pass

__copyright__ = '''\
© Timothy Pelham 2016-2022
'''
__license__ = 'GNU Affero General Public License v3'
from . import _version
__version__ = _version.get_versions()['version']
