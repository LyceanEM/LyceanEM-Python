import numpy as np
from tqdm import tqdm
from ..raycasting import rayfunctions as RF
from ..electromagnetics import empropagation as EM
from ..geometry import targets as TL
from ..geometry import geometryfunctions as GF
import open3d as o3d
from ..base import scattering_t