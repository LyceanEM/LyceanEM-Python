import numpy as np
import lyceanem.tests.reflectordata as data
import open3d as o3d
import copy
from lyceanem.models.frequency_domain import aperture_projection
from lyceanem.base import structures
#import example UAV and nose mounted array from examples
body,array=data.exampleUAV()
#visualise UAV and Array
o3d.visualization.draw_geometries([body,array])
#crop the inner surface of the array trianglemesh (not strictly required, as the UAV main body provides blocking to the hidden surfaces, but correctly an aperture will only have an outer face.
surface_array=copy.deepcopy(array)
surface_array.triangles=o3d.utility.Vector3iVector(
        np.asarray(array.triangles)[:len(array.triangles) // 2, :])
surface_array.triangle_normals=o3d.utility.Vector3dVector(
        np.asarray(array.triangle_normals)[:len(array.triangle_normals) // 2, :])

#set wavelength of interest, say 10GHz
wavelength=3e8/10e9
blockers=structures([body])

directivity_envelop,pcd=aperture_projection(surface_array,
                                            environment=blockers,
                                            wavelength=wavelength,
                                            az_range=np.linspace(-180.0, 180.0, 19),
                                            elev_range=np.linspace(-90.0, 90.0, 19))