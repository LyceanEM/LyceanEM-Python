
import numpy as np

##translation matrix from eath centre to bristol

import meshio
import numpy as np
import scipy.stats as stats
mesh = meshio.read("/home/tf17270/Downloads/ST47sw250k.xdmf")
n = np.array(mesh.points)
radius_eath = 6371000
lat_bristol = np.radians(51.4545)
lon_bristol = np.radians(-2.5879)
## affine transformation matrix
translation_matrix = np.zeros((4,4))
translation_matrix[0,0] = 1
translation_matrix[1,1] = 1
translation_matrix[2,2] = 1
translation_matrix[3,3] = 1
translation_matrix[0,3] = radius_eath

## rotation matrix
rotation_matrix_lat = np.zeros((4,4))
rotation_matrix_lat[1,1] = 1
rotation_matrix_lat[3,3] = 1
rotation_matrix_lat[0,0] = np.cos(lat_bristol)
rotation_matrix_lat[0,2] = -np.sin(lat_bristol)
rotation_matrix_lat[2,0] = np.sin(lat_bristol)
rotation_matrix_lat[2,2] = np.cos(lat_bristol)

## rotation matrix

rotation_matrix_lon = np.zeros((4,4))
rotation_matrix_lon[2,2] = 1
rotation_matrix_lon[3,3] = 1
rotation_matrix_lon[0,0] = np.cos(lon_bristol)
rotation_matrix_lon[0,1] = -np.sin(lon_bristol)
rotation_matrix_lon[1,0] = np.sin(lon_bristol)
rotation_matrix_lon[1,1] = np.cos(lon_bristol)
#multiply up the matrices
translation_matrix = np.matmul(np.matmul(rotation_matrix_lon,rotation_matrix_lat), translation_matrix)
inverse_matrix = np.linalg.inv(translation_matrix)
for i in range(n.shape[0]):
    n[i] = np.matmul(inverse_matrix, np.append(n[i],1))[0:3]
print(np.min(n[:,0]), np.min(n[:,1]), np.min(n[:,2]))
print(np.max(n[:,0]), np.max(n[:,1]), np.max(n[:,2]))
#bin
s = stats.binned_statistic_2d( n[:,1], n[:,2],None, 'count', bins=100)
arr = s.statistic
print(np.mean(arr))
print(np.max(arr))
print(np.min(arr))





