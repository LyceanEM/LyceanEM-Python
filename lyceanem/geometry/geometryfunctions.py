import open3d as o3d
import numpy as np

def tri_areas(solid):
    """
    Takes as an import an open3D triangle mesh, then calculates the area of all triangles in the global units, and
    returns as an array of areas
    """
    triangle_vertices = np.asarray(solid.vertices)
    triangleidx = np.asarray(solid.triangles)
    a_vectors=triangle_vertices[triangleidx[:,1],:]-triangle_vertices[triangleidx[:,0],:]
    b_vectors = triangle_vertices[triangleidx[:, 2], :] - triangle_vertices[triangleidx[:, 0], :]
    u = np.cross(a_vectors, b_vectors)
    areas=0.5*((u[:,0]**2+u[:,1]**2+u[:,2]**2)**0.5)

    return areas

def tri_centroids(solid):
    """
    In order to calculate the centroid of the triangle, take vertices from open3d triangle mesh, then put each triangle
    into origin space, creating oa,ob,oc vectors, then the centroid is a third of the sum of oa+ob+oc, converted back
    into global coordinates
    """
    triangle_vertices = np.asarray(solid.vertices)
    triangleidx = np.asarray(solid.triangles)
    oa = triangle_vertices[triangleidx[:,0],:]
    ob = triangle_vertices[triangleidx[:,1],:]
    oc = triangle_vertices[triangleidx[:,2],:]
    centroids=((1/3)*(oa+ob+oc))
    centroid_cloud=o3d.geometry.PointCloud()
    centroid_cloud.points=o3d.utility.Vector3dVector(centroids)
    centroid_cloud.normals = solid.triangle_normals
    return centroids, centroid_cloud

def decimate_mesh(solid,mesh_sep):
    """
    In order to calculate the scattering appropriately the triangle mesh should be decimated so that the vertices
    are spaced mesh_sep apart.
    inputs are the trianglemesh object solid, and the mesh_sep, and the output is a new solid. This is only required for
    the discrete scattering model, using the centroids or vertices
    """
    new_solid=o3d.geometry.TriangleMesh()

    return new_solid
