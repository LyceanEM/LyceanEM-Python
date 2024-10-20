# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 08:57:48 2024

@author: lycea
"""
import numpy as np
import pyvista as pv
from scipy.constants import speed_of_light

def distances_angles(transmit_aperture,receive_point):
    
    distances=np.linalg.norm(receive_point-transmit_aperture.points,axis=1).reshape(-1,1)
    relative_vector=(receive_point-transmit_aperture.points)/distances.reshape(-1,1)
    angle=np.arccos(np.clip(np.sum(transmit_aperture.point_data['Normals']*relative_vector,axis=1), -1.0, 1.0)).reshape(-1,1)
    return distances,angle

    
def compute_tau(transmit_area,receive_area,distance,frequency):
    
    tau=np.sqrt(transmit_area*receive_area)/((speed_of_light/frequency)*distance)
    
    return tau

def theoretical_efficiency(transmit_area,receive_area,distance,frequency):
    
    tau=compute_tau(transmit_area,receive_area,distance,frequency)
    efficiency=1-np.exp(-tau**2)
    return efficiency

def simple_e_field(weights,distances,angles,beta,wavelength):
    
    components=weights*(np.exp(-1j*(beta*distances))/distances)
    return components

def hutson_e_field(transmit_aperture,receive_point,weights,distances,angles,beta,wavelength):
    #distance=transmit_aperture.center[2]-receive_point[2]
    #rayz=(1-(transmit_aperture.points[:,0]**2+transmit_aperture.points[:,1]**2)/(transmit_aperture.points[:,0]**2+transmit_aperture.points[:,1]**2+distance**2))**0.5
    #front=-(1/(1j*wavelength*distances))
    #G=np.exp(-1j*beta*distances)
    #theta=np.arctan2((receive_point[0]**2+receive_point[1]**2)**0.5,distance)
    #ftheta=((np.cos(theta)+rayz)/2.0)
    #components=front*weights*G*ftheta
    
    # R is disance of point of receive_point from origin (transmit center)
    # r is the distance between te receive point and the transmit points
    # ra is the distance in x and y of the point of interest from all source points, divided by twice the z seperation between the apertures
    # theta is the atan of the distance of the point of interest from the center of the receive array divided by the speration between the two arrays
    # ftheta is cos of theta plus rayz, all divided by 2
    # rayz is the sqrt of (1 - (x**2 +y**2)/(x**2 +y**2+z**2)) where x and y are the transmit aperture coordiantes, and z is the seperation between the apertures
    R=np.linalg.norm(receive_point-transmit_aperture.center)
    array_seperation=transmit_aperture.center[2]-receive_point[2]
    r=distances
    ra=distances/(2*(array_seperation))
    theta=np.arctan2(np.linalg.norm(receive_point),array_seperation).reshape(-1,1)
    rayz=np.sqrt(1 - (transmit_aperture.points[:,0]**2 +transmit_aperture.points[:,1]**2)/(transmit_aperture.points[:,0]**2 +transmit_aperture.points[:,1]**2+transmit_aperture.points[:,2]**2)).reshape(-1,1)
    ftheta=(np.cos(theta)+rayz)/2.0
    
    front=-(1/(1j*wavelength*r))
    components=front*ftheta*weights*np.exp(-1j*beta*r)
    return components

def hutson_alt_e_field(transmit_aperture,receive_point,weights,distances,angles,beta,wavelength):
    # distance=transmit_aperture.center[2]-receive_point[2]
    # ra=(transmit_aperture.points[:,0]-receive_point[0])/(2*transmit_aperture.points[:,2])+(transmit_aperture.points[:,1]-receive_point[1])/(2*transmit_aperture.points[:,2])
    # front=(1j/(2*wavelength))
    # G=(np.exp(-1j*beta*transmit_aperture.points[:,2]))/distances
    # G2=(np.exp(-1j*beta*ra))
    # ftheta=((np.cos(angles)+1.0)/2.0)
    # components=front*weights*G*G2*ftheta
    # R is disance of point of receive_point from origin (transmit center)
    # r is the distance between te receive point and the transmit points
    # ra is the distance in x and y of the point of interest from all source points, divided by twice the z seperation between the apertures
    # theta is the atan of the distance of the point of interest from the center of the receive array divided by the speration between the two arrays
    # ftheta is cos of theta plus rayz, all divided by 2
    # rayz is the sqrt of (1 - (x**2 +y**2)/(x**2 +y**2+z**2)) where x and y are the transmit aperture coordiantes, and z is the seperation between the apertures
    R=np.linalg.norm(receive_point-transmit_aperture.center)
    array_seperation=transmit_aperture.center[2]-receive_point[2]
    r=distances
    ra=distances/(2*(array_seperation))
    theta=np.arctan2(np.linalg.norm(receive_point),array_seperation)
    rayz=np.sqrt(1 - (transmit_aperture.points[:,0]**2 +transmit_aperture.points[:,1]**2)/(transmit_aperture.points[:,0]**2 +transmit_aperture.points[:,1]**2+transmit_aperture.points[:,2]**2))
    ftheta=(np.cos(theta)+rayz)/2.0
    
    front=(1j/(2*wavelength))
    G1=(np.exp(-1j*beta*transmit_aperture.points[:,2]))/R
    G2=(np.exp(-1j*beta*ra))
    components=front*ftheta*weights*G1*G2*(np.cos(theta)+1)
    return components
    
def rs_e_field(weights,distances,angles,beta):
    alpha=0.0
    
    
    front=-1/(2*np.pi)
    s = 2.5
    distance_loss = 1.0 / ((1 + distances ** s) ** (1 / s))
    G = (np.exp(-(alpha + 1j * beta) * distances)) * distance_loss

    dG = (-(alpha + 1j * beta) - (distance_loss)) * G
    #dG=np.cos(angles)*(-(alpha+1j*beta)-(1/distances))*G
    #dG = (-(alpha + 1j * beta) - (1 / distances)) * G
    components=front*weights*dG
    
    return components

def rs_e_field_new(weights,distances,angles,beta,wavelength):
    alpha=0.0
    
    
    front=-1/(2*np.pi)
    s = 2.5
    distance_loss = 1.0 / ((1 + distances ** s) ** (1 / s))
    G = (np.exp(-(alpha + 1j * beta) * distances)) * distance_loss

    dG = (-(alpha + 1j * beta) - (distance_loss)) * G
    #dG=np.cos(angles)*(-(alpha+1j*beta)-(1/distances))*G
    #dG = (-(alpha + 1j * beta) - (1 / distances)) * G
    components=front*weights*dG*np.cos(angles)
    
    return components

def friis(weights,distances,transmit_area,receive_area,wavelength):
    #Pr=Pt*(Ar*At/d**2*wavelength**2)
    components=weights*(wavelength/(4*np.pi*distances))
    return components

def id_cells(faces):
    cell_types={1:"vertex",
                2:"line",
                3:"triangle",
                4:"quad"
        }
    cells={"vertex":[],
           "line":[],
           "triangle":[],
           "quad":[]
           }
    
    while (faces.shape[0]>=1):
        trim_num=faces[0]
        temp_array=faces[1:trim_num+1]
        cells[cell_types[trim_num]].append(temp_array.tolist())
        faces=np.delete(faces,np.arange(0,trim_num+1))
        
    
    meshio_cells=[]
    for key in cells:
        if len(cells[key])>0:
            meshio_cells.append((key,cells[key]))
        
    return meshio_cells

def pyvista_to_meshio(polydata_object):
    import meshio
    #polydata_object=polydata_object.extract_cells_by_type(5) #extract only the triangles
    if type(polydata_object)==pv.core.pointset.UnstructuredGrid:
        cells=id_cells(polydata_object.cells)    
    else:
        cells=id_cells(polydata_object.faces)
    meshio_object = meshio.Mesh(
        points=polydata_object.points,
        cells=cells,
        point_data=polydata_object.point_data,
    )
    return meshio_object

def lycean(weights1,weights2,weights3,transmit_aperture,receive_aperture,wavelength):
    import copy
    import lyceanem.models.frequency_domain as FD
    excitation_function=np.zeros((weights1.shape[0],3),dtype="complex")
    excitation_function[:,0]=np.ones((weights1.shape[0]))
    transmit=pyvista_to_meshio(transmit_aperture)
    receive=pyvista_to_meshio(receive_aperture)
    Ex,Ey,Ez=FD.calculate_scattering(transmit, 
                                     receive, 
                                     [],
                                     excitation_function,
                                     wavelength=wavelength,
                                     scattering=0,
                                     beta=(np.pi*2)/wavelength,
                                     elements=True
                                     )
    
    sum_axis=0
    total1=np.array([np.sum(copy.deepcopy(Ex)*weights1,axis=sum_axis),np.sum(copy.deepcopy(Ey)*weights1,axis=sum_axis),np.sum(copy.deepcopy(Ez)*weights1,axis=sum_axis)]).transpose()
    components1=np.linalg.norm(total1,axis=1).reshape(-1,1)
    total2=np.array([np.sum(copy.deepcopy(Ex)*weights2,axis=sum_axis),np.sum(copy.deepcopy(Ey)*weights2,axis=sum_axis),np.sum(copy.deepcopy(Ez)*weights2,axis=sum_axis)]).transpose()
    components2=np.linalg.norm(total2,axis=1).reshape(-1,1)
    total3=np.array([np.sum(copy.deepcopy(Ex)*weights3,axis=sum_axis),np.sum(copy.deepcopy(Ey)*weights3,axis=sum_axis),np.sum(copy.deepcopy(Ez)*weights3,axis=sum_axis)]).transpose()
    components3=np.linalg.norm(total3,axis=1).reshape(-1,1)
    return components1,components2,components3

def contributions(transmit_aperture,receive_aperture,weights,frequency):
    wavelength=(speed_of_light/frequency)
    beta=(np.pi*2)/wavelength
    components=0.0
    wavelength=(speed_of_light/frequency)
    beta=(np.pi*2)/wavelength
    components=0.0
    equiphase_weights=equiphase(transmit_aperture, frequency)
    window_length=(transmit_area**2+transmit_area**2)**0.5
    kaiser_weights=kaiser_amplitude_taper(transmit_aperture,2.3614,window_length).reshape(-1,1)
    kaiser_weights_equiphase=kaiser_weights * equiphase_weights.reshape(-1,1)
    flat_weights=discrete_transmit_power(np.ones((transmit_aperture.n_points,1)), transmit_aperture.point_data['Area'],transmit_power=transmit_power)
    equi_weights=discrete_transmit_power(equiphase_weights, transmit_aperture.point_data['Area'],transmit_power=transmit_power)
    kai_weights=discrete_transmit_power(kaiser_weights_equiphase, transmit_aperture.point_data['Area'],transmit_power=transmit_power)
    from tqdm import tqdm
    #Lycean
    results1,results2,results3=lycean(flat_weights*transmit_aperture.point_data['Area'].reshape(-1,1),
                                      equi_weights*transmit_aperture.point_data['Area'].reshape(-1,1),
                                      kai_weights*transmit_aperture.point_data['Area'].reshape(-1,1),
                                      transmit_aperture,
                                      receive_aperture,
                                      wavelength)
    receive_aperture.point_data['Lycean_Amplitude-Flat']=np.abs(results1)
    receive_aperture.point_data['Lycean_Amplitude-Equiphase']=np.abs(results2)
    receive_aperture.point_data['Lycean_Amplitude-Kaiser']=np.abs(results3)
    for receive_index in range(receive_aperture.n_points):
        distances,angles=distances_angles(transmit_aperture,receive_aperture.points[receive_index,:])
        receive_aperture.point_data['Amplitude-Simple'][receive_index]=np.abs(np.sum(simple_e_field(weights, distances, angles, beta, wavelength)))
        receive_aperture.point_data['Amplitude-Hutson'][receive_index]=np.abs(np.sum(hutson_e_field(transmit_aperture,receive_aperture.points[receive_index,:],weights, distances, angles, beta, wavelength)))
        receive_aperture.point_data['Amplitude-RS+'][receive_index]=np.abs(np.sum(rs_e_field(weights, distances, angles, beta)))
        receive_aperture.point_data['Amplitude-RS'][receive_index]=np.abs(np.sum(rs_e_field_new(weights, distances, angles, beta,wavelength)))
        
    return receive_aperture

def planar_surface(u_size,v_size,mesh_size):
    """
    Minimal Example, generating a structured mesh, then converting to triangles and points with normal vectors, orientated with boresight along the +z direction.

    Parameters
    ----------
    u_size : float
        size in the u direction (x until rotated)
    v_size : TYPE
        size in the y direction (y until rotated)
    mesh_size : float
        target mesh size for the surface, default for aperture should be half a wavelength

    Returns
    -------
    surface : TYPE
        

    """
    u_points=np.linspace(-u_size/2,u_size/2,np.ceil(u_size/mesh_size).astype(int))
    v_points=np.linspace(-v_size/2,v_size/2,np.ceil(v_size/mesh_size).astype(int))
    n_points=np.linspace(0,0.0,1)
    u_mesh,v_mesh,n_mesh=np.meshgrid(u_points,v_points,n_points,indexing='ij')
    points=pv.PolyData(np.c_[u_mesh.reshape(-1),v_mesh.reshape(-1),n_mesh.reshape(-1)])
    surface=points.delaunay_2d()
    surface.point_data['Normals'] =np.zeros((surface.number_of_points,3))
    surface.point_data['Normals'][:,2]=1.0
    surface.point_data['Area']=mesh_size**2
    surface.cell_data['Normals'] =np.zeros((surface.number_of_cells,3))
    surface.cell_data['Normals'][:,2]=1.0
    return surface

def discrete_transmit_power(
    weights, element_area, transmit_power=100.0, impedance=np.pi * 120.0
):
    """
    Calculate the transmitting aperture amplitude density required for a given transmit power in watts.
    Parameters
    ----------
    weights
    element_area
    transmit_power
    impedance

    Returns
    -------

    """
    # Calculate the Power at each mesh point from the Intensity at each point
    power_at_point = np.abs(weights.reshape(-1,1))* element_area.reshape(
        -1, 1
    )  # calculate power
    # integrate power over aperture and normalise to desired power for whole aperture
    power_normalisation = np.abs(transmit_power / np.abs(np.sum(power_at_point)))
    transmit_power_density = (
        power_at_point * power_normalisation
    ) / element_area.reshape(-1, 1)
    # calculate amplitude (V/m)
    transmit_amplitude = ((transmit_power_density * impedance) ** 0.5) * np.exp(1j*np.angle(weights.reshape(-1, 1)))
    # transmit_excitation=transmit_amplitude_density.reshape(-1,1)*element_area.reshape(-1,1)
    return transmit_amplitude

def equiphase(transmit_aperture,frequency):
    wavelength=(speed_of_light/frequency)
    beta=(np.pi*2)/wavelength
    vector_offset=transmit_aperture.points
    
    weights=np.exp(1j*(np.linalg.norm(vector_offset,axis=1))*beta)
    return weights

def prep_scenario(transmit_area,receive_area,distance,frequency,mesh_size=0.4941690299231883):
    wavelength=(speed_of_light/frequency)
    transmit_aperture=planar_surface(transmit_area**0.5,transmit_area**0.5,wavelength*mesh_size)
    transmit_aperture.rotate_x(180,inplace=True,transform_all_input_vectors=True)
    transmit_aperture.translate([0.0,0.0,distance],inplace=True)
    receive_aperture=planar_surface(receive_area**0.5,receive_area**0.5,wavelength*mesh_size)
    
    return transmit_aperture,receive_aperture
def kaiser_amplitude_taper(mesh,alpha,window_length):
    radial_distances=np.linalg.norm(mesh.points-mesh.center,axis=1)+1e-9
    kaiser_weights=np.i0(np.pi*alpha*(1-((2*radial_distances)/window_length)**2)**0.5)/np.i0(np.pi*alpha)
    kaiser_weights[np.isnan(kaiser_weights)]=1e-12
    
    return kaiser_weights
if __name__ == '__main__':
    
    
    num_sample_points=50
    transmit_power=100.0
    frequency=10e9
    wavelength=speed_of_light/frequency
    transmit_area=4.0
    receive_area=4.0
    transmit_normal=np.zeros((1,3))
    transmit_normal[0,2]=1.0
    receive_normal=np.zeros((num_sample_points,3))
    receive_normal[0,2]=-1.0
    distance=np.logspace(0,3,num=num_sample_points)
    points=np.zeros((distance.shape[0],3))
    points[:,2]=distance
    sample_line=pv.PolyData(points)
    #sample_line.point_data['Normals']=receive_normal
    #sample_line.point_data['Area']=receive_area
    sample_line.point_data['Amplitude-Simple']=0.0
    sample_line.point_data['Amplitude-Hutson']=0.0
    sample_line.point_data['Amplitude-RS+']=0.0
    sample_line.point_data['Amplitude-RS']=0.0
    sample_line.point_data['Amplitude-Friis']=0.0
    sample_line.point_data['Lycean_Amplitude-Flat']=0.0
    sample_line.point_data['Lycean_Amplitude-Equiphase']=0.0
    sample_line.point_data['Lycean_Amplitude-Kaiser']=0.0
    sample_line.point_data['Theoretical Max']=theoretical_efficiency(transmit_area, receive_area, distance, frequency)*transmit_power
    #transmit_point=pv.PolyData(np.zeros((1,3)))
    #transmit_point.point_data['Normals']=transmit_normal
    #transmit_point.point_data['Area']=transmit_area
    from tqdm import tqdm
    for inc in tqdm(range(num_sample_points)):
        transmit_aperture,receive_aperture=prep_scenario(transmit_area,receive_area,distance[inc],frequency)
        
        receive_aperture.point_data['Amplitude-Simple']=0.0
        receive_aperture.point_data['Amplitude-Hutson']=0.0
        receive_aperture.point_data['Amplitude-RS+']=0.0
        receive_aperture.point_data['Amplitude-RS']=0.0
        receive_aperture.point_data['Amplitude-Friis']=0.0
        receive_aperture.point_data['Lycean_Amplitude-Flat']=0.0
        receive_aperture.point_data['Lycean_Amplitude-Equiphase']=0.0
        receive_aperture.point_data['Lycean_Amplitude-Kaiser']=0.0
        equiphase_weights=equiphase(transmit_aperture, frequency)
        window_length=(transmit_area**2+transmit_area**2)**0.5
        kaiser_weights=kaiser_amplitude_taper(transmit_aperture,2.3614,window_length).reshape(-1,1)
        kaiser_weights_equiphase=kaiser_weights * equiphase_weights.reshape(-1,1)
        weights=discrete_transmit_power(kaiser_weights_equiphase, transmit_aperture.point_data['Area'],transmit_power=transmit_power)
        print("Power Density ",np.mean((np.abs(weights)**2)/(np.pi*120)))
        receive_aperture=contributions(transmit_aperture,receive_aperture,weights*transmit_aperture.point_data['Area'].reshape(-1,1),frequency)
        receive_aperture.save("Receive_Aperture{}.vtk".format(inc))
        sample_line.point_data['Amplitude-Simple'][inc]=np.mean(np.abs(((receive_aperture.point_data['Amplitude-Simple']**2)/(np.pi*120))))*receive_area
        sample_line.point_data['Amplitude-Hutson'][inc]=np.mean(np.abs(((receive_aperture.point_data['Amplitude-Hutson']**2)/(np.pi*120))))*receive_area
        sample_line.point_data['Amplitude-RS+'][inc]=np.mean(np.abs(((receive_aperture.point_data['Amplitude-RS+']**2)/(np.pi*120))))*receive_area
        sample_line.point_data['Amplitude-RS'][inc]=np.mean(np.abs(((receive_aperture.point_data['Amplitude-RS']**2)/(np.pi*120))))*receive_area
        sample_line.point_data['Amplitude-Friis'][inc]=np.mean(np.abs(((receive_aperture.point_data['Amplitude-Friis']**2)/(np.pi*120))))*receive_area
        sample_line.point_data['Lycean_Amplitude-Flat'][inc]=np.mean(np.abs(((receive_aperture.point_data['Lycean_Amplitude-Flat']**2)/(np.pi*120))))*receive_area
        sample_line.point_data['Lycean_Amplitude-Equiphase'][inc]=np.mean(np.abs(((receive_aperture.point_data['Lycean_Amplitude-Equiphase']**2)/(np.pi*120))))*receive_area
        sample_line.point_data['Lycean_Amplitude-Kaiser'][inc]=np.mean(np.abs(((receive_aperture.point_data['Lycean_Amplitude-Kaiser']**2)/(np.pi*120))))*receive_area
        del receive_aperture,transmit_aperture,equiphase_weights,kaiser_weights,weights
    
    color_list=['b','r','g','c','m','y','k']
    chart=pv.Chart2D()#,off_screen=True)
    #plot=chart.line(sample_line.points[:,2],sample_line.point_data['Amplitude-Simple'].reshape(-1,1),color=color_list[0],label="Simple",width=2.0)
    #plot=chart.line(sample_line.points[:,2],sample_line.point_data['Amplitude-RS+'].reshape(-1,1),color=color_list[1],label="RS+",width=2.0)
    #plot=chart.line(sample_line.points[:,2],sample_line.point_data['Amplitude-RS'].reshape(-1,1),color=color_list[2],label="RS",width=2.0)
    #plot=chart.line(sample_line.points[:,2],sample_line.point_data['Amplitude-Hutson'].reshape(-1,1),color=color_list[3],label="Hutson",width=2.0)
    #plot=chart.line(sample_line.points[:,2],sample_line.point_data['Amplitude-Friis'].reshape(-1,1),color=color_list[4],label="Friis",width=2.0)
    plot=chart.line(sample_line.points[:,2],sample_line.point_data['Lycean_Amplitude-Flat'].reshape(-1,1),color=color_list[0],label="Flat Weights",width=3.0)#,style="--")
    plot=chart.line(sample_line.points[:,2],sample_line.point_data['Lycean_Amplitude-Equiphase'].reshape(-1,1),color=color_list[1],label="Equiphase",width=3.0)#,style=":")
    plot=chart.line(sample_line.points[:,2],sample_line.point_data['Lycean_Amplitude-Kaiser'].reshape(-1,1),color=color_list[2],label="Kaiser",width=3.0)#,style="--")
    plot=chart.line(sample_line.points[:,2],sample_line.point_data['Theoretical Max'].reshape(-1,1),color=color_list[-1],label="Theoretical Max",width=3.0)
    chart.x_range=[5,1000]
    chart.y_range=[0,100]
    chart.x_axis.log_scale=True
    chart.y_axis.log_scale=True
    chart.x_label="Seperation (m)"
    chart.y_label="Received Power (W)"
    chart.show(window_size=[720,720//2],screenshot="BeamformingComparison.png")
    