import numpy as np
import pyvista as pv
import tempfile
import os
import pytest
import meshio
import timeit
from acceleration_structure import Tile_acceleration_structure
import scipy

#from em_MVP import em_propagation_03  # Replace 'your_module' with the actual module name


def WavefrontWeights(source_coords,wavelength, steering_vector = np.array([ 5.15129018e-01,  3.28602175e-08, -8.57112650e-01])):
    """

    calculate the weights for a given set of element coordinates, wavelength, and steering vector (cartesian)

    """
    weights = np.zeros((source_coords.shape[0]), dtype=np.complex64)
    # calculate distances of coords from steering_vector by using it to calculate arbitarily distant point
    # dist=distance.cdist(source_coords,(steering_vector*1e9))
    # _,_,_,dist=calc_dv(source_coords,(steering_vector*1e9))
    dist = np.zeros((source_coords.shape[0]), dtype=np.float32)
    dist = np.sqrt(
        np.abs(
            (source_coords[:, 0] - steering_vector.ravel()[0] * 1e9) ** 2
            + (source_coords[:, 1] - steering_vector.ravel()[1] * 1e9) ** 2
            + (source_coords[:, 2] - steering_vector.ravel()[2] * 1e9) ** 2
        )
    )
    # RF.fast_calc_dv(source_coords,target,dv,dist)
    dist = dist - np.min(dist)
    # calculate required time delays, and then convert to phase delays
    delays = dist / scipy.constants.speed_of_light
    weights[:] = np.exp(
        1j * 2 * np.pi * (scipy.constants.speed_of_light / wavelength) * delays
    )
    return weights
    
def ArbitaryCoherenceWeights(source_coords, target_coord, wavelength):
    """

    Generate Wavefront coherence weights based upon the desired wavelength and the coordinates of the target point

    """
    from scipy.spatial import distance
    from scipy.constants import speed_of_light
    weights = np.zeros((len(source_coords), 1), dtype=np.complex64)
    # calculate distances of coords from steering_vector by using it to calculate arbitarily distant point
    dist = distance.cdist(source_coords, target_coord)
    dist = dist - np.min(dist)
    # calculate required time delays, and then convert to phase delays
    delays = dist / speed_of_light
    weights[:] = np.exp(
        1j * 2 * np.pi * (speed_of_light / wavelength) * delays
    )
    return weights

def discrete_transmit_power(weights,element_area,transmit_power=100.0,impedance=np.pi*120.0):
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
    #to start with assume area per element is consistent
    power_at_point=np.linalg.norm(weights,axis=1).reshape(-1,1)*element_area.reshape(-1,1) # calculate power
    #integrate power over aperture and normalise to desired power for whole aperture
    power_normalisation=transmit_power/np.sum(power_at_point)
    transmit_power_density=(power_at_point*power_normalisation)/element_area.reshape(-1,1)
    #calculate amplitude density
    transmit_amplitude_density=(transmit_power_density*impedance)**0.5
    transmit_excitation=transmit_amplitude_density.reshape(-1,1)*element_area.reshape(-1,1)*(weights/np.linalg.norm(weights,axis=1).reshape(-1,1))
    return transmit_excitation
def main():
    """
    End to end test for example 3 in the documentation of old code. Test passes if the result of the new method matches
    the result of the old one for the range of angles specified in the old code.
    This comparison is made to a reference saved .npy file found in the reference_files folder.
    The test will fail if any step is wrong, including
    - the generation of the input data
    - the EM propagation
    - the conversion of the result to a numpy array of the same shape as the reference result
    """

    # Run EM propagation
    # load points
    ##end = meshio.read("/home/tf17270/Downloads/Receiving_aperture (1).vtk")
    source = meshio.read("../Downloads//Transmitting_Aperture_x_vertical.vtk")
    source.point_data['is_electric'] = np.ones(source.points.shape[0])
    source.point_data['permittivity'] = np.ones(source.points.shape[0],dtype=np.complex64)
    source.point_data['permeability'] = np.ones(source.points.shape[0],dtype=np.complex64)

    end = meshio.read("../Downloads//Receiving_aperture_x_vertical.vtk")
    freq = np.asarray(2.45e9)
    from scipy.constants import speed_of_light
    wavelength = speed_of_light / freq   


    # stack point data


    weights = ArbitaryCoherenceWeights(source.points, np.array([[18.0,0,0]]),wavelength)[:,0]
    wavefrontweights = WavefrontWeights(source.points,wavelength)
    ## combine ex real imag, ey real imag, ez real imag into a single array
    ex = source.point_data['Ex%20-%20Real'] + 1j*source.point_data['Ex%20-%20Imag']
    ey = source.point_data['Ey%20-%20Real'] + 1j*source.point_data['Ey%20-%20Imag'][:,0]
    ez = source.point_data['Ez%20-%20Real'] + 1j*source.point_data['Ez%20-%20Imag']
    w = np.stack((ex,ey,ez),axis=1)
    power = discrete_transmit_power(w,wavelength*wavelength/4,transmit_power=223408)
    source.point_data['Ex%20-%20Real'] = power[:,0].real
    source.point_data['Ex%20-%20Imag'] = power[:,0].imag
    source.point_data['Ey%20-%20Real'] = power[:,1].real
    source.point_data['Ey%20-%20Imag'] = power[:,1].imag
    source.point_data['Ez%20-%20Real'] = power[:,2].real
    source.point_data['Ez%20-%20Imag'] = power[:,2].imag


    terrain = meshio.read("../Downloads//ST47sw1M_x_vertical.xdmf")
    tile = Tile_acceleration_structure(terrain, 200)
    print("Done building acceleration structure")
    #bin_counts = bin_counts_to_numpy(np.array(n), np.array(triangles), ncellsy, ncellsz, min_y, diff, min_z)
    #binned_triangles = bin_triangles_to_numpy(n, triangles, ncellsy, ncellsz, min_y, diff, min_z, bin_counts, np.sum(bin_counts))


    final_result = np.zeros((end.points.shape[0],3),dtype=np.complex64)
    final_result_formed2 = np.zeros((end.points.shape[0],3),dtype=np.complex64)
    final_result_not_formed = np.zeros((end.points.shape[0],3),dtype=np.complex64)
    
    ## get result slice by slice of sink_points
    x = 2000
    y = 1000
    ones = np.ones(y)
    
    k = 0
    now = timeit.default_timer()

    for i in range(0,1000277,x):
        print("i", i)
        for j in range(0,250000,y):
            top_ind = i + x
            if top_ind > 1000277:
                top_ind = 1000277
            top_ind_source = j + y
            if top_ind_source > source.points.shape[0]:
                top_ind_source = source.points.shape[0]
            sink_sizess = top_ind - i
            source_sizess = top_ind_source - j



            result = tile.calculate_scattering_sliced(source, j, top_ind_source, end, i, top_ind, 51.363, 1.535e-5, wavelength)
            result = result.reshape((source_sizess,top_ind-i,3))
            result = result.transpose(1,0,2)


            final_result[i:top_ind,0] +=  np.dot(result[:,:,0], weights[j:top_ind_source])
            final_result[i:top_ind,1] += np.dot(result[:,:,1], weights[j:top_ind_source])
            final_result[i:top_ind,2] += np.dot(result[:,:,2], weights[j:top_ind_source])
            final_result_formed2[i:top_ind,0] +=  np.dot(result[:,:,0], wavefrontweights[j:top_ind_source])
            final_result_formed2[i:top_ind,1] += np.dot(result[:,:,1], wavefrontweights[j:top_ind_source])
            final_result_formed2[i:top_ind,2] += np.dot(result[:,:,2], wavefrontweights[j:top_ind_source])
            final_result_not_formed[i:top_ind,0] +=  np.dot(result[:,:,0], ones)
            final_result_not_formed[i:top_ind,1] += np.dot(result[:,:,1], ones)
            final_result_not_formed[i:top_ind,2] += np.dot(result[:,:,2], ones)



            k+=1
            if k%20 == 0:
                end = timeit.default_timer()
                print(k,"kth 20 loops took", end-now)
                print("saving", k)
                np.save("RECIEVER_beamformed_ARB.npy", final_result)
                np.save("RECIEVER_no_beamforming.npy", final_result_not_formed)
                np.save("RECIEVER_beamformed_r2.npy", final_result_formed2)
                now = timeit.default_timer()




    np.save("DONERECIEVER_beamformed_ARB.npy", final_result)
    np.save("DONERECIEVER_no_beamforming.npy", final_result_not_formed)
    np.save("DONERECIEVER_beamformed_r2.npy", final_result_formed2)
main()
   
