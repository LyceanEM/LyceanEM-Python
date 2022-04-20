

import cmath

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import open3d as o3d
import scipy.stats
from matplotlib import cm
from matplotlib.patches import Wedge
from numba import cuda, float32, njit, prange
from scipy.spatial import distance

from ..raycasting import rayfunctions as RF


def Steering_Efficiency(Dtheta,Dphi,Dtot,first_dimension_angle,second_dimension_angle,angular_coverage):
    """
    Calculate Steering Efficiency for the provided pattern, in radians

    Parameters
    ----------
    Dtheta : TYPE
        DESCRIPTION.
    Dphi : TYPE
        DESCRIPTION.
    Dtot : TYPE
        DESCRIPTION.
    angular coverage : TYPE
        DESCRIPTION.

    Returns
    -------
    setheta : TYPE
        DESCRIPTION.
    sephi : TYPE
        DESCRIPTION.
    setot : TYPE
        DESCRIPTION.

    """
    with np.errstate(divide='ignore'):
        a_index=10*np.log10(np.abs(Dtheta))>=(10*np.log10(np.nanmax(np.abs(Dtheta)))-3)
        b_index=10*np.log10(np.abs(Dphi))>=(10*np.log10(np.nanmax(np.abs(Dphi)))-3)
        tot_index=10*np.log10(np.abs(Dtot))>=(10*np.log10(np.nanmax(np.abs(Dtot)))-3)
        setheta=(np.sum(a_index)/(Dtheta.shape[0]*Dtheta.shape[1]))*100
        sephi=(np.sum(b_index)/(Dphi.shape[0]*Dphi.shape[1]))*100
        setot=(np.sum(tot_index)/(Dtot.shape[0]*Dtot.shape[1]))*100
        #setheta=((np.sum(a_index)*(first_dimension_angle*second_dimension_angle))/angular_coverage)*100
        #sephi=((np.sum(b_index)*(first_dimension_angle*second_dimension_angle))/angular_coverage)*100
        #setot=((np.sum(tot_index)*(first_dimension_angle*second_dimension_angle))/angular_coverage)*100

    return setheta,sephi,setot

@njit(cache=True, nogil=True)
def WavefrontWeights(source_coords,steering_vector,wavelength):
    """
    calculate the weights for a given set of element coordinates, wavelength, and steering vector (cartesian)
    """
    weights=np.zeros((source_coords.shape[0]),dtype=np.float64)
    #calculate distances of coords from steering_vector by using it to calculate arbitarily distant point
    #dist=distance.cdist(source_coords,(steering_vector*1e9))
    #_,_,_,dist=calc_dv(source_coords,(steering_vector*1e9))
    dist=np.zeros((source_coords.shape[0]),dtype=np.float32)
    dist=np.sqrt(np.abs((source_coords[:,0]-steering_vector.ravel()[0]*1e9)**2+(source_coords[:,1]-steering_vector.ravel()[1]*1e9)**2+(source_coords[:,2]-steering_vector.ravel()[2]*1e9)**2))
    #RF.fast_calc_dv(source_coords,target,dv,dist)
    dist=dist-np.min(dist)
    #calculate required time delays, and then convert to phase delays
    delays=dist/scipy.constants.speed_of_light
    weights=np.exp(-1j*2*np.pi*(scipy.constants.speed_of_light/wavelength)*delays)
    return weights

def ArbitaryCoherenceWeights(source_coords,target_coord,wavelength):
    """
    Generate Wavefront coherence weights based upon the desired wavelength and the coordinates of the target point
    """
    weights=np.zeros((len(source_coords)),dtype=np.float64)
    #calculate distances of coords from steering_vector by using it to calculate arbitarily distant point
    dist=distance.cdist(source_coords,target_coord)
    dist=dist-np.min(dist)
    #calculate required time delays, and then convert to phase delays
    delays=dist/scipy.constants.speed_of_light
    weights=np.exp(-1j*2*np.pi*(scipy.constants.speed_of_light/wavelength)*delays)
    return weights

def TimeDelayWeights(source_coords,target_coord,magnitude=1.0):
    """
    Generate time delay weights to focus on a target coordinate
    """
    weights=np.zeros((len(source_coords)),dtype=np.complex64)
    #calculate distances of coords from steering_vector by using it to calculate arbitarily distant point
    dist=distance.cdist(source_coords,target_coord)
    dist=dist-np.min(dist)
    #calculate required time delays, and then convert to phase delays
    delays=dist/scipy.constants.speed_of_light
    weights=magnitude+delays*1j
    return weights

@njit(cache=True, nogil=True)
def MRCWeights(Etheta,Ephi,command_angles,polarization_switch='Etheta',az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-90.0,90.0,19)):
    """
    calculate the maximal ratio combining weights for a given set of element coordinates, wavelength, and command angles (az,elev)
    """
    weights=np.zeros((Etheta.shape[0]),dtype=np.complex64)
    az_index=np.argmin(np.abs(az_range-command_angles[0]))
    elev_index=np.argmin(np.abs(elev_range-command_angles[1]))
    if polarization_switch=='Etheta':
        angle_vector=np.angle(Etheta[:,elev_index,az_index].astype(np.complex64))
    else:
        angle_vector=np.angle(Ephi[:,elev_index,az_index].astype(np.complex64))

    weights=np.exp(-1j*angle_vector)
    return weights

def OAMWeights(x,y,mode):
    """
    generate OAM mode weights, based upon the radial angle of each element.
    """
    #assumed array is x directed
    angles=np.arctan2(x,y)
    weights=np.zeros((len(x)),dtype=np.complex64)
    weights=np.exp(-1j*mode*angles)
    return weights

def OAMFourier(Ex,Ey,Ez,coordinates,prime_vector,mode_limit,first_dimension,second_dimension,coord_format='AzEl'):
    """
    producing mode index, mode coefficiencts, and mode probabilities with the co and crosspolar (Etheta,Ephi), but can probably be done the same for Ex,Ey,Ez
    """
    #establish coordinate set, in this case theta and phi, but would work just as well with elevation and azimuth, assume that array is propagating in the z+ direction.
    if coord_format=='xyz':
        mode_index,mode_coefficients,mode_probabilites=OAMFourierCartesian(Ex,Ey,Ez,coordinates,mode_limit,first_dimension,second_dimension)
    elif coord_format=='AzEl':
        mode_index,mode_coefficients,mode_probabilites=OAMFourierSpherical(Ex,Ey,Ez,coordinates,mode_limit,first_dimension,second_dimension)

    return mode_index,mode_coefficients,mode_probabilites

def OAMFourierCartesian(Ex,Ey,Ez,coordinates,mode_limit):
    """
    assume probagation is in the Ez dimension
    """
    mode_index=np.linspace(-mode_limit,mode_limit,mode_limit*2+1)
    mode_coefficients=np.zeros((mode_index.shape[0],3),dtype=np.complex64)
    az,el,r=RF.cart2sph(coordinates[:,0], coordinates[:,1], coordinates[:,2])
    #a coefficient of mode m, at angle theta is defined in terms of the
    #integral of the phi dimension in the range 0 to 2pi.
    for oam_m in range(len(mode_index)):
        mode_coefficients[oam_m,0]=(1/(2*np.pi))*np.sum(Ex.ravel()*np.exp(1j*mode_index[oam_m]*az),axis=0)
        mode_coefficients[oam_m,1]=(1/(2*np.pi))*np.sum(Ey.ravel()*np.exp(1j*mode_index[oam_m]*az),axis=0)
        mode_coefficients[oam_m,2]=(1/(2*np.pi))*np.sum(Ez.ravel()*np.exp(1j*mode_index[oam_m]*az),axis=0)

    powers=np.sum(np.abs(mode_coefficients**2),axis=0)


    mode_probabilities=np.zeros((mode_index.shape[0],3),dtype=np.float32)
    mode_probabilities[:,0]=np.abs((1/np.sum(powers))*(mode_coefficients[:,0]**2))
    mode_probabilities[:,1]=np.abs((1/np.sum(powers))*(mode_coefficients[:,1]**2))
    mode_probabilities[:,2]=np.abs((1/np.sum(powers))*(mode_coefficients[:,2]**2))

    return mode_index,mode_coefficients,mode_probabilities

def OAMFourierSpherical(Ex,Ey,Ez,coordinates,mode_limit,az_range,elev_range):
    """
    assume probagation is in the Ez dimension
    """
    mode_index=np.linspace(-mode_limit,mode_limit,mode_limit*2+1)
    mode_coefficients=np.zeros((mode_index.shape[0],len(elev_range),3))
    #a coefficient of mode m, at angle theta is defined in terms of the
    #integral of the azimuth dimension in the range -pi to pi.
    for oam_m in range(mode_index):
        #mode_coefficients(oam_m,:,1)=(1/(2*pi))*sum(CoPolar(:,1:theta_limit).*exp(-sqrt(-1)*mode_index(oam_m)*p'*ones(1,theta_limit)));
        #mode_coefficients(oam_m,:,2)=(1/(2*pi))*sum(CrossPolar(:,1:theta_limit).*exp(-sqrt(-1)*mode_index(oam_m)*p'*ones(1,theta_limit)));
        mode_coefficients[oam_m,:,0]=(1/(2*np.pi))*np.sum(Ex[:,:]*np.exp(-1j*mode_index[oam_m]),axis=0)

     #copolar_power=sum(sum(abs(mode_coefficients(:,:,1)).^2));
     #crosspolar_power=sum(sum(abs(mode_coefficients(:,:,2)).^2));

    mode_probabilities=np.zeros((mode_index.shape[0],2),dtype=np.float32)
    #mode_prob(:,1)=(1/sum([copolar_power,crosspolar_power]))*sum(abs(mode_coefficients(:,:,1)').^2);
    #mode_prob(:,2)=(1/sum([copolar_power,crosspolar_power]))*sum(abs(mode_coefficients(:,:,2)').^2);

    return mode_index,mode_coefficients,mode_probabilities

#@njit(parallel=True,cache=True, nogil=True)
def MaximumDirectivityMap(Etheta,Ephi,source_coords,wavelength,az_res,elev_res,az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-180.0,180.0,19),forming='Total',total_solid_angle=(4*np.pi),phase_resolution=24):
    directivity_map=np.zeros((elev_res,az_res,3))
    command_angles=np.zeros((2),dtype=np.float32)
    for az_inc in range(az_res):
        for elev_inc in range(elev_res):
            command_angles[0]=az_range[az_inc]
            command_angles[1]=elev_range[elev_inc]
            steering_vector=np.zeros((1,3))
            steering_vector[0,0],steering_vector[0,1],steering_vector[0,2]=RF.sph2cart(np.radians(command_angles[0]), np.radians(command_angles[1]), 1)
            WS_weights=WavefrontWeights(source_coords,steering_vector,wavelength)
            MRC_weights=MRCWeights(Etheta, Ephi, command_angles,az_range=az_range,elev_range=elev_range)
            MRC_weights2=MRCWeights(Etheta, Ephi, command_angles,az_range=az_range,elev_range=elev_range,polarization_switch='Ephi')
            if phase_resolution<=12:
                WS_weights=WeightTruncation(WS_weights,phase_resolution)
                MRC_weights=WeightTruncation(MRC_weights,phase_resolution)
                MRC_weights2=WeightTruncation(MRC_weights2,phase_resolution)

            Ethetabeamformed=np.sum(MRC_weights.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
            Ephibeamformed=np.sum(MRC_weights.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
            Ethetabeamformed2=np.sum(MRC_weights2.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
            Ephibeamformed2=np.sum(MRC_weights2.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
            Ethetabeamformed3=np.sum(WS_weights.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
            Ephibeamformed3=np.sum(WS_weights.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
            Dtheta,Dphi,Dtot,_=directivity_transform(Ethetabeamformed,Ephibeamformed,az_range=az_range,elev_range=elev_range)
            Dtheta2,Dphi2,Dtot2,_=directivity_transform(Ethetabeamformed2,Ephibeamformed2,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
            Dtheta3,Dphi3,Dtot3,_=directivity_transform(Ethetabeamformed3,Ephibeamformed3,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
            if forming=='Total':
                comparitor=np.asarray([Dtot[elev_inc,az_inc],Dtot2[elev_inc,az_inc],Dtot3[elev_inc,az_inc]])
            elif forming=='Etheta':
                comparitor=np.asarray([Dtheta[elev_inc,az_inc],Dtheta2[elev_inc,az_inc],Dtheta3[elev_inc,az_inc]])
            elif forming=='Ephi':
                comparitor=np.asarray([Dphi[elev_inc,az_inc],Dphi2[elev_inc,az_inc],Dphi3[elev_inc,az_inc]])

            if np.any(np.isnan(comparitor)):
                print('error')

            best_index=np.where(comparitor==np.max(comparitor))[0][0]
            if best_index==0:
                directivity_map[elev_inc,az_inc,0]=Dtheta[elev_inc,az_inc]
                directivity_map[elev_inc,az_inc,1]=Dphi[elev_inc,az_inc]
                directivity_map[elev_inc,az_inc,2]=Dtot[elev_inc,az_inc]
            elif best_index==1:
                directivity_map[elev_inc,az_inc,0]=Dtheta2[elev_inc,az_inc]
                directivity_map[elev_inc,az_inc,1]=Dphi2[elev_inc,az_inc]
                directivity_map[elev_inc,az_inc,2]=Dtot2[elev_inc,az_inc]
            elif best_index==2:
                directivity_map[elev_inc,az_inc,0]=Dtheta3[elev_inc,az_inc]
                directivity_map[elev_inc,az_inc,1]=Dphi3[elev_inc,az_inc]
                directivity_map[elev_inc,az_inc,2]=Dtot3[elev_inc,az_inc]

    return directivity_map

@njit(parallel=False,cache=True, nogil=True)
def MaximumDirectivityMapDiscrete(Etheta,Ephi,source_coords,wavelength,az_res,elev_res,az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-180.0,180.0,19),forming='Total',total_solid_angle=(4*np.pi),phase_resolution=np.asarray([24])):
    directivity_map=np.zeros((elev_res,az_res,3,phase_resolution.shape[0]),dtype=np.float32)
    command_angles=np.zeros((2),dtype=np.float32)
    for az_inc in range(az_res):
        for elev_inc in range(elev_res):
            inc_res=0
            for res_inc in range(phase_resolution.shape[0]):
                resolution=phase_resolution[res_inc]
                command_angles[0]=az_range[az_inc]
                command_angles[1]=elev_range[elev_inc]
                steering_vector=np.zeros((1,3))
                steering_vector[0,0],steering_vector[0,1],steering_vector[0,2]=RF.sph2cart(np.radians(command_angles[0]), np.radians(command_angles[1]), 1)
                WS_weights=WavefrontWeights(source_coords,steering_vector,wavelength)
                MRC_weights=MRCWeights(Etheta, Ephi, command_angles,az_range=az_range,elev_range=elev_range)
                MRC_weights2=MRCWeights(Etheta, Ephi, command_angles,az_range=az_range,elev_range=elev_range,polarization_switch='Ephi')

                WS_weights=WeightTruncation(WS_weights,resolution)
                MRC_weights=WeightTruncation(MRC_weights,resolution)
                MRC_weights2=WeightTruncation(MRC_weights2,resolution)

                Ethetabeamformed=np.sum(MRC_weights.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
                Ephibeamformed=np.sum(MRC_weights.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
                Ethetabeamformed2=np.sum(MRC_weights2.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
                Ephibeamformed2=np.sum(MRC_weights2.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
                Ethetabeamformed3=np.sum(WS_weights.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
                Ephibeamformed3=np.sum(WS_weights.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
                Dtheta,Dphi,Dtot,_=directivity_transform(Ethetabeamformed,Ephibeamformed,az_range=az_range,elev_range=elev_range)
                Dtheta2,Dphi2,Dtot2,_=directivity_transform(Ethetabeamformed2,Ephibeamformed2,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
                Dtheta3,Dphi3,Dtot3,_=directivity_transform(Ethetabeamformed3,Ephibeamformed3,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
                if forming=='Total':
                    comparitor=np.asarray([Dtot[elev_inc,az_inc],Dtot2[elev_inc,az_inc],Dtot3[elev_inc,az_inc]])
                elif forming=='Etheta':
                    comparitor=np.asarray([Dtheta[elev_inc,az_inc],Dtheta2[elev_inc,az_inc],Dtheta3[elev_inc,az_inc]])
                elif forming=='Ephi':
                    comparitor=np.asarray([Dphi[elev_inc,az_inc],Dphi2[elev_inc,az_inc],Dphi3[elev_inc,az_inc]])

                if np.any(np.isnan(comparitor)):
                    print('error')

                best_index=np.where(comparitor==np.max(comparitor))[0][0]
                if best_index==0:
                    directivity_map[elev_inc,az_inc,0,inc_res]=Dtheta[elev_inc,az_inc]
                    directivity_map[elev_inc,az_inc,1,inc_res]=Dphi[elev_inc,az_inc]
                    directivity_map[elev_inc,az_inc,2,inc_res]=Dtot[elev_inc,az_inc]
                elif best_index==1:
                    directivity_map[elev_inc,az_inc,0,inc_res]=Dtheta2[elev_inc,az_inc]
                    directivity_map[elev_inc,az_inc,1,inc_res]=Dphi2[elev_inc,az_inc]
                    directivity_map[elev_inc,az_inc,2,inc_res]=Dtot2[elev_inc,az_inc]
                elif best_index==2:
                    directivity_map[elev_inc,az_inc,0,inc_res]=Dtheta3[elev_inc,az_inc]
                    directivity_map[elev_inc,az_inc,1,inc_res]=Dphi3[elev_inc,az_inc]
                    directivity_map[elev_inc,az_inc,2,inc_res]=Dtot3[elev_inc,az_inc]

                inc_res+=1

    return directivity_map

@njit(cache=True, nogil=True)
def MaximumfieldMapDiscrete(Etheta,Ephi,source_coords,wavelength,az_res,elev_res,az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-180.0,180.0,19),forming='Total',total_solid_angle=(4*np.pi),phase_resolution=[24]):
    efield_map=np.zeros((elev_res,az_res,3,len(phase_resolution)),dtype=np.complex64)
    command_angles=np.zeros((2),dtype=np.float32)
    for az_inc in prange(az_res):
        for elev_inc in range(elev_res):
            inc_res=0
            for resolution in phase_resolution:
                command_angles[0]=az_range[az_inc]
                command_angles[1]=elev_range[elev_inc]
                steering_vector=np.zeros((1,3))
                steering_vector[0,0],steering_vector[0,1],steering_vector[0,2]=RF.sph2cart(np.radians(command_angles[0]), np.radians(command_angles[1]), 1)
                WS_weights=WavefrontWeights(source_coords,steering_vector,wavelength)
                MRC_weights=MRCWeights(Etheta, Ephi, command_angles,az_range=az_range,elev_range=elev_range)
                MRC_weights2=MRCWeights(Etheta, Ephi, command_angles,az_range=az_range,elev_range=elev_range,polarization_switch='Ephi')

                WS_weights=WeightTruncation(WS_weights,resolution)
                MRC_weights=WeightTruncation(MRC_weights,resolution)
                MRC_weights2=WeightTruncation(MRC_weights2,resolution)

                Ethetabeamformed=np.sum(MRC_weights.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
                Ephibeamformed=np.sum(MRC_weights.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
                Ethetabeamformed2=np.sum(MRC_weights2.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
                Ephibeamformed2=np.sum(MRC_weights2.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
                Ethetabeamformed3=np.sum(WS_weights.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
                Ephibeamformed3=np.sum(WS_weights.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
                Etotal=Ethetabeamformed**2+Ephibeamformed**2
                Etotal2=Ethetabeamformed2**2+Ephibeamformed2**2
                Etotal3=Ethetabeamformed3**2+Ephibeamformed3**2
                #Dtheta,Dphi,Dtot,_=directivity_transform(Ethetabeamformed,Ephibeamformed,az_range=az_range,elev_range=elev_range)
                #Dtheta2,Dphi2,Dtot2,_=directivity_transform(Ethetabeamformed2,Ephibeamformed2,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
                #Dtheta3,Dphi3,Dtot3,_=directivity_transform(Ethetabeamformed3,Ephibeamformed3,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
                if forming=='Total':
                    comparitor=np.asarray([Etotal[elev_inc,az_inc],Etotal2[elev_inc,az_inc],Etotal3[elev_inc,az_inc]])
                if forming=='Etheta':
                    comparitor=np.asarray([Ethetabeamformed[elev_inc,az_inc],Ethetabeamformed2[elev_inc,az_inc],Ethetabeamformed3[elev_inc,az_inc]])
                elif forming=='Ephi':
                    comparitor=np.asarray([Ephibeamformed[elev_inc,az_inc],Ephibeamformed2[elev_inc,az_inc],Ephibeamformed3[elev_inc,az_inc]])

                if np.any(np.isnan(comparitor)):
                    print('error')

                best_index=np.where(comparitor==np.max(comparitor))[0][0]
                if best_index==0:
                    efield_map[elev_inc,az_inc,0,inc_res]=Ethetabeamformed[elev_inc,az_inc]
                    efield_map[elev_inc,az_inc,1,inc_res]=Ephibeamformed[elev_inc,az_inc]
                    efield_map[elev_inc,az_inc,2,inc_res]=Etotal[elev_inc,az_inc]
                elif best_index==1:
                    efield_map[elev_inc,az_inc,0,inc_res]=Ethetabeamformed2[elev_inc,az_inc]
                    efield_map[elev_inc,az_inc,1,inc_res]=Ephibeamformed2[elev_inc,az_inc]
                    efield_map[elev_inc,az_inc,2,inc_res]=Etotal2[elev_inc,az_inc]
                elif best_index==2:
                    efield_map[elev_inc,az_inc,0,inc_res]=Ethetabeamformed3[elev_inc,az_inc]
                    efield_map[elev_inc,az_inc,1,inc_res]=Ephibeamformed3[elev_inc,az_inc]
                    efield_map[elev_inc,az_inc,2,inc_res]=Etotal3[elev_inc,az_inc]

                inc_res+=1

    return efield_map

def PatternTransform3D(norm_magnitudes,min_level=-40,linearswitch='log',logswitch='amplitude',az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-90.0,90.0,19),shell_range=1.0,centre=np.zeros((3),dtype=np.float32)):
    #create point cloud, coloured and radius scaled with the magnitudes
    if logswitch=='amplitude':
        log_multiplier=20.0
    elif logswitch=='power':
        log_multiplier=10.0


    azaz,elel=np.meshgrid(az_range,elev_range)
    sinks=np.zeros((len(np.ravel(azaz)),3),dtype=np.float32)
    if linearswitch=='log':
        norm_ranges=log_multiplier*np.log10(np.ravel(norm_magnitudes)+1e-24)
        norm_ranges[norm_ranges<min_level]=min_level
        norm_ranges=((norm_ranges-min_level)/np.abs(min_level))*shell_range
    elif linearswitch=='linear':
        norm_ranges=np.ravel(norm_magnitudes)*shell_range

    sinks[:,0],sinks[:,1],sinks[:,2]=RF.azeltocart(np.ravel(azaz),np.ravel(elel),np.ravel(norm_ranges))
    point_cloud=RF.points2pointcloud(sinks+centre)

    #normdata
    logdata=log_multiplier*np.log10(np.ravel(norm_magnitudes)+1e-24)
    viridis = cm.get_cmap('viridis', 40)
    np_colors=viridis(np.ravel(norm_magnitudes))
    point_cloud.colors = o3d.utility.Vector3dVector(np_colors[:,0:3])

    return point_cloud

def PatternTransformArbitary(norm_magnitudes,coordinates,min_level=-40,linearswitch='log',logswitch='amplitude'):
    #create point cloud, coloured and radius scaled with the magnitudes
    if logswitch=='amplitude':
        log_multiplier=20.0
    elif logswitch=='power':
        log_multiplier=10.0


    point_cloud=coordinates
    if linearswitch=='log':
        norm_ranges=log_multiplier*np.log10(np.ravel(norm_magnitudes)+1e-24)
        norm_ranges[norm_ranges<min_level]=min_level
        norm_ranges=((norm_ranges-min_level)/np.abs(min_level))*1.0
    elif linearswitch=='linear':
        norm_ranges=np.ravel(norm_magnitudes)*1.0

    #normdata
    logdata=log_multiplier*np.log10(np.ravel(norm_magnitudes)+1e-24)
    viridis = cm.get_cmap('viridis', 1000)
    np_colors=viridis(np.ravel(norm_magnitudes))
    point_cloud.colors = o3d.utility.Vector3dVector(np_colors[:,0:3])

    return point_cloud

def PatternTransformPhase3D(norm_magnitudes,phases,min_level=-40,az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-90.0,90.0,19),shell_range=1.0):
    #create point cloud, radius scaled with the magnitudes, and coloured by the phases
    azaz,elel=np.meshgrid(az_range,elev_range)
    sinks=np.zeros((len(np.ravel(azaz)),3),dtype=np.float32)
    norm_ranges=20*np.log10(np.ravel(norm_magnitudes)+1e-24)
    norm_ranges[norm_ranges<min_level]=min_level
    norm_ranges=((norm_ranges-min_level)/np.abs(min_level))*shell_range
    sinks[:,0],sinks[:,1],sinks[:,2]=RF.azeltocart(np.ravel(azaz),np.ravel(elel),np.ravel(norm_ranges))
    point_cloud=RF.points2pointcloud(sinks)

    #normdata in interval - pi to pi, and convert to interval from 0 to 1
    logdata=(np.ravel(phases)+np.pi)/(np.pi*2)
    viridis = cm.get_cmap('viridis', 181)
    np_colors=viridis(np.ravel(logdata))
    point_cloud.colors = o3d.utility.Vector3dVector(np_colors[:,0:3])

    return point_cloud

@njit(cache=True, nogil=True)
def directivity_transform(Etheta,Ephi,az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-90.0,90.0,19),total_solid_angle=(4*np.pi)):
    """
    transform Etheta and Ephi data into antenna directivity
    directivity is defined in terms of the power radiated in a specific direction, over the average radiated power
    power per unit solid angle
    """
    Dmax=np.zeros((3),dtype=np.float32)
    Umax=np.zeros((3),dtype=np.float32)
    Utheta=np.abs(Etheta)**2
    Uphi=np.abs(Ephi)**2
    Utotal=np.abs(Etheta)**2+np.abs(Ephi)**2
    power_sum=0
    temp_vector=np.zeros((4),dtype=np.float32)
    for elinc in range(len(elev_range)-1):
        for azinc in range(len(az_range)-1):
            temp_vector[0]=Utheta[elinc,azinc]
            temp_vector[1]=Utheta[elinc+1,azinc]
            temp_vector[2]=Utheta[elinc,azinc+1]
            temp_vector[3]=Utheta[elinc+1,azinc+1]
            r=np.mean(temp_vector)
            power_sum=power_sum+r*np.abs(np.sin(np.radians(az_range[azinc]+az_range[azinc+1])/2.0))

    etheta_power_sum=power_sum
    for elinc in range(len(elev_range)-1):
        for azinc in range(len(az_range)-1):
            temp_vector[0]=Uphi[elinc,azinc]
            temp_vector[1]=Uphi[elinc+1,azinc]
            temp_vector[2]=Uphi[elinc,azinc+1]
            temp_vector[3]=Uphi[elinc+1,azinc+1]
            r=np.mean(temp_vector)
            power_sum=power_sum+r*np.abs(np.sin(np.radians(az_range[azinc]+az_range[azinc+1])/2.0))

    ephi_power_sum=power_sum-etheta_power_sum
    Umax[0]=np.max(Utheta)
    Umax[1]=np.max(Uphi)
    Umax[2]=np.max(Utotal)
    Uav = power_sum*((np.radians(np.abs(az_range[1]-az_range[0])))*(np.radians(np.abs(elev_range[1]-elev_range[0]))))/(total_solid_angle)
    Dmax=Umax/Uav
    Dtheta=Utheta/Uav
    Dphi=Uphi/Uav
    Dtot=Utotal/Uav

    return Dtheta,Dphi,Dtot,Dmax

@njit(cache=True, nogil=True)
def WeightTruncation(weights,resolution):
    quant=2**resolution
    levels=np.zeros((weights.shape),dtype=np.float32)
    #shift numpy complex angles from -pi to pi, into 0 to 2pi, then quantise for `quant' levels
    levels=np.round((quant-1)*((np.angle(weights)+np.pi)/(2*np.pi)),0,levels)/(quant-1)
    new_weights=np.abs(weights)*np.exp(1j*((levels*2*np.pi)-np.pi))
    return new_weights

def PatternPlot(data,
                az,
                elev,
                pattern_min=-40,
                plot_max=0.0,
                plottype='Polar',
                logtype='amplitude',
                ticknum=6,
                title_text=None):
    """
    Plot the relavent 3D data in relative power (dB) or normalised directivity (dBi)

    Parameters
    -----------
    data : (2D array of floats or complex)
        the data to plot
    az : (2D array of floats)
        the azimuth angles for each datapoint in [data] in degrees
    elev : (2D array of floats)
        the elevation angles for each datapoint in [data] in degrees
    pattern_min : (float)
        the desired scale minimum in dB, default is [-40]
    plot_max : (float)
        the desired scale maximum in dB, default is [0]
    plottype : (string)
        the plot type, either [Polar], or [Cartesian-Surf], the default is [Polar]
    logtype : (string)
        the type of data being considered, either [amplitude] or [power], to ensure the correct logarithm is used, default is [amplitude]
    ticknum : (int)
        the number of ticks on the colorbar, default is [6]
    title_text : (string)
        the graph title, defaults to [None]

    Returns
    --------
    None
    """
    #condition data
    data=np.abs(data)
    #calculate log profile
    if logtype=='power':
        logdata=10*np.log10(data)
        bar_label='Relative Power (dB)'
    else:
        logdata=20*np.log10(data)

    if plot_max==0.0:
        logdata-=np.nanmax(logdata)
        bar_label = 'Normalised Directivity (dBi)'
    else:
        bar_label = 'Directivity (dBi)'

    logdata[logdata<=pattern_min]=pattern_min

    if plottype=='Polar':
        norm_log = (logdata - pattern_min) / np.abs(pattern_min)
        sinks=np.zeros((len(np.ravel(az)),3),dtype=np.float32)
        sinks[:,0],sinks[:,1],sinks[:,2]=RF.azeltocart(np.ravel(az),np.ravel(elev),np.ravel(norm_log))
        dist = np.sqrt(sinks[:,0].reshape(az.shape)**2 + sinks[:,1].reshape(az.shape)**2 + sinks[:,2].reshape(az.shape)**2)
        dist_max = np.max(dist)
        my_col = cm.viridis(dist/dist_max)
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        V = np.array([[1.1,0,0], [0,1.1,0], [0,0,1.1]],dtype=np.float32)
        origin = np.zeros((3,3),dtype=np.float32) # origin point
        offset=np.array([0.8,0.8,0.8],dtype=np.float32).reshape(1,3)
        ax.quiver(origin[0,0]-offset,origin[0,0]-offset,origin[0,0]-offset, V[0,0], V[1,0],V[2,0], color=['red'])
        ax.quiver(origin[0, 1] - offset, origin[0, 1] - offset, origin[0, 1] - offset, V[0, 1], V[1, 1], V[2, 1],color=['green'])
        ax.quiver(origin[0, 2] - offset, origin[0, 2] - offset, origin[0, 2] - offset, V[0, 2], V[1, 2], V[2, 2],color=['blue'])
        plot_handle=ax.plot_surface(sinks[:,0].reshape(az.shape),
                        sinks[:,1].reshape(az.shape),
                        sinks[:,2].reshape(az.shape),
                        facecolors=my_col,
                        linewidth=0,
                        antialiased=False,
                        clim=[0,1])

        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        plt.axis('off')
        #plot_handle.set_clim([0,1])
        cbar=fig.colorbar(plot_handle,ticks=np.linspace(0,1.0,ticknum),extend='both')
        cbar.ax.set_ylabel(bar_label)
        c_labels=np.linspace(pattern_min,plot_max,ticknum).astype('str')
        cbar.set_ticklabels(c_labels.tolist())
        p = Wedge((0, 0), 1.01, 0, 360, width=0.0001,color='gray')
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="x")
        p = Wedge((0, 0), 1.01, 0, 360, width=0.0001,color='gray')
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="y")
        p = Wedge((0, 0), 1.01, 0, 360, width=0.0001,color='gray')
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
        ax.view_init(elev=45., azim=-45)
        if title_text!=None:
            ax.set_title(title_text)

    elif plottype=='Cartesian-Surf':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(az, elev, logdata, cmap='viridis', edgecolor='none')
        ax.set_xlim([np.min(az),np.max(az)])
        ax.set_ylim([np.min(elev), np.max(elev)])
        ax.set_zlim([pattern_min,plot_max])
        ax.set_zticks(np.linspace(pattern_min, plot_max, ticknum))
        ax.set_xticks(np.linspace(-180, 180, 9))
        ax.set_yticks(np.linspace(-90, 90., 5))
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Elevation (degrees)')
        ax.set_zlabel(bar_label)
        if title_text!=None:
            ax.set_title(title_text)
    elif plottype=='Contour':
        fig, ax = plt.subplots(constrained_layout=True)
        origin = 'lower'
        levels=np.linspace(pattern_min,plot_max,ticknum*10)
        CS = ax.contourf(az, elev, logdata, levels,
                         cmap='viridis',
                         origin=origin)
        cbar = fig.colorbar(CS, ticks=np.linspace(pattern_min, plot_max, ticknum))
        cbar.ax.set_ylabel(bar_label)
        c_labels = np.linspace(pattern_min, plot_max, ticknum).astype('str')
        cbar.set_ticklabels(c_labels.tolist())
        ax.set_xlim([np.min(az), np.max(az)])
        ax.set_ylim([np.min(elev), np.max(elev)])
        ax.set_xticks(np.linspace(-180, 180, 9))
        ax.set_yticks(np.linspace(-90, 90., 13))
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Elevation (degrees)')
        levels2 = np.linspace(pattern_min, plot_max, ticknum)
        CS4 = ax.contour(az, elev, logdata, levels2,
                          colors=('k',),
                          linewidths=(2,),
                          origin=origin)
        ax.grid()
        if title_text!=None:
            ax.set_title(title_text)


# noinspection PyTypeChecker
@cuda.jit(device=True)
def point_directivity(Ea,Eb,az_range,el_range,interest_index):
    """
    compute the directivity at the point of interest in the farfield pattern
    """

    average_power=0.0
    directivity_results=cuda.local.array(shape=(3), dtype=float32)
    return directivity_results

@cuda.jit(device=True)
def EqualGainCombiningGPU(SteeringPattern,CommandIndex,weights):
    """
    equal gain combining algorithm based on the provided steering pattern and command index
    """
    weights=cmath.exp(-1j*np.angle(SteeringPattern[:,CommandIndex]))

    return weights

@cuda.jit
def GPUBeamformingMap(Etheta,Ephi,DirectivityMap,az_range,el_range,wavelength):
    """

    """
    az_inc, el_inc = cuda.grid(2)
    if az_inc < az_range.shape[0] and el_inc < el_range.shape[0]:

        DirectivityMap[az_inc, el_inc]=0