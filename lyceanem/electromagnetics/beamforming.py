import cmath
import copy

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import pylab as pl
import pyvista as pv
import scipy.stats
from matplotlib import cm
from matplotlib.patches import Wedge
from numba import cuda, float32, njit, prange
from scipy.spatial import distance

from ..raycasting import rayfunctions as RF
from ..utility import math_functions


def Steering_Efficiency(
    Dtheta, Dphi, Dtot, first_dimension_angle, second_dimension_angle, angular_coverage
):
    """
    Calculate Steering Efficiency for the provided pattern, in radians

    Parameters
    ----------
    Dtheta : numpy.ndarray of complex or float
        Directivity for the :math:`E\\theta` polarisation.
    Dphi : numpy.ndarray of complex or float
        Directivity for the :math:`E\\phi` polarisation.
    Dtot : numpy.ndarray of complex or float
        Total Directivity pattern
    first_dimension_angle : float
        the angle in radians of the first dimension, should be :math:`2\\pi` for a full azimuth sweep
    second_dimension_angle : float
        the angle in radians of the second dimension, should be :math:`\\pi` for a full elevation sweep
    angular_coverage : float
        the total angular coverage to be considered, should be :math:`4\\pi` steradians for a full sphere.

    Returns
    -------
    setheta : float
        steering efficiency in Dtheta
    sephi : float
        steering efficiency in Dphi
    setot : float
        steering efficiency in Dtotal

    """
    with np.errstate(divide="ignore"):
        a_index = 10 * np.log10(np.abs(Dtheta)) >= (
            10 * np.log10(np.nanmax(np.abs(Dtheta))) - 3
        )
        b_index = 10 * np.log10(np.abs(Dphi)) >= (
            10 * np.log10(np.nanmax(np.abs(Dphi))) - 3
        )
        tot_index = 10 * np.log10(np.abs(Dtot)) >= (
            10 * np.log10(np.nanmax(np.abs(Dtot))) - 3
        )
        # setheta = (np.sum(a_index) / (Dtheta.shape[0] * Dtheta.shape[1])) * 100
        # sephi = (np.sum(b_index) / (Dphi.shape[0] * Dphi.shape[1])) * 100
        # setot = (np.sum(tot_index) / (Dtot.shape[0] * Dtot.shape[1])) * 100
        setheta = (
            (np.sum(a_index) * (first_dimension_angle * second_dimension_angle))
            / angular_coverage
        ) * 100
        sephi = (
            (np.sum(b_index) * (first_dimension_angle * second_dimension_angle))
            / angular_coverage
        ) * 100
        setot = (
            (np.sum(tot_index) * (first_dimension_angle * second_dimension_angle))
            / angular_coverage
        ) * 100

    return setheta, sephi, setot


@njit(cache=True, nogil=True)
def WavefrontWeights(source_coords, steering_vector, wavelength):
    """

    calculate the weights for a given set of element coordinates, wavelength, and steering vector (cartesian)

    Parameters
    ----------
    source_coords : numpy.ndarray of float
        The coordinates of the elements, arranged in terms of x, y, and z coordinates
    steering_vector : numpy.ndarray of float
        The steering vector for the beamforming, arranged in terms of x, y, and z coordinates
    wavelength : float
        The wavelength of interest, used to calculate the phase delays

    Returns
    -------
    weights : numpy.ndarray of complex
        The wavefront weights for the specified steering vector, arranged as a 1D array of complex numbers

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

    Parameters
    ----------
    source_coords : numpy.ndarray of float
        The coordinates of the elements, arranged in terms of x, y, and z coordinates
    target_coord : numpy.ndarray of float
        The target coordinate for the beamforming, arranged in terms of x, y, and z coordinates
    wavelength : float
        The wavelength of interest, used to calculate the phase delays

    Returns
    -------
    weights : numpy.ndarray of complex
        The wavefront coherence weights for the specified target coordinate, arranged as a 1D array of complex numbers

    """
    weights = np.zeros((len(source_coords), 1), dtype=np.complex64)
    # calculate distances of coords from steering_vector by using it to calculate arbitarily distant point
    dist = distance.cdist(source_coords, target_coord)
    dist = dist - np.min(dist)
    # calculate required time delays, and then convert to phase delays
    delays = dist / scipy.constants.speed_of_light
    weights[:] = np.exp(
        1j * 2 * np.pi * (scipy.constants.speed_of_light / wavelength) * delays
    )
    return weights


def TimeDelayWeights(source_coords, steering_vector, magnitude=1.0, maximum_delay=10):
    """
    Generate time delay weights to focus on a target coordinate, with delays in nanoseconds

    Parameters
    ----------
    source_coords : numpy.ndarray of float
        The coordinates of the elements, arranged in terms of x, y, and z coordinates
    steering_vector : numpy.ndarray of float
        The steering vector for the beamforming, arranged in terms of x, y, and z coordinates
    magnitude : numpy.ndarray of float
        The magnitude of the weights, default is 1.0 but can be set to any value to allow for both calibrated transmit power, and amplitude weighting.
    maximum_delay : float
        The maximum delay to be applied to the weights, in nanoseconds. This is used to ensure that no delay is less than 0ns.

    Returns
    -------
    weights : numpy.ndarray of complex
        The time delay weights for the specified steering vector, arranged as a 1D array of complex numbers. For these weights for convineince the magnitude is the real part, while the time delay is the imaginary part.

    """
    weights = np.zeros((len(source_coords)), dtype=np.complex64)
    # calculate distances of coords from steering_vector by using it to calculate arbitarily distant point
    dist = np.sqrt(
        np.abs(
            (source_coords[:, 0] - steering_vector.ravel()[0] * 1e9) ** 2
            + (source_coords[:, 1] - steering_vector.ravel()[1] * 1e9) ** 2
            + (source_coords[:, 2] - steering_vector.ravel()[2] * 1e9) ** 2
        )
    )
    dist = dist - np.min(dist)
    # dist is now relative distance from target vector, want to delay closest maximum, then have farthest be delayed zero
    # calculate required time delays, and then convert to nanoseconds, stored as a complex number with the magnitude weights
    delays = (dist / scipy.constants.speed_of_light) * 1e9
    delays = np.max(delays) - delays
    weights[:] = magnitude + delays * 1j
    return weights


def TimeDelayBeamform(excitation_function, weights, sampling_rate):
    """

    The time delay beamform function takes an n by 2 or n by 3 array, and applies the supplied time delay weights to each by rolling each slice of the array by the required number of sampling intervals.
    Only positive time delays should be applied, but if positive and negative values are required for weighting, a constant value should be applied so that no delay is less than 0ns.


    Parameters
    ----------
    excitation_function : numpy.ndarray of float
        excitation function for time domain antenna array
    weights : numpy.ndarray of complex
        magnitude is the real part, while the imaginary part is the time_delay in ns
    sampling_rate : float
        sampling rate in Hz, used to calculate the shifts to align the excitation function.

    Returns
    -------
    beamformed_function : numpy.ndarray of float
        The beamformed excitation function, with the time delays applied.

    """
    # extract delays and convert to seconds, then integrer shifts
    delay = (sampling_rate * (np.imag(weights) * 1e-9)).astype(int).ravel()
    if len(excitation_function.shape) == 2:
        magnitudes = np.real(weights).reshape(-1, 1)
    elif len(excitation_function.shape) == 3:
        magnitudes = np.real(weights).reshape(-1, 1, 1)
    elif len(excitation_function.shape) == 4:
        magnitudes = np.real(weights).reshape(-1, 1, 1, 1)

    for row in range(excitation_function.shape[0]):
        excitation_function = shift_slice(excitation_function, row, delay[row])

    excitation_function = excitation_function * magnitudes

    return excitation_function


def shift_slice(array, row, shift):
    """
    :meta private:
    Shift a slice of a 2D, 3D, or 4D array by a specified number of elements, filling the shifted elements with zeros.

    Parameters
    ----------
    array : TYPE
        DESCRIPTION.
    row : TYPE
        DESCRIPTION.
    shift : TYPE
        DESCRIPTION.

    Returns
    -------
    array : TYPE
        DESCRIPTION.

    """
    if len(array.shape) == 2:
        array[row, :] = np.roll(array[row, :], shift)
        if shift > 0:
            array[row, :shift] = 0
        elif shift < 0:
            array[row, -shift:] = 0

    if len(array.shape) == 3:
        array[row, :, :] = np.roll(array[row, :, :], shift)
        if shift > 0:
            array[row, :, :shift] = 0
        elif shift < 0:
            array[row, :, -shift:] = 0

    if len(array.shape) == 4:
        array[row, :, :, :] = np.roll(array[row, :, :, :], shift)
        if shift > 0:
            array[row, :, :, :shift] = 0
        elif shift < 0:
            array[row, :, :, -shift:] = 0

    return array


@njit(cache=True, nogil=True)
def EGCWeights(
    Etheta,
    Ephi,
    command_angles,
    polarization_switch="Etheta",
    az_range=None,
    elev_range=None,
):
    """

    calculate the equal gain combining weights for a given set of element coordinates, wavelength, and command angles (az,elev)

    Parameters
    ----------
    Etheta : numpy.ndarray of complex
        The Etheta polarisation farfield patterns, arranged in terms of the number of elements, azimuth resolution, and elevation resolution
    Ephi : numpy.ndarray of complex
        The Ephi polarisation farfield patterns, arranged in terms of the number of elements, azimuth resolution, and elevation resolution
    command_angles : numpy.ndarray of float
        The command angles for the beamfroming, arranged as [azimuth,elevation]
    polarization_switch : str
        The polarisation to be used for the beamforming, either [Etheta] or [Ephi]. Default is [Etheta]
    az_range : numpy.ndarray of float
        The azimuth values for the farfield mesh, arranged from smallest to largest
    elev_range : numpy.ndarray of float
        The elevation values for the farfield mesh, arranged from smallest to largest

    Returns
    -------
    weights : numpy.ndarray of complex
        The equal gain combining weights for the specified command angles, arranged as a 1D array of complex numbers


    """
    if az_range is None:
        az_range = np.linspace(-180.0, 180.0, 19)

    if elev_range is None:
        elev_range = np.linspace(-90.0, 90.0, 19)
    weights = np.zeros((Etheta.shape[0]), dtype=np.complex64)
    az_index = np.argmin(np.abs(az_range - command_angles[0]))
    elev_index = np.argmin(np.abs(elev_range - command_angles[1]))
    if polarization_switch == "Etheta":
        angle_vector = -np.angle(Etheta[:, elev_index, az_index].astype(np.complex64))
    else:
        angle_vector = -np.angle(Ephi[:, elev_index, az_index].astype(np.complex64))

    weights[:] = np.exp(1j * angle_vector)
    return weights


def OAMWeights(x, y, mode):
    """

    Generate Orbital Angular Momentum (OAM) mode weights, based upon the radial angle of each element.

    Parameters
    ----------
    x : numpy.ndarray of float
        The x coordinates of the elements
    y : numpy.ndarray of float
        The y coordinates of the elements
    mode : int
        The OAM mode to be used for the weights, should be an integer value

    Returns
    -------
    weights : numpy.ndarray of complex
        The OAM weights for the specified mode, arranged as a 1D array of complex numbers

    """
    # assumed array is x directed
    angles = np.arctan2(x, y)
    weights = np.zeros((len(x)), dtype=np.complex64)
    weights = np.exp(1j * mode * angles)
    return weights


def OAMFourier(
    Ex,
    Ey,
    Ez,
    coordinates,
    mode_limit,
    first_dimension,
    second_dimension,
    coord_format="AzEl",
):
    """
    Compute the mode index, mode coefficients, and mode probabilities with the co and crosspolar (Etheta,Ephi), tentatively also supports for Ex,Ey,Ez

    Parameters
    ----------
    Ex : numpy.ndarray of complex
        The Ex component of the electric field, arranged in terms of azimuth resolution and elevation resolution
    Ey : numpy.ndarray of complex
        The Ey component of the electric field, arranged in terms of azimuth resolution and elevation resolution
    Ez : numpy.ndarray of complex
        The Ez component of the electric field, arranged in terms of azimuth resolution and elevation resolution
    coordinates : numpy.ndarray of float
        The coordinates of the elements, arranged in terms of x, y, and z coordinates
    mode_limit : int
        The maximum OAM mode to be considered for the calculation, should be an integer value
    first_dimension : numpy.ndarray of float
        if the coordinate system is spherical, this should be the azimuth range
    second_dimension : numpy.ndarray of float
        if the coordinate system is spherical, this should be the elevation range
    coord_format : str
        The coordinate system to be used for the calculation, either [xyz] or [AzEl]. Default is [AzEl]

    Returns
    -------
    mode_index : numpy.ndarray of int
        The OAM mode index, arranged as a 1D array of integers

    mode_coefficients : numpy.ndarray of float
        The OAM mode coefficients, arranged in terms of the number of modes and the number of polarisation components

    mode_probabilites : numpy.ndarray of float
        The OAM mode probabilities, arranged in terms of the number of modes and the number of polarisation components

    """
    # establish coordinate set, in this case theta and phi, but would work just as well with elevation and azimuth, assume that array is propagating in the z+ direction.
    if coord_format == "xyz":
        mode_index, mode_coefficients, mode_probabilites = OAMFourierCartesian(
            Ex, Ey, Ez, coordinates, mode_limit
        )
    elif coord_format == "AzEl":
        mode_index, mode_coefficients, mode_probabilites = OAMFourierSpherical(
            Ex, Ey, Ez, coordinates, mode_limit, first_dimension, second_dimension
        )

    return mode_index, mode_coefficients, mode_probabilites


def OAMFourierCartesian(Ex, Ey, Ez, coordinates, mode_limit):
    """

    assume propagation is in the Ez dimension

    :meta private:
    """
    mode_index = np.linspace(-mode_limit, mode_limit, mode_limit * 2 + 1)
    mode_coefficients = np.zeros((mode_index.shape[0], 3), dtype=np.complex64)
    az, el, r = math_functions.cart2sph(
        coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    )
    # a coefficient of mode m, at angle theta is defined in terms of the
    # integral of the phi dimension in the range 0 to 2pi.
    for oam_m in range(len(mode_index)):
        mode_coefficients[oam_m, 0] = (1 / (2 * np.pi)) * np.sum(
            Ex.ravel() * np.exp(1j * mode_index[oam_m] * az), axis=0
        )
        mode_coefficients[oam_m, 1] = (1 / (2 * np.pi)) * np.sum(
            Ey.ravel() * np.exp(1j * mode_index[oam_m] * az), axis=0
        )
        mode_coefficients[oam_m, 2] = (1 / (2 * np.pi)) * np.sum(
            Ez.ravel() * np.exp(1j * mode_index[oam_m] * az), axis=0
        )

    powers = np.sum(np.abs(mode_coefficients**2), axis=0)

    mode_probabilities = np.zeros((mode_index.shape[0], 3), dtype=np.float32)
    mode_probabilities[:, 0] = np.abs(
        (1 / np.sum(powers)) * (mode_coefficients[:, 0] ** 2)
    )
    mode_probabilities[:, 1] = np.abs(
        (1 / np.sum(powers)) * (mode_coefficients[:, 1] ** 2)
    )
    mode_probabilities[:, 2] = np.abs(
        (1 / np.sum(powers)) * (mode_coefficients[:, 2] ** 2)
    )

    return mode_index, mode_coefficients, mode_probabilities


def OAMFourierSpherical(Ex, Ey, Ez, coordinates, mode_limit, az_range, elev_range):
    """

    assume propagation is in the Ez dimension

    :meta private:

    """
    mode_index = np.linspace(-mode_limit, mode_limit, mode_limit * 2 + 1)
    mode_coefficients = np.zeros((mode_index.shape[0], len(elev_range), 3))
    # a coefficient of mode m, at angle theta is defined in terms of the
    # integral of the azimuth dimension in the range -pi to pi.
    for oam_m in range(mode_index):
        # mode_coefficients(oam_m,:,1)=(1/(2*pi))*sum(CoPolar(:,1:theta_limit).*exp(-sqrt(-1)*mode_index(oam_m)*p'*ones(1,theta_limit)));
        # mode_coefficients(oam_m,:,2)=(1/(2*pi))*sum(CrossPolar(:,1:theta_limit).*exp(-sqrt(-1)*mode_index(oam_m)*p'*ones(1,theta_limit)));
        mode_coefficients[oam_m, :, 0] = (1 / (2 * np.pi)) * np.sum(
            Ex[:, :] * np.exp(-1j * mode_index[oam_m]), axis=0
        )

    # copolar_power=sum(sum(abs(mode_coefficients(:,:,1)).^2));
    # crosspolar_power=sum(sum(abs(mode_coefficients(:,:,2)).^2));

    mode_probabilities = np.zeros((mode_index.shape[0], 2), dtype=np.float32)
    # mode_prob(:,1)=(1/sum([copolar_power,crosspolar_power]))*sum(abs(mode_coefficients(:,:,1)').^2);
    # mode_prob(:,2)=(1/sum([copolar_power,crosspolar_power]))*sum(abs(mode_coefficients(:,:,2)').^2);

    return mode_index, mode_coefficients, mode_probabilities


# @njit(parallel=True,cache=True, nogil=True)
def MaximumDirectivityMap(
    Etheta,
    Ephi,
    source_coords,
    wavelength,
    az_range=None,
    elev_range=None,
    az_index=None,
    elev_index=None,
    forming="Total",
    total_solid_angle=(4 * np.pi),
    phase_resolution=24,
):
    """

    Uses wavefront beamsteering, and equal gain combining algorithms to steer the antenna array to each possible
    command angle in the farfield, mapping out the maximum achieved directivity at the command angle for each command
    angle set.

    Parameters
    ----------
    Etheta : numpy.ndarray of complex
        The :math:`E\\theta` polarisation farfield patterns, arranged in terms of the number of elements, azimuth resolution, and elevation resolution
    Ephi : numpy.ndarray of complex
        The :math:`E\\phi` polarisation farfield patterns, arranged in terms of the number of elements, azimuth resolution, and elevation resolution
    source_coords : :class:`meshio.Mesh`
        The source coordinates of each element, corresponding to the order of element patterns in :math:`E\\theta` or :math:`E\\phi`. Units should be m
    wavelength : float
        The wavelength of interest
    az_range : numpy.ndarray of float
        The azimuth values for the farfield mesh, arranged from smallest to largest
    elev_range : numpy.ndarray of float
        The elevation values for the farfield mesh, arranged from smallest to largest
    az_index : numpy.ndarray of int
        optional parameter, can be specified as an index of the azimuth values of interest via indexing, defaults to [None], which ensures all values are covered
    elev_index : numpy.ndarray of int
        optional parameter, can be specified as an index of the elevation values of interest via indexing, defaults to [None], which ensures all values are covered
    forming : str
        Which polarisation should be beamformed, the default is [Total], beamforming the total directivity pattern,
        avoiding issues with elements which have a strongly :math:`E\\theta` or :math:`E\\phi` pattern. This can also be set
        to [Etheta] or [Ephi]
    total_solid_angle : float
        the total solid angle covered by the farfield patterns, this defaults to :math:`4\\pi` for a full spherical pattern
    phase_resolution : int
        the desired phase resolution of the beamforming architecture in bits. Default is [24], which means no practical
        truncation will occur. If beam mapping at a single resolution is required, then this can be set
        between 2 and 24. If multiple values are required, it may be more efficient to
        use :func:`lyceanem.electromagnetics.beamforming.MaximumDirectivityMapDiscrete`, which allows a 1D array of
        resolutions to be supplied, and produces a maximum directivity map for each.

    Returns
    -------
    directivity_map : numpy.ndarray of float
        The achieved maximum directivity map. At each point the directivity corresponds to the achieved directivity at
        that command angle. Arranged as elev axis, azimuth axis, Dtheta,Dphi,Dtot

    """
    if az_range is None:
        az_range = np.linspace(-180.0, 180.0, 19)
    if elev_range is None:
        elev_range = np.linspace(-90.0, 90.0, 19)
    if elev_index is None:
        # if no elev index is provided then generate for all possible values (assumes every elevation point is of interest)
        elev_index = np.linspace(0, len(elev_range) - 1, len(elev_range)).astype(int)

    if az_index is None:
        # if no az index is provided then generate for all possible values (assumes every azimuth point is of interest)
        az_index = np.linspace(0, len(az_range) - 1, len(az_range)).astype(int)

    az_res = len(az_index)
    elev_res = len(elev_index)
    source_points = np.asarray(source_coords.points)
    directivity_map = np.zeros((elev_res, az_res, 3))
    command_angles = np.zeros((2), dtype=np.float32)

    for az_inc in range(az_res):
        for elev_inc in range(elev_res):
            command_angles[0] = az_range[az_index[az_inc]]
            command_angles[1] = elev_range[elev_index[elev_inc]]
            steering_vector = np.zeros((1, 3))
            (
                steering_vector[0, 0],
                steering_vector[0, 1],
                steering_vector[0, 2],
            ) = math_functions.sph2cart(
                np.radians(command_angles[0]), np.radians(command_angles[1]), 1
            )
            WS_weights = WavefrontWeights(source_points, steering_vector, wavelength)
            EGC_weights = EGCWeights(
                Etheta.astype(np.complex64),
                Ephi.astype(np.complex64),
                command_angles,
                az_range=az_range,
                elev_range=elev_range,
            )
            EGC_weights2 = EGCWeights(
                Etheta.astype(np.complex64),
                Ephi.astype(np.complex64),
                command_angles,
                az_range=az_range,
                elev_range=elev_range,
                polarization_switch="Ephi",
            )
            if phase_resolution <= 12:
                WS_weights = WeightTruncation(WS_weights, phase_resolution)
                EGC_weights = WeightTruncation(EGC_weights, phase_resolution)
                EGC_weights2 = WeightTruncation(EGC_weights2, phase_resolution)

            Ethetabeamformed = np.sum(
                EGC_weights.reshape(Etheta.shape[0], 1, 1) * Etheta, axis=0
            )
            Ephibeamformed = np.sum(
                EGC_weights.reshape(Etheta.shape[0], 1, 1) * Ephi, axis=0
            )
            Ethetabeamformed2 = np.sum(
                EGC_weights2.reshape(Etheta.shape[0], 1, 1) * Etheta, axis=0
            )
            Ephibeamformed2 = np.sum(
                EGC_weights2.reshape(Etheta.shape[0], 1, 1) * Ephi, axis=0
            )
            Ethetabeamformed3 = np.sum(
                WS_weights.reshape(Etheta.shape[0], 1, 1) * Etheta, axis=0
            )
            Ephibeamformed3 = np.sum(
                WS_weights.reshape(Etheta.shape[0], 1, 1) * Ephi, axis=0
            )
            Dtheta, Dphi, Dtot, _ = directivity_transform(
                Ethetabeamformed,
                Ephibeamformed,
                az_range=az_range,
                elev_range=elev_range,
            )
            Dtheta2, Dphi2, Dtot2, _ = directivity_transform(
                Ethetabeamformed2,
                Ephibeamformed2,
                az_range=az_range,
                elev_range=elev_range,
                total_solid_angle=total_solid_angle,
            )
            Dtheta3, Dphi3, Dtot3, _ = directivity_transform(
                Ethetabeamformed3,
                Ephibeamformed3,
                az_range=az_range,
                elev_range=elev_range,
                total_solid_angle=total_solid_angle,
            )
            if forming == "Total":
                comparitor = np.asarray(
                    [
                        Dtot[elev_inc, az_inc],
                        Dtot2[elev_inc, az_inc],
                        Dtot3[elev_inc, az_inc],
                    ]
                )
            elif forming == "Etheta":
                comparitor = np.asarray(
                    [
                        Dtheta[elev_inc, az_inc],
                        Dtheta2[elev_inc, az_inc],
                        Dtheta3[elev_inc, az_inc],
                    ]
                )
            elif forming == "Ephi":
                comparitor = np.asarray(
                    [
                        Dphi[elev_inc, az_inc],
                        Dphi2[elev_inc, az_inc],
                        Dphi3[elev_inc, az_inc],
                    ]
                )

            if np.any(np.isnan(comparitor)):
                print("error")

            best_index = np.where(comparitor == np.max(comparitor))[0][0]
            if best_index == 0:
                directivity_map[elev_inc, az_inc, 0] = Dtheta[elev_inc, az_inc]
                directivity_map[elev_inc, az_inc, 1] = Dphi[elev_inc, az_inc]
                directivity_map[elev_inc, az_inc, 2] = Dtot[elev_inc, az_inc]
            elif best_index == 1:
                directivity_map[elev_inc, az_inc, 0] = Dtheta2[elev_inc, az_inc]
                directivity_map[elev_inc, az_inc, 1] = Dphi2[elev_inc, az_inc]
                directivity_map[elev_inc, az_inc, 2] = Dtot2[elev_inc, az_inc]
            elif best_index == 2:
                directivity_map[elev_inc, az_inc, 0] = Dtheta3[elev_inc, az_inc]
                directivity_map[elev_inc, az_inc, 1] = Dphi3[elev_inc, az_inc]
                directivity_map[elev_inc, az_inc, 2] = Dtot3[elev_inc, az_inc]

    return directivity_map


# @njit(parallel=False, cache=True, nogil=True)
def MaximumDirectivityMapDiscrete(
    Etheta,
    Ephi,
    source_coords,
    wavelength,
    az_range=None,
    elev_range=None,
    az_index=None,
    elev_index=None,
    forming="Total",
    total_solid_angle=(4 * np.pi),
    phase_resolution=np.asarray([24]),
):
    """

    Uses wavefront beamsteering, and equal gain combining algorithms to steer the antenna array to each possible
    command angle in the farfield, mapping out the maximum achieved directivity at the command angle for each command
    angle set.


    Parameters
    ----------
    Etheta : numpy.ndarray of complex
        The :math:`E\\theta` polarisation farfield patterns, arranged in terms of the number of elements, azimuth resolution, and elevation resolution
    Ephi : numpy.ndarray of complex
        The :math:`E\\phi` polarisation farfield patterns, arranged in terms of the number of elements, azimuth resolution, and elevation resolution
    source_coords : :class:`meshio.Mesh`
        The source coordinates of each element, corresponding to the order of element patterns in :math:`E\theta` and :math:`E\phi`. Units should be m
    wavelength : float
        The wavelength of interest
    az_res : int
        Azimuth resolution
    elev_res : int
        Elevation resolution
    az_range : numpy.ndarray of float
        The azimuth values for the farfield mesh, arranged from smallest to largest
    elev_range : numpy.ndarray of float
        The elevation values for the farfield mesh, arranged from smallest to largest
    forming : str
        Which polarisation should be beamformed, the default is [Total], beamforming the total directivity pattern,
        avoiding issues with elements which have a strongly $E\theta$ or $E\phi$ pattern. This can also be set
        to [Etheta] or [Ephi]
    total_solid_angle : float
        the total solid angle covered by the farfield patterns, this defaults to :math:`4\pi` for a full spherical pattern
    phase_resolution : numpy.ndarray of int
        the desired phase resolution of the beamforming architecture in bits. Default is [24], which means no practical
        truncation will occur. If beam mapping is desired at a single resolution is required, then this can be set
        between 2 and 24, if more than one resolution value is required, then a 1D array of values can be specified.
        resolutions to be supplied, and produces a maximum directivity map for each.

    Returns
    -------
    directivity_map : numpy.ndarray of float
        The achieved maximum directivity map (a 3D numpy array for each phase resolution) . For each phase resolution for each point the directivity corresponds to the achieved directivity at
        that command angle.

    """
    if az_range is None:
        az_range = np.linspace(-180.0, 180.0, 19)
    if elev_range is None:
        elev_range = np.linspace(-90.0, 90.0, 19)

    if elev_index is None:
        # if no elev index is provided then generate for all possible values (assumes every elevation point is of interest)
        elev_index = np.linspace(0, len(elev_range) - 1, len(elev_range)).astype(int)

    if az_index is None:
        # if no az index is provided then generate for all possible values (assumes every azimuth point is of interest)
        az_index = np.linspace(0, len(az_range) - 1, len(az_range)).astype(int)

    az_res = len(az_index)
    elev_res = len(elev_index)
    source_points = np.asarray(source_coords.points)
    directivity_map = np.zeros(
        (elev_res, az_res, 3, phase_resolution.shape[0]), dtype=np.float32
    )
    command_angles = np.zeros((2), dtype=np.float32)
    for az_inc in range(az_res):
        for elev_inc in range(elev_res):
            inc_res = 0
            for res_inc in range(phase_resolution.shape[0]):
                resolution = phase_resolution[res_inc]
                command_angles[0] = az_range[az_inc]
                command_angles[1] = elev_range[elev_inc]
                steering_vector = np.zeros((1, 3))
                (
                    steering_vector[0, 0],
                    steering_vector[0, 1],
                    steering_vector[0, 2],
                ) = math_functions.sph2cart(
                    np.radians(command_angles[0]), np.radians(command_angles[1]), 1
                )
                WS_weights = WavefrontWeights(
                    source_points, steering_vector, wavelength
                )
                EGC_weights = EGCWeights(
                    Etheta,
                    Ephi,
                    command_angles,
                    az_range=az_range,
                    elev_range=elev_range,
                )
                EGC_weights2 = EGCWeights(
                    Etheta,
                    Ephi,
                    command_angles,
                    az_range=az_range,
                    elev_range=elev_range,
                    polarization_switch="Ephi",
                )

                WS_weights = WeightTruncation(WS_weights, resolution)
                EGC_weights = WeightTruncation(EGC_weights, resolution)
                EGC_weights2 = WeightTruncation(EGC_weights2, resolution)

                Ethetabeamformed = np.sum(
                    EGC_weights.reshape(Etheta.shape[0], 1, 1) * Etheta, axis=0
                )
                Ephibeamformed = np.sum(
                    EGC_weights.reshape(Etheta.shape[0], 1, 1) * Ephi, axis=0
                )
                Ethetabeamformed2 = np.sum(
                    EGC_weights2.reshape(Etheta.shape[0], 1, 1) * Etheta, axis=0
                )
                Ephibeamformed2 = np.sum(
                    EGC_weights2.reshape(Etheta.shape[0], 1, 1) * Ephi, axis=0
                )
                Ethetabeamformed3 = np.sum(
                    WS_weights.reshape(Etheta.shape[0], 1, 1) * Etheta, axis=0
                )
                Ephibeamformed3 = np.sum(
                    WS_weights.reshape(Etheta.shape[0], 1, 1) * Ephi, axis=0
                )
                Dtheta, Dphi, Dtot, _ = directivity_transform(
                    Ethetabeamformed,
                    Ephibeamformed,
                    az_range=az_range,
                    elev_range=elev_range,
                )
                Dtheta2, Dphi2, Dtot2, _ = directivity_transform(
                    Ethetabeamformed2,
                    Ephibeamformed2,
                    az_range=az_range,
                    elev_range=elev_range,
                    total_solid_angle=total_solid_angle,
                )
                Dtheta3, Dphi3, Dtot3, _ = directivity_transform(
                    Ethetabeamformed3,
                    Ephibeamformed3,
                    az_range=az_range,
                    elev_range=elev_range,
                    total_solid_angle=total_solid_angle,
                )
                if forming == "Total":
                    comparitor = np.asarray(
                        [
                            Dtot[elev_inc, az_inc],
                            Dtot2[elev_inc, az_inc],
                            Dtot3[elev_inc, az_inc],
                        ]
                    )
                elif forming == "Etheta":
                    comparitor = np.asarray(
                        [
                            Dtheta[elev_inc, az_inc],
                            Dtheta2[elev_inc, az_inc],
                            Dtheta3[elev_inc, az_inc],
                        ]
                    )
                elif forming == "Ephi":
                    comparitor = np.asarray(
                        [
                            Dphi[elev_inc, az_inc],
                            Dphi2[elev_inc, az_inc],
                            Dphi3[elev_inc, az_inc],
                        ]
                    )

                if np.any(np.isnan(comparitor)):
                    print("error")

                best_index = np.where(comparitor == np.max(comparitor))[0][0]
                if best_index == 0:
                    directivity_map[elev_inc, az_inc, 0, inc_res] = Dtheta[
                        elev_inc, az_inc
                    ]
                    directivity_map[elev_inc, az_inc, 1, inc_res] = Dphi[
                        elev_inc, az_inc
                    ]
                    directivity_map[elev_inc, az_inc, 2, inc_res] = Dtot[
                        elev_inc, az_inc
                    ]
                elif best_index == 1:
                    directivity_map[elev_inc, az_inc, 0, inc_res] = Dtheta2[
                        elev_inc, az_inc
                    ]
                    directivity_map[elev_inc, az_inc, 1, inc_res] = Dphi2[
                        elev_inc, az_inc
                    ]
                    directivity_map[elev_inc, az_inc, 2, inc_res] = Dtot2[
                        elev_inc, az_inc
                    ]
                elif best_index == 2:
                    directivity_map[elev_inc, az_inc, 0, inc_res] = Dtheta3[
                        elev_inc, az_inc
                    ]
                    directivity_map[elev_inc, az_inc, 1, inc_res] = Dphi3[
                        elev_inc, az_inc
                    ]
                    directivity_map[elev_inc, az_inc, 2, inc_res] = Dtot3[
                        elev_inc, az_inc
                    ]

                inc_res += 1

    return directivity_map


@njit(cache=True, nogil=True)
def MaximumfieldMapDiscrete(
    Etheta,
    Ephi,
    source_coords,
    wavelength,
    az_res,
    elev_res,
    az_range=None,
    elev_range=None,
    forming="Total",
    total_solid_angle=(4 * np.pi),
    phase_resolution=[24],
):
    if az_range is None:
        az_range = np.linspace(-180.0, 180.0, 19)
    if elev_range is None:
        elev_range = np.linspace(-90.0, 90.0, 19)
    efield_map = np.zeros(
        (elev_res, az_res, 3, len(phase_resolution)), dtype=np.complex64
    )
    command_angles = np.zeros((2), dtype=np.float32)
    for az_inc in prange(az_res):
        for elev_inc in range(elev_res):
            inc_res = 0
            for resolution in phase_resolution:
                command_angles[0] = az_range[az_inc]
                command_angles[1] = elev_range[elev_inc]
                steering_vector = np.zeros((1, 3))
                (
                    steering_vector[0, 0],
                    steering_vector[0, 1],
                    steering_vector[0, 2],
                ) = math_functions.sph2cart(
                    np.radians(command_angles[0]), np.radians(command_angles[1]), 1
                )
                WS_weights = WavefrontWeights(
                    source_coords, steering_vector, wavelength
                )
                EGC_weights = EGCWeights(
                    Etheta,
                    Ephi,
                    command_angles,
                    az_range=az_range,
                    elev_range=elev_range,
                )
                EGC_weights2 = EGCWeights(
                    Etheta,
                    Ephi,
                    command_angles,
                    az_range=az_range,
                    elev_range=elev_range,
                    polarization_switch="Ephi",
                )

                WS_weights = WeightTruncation(WS_weights, resolution)
                EGC_weights = WeightTruncation(EGC_weights, resolution)
                EGC_weights2 = WeightTruncation(EGC_weights2, resolution)

                Ethetabeamformed = np.sum(
                    EGC_weights.reshape(Etheta.shape[0], 1, 1) * Etheta, axis=0
                )
                Ephibeamformed = np.sum(
                    EGC_weights.reshape(Etheta.shape[0], 1, 1) * Ephi, axis=0
                )
                Ethetabeamformed2 = np.sum(
                    EGC_weights2.reshape(Etheta.shape[0], 1, 1) * Etheta, axis=0
                )
                Ephibeamformed2 = np.sum(
                    EGC_weights2.reshape(Etheta.shape[0], 1, 1) * Ephi, axis=0
                )
                Ethetabeamformed3 = np.sum(
                    WS_weights.reshape(Etheta.shape[0], 1, 1) * Etheta, axis=0
                )
                Ephibeamformed3 = np.sum(
                    WS_weights.reshape(Etheta.shape[0], 1, 1) * Ephi, axis=0
                )
                Etotal = Ethetabeamformed**2 + Ephibeamformed**2
                Etotal2 = Ethetabeamformed2**2 + Ephibeamformed2**2
                Etotal3 = Ethetabeamformed3**2 + Ephibeamformed3**2
                # Dtheta,Dphi,Dtot,_=directivity_transform(Ethetabeamformed,Ephibeamformed,az_range=az_range,elev_range=elev_range)
                # Dtheta2,Dphi2,Dtot2,_=directivity_transform(Ethetabeamformed2,Ephibeamformed2,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
                # Dtheta3,Dphi3,Dtot3,_=directivity_transform(Ethetabeamformed3,Ephibeamformed3,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
                if forming == "Total":
                    comparitor = np.asarray(
                        [
                            Etotal[elev_inc, az_inc],
                            Etotal2[elev_inc, az_inc],
                            Etotal3[elev_inc, az_inc],
                        ]
                    )
                if forming == "Etheta":
                    comparitor = np.asarray(
                        [
                            Ethetabeamformed[elev_inc, az_inc],
                            Ethetabeamformed2[elev_inc, az_inc],
                            Ethetabeamformed3[elev_inc, az_inc],
                        ]
                    )
                elif forming == "Ephi":
                    comparitor = np.asarray(
                        [
                            Ephibeamformed[elev_inc, az_inc],
                            Ephibeamformed2[elev_inc, az_inc],
                            Ephibeamformed3[elev_inc, az_inc],
                        ]
                    )

                if np.any(np.isnan(comparitor)):
                    print("error")

                best_index = np.where(comparitor == np.max(comparitor))[0][0]
                if best_index == 0:
                    efield_map[elev_inc, az_inc, 0, inc_res] = Ethetabeamformed[
                        elev_inc, az_inc
                    ]
                    efield_map[elev_inc, az_inc, 1, inc_res] = Ephibeamformed[
                        elev_inc, az_inc
                    ]
                    efield_map[elev_inc, az_inc, 2, inc_res] = Etotal[elev_inc, az_inc]
                elif best_index == 1:
                    efield_map[elev_inc, az_inc, 0, inc_res] = Ethetabeamformed2[
                        elev_inc, az_inc
                    ]
                    efield_map[elev_inc, az_inc, 1, inc_res] = Ephibeamformed2[
                        elev_inc, az_inc
                    ]
                    efield_map[elev_inc, az_inc, 2, inc_res] = Etotal2[elev_inc, az_inc]
                elif best_index == 2:
                    efield_map[elev_inc, az_inc, 0, inc_res] = Ethetabeamformed3[
                        elev_inc, az_inc
                    ]
                    efield_map[elev_inc, az_inc, 1, inc_res] = Ephibeamformed3[
                        elev_inc, az_inc
                    ]
                    efield_map[elev_inc, az_inc, 2, inc_res] = Etotal3[elev_inc, az_inc]

                inc_res += 1

    return efield_map


@njit(cache=True, nogil=True)
def directivity_transform(
    Etheta,
    Ephi,
    az_range=None,
    elev_range=None,
    total_solid_angle=(4 * np.pi),
):
    """
    Transform the Etheta and Ephi field vectors into antenna directivity.
    The directivity is defined in terms of the power radiated in a specific direction, over the average radiated power

    Parameters
    ----------
    Etheta : numpy.ndarray of complex
        The :math:`E\\theta` polarisation farfield patterns, arranged in terms of the number of elements, azimuth resolution, and elevation resolution
    Ephi : numpy.ndarray of complex
        The :math:`E\\phi` polarisation farfield patterns, arranged in terms of the number of elements, azimuth resolution, and elevation resolution
    az_range : numpy.ndarray of float
        The azimuth values for the farfield mesh, arranged from smallest to largest
    elev_range : numpy.ndarray of float
        The elevation values for the farfield mesh, arranged from smallest to largest
    total_solid_angle : float
        the total solid angle covered by the farfield patterns, this defaults to $4\pi$ for a full spherical pattern

    Returns
    -------
    Dtheta : numpy.ndarray of float
        Directivity for the :math:`E\\theta` polarisation.
    Dphi : numpy.ndarray of float
        Directivity for the :math:`E\\phi` polarisation.
    Dtot : numpy.ndarray of float
        The achieved directivity map for the total polarisation.
    Dmax : numpy.ndarray of float
        The maximum directivity for each polarisation.


    """
    if az_range is None:
        az_range = np.linspace(-180.0, 180.0, 19)
    if elev_range is None:
        elev_range = np.linspace(-90.0, 90.0, 19)
    # transform Etheta and Ephi data into antenna directivity
    # directivity is defined in terms of the power radiated in a specific direction, over the average radiated power
    # power per unit solid angle
    Dmax = np.zeros((3), dtype=np.float32)
    Umax = np.zeros((3), dtype=np.float32)
    Utheta = np.abs(Etheta**2)
    Uphi = np.abs(Ephi**2)
    Utotal = np.abs(Etheta**2) + np.abs(Ephi**2)
    power_sum = 0.0
    temp_vector = np.zeros((4), dtype=np.float32)
    temp_vector2 = np.zeros((4), dtype=np.float32)
    for elinc in range(len(elev_range) - 1):
        for azinc in range(len(az_range) - 1):
            temp_vector[0] = Utheta[elinc, azinc]
            temp_vector[1] = Utheta[elinc + 1, azinc]
            temp_vector[2] = Utheta[elinc, azinc + 1]
            temp_vector[3] = Utheta[elinc + 1, azinc + 1]
            temp_vector2[0] = Uphi[elinc, azinc]
            temp_vector2[1] = Uphi[elinc + 1, azinc]
            temp_vector2[2] = Uphi[elinc, azinc + 1]
            temp_vector2[3] = Uphi[elinc + 1, azinc + 1]
            r1 = np.mean(temp_vector)
            r2 = np.mean(temp_vector2)
            power_sum = (
                power_sum
                + r1
                * np.abs(
                    np.sin(np.radians(az_range[azinc] + az_range[azinc + 1]) / 2.0)
                )
                + r2
                * np.abs(
                    np.sin(np.radians(az_range[azinc] + az_range[azinc + 1]) / 2.0)
                )
            )

    Umax[0] = np.nanmax(Utheta)
    Umax[1] = np.nanmax(Uphi)
    Umax[2] = np.nanmax(Utotal)
    Uav = (
        power_sum
        * (
            (np.radians(az_range[1] - az_range[0]))
            * (np.radians(elev_range[1] - elev_range[0]))
        )
        / total_solid_angle
    )
    Dmax = Umax / Uav
    Dtheta = Utheta / Uav
    Dphi = Uphi / Uav
    Dtot = Utotal / Uav

    return Dtheta, Dphi, Dtot, Dmax


@njit(cache=True, nogil=True)
def WeightTruncation(weights, resolution):
    """
    :meta private:

    Truncates the phase of the weights to a specified resolution in bits.

    """
    quant = 2**resolution
    levels = np.zeros((weights.shape), dtype=np.complex64)
    # shift numpy complex angles from -pi to pi, into 0 to 2pi, then quantise for `quant' levels
    levels = np.round(
        ((quant - 1) * ((np.angle(weights) + np.pi) / (2 * np.pi))), 0, levels
    ) / (quant - 1)
    new_weights = np.abs(weights) * np.exp(1j * ((levels * 2 * np.pi) - np.pi))
    return new_weights


def AnimatedPlot(
    dimension1,
    dimension2,
    dimension3,
    data,
    pattern_min=-40.0,
    pattern_max=0.0,
    dimension1label=None,
    dimension2label=None,
    colorbarlabel=None,
    title_text=None,
    ticknum=9,
    fps=5,
    save_location=None,
):
    def animate(i):
        # title_text='Sampled Phase {:.2f} Wavelengths from the Transmitting Antenna'.format(dimension3[i])
        ax.clear()
        ax.contourf(
            dimension1, dimension2, data[:, :, i], levels, cmap="viridis", origin=origin
        )
        ax.contour(
            dimension1, dimension2, data[:, :, i], levels, colors=("k",), origin=origin
        )
        ax.set_xlim([np.min(dimension1), np.max(dimension1)])
        ax.set_ylim([np.min(dimension2), np.max(dimension2)])
        ax.grid()
        # ax.set_xticks(np.linspace(-180, 180, 9))
        # ax.set_yticks(np.linspace(-90, 90.0, 13))
        if dimension1label != None:
            ax.set_xlabel(dimension1label)
        if dimension2label != None:
            ax.set_ylabel(dimension2label)
        if title_text != None:
            ax.set_title(title_text.format(dimension3[i]))

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ticknum = 9
    fig, ax = plt.subplots(constrained_layout=True)
    origin = "lower"
    # pattern_min = -np.pi / 2
    # pattern_max = np.pi / 2
    levels = np.linspace(pattern_min, pattern_max, 73)
    CS = ax.contourf(
        dimension1, dimension2, data[:, :, 0], levels, cmap="viridis", origin=origin
    )
    cbar = fig.colorbar(CS, ticks=np.linspace(pattern_min, pattern_max, ticknum))
    # bar_label="Sampled Phase Angle (Radians)"

    c_label_values = np.linspace(pattern_min, pattern_max, ticknum)
    c_labels = np.char.mod("%.2f", c_label_values)
    cbar.set_ticklabels(c_labels.tolist())
    ax.set_xlim([np.min(dimension1), np.max(dimension1)])
    ax.set_ylim([np.min(dimension2), np.max(dimension2)])
    # ax.set_xticks(np.linspace(-180, 180, 9))
    # ax.set_yticks(np.linspace(-90, 90.0, 13))
    ax.set_xlabel("x ($\lambda$)")
    ax.set_ylabel("y ($\lambda$)")
    # setup for 3dB contours
    # contournum = np.ceil((pattern_max - pattern_min) / 3).astype(int)
    # levels2 = np.linspace(-contournum * 3, plot_max, contournum + 1)
    # title_text=None
    ax.grid()
    if dimension1label != None:
        ax.set_xlabel(dimension1label)
    if dimension2label != None:
        ax.set_ylabel(dimension2label)
    if colorbarlabel != None:
        cbar.ax.set_ylabel(colorbarlabel)
    if title_text != None:
        ax.set_title(title_text.format(dimension3[0]))

    ani = animation.FuncAnimation(fig, animate, 100, interval=50, blit=False)
    plt.show()

    if save_location != None:
        # f = r"C:/Users/lycea/Documents/10-19 Research Projects/farfieldanimation.gif"
        writergif = animation.PillowWriter(fps=fps)
        ani.save(save_location, writer=writergif)


def PatternPlot(
    data,
    az,
    elev,
    pattern_min=-40,
    plot_max=0.0,
    plottype="Polar",
    logtype="amplitude",
    ticknum=6,
    title_text=None,
):
    """

    Plot the relevant 3D data in relative power (dB) or normalised directivity (dBi)


    Parameters
    -----------
    data : numpy.ndarray of float
        the data to plot
    az : numpy.ndarray of float
        the azimuth angles for each datapoint in [data] in degrees
    elev : numpy.ndarray of float
        the elevation angles for each datapoint in [data] in degrees
    pattern_min : float
        the desired scale minimum in dB, default is [-40]
    plot_max : float
        the desired scale maximum in dB, default is [0]
    plottype : str
        the plot type, either [Polar], [Cartesian-Surf], or [Contour]. The default is [Polar]
    logtype : str
        the type of data being considered, either [amplitude] or [power], to ensure the correct logarithm is used, default is [amplitude]
    ticknum : int
        the number of ticks on the colorbar, default is [6]
    title_text : str
        the graph title, defaults to [None]

    Returns
    --------
    None
    """
    # condition data
    data = np.abs(data)
    # calculate log profile
    if logtype == "power":
        logdata = 10 * np.log10(data)
        bar_label = "Relative Power (dB)"
    else:
        logdata = 20 * np.log10(data)

    if plot_max == 0.0:
        logdata -= np.nanmax(logdata)
        bar_label = "Normalised Directivity (dBi)"
    else:
        bar_label = "Directivity (dBi)"

    logdata[logdata <= pattern_min] = pattern_min

    if plottype == "Polar":
        norm_log = (logdata - pattern_min) / np.abs(pattern_min)
        sinks = np.zeros((len(np.ravel(az)), 3), dtype=np.float32)
        sinks[:, 0], sinks[:, 1], sinks[:, 2] = RF.azeltocart(
            np.ravel(az), np.ravel(elev), np.ravel(norm_log)
        )
        dist = np.sqrt(
            sinks[:, 0].reshape(az.shape) ** 2
            + sinks[:, 1].reshape(az.shape) ** 2
            + sinks[:, 2].reshape(az.shape) ** 2
        )
        dist_max = np.max(dist)
        my_col = cm.viridis(dist / dist_max)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        V = np.array([[1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]], dtype=np.float32)
        origin = np.zeros((3, 3), dtype=np.float32)  # origin point
        offset = np.array([0.8, 0.8, 0.8], dtype=np.float32).reshape(1, 3)
        ax.quiver(
            origin[0, 0] - offset,
            origin[0, 0] - offset,
            origin[0, 0] - offset,
            V[0, 0],
            V[1, 0],
            V[2, 0],
            color=["red"],
        )
        ax.quiver(
            origin[0, 1] - offset,
            origin[0, 1] - offset,
            origin[0, 1] - offset,
            V[0, 1],
            V[1, 1],
            V[2, 1],
            color=["green"],
        )
        ax.quiver(
            origin[0, 2] - offset,
            origin[0, 2] - offset,
            origin[0, 2] - offset,
            V[0, 2],
            V[1, 2],
            V[2, 2],
            color=["blue"],
        )
        plot_handle = ax.plot_surface(
            sinks[:, 0].reshape(az.shape),
            sinks[:, 1].reshape(az.shape),
            sinks[:, 2].reshape(az.shape),
            facecolors=my_col,
            linewidth=0,
            antialiased=False,
            clim=[0, 1],
        )

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.axis("off")
        # plot_handle.set_clim([0,1])
        cbar = fig.colorbar(
            plot_handle, ticks=np.linspace(0, 1.0, ticknum), extend="both"
        )
        cbar.ax.set_ylabel(bar_label)
        c_labels = np.linspace(pattern_min, plot_max, ticknum).astype("str")
        cbar.set_ticklabels(c_labels.tolist())
        p = Wedge((0, 0), 1.01, 0, 360, width=0.0001, color="gray")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="x")
        p = Wedge((0, 0), 1.01, 0, 360, width=0.0001, color="gray")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="y")
        p = Wedge((0, 0), 1.01, 0, 360, width=0.0001, color="gray")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
        ax.view_init(elev=45.0, azim=-45)
        if title_text != None:
            ax.set_title(title_text)

    elif plottype == "Cartesian-Surf":
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(az, elev, logdata, cmap="viridis", edgecolor="none")
        ax.set_xlim([np.min(az), np.max(az)])
        ax.set_ylim([np.min(elev), np.max(elev)])
        ax.set_zlim([pattern_min, plot_max])
        ax.set_zticks(np.linspace(pattern_min, plot_max, ticknum))
        ax.set_xticks(np.linspace(-180, 180, 9))
        ax.set_yticks(np.linspace(-90, 90.0, 5))
        ax.set_xlabel("Azimuth (degrees)")
        ax.set_ylabel("Elevation (degrees)")
        ax.set_zlabel(bar_label)
        if title_text != None:
            ax.set_title(title_text)
    elif plottype == "Contour":
        fig, ax = plt.subplots(constrained_layout=True)
        origin = "lower"
        levels = np.linspace(pattern_min, plot_max, ticknum * 10)
        CS = ax.contourf(az, elev, logdata, levels, cmap="viridis", origin=origin)
        cbar = fig.colorbar(CS, ticks=np.linspace(pattern_min, plot_max, ticknum))
        cbar.ax.set_ylabel(bar_label)
        c_labels = np.linspace(pattern_min, plot_max, ticknum).astype("str")
        cbar.set_ticklabels(c_labels.tolist())
        ax.set_xlim([np.min(az), np.max(az)])
        ax.set_ylim([np.min(elev), np.max(elev)])
        ax.set_xticks(np.linspace(-180, 180, 9))
        ax.set_yticks(np.linspace(-90, 90.0, 13))
        ax.set_xlabel("Azimuth (degrees)")
        ax.set_ylabel("Elevation (degrees)")
        # setup for 3dB contours
        contournum = np.ceil((plot_max - pattern_min) / 3).astype(int)
        levels2 = np.linspace(-contournum * 3, plot_max, contournum + 1)
        if pattern_min < -40:
            line_spec_width = 0.5
        else:
            line_spec_width = 1

        CS4 = ax.contour(
            az,
            elev,
            logdata,
            levels2,
            colors=("k",),
            linewidths=(line_spec_width,),
            origin=origin,
        )
        ax.grid()
        if title_text != None:
            ax.set_title(title_text)

    plt.show()


def PatternPlot2D(
    data,
    az,
    pattern_min=-40,
    plot_max=0.0,
    logtype="amplitude",
    ticknum=6,
    line_labels=None,
    title_text=None,
    fontsize=16,
):
    """
    Plot the relevant 2D data in relative power (dB) or normalised directivity (dBi) in a polar plot.

    Parameters
    ----------
    data : numpy.ndarray of float
        The data to plot, should be a 1D array for a single line or a 2D array for multiple lines.
    az : numpy.ndarray of float
        The azimuth angles for each datapoint in [data] in degrees.
    pattern_min : float
        The desired scale minimum in dB, default is [-40].
    plot_max : float
        The desired scale maximum in dB, default is [0].
    logtype : str
        The type of data being considered, either "amplitude" or "power", to ensure the correct logarithm is used, default is "amplitude".
    ticknum : int
        The number of ticks on the radial axis, default is [6].
    line_labels : list of str, optional
        Labels for each line in the plot, if provided, will be used in the legend.
    title_text : str, optional
        The graph title, defaults to [None].
    fontsize : int, optional
        The font size for the plot, default is [16].

    Returns
    -------
    None

    """
    # condition data
    data = np.abs(data)

    if data.ndim > 1:
        # multi line plot, condition data, as second axis should be the number of different lines.
        multiline = True
        num_lines = data.shape[1]
    else:
        multiline = False

    # calculate log profile
    if logtype == "power":
        logdata = 10 * np.log10(data)
        bar_label = "Relative Power (dB)"
    else:
        logdata = 20 * np.log10(data)

    if plot_max == 0.0:
        logdata -= np.nanmax(logdata)
        bar_label = "Normalised Directivity (dBi)"
    else:
        bar_label = "Directivity (dBi)"

    logdata[logdata <= pattern_min] = pattern_min
    tick_marks = np.linspace(pattern_min, plot_max, ticknum)
    az_rad = np.radians(az)
    pl.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    if multiline == True:
        for line in range(num_lines):
            if not (line_labels is None):
                ax.plot(az_rad, logdata[:, line], label=line_labels[line])
            else:
                ax.plot(az_rad, logdata[:, line])
    else:
        ax.plot(az_rad, logdata)

    ax.set_rmax(plot_max)
    ax.set_rticks(tick_marks, fontsize=fontsize)  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    if not (line_labels is None):
        # legend position
        legend_angle = np.deg2rad(30)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(
                0.5 + np.cos(legend_angle) / 2,
                0.5 + np.sin(legend_angle) / 2,
            ),
            fontsize=fontsize,
        )

    if not (title_text is None):
        ax.set_title(title_text, va="bottom", fontsize=fontsize)

    label_angle = np.deg2rad(280)
    ax.text(label_angle, plot_max * 1.2, bar_label)
    plt.show()


# noinspection PyTypeChecker
@cuda.jit(device=True)
def point_directivity(Ea, Eb, az_range, el_range, interest_index):
    """

    compute the directivity at the point of interest in the farfield pattern

    :meta private:

    """

    average_power = 0.0
    directivity_results = cuda.local.array(shape=(3), dtype=float32)
    return directivity_results


@cuda.jit(device=True)
def EqualGainCombiningGPU(SteeringPattern, CommandIndex, weights):
    """
    :meta private:

    equal gain combining algorithm based on the provided steering pattern and command index

    """
    weights = cmath.exp(-1j * np.angle(SteeringPattern[:, CommandIndex]))

    return weights


@cuda.jit
def GPUBeamformingMap(Etheta, Ephi, DirectivityMap, az_range, el_range, wavelength):
    """
    :meta private:

    Unfinished GPU beamforming map function.
    """
    az_inc, el_inc = cuda.grid(2)
    if az_inc < az_range.shape[0] and el_inc < el_range.shape[0]:
        DirectivityMap[az_inc, el_inc] = 0


def create_display_mesh(
    field_data,
    field_radius=1.0,
    label="Poynting_Vector_(Magnitude_(W/m2))",
    log_type="power",
    plot_max=None,
    dynamic_range=40,
):
    """
    Create a display mesh for the field data, with normals and scaled points based on the field radius.

    Parameters
    ----------
    field_data : :type:`meshio.Mesh`
        The field data to create the display mesh
    field_radius : float
        The radius of the field, used to scale the normals and points in the mesh, the maximum of the selected data will be scaled to this value.
    label : str
        The label for the data to be plotted, this should be a key in the point_data of the field_data mesh.
    log_type : str
        The type of logarithm to use for the data, either "amplitude" or "power". Defaults to "power".
    plot_max : float, optional
        The maximum value for the plot, if None, it will be calculated as the maximum of the log data rounded up to the nearest 5. Defaults to None.
    dynamic_range : float
        The desired maximum plot range in decibels. The default is 40dB.

    Returns
    -------
    :type:`pyvista.PolyData`
        A PyVista PolyData object containing the display mesh with normals and scaled points.

    """
    from ..utility.mesh_functions import meshio_to_pyvista

    pattern_mesh = meshio_to_pyvista(field_data).copy().clean(tolerance=1e-6)
    # pattern_mesh=pv.PolyData(normals).delaunay_2d(inplace=True)
    # pattern_mesh.compute_normals(inplace=True,flip_normals=False) # for some reason the computed normals are all inwards unless I set flip_normals to True

    if log_type == "amplitude":
        log_multiplier = 20.0
    elif log_type == "power":
        log_multiplier = 10.0
        # create Normed dB Values for labelled value
    if len(pattern_mesh.point_data[label].shape) > 1:
        # assumes complex data in iq format is to be plotted.
        logdata = log_multiplier * np.log10(
            np.abs(
                pattern_mesh.point_data[label][:, 0]
                + 1j * pattern_mesh.point_data[label][:, 1]
            )
        )
    else:
        logdata = log_multiplier * np.log10(pattern_mesh.point_data[label])
    if np.nanmax(logdata) <= 0.0:
        logdata -= np.nanmax(logdata)

    if plot_max is None:
        pattern_max = np.ceil(np.nanmax(logdata) / 5) * 5.0
    else:
        pattern_max = plot_max

    pattern_min = pattern_max - dynamic_range

    normed_logdata = copy.deepcopy(logdata)
    normed_logdata -= np.max(logdata)
    normed_logdata[normed_logdata < pattern_min] = pattern_min

    norm_log = ((normed_logdata - pattern_min) / np.abs(pattern_min)) * field_radius
    pattern_mesh.point_data["Normed Power"] = norm_log
    pattern_mesh.point_data["Gain (dB)"] = logdata
    pattern_mesh.points = pattern_mesh.center + pattern_mesh[
        "Normals"
    ] * norm_log.reshape(-1, 1)
    return pattern_mesh


def AnimatedPatterns(
    pattern_list,
    command_vectors,
    field_radius=1.0,
    plot_scalar="Poynting_Vector_(Magnitude_(W/m2))",
    log_type="power",
    plot_max=None,
    filename="BeamformingExport.mp4",
):
    """
    Create an animated plot of the beamforming patterns, with a command vector overlay.

    Parameters
    ----------
    pattern_list : list
        A list of :type:`meshio.Mesh` objects, each containing the field data for a different beamformed response.
    command_vectors : numpy.ndarray of float
        The (N,Azimuth,Elevation) command vectors for each pattern in the pattern_list, where N is the number of patterns.
    field_radius : float
        The radius of the field, used to scale the normals and points in the mesh, the maximum of the selected data will be scaled to this value.
    plot_scalar : str, optional
        The label for the data to be plotted, this should be a key in the point_data of the field_data mesh. Defaults to "Poynting_Vector_(Magnitude_(W/m2))".
    log_type : str, optional
        The type of logarithm to use for the data, either "amplitude" or "power". Defaults to "power".
    plot_max : float, optional
        The maximum value for the plot, if None, it will be calculated as the maximum of the log data rounded up to the nearest 5. Defaults to None.
    filename : str, optional
        The filename for the output video file, defaults to "BeamformingExport.mp4".

    Returns
    -------
    None
        This function creates an animated video file of the beamforming patterns, which will be saved to the specified filename.

    """
    pattern_max = plot_max

    pattern_min = pattern_max - 40
    display_list = []
    for inc in range(len(pattern_list)):
        display_list.append(
            create_display_mesh(
                pattern_list[inc],
                field_radius=field_radius,
                label=plot_scalar,
                log_type=log_type,
                plot_max=plot_max,
            )
        )

    time_horizontal_fraction = 0.25
    screen_size = 3008
    pl = pv.Plotter(off_screen=True)
    # Open a movie file
    pl.open_movie(filename)
    pl.ren_win.SetSize([screen_size, screen_size])
    actor = pl.add_mesh(
        display_list[0], scalars="Gain (dB)", clim=[pattern_min, pattern_max]
    )
    steering_vector = pl.add_text(
        "Command Vector = ({:3.0f},{:3.0f})".format(
            command_vectors[0, 0], command_vectors[0, 1]
        ),
        position=(time_horizontal_fraction * screen_size, 0.95 * screen_size),
        font_size=50,
    )
    pl.add_axes()
    pl.write_frame()
    from tqdm import tqdm

    for frame in tqdm(range(len(pattern_list))):
        pl.remove_actor(actor)
        pl.remove_actor(steering_vector)
        actor = pl.add_mesh(
            display_list[frame], scalars="Gain (dB)", clim=[pattern_min, pattern_max]
        )
        steering_vector = pl.add_text(
            "Command Vector = ({:3.0f},{:3.0f})".format(
                command_vectors[frame, 0], command_vectors[frame, 1]
            ),
            position=(time_horizontal_fraction * screen_size, 0.95 * screen_size),
            font_size=50,
        )
        pl.write_frame()

    pl.close()
