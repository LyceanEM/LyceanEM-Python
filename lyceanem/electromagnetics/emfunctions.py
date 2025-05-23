import numpy as np
import pyvista as pv
from importlib_resources import files
import lyceanem.electromagnetics.data as data
from lyceanem.utility.mesh_functions import pyvista_to_meshio


def excitation_function(
    aperture_points,
    desired_e_vector,
    phase_shift="none",
    wavelength=1.0,
    steering_vector=np.zeros((1, 3)),
    transmit_power=1.0,
    local_projection=True,
):
    """
    Calculate the excitation function for the given aperture points, desired electric field vector, phase shift, wavelength, steering vector, transmit power, and local projection. This will generate the normalised field intensities for the desired total transmit power, and beamforming algorithm. The aperture mesh should have Normals and Area associated with each point for full functionality.

    The phase shift can be set to 'none', 'wavefront', or 'coherence'. The steering vector is the command vector for the desired beamforming algorithm, and the transmit power is the total power to be transmitted. The local projection flag indicates whether the electric field vectors should be projected based upon the local surface normals.

    Parameters
    ----------
    aperture_points : meshio.Mesh

    desired_e_vector : numpy.ndarray

    phase_shift : str

    wavelength : float

    steering_vector : numpy.ndarray

    transmit_power : float

    local_projection : bool

    Returns
    -------
    calibrated_amplitude_weights : numpy.ndarray
        The calibrated amplitude weights for the given aperture points, desired electric field vector, phase shift, wavelength, steering vector, and total transmit power.
    """

    if local_projection:
        # as export all points imposes the transformation from local to global frame on the points and associated normal vectors, no rotation is required within calculate_conformalVectors
        from .empropagation import calculate_conformalVectors

        aperture_weights = calculate_conformalVectors(
            desired_e_vector, aperture_points.point_data["Normals"], np.eye(3)
        )
    else:
        aperture_weights = np.repeat(
            desired_e_vector, aperture_points.points.shape[0], axis=0
        )

    if phase_shift == "wavefront":
        from .beamforming import WavefrontWeights

        source_points = np.asarray(aperture_points.points)
        phase_weights = WavefrontWeights(source_points, steering_vector, wavelength)
        aperture_weights = aperture_weights * phase_weights.reshape(-1, 1)
    if phase_shift == "coherence":
        from .beamforming import ArbitaryCoherenceWeights

        source_points = np.asarray(aperture_points.points)
        phase_weights = ArbitaryCoherenceWeights(
            source_points, steering_vector, wavelength
        )
        aperture_weights = aperture_weights * phase_weights.reshape(-1, 1)

    from ..utility.math_functions import discrete_transmit_power

    if "Area" in aperture_points.point_data.keys():
        areas = aperture_points.point_data["Area"]
    else:
        areas = np.zeros((aperture_points.points.shape[0]))
        areas[:] = (wavelength * 0.5) ** 2

    calibrated_amplitude_weights = discrete_transmit_power(
        aperture_weights, areas, transmit_power
    )
    return calibrated_amplitude_weights


def fresnel_zone(pointA, pointB, wavelength, zone=1):
    """
    based on the provided points, wavelength, and zone number, calculate the fresnel zone. This is defined as an ellipsoid for which the difference in distance between the line AB (line of sight), and AP+PB (reflected wave) is a constant multiple of ($n\dfrac{\lambda}{2}$).

    Parameters
    -----------
    pointA : numpy.ndarray
        Point A as a number array of floats
    pointB : numpy.ndarray
        Point B as a number array of floats
    wavelength : float
        wavelength of interest
    zone : int
        highest order Fresnel zone to be calculated. The default is 1, which is the first order fresnel zone.


    Returns
    --------
    ellipsoid : meshio.Mesh
        A meshio object representing the fresnel zone, allowing for visualisation and boolean operations for decimating a larger triangle mesh.
    """

    foci = np.stack([pointA, pointB])
    center = np.mean(foci, axis=0)
    ellipsoid = pv.ParametricEllipsoid()
    major_axis = (pointB - pointA) * 0.5

    separation = np.linalg.norm(major_axis)
    # using binomial approximation via assuming the distance between A and B is much larger than the maximum radius of the zone
    fresnel_radius = 0.5 * ((zone * wavelength * separation) ** 0.5)
    import scipy.spatial.transform as spt

    pose = np.eye(4)
    pose[:3, :3] = spt.Rotation.align_vectors(major_axis / separation, [1, 0, 0])[
        0
    ].as_matrix()
    pose[:3, 3] = centere

    ellipsoid = pyvista_to_meshio(pv.ParametricEllipsoid(
        separation * 0.5, fresnel_radius, fresnel_radius
    ).transform(pose))
    return ellipsoid


def field_magnitude_phase(field_data):
    """
    Calculate the magnitude and phase of the electric field vectors in the given field data. The function checks for the presence of the electric field components in both Cartesian (Ex, Ey, Ez) and spherical (E(theta), E(phi)) coordinates. If the components are present, it calculates the magnitude and phase for each component and adds them to the point data.

    Parameters
    ----------
    field_data : meshio.Mesh
        The field data containing the electric field components in either Cartesian or spherical coordinates.

    Returns
    -------
    field_data : meshio.Mesh
        The field data containing the resultant magnitude and phase components for each electric field vector. The magnitude and phase are stored as `Ex-Magnitude``, ``Ex-Phase``, ``Ey-Magnitude``, ``Ey-Phase``, ``Ez-Magnitude``, and ``Ez-Phase`` for Cartesian coordinates, and `E(theta)-Magnitude`, `E(theta)-Phase`, `E(phi)-Magnitude`, and `E(phi)-Phase` for spherical coordinates.
    """

    if all(
        k in field_data.point_data.keys()
        for k in (
            "Ex-Real",
            "Ex-Imag",
            "Ey-Real",
            "Ey-Imag",
            "Ez-Real",
            "Ez-Imag",
        )
    ):
        # Exyz exsists in the dataset
        Ex = field_data.point_data["Ex-Real"] + 1j * field_data.point_data["Ex-Imag"]
        Ey = field_data.point_data["Ey-Real"] + 1j * field_data.point_data["Ey-Imag"]
        Ez = field_data.point_data["Ez-Real"] + 1j * field_data.point_data["Ez-Imag"]
        field_data.point_data["Ex-Magnitude"] = np.abs(Ex)
        field_data.point_data["Ex-Phase"] = np.angle(Ex)
        field_data.point_data["Ey-Magnitude"] = np.abs(Ey)
        field_data.point_data["Ey-Phase"] = np.angle(Ey)
        field_data.point_data["Ez-Magnitude"] = np.abs(Ez)
        field_data.point_data["Ez-Phase"] = np.angle(Ez)

    if all(
        k in field_data.point_data.keys()
        for k in (
            "E(theta)-Real",
            "E(theta)-Imag",
            "E(phi)-Real",
            "E(phi)-Imag",
        )
    ):
        # E(theta) and E(phi) are not present in the data
        Etheta = (
            field_data.point_data["E(theta)-Real"]
            + 1j * field_data.point_data["E(theta)-Imag"]
        )
        Ephi = (
            field_data.point_data["E(phi)-Real"]
            + 1j * field_data.point_data["E(phi)-Imag"]
        )
        field_data.point_data["E(theta)-Magnitude"] = np.abs(Etheta)
        field_data.point_data["E(theta)-Phase"] = np.angle(Etheta)
        field_data.point_data["E(phi)-Magnitude"] = np.abs(Ephi)
        field_data.point_data["E(phi)-Phase"] = np.angle(Ephi)

    return field_data


def extract_electric_fields(field_data):
    """
    Return the electric field vectors in the provided mesh.

    Parameters
    ----------
    field_data : meshio.Mesh
        The field data containing the electric field components in either cartesian format

    Returns
    -------
    fields : numpy.ndarray
        The electric field vectors in the provided mesh. The shape of the array is (n_points, 3), where n_points is the number of points in the mesh.
    """
    fields = np.array(
        [
            field_data.point_data["Ex-Real"].ravel()
            + 1j * field_data.point_data["Ex-Imag"].ravel(),
            field_data.point_data["Ey-Real"].ravel()
            + 1j * field_data.point_data["Ey-Imag"].ravel(),
            field_data.point_data["Ez-Real"].ravel()
            + 1j * field_data.point_data["Ez-Imag"].ravel(),
        ]
    ).transpose()

    return fields


def EthetaEphi_to_Exyz(field_data):
    if not all(
        k in field_data.point_data.keys() for k in ("theta_(Radians)", "phi_(Radians)")
    ):
        # theta and phi don't exist in the dataset
        from lyceanem.geometry.geometryfunctions import theta_phi_r

        field_data = theta_phi_r(field_data)
    #    # output Ex,Ey,Ez on point data.
    Espherical = np.array(
        [
            (
                field_data.point_data["E(theta)-Real"]
                + 1j * field_data.point_data["E(theta)-Imag"]
            ).ravel(),
            (
                field_data.point_data["E(phi)-Real"]
                + 1j * field_data.point_data["E(phi)-Imag"]
            ).ravel(),
            np.zeros((field_data.points.shape[0])),
        ]
    ).transpose()
    # local_coordinates=field_data.points-field_data.center
    # radial_distance=np.linalg.norm(local_coordinates,axis=1)
    # theta=np.arccos(local_coordinates[:,2]/radial_distance)
    # phi=np.arctan2(local_coordinates[:,1],local_coordinates[:,0])
    # prime_vector=np.array([1.0,0,0])
    # etheta=np.zeros(field_data.points.shape[0])
    # ephi=np.zeros(field_data.points.shape[0])
    # conversion matrix from EthetaEphi to Exyz, the inverse operation is via transposing the matrix.
    ex = (
        Espherical[:, 2]
        * np.sin(field_data.point_data["theta_(Radians)"])
        * np.cos(field_data.point_data["phi_(Radians)"])
        + Espherical[:, 0]
        * np.cos(field_data.point_data["theta_(Radians)"])
        * np.cos(field_data.point_data["phi_(Radians)"])
        - Espherical[:, 1] * np.sin(field_data.point_data["phi_(Radians)"])
    )
    ey = (
        Espherical[:, 2]
        * np.sin(field_data.point_data["theta_(Radians)"])
        * np.sin(field_data.point_data["phi_(Radians)"])
        + Espherical[:, 0]
        * np.cos(field_data.point_data["theta_(Radians)"])
        * np.sin(field_data.point_data["phi_(Radians)"])
        + Espherical[:, 1] * np.cos(field_data.point_data["phi_(Radians)"])
    )
    ez = Espherical[:, 2] * np.cos(
        field_data.point_data["theta_(Radians)"]
    ) + Espherical[:, 0] * np.sin(field_data.point_data["theta_(Radians)"])

    field_data.point_data["Ex-Real"] = np.array([np.real(ex)]).transpose()
    field_data.point_data["Ex-Imag"] = np.array([np.imag(ex)]).transpose()
    field_data.point_data["Ey-Real"] = np.array([np.real(ey)]).transpose()
    field_data.point_data["Ey-Imag"] = np.array([np.imag(ey)]).transpose()
    field_data.point_data["Ez-Real"] = np.array([np.real(ez)]).transpose()
    field_data.point_data["Ez-Imag"] = np.array([np.imag(ez)]).transpose()
    return field_data


def Exyz_to_EthetaEphi(field_data):
    # this function assumes a spherical field definition, will need to write a function which works based on the poynting vector/normal vector of the point
    if not all(
        k in field_data.point_data.keys() for k in ("theta_(Radians)", "phi_(Radians)")
    ):
        # theta and phi don't exist in the dataset
        from lyceanem.geometry.geometryfunctions import theta_phi_r

        field_data = theta_phi_r(field_data)

    electric_fields = extract_electric_fields(field_data)
    theta = field_data.point_data["theta_(Radians)"]
    phi = field_data.point_data["phi_(Radians)"]
    etheta = (
        electric_fields[:, 0] * np.cos(phi) * np.cos(theta)
        + electric_fields[:, 1] * np.sin(phi) * np.cos(theta)
        - electric_fields[:, 2] * np.sin(theta)
    )
    ephi = -electric_fields[:, 0] * np.sin(phi) + electric_fields[:, 1] * np.cos(phi)
    field_data.point_data["E(theta)-Real"] = np.zeros((electric_fields.shape[0]))
    field_data.point_data["E(phi)-Real"] = np.zeros((electric_fields.shape[0]))
    field_data.point_data["E(theta)-Imag"] = np.zeros((electric_fields.shape[0]))
    field_data.point_data["E(phi)-Imag"] = np.zeros((electric_fields.shape[0]))
    field_data.point_data["E(theta)-Real"] = np.real(etheta)
    field_data.point_data["E(theta)-Imag"] = np.imag(etheta)
    field_data.point_data["E(phi)-Real"] = np.real(ephi)
    field_data.point_data["E(phi)-Imag"] = np.imag(ephi)
    return field_data


def field_vectors(field_data):
    fields = np.array(
        [
            field_data.point_data["Ex-Real"] + 1j * field_data.point_data["Ex-Imag"],
            field_data.point_data["Ey-Real"] + 1j * field_data.point_data["Ey-Imag"],
            field_data.point_data["Ez-Real"] + 1j * field_data.point_data["Ez-Imag"],
        ]
    ).transpose()
    directions = np.abs(fields)
    return directions


def transform_em(field_data, r):
    """
    Transform the electric current vectors into the new coordinate system defined by the rotation matrix.

    Parameters
    ----------
    field_data: :class: meshio.Mesh
    r: scipy.rotation

    Returns
    -------

    """
    if all(
        k in field_data.point_data.keys() for k in ("Ex-Real", "Ey-Real", "Ez-Real")
    ):
        # print("Rotating Ex,Ey,Ez")
        fields = (
            np.array(
                [
                    field_data.point_data["Ex-Real"]
                    + 1j * field_data.point_data["Ex-Imag"],
                    field_data.point_data["Ey-Real"]
                    + 1j * field_data.point_data["Ey-Imag"],
                    field_data.point_data["Ez-Real"]
                    + 1j * field_data.point_data["Ez-Imag"],
                ]
            )
            .reshape(3, -1)
            .transpose()
        )
        rot_fields = r.apply(fields)
        field_data.point_data["Ex-Real"] = np.real(
            rot_fields[:, 0]
        )  # np.array([np.real(rot_fields[:,0]),np.imag(rot_fields[:,0])]).transpose()
        field_data.point_data["Ey-Real"] = np.real(rot_fields[:, 1])
        field_data.point_data["Ez-Real"] = np.real(rot_fields[:, 2])
        field_data.point_data["Ex-Imag"] = np.imag(
            rot_fields[:, 0]
        )  # np.array([np.real(rot_fields[:,0]),np.imag(rot_fields[:,0])]).transpose()
        field_data.point_data["Ey-Imag"] = np.imag(rot_fields[:, 1])
        field_data.point_data["Ez-Imag"] = np.imag(rot_fields[:, 2])
        # if all(k in field_data.point_data.keys() for k in ('E(theta)','E(phi)')):
    elif all(k in field_data.point_data.keys() for k in ("E(theta)", "E(phi)")):
        # print("Converting Fields and Rotating Ex,Ey,Ez")
        from lyceanem.geometry.geometryfunctions import theta_phi_r

        field_data = theta_phi_r(field_data)
        field_data = EthetaEphi_to_Exyz(field_data)
        fields = (
            np.array(
                [
                    field_data.point_data["Ex-Real"]
                    + 1j * field_data.point_data["Ex-Imag"],
                    field_data.point_data["Ey-Real"]
                    + 1j * field_data.point_data["Ey-Imag"],
                    field_data.point_data["Ez-Real"]
                    + 1j * field_data.point_data["Ez-Imag"],
                ]
            )
            .reshape(3, -1)
            .transpose()
        )
        rot_fields = r.apply(fields)
        field_data.point_data["Ex-Real"] = np.real(rot_fields[:, 0])
        field_data.point_data["Ey-Real"] = np.real(rot_fields[:, 1])
        field_data.point_data["Ez-Real"] = np.real(rot_fields[:, 2])
        field_data.point_data["Ex-Imag"] = np.imag(rot_fields[:, 0])
        field_data.point_data["Ey-Imag"] = np.imag(rot_fields[:, 1])
        field_data.point_data["Ez-Imag"] = np.imag(rot_fields[:, 2])
    return field_data


def update_electric_fields(field_data, ex, ey, ez):
    field_data.point_data["Ex-Real"] = np.zeros((field_data.points.shape[0], 1))
    field_data.point_data["Ey-Real"] = np.zeros((field_data.points.shape[0], 1))
    field_data.point_data["Ez-Real"] = np.zeros((field_data.points.shape[0], 1))
    field_data.point_data["Ex-Imag"] = np.zeros((field_data.points.shape[0], 1))
    field_data.point_data["Ey-Imag"] = np.zeros((field_data.points.shape[0], 1))
    field_data.point_data["Ez-Imag"] = np.zeros((field_data.points.shape[0], 1))
    field_data.point_data["Ex-Real"] = np.array([np.real(ex)]).transpose()
    field_data.point_data["Ex-Imag"] = np.array([np.imag(ex)]).transpose()
    field_data.point_data["Ey-Real"] = np.array([np.real(ey)]).transpose()
    field_data.point_data["Ey-Imag"] = np.array([np.imag(ey)]).transpose()
    field_data.point_data["Ez-Real"] = np.array([np.real(ez)]).transpose()
    field_data.point_data["Ez-Imag"] = np.array([np.imag(ez)]).transpose()
    return field_data


def PoyntingVector(field_data, measurement=False, aperture=None):
    """
    Calculate the poynting vector for the given field data. If the magnetic field data is present, then the poynting vector is calculated using the cross product of the electric and magnetic field vectors.
    If the magnetic field data is not present, then the poynting vector is calculated using the electric field vectors and the impedance of the material. If material parameters are not included in the field data, then the impedance is assumed to be that of free space.
    Measurement is a boolean value which indicates whether the field data represents measurements with a finite aperture, rather than point data. If measurement is True, then the aperture parameter must be provided, which is the area of the measurement aperture. This allows the power density at each measurement location to be calculated consistently with the point approach.
    field_data: meshio.Mesh

    measurement: bool

    aperture: float
        The area of the measurement aperture, if measurement is True, so the field data represents measurements with a finite aperture, rather than point data.

    """
    if all(
        k in field_data.point_data.keys()
        for k in (
            "Permittivity-Real",
            "Permittivity-Imag",
            "Permeability-Real",
            "Permeability-Imag",
            "Conductivity",
        )
    ):
        eta = (
            field_data.point["Permeability-Real"]
            + 1j
            * field_data.point["Permeability-Imag"]
            / field_data.point["Permittivity-Real"]
            + 1j * field_data.point["Permittivity-Imag"]
        ) ** 0.5
    else:
        from scipy.constants import physical_constants

        eta = (
            np.ones((field_data.point_data["Ex-Real"].shape[0]))
            * physical_constants["characteristic impedance of vacuum"][0]
        )

    electric_field_vectors = np.array(
        [
            field_data.point_data["Ex-Real"].ravel()
            + 1j * field_data.point_data["Ex-Imag"].ravel(),
            field_data.point_data["Ey-Real"].ravel()
            + 1j * field_data.point_data["Ey-Imag"].ravel(),
            field_data.point_data["Ez-Real"].ravel()
            + 1j * field_data.point_data["Ez-Imag"].ravel(),
        ]
    ).transpose()
    if all(
        k in field_data.point_data.keys() for k in ("Hx-Real", "Hy-Real", "Hz-Real")
    ):
        # magnetic field data present, so use
        magnetic_field_vectors = np.array(
            [
                field_data.point_data["Hx-Real"]
                + 1j * field_data.point_data["Hx-Imag"],
                field_data.point_data["Hy-Real"]
                + 1j * field_data.point_data["Hy-Imag"],
                field_data.point_data["Hz-Real"]
                + 1j * field_data.point_data["Hz-Imag"],
            ]
        ).transpose()
        # calculate poynting vector using electric and magnetic field vectors
        poynting_vector_complex = np.cross(
            electric_field_vectors, magnetic_field_vectors
        )
    else:
        # use normal vectors instead
        # poynting_vector_complex=field_data.point_data['Normals']*((np.linalg.norm(electric_field_vectors,axis=1)**2)/eta).reshape(-1,1)
        poynting_vector_complex = (
            (np.linalg.norm(electric_field_vectors, axis=1) ** 2) / eta
        ).reshape(-1, 1)

    if measurement:
        poynting_vector_complex = poynting_vector_complex / aperture

    field_data.point_data["Poynting_Vector_(Magnitude_(W/m2))"] = np.zeros(
        (field_data.points.shape[0], 1)
    )
    field_data.point_data["Poynting_Vector_(Magnitude_(dBW/m2))"] = np.zeros(
        (field_data.points.shape[0], 1)
    )
    field_data.point_data["Poynting_Vector_(Magnitude_(W/m2))"] = np.linalg.norm(
        poynting_vector_complex, axis=1
    ).reshape(-1, 1)
    field_data.point_data["Poynting_Vector_(Magnitude_(dBW/m2))"] = 10 * np.log10(
        np.linalg.norm(poynting_vector_complex.reshape(-1, 1), axis=1)
    )
    return field_data


def Directivity(field_data):
    # calculate directivity for the given pattern

    if not all(
        k in field_data.point_data.keys() for k in ("theta_(Radians)", "phi_(Radians)")
    ):
        # theta and phi don't exist in the dataset
        from lyceanem.geometry.geometryfunctions import theta_phi_r

        field_data = theta_phi_r(field_data)

    if not all(
        k in field_data.point_data.keys() for k in ("E(theta)-Real", "E(phi)-Real")
    ):
        # E(theta) and E(phi) are not present in the data
        field_data = Exyz_to_EthetaEphi(field_data)

    if not all(k in field_data.point_data.keys() for k in ("Area")):
        from lyceanem.geometry.geometryfunctions import compute_areas

        # E(theta) and E(phi) are not present in the data
        field_data = compute_areas(field_data)

    Utheta = np.abs(
        (
            field_data.point_data["E(theta)-Real"]
            + 1j * field_data.point_data["E(theta)-Imag"]
        )
        ** 2
    )
    Uphi = np.abs(
        (
            field_data.point_data["E(phi)-Real"]
            + 1j * field_data.point_data["E(phi)-Imag"]
        )
        ** 2
    )
    # Calculate Solid Angle
    solid_angle = (
        field_data.point_data["Area"]
        / field_data.point_data["Radial_Distance_(m)"] ** 2
    )
    Utotal = Utheta + Uphi

    Uav = np.nansum(Utotal.ravel() * solid_angle) / (4 * np.pi)
    # sin_factor = np.abs(
    #    np.sin(field_data.point_data["theta (Radians)"])
    # ).reshape(-1,1)  # only want positive factor
    # power_sum = np.sum(np.abs(Utheta * sin_factor)) + np.sum(np.abs(Uphi * sin_factor))
    # need to dynamically account for the average contribution of each point, this is only true for a theta step of 1 degree, and phi step of 10 degrees
    # Uav = (power_sum * (np.radians(1.0) * np.radians(10.0))) / (4 * np.pi)
    Dtheta = Utheta / Uav
    Dphi = Uphi / Uav
    Dtot = Utotal / Uav
    field_data.point_data["D(theta)"] = Dtheta
    field_data.point_data["D(phi)"] = Dphi
    field_data.point_data["D(Total)"] = Dtot

    return field_data




def oxygen_lines():
    data_lines = []
    oxy_data = str(files(data).joinpath("Oxy.txt"))
    with open(oxy_data, "r") as file:
        for line in file:
            if line.strip():
                values = [float(x) for x in line.split()]
                data_lines.append(values[:7])

    return data_lines


def water_vapour_lines():

    data_lines = []
    water_data = str(files(data).joinpath("Vapour.txt"))
    with open(water_data, "r") as file:
        for line in file:
            if line.strip():
                values = [float(x) for x in line.split()]
                data_lines.append(values[:7])

    return data_lines

def calculate_oxygen_attenuation(frequency, pressure, temperature, oxygen_lines):
    """
    Calculate the specific attenuation due to oxygen using the ITU-R P.676-11 model.

    Parameters:
    frequency (GHz): The frequency of the signal in GHz.
    pressure (hPa): The atmospheric pressure in hectopascals.
    temperature (C): The temperature in degrees Celsius.
    oxygen_lines (list): A list of spectroscopic data lines for oxygen.

    Returns:
    float: The calculated oxygen attenuation in dB/km.
    """
    temperature_k = temperature + 273.15
    theta = 300 / temperature_k
    specific_attenuation = 0

    for line in oxygen_lines:
        f_line, a1, a2, a3, a4, a5, a6 = line
        S = a1 * 10**-7 * pressure * theta**3 * math.exp(a2 * (1 - theta))
        ffo = a3 * 10**-4 * (pressure * theta ** (0.8 - a4) + 1.1 * pressure * theta)
        delta = (a5 + a6 * theta) * 10**-4 * (pressure) * theta**0.8
        F = (frequency / f_line) * (
            (ffo - delta * (f_line - frequency)) / ((f_line - frequency) ** 2 + ffo**2)
            + (ffo - delta * (f_line + frequency))
            / ((f_line + frequency) ** 2 + ffo**2)
        )
        specific_attenuation += (frequency / f_line) * S * F

    return specific_attenuation


def calculate_water_vapor_attenuation(
    frequency, pressure, temperature, water_vapor_lines
):
    """
    Calculate the specific attenuation due to water vapor using the ITU-R P.676-11 model.

    Parameters:
    frequency (GHz): The frequency of the signal in GHz.
    pressure (hPa): The atmospheric pressure in hectopascals.
    temperature (C): The temperature in degrees Celsius.
    water_vapor_lines (list): A list of spectroscopic data lines for water vapor.

    Returns:
    float: The calculated water vapor attenuation in dB/km.
    """
    temperature_k = temperature + 273.15
    theta = 300 / temperature_k
    e = pressure * 0.622 / (0.622 + 0.378)  # Partial pressure of water vapor (hPa)
    specific_attenuation = 0

    for line in water_vapor_lines:
        f_line, a1, a2, a3, a4, a5, a6 = line
        S = a1 * 10**-1 * e * theta**3.5 * math.exp(a2 * (1 - theta))
        ffo = a3 * 10**-4 * (pressure * theta**a4 + a5 * e * theta**a6)
        F = (frequency / f_line) * (
            (ffo - 0 * (f_line - frequency)) / ((f_line - frequency) ** 2 + ffo**2)
            + (ffo - 0 * (f_line + frequency)) / ((f_line + frequency) ** 2 + ffo**2)
        )
        specific_attenuation += (frequency / f_line) * S * F

    return specific_attenuation


def calculate_total_gaseous_attenuation(
    frequency,
    pressure,
    temperature,
    oxygen_lines=oxygen_lines(),
    water_vapor_lines=water_vapour_lines(),
):
    """
    Calculate the total gaseous attenuation due to both oxygen and water vapor.

    Parameters:
    --------------
    frequency (GHz): float
        The frequency of the signal in GHz.
    pressure (hPa): float
        The atmospheric pressure in hectopascals.
    temperature (C): float
     The temperature in degrees Celsius.


    Returns:
    -----------
    float: The calculated total gaseous attenuation in Np/m.
    """
    # Calculate specific attenuation
    oxygen_attenuation = calculate_oxygen_attenuation(
        frequency, pressure, temperature, oxygen_lines
    )
    water_vapor_attenuation = calculate_water_vapor_attenuation(
        frequency, pressure, temperature, water_vapor_lines
    )
    specific_attenuation = (
        0.1820 * frequency * (oxygen_attenuation + water_vapor_attenuation)
    )
    specific_attenuation = specific_attenuation / (8.686 * 1000)

    return specific_attenuation


def calculate_phase_constant(frequency, temperature, pressure, water_vapor_density):
    """
    Calculate the phase constant as a function of frequency (GHz), temperature (Celsius), atmospheric pressure (hectoPascals) and water vapour density (g/m^3).

    Parameters
    ----------
    frequency : float
        Frequency in GHz
    temperature : float
        Temperature in Celsius
    pressure : float
        Atmospheric pressure in hectoPascals
    water_vapor_density : float
        Water Vapour Density in g/m^3

    Returns
    -------
    phase_constant : float
        Phase constant in radians/m

    """
    # Constants
    from scipy.constants import speed_of_light

    # c = 3e8  # Speed of light in vacuum (m/s)
    T0 = 273.15  # Standard temperature in Kelvin
    e_s0 = 611  # Saturation vapor pressure at T0 in Pa
    Lv = 2.5e6  # Latent heat of vaporization of water in J/kg
    Rv = 461.5  # Specific gas constant for water vapor in J/(kg·K)

    # Convert temperature to Kelvin
    temperature_K = temperature + T0

    # Saturation vapor pressure at given temperature
    e_s = e_s0 * math.exp((Lv / Rv) * ((1 / T0) - (1 / temperature_K)))

    # Actual vapor pressure
    e = water_vapor_density * e_s

    # Pressure in Pa
    P = pressure * 100  # Convert hPa to Pa

    # Refractivity N(h)
    N = 77.6 * (P / temperature_K) + (3.73e5 * e) / (temperature_K**2)

    # Refractive index n
    n = 1 + N * 1e-6

    # Phase constant beta
    beta = (2 * math.pi * frequency * 1e9 * n) / speed_of_light

    return beta


def calculate_atmospheric_propagation_constant(
    frequency, temperature, pressure, water_vapor_density
):
    """
    Calculate the propagation constant as a function of frequency (GHz), temperature (Celsius), atmospheric pressure (hectoPascals) and water vapour density (g/m^3).

    Parameters
    ----------
    frequency : float
        Frequency in GHz
    temperature : float
        Temperature in Celsius
    pressure : float
        Atmospheric pressure in hectoPascals
    water_vapor_density : float
        Water Vapour Density in g/m^3

    Returns
    -------
    propagation constant : complex

    """
    alpha = calculate_total_gaseous_attenuation(
        frequency, temperature, pressure, water_vapor_density
    )
    beta = calculate_phase_constant(
        frequency, temperature, pressure, water_vapor_density
    )
    gamma = alpha + 1j * beta
    return gamma