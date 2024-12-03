import numpy as np
import pyvista as pv


def fresnel_zone(pointA, pointB, wavelength, zone=1):
    # based on the provided points, wavelength, and zone number, calculate the fresnel zone. This is defined as an ellipsoid for which the difference in distance between the line AB (line of sight), and AP+PB (reflected wave) is a constant multiple of ($n\dfrac{\lambda}{2}$).
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
    pose[:3, 3] = center
    ellipsoid = pv.ParametricEllipsoid(
        separation * 0.5, fresnel_radius, fresnel_radius
    ).transform(pose)
    return ellipsoid


def field_magnitude_phase(field_data):
    # calculate magnitude and phase representation

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
        Ex = (
            field_data.point_data["Ex-Real"] + 1j * field_data.point_data["Ex-Imag"]
        )
        Ey = (
            field_data.point_data["Ey-Real"] + 1j * field_data.point_data["Ey-Imag"]
        )
        Ez = (
            field_data.point_data["Ez-Real"] + 1j * field_data.point_data["Ez-Imag"]
        )
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
            field_data.point_data["Ex-Real"]
            + 1j * field_data.point_data["Ex-Imag"],
            field_data.point_data["Ey-Real"]
            + 1j * field_data.point_data["Ey-Imag"],
            field_data.point_data["Ez-Real"]
            + 1j * field_data.point_data["Ez-Imag"],
        ]
    ).transpose()
    directions = np.abs(fields)
    return directions


def transform_em(field_data, r):
    """
    Transform the electric current vectors into the new coordinate system defined by the rotation matrix.

    Parameters
    ----------
    field_data: meshio.Mesh
    r: scipy.rotation

    Returns
    -------

    """
    if all(
        k in field_data.point_data.keys()
        for k in ("Ex-Real", "Ey-Real", "Ez-Real")
    ):
        # print("Rotating Ex,Ey,Ez")
        fields = np.array(
            [
                field_data.point_data["Ex-Real"]
                + 1j * field_data.point_data["Ex-Imag"],
                field_data.point_data["Ey-Real"]
                + 1j * field_data.point_data["Ey-Imag"],
                field_data.point_data["Ez-Real"]
                + 1j * field_data.point_data["Ez-Imag"],
            ]
        ).reshape(3,-1).transpose()
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
        fields = np.array(
            [
                field_data.point_data["Ex-Real"]
                + 1j * field_data.point_data["Ex-Imag"],
                field_data.point_data["Ey-Real"]
                + 1j * field_data.point_data["Ey-Imag"],
                field_data.point_data["Ez-Real"]
                + 1j * field_data.point_data["Ez-Imag"],
            ]
        ).reshape(3,-1).transpose()
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


def PoyntingVector(field_data,measurement=False,aperture=None):
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
        k in field_data.point_data.keys()
        for k in ("Hx-Real", "Hy-Real", "Hz-Real")
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
        poynting_vector_complex = poynting_vector_complex/aperture

    field_data.point_data["Poynting_Vector_(Magnitude)"] = np.zeros(
        (field_data.points.shape[0], 1)
    )
    field_data.point_data["Poynting_Vector_(Magnitude_(dB))"] = np.zeros(
        (field_data.points.shape[0], 1)
    )
    field_data.point_data["Poynting_Vector_(Magnitude)"] = np.linalg.norm(
        poynting_vector_complex, axis=1
    ).reshape(-1, 1)
    field_data.point_data["Poynting_Vector_(Magnitude_(dB))"] = 10 * np.log10(
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

    if not all(
            k in field_data.point_data.keys() for k in ("Area")
        ):
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
    #Calculate Solid Angle
    solid_angle=field_data.point_data["Area"]/field_data.point_data["Radial_Distance_(m)"]**2
    Utotal = Utheta + Uphi

    Uav=(np.nansum(Utotal.ravel()*solid_angle)/(4*np.pi))
    #sin_factor = np.abs(
    #    np.sin(field_data.point_data["theta (Radians)"])
    #).reshape(-1,1)  # only want positive factor
    #power_sum = np.sum(np.abs(Utheta * sin_factor)) + np.sum(np.abs(Uphi * sin_factor))
    # need to dynamically account for the average contribution of each point, this is only true for a theta step of 1 degree, and phi step of 10 degrees
    #Uav = (power_sum * (np.radians(1.0) * np.radians(10.0))) / (4 * np.pi)
    Dtheta = Utheta / Uav
    Dphi = Uphi / Uav
    Dtot = Utotal / Uav
    field_data.point_data["D(theta)"] = Dtheta
    field_data.point_data["D(phi)"] = Dphi
    field_data.point_data["D(Total)"] = Dtot

    return field_data
