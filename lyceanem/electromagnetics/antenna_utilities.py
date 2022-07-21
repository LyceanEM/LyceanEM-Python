import numpy as np
import open3d as o3d


def antenna_stats(freq, x, y, z):
    """
    Utility for evaluating electrical size

    Parameters
    ----------
    freq : float
        frequency of the antenna in Hz
    x : float
        largest dimension in x, or half width in x direction (m)
    y : float
        largest dimension in y, or half width in y direction (m)
    z : float
        largest dimension in z, or half width in z direction (m)

    Returns
    -------

    """
    wavelength = 3e8 / freq
    k = (2 * np.pi) / wavelength
    a = np.sqrt((x) ** 2 + (y) ** 2 + (z) ** 2) * 0.5

    print("ka={:1.5f}".format(k * a))
    print("Radius in Wavelengths={:1.5f}".format(a / wavelength))


def antenna_size(freq, antenna):
    """
    Utility for calculating the electrical size of a given antenna structure

    Parameters
    ----------
    freq : float
        frequency of the antenna in Hz
    antenna_structure :
        antenna structure

    Returns
    -------
    ka : float
        electrical size of antenna structure for given frequency
    """
    total_points = antenna.export_all_points()
    # calculate bounding box
    bounding_box = total_points.get_oriented_bounding_box()
    max_points = bounding_box.get_max_bound()
    min_points = bounding_box.get_min_bound()
    center = bounding_box.get_center()
    max_dist = np.sqrt(
        (max_points[0] - center[0]) ** 2
        + (max_points[1] - center[1]) ** 2
        + (max_points[2] - center[2]) ** 2
    )
    min_dist = np.sqrt(
        (center[0] - min_points[0]) ** 2
        + (center[1] - min_points[1]) ** 2
        + (center[2] - min_points[2]) ** 2
    )
    a = np.mean([max_dist, min_dist])
    wavelength = 3e8 / freq
    k = (2 * np.pi) / wavelength
    ka = k * a
    return ka
