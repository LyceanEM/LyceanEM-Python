import numpy as np


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
