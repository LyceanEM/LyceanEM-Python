import numpy as np

def antenna_stats(freq, x, y, z):
    """
    Assume Frequency is in Hz, and xyz are in m, calculate the electrical size
    in ka format and radius in wavelengths
    """
    wavelength = 3e8 / freq
    k = (2 * np.pi) / wavelength
    a = np.sqrt((x) ** 2 + (y) ** 2 + (z) ** 2) * 0.5

    print("ka={:1.5f}".format(k * a))
    print("Radius in Wavelengths={:1.5f}".format(a / wavelength))