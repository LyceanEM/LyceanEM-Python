.. _tutorials:
Tutorials
============

This collection of tutorials is intended to give an introduction to the different models included with LyceanEM, and some of the different use cases.

Aperture Projection
--------------------
Aperture Projection as a technique is based upon Hannan's formulation of the gain of an aperture based upon its surface area and the freuqency of interest.
This is defined in terms of the maximum gain :math:`G_{max}`, the effective area of the aperture :math:`A_{e}`, and the wavelength of interest :math:`\lambda`.

.. math::
    G_{max}=\dfrac{4 \pi A_{e}}{\lambda^{2}}

While this has been in common use since the 70s, as a formula it is limited to planar surfaces, and only providing the maximum gain in the boresight direction for that surface.

Aperture projection as a function is based upon the rectilinear projection of the aperture into the farfield. This can then be used with Hannan's formula to predict the maximum achievable directivity for all farfield directions of interest :footcite:p:`Pelham2017`.

As this method is built into a raytracing environment, the maximum performance for an aperture on any platform can also be predicted using the :func:`lyceanem.models.frequency_domain.aperture_projection` function.






.. footbibliography::

