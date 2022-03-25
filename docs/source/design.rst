.. _design:
Design
=======

LyceanEM is the development of initial PhD research into conformal antenna array design :footcite:p:`Pelham2021a`.

The intention is the LyceanEM can be used both in a systems design context to allow the user to evaluate different antenna array on platform configurations for performance against system requirements, while also allowing more fundamental electromangetics scattering research.

There are two fundamentally different approaches considered. The first is that of antenna apertures on platforms, buildings, airplanes, cars, and boats. In this case the interest is in the effects of location and arrangement on the farfield, and LyceanEM allows for rapid assesments of the possible using `aperture projection`. This can then be built into more careful performance modelling of the effects of array polarisation, beamforming architecture, and beamforming algorithms on overall performance using the `calculate farfield` function and others. The second approach is concerned with the effects of scattering in a complex environment, predicting the channel for communications or sensors. In this case the `calculate scattering` functions in both frequency and time domain come into use.

If LyceanEM has been useful to your research and you would like to acknowledge the projection in your publication,
we would like to encourage you to cite the following paper :footcite:p:`Pelham2021a`.

The base class of LyceanEM is that of the `antenna pattern'. This class provides a useful container for individual antenna patterns, with builtin functions to generate simple patterns, import, rotate, and display them.

.. toctree::
    base

.. footbibliography::
