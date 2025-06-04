.. _design:
Design
=======

LyceanEM is the development of initial PhD research into conformal antenna array design :footcite:p:`Pelhamb2021`.

The intention is the LyceanEM can be used both in a systems design context to allow the user to evaluate different antenna array on platform configurations for performance against system requirements, while also allowing more fundamental electromangetics scattering research.

There are two fundamentally different approaches considered. The first is that of antenna apertures on platforms, buildings, airplanes, cars, and boats. In this case the interest is in the effects of location and arrangement on the farfield, and LyceanEM allows for rapid assesments of the possible using `aperture projection`. This can then be built into more careful performance modelling of the effects of array polarisation, beamforming architecture, and beamforming algorithms on overall performance using the `calculate farfield` function and others. The second approach is concerned with the effects of scattering in a complex environment, predicting the channel for communications or sensors. In this case the `calculate scattering` functions in both frequency and time domain come into use.

If LyceanEM has been useful to your research and you would like to acknowledge the projection in your publication,
we would like to encourage you to cite the following papers `LyceanEM: a python package for virtual prototyping of antenna arrays, time and frequency domain channel modelling` :footcite:p:`Pelham2023`, and `A scalable open-source electromagnetics model for wireless power transfer and space-based solar power` :footcite:p:`Pelhama2025`. The support of the Net Zero Innovation Portfolio has enabled the inclusion of a new CUDA propagation engine, lossy propagation models, propagation constant calculators to support the new engine using ITU recommended approaches for predicting the propagation constant of the atmosphere under different conditions, and the inclusion of gmsh and meshio as the new geometry engine for LyceanEM. This allows not only more accurate, faster simulations, but the ability to import, meshing and export of a wide range of CAD and mesh formats for inclusion in LyceanEM simulations.

The version 0.1.0 release marks the completion of the first stage of the development roadmap.
* Computational efficiency and scalability: The computational efficiency of `LyceanEM` has been improved, and it can now be used on a wider range of hardware platforms, including desktop computers and high-performance computing (HPC) clusters. This will make `LyceanEM` more accessible to a wider range of users. This has been developed with the support of the Net Zero Innovation Portfolio to support antenna arrays with multiple billon antenna elements, `Giga-scale` antenna arrays suitable for space-based solar power and other applications.
* Core propagation engine: The core propagation engine of `LyceanEM` has been upgraded to include more realistic models of lossy propagation and atmospheric effects for each layer of the atmosphere. This will make `LyceanEM` more accurate and versatile for a wider range of applications.
* Modelling fidelity: New features and functions have been added to `LyceanEM` that will allow users to model electromagnetic systems with greater fidelity. This includes support for importing antenna patterns and time domain sources, as well as the development of open source standards for antenna array designs, antenna patterns and field sources, and wireless power transfer.

The basic class of LyceanEM for farfield patterns is the `antenna pattern`. This class provides a useful container for individual antenna patterns, with builtin functions to generate simple patterns, import, rotate, and display them.

In order to handle solids in a consistent manner for raycasting, LyceanEM implements the class `structures` as a way to bundle multiple trianglemesh solids that are part of a whole. This enables not just packaging for the raycaster, but also further development for multimaterial modelling.


.. footbibliography::
