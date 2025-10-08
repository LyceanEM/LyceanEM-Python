.. _contributing:
Contributing
===============
The `LyceanEM` project welcomes interested contributors. This isn't limited to programming, and additional examples, documention, or demonstrations are welcomed.

If you are unsure where to start or how you can contribute reach out on the `GitHub <https://github.com/LyceanEM/LyceanEM-Python>`_ by opening an issue or commenting on an issue you are interested in.

All accepted contributions must be accepted under the :ref:`GNU 3 license<copyright>`.

Development Roadmap
---------------------
`LyceanEM` is electromagnetics simulation software that is used by researchers and engineers in a variety of fields. The software is currently under development, and the developers have outlined a roadmap for future changes. The roadmap includes three key areas:

* Computational Efficiency and Scalability: The inclusion of the new CUDA propagation engine has significantly improved the computational efficiency of `LyceanEM`, future development will focus on the inclusion of chunking algorithms that will allow `LyceanEM` models to be distributed across multiple GPUs, enabling the simulation of larger and more complex models. New modelling approaches will also be developed to take advantage of the improved computational speed of tensor-cores.

* Time Domain Model: The time domain model will be developed to include it's own variant of the CUDA propagation engine, together with support for dynamic environments allowing the simulation of doppler effects.



Here are some specific ways that users can contribute to the development of `LyceanEM`:

* Report bugs: If you find a bug in `LyceanEM`, please report it to the developers so that they can fix it.
* Submit patches: If you know how to fix a bug or add a new feature, please submit a pull request to the developers.
* Donate: If you would like to support the development of `LyceanEM`, you can make a donation to the developers.

Your contributions will help to make `LyceanEM` the best possible electromagnetics simulation software for a wide range of users. Thank you for your support!

Documentation
--------------
`NumPy Doc` style documentation is used for `LyceanEM`, generated using `Sphinx`.