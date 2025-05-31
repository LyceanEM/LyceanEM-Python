#!/bin/bash

# Use Ninja explicitly (optional but recommended for consistency)
export CMAKE_GENERATOR="Ninja"

# Install the package using pip with scikit-build
"${PYTHON}" -m pip install . --no-deps --no-build-isolation -vv