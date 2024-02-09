#!/usr/bin/env bash

# Install devito
pip install --upgrade pip
pip install -U --user "devito[tests,extras]"

# Set devito enviroment variables
export DEVITO_ARCH="gcc"
export DEVITO_LANGUAGE="openmp"

# Point PyCall to correct Python version
PYTHON=$(which python) julia -e 'using Pkg; Pkg.build("PyCall")'


