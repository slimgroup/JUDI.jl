#!/usr/bin/env bash

# Install devito
pip install --upgrade pip
pip install --user git+https://github.com/opesci/devito.git@v3.2.0
pip install --user -r docker/devito_requirements.txt

# Set devito enviroment variables
export DEVITO_ARCH="gcc"
export DEVITO_OPENMP="0"

# Point PyCall to correct Python version
export PYTHON=$(which python)
julia -e 'Pkg.build("PyCall")'


