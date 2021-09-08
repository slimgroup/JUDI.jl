#!/bin/bash

# Read in the file of environment settings
source /opt/intel/oneapi/setvars.sh intel64

# gu cleanup and path
find /app -type f -name '*.pyc' -delete

export PATH=/venv/bin:$PATH
export PYTHONPATH=$PYTHONPATH:/app

# Then run the CMD
exec "$@"
