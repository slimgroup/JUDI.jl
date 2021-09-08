#!/bin/bash
# Read in the file of environment settings
source /opt/intel/oneapi/setvars.sh intel64
# Then run the CMD
exec "$@"
