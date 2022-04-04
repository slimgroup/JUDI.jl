#!/bin/bash

# Check if we have a gpu available
# Or defaults to cpu (gcc)
if [[ $(type -p nvidia-smi) ]]; then
  export DEVITO_ARCH="nvc"
  export DEVITO_PLATFORM="nvidiaX"
  export DEVITO_LANGUAGE="openacc"
fi

# gu cleanup and path
find /app -type f -name '*.pyc' -delete

# Then run the CMD
exec "$@"
