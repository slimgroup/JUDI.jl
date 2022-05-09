#!/bin/bash

JVER=$1
JVERMIN=$(wget -O - -o /dev/null https://raw.githubusercontent.com/docker-library/julia/master/versions.json | jq -r --arg JVER "${JVER}" '.[$JVER].version')
wget "https://julialang-s3.julialang.org/bin/linux/x64/${JVER}/julia-${JVERMIN}-linux-x86_64.tar.gz"
tar -xvzf julia-${JVERMIN}-linux-x86_64.tar.gz
rm -rf julia-${JVERMIN}-linux-x86_64.tar.gz
ln -s /julia-${JVERMIN}/bin/julia /usr/local/bin/julia