FROM ubuntu:20.04

# Install python
ENV DEBIAN_FRONTEND=noninteractive 
# Required packages
RUN apt-get update && \
    apt-get install -y gfortran && \
    apt-get install -y git wget vim htop hdf5-tools

RUN apt-get install -y python3 python3-dev python3-pip

# Install devito
RUN pip3 install --upgrade pip
RUN pip3 install --user devito

# Devito requirements
RUN pip3 install --user jupyter matplotlib

# Compiler and Devito environment variables
ENV DEVITO_ARCH=gcc
ENV DEVITO_LANGUAGE=openmp
ENV OMP_NUM_THREADS=2
ENV PYTHONPATH=/app/devito

# Install Julia
WORKDIR /julia
RUN wget --no-check-certificate "https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.1-linux-x86_64.tar.gz" && \
	tar -xvzf julia-1.5.1-linux-x86_64.tar.gz && \
	rm -rf julia-1.5.1-linux-x86_64.tar.gz && \
	ln -s /julia/julia-1.5.1/bin/julia /usr/local/bin/julia

# Manually install unregistered packages and JUDI
RUN julia -e 'using Pkg;Pkg.update()'
RUN julia -e 'using Pkg;Pkg.Registry.add(RegistrySpec(url="https://Github.com/slimgroup/SLIMregistryJL.git"))'

RUN for pkg in "SegyIO" "JOLI" "JUDI" "SlimOptim" "InvertibleNetworks" "JUDI4Flux" "JOLI4Flux"; do julia -e "using Pkg;Pkg.develop(\"$pkg\");using $pkg"; done
RUN for pkg in "LineSearches" "Optim" "DSP" "Dierckx" "NLopt" "FFTW" "SetIntersectionProjection" "IterativeSolvers" "JLD" "Flux" "Zygote" "JLD2" "HDF5" "PyPlot" "PyCall" "Distributed" "Images"; do julia -e "using Pkg;Pkg.add(\"$pkg\"); using $pkg"; done

RUN PYTHON=$(which python3) julia -e 'using Pkg; Pkg.build("PyCall")'

# Install and build IJulia
ENV JUPYTER="/root/.local/bin/jupyter"
RUN julia -e 'using Pkg; Pkg.add("IJulia")'

RUN apt clean all

WORKDIR /app/judi/notebooks

EXPOSE 8888

CMD /root/.local/bin/jupyter-notebook --ip="*" --no-browser --allow-root
