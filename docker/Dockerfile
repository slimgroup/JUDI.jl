FROM python:3.6

# Install devito
RUN pip3 install --upgrade pip
RUN pip3 install --user git+https://github.com/devitocodes/devito.git@v4.2

# Devito requirements
RUN pip3 install --user jupyter

# Compiler and Devito environment variables
ENV DEVITO_ARCH=gcc
ENV DEVITO_LANGUAGE=openmp
ENV OMP_NUM_THREADS=2
ENV PYTHONPATH=/app/devito

# Required packages
RUN apt-get update && \
	apt-get install -y gfortran && \
	apt-get install -y hdf5-tools

# Install Julia
WORKDIR /julia
RUN wget "https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.1-linux-x86_64.tar.gz" && \
	tar -xvzf julia-1.4.1-linux-x86_64.tar.gz && \
	rm -rf julia-1.4.1-linux-x86_64.tar.gz && \
	ln -s /julia/julia-1.4.1/bin/julia /usr/local/bin/julia

# Manually install unregistered packages and JUDI
RUN julia -e 'using Pkg; Pkg.develop(PackageSpec(url="https://github.com/slimgroup/SegyIO.jl"))' && \
	julia -e 'using Pkg; Pkg.develop(PackageSpec(url="https://github.com/slimgroup/JOLI.jl"))' && \
	julia -e 'using Pkg; Pkg.develop(PackageSpec(url="https://github.com/slimgroup/JUDI.jl"))'

RUN	julia -e 'using Pkg; Pkg.add("NLopt")'

# Install and build IJulia
ENV JUPYTER="/root/.local/bin/jupyter"
RUN julia -e 'using Pkg; Pkg.add("IJulia")'

# Add JUDI examples
ADD ./data /app/judi/data
ADD ./test /app/judi/test
ADD ./examples /app/judi/examples

WORKDIR /app/judi/notebooks

EXPOSE 8888

CMD /root/.local/bin/jupyter-notebook --ip="*" --no-browser --allow-root
