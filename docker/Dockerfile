FROM python:3.6

# Install devito
RUN pip install --upgrade pip
RUN pip install --user git+https://github.com/opesci/devito.git@v3.2.0

# Devito requirements
ADD docker/devito_requirements.txt /app/devito_requirements.txt
RUN pip install --user -r /app/devito_requirements.txt && \
	pip install --user jupyter

# Compiler and Devito environment variables
ENV DEVITO_ARCH="gcc"
ENV DEVITO_OPENMP="0"
ENV PYTHONPATH=/app/devito

# Required packages
RUN apt-get update && \
	apt-get install -y gfortran && \
	apt-get install -y hdf5-tools

# Install Julia
WORKDIR /julia
RUN wget "https://julialang-s3.julialang.org/bin/linux/x64/0.6/julia-0.6.2-linux-x86_64.tar.gz" && \
	tar -xvzf julia-0.6.2-linux-x86_64.tar.gz && \
	rm -rf julia-0.6.2-linux-x86_64.tar.gz && \
	ln -s /julia/julia-d386e40c17/bin/julia /usr/local/bin/julia

# Install JUDI requirements
ADD ./REQUIRE /app/judi/REQUIRE
RUN julia -e "Pkg.resolve()"

# Manually install unregistered packages and JUDI
RUN julia -e 'Pkg.clone("https://github.com/slimgroup/SeisIO.jl")' && \
	julia -e 'Pkg.clone("https://github.com/slimgroup/JOLI.jl.git")' && \
	julia -e 'Pkg.clone("https://github.com/floswald/ApproXD.jl.git")' && \
	julia -e 'Pkg.clone("https://github.com/slimgroup/JUDI.jl.git")'

# Additional packages, which are not in requirements but used in notebooks
RUN julia -e 'Pkg.add("PyPlot")' && \
	julia -e 'Pkg.add("NLopt")' && \
	julia -e 'Pkg.add("HDF5")' && \
	julia -e 'Pkg.add("JLD")'

# Install and build IJulia
ENV JUPYTER="/root/.local/bin/jupyter"
RUN julia -e 'Pkg.add("IJulia")'

# Add JUDI examples
ADD ./data /app/judi/data
ADD ./examples /app/judi/examples
ADD ./test /app/judi/test

WORKDIR /app/judi/examples/notebook

EXPOSE 8888

CMD /root/.local/bin/jupyter-notebook --ip="*" --no-browser --allow-root

