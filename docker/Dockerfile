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
RUN wget "https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.2-linux-x86_64.tar.gz" && \
	tar -xvzf julia-1.0.2-linux-x86_64.tar.gz && \
	rm -rf julia-1.0.2-linux-x86_64.tar.gz && \
	ln -s /julia/julia-1.0.2/bin/julia /usr/local/bin/julia

# Manually install unregistered packages and JUDI
RUN julia -e 'using Pkg; Pkg.clone("https://github.com/slimgroup/SeisIO.jl")' && \
	julia -e 'using Pkg; Pkg.clone("https://github.com/slimgroup/JOLI.jl.git")' && \
	julia -e 'using Pkg; Pkg.clone("https://github.com/slimgroup/JUDI.jl.git")'

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

