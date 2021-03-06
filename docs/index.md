# The Julia Devito Inversion framework (JUDI.jl)

JUDI is a framework for large-scale seismic modeling and inversion and designed to enable rapid translations of algorithms to fast and efficient code that scales to industry-size 3D problems. Wave equations in JUDI are solved with [Devito](https://www.devitoproject.org/), a Python domain-specific language for automated finite-difference (FD) computations. 

## Docs overview

This documentation provides an overview over JUDI's basic data structures and abstract operators:

 * **Tutorials**: Shows basic functionalities and some common applications.

 * **Data structures**: Explains the `Model`, `Geometry` and `Info` data structures and how to set up acquisition geometries.

 * **Abstract vectors**: Documents JUDI's abstract vector classes `judiVector`, `judiWavefield`, `judiRHS`, `judiWeights` and `judiExtendedSource`.

 * **Abstract operators**: Lists and explains JUDI's abstract linear operators `judiModeling`, `judiJacobian`, `judiProjection` and `judiLRWF`.

 * **Input/Output**: Read SEG-Y data and set up `judiVectors` for shot records and sources. Read velocity models.

 * **Helper functions**: API of functions that make your life easier.

 * **Preconditioners**: Basic preconditioners for seismic imaging.

## Installation

First, install Devito using `pip`, or see the [Devito's GitHub page](https://github.com/devitocodes/devito) for installation with Conda and further information. The current release of JUDI requires Python 3 and the current Devito version. Run all of the following commands from the (bash) terminal command line (not in the Julia REPL):

```julia
pip3 install --user git+https://github.com/devitocodes/devito.git
```

For reading and writing seismic SEG-Y data, JUDI uses the [SegyIO](https://github.com/slimgroup/SegyIO.jl) package and matrix-free linear operators are based the [Julia Operator LIbrary](https://github.com/slimgroup/JOLI.jl/tree/master/src) (JOLI):

```julia
julia -e 'using Pkg; Pkg.develop(PackageSpec(url="https://github.com/slimgroup/SegyIO.jl"))'
julia -e 'using Pkg; Pkg.develop(PackageSpec(url="https://github.com/slimgroup/JOLI.jl"))'
```

Once Devito, SegyIO and JOLI are installed, you can install JUDI as follows:

```julia
julia -e 'using Pkg; Pkg.develop(PackageSpec(url="https://github.com/slimgroup/JUDI.jl"))'
```

Once you have JUDI installed, you need to point Julia's PyCall package to the Python version for which we previsouly installed Devito. To do this, copy-paste the following commands into the (bash) terminal:

```julia
export PYTHON=$(which python3)
julia -e 'using Pkg; Pkg.build("PyCall")'
```

## Running with Docker

If you do not want to install JUDI, you can run JUDI as a docker image. The first possibility is to run the docker container as a Jupyter notebook:

```
docker run -p 8888:8888 philippwitte/judi:v1.3
```

This command downloads the image and launches a container. You will see a link that you can copy-past to your browser to access the notebooks. Alternatively, you can run a bash session, in which you can start a regular interactive Julia session and run the example scripts. Download/start the container as a bash session with:

```
docker run -it philippwitte/judi:v1.3 /bin/bash
```

Inside the container, all examples are located in the directory `/app/judi/examples/scripts`.

## Configure compiler and OpenMP

Devito uses just-in-time compilation for the underlying wave equation solves. The default compiler is intel, but can be changed to any other specified compiler such as `gnu`. Either run the following command from the command line or add it to your ~/.bashrc file:

```
export DEVITO_ARCH=gnu
```

Devito uses shared memory OpenMP parallelism for solving PDEs. OpenMP is disabled by default, but you can enable OpenMP and define the number of threads (per PDE solve) as follows:

```
export DEVITO_LANGUAGE=openmp  # Enable OpenMP. 
export OMP_NUM_THREADS=4    # Number of OpenMP threads
```


## Troubleshooting

For troubleshooting please raise an issue on the [JUDI github page](https://github.com/slimgroup/JUDI.jl) or contact Philipp Witte at `pwitte3@gatech.edu` or Mathias Louboutin at `mlouboutin3@gatech.edu`