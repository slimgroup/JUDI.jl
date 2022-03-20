
# Installation

JUDI is a linear algebra abstraction built on top of [Devito](https://github.com/devitocodes/devito). Because [Devito](https://github.com/devitocodes/devito) is a just-in-time compiler, you will need to have a standard C compiler installed. by default most system come with a gcc compiler (except Windows where we recommend to use docker or WSL) which unfortunately isnt' very reliable. It is therefore recommended to install a proper compiler (gcc>=7, icc). For GPU offloading, you will then need to install a proper offloading compiler such as Nvidia's nvc or the latest version of clang (*not Apple clang*).

## Standard installation

JUDI is registered and can be installed directly in julia REPL

```julia
] add JUDI
```

This will install JUDI, and the `build` will install the necessary dependencies including [Devito](https://github.com/devitocodes/devito).

## Custom installation

In some case you may want to have your own installation of Devtio you want JUDI to use in which case you should foloow these steps.

First, install Devito using `pip`, or see the [Devito's GitHub page](https://github.com/devitocodes/devito) for installation with Conda and further information. The current release of JUDI requires Python 3 and the current Devito version. Run all of the following commands from the (bash) terminal command line (not in the Julia REPL):

```bash
pip3 install --user git+https://github.com/devitocodes/devito.git
```

Once Devito, SegyIO and JOLI are installed, you can install JUDI as follows:

```julia
] add JUDI
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
