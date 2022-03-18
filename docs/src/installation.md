
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

You can find installation instruction in our Wiki at [Installation](https://github.com/slimgroup/JUDI.jl/wiki/Installation).

JUDI is a registered package and can therefore be easily installed from the General registry with `]add/dev JUDI`

## GPU

JUDI supports the computation of the wave equation on GPU via [Devito](https://www.devitoproject.org)'s GPU offloading support.

**NOTE**: Only the wave equation part will be computed on GPU, the julia arrays will still be CPU arrays and `CUDA.jl` is not supported.

### Installation

To enable gpu support in JUDI, you will need to install one of [Devito](https://www.devitoproject.org)'s supported offloading compilers. We strongly recommend checking the [Wiki](https://github.com/devitocodes/devito/wiki) for installation steps and to reach out to the Devito community for GPU compiler related issues.

- [x] `nvc/pgcc`. This is recommended and the simplest installation. You can install the compiler following Nvidia's installation instruction at [HPC-sdk](https://developer.nvidia.com/hpc-sdk)
- [ ] `aompcc`. This is the AMD compiler that is necessary for running on AMD GPUs. This installation is not tested with JUDI and we recommend to reach out to Devito's team for installation guidelines.
- [ ] `openmp5/clang`. This installation requires the compilation from source `openmp`, `clang` and `llvm` to install the latest version of `openmp5` enabling gpu offloading. You can find instructions on this installation in Devito's [Wiki](https://github.com/devitocodes/devito/wiki)

### Setup

The only required setup for GPU support are the environment variables for [Devito](https://www.devitoproject.org). For the currently supported `nvc+openacc` setup these are:

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
docker run -p 8888:8888 mloubout/judi-base:1.0
```

This command downloads the image and launches a container. You will see a link that you can copy-past to your browser to access the notebooks. Alternatively, you can run a bash session, in which you can start a regular interactive Julia session and run the example scripts. Download/start the container as a bash session with:

```
docker run -it mloubout/judi-base:1.0 /bin/bash
```

Inside the container, all examples are located in the directory `/app/judi/examples/scripts`.


Additionaly, we provide two runtime docker images `mloubout/judi-cpu:1.4.3` and `mloubout/judi-gpu:1.0` that provide runtime (bash session) containers with additional librairies and compilers installed (`icc`, `nvcc`). These image do not offer  jupyter notebook as they are designed to be used as remote image for HPC (i.e [JUDI4Cloud.jl](https://github.com/slimgroup/JUDI4Cloud.jl)). The image `mloubout/judi-cpu:1.4.3` is recommended to be used with [JUDI4Cloud.jl](https://github.com/slimgroup/JUDI4Cloud.jl).

## Testing

A complete test suite is inculded with JUDI and is tested via GitHub Actions. You can also run the test locally
via:

```julia
    julia --project -e 'using Pkg;Pkg.test(coverage=false)'
```

By default, only the JUDI base API will be tested, however the testing suite supports other modes controlled via the environemnt variable `GROUP` such as:

```julia
	GROUP=JUDI julia --project -e 'using Pkg;Pkg.test(coverage=false)'
```

The supported modes are:

- JUDI : Only the base API (linear operators, vectors, ...)
- ISO_OP : Isotropic acoustic operators
- ISO_OP_FS : Isotropic acoustic operators with free surface
- TTI_OP : Transverse tilted isotropic operators
- TTI_OP_FS : Transverse tilted isotropic operators with free surface
- filename : you can also provide just a filename (i.e `GROUP=test_judiVector.jl`) and only this one test file will be run. Single files with TTI or free surface are not currently supported as it relies on `Base.ARGS` for the setup.


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
