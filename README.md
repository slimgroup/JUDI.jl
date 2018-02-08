
# The Julia Devito Inversion framework (JUDI)

## Overview

JUDI is a framework for large-scale seismic modeling and inversion and designed to enable rapid translations of algorithms to fast and efficient code that scales to industry-size 3D problems. The focus of the package lies on seismic modeling as well as PDE-constrained optimization such as full-waveform inversion (FWI) and imaging (LS-RTM). Wave equations in JUDI are solved with [Devito](https://github.com/opesci/devito), a Python domain-specific language for automated finite-difference (FD) computations.

## Installation and prerequisites

First, install Devito using `pip`, or see the [Devito homepage](https://github.com/opesci/devito) for installation with Conda and further information:

```julia
pip install --user git+https://github.com/opesci/devito.git@v3.1.0
```

Once Devito is installed, you can install JUDI with Julia's `Pkg.clone`:

```julia
Pkg.clone("https://github.com/slimgroup/JUDI.jl")
```

Devito uses just-in-time compilation for the underlying wave equation solves. The default compiler is intel, but can be changed to any other specified compiler such as `gnu`. Either run the following command from the command line or add it to your ~/.bashrc file:

```
export DEVITO_ARCH=gnu
```

For reading and writing seismic SEG-Y data, Julia Devito uses the [SeisIO](https://github.com/slimgroup/SeisIO.jl) package and matrix-free linear operators are based the [Julia Operator LIbrary](https://github.com/slimgroup/JOLI.jl/tree/master/src) (JOLI):

```julia
Pkg.clone("https://github.com/slimgroup/SeisIO.jl.git")
Pkg.clone("https://github.com/slimgroup/JOLI.jl.git")
```

## Full-waveform inversion

JUDI is designed to let you set up objective functions that can be passed to standard packages for (gradient-based) optimization. The following example demonstrates how to perform FWI on the 2D Overthrust model using a spectral projected gradient algorithm from the minConf library, which is included in the software. A small test dataset (62 MB) and the model can be downloaded from this FTP server:

```julia
run(`wget ftp://slim.eos.ubc.ca/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_2D.segy`)
run(`wget ftp://slim.eos.ubc.ca/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_2D_initial_model.h5`)
```

The first step is to load the velocity model and the observed data into Julia, as well as setting up bound constraints for the inversion, which prevent too high or low velocities in the final result. Furthermore, we define an 8 Hertz Ricker wavelet as the source function:

```julia
using PyPlot, HDF5, SeisIO, JUDI.TimeModeling, JUDI.SLIM_optim

# Load starting model
n,d,o,m0 = read(h5open("overthrust_2D_initial_model.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)	# need n,d,o as tuples and m0 as array

# Bound constraints
vmin = ones(Float32,model0.n) + 0.3f0
vmax = ones(Float32,model0.n) + 5.5f0
mmin = vec((1f0./vmax).^2)	# convert to slowness squared [s^2/km^2]
mmax = vec((1f0./vmin).^2)

# Load segy data
block = segy_read("overthrust_2D.segy")
dobs = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key="source", segy_depth_key="SourceDepth")	# read source position geometry
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1],0.008f0)	# 8 Hz wavelet
q = judiVector(src_geometry,wavelet)

```

For this FWI example, we define an objective function that can be passed to the minConf optimization library, which is included in the Julia Devito software package. We allow a maximum of 20 function evaluations using a spectral-projected gradient (SPG) algorithm. To save computational cost, each function evaluation uses a randomized subset of 20 shot records, instead of all 97 shots:

```julia
# Optimization parameters
srand(1)	# reset seed of random number generator
fevals = 20	# number of function evaluations
batchsize = 20	# number of sources per iteration
fvals = zeros(21)

# Objective function for minConf library
count = 0
function objective_function(x)
	model0.m = reshape(x,model0.n);

	# fwi function value and gradient
	i = randperm(dobs.nsrc)[1:batchsize]
	fval,grad = fwi_objective(model0, q[i], dobs[i])
	grad = reshape(grad, model0.n); grad[:,1:21] = 0f0	# reset gradient in water column to 0.
	grad = .1f0*grad/maximum(abs.(grad))	# scale gradient for line search
	
	global count; count += 1; fvals[count] = fval
    return fval, vec(grad)
end

# FWI with SPG
ProjBound(x) = median([mmin x mmax],2)	# Bound projection
options = spg_options(verbose=3, maxIter=fevals, memory=3)
x, fsave, funEvals= minConf_SPG(objective_function, vec(model0.m), ProjBound, options)
```

This example script can be run in parallel and requires roughly 220 MB of memory per source location. Execute the following code to generate figures of the initial model and the result, as well as the function values:

```julia
figure(); imshow(sqrt.(1./m0)'); title("Initial model")
figure(); imshow(sqrt.(1./reshape(x, model0.n))'); title("FWI")
figure(); plot(fvals); title("Function value")
```

#### Figure: {#f1}
![](docs/fwi.png){width=70%} 


## Least squares reverse-time migration

JUDI includes matrix-free linear operators for modeling and linearized (Born) modeling, that let you write algorithms for migration that follow the mathematical notation of standard least squares problems. This example demonstrates how to use Julia Devito to perform least-squares reverse-time migration on the 2D Marmousi model. Start by downloading the test data set (1.1 GB) and the model:

```julia
run(`wget ftp://slim.eos.ubc.ca/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_2D.segy`)
run(`wget ftp://slim.eos.ubc.ca/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_migration_velocity.h5`)
```

Once again, load the starting model and the data and set up the source wavelet. For this example, we use a Ricker wavelet with 30 Hertz peak frequency. For setting up matrix-free linear operators, an `info` structure with the dimensions of the problem is required:

```julia
using PyPlot, HDF5, JUDI.TimeModeling, SeisIO

# Load smooth migration velocity model
n,d,o,m0 = read(h5open("marmousi_migration_velocity.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)

# Load data
block = segy_read("marmousi_2D.segy")
dD = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key="source", segy_depth_key="SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1],0.03)	# 30 Hz wavelet
q = judiVector(src_geometry,wavelet)

# Set up info structure
ntComp = get_computational_nt(q.geometry,dD.geometry,model0)	# no. of computational time steps
info = Info(prod(model0.n),dD.nsrc,ntComp)
```

To speed up the convergence of our imaging example, we set up a basic preconditioner for each the model- and the data space, consisting of mutes to suppress the ocean-bottom reflection in the data and the source/receiver imprint in the image. The operator `J` represents the linearized modeling operator and its adjoint `J'` corresponds to the migration (RTM) operator. The forward and adjoint pair can be used for a basic LS-RTM example with (stochastic) gradient descent:

```julia
# Set up matrix-free linear operators
F = judiModeling(info,model0,q.geometry,dD.geometry)
J = judiJacobian(F,q)

# Right-hand preconditioners (model topmute)
Mr = judiTopmute(model0.n,42,10)	# mute up to grid point 42, with 10 point taper

# Stochastic gradient
x = zeros(Float32,info.n)	# zero initial guess
batchsize = 10	# use subset of 10 shots per iteration
niter = 32
fval = zeros(Float32,niter)
for j=1:niter
	println("Iteration: ", j)

	# Select batch and set up left-hand preconditioner
	i = randperm(dD.nsrc)[1:batchsize]
	Ml = judiMarineTopmute2D(30,dD[i].geometry)	# data topmute starting at time sample 30

	# Compute residual and gradient
	r = Ml*J[i]*Mr*x - Ml*dD[i]
	g = Mr'*J[i]'*Ml'*r

	# Step size and update variable
	fval[j] = .5*norm(r)^2
	t = norm(r)^2/norm(g)^2
	x -= t*g
end
```

#### Figure: {#f1}
![](docs/lsrtm.png){width=80%} 

## Authors

This package was written by [Philipp Witte](https://www.slim.eos.ubc.ca/philip) and [Mathias Louboutin](https://www.slim.eos.ubc.ca/content/mathias-louboutin) from the Seismic Laboratory for Imaging and Modeling (SLIM) at the University of British Columbia (UBC).

Contact authors via: pwitte@eoas.ubc.ca, mloubout@eoas.ubc.ca


