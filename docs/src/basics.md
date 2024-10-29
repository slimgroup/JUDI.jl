# Getting Started

These tutorials provide instructions of how to set up various modeling or inversion scenarios with JUDI. For a list of runnable Julia scripts and reproducable research, please also check out the examples:
 - The [examples](https://github.com/slimgroup/JUDI.jl/tree/master/examples/scripts) scripts contain simple modeling and inversion examples such as FWI, LSRTM, and medical modeling.
 - The [machine-learning](https://github.com/slimgroup/JUDI.jl/tree/master/examples/machine-learning) scripts contain examples of machine learning using Flux.

```@contents
Pages = ["tutorials.md"]
```

## 2D Modeling Quickstart

To set up a simple 2D modeling experiment with JUDI with an OBN-type acquisition (receivers everywhere), we start by loading the module and building a two layer model:

```julia
using JUDI

# Grid
n = (120, 100)   # (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32, n) .* 1.4f0
v[:, 50:end] .= 5f0

# Squared slowness
m = (1f0 ./ v).^2
```

For working with JUDI operators, we need to set up a model structure, which contains the grid information, as well as the slowness. Optionally, we can provide an array of the density in `g/cm^3` (by default a density of 1 is used):

```julia
# Density (optional)
rho = ones(Float32, n)

# Model structure:
model = Model(n, d, o, m; rho=rho)
```

Next, we define our source acquisition geometry, which needs to be defined as a `Geometry` structure. The `Geometry` function requires the x-, y- and z-coordinates of the source locations as input, as well as the modeling time and samping interval of the wavelet. In general, each parameter can be passed as a cell array, where each cell entry provides the information for the respective source location. The helper function `convertToCell` converts a Julia `range` to a cell array, which makes defining the source geometry easier:

```julia
# Set up source geometry
nsrc = 4    # no. of sources
xsrc = convertToCell(range(400f0, stop=800f0, length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(20f0, stop=20f0, length=nsrc))

# Modeling time and sampling interval
time = 1000f0  # ms
dt = 2f0   # ms

# Set up source structure
src_geometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time)
```

Now we can define our source wavelet. The source must be defined as a `judiVector`, which takes the source geometry, as well as the source data (i.e. the wavelet) as an input argument:

```julia
# Source wavelet
f0 = 0.01f0     # kHz
wavelet = ricker_wavelet(time, dt, f0)
q = judiVector(src_geometry, wavelet)
```

In general, `wavelet` can be a cell array with a different wavelet in each cell, i.e. for every source location. Here, we want to use the same wavelet for all 4 source experiments, so we can simply pass a single vector. As we already specified in our `src_geometry` object that we want to have 4 source locations, `judiVector` will automaticallty copy the wavelet for every experiment.

Next, we set up the receiver acquisition geometry. Here, we define an OBN acquisition, where the receivers are spread out over the entire domain and each source experiment uses the same set of receivers. Again, we can in principle pass the coordinates as cell arrays, with one cell per source location. Since we want to use the same geometry for every source, we can use a short cut and define the coordinates as Julia `ranges` and pass `nsrc=nsrc` as an optional argument to the `Geometry` function. This tells the function that we want to use our receiver set up for `nsrc` distinct source experiments:

```julia
# Set up receiver geometry (for 2D, set yrec to zero)
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# Set up receiver structure
rec_geometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)
```

Next, we can define separate operators for source/receiver projections and a forward modeling operator:

```julia
# Setup operators
Pr = judiProjection(rec_geometry)
A_inv = judiModeling(model)
Ps = judiProjection(src_geometry)
```

We can see, that from JUDI's perspective, source and receivers are treated equally and are represented by the same operators (`judiProjection`) and vectors (`judiVector`).

We also could've skipped setting up the projection operators and directly created:

```julia
F = judiModeling(model, src_geometry, rec_geometry)
```

which is equivalent to creating the combined operator:

```julia
F = Pr*A_inv*Ps'
```

Finally, to model our seismic data, we run:

```julia
d_obs = Pr*A_inv*Ps'*q
# or
d_obs = F*q
```

We can plot a 2D shot record by accessing the `.data` field of the `judiVector`, which contains the data in the original (non-vectorized) dimensions:

```julia
using PythonPlot
imshow(d_obs.data[1], vmin=-5, vmax=5, cmap="seismic", aspect="auto")
```

We can also set up a Jacobian operator for Born modeling and reverse-time migration. First we set up a (constant) migration velocity model:

```julia
v0 = ones(Float32, n) .* 1.4f0
m0 = (1f0 ./ v0).^2
dm = m - m0     # model perturbation/image

# Model structure
model0 = Model(n, d, o, m0)
```

We can create the Jacobian directly from a (non-linear) modeling operator and a source vector:

```julia
A0_inv = judiModeling(model0) # modeling operator for migration velocity
J = judiJacobian(Pr*A0_inv*Ps', q)
```

We can use this operator to model single scattered data, as well as for migration our previous data:

```julia
d_lin = J*vec(dm)

# RTM
rtm = J'*d_obs
```

To plot, first reshape the image:

```julia
rtm = reshape(rtm, model0.n)
imshow(rtm', cmap="gray", vmin=-1e3, vmax=1e3)
```

## 3D Modeling Quickstart

Setting up a 3D experiment largely follows the instructions for the 2D example. Instead of a 2D model, we define our velocity model as:

```julia
using JUDI

# Grid
n = (120, 100, 80)   # (x,y,z)
d = (10., 10., 10.)
o = (0., 0., 0.)

# Velocity [km/s]
v = ones(Float32, n) .* 1.4f0
v[:, :, 40:end] .= 5f0

# Squared slowness and model structure
m = (1f0 ./ v).^2
model = Model(n, d, o, m)
```

Our source coordinates now also need to have the y-coordinate defined:

```julia
# Set up source geometry
nsrc = 4    # no. of sources
xsrc = convertToCell(range(400f0, stop=800f0, length=nsrc))
ysrc = convertToCell(range(200f0, stop=1000f0, length=nsrc))
zsrc = convertToCell(range(20f0, stop=20f0, length=nsrc))

# Modeling time and sampling interval
time = 1000f0  # ms
dt = 2f0   # ms

# Set up source structure
src_geometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time)
```

Our source wavelet, is set up as in the 2D case:

```julia
# Source wavelet
f0 = 0.01f0     # kHz
wavelet = ricker_wavelet(time, dt, f0)
q = judiVector(src_geometry, wavelet)
```

For the receivers, we generally need to define each coordinate (x, y, z) for every receiver. I.e. `xrec`, `yrec` and `zrec` each have the length of the total number of receivers. However, oftentimes we are interested in a regular receiver grid, which can be defined by two basis vectors and a constant depth value for all receivers. We can then use the `setup_3D_grid` helper function to create the full set of coordinates:

```julia
# Receiver geometry
nxrec = 120
nyrec = 100
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = range(100f0, stop=900f0, length=nyrec)
zrec = 50f0

# Construct 3D grid from basis vectors
(xrec, yrec, zrec) = setup_3D_grid(xrec, yrec, zrec)

# Set up receiver structure
rec_geometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)
```

Setting up the modeling operators is done as in the previous 2D case:

```julia
# Setup operators
Pr = judiProjection(rec_geometry)
A_inv = judiModeling(model)
Ps = judiProjection(src_geometry)

# Model data
d_obs = Pr*A_inv*Ps'*q
```

The 3D shot records are still saved as 2D arrays of dimensions `time x (nxrec*nyrec)`:

```julia
using PythonPlot
imshow(d_obs.data[1], vmin=-.4, vmax=.4, cmap="seismic", aspect="auto")
```

## Vertical and tilted-transverse isotropic modeling (VTI, TTI)

JUDI supports both VTI and TTI modeling based on a coupled pseudo-acoustic wave equation. To enable VTI/TTI modeling, simply pass Thomsen parameters as well as the tilt angles to the `Model` structure as optional keyword arguments:

```julia
# Grid and model
n = (120, 100, 80)
d = (10., 10., 10)
o = (0., 0., 0.)

# Velocity
v = ones(Float32, n) .* 1.5f0
m = 1f0 ./ v.^2

# Thomsen parameters
epsilon = ones(Float32, n) .* 0.2f0
delta = ones(Float32, n) .* 0.1f0

# Tile angles for TTI
theta = ones(Float32, n) .* pi/2f0
phi = ones(Float32, n) .* pi/3f0    # 3D only

# Set up model structure with Thomsen parameters
model = Model(n, d, o, m; rho=rho, epsilon=epsilon, delta=delta, theta=theta, delta=delta)
```

## Modeling with density

To use density, pass `rho` in the units of `[g/cm^3]` as an optional keyword argument to the Model structure. The default density is `rho=1f0` (i.e. density of water):

```julia
# Grid and model
n = (120, 100)
d = (10., 10.)
o = (0., 0.)
v = ones(Float32, n) .* 1.5f0
m = 1f0 ./ v.^2
rho = ones(Float32, n) .* 1.1f0

# Set up model structure with density
model = Model(n, d, o, m; rho=rho)
```

## 2D Marine streamer acquisition

For a marine streamer acquisition, we need to define a moving set of receivers representing a streamer that is towed behind a seismic source vessel. In JUDI, this is easily done by defining a different set of receivers for each source location. Here, we explain how to set up the `Geometry` objects for a 2D marine streamer acquisition.

If we define that our streamer is to the right side of the source vessel, this has the effect that part of the streamer is outside the grid while our vessel is in the right side of the model. To circumvent this, we can say that our streamer is on the right side of the source while the vessel is in the left-hand side of the model and vice versa. This way, we get the full maximum offset coverage for every source location (assuming that the maximum offset is less or equal than half the domain size). 

First, we have to specify our domain size (the physical extent of our model), as well as the number of receivers and the minimum and maximum offset:

```julia
domain_x = (model.n[1] - 1)*model.d[1]    # horizontal extent of model
nrec = 120     # no. of receivers
xmin = 50f0    # leave buffer zone w/o source and receivers of this size
xmax = domain_x - 50f0
min_offset = 10f0      # distance between source and first receiver
max_offset = 400f0    # distance between source and last
xmid = domain_x / 2     # midpoint of model
source_spacing = 25f0   # source interval [m]
```

For the JUDI `Geometry` objects, we need to create cell arrays for the source and receiver coordinates, with one cell entry per source location:

```julia
# Source/receivers
nsrc = 20   # number of shot locations

# Receiver coordinates
xrec = Array{Any}(undef, nsrc)
yrec = Array{Any}(undef, nsrc)
zrec = Array{Any}(undef, nsrc)

# Source coordinates
xsrc = Array{Any}(undef, nsrc)
ysrc = Array{Any}(undef, nsrc)
zsrc = Array{Any}(undef, nsrc)
```

Next, we compute the source and receiver coordinates for when the vessel moves from left to right in the right-hand side of the model:

```julia
# Vessel goes from left to right in right-hand side of model
nsrc_half = Int(nsrc/2)
for j=1:nsrc_half
    xloc = xmid + (j-1)*source_spacing

    # Current receiver locations
    xrec[j] = range(xloc - max_offset, xloc - min_offset, length=nrec)
    yrec[j] = 0.
    zrec[j] = range(50f0, 50f0, length=nrec)
    
    # Current source
    xsrc[j] = xloc
    ysrc[j] = 0f0
    zsrc[j] = 20f0
end
```

Then, we repeat this for the case where the vessel goes from right to left in the left-hand model side:

```julia
# Vessel goes from right to left in left-hand side of model
for j=1:nsrc_half
    xloc = xmid - (j-1)*source_spacing
    
    # Current receiver locations
    xrec[nsrc_half + j] = range(xloc + min_offset, xloc + max_offset, length=nrec)
    yrec[nsrc_half + j] = 0f0
    zrec[nsrc_half + j] = range(50f0, 50f0, length=nrec)
    
    # Current source
    xsrc[nsrc_half + j] = xloc
    ysrc[nsrc_half + j] = 0f0
    zsrc[nsrc_half + j] = 20f0
end
```

Finally, we can set the modeling time and sampling interval and create the `Geometry` objects:

```julia
# receiver sampling and recording time
time = 10000f0   # receiver recording time [ms]
dt = 4f0    # receiver sampling interval

# Set geometry objects
rec_geometry = Geometry(xrec, yrec, zrec; dt=dt, t=time)
src_geometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time)
```

You can find a full (reproducable) example for generating a marine streamer data set for the Sigsbee 2A model [here](https://github.com/slimgroup/JUDI.jl/blob/master/examples/compressive_splsrtm/Sigsbee2A/generate_data_sigsbee.jl).


## Simultaneous sources

To set up a simultaneous source with JUDI, we first create a cell array with `nsrc` cells, where `nsrc` is the number of separate experiments (here `nsrc=1`). For a simultaneous source, we create an array of source coordinates for each cell entry. In fact, this is exactly like setting up the receiver geometry, in which case we define multiple receivers per shot location. Here, we define a single experiment with a simultaneous source consisting of four sources:

```julia
nsrc = 1    # single simultaneous source
xsrc = Vector{Float32}(undef, nsrc)
ysrc = Vector{Float32}(undef, nsrc)
zsrc = Vector{Float32}(undef, nsrc)

# Set up source geometry
xsrc[1] = [250f0, 500f0, 750f0, 1000f0]     # four simultaneous sources
ysrc[1] = 0f0
zsrc[1] = [50f0, 50f0, 50f0, 50f0]	

# Source sampling and number of time steps
time = 2000f0
dt = 4f0

# Set up source structure
src_geometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time)
```

With the simultaneous source geometry in place, we can now create our simultaneous data. As we have four sources per sim. source, we create an array of dimensions `4 x src_geometry.nt[1]` and fill it with wavelets of different time shifts:

```julia
# Create wavelet
f0 = 0.01	# source peak frequencies
q = ricker_wavelet(500f0, dt, f0)  # 500 ms wavelet

# Create array with different time shifts of the wavelet
wavelet = zeros(Float32, 4, src_geometry.nt[1])
wavelet[1, 1:1+length(q)-1] = q
wavelet[2, 41:41+length(q)-1] = q
wavelet[3, 121:121+length(q)-1] = q
wavelet[4, 201:201+length(q)-1] = q
```

Finally, we create our simultaneous source as a `judiVector`:

```julia
# Source wavelet
q = judiVector(src_geometry, wavelet)
```

## Computational simultaneous sources (super shots)

The computational simultaneous sources refer to superposition of sequentially-fired sources and shot records from the field. These computational simultaneous shot records are not acquired in the field simultaneously, but computational simultaneous sources introduce randomness to the optimization problems like FWI and LS-RTM, which takes advantage of the knowledge in randomized linear algebra to make optimization faster and more robust.

The implementation of computational simultaneous sources follows the journal article [Fast randomized full-waveform inversion with compressive sensing](https://slim.gatech.edu/content/fast-randomized-full-waveform-inversion-compressive-sensing)
The simultaneous sources experiments are generated by superposition of shot records with random weights drawn from Gaussian distribution.

```julia
# assume dobs is generated by sequentially fired point sources q
nsimsrc = 8
# Set up random weights
weights = randn(Float32,nsimsrc,q.nsrc)
# Create SimSource
q_sim = weights * q
data_sim = weights * dobs
# We can also apply the weights to the operator and check the equivalence
d_sim = (weights * F) * q_sim
isapprox(data_sim, d_sim)
```

## Working with wavefields

JUDI allows computing full time domain wavefields and using them as right-hand sides for wave equations solves. This tutorial shows how. We start by setting up a basic 2D experiment:


```julia
using JUDI

# Grid
n = (120, 100)   # (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32, n) .* 1.4f0
v[:, 50:end] .= 5f0

# Squared slowness
m = (1f0 ./ v).^2

# Model structure:
model = Model(n, d, o, m)
```

Next, we set up the source geometry for a single source experiment:

```julia
# Set up source geometry
nsrc = 1    # no. of sources
xsrc = convertToCell([600f0])
ysrc = convertToCell([0f0])
zsrc = convertToCell([20f0])

# Modeling time and sampling interval
time = 600f0  # ms
dt = 4f0   # ms

# Set up source structure
src_geometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time)

# Source wavelet
f0 = 0.01f0     # kHz
wavelet = ricker_wavelet(time, dt, f0)
q = judiVector(src_geometry, wavelet)
```

As in the 2D quick start tutorial, we create our modeling operator and source projection operator:

```julia
# Setup operators
A_inv = judiModeling(model)
Ps = judiProjection(src_geometry)
```

To model a wavefield, we simply omit the receiver sampling operator:

```julia
u = A_inv*Ps'*q
```

This return an abstract data vector called `judiWavefield`. Similar to `judiVectors`, we can access the data for each source number `i` via `u.data[i]`. The data is a 3D array of size `(nt, nx, nz)` for 2D and a 4D array of size `(nt, nx, ny, nz)` for 3D. We can plot the wavefield of the 600th time step with:

```julia
using PythonPlot
imshow(u.data[1][600, :, :]', vmin=-5, vmax=5, cmap="seismic", aspect="auto")
```

We can also use the computed wavefield `u` as a right-hand side for forward and adjoint wave equation solves:

```julia
v = A_inv*u
w = A_inv'*u
```

Similarly, by setting up a receiver projection operator, we can use wavefields as right-hand sides, but restrict the output to the receiver locations.

## Extended source modeling

JUDI supports extened source modeling, which injects a 1D wavelet `q` at every point in the subsurface weighted by a spatially varying extended source. To demonstrate extended source modeling, we first set up a runnable 2D experiment with JUDI. We start with defining the model:

```julia
using JUDI

# Grid
n = (120, 100)   # (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32, n) .* 1.4f0
v[:, 50:end] .= 5f0

# Squared slowness
m = (1f0 ./ v).^2

# Model structure:
model = Model(n, d, o, m)
```

Next, we set up the receiver geometry:

```julia
# Number of experiments
nsrc = 2

# Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# Modeling time and receiver sampling interval
time = 2000
dt = 4

# Set up receiver structure
rec_geometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)
```

For the extended source, we do not need to set up a source geometry object, but we need to define a wavelet function:

```julia
# Source wavelet
f0 = 0.01f0     # MHz
wavelet = ricker_wavelet(time, dt, f0)
```

As before, we set up a modeling operator and a receiver sampling operator:

```julia
# Setup operators
A_inv = judiModeling(model)
Pr = judiProjection(rec_geometry)
```

We define our extended source as a so called `judiWeights` vector. Similar to a `judiVector`, the data of this abstract vector is stored as a cell array, where each cell corresponds to one source experiment. We create a cell array of length two and create a random array of the size of the model as our extended source:

```julia
weights = Array{Array}(undef, nsrc)
for j=1:nsrc
    weights[j] = randn(Float32, model.n)
end
w = judiWeights(weights)
```

To inject the extended source into the model and weight it by the wavelet, we create a special projection operator called `judiLRWF` (for JUDI low-rank wavefield). This operator needs to know the wavelet we defined earlier. We can then create our full modeling operator, by combining `Pw` with `A_inv` and the receiver sampling operator:

```julia
# Create operator for injecting the weights, multiplied by the provided wavelet(s)
Pw = judiLRWF(wavelet)

# Model observed data w/ extended source
F = Pr*A_inv*adjoint(Pw)
```

Extended source modeling supports both forward and adjoint modeling:

```julia
# Simultaneous observed data
d_sim = F*w
dw = adjoint(F)*d_sim
```

As for regular modeling, we can create a Jacobian for linearized modeling and migration. First we define a migration velocity model and the corresponding modeling operator `A0_inv`:

```julia
# Migration velocity and squared slowness
v0 = ones(Float32, n) .* 1.4f0
m0 = (1f0 ./ v0).^2

# Model structure and modeling operator for migration velocity
model0 = Model(n, d, o, m0)
A0_inv = judiModeling(model0)

# Jacobian and RTM
J = judiJacobian(Pr*A0_inv*adjoint(Pw), w)
rtm = adjoint(J)*d_sim
```

As before, we can plot the image after reshaping it into its original dimensions:

```julia
rtm = reshape(rtm, model.n)
imshow(rtm', cmap="gray", vmin=-3e6, vmax=3e6)
```

Please also refer to the reproducable example on github for [2D](https://github.com/slimgroup/JUDI.jl/blob/master/examples/scripts/modeling_extended_source_2D.jl) and [3D](https://github.com/slimgroup/JUDI.jl/blob/master/examples/scripts/modeling_extended_source_3D.jl) extended modeling.

## Impedance imaging (inverse scattering)

JUDI supports imaging (RTM) and demigration (linearized modeling) using the linearized inverse scattering imaging condition (ISIC) and its corresponding adjoint. ISIC can be enabled via the `Options` class. You can set this options when you initially create the modeling operator:


```julia
# Options strucuture
opt = Options(isic=true)

# Set up modeling operator
A0_inv = judiModeling(model0; options=opt)
```

When you create a Jacobian from a forward modeling operator, the Jacobian inherits the options from `A0_inv`:

```julia
J = judiJacobian(Pr*A0_inv*Ps', q)
J.options.isic
# -> true
```

Alternatively, you can directly set the option in your Jacobian:

```julia
J.options.isic = true   # enable isic
J.options.isic = false  # disable isic
```

## Optimal checkpointing

JUDI supports optimal checkpointing via Devito's interface to the Revolve library. To enable checkpointing, use the `Options` class:

```julia
# Options strucuture
opt = Options(optimal_checkpointing=true)

# Set up modeling operator
A0_inv = judiModeling(model0; options=opt)
```

When you create a Jacobian from a forward modeling operator, the Jacobian inherits the options from `A0_inv`:

```julia
J = judiJacobian(Pr*A0_inv*Ps', q)
J.options.optimal_checkpointing
# -> true
```

Alternatively, you can directly set the option in your Jacobian:

```
J.options.optimal_checkpointing = true   # enable checkpointing
J.options.optimal_checkpointing = false  # disable checkpointing
```


## On-the-fly Fourier transforms

JUDI supports seismic imaging in the frequency domain using on-the-fly discrete Fourier transforms (DFTs). To compute an RTM image in the frequency domain for a given set of frequencies, we first create a cell array for the frequencies of each source experiment:


```julia
nsrc = 4    # assume 4 source experiments
frequencies = Array{Any}(undef, nsrc)
```

Now we can define single or multiple frequencies for each shot location for which the RTM image will be computed:

```julia
# For every source location, compute RTM image for 10 and 20 Hz
for j=1:nsrc
    frequencies[j] = [0.001, 0.002]
end
```

The frequencies are passed to the Jacobian via the options field. Assuming we already have a Jacobian set up, we set the frequencies via:

```julia
J.options.frequencies = frequencies
```

Instead of the same two frequencies for each source experiment, we could have chosen different random sets of frequencies, which creates an RTM with incoherent noise. We can also draw random frequencies using the frequency spectrum of the true source as the probability density function. To create a distribution for a given source `q` (`judiVector`) from which we can draw frequency samples, use:

```julia
q_dist = generate_distribution(q)
```

Then we can assigne a random set of frequencies in a specified range as follows:

```julia
nfreq = 10  # no. of frequencies per source location
for j=1:nsrc
    J.options.frequencies[j] = select_frequencies(q_dist; fmin=0.003, fmax=0.04, nf=nfreq)
end
```

Once the `options.frequencies` field is set, on-the-fly DFTs are used for both born modeling and RTM.
To save computational cost, we can limit the number of DFTs that are performed. Rather than computing the DFT at every time step, we can define a subsampling factor as follows:

```julia
# Compute DFT every 4 time steps
J.options.dft_subsampling_factor=4
```