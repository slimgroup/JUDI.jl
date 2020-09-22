# Data structures

## Physical Parameter

Data structure for physicall parameter array in JUDI. A `PhysicalParameter` inherits from julia `AbstractVector`

**Construction**

A `PhysicalParameter` can be constructed in various ways but always require the origin `o` and grid spacing `d` that
cannot be infered from the array.

```julia
p = PhysicalParameter(v::Array{vDT}, d, o)
```
where `v` is an n-dimensional array and n=size(v).

```julia
p = PhysicalParameter(n, d, o; vDT=Float32)
```
Creates a zero PhysicalParameter.

```julia
p = PhysicalParameter(v::Array{vDT}, A::PhysicalParameter)
```
Creates a PhysicalParameter from the Array `v` with n, d, o from `A`.

```julia
p = PhysicalParameter(v::Array{vDT, N}, n::Tuple, d::Tuple, o::Tuple)
```
where `v` is a vector or nd-array that is reshaped into shape `n`.

```julia
p = PhysicalParameter(v::vDT, n::Tuple, d::Tuple, o::Tuple)
```
Creates a constant (single number) PhyicalParameter.

**Access fields:**


## Model structure

Data structure for velocity models in JUDI.

**Construction:**

`Model` requires the following input arguments:

```julia
model = Model(n, d, o, m; nb=40, rho=1f0, epsilon=0f0, delta=0f0, theta=0f0, phi=0f0)
```

Accessible fields include all of the above parameters `p.n, p.d, p.o, p.data`. Additionaly, arithmetic operation are all impemented such as addition, multiplication, broadcasting and indexing. Linear algebra operation are implemented as well but will return a standard Julia vector if the matrix used is external to JUDI.

**Parameters:**

 * `n`: Integer tuple with number of grid points in each dimension, e.g. `n = (120, 100)` (2D) or `n = (120, 100, 80)` (3D). **The order of dimenions in all tuples is `(x, z)` for 2D and `(x, y, z)` for 3D**.

 * `d`: Real tuple with grid spacing in each dimension.

 * `o`: Real tuple with coordinate origin (typically `o = (0f0, 0f0)`).

 * `m`: 2D or 3D array of the velocity model in squared slowness ``[s^2/km^2]``.

 * `nb`: Number of absorbing boundary points on each edge. Default is ``nb = 40``.

 * `rho`: 2D or 3D array of the density in ``[g / cm^3]`` (default is 1)

 * `epsilon`: Thomsen parameter epsilon for VTI/TTI modeling (default is 0)

 * `delta`: Thomsen parameter delta for VTI/TTI modeling (default is 0)

 * `theta`: Vertical tilt angle of TTI symmetry axis (default is 0) ``[rad]``

 * `phi`: Horizontal tilt angle of TTI symmetry axis (default is 0) ``[rad]``


**Access fields:**

Accessible fields include all of the above parameters, which can be accessed as follows:

```julia
# Access model
model.m

# Access number of grid points
model.n
```

## Geometry structure

JUDI's geometry structure contains the information of either the source **or** the receiver geometry. 

**Construction:**

Construct an (in-core) geometry object for **either** a source or receiver set up:

```julia
geometry = Geometry(xloc, yloc, zloc; dt=[], nt=[], t=[])
```

**Parameters:**

 * `xloc`: Cell array, with one cell per source location. Each cell contains a 1D Julia array with the coordinates in the horizontal x direction. Coordinates are specified as distances in meters `[m]` relative to the model origin.

 * `yloc`: Cell array for horizontal y coordinates. For 2D, set each cell entry to `0f0`.

 * `zloc`: Cell array for depth coordinates (z) at each source location.

 * `dt`: Cell array with the time intervals at which the data was sampled (i.e. a shot record or source wavelet was sampled). Units in milliseconds `[ms]`.

* `nt`: Cell array with number of time samples.

* `t`: Cell array with the recording lengths in milliseconds `[ms]`.

From the optional arguments, you have to pass (at least) **two** of `dt`, `nt` and `t`. The third value is automatically determined and set from the two other values.


**Access fields:**

Accessible fields include all of the above parameters, which can be accessed as follows:

```julia
# Access cell arrays of x coordinates:
geometry.xloc

# Access x coordinates of the i-th source location
geometry.xloc[i]

# Access j-th receiver location (in x) of the i-th source location
geometry.xloc[i][j]
```


## Info structure

The info structure contains some basic dimensionality information that needs to be available to any type of linear operator:

**Construction:**

```julia
info = Info(n, nsrc, nt)
```

**Parameters**:

 * `n`: Total number of grid points in all dimensions. Given by `prod(model.n)`.

 * `nsrc`: Number of source/shot locations in the seismic experiment.

 * `nt`: Number of computational time steps.

You can automatically obtain the number of computational time steps as follows:

```julia
nt = get_computational_nt(src_geometry, rec_geometry, model)
```

where `src_geometry` is a `Geometry` object with the source geometry, `rec_geometry` is a `Geometry` object with the receiver geometry and `model` is a `Model` structure.


## Options structure

The options structure allows setting several modeling parameters.

**Construction:**

 * Choose all default options:

```julia
opt = Options()
```

 * List of all possible options:

```julia
opt = Options(
    space_order:: Integer
    space_order:: Integer
    free_surface:: Bool
    limit_m:: Bool
    buffer_size:: Real
    save_data_to_disk:: Bool
    save_wavefield_to_disk:: Bool
    file_path:: String
    file_name:: String
    sum_padding:: Bool
    optimal_checkpointing:: Bool
    num_checkpoints:: Union{Integer, Nothing}
    checkpoints_maxmem:: Union{Real, Nothing}
    frequencies:: Array
    subsampling_factor:: Integer
    dft_subsampling_factor:: Integer
    isic:: Bool
    return_array:: Bool
    dt_comp:: Union{Real, Nothing}
)
```

**Parameters:**

 * `space_order`: Finite difference space order for wave equation (default is `8`, needs to be multiple of 4).
 
 * `free_surface`: Set to `true` to enable a free surface boundary condition (default is `false`).
 
 * `limit_m`: For 3D modeling, limit modeling domain to area with receivers (default is `false`).
 
 * `buffer_size`: If `limit_m=true`, define buffer area on each side of modeling domain (in meters)
 
 * `save_data_to_disk`: If `true`, saves shot records as separate SEG-Y files (default is `false`).
 
 * `save_wavefield_to_disk`: If wavefield is return value, save wavefield to disk as pickle file (default is `false`).
 
 * `file_path`: Path to directory where data is saved.
 
 * `file_name`: Shot records will be saved as specified file name plus its source coordinates.
 
 * `sum_padding`: When removing the padding area of the gradient, sum values into the most outer rows/columns (default is `false`). Required to pass adjoint tests.
 
 * `optimal_checkpointing`: Use optimal wavefield checkpointing (default is `false`).
 
 * `num_checkpoints`: Number of checkpoints. If not supplied, is set to `log(num_timesteps)`.
 
 * `checkpoints_maxmem`: Maximum amount of memory that can be allocated for checkpoints (MB).
 
 * `frequencies`: Provide a cell array (one cell per shot location), where each cell contains an array of frequencies. In this case, the RTM/FWI gradient is computed for the given set of frequencies using on-the-fly Fourier transforms (default is `nothing`, i.e. the gradient is computed in the time domain).
 
 * `subsampling_factor`: Compute forward wavefield on a time axis that is reduced by a given factor (default is `1`).
 
 * `dft_subsampling_factor`: Compute on-the-fly DFTs on a time axis that is reduced by a given factor (default is `1`).
 
 * `isic`: Use linearized inverse scattering imaging condition for the Jacobian (default is `false`).
 
 * `return_array`: Return data from nonlinear/linear modeling as a plain Julia array instead of as a `judiVector` (default is `false`).
 
 * `dt_comp`: Overwrite automatically computed computational time step (default option) with this value.
