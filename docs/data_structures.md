# Data structures

## Model structure

Velocity models in JUDI are defined as `Model` structures. `Model` requires the following input arguments:

```julia
model = Model(n, d, o, m; nb=40, rho=1f0)
```

**Parameters:**

 * `n`: Integer tuple with number of grid points in each dimension, e.g. `n = (120, 100)` (2D) or `n = (120, 100, 80)` (3D). **The order of dimenions in all tuples is `(x, z)` for 2D and `(x, y, z)` for 3D**.

 * `d`: Real tuple with grid spacing in each dimension.

 * `o`: Real tuple with coordinate origin (typically `o = (0f0, 0f0)`).

 * `m`: 2D or 3D array of the velocity model in squared slowness ``[s^2/km^2]``.

 * `nb`: Number of absorbing boundary points on each edge. Default is ``nb = 40``.

 * `rho`: 2D or 3D array of the density in ``[g / cm^3]``


**Access fields:**

```julia
# Access model
model.m

# Access number of grid points
model.n
```

## Geometry structure

JUDI's geometry structure contains the information of either the source **or** the receiver geometry. Each geometry object contains 6 fields:


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

Example of how to access fields of geometry objects:

```julia
# Access cell arrays of x coordinates:
geometry.xloc

# Access x coordinates at first source location
geometry.xloc[1]

# Access first receiver location (in x) at the second source location
geometry.xloc[2][1]
```


## Info structure

The info structure contains some basic dimensionality information that needs to be available to any type of linear operator:

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

