# Helper functions

## Ricker wavelet

Create a 1D Ricker wavelet:

```julia
wavelet = ricker_wavelet(tmax, dt, f0)
```

**Parameters:**

 * `tmax`: Modeling time in milliseconds `[ms]`.

 * `dt`: Sampling interval in milliseconds (can be different from `dt` of shot data).

 * `f0`: Peak frequency in Kilohertz `[kHz]`.

## Compute CFL time stepping interval 

Calculate the time stepping interval based on the CFL condition

```julia
dt = calculate_dt(n, d, o, v, rho)
```

**Parameters:**

* `n`: Tuple with number of grid points.

* `d`: Tuple with grid spacing.

* `o`: Tuple with coordiante system origin.

* `v`: 2D/3D Julia array with velocity in `[km/s]`.

* `rho`: 2D/3D Julia array with density in `[g/cm^3]`.


## Compute number of computational time steps

Estimate the number of computational time steps. Required for calculating the dimensions of the matrix-free linear modeling operators:

```julia
nt = get_computational_nt(src_geometry, rec_geometry, model)
```

**or** (for extended source modeling, where `src_geometry` is not available):

```julia
nt = get_computational_nt(rec_geometry, model)
```

**Parameters:**

* `src_geometry`: A JUDI `Geometry` object with the source geometry.

* `rec_geometry`: A JUDI `Geometry` object with the receiver geometry.

* `model`: A JUDI `Model` object containing the velocity model.

## Set up 3D acquisition grid

Helper function to create a regular acquisition grid for a 3D survey.

```julia
x_coord_full, y_coord_full, z_coord_full = setup_3D_grid(x_coord, y_coord, z_coord)
```

**Parameters:**

 * `x_coord`: 1D julia vector of length `nx`, where `nx` is the number of distinct source/receiver locations in x direction. 

 * `y_coord`: 1D julia vector of length `ny`, where `ny` is the number of distinct source/receiver locations in y direction. 

 * `z_coord`: Single scalar for depth of sources/receivers.


**Returns:**

 * `x_coord_full`: 1D julia vector of length `nx * ny` with source/receiver locations in x direction.

 * `y_coord_full`: 1D julia vector of length `nx * ny` with source/receiver locations in y direction.

 * `z_coord_full`: 1D julia vector of length `nx * ny` with source/receiver locations in z direction.


## Data interpolation

Time interpolation for source/receiver data using splines. For modeling, the data is interpolated automatically onto the computational time axis, so generally, these functions are not needed for users.

```julia
data_interp, geometry_out = time_resample(data, geometry_in, dt_out; order=2)
```

**or**:

```julia
data_interp = time_resample(data, geometry_out, dt_in; order=2)
```

**Parameters:**

 * `data`: 2D Julia array of source/receiver data.

 * `geometry_in`: A JUDI `Geometry` object of the input data before interpolation.

 * `dt_out`: Sampling interval of interpolated shot record in milliseconds `[ms]`.

 * `order`: Order of splines for interpolation. 

**or:**
 
 * `data`: 2D Julia array of source/receiver data.

 * `geometry_out`: A JUDI `Geometry` object of the data after interpolation.

 * `dt_in`: Sampling interval of input shot record in milliseconds `[ms]`.


## Generate and sample from frequency distribution

Create a probability distribution with the shape of the source spectrum from which we can draw random frequencies.

```
dist = generate_distribution(q; src_no=1)
```

**Parameters:**

 * `q`: Source vector of type `judiVector` from which to create the distribution.

 * `src_no`: Source number for which to create the distribution (i.e. `q[src_no]`).

**Returns:**

 * `dist`: probability distribution. 
 
We can draw random samples from `dist` by passing it values between 0 and 1:

```julia
# Draw a single random frequency
f = dist(rand(1))

# Draw 10 random frequencies
f = dist(rand(10))
```

Alternatively, we can use the function:

```julia
f = select_frequencies(dist; fmin=0f0, fmax=Inf, nf=1)
```

to draw `nf` number of frequencies for a given distribution `dist` in the frequency range of `fmin` to `fmax` (both in kHz).