# Helper functions

JUDI provides numerous helper and utility functions need for seismic modeling and inversion.

```@contents
Pages = ["helper.md"]
```

## Ricker wavelet

Create a 1D Ricker wavelet:

```@docs
ricker_wavelet(tmax, dt, f0; t0=nothing)
```

## Compute CFL time stepping interval 

Calculate the time stepping interval based on the CFL condition

```@dcs
calculate_dt
```

## Compute number of computational time steps

Estimate the number of computational time steps. Required for calculating the dimensions of the matrix-free linear modeling operators:

```@docs
get_computational_nt
```

## Set up 3D acquisition grid

Helper function to create a regular acquisition grid for a 3D survey.

```julia
setup_3D_grid
```


## Data interpolation

Time interpolation for source/receiver data using splines. For modeling, the data is interpolated automatically onto the computational time axis, so generally, these functions are not needed for users.

```@docs
time_resample
```

## Generate and sample from frequency distribution

Create a probability distribution with the shape of the source spectrum from which we can draw random frequencies.

```@docs
generate_distribution
select_frequencies
```

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

## Read data from out of core container

In the case where a `judiVector` is out of core (points to a segy file) it is possible to convert it or part of it into an in core `judiVecor` with the `get_data` function.


```julia
d_ic = get_data(d_ooc, inds)
```

where `inds` is either a single index, a list of index or a range of index.

## Restrict model to acquisition

In practice, and in particular for marine data, the aperture of a single shot is much smaller than the full model size. We provide a function (automatically used when the option `limit_m` is set in [`Options`](@ref)) that limits the model to the acquisition area.

```@docs
limit_model_to_receiver_area
```

## Additional miscellanous utilities

```@docs
devito_model
setup_grid
remove_padding
convertToCell
process_input_data
reshape
transducer
as_vec
```