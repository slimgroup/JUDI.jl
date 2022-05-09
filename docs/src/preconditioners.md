# Seismic Preconditioners

## Model topmute

Create a linear operator for a 2D model topmute, i.e. for muting the water column:

```julia
Mr = judiTopmute(n, mute_start, length)
```

**Parameters:**

 * `n`: Tuple of model dimensions (e.g. from `model.n`)

 * `mute_start`: First grid point in z-direction from where on to mute the image. Can be a single integer or a vector of length `nx`, where `nx` is the number of grid points in x direction.

 * `length`: The mask is created with a linear taper from 0 to 1. The width of the taper is `length`.

**Usage:**

```julia
# Forward
m_mute = Mr*vec(m)

# Adjoint
m_mute = Mr'*vec(m)
```

As `Mr` is self adjoint, `Mr` is equal to `Mr'`.

## Model depth scaling

Create a 2D model depth scaling:

```julia
Mr = judiDepthScaling(model)
```

**Parameters:**

 * `model`: JUDI `Model` structure.


## Data topmute (experimental)

Create a data topmute for a 2D marine shot record (i.e. for a shot record with an end-on-spread acquisition geometry).

```julia
Ml = judiMarineTopmute2D(muteStart, geometry; flipmask=false)
```

**Parameters:**

 * `muteStart`: Vertical index of the apex of the shot record (i.e. the earliest point from where to mute).

 * `geometry`: A JUDI `Geometry` object with the receiver geometry.

 * `flipmask`: If the source is on the left side, set to `false` (default). If the source is on the right side, set to `true`.



