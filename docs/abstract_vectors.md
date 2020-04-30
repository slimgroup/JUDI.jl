# Abstract JUDI vectors

## judiVector

The class `judiVector` is the basic data structure for seismic shot records or seismic sources. From JUDI's perspective, both are treated the same and can be multiplied with modeling operators.

**Construction:**

In the most basic way, `judiVectors` are contstructed from a `Geometry` object (containing either source or receiver geometry) and a cell array of data:

```julia
x = judiVector(geometry, data)
```

**Parameters:**

 * `geometry`: A `Geometry` object containing source or receiver geometries.

 * `data`: A cell array with one cell per source location, where each cell contains a 1D/2D Julia array with either the receiver data or the source wavelet. Alternatively: pass a single Julia array which will be used for all source locations.


**Access fields (in-core data containers):**

```julia
# Access i-th shot record
x.data[i]

# Extract judiVector for i-th shot
x1 = x[i]

# Access j-th receiver location of i-th shot
x.geometry.xloc[i][j]
```

**Access fields (out-of-core data containers):**

```julia
# Access data container of i-th shot
x.data[i]

# Read data from i-th shot into memory
x.data[i][1].data

# Access out-of-core geometry
x.geometry

# Load OOC geometry into memory
Geometry(x.geometry)
```

**Operations:**

In-core `judiVectors` can be used like regular Julia arrays and support common operations such as:


```julia
x = judiVector(geometry, data)

# Size (as if all data was vectorized)
size(x)

# Norms
norm(x)

# Inner product
dot(x, x)

# Addition, subtraction (geometries must match)
y = x + x
z = x - y

# Scaling
α = 2f0
y = x * α

# Concatenate
y = vcat(x, x)
```


## judiWavefield

Abstract vector class for wavefields. 

**Construction:**

```julia
u = judiWavefield(info, dt, data)
```

**Parameters:**

 * `info`: An `Info` structure.

 * `dt`: Time sampling interval of wavefield.

 * `data`: Cell array with one cell per source location. Each cell contains a 3D or 4D array for a seismic wavefield. The order of dimensions is `(nt, nx, nz)` (2D) and `(nt, nx, ny, nz)` (3D), where `nt` is the number of time steps.


**Access fields:**

```julia
# Access wavefield from i-th shot location
u.data[i]
```

**Operations:**

Supports some basic arithmetric operations:

```julia
# Size 
size(u)

# Norms
norm(u)

# Inner product 
dot(u, y)

# Addition, subtraction
v = u + u
z = u - v

# Absolute value
abs(u)

# Concatenation
v = vcat(u, u)
```

## judiRHS

Abstract vector class for a right-hand-side (RHS). A RHS has the size of a full wavefield, but only contains the data of the source wavelet of shot records in memory, as well as the geometry information of where the data is injected during modeling.

**Construction:**

```julia
rhs = judiRHS(info, geometry, data)
```

A JUDI RHS can also be constructed by multplying a `judiVector` and the corresponding transpose of a `judiProjection` operator:

```julia
rhs1 = Ps'*q
rhs2 = Pr'*d_obs
```

where `Ps` and `Pr` are `judiProjection` operators for sources and receivers respectively and `q` and `d_obs` are `judiVectors` with the source and receiver data.

**Parameters:**

 * `info`: An `Info` structure.

 * `geometry`: A JUDI `Geometry` structure, containing the source or receiver geometry.

 * `data`: A cell array with one cell per source location. Each cell contains a 1D/2D Julia array with the source or receiver data.

 **Access fields:**

Accessible fields include:

```julia
# Source/receiver data
rhs.data

# Source/receiver geometry
rhs.geometry

# Info structure
rhs.info
```

## judiWeights

Abstract vector class for extended source weights. The weights for each shot location have the dimensions of the model (namely `model.n`).

**Construction:**

```julia
w = judiWeights(weights)
```

**Parameters:**

 * `weights`: Cell array with one cell per shot location. Each cell contains a 2D/3D Julia array with the weights for the spatially extended source. Alternatively: pass a single Julia array which will be used for all source locations.

**Access fields:**

```julia
# Access weights of i-th shot locatoin
w.weights[i]
```

**Operations:**

Supports the same arithmetric operations as a `judiVector`.


## judiExtendedSource

Abstract data vector for an extended source. This data structure is the equivalent type of `judiRHS` for extended source modeling. A `judiExtendedSource` has the dimension of the full wavefield, but only contains the 1D wavelet and the 2D/3D spatially varying weights in memory.

**Construction:**

Construction from weights and source wavelets:

```julia
ex_src = judiExtendedSource(info, wavelet, weights)
```

Construction from a `judiWeights` vector and a `judiLRWF` injection operator:

```julia
ex_src = Pw'*w
```

where `Pw` is a `judiLRWF` operator and `w` is a `judiWeights` vector.

**Parameters:**

 * `info`: An `Info` structure.

 * `wavelet`: A cell array with one cell per source location containing a 1D Julia array with the time varying source wavelet **or** a single 1D Julia array, which is used for all source locations.

 * `weights`: A cell array with one cell per source location containing a 2D/3D Julia array with the spatially varying source weights **or** a single 1D Julia array, which is used for all source locations.

**Access fields:**

```julia
# Access weights of i-th source location
ex_src.weights[i]

# Access wavelet of i-th source location
ex_src.wavelet[i]
```
