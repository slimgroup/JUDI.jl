# Data structures


```@contents
Pages = ["data_structures.md"]
```

## Physical Parameter

Data structure for physical parameter array in JUDI. A `PhysicalParameter` inherits from julia `AbstractVector`

```@docs
PhysicalParameter
```

Unless specified otherwise with the `return_array` option in [`Options`](@ref), the result of a migration/FWIgradient([`judiJacobian`](@ref), [`fwi_objective`](@ref), [`lsrtm_objective`](@ref)) will be wrapped into a `PhysicalParameter`. THis allow better handling of different model parts and a better representation of the dimensional array.

## Model structure

Data structure for velocity models in JUDI.

```@docs
Model
```

Accessible fields include all of the above parameters `p.n, p.d, p.o, p.data`. Additionaly, arithmetic operation are all impemented such as addition, multiplication, broadcasting and indexing. Linear algebra operation are implemented as well but will return a standard Julia vector if the matrix used is external to JUDI.

**Access fields:**

Accessible fields include all of the above parameters, which can be accessed as follows:

```julia
# Access model
model.m

# Access number of grid points
model.n
```

## Geometry structure

JUDI's geometry structure contains the information of either the source **or** the receiver geometry. Construct an (in-core) geometry object for **either** a source or receiver set up:

```@docs
 Geometry
```

From the optional arguments, you have to pass (at least) **two** of `dt`, `nt` and `t`. The third value is automatically determined and set from the two other values. a `Geometry` can be constructed in a number of different ways for in-core and out-of-core cases. Check our examples and the source for additional details while the documentation is being extended.

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

### Geometry utilities

A few utilities to manipulates geometries are provided as well.

```@docs
super_shot_geometry
reciprocal_geom
```
## Options structure

The options structure allows setting several modeling parameters.

```@docs
Options
```

**notes**

`Option` has been renamed `JUDIOptions` as of JUDI version `4.0` to avoid potential (and exisiting) conflicts wiht other packages exporting an Options structure.