# Input/Output

For reading and writing SEG-Y data, JUDI uses the [SegyIO.jl](https://github.com/slimgroup/SegyIO.jl) package. JUDI supports reading SEG-Y from disk into memory, as well as working with out-of-core (OOC) data containers. In the latter case, `judiVectors` contain look-up tables that allow accessing the underlying data in constant time.

## Reading SEG-Y files into memory

To read a single SEG-Y file into memory, use the `segy_read` function:

```julia
using SegyIO

block = segy_read("data.segy")
```

From a `SegyIO` data block, you can create an in-core `judiVector`, as well as a `Geometry` object for the source:

```
# judiVector for observed data
d_obs = judiVector(block; segy_depth_key="RecGroupElevation")

# Source geometry
src_geometry = Geometry(block; key="source", segy_depth_key="SourceDepth")
```

The optional keyword `segy_depth_key` specifies which SEG-Y header stores the depth coordinate. After reading a `block`, you can check `block.traceheaders` to see which trace headers are set and where to find the depth coordinates for sources or receivers.

The `d_obs` vector constains the receiver geometry in `d_obs.geometry`, so there is no need to set up a separate geometry object manually. However, in principle we can set up a receiver `Geometry` object as follows:

```
rec_geometry = Geometry(block; key="receiver", segy_depth_key="RecGroupElevation")
```


## Writing SEG-Y files

To write a `judiVector` as a SEG-Y file, we need a `judiVector` containing the receiver data and geometry, as well as a `judiVector` with the source coordinates. From the `judiVectors`, we first create a `SegyIO` block:

```julia
block = judiVector_to_SeisBlock(d_obs, q)
```

where `d_obs` and `q` are `judiVectors` for receiver and source data respectively. To save only the source `q`, we can do

```julia
block = src_to_SeisBlock(q)
```

Next, we can write a SEG-Y file from a `SegyIO block`:

```julia
segy_write("new_file.segy", block)  # writes a SEG-Y file called new_file.segy
```

## Reading out-of-core SEG-Y files

For SEG-Y files that do not fit into memory, JUDI provides the possibility to work with OOC data containers. First, `SegyIO` scans also available files and then creates a lookup table, including a summary of the most important SEG-Y header values. See `SegyIO's` [documentation](https://github.com/slimgroup/SegyIO.jl/wiki/Scanning) for more information.

First we provide the path to the directory that we want to scan, as well as a string that appears in all the files we want to scan. For example, here we want to scan all files that contain the string `"bp_observed_data"`. The third argument is a list of SEG-Y headers for which we create a summary. For creating OOC `judiVectors`, **always** include the `"GroupX"`, `"GroupY"` and `"dt"` keyworkds, as well as the keywords that carry the source and receiver depth coordinates:

```julia
# Specify direcotry to scan
path_to_data = "/home/username/data_directory/"

# Scan files in given directory and create OOC data container
container = segy_scan(path_to_data, "bp_observed_data", ["GroupX", "GroupY", 
    "RecGroupElevation", "SourceDepth", "dt"])
```

Depending of the number and size of the underlying files, this process can take multiple hours, but it only has to be executed once! Furthermore, [parallel scanning](https://github.com/slimgroup/SegyIO.jl/wiki/Scanning) is supported as well. 

Once we have scanned all files in the directory, we can create an OOC `judiVector` and source `Geometry` object as follows:

```julia
# Create OOC judiVector
d_obs = judiVector(container; segy_depth_key="RecGroupElevation")

# Create OOC source geometry object
src_geometry = Geometry(container; key="source", segy_depth_key="SourceDepth")
```

## Reading and writing velocity models

JUDI does not require velocity models to be read or saved in any specific format. Any file format that allows reading the velocity model as a two or three-dimensional Julia array will work.

In our examples, we often use the [JLD](https://github.com/JuliaIO/JLD.jl) or [HDF5](https://github.com/JuliaIO/HDF5.jl) packages to read/write velocity models and the corresponing meta data (i.e. grid spacings and origins). If your model is a SEG-Y file, use the `segy_read` function from `SegyIO` as shown above.

 * Create an example model to write and read:

```julia
n = (120, 100)
d = (10.0, 10.0)
o = (0.0, 0.0)
v = ones(Float32, n) .* 1.5f0
m = 1f0 ./ v.^2
```

 * Write a model as a `.jld` file:

```julia
using JLD

save("my_model.jld", "n", n, "d", d, "o", o, "m", m)
```

 * Read a model from a `.jld` file:
 
```julia
# Returns a Julia dictionary
M = load("my_model.jld")

n = M["n"]
d = M["d"]
o = M["o"]
m = M["m"]

# Set up a Model object
model = Model(n, d, o, m)
```

