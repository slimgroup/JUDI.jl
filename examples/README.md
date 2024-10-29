# JUDI Examples

This directory contains examples of how to use JUDI for modeling and inversion, as well as code to reproduce results from journal papers.

## Required packages

The examples require some additional Julia packages. These packages are not required for the core package, but need to be installed in order to run the following examples. You can install the packages by running the following code from the Julia terminal:

```
using Pkg

# IO
Pkg.add("HDF5")
Pkg.add("JLD")
Pkg.add("JLD2")

# Plotting
Pkg.add("PythonPlot")
Pkg.add("SlimPlotting")

# Optimization
Pkg.add("NLopt")
Pkg.add("IterativeSolvers")
Pkg.add("Optim")
Pkg.add("LineSearches")
```

## Overview

 * Generic JUDI examples can be found in [scripts](https://github.com/slimgroup/JUDI.jl/tree/master/examples/scripts)

 * Jupyter notebooks for FWI can be found in [notebooks](https://github.com/slimgroup/JUDI.jl/tree/master/examples/notebooks)

 * Reproducable examples for *A large-scale framework for symbolic implementations of seismic inversion algorithms in Julia* are available in [software_paper](https://github.com/slimgroup/JUDI.jl/tree/master/examples/software_paper)

 * Reproducable examples for *Compressive least squares migration with on-the-fly Fourier transforms* are available in [compressive_splsrtm](https://github.com/slimgroup/JUDI.jl/tree/master/examples/compressive_splsrtm)

 * Examples related to *A dual formulation of wavefield reconstruction inversion for large-scale seismic inversion* are available in [twri](https://github.com/slimgroup/JUDI.jl/tree/master/examples/twri)

## References

 * Philipp A. Witte, Mathias Louboutin, Navjot Kukreja, Fabio Luporini, Michael Lange, Gerard J. Gorman and Felix J. Herrmann. A large-scale framework for symbolic implementations of seismic inversion algorithms in Julia. GEOPHYSICS, Vol. 84 (3), pp. F57-F71, 2019. <https://library.seg.org/doi/abs/10.1190/geo2018-0174.1>

 * Philipp A. Witte, Mathias Louboutin, Fabio Luporini, Gerard J. Gorman and Felix J. Herrmann. Compressive least-squares migration with on-the-fly Fourier transforms. GEOPHYSICS, vol. 84 (5), pp. R655-R672, 2019. <https://library.seg.org/doi/abs/10.1190/geo2018-0490.1>


 * Gabrio Rizzuti, Mathias Louboutin, Rongrong Wang, and Felix J. Herrmann. A dual formulation of wavefield reconstruction inversion for large-scale seismic inversion. Submitted to Geophysics. <https://slim.gatech.edu/content/dual-formulation-wavefield-reconstruction-inversion-large-scale-seismic-inversion>
