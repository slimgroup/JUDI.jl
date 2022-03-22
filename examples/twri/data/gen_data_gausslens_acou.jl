################################################################################
#
# Generate synthetic data for the BG Compass model
#
# Author: Mathias Loubout and Gabbrio Rizzuti
# Date: September 2020
################################################################################
### Module loading

using JUDI, JLD2

### Load true model
base_path = dirname(pathof(JUDI))*"/../examples/twri/"
@load string(base_path*"data/GaussLens.jld") n d o m

# Model

model_true = Model(n, d, o, m; nb=80)

### Acquisition geometrye
dt = 1f0
T = 2000f0 # total recording time [ms]

# Source wavelet
freq_peak = 0.006f0
wavelet = ricker_wavelet(T, dt, freq_peak)

# Sources
ix_src = 3:14:199
nsrc = length(ix_src)
iz_src = 3
x_src = convertToCell((ix_src.-1)*d[1])
y_src = convertToCell(range(0f0, stop = 0f0, length = nsrc))
z_src = convertToCell(range((iz_src-1)*d[2], stop = (iz_src-1)*d[2], length = nsrc))

# Receivers
ix_rcv = 3:199
iz_rcv = 199
nrcv = length(ix_rcv)
x_rcv = (ix_rcv.-1)*d[1]
y_rcv = 0f0
z_rcv = range((iz_rcv-1)*d[2], stop = (iz_rcv-1)*d[2], length = nrcv)


# Geometry structures
src_geom = Geometry(x_src, y_src, z_src; dt = dt, t = T)
rcv_geom = Geometry(x_rcv, y_rcv, z_rcv; dt = dt, t = T, nsrc = nsrc)

# Source function
fsrc = judiVector(src_geom, wavelet)

# Setup operators
F = judiModeling(model_true, src_geom,rcv_geom)
dat = F*fsrc

### Saving data
base_path = dirname(pathof(JUDI))*"/../examples/twri/"
@save string(base_path*"data/GaussLens_data_acou.jld") model_true fsrc dat
