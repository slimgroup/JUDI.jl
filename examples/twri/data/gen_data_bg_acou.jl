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
@load string(base_path*"data//BGCompass_tti.jld") n d o m

# Model

model_true = Model(n, d, o, m; nb=80)

### Acquisition geometrye
dt = 4f0
T = 2500f0 # total recording time [ms]

# Source wavelet
freq_peak = 0.01f0
wavelet = ricker_wavelet(T, dt, freq_peak)

# Sources
nsrc = 51
x_src = convertToCell(range(0f0, stop = (n[1]-1)*d[1], length = nsrc))
y_src = convertToCell(range(0f0, stop = 0f0, length = nsrc))
z_src = convertToCell(range(0f0, stop = 0f0, length = nsrc))

# Receivers
nrcv = n[1]
x_rcv = range(0f0, stop = (n[1]-1)*d[1], length = nrcv)
y_rcv = 0f0
z_rcv = range(12.5f0, stop = 12.5f0, length = nrcv)

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
@save string(base_path*"data/BGCompass_data_acou.jld") model_true fsrc dat
