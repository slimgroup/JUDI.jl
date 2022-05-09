# Generate 2D data for the overthrust model
# Author: pwitte.slim@gmail.com
# Date: December 2018
#

using JUDI, HDF5, SegyIO

# Load overthrust model
if ~isfile("$(JUDI.JUDI_DATA)/overthrust_model_2D.h5")
    ftp_data("ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5")
end
n, d, o, m = read(h5open("$(JUDI.JUDI_DATA)/overthrust_model_2D.h5","r"), "n", "d", "o", "m")

# Set up model structure
model = Model((n[1], n[2]), (d[1], d[2]), (o[1], o[2]), m)

# Set up source geometry (cell array with source locations for each shot)
nsrc = 97
xsrc = convertToCell(range(400f0, stop=19600f0, length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(50f0, stop=50f0, length=nsrc))

# source sampling and number of time steps
timeS = 3000f0  # ms
dtS = 4f0   # ms

# Set up source structure
src_geometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

## Set up receivers
xmin = 100f0; xmax = 19900f0
offset = 6000f0
nxrec = 240
xrec = Array{Any}(undef, nsrc)
yrec = Array{Any}(undef, nsrc)
zrec = Array{Any}(undef, nsrc)
for j = 1: nsrc
	# 6 km max. offset around current source
    xsrc[j] - offset < xmin ? xlocal1 = xmin : xlocal1 = xsrc[j] - offset
    xsrc[j] + offset > xmax ? xlocal2 = xmax : xlocal2 = xsrc[j] + offset

    xrec[j] = xlocal1: 50f0: xlocal2
    yrec[j] = range(0f0, stop=0f0, length=length(xrec[j]))
    zrec[j] = range(500f0, stop=500f0, length=length(xrec[j]))
end
timeR = 3000f0	# receiver recording time [ms]
dtR = 4f0	# receiver sampling interval
rec_geometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR)

# Set up source
f0 = 0.008
wavelet = ricker_wavelet(timeS, dtS, f0)
#######################################################################

# Set up operators
Pr = judiProjection(rec_geometry)
Ps = judiProjection(src_geometry)
A_inv = judiModeling(model)
q = judiVector(src_geometry, wavelet)

# Generate data
d_obs = Pr*A_inv*adjoint(Ps)*q

block_out = judiVector_to_SeisBlock(d_obs, q; source_depth_key="SourceDepth")
segy_write("overthrust_2D.segy", block_out)
