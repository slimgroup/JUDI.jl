# Generate observed Born data for Marmousi examples
# Author: pwitte.slim@gmail.com
# Date: December 2018
#

using JUDI, SegyIO, HDF5, LinearAlgebra

# Load migration velocity model
if ~isfile("$(JUDI.JUDI_DATA)/marmousi_model.h5")
    ftp_data("ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_model.h5")
end
n, d, o, m0, "dm" = read(h5open("$(JUDI.JUDI_DATA)/marmousi_model.h5", "r"), "n", "d", "o", "m0", "dm")

# Set up model structure
model = Model((n[1], n[2]), (d[1], d[2]), (o[1], o[2]), m0)
dm = vec(dm)

# Set up source geometry (cell array with source locations for each shot)
nsrc = 320
xsrc = convertToCell(range(25f0, stop=7965f0, length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(10f0, stop=10f0, length=nsrc))

# source sampling and number of time steps
timeS = 4000f0  # ms
dtS = 4f0   # ms

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

## Set up receiver geometry
nxrec = 800
max_offset = 3990
nsrc_half = Int(round(nsrc/2))

xrec = Array{Any}(undef, nsrc)
yrec = Array{Any}(undef, nsrc)
zrec = Array{Any}(undef, nsrc)

for j=1:nsrc_half
    xrec[j] = range(xsrc[j], stop=xsrc[j] + max_offset, length=nxrec)
    yrec[j] = range(0f0, stop=0f0, length=nxrec)
    zrec[j] = range(210f0, 210f0, length=nxrec)
end
for j=nsrc_half+1:nsrc
    xrec[j] = range(xsrc[j], stop=xsrc[j] - max_offset, length=nxrec)
    yrec[j] = range(0f0, stop=0f0, length=nxrec)
    zrec[j] = range(210f0, 210f0, length=nxrec)
end

# receiver sampling and recording time
timeR = 4000f0   # receiver recording time [ms]
dtR = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR)

# setup wavelet
f0 = 0.03f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# Setup operators
Pr = judiProjection(recGeometry)
F = judiModeling(model)
Ps = judiProjection(srcGeometry)
J = judiJacobian(Pr*F*adjoint(Ps), q)

# Born modeling
dobs = J*dm
block_out = judiVector_to_SeisBlock(dobs, q; source_depth_key="SourceDepth")
segy_write("marmousi_2D.segy", block_out)
