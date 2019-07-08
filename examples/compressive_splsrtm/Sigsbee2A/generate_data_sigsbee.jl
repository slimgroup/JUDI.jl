# Generate observed linearized data for the Sigsbee2A velocity model
# Author: Philipp Witte, pwitte.slim@gmail.com
# Date: May 2018
#

# TO DO:
# Replace w/ full path the directory where the observed data will be stored
file_path="/path/to/data/"
file_name="sigsbee2A_marine"

using JUDI.TimeModeling, PyPlot, JLD, SeisIO

# Load Sigsbee model
if !isfile("sigsbee2A_model.jld")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/CompressiveLSRTM/sigsbee2A_model.jld`)
end
M = load("sigsbee2A_model.jld")

# Setup info and model structure
model0 = Model(M["n"], M["d"], M["o"], M["m0"])
dm = vec(M["dm"])

# Source interval [m]
source_spacing = 25

## Set up receiver geometry
domain_x = (model0.n[1] - 1)*model0.d[1]
nrec = 1200     # no. of receivers
xmin = 500f0
xmax = domain_x - 500f0
min_offset = 100f0  # [m]
max_offset = 12000f0
xmid = domain_x / 2

# Source/receivers
nsrc = Int(length(xmin:source_spacing:xmax))
xrec = Array{Any}(undef, nsrc)
yrec = Array{Any}(undef, nsrc)
zrec = Array{Any}(undef, nsrc)
xsrc = Array{Any}(undef, nsrc)
ysrc = Array{Any}(undef, nsrc)
zsrc = Array{Any}(undef, nsrc)

# Vessel goes from left to right in right-hand side of model
nsrc_half = Int(nsrc/2)
for j=1:nsrc_half
    xloc = xmid + (j-1)*source_spacing
    xrec[j] = range(xloc - max_offset, xloc - min_offset, length=nrec)
    yrec[j] = 0.
    zrec[j] = range(50f0, 50f0, length=nrec)
    xsrc[j] = xloc
    ysrc[j] = 0f0
    zsrc[j] = 20f0
end

# Vessel goes from right to left in left-hand side of model
for j=1:nsrc_half
    xloc = xmid - (j-1)*source_spacing
    xrec[nsrc_half + j] = range(xloc + min_offset, xloc + max_offset, length=nrec)
    yrec[nsrc_half + j] = 0f0
    zrec[nsrc_half + j] = range(50f0, 50f0, length=nrec)
    xsrc[nsrc_half + j] = xloc
    ysrc[nsrc_half + j] = 0f0
    zsrc[nsrc_half + j] = 20f0
end

# receiver sampling and recording time
timeR = 10000f0   # receiver recording time [ms]
dtR = 4f0    # receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR)

# Source sampling and number of time steps
timeS = 10000f0
dtS = 2f0

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

# Setup wavelet
f0 = 0.015  # dominant frequency in [kHz]
wavelet = ricker_wavelet(timeS,dtS,f0)
q = judiVector(srcGeometry,wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry,recGeometry,model0)
info = Info(prod(model0.n),nsrc,ntComp)

#################################################################################################

opt = Options(isic=true,    # impedance modeling
              save_data_to_disk=true,
              file_path = file_path,  # directory for saving generated shots
              file_name = file_name
              )

# Setup operators
Pr = judiProjection(info, recGeometry)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, srcGeometry)
J = judiJacobian(Pr*F0*Ps', q)

# Linearized modeling (shots written to disk as SEG-Y files automatically)
J*dm
