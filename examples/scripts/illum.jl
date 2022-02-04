# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# February 2022

using JUDI, HDF5, SlimPlotting, PyPlot, LinearAlgebra, Downloads

# Load model
data_path = dirname(pathof(JUDI))*"/../data/"
# Load migration velocity model
if ~isfile(data_path*"marmousi_model.h5")
    Downloads.download("ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_model.h5", data_path*"marmousi_model.h5")
end
n, d, o, m0, m = read(h5open(data_path*"marmousi_model.h5","r"), "n", "d", "o", "m0", "m")

# Subsample tso that runs ok with ci
fact = 3
m = m[1:fact:end, 1:fact:end]
m0 = m0[1:fact:end, 1:fact:end]
n = size(m)
d = (fact*d[1], fact*d[2]) # Base is 5m spacing
# Setup info and model structure
nsrc = 3	# number of sources
model = Model(n, d, o, m)
model0 = Model(n, d, o, m0)

# Set up receiver geometry
nxrec = n[1]
xrec = range(0f0, stop=(n[1] - 1)*d[1], length=nxrec)
yrec = 0f0
zrec = range(d[end], stop=d[end], length=nxrec)

# receiver sampling and recording time
td = 3000f0   # receiver recording time [ms]
dtd = 2f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtd, t=td, nsrc=nsrc)

# Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(range(0f0, stop=(n[1] - 1)*d[1], length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(d[2], stop=d[2], length=nsrc))

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtd, t=td)

# setup wavelet
f0 = 0.015f0     # kHz
wavelet = ricker_wavelet(td, dtd, f0)
q = judiVector(srcGeometry, wavelet)

###################################################################################################

# Write shots as segy files to disk
opt = Options(isic=true)

# Setup operators
F = judiModeling(model, srcGeometry, recGeometry; options=opt)
F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
J = judiJacobian(F0, q)

# Nonlinear modeling
dobs = F*q
d0 = F0*q


I = inv(judiIllumination(J))

rtm = J'*(d0 - dobs)

# Plot illum
c1, c2 = maximum(I.illum["u"])/10, maximum(I.illum["v"])/2
figure()
subplot(311)
imshow(I.illum["u"].data', cmap="gist_ncar", vmin=0, vmax=c1, aspect="auto")
subplot(312)
imshow(I.illum["v"].data', cmap="gist_ncar", vmin=0, vmax=c2, aspect="auto")
subplot(313)
imshow(I.illum["u"].data' .* I.illum["v"].data', cmap="gist_ncar", vmin=0, vmax=c1*c2, aspect="auto")
savefig("Illums.png", bbox_inches="tight")

# Plot rtm with corrections
figure(figsize=(10, 5))
subplot(221)
plot_simage(rtm.data', rtm.d; new_fig=false, name="rtm", aspect="auto")
subplot(222)
plot_simage((I*rtm).data', rtm.d; new_fig=false, name="rtmu", aspect="auto")
subplot(223)
plot_simage((I("v")*rtm).data', rtm.d; new_fig=false, name="rtmv", aspect="auto")
subplot(224)
plot_simage((I("uv")*rtm).data', rtm.d; new_fig=false, name="rtmuv", aspect="auto")
tight_layout()
savefig("rtms.png", bbox_inches="tight")
