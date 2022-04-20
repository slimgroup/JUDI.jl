# Generate 3D data for the overthrust model
# Author: pwitte.slim@gmail.com
# Date: January 2017
#

using JUDI, HDF5

# Load overthrust model
if ~isfile("$(JUDI.JUDI_DATA)/overthrust_3D_true_model.h5")
    ftp_data("ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_true_model.h5")
end
n, d, o, m = read(h5open("$(JUDI.JUDI_DATA)/overthrust_3D_true_model.h5","r"), "n", "d", "o", "m")

# Set up model structure
model = Model((n[1], n[2], n[3]), (d[1], d[2], d[3]), (o[1], o[2], o[3]), m)

# Set up source grid
nsrc = 97^2
xsrc = 400f0: 200f0: 19600f0	# 400 m space from boundary
ysrc = 400f0: 200f0: 19600f0
zsrc = 50f0
(xsrc, ysrc, zsrc) = setup_3D_grid(xsrc, ysrc, zsrc)
xsrc = convertToCell(xsrc)
ysrc = convertToCell(ysrc)
zsrc = convertToCell(zsrc)
timeS = 3000f0
dtS = 4f0
src_geometry = Geometry(xsrc, ysrc, zsrc; dt = dtS, t = timeS)

## Set up receivers
xmin = 100f0; xmax = 19900f0
ymin = 100f0; ymax = 19900f0
offset = 6000f0
nxrec = 240
nyrec = 240
xrec = Array{Any,1}(undef, nsrc)
yrec = Array{Any,1}(undef, nsrc)
zrec = Array{Any}(undef, nsrc)
for j = 1: nsrc
	# 6 km max. offset around current source
    xsrc[j] - offset < xmin ? xlocal1 = xmin : xlocal1 = xsrc[j] - offset
    xsrc[j] + offset > xmax ? xlocal2 = xmax : xlocal2 = xsrc[j] + offset
	ysrc[j] - offset < ymin ? ylocal1 = ymin : ylocal1 = ysrc[j] - offset
	ysrc[j] + offset > ymax ? ylocal2 = ymax : ylocal2 = ysrc[j] + offset
	xlocal = xlocal1: 50f0: xlocal2
	ylocal = ylocal1: 50f0: ylocal2
	zlocal = 500f0
	(xlocal,ylocal,zlocal) = setup_3D_grid(xlocal, ylocal, zlocal)
	xrec[j] = xlocal
	yrec[j] = ylocal
	zrec[j] = zlocal
end
timeR = 3000f0	# receiver recording time [ms]
dtR = 4f0	# receiver sampling interval
rec_geometry = Geometry(xrec, yrec, zrec; dt = dtR, t = timeR)

# Set up source
f0 = 0.008
wavelet = ricker_wavelet(timeS, dtS, f0)

#######################################################################

# Set up modeling options
opt = Options(limit_m = true,
			  save_data_to_disk = true,
			  file_path = "path/to/shot/records",   # replace w/ path to directory in which data will be stored (~total of 1.7 TB)
 			  file_name = "overthrust_3D_shot_"   # adds x-y coordinates to name
              )

# Set up operators
Pr = judiProjection(rec_geometry)
Ps = judiProjection(src_geometry)
A_inv = judiModeling(model; options=opt)
q = judiVector(src_geometry, wavelet)

# Generate data and save as individual SEG-Y files to disk
d_obs = Pr*A_inv*adjoint(Ps)*q
