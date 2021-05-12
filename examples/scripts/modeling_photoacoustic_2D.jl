using JUDI, PyPlot

# Set up model structure
n = (64,64) 
d = (0.1f0,0.1f0) # in mm
o = (0.0f0,0.0f0)

# Velocity [mm/microsec]
v = 1.5*ones(Float32,(n...)) ; #constant water velocity

# Slowness squared [microsec^2/mm^2]
m = (1f0 ./ v).^2.0f0;

# Setup info and model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m;nb=80) 

nxrec = 64
xrec = range(0, stop=d[1]*(n[1]-1), length=nxrec)
yrec = [0f0]
zrec = range(0, stop=0, length=nxrec)

timeR = 4f0  # receiver sampling interval [micro sec]
dtR = 1.5f-3 # receiver sampling interval [micro sec]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# Set up info structure for linear operators
ntComp = get_computational_nt(recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

###################################################################################################
space_order = 16 #high space order to reduce dispersion
opt = Options(space_order=space_order, dt_comp=dtR)

# Setup operators
Pr = judiProjection(info, recGeometry)
F  = judiModeling(info, model;options=opt)
Pi = judiInitial(info)

A = Pr*F*adjoint(Pi)

# initial photoacoustic source distribution
PA_dist_0 = randn(Float32, model.n);
PA_dist_1 = randn(Float32, model.n);

PA_iv = judiInitialValue(PA_dist_0, PA_dist_1)

# photoacoustic modeling
dobs = A*PA_iv
PA_adj = A'*dobs #adjiont variable for photoacoustic is in PA_adj.firstValue

figure();
imshow(dobs.data[1], cmap="seismic", vmin = -200, vmax=200, aspect="auto")

figure();
imshow(PA_adj.firstValue',cmap="seismic")





