# Test reciprocity/simultaneou sources
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: July 2021
#

parsed_args = parse_commandline()

# Set parallel if specified
nw = parsed_args["parallel"]
if nw > 1 && nworkers() < nw
   addprocs(nw-nworkers() + 1; exeflags=["--code-coverage=user", "--inline=no", "--check-bounds=yes"])
end

@everywhere using JOLI
@everywhere using JUDI, LinearAlgebra, Test, Distributed

### Model
model, model0, dm = setup_model(parsed_args["tti"], parsed_args["nlayer"]; n=(101, 101), d=(10., 10.))
n = model.n
nsrc = 4
nxrec = 8
q, srcGeometry, recGeometry, info = setup_geom(model; nsrc=nsrc, nxrec=nxrec, tn=500f0)
dt = srcGeometry.dt[1]

# Test if reciprocity works
@testset "Reciprocity test with $(parsed_args["nlayer"]) layers with isic: $(parsed_args["isic"]), free surface: $(parsed_args["fs"]) and tti: $(parsed_args["tti"])" begin

    ftol = 1f-3
    ntComp = get_computational_nt(srcGeometry, recGeometry, model)
    info = Info(prod(n), nsrc, ntComp)

    opt = Options(isic=parsed_args["isic"],free_surface=parsed_args["fs"])

    F = judiProjection(info, recGeometry)*judiModeling(info, model; options=opt)*adjoint(judiProjection(info, srcGeometry))

    dobs = F*q

    dreci, qreci = SrcRecReciprocal(dobs,q)
    info1 = Info(prod(n), nxrec, ntComp)
    Freci = judiProjection(info1, dreci.geometry)*judiModeling(info1, model; options=opt)*adjoint(judiProjection(info1, qreci.geometry))

    @test isapprox(Freci*qreci, dreci, rtol=ftol)

end

# Test computational simultaneous sources (superposition)
@testset "Simultaneous sources test with $(parsed_args["nlayer"]) layers with isic: $(parsed_args["isic"]), free surface: $(parsed_args["fs"]) and tti: $(parsed_args["tti"])" begin

    ftol = 1f-3
    ntComp = get_computational_nt(srcGeometry, recGeometry, model)
    info = Info(prod(n), nsrc, ntComp)

    opt = Options(isic=parsed_args["isic"],free_surface=parsed_args["fs"])

    F = judiProjection(info, recGeometry)*judiModeling(info, model; options=opt)*adjoint(judiProjection(info, srcGeometry))

    dobs = F*q

    nsimsrc = 8
    xsrc = [[q.geometry.xloc[i][1] for i = 1:nsrc] for j = 1:nsimsrc]
    ysrc = [[0.0f0] for j = 1:nsimsrc]
    zsrc = [[q.geometry.zloc[i][1] for i = 1:nsrc] for j = 1:nsimsrc]

    # Set up source structure
    src_geometry = Geometry(xsrc, ysrc, zsrc; dt=q.geometry.dt[1], t=q.geometry.t[1])

    # Set up random weights
    weights = randn(Float32,nsimsrc,nsrc)

    # Create wavelet
    wavelet = [q.data[1]*weights[k:k,:] for k = 1:nsimsrc]

    q_sim = judiVector(src_geometry, wavelet)
    data_sim = [sum(weights[k,:].*dobs.data) for k = 1:nsimsrc]
    dobs_sim = judiVector(Geometry(dobs.geometry.xloc[1],dobs.geometry.yloc[1],dobs.geometry.zloc[1]; dt=dobs.geometry.dt[1], t=dobs.geometry.t[1], nsrc=nsimsrc),data_sim)

    info1 = Info(prod(n), nsimsrc, ntComp)
    F_sim = judiProjection(info1, dobs_sim.geometry)*judiModeling(info1, model; options=opt)*adjoint(judiProjection(info1, q_sim.geometry))

    @test isapprox(F_sim*q_sim, dobs_sim, rtol=ftol)

end
