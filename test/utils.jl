# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: July 2020

export setup_model, parse_commandline, setup_geom
export example_src_geometry, example_rec_geometry, example_model, example_info


"""
Simple 2D model setup used for the tests.
"""

function smooth(v, sigma=3)
    v0 = 1f0 .* v
    for i=-sigma:sigma
        i != 0 && (v0[:, sigma+1:end-sigma] .+= v[:, sigma+i+1:end-sigma+i])
    end
    v0[:, sigma+1:end-sigma] .*= 1/(2*sigma+1)
    return v0
end

"""
Sets up a simple 2D layered model for the wave equation operators tests
"""
function setup_model(tti=false, viscoacoustic=false, nlayer=2; n=(301, 151), d=(10., 10.))
    tti && viscoacoustic && throw("The inputs can't be tti and viscoacoustic at the same time")

    ## Set up model structure
    o = (0., 0.)
    lw = n[2] ÷ nlayer

    v = ones(Float32, n) .* 1.5f0
    vp_i = range(1.5f0, 3.5f0, length=nlayer)
    for i in range(2, nlayer, step=1)
        v[:, (i-1)*lw+ 1:end] .= vp_i[i]  # Bottom velocity
    end

    v0 = smooth(v, 7)
    rho0 = (v .+ .5f0) ./ 2
    # Slowness squared [s^2/km^2]
    m = (1f0 ./ v).^2
    m0 = (1f0 ./ v0).^2

    # Setup model structure
    if tti
        println("TTI Model")
        epsilon = smooth((v0[:, :] .- 1.5f0)/12f0, 3)
        delta =  smooth((v0[:, :] .- 1.5f0)/14f0, 3)
        theta =  smooth((v0[:, :] .- 1.5f0)/4, 3) .* rand([0, 1]) # makes sure VTI is tested
        model0 = Model(n, d, o, m0; epsilon=epsilon, delta=delta, theta=theta)
        model = Model(n, d, o, m; epsilon=epsilon, delta=delta, theta=theta)
    elseif viscoacoustic
        println("Viscoacoustic Model")
        qp0 = 3.516f0 .* ((v .* 1000f0).^2.2f0) .* 10f0^(-6f0)
        model = Model(n,d,o,m,rho0,qp0)
        model0 = Model(n,d,o,m0,rho0,qp0)
    else
        model = Model(n, d, o, m, rho0)
        model0 = Model(n, d, o, m0, rho0)
        @test Model(n, d, o, m0; rho=rho0).rho == model0.rho
    end
    dm = model.m - model0.m
    return model, model0, dm
end

"""
Sets up a simple 2D acquisition for the wave equation operators tests
"""

function setup_geom(model; nsrc=1, tn=1500f0)
    ## Set up receiver geometry
    nxrec = model.n[1] - 2
    xrec = collect(range(model.d[1], stop=(model.n[1]-2)*model.d[1], length=nxrec))
    yrec = collect(range(0f0, stop=0f0, length=nxrec))
    zrec = collect(range(50f0, stop=50f0, length=nxrec))

    # receiver sampling and recording time
    T = tn   # receiver recording time [ms]
    dt = .75f0    # receiver sampling interval [ms]

    # Set up receiver structure
    recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=T, nsrc=nsrc)

    ## Set up source geometry (cell array with source locations for each shot)
    ex =  (model.n[1] - 1) * model.d[1]
    nsrc > 1 ? xsrc = range(.25f0 * ex, stop=.75f0 * ex, length=nsrc) : xsrc = .5f0 * ex
    xsrc = convertToCell(xsrc)
    ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
    zsrc = convertToCell(range(2*model.d[2], stop=2*model.d[2], length=nsrc))

    # Set up source structure
    srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=T)

    # setup wavelet
    f0 = 0.015f0     # MHz
    wavelet = ricker_wavelet(T, dt, f0)
    q = judiVector(srcGeometry, wavelet)

    return q, srcGeometry, recGeometry, f0
end


### Process command line args
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--tti"
            help = "TTI, default False"
            action = :store_true
        "--viscoacoustic"
            help = "Viscoacoustic, default False"
            action = :store_true
        "--fs"
            help = "Free surface, default False"
            action = :store_true
        "--nlayer", "-n"
            help = "Number of layers"
            arg_type = Int
            default = 2
        "--parallel", "-p"
            help = "Number of workers"
            arg_type = Int
            default = 1
    end
    return parse_args(s)
end


# Example structures
example_model(; n=(120,100), d=(10f0, 10f0), o=(0f0, 0f0), m=randn(Float32, n)) = Model(n, d, o, m)

function example_rec_geometry(; nsrc=2, nrec=120, cut=false)
    startr = cut ? 150f0 : 50f0
    endr = cut ? 1050f0 : 1150f0
    xrec = range(startr, stop=endr, length=nrec)
    yrec = 0f0
    zrec = range(50f0, stop=50f0, length=nrec)
    return Geometry(xrec, yrec, zrec; dt=4f0, t=1000f0, nsrc=nsrc)
end

function example_src_geometry(; nsrc=2)
    xsrc = nsrc == 1 ? 500f0 : range(300f0, stop=900f0, length=nsrc)
    ysrc = nsrc == 1 ? 0f0 : zeros(Float32, nsrc)
    zsrc = range(50f0, stop=50f0, length=nsrc)
    return Geometry(convertToCell(xsrc), convertToCell(ysrc), convertToCell(zsrc); dt=4f0, t=1000f0)
end
