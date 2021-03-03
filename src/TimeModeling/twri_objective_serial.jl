
export twri_objective

function twri_objective(model_full::Model, source::judiVector, dObs::judiVector, y::Union{judiVector, Nothing},
                        srcnum::Integer; options=Options(), optionswri=TWRIOPtions())
    # Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito
    length(optionswri.eps) == 1 ? eps_loc=optionswri.eps : eps_loc=optionswri.eps[srcnum]

    # Load full geometry for out-of-core geometry containers
    typeof(dObs.geometry) == GeometryOOC && (dObs.geometry = Geometry(dObs.geometry))
    typeof(source.geometry) == GeometryOOC && (source.geometry = Geometry(source.geometry))

    # for 3D modeling, limit model to area with sources/receivers
    if options.limit_m == true # only supported for 3D
        model = deepcopy(model_full)
        model = limit_model_to_receiver_area(source.geometry,dObs.geometry,model,options.buffer_size)
    else
        model = model_full
    end

    # Source/receiver parameters
    tmaxSrc = source.geometry.t[1]
    tmaxRec = dObs.geometry.t[1]

    # Set up Python model structure (force origin to be zero due to current devito bug)
    # Set up Python model structure
    modelPy = devito_model(model, options)
    dtComp = modelPy.critical_dt

    # Extrapolate input data to computational grid
    qIn = time_resample(source.data[1],source.geometry,dtComp)[1]
    if typeof(dObs.data[1]) == SegyIO.SeisCon
        data = convert(Array{Float32,2},dObs.data[1][1].data)
        dObs = judiVector(dObs.geometry,data)
    end
    dObserved = time_resample(dObs.data[1], dObs.geometry, dtComp)[1]
    isnothing(y) ? Y = nothing : Y = time_resample(y.data[1], y.geometry, dtComp)[1]
    ntComp = size(dObserved, 1)
    ntSrc = Int(trunc(tmaxSrc/dtComp+1))
    ntRec = Int(trunc(tmaxRec/dtComp+1))

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(dObs.geometry, model.n)    # shifts rec coordinates by origin

    ac = load_devito_jit()
    ~isempty(options.frequencies) ? freqs = options.frequencies : freqs = nothing
    ~isempty(options.frequencies) ? (wfilt, freqs) =  filter_w(qIn, dtComp, freqs) : wfilt = nothing
    obj, gradm, grady = pycall(ac."wri_func", PyObject,
                               modelPy, src_coords, qIn, rec_coords, dObserved, Y,
                               t_sub=options.subsampling_factor, space_order=options.space_order,
                               grad=optionswri.params, grad_corr=optionswri.grad_corr, eps=eps_loc,
                               alpha_op=optionswri.comp_alpha, w_fun=optionswri.weight_fun,
                               freq_list=freqs, wfilt=wfilt)

    if (optionswri.params in [:m, :all])
        gradm = remove_padding(gradm, modelPy.padsizes; true_adjoint=options.sum_padding)
        options.limit_m==true && (gradm = extend_gradient(model_full, model, gradm))
        gradm = PhysicalParameter(gradm, model_full.d, model_full.o)
    end
    if ~isnothing(grady)
        ntRec > ntComp && (grady = [grady zeros(size(grady,1), ntRec - ntComp)])
        grady = time_resample(grady, dtComp, dObs.geometry)
        grady = judiVector(dObs.geometry, grady)
    end

    return obj, gradm, grady
end


function filter_w(qIn, dt, freqs)
    ff = FFTW.fftfreq(length(qIn), 1/dt)
    df = ff[2] - ff[1]
    inds = [findmin(abs.(ff.-f))[2] for f in freqs]
    DFT = joDFT(length(qIn); DDT=Float32)
    R = joRestriction(size(DFT,1), inds; RDT=Complex{Float32}, DDT=Complex{Float32})
    qfilt = DFT'*R'*R*DFT*qIn
    return qfilt, freqs
end