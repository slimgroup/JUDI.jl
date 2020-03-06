
export fwi_objective

function fwi_objective(model_full::Modelall, source::judiVector, dObs::judiVector, srcnum::Integer; options=Options(), frequencies=[])
# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito

    # Load full geometry for out-of-core geometry containers
    typeof(dObs.geometry) == GeometryOOC && (dObs.geometry = Geometry(dObs.geometry))
    typeof(source.geometry) == GeometryOOC && (source.geometry = Geometry(source.geometry))
    length(model_full.n) == 3 ? dims = [3,2,1] : dims = [2,1]   # model dimensions for Python are (z,y,x) and (z,x)

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
    modelPy = devito_model(model, "F", 1, options, nothing)
    dtComp = modelPy.critical_dt

    # Extrapolate input data to computational grid
    qIn = time_resample(source.data[1],source.geometry,dtComp)[1]
    if typeof(dObs.data[1]) == SegyIO.SeisCon
        data = convert(Array{Float32,2},dObs.data[1][1].data)
        dObs = judiVector(dObs.geometry,data)
    end
    dObserved = time_resample(dObs.data[1],dObs.geometry,dtComp)[1]
    ntComp = size(dObserved,2)
    ntSrc = Int(trunc(tmaxSrc/dtComp+1))
    ntRec = Int(trunc(tmaxRec/dtComp+1))

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(dObs.geometry, model.n)    # shifts rec coordinates by origin

    ac = load_devito_jit(model)

    if options.optimal_checkpointing == true
        op_F = pycall(ac."forward_modeling", PyObject, modelPy, src_coords, qIn, rec_coords, op_return=true)
        argout1, argout2 = pycall(ac."adjoint_born", PyObject, modelPy, rec_coords, dObserved, op_forward=op_F, is_residual=false, n_checkpoints=options.num_checkpoints, maxmem=options.checkpoints_maxmem)
    elseif ~isempty(options.frequencies)
                typeof(options.frequencies) == Array{Any,1} && (options.frequencies = options.frequencies[srcnum])
                dPredicted, uf_real, uf_imag = pycall(ac."forward_freq_modeling", PyObject, modelPy, src_coords, qIn, rec_coords, options.frequencies, space_order=options.space_order, nb=model.nb, factor=options.dft_subsampling_factor)
                argout1 = .5f0*dot(vec(dPredicted) - vec(dObserved), vec(dPredicted) - vec(dObserved))*dtComp  # FWI objective function value
                argout2 = pycall(ac."adjoint_freq_born", Array{Float32, length(model.n)}, modelPy, rec_coords, (dPredicted - dObserved), options.frequencies, factor=options.dft_subsampling_factor, uf_real, uf_imag, space_order=options.space_order, nb=model.nb)
    else
        dPredicted, u0 = pycall(ac."forward_modeling", PyObject, modelPy, src_coords, qIn, rec_coords, save=true, tsub_factor=options.subsampling_factor, return_devito_obj=true)
        argout1 = .5f0*dot(vec(dPredicted) - vec(dObserved), vec(dPredicted) - vec(dObserved))*dtComp # FWI objective function value
        argout2 = pycall(ac."adjoint_born", Array{Float32}, modelPy, rec_coords, (dPredicted  - dObserved), tsub_factor=options.subsampling_factor, u=u0, is_residual=true)
    end
    argout2 = remove_padding(argout2, model.nb, true_adjoint=options.sum_padding)
    if options.limit_m==true
        argout2 = extend_gradient(model_full, model, argout2)
    end
    return [argout1; vec(argout2)]
end
