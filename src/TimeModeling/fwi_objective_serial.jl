
export fwi_objective

function fwi_objective(model_full::Model, source::judiVector, dObs::judiVector, srcnum::Integer; options=Options(), frequencies=[])
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
    if typeof(dObs.data[1]) == SeisIO.SeisCon
        data = convert(Array{Float32,2},dObs.data[1][1].data)
        dObs = judiVector(dObs.geometry,data)
    end
    dObserved = time_resample(dObs.data[1],dObs.geometry,dtComp)[1]
    ntComp = size(dObserved,2)
    ntSrc = Int(trunc(tmaxSrc/dtComp+1))
    ntRec = Int(trunc(tmaxRec/dtComp+1))

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n, model.o)  # shifts source coordinates by origin
    rec_coords = setup_grid(dObs.geometry, model.n, model.o)    # shifts rec coordinates by origin

    argout2 = compute_grad(model, modelPy, src_coords, rec_coords, qIn, dObserved, options)
    if options.limit_m==true
        argout2 = extend_gradient(model_full,model,argout2)
    end
    return [argout1; vec(argout2)]
end


function compute_grad(model, modelPy, src_coords, rec_coords, qIn, dObserved, options)
    ac = load_devito_jit(modelPy)

    if options.optimal_checkpointing == true
        op_F = pycall(ac."forward_modeling", PyObject, modelPy,
					  PyReverseDims(copy(transpose(src_coords))),
					  PyReverseDims(copy(transpose(qIn))),
					  PyReverseDims(copy(transpose(rec_coords))), op_return=true)
        argout1, argout2 = pycall(ac."adjoint_born", PyObject, modelPy,
								  PyReverseDims(copy(transpose(rec_coords))),
								  PyReverseDims(copy(transpose(dObserved))),
                                  op_forward=op_F, is_residual=false, free_surface=options.free_surface)
    elseif ~isempty(options.frequencies)
        typeof(options.frequencies) == Array{Any,1} && (options.frequencies = options.frequencies[srcnum])
        dPredicted, uf_real, uf_imag = pycall(ac."forward_freq_modeling", PyObject, modelPy,
											  PyReverseDims(copy(transpose(src_coords))),
											  PyReverseDims(copy(transpose(qIn))),
											  PyReverseDims(copy(transpose(rec_coords))),
                                              options.frequencies, space_order=options.space_order,
											  nb=model.nb, free_surface=options.free_surface)

        argout1 = .5f0*norm(vec(dPredicted) - vec(dObserved),2)^2.f0    # data misfit
        argout2 = pycall(ac."adjoint_freq_born", Array{Float32, length(model.n)}, modelPy,
						 PyReverseDims(copy(transpose(src_coords))),
						 PyReverseDims(copy(transpose(dPredicted - dObserved))),
                         options.frequencies, uf_real, uf_imag, space_order=options.space_order,
						 nb=model.nb, free_surface=options.free_surface)
    else
        dPredicted, u0 = pycall(ac."forward_modeling", PyObject, modelPy,
								PyReverseDims(copy(transpose(src_coords))),
								PyReverseDims(copy(transpose(qIn))),
								PyReverseDims(copy(transpose(rec_coords))),
								save=true, free_surface=options.free_surface)
    	# Data misfit
        if isempty(options.gs)
            # wrong misfit for trace based
            argout1 = misfit(dPredicted, dObserved, options.normalize)
            residual = adjoint_src(dPredicted, dObserved, options.normalize)
        else
            argout1 = misfit(dPredicted, dObserved, options.normalize; trace=options.gs["strategy"])
            residual = gs_residual(options.gs, dtComp, dPredicted, dObserved, options.normalize)
        end
        argout2 = pycall(ac."adjoint_born", Array{Float32}, modelPy,
						 PyReverseDims(copy(transpose(rec_coords))),
						 PyReverseDims(copy(transpose(residual))), u=u0,
						 is_residual=true, free_surface=options.free_surface)
    end
    argout2 = remove_padding(argout2, model.nb, true_adjoint=options.sum_padding)
	
	return argout2
end


function fwi_objective(model_full::Model_TTI, source::judiVector, dObs::judiVector, srcnum::Integer; options=Options(), frequencies=[])
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
    ac = load_devito_jit(modelPy)

    # Extrapolate input data to computational grid
    qIn = time_resample(source.data[1],source.geometry,dtComp)[1]
    if typeof(dObs.data[1]) == SeisIO.SeisCon
        data = convert(Array{Float32,2},dObs.data[1][1].data)
        dObs = judiVector(dObs.geometry,data)
    end
    dObserved = time_resample(dObs.data[1],dObs.geometry,dtComp)[1]
    ntComp = size(dObserved,2)
    ntSrc = Int(trunc(tmaxSrc/dtComp+1))
    ntRec = Int(trunc(tmaxRec/dtComp+1))

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n, model.o)  # shifts source coordinates by origin
    rec_coords = setup_grid(dObs.geometry, model.n, model.o)    # shifts rec coordinates by origin

    # Forward modeling to generate synthetic data and background wavefields
    if options.optimal_checkpointing == true
        op_F = pycall(ac."forward_modeling", PyObject, modelPy, PyReverseDims(copy(transpose(src_coords))), PyReverseDims(copy(transpose(qIn))), PyReverseDims(copy(transpose(rec_coords))), op_return=true)
        argout1, argout2 = pycall(ac."adjoint_born", PyObject, modelPy, PyReverseDims(copy(transpose(rec_coords))), PyReverseDims(copy(transpose(dObserved))),
                                  op_forward=op_F, is_residual=false, n_checkpoints=options.num_checkpoints, maxmem=options.checkpoints_maxmem)
    elseif ~isempty(options.frequencies)
                typeof(options.frequencies) == Array{Any,1} && (options.frequencies = options.frequencies[srcnum])
                dPredicted, uf_real, uf_imag = pycall(ac."forward_freq_modeling", PyObject, modelPy, PyReverseDims(copy(transpose(src_coords))), PyReverseDims(copy(transpose(qIn))), PyReverseDims(copy(transpose(rec_coords))),
                                                      options.frequencies, space_order=options.space_order, nb=model.nb, factor=options.dft_subsampling_factor)
                argout1 = .5f0*dot(vec(dPredicted) - vec(dObserved), vec(dPredicted) - vec(dObserved))*dtComp  # FWI objective function value
                argout2 = pycall(ac."adjoint_freq_born", Array{Float32, length(model.n)}, modelPy, PyReverseDims(copy(transpose(rec_coords))), PyReverseDims(copy(transpose((dPredicted - dObserved)))),
                                 options.frequencies, factor=options.dft_subsampling_factor, uf_real, uf_imag, space_order=options.space_order, nb=model.nb)
    else
        dPredicted, u0 = pycall(ac."forward_modeling", PyObject, modelPy, PyReverseDims(copy(transpose(src_coords))), PyReverseDims(copy(transpose(qIn))), PyReverseDims(copy(transpose(rec_coords))), save=true, tsub_factor=options.subsampling_factor)
        argout1 = .5f0*dot(vec(dPredicted) - vec(dObserved), vec(dPredicted) - vec(dObserved))*dtComp # FWI objective function value
        argout2 = pycall(ac."adjoint_born", Array{Float32}, modelPy, PyReverseDims(copy(transpose(rec_coords))), PyReverseDims(copy(transpose((dPredicted  - dObserved)))), tsub_factor=options.subsampling_factor,
                         u=u0, is_residual=true)
    end
    argout2 = remove_padding(argout2, model.nb, true_adjoint=options.sum_padding)
    if options.limit_m==true
        argout2 = extend_gradient(model_full,model,argout2)
    end
    u0 = []
    u0 = 0.
    u0 = []
    return [argout1; vec(argout2)]
end


function misfit(dPredicted, dObserved, normalize; trace="shot")
    if normalize
        if trace=="shot"
            obj = norm(vec(dPredicted)) - dot(vec(dPredicted),vec(dObserved))/norm(vec(dObserved))
        else
            indnz = [i for i in 1:size(dObserved, 2) if norm(dObserved[:, i])>0]
            obj = sum(norm(vec(dPredicted[:, i])) - dot(vec(dPredicted[:, i]),vec(dObserved[:, i]))/norm(vec(dObserved[:, i])) for i in indnz)
        end
    else
        obj = .5f0*norm(vec(dPredicted) - vec(dObserved),2)^2.f0
    end
end

function adjoint_src(dPredicted, dObserved, normalize)
    if normalize
        adj_src = dPredicted/norm(vec(dPredicted)) - dObserved/norm(vec(dObserved))
    else
        adj_src = dPredicted - dObserved
    end
end
