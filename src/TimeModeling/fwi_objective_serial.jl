
export fwi_objective

function fwi_objective(model_full::Model, source::judiVector, dObs::judiVector, srcnum::Integer; options=Options(), frequencies=[])
# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito
    pm = load_pymodel()
    ac = load_acoustic_codegen()

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
    modelPy = pm[:Model](origin=(0.,0.,0.), spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml=model.nb,
                       rho=process_physical_parameter(model.rho, dims), space_order=options.space_order)
    dtComp = modelPy[:critical_dt]

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
        op_F = pycall(ac[:forward_modeling], PyObject, modelPy, PyReverseDims(copy(transpose(src_coords))), PyReverseDims(qIn'), PyReverseDims(copy(transpose(rec_coords))), op_return=true)
        argout1, argout2 = pycall(ac[:adjoint_born], PyObject, modelPy, PyReverseDims(copy(transpose(rec_coords))), PyReverseDims(dObserved'),
                                  op_forward=op_F, is_residual=false, freesurface=options.freesurface)
    elseif ~isempty(options.frequencies)
        typeof(options.frequencies) == Array{Any,1} && (options.frequencies = options.frequencies[srcnum])
        dPredicted, uf_real, uf_imag = pycall(ac[:forward_freq_modeling], PyObject, modelPy, PyReverseDims(copy(transpose(src_coords))), PyReverseDims(qIn'), PyReverseDims(copy(transpose(rec_coords))),
                                              options.frequencies, space_order=options.space_order, nb=model.nb, freesurface=options.freesurface)
        argout1 = .5f0*norm(vec(dPredicted) - vec(dObserved),2)^2.f0    # data misfit
        argout2 = pycall(ac[:adjoint_freq_born], Array{Float32, length(model.n)}, modelPy, PyReverseDims(copy(transpose(src_coords))), PyReverseDims((dPredicted - dObserved)'),
                         options.frequencies, uf_real, uf_imag, space_order=options.space_order, nb=model.nb, freesurface=options.freesurface)
    else
        dPredicted, u0 = pycall(ac[:forward_modeling], PyObject, modelPy, PyReverseDims(transpose(src_coords)), PyReverseDims(qIn'), PyReverseDims(copy(transpose(rec_coords))), save=true, freesurface=options.freesurface)
    	# Data misfit
        if isempty(options.gs)
            # wrong misfit for trace based
            argout1 = misfit(dPredicted, dObserved, options.normalize)
            residual = adjoint_src(dPredicted, dObserved, options.normalize)
        else
            argout1 = misfit(dPredicted, dObserved, options.normalize; trace=options.gs["strategy"])
            residual = gs_residual(options.gs, dtComp, dPredicted, dObserved, options.normalize)
        end
        argout2 = pycall(ac[:adjoint_born], Array{Float32}, modelPy, PyReverseDims(copy(transpose(rec_coords))), PyReverseDims(residual'),
                         u=u0, is_residual=true, freesurface=options.freesurface)
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

function fwi_objective(model_full::Model_TTI, source::judiVector, dObs::judiVector, srcnum::Int64; options=Options(), frequencies=[])
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
    modelPy = pm[:Model](origin=(0.,0.,0.), spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims),
                       epsilon=process_physical_parameter(model.epsilon, dims),
                       delta=process_physical_parameter(model.delta, dims),
                       theta=process_physical_parameter(model.theta, dims),
                       phi=process_physical_parameter(model.phi, dims), nbpml=model.nb)
    dtComp = modelPy[:critical_dt]

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
        op_F = pycall(tti[:forward_modeling], PyObject, modelPy, PyReverseDims(copy(transpose(src_coords))), PyReverseDims(qIn'), PyReverseDims(copy(transpose(rec_coords))), op_return=true)
        argout1, argout2 = pycall(tti[:adjoint_born], PyObject, modelPy, PyReverseDims(copy(transpose(rec_coords))), PyReverseDims(dObserved'),
                                  op_forward=op_F, is_residual=false)
    else
        dPredicted, u0, v0 = pycall(tti[:forward_modeling], PyObject, modelPy, PyReverseDims(copy(transpose(src_coords))), PyReverseDims(qIn'), PyReverseDims(copy(transpose(rec_coords))), save=true)
        argout1 = .5f0*norm(vec(dPredicted) - vec(dObserved),2)^2.f0    # data misfit
        argout2 = pycall(tti[:adjoint_born], Array{Float32}, modelPy, PyReverseDims(copy(transpose(rec_coords))), PyReverseDims((dPredicted  - dObserved)'),
                         u=u0,  is_residual=true)
    end
    argout2 = remove_padding(argout2, model.nb, true_adjoint=options.sum_padding)
    if options.limit_m==true
        argout2 = extend_gradient(model_full,model,argout2)
    end
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
