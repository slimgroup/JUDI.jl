
export lsrtm_objective

lsrtm_objective(model_full::Model, source::judiVector, dObs::judiVector, dm::Union{Array, PhysicalParameter}, nlind::Bool, options::Options) = lsrtm_objective(model_full, source, dObs, dm, options, nlind)
lsrtm_objective(model_full::Model, source::judiVector, dObs::judiVector, dm::Union{Array, PhysicalParameter}, options::Options) = lsrtm_objective(model_full, source, dObs, dm, options, false)
lsrtm_objective(model_full::Model, source::judiVector, dObs::judiVector, dm::Union{Array, PhysicalParameter}) = lsrtm_objective(model_full, source, dObs, dm, Options(), false)
lsrtm_objective(model_full::Model, source::judiVector, dObs::judiVector, dm::Union{Array, PhysicalParameter}, nlind::Bool) = lsrtm_objective(model_full, source, dObs, dm, Options(), nlind)


function lsrtm_objective(model_full::Model, source::judiVector, dObs::judiVector, dm::Union{Array, PhysicalParameter}, options::Options, nlind::Bool)
    # assert this is for single source LSRTM
    @assert source.nsrc == 1 "Multiple sources are used in a single-source lsrtm_objective"
    @assert dObs.nsrc == 1 "Multiple-source data is used in a single-source lsrtm_objective"

    # Load full geometry for out-of-core geometry containers
    dObs.geometry = Geometry(dObs.geometry)
    source.geometry = Geometry(source.geometry)

    # Limit model to area with sources/receivers
    if options.limit_m == true
        model = deepcopy(model_full)
        model, dm = limit_model_to_receiver_area(source.geometry,dObs.geometry,model,options.buffer_size; pert=dm)
    else
        model = model_full
    end

    # Set up Python model structure
    modelPy = devito_model(model, options; dm=dm)
    dtComp = get_dt(model; dt=options.dt_comp)

    # Extrapolate input data to computational grid
    qIn = time_resample(source.data[1],source.geometry,dtComp)[1]
    obsd = typeof(dObs.data[1]) == SegyIO.SeisCon ? convert(Array{Float32,2}, dObs.data[1][1].data) : dObs.data[1]
    dObserved = time_resample(obsd, dObs.geometry, dtComp)[1]

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(dObs.geometry, model.n)    # shifts rec coordinates by origin

    if options.optimal_checkpointing == true
        argout1, argout2 = pycall(ac."J_adjoint_checkpointing", Tuple{Float32,  Array{Float32, modelPy.dim}},
                                  modelPy, src_coords, qIn,
                                  rec_coords, dObserved, is_residual=false, return_obj=true,
                                  t_sub=options.subsampling_factor, space_order=options.space_order,
                                  born_fwd=true, nlind=nlind, isic=options.isic)
    elseif ~isempty(options.frequencies)
        argout1, argout2 = pycall(ac."J_adjoint_freq", Tuple{Float32, Array{Float32, modelPy.dim}},
                                  modelPy, src_coords, qIn,
                                  rec_coords, dObserved, is_residual=false, return_obj=true, nlind=nlind,
                                  freq_list=options.frequencies, t_sub=options.subsampling_factor,
                                  space_order=options.space_order, born_fwd=true, isic=options.isic)
    else
        argout1, argout2 = pycall(ac."J_adjoint_standard", Tuple{Float32, Array{Float32, modelPy.dim}},
                                  modelPy, src_coords, qIn,
                                  rec_coords, dObserved, is_residual=false, return_obj=true,
                                  t_sub=options.subsampling_factor, space_order=options.space_order,
                                  isic=options.isic, born_fwd=true, nlind=nlind)
    end
    argout2 = remove_padding(argout2, modelPy.padsizes; true_adjoint=options.sum_padding)
    if options.limit_m==true
        argout2 = extend_gradient(model_full, model, argout2)
    end

    return Ref{Float32}(argout1),  PhysicalParameter(argout2, model_full.d, model_full.o)
end
