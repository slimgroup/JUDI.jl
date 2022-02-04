
export lsrtm_objective

# Other potential calls
lsrtm_objective(model_full::Model, source::judiVector, dObs::judiVector, dm::Union{Array, PhysicalParameter}, nlind::Bool, options::Options) = lsrtm_objective(model_full, source, dObs, dm, options, nlind)
lsrtm_objective(model_full::Model, source::judiVector, dObs::judiVector, dm::Union{Array, PhysicalParameter}, options::Options) = lsrtm_objective(model_full, source, dObs, dm, options, false)
lsrtm_objective(model_full::Model, source::judiVector, dObs::judiVector, dm::Union{Array, PhysicalParameter}) = lsrtm_objective(model_full, source, dObs, dm; options=Options(), nlind=false)
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
        f, im, Iu, Iv = pycall(ac."J_adjoint_checkpointing", fg_I_I(model), modelPy, src_coords, qIn,
                                  rec_coords, dObserved, is_residual=false, return_obj=true,
                                  t_sub=options.subsampling_factor, space_order=options.space_order,
                                  born_fwd=true, nlind=nlind, isic=options.isic)
    else
        save = isempty(options.frequencies)
        f, im, Iu, Iv = pycall(ac."J_adjoint_standard", fg_I_I(model), modelPy, src_coords, qIn,
                               rec_coords, dObserved, is_residual=false, return_obj=true, save=save,
                               t_sub=options.subsampling_factor, space_order=options.space_order,
                               freq_list=options.frequencies, dft_sub=options.dft_subsampling_factor,
                               isic=options.isic, born_fwd=true, nlind=nlind)
    end
    im = phys_out(im, modelPy, model, options)
    Iu, Iv = illum_out((Iu, Iv), modelPy, model, options)
    return post_process((Ref{Float32}(f), im, Iu, Iv), model_full, model)
end
