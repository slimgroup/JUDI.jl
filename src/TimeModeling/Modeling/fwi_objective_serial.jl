
export fwi_objective

fwi_objective(model_full::Model, source::judiVector, dObs::judiVector) = fwi_objective(model_full, source, dObs; options=Options())

function fwi_objective(model_full::Model, source::judiVector, dObs::judiVector, options::Options)
# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito

    # assert this is for single source LSRTM
    @assert source.nsrc == 1 "Multiple sources are used in a single-source fwi_objective"
    @assert dObs.nsrc == 1 "Multiple-source data is used in a single-source fwi_objective"    

    # Load full geometry for out-of-core geometry containers
    dObs.geometry = Geometry(dObs.geometry)
    source.geometry = Geometry(source.geometry)

    # Limit model to area with sources/receivers
    if options.limit_m == true
        model = deepcopy(model_full)
        model, _ = limit_model_to_receiver_area(source.geometry,dObs.geometry,model,options.buffer_size)
    else
        model = model_full
    end

    # Set up Python model
    modelPy = devito_model(model, options)
    dtComp = get_dt(model; dt=options.dt_comp)

    # Extrapolate input data to computational grid
    qIn = time_resample(source.data[1], source.geometry, dtComp)[1]
    obsd = typeof(dObs.data[1]) == SegyIO.SeisCon ? convert(Array{Float32,2}, dObs.data[1][1].data) : dObs.data[1]
    dObserved = time_resample(obsd, dObs.geometry, dtComp)[1]

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(dObs.geometry, model.n)    # shifts rec coordinates by origin


    if options.optimal_checkpointing == true
        f, g, Iu, Iv = pycall(ac."J_adjoint_checkpointing", fg_I_I(model), modelPy, src_coords, qIn,
                              rec_coords, dObserved, is_residual=false, return_obj=true, isic=options.isic,
                              t_sub=options.subsampling_factor, space_order=options.space_order)
    else
        length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
        f, g, Iu, Iv = pycall(ac."J_adjoint_standard", fg_I_I(model), modelPy, src_coords, qIn, rec_coords, dObserved,
                              is_residual=false, return_obj=true, save=isnothing(freqs),
                              t_sub=options.subsampling_factor, space_order=options.space_order,
                              isic=options.isic, freq_list=freqs, dft_sub=options.dft_subsampling_factor)
    end
    g = phys_out(g, modelPy, model, options)
    Iu, Iv = illum_out((Iu, Iv), modelPy, model, options)
    return post_process((Ref{Float32}(f), g, Iu, Iv), model_full, model)
end
