
export fwi_objective

fwi_objective(model_full::Model, source::judiVector, dObs::judiVector) = fwi_objective(model_full, source, dObs; options=Options())

"""
    fwi_objective(model, source, dobs; options=Options())

Evaluate the full-waveform-inversion (reduced state) objective function. Returns a tuple with function value and
gradient. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and
observed data of type `judiVector`.

Example
=======
    function_value, gradient = fwi_objective(model, source, dobs)
"""
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
    dtComp = convert(Float32, modelPy."critical_dt")

    # Extrapolate input data to computational grid
    qIn = time_resample(source.data[1], source.geometry, dtComp)[1]
    dObserved = time_resample(convert(Matrix{Float32}, dObs.data[1]), dObs.geometry, dtComp)[1]

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(dObs.geometry, model.n)    # shifts rec coordinates by origin


    if options.optimal_checkpointing == true
        argout1, argout2 = pycall(ac."J_adjoint_checkpointing", Tuple{Float32, PyArray},
                                  modelPy, src_coords, qIn,
                                  rec_coords, dObserved, is_residual=false, return_obj=true, isic=options.isic,
                                  t_sub=options.subsampling_factor, space_order=options.space_order, f0=options.f0)
    elseif ~isempty(options.frequencies)
        argout1, argout2 = pycall(ac."J_adjoint_freq", Tuple{Float32,  PyArray},
                                  modelPy, src_coords, qIn,
                                  rec_coords, dObserved, is_residual=false, return_obj=true, isic=options.isic,
                                  freq_list=options.frequencies, t_sub=options.subsampling_factor,
                                  space_order=options.space_order, f0=options.f0)
    else
        argout1, argout2 = pycall(ac."J_adjoint_standard", Tuple{Float32, PyArray},
                                  modelPy, src_coords, qIn,
                                  rec_coords, dObserved, is_residual=false, return_obj=true,
                                  t_sub=options.subsampling_factor, space_order=options.space_order,
                                  isic=options.isic, f0=options.f0)
    end
    argout2 = remove_padding(argout2, modelPy.padsizes; true_adjoint=options.sum_padding)
    if options.limit_m==true
        argout2 = extend_gradient(model_full, model, argout2)
    end
    return Ref{Float32}(argout1), PhysicalParameter(argout2, model_full.d, model_full.o)
end
