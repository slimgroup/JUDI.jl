
export lsrtm_objective

# Other potential calls
lsrtm_objective(model_full::Model, source::judiVector, dObs::judiVector, dm::Union{Array, PhysicalParameter}, nlind::Bool, options::Options) = lsrtm_objective(model_full, source, dObs, dm, options, nlind)
lsrtm_objective(model_full::Model, source::judiVector, dObs::judiVector, dm::Union{Array, PhysicalParameter}, options::Options) = lsrtm_objective(model_full, source, dObs, dm, options, false)
lsrtm_objective(model_full::Model, source::judiVector, dObs::judiVector, dm::Union{Array, PhysicalParameter}) = lsrtm_objective(model_full, source, dObs, dm; options=Options(), nlind=false)
lsrtm_objective(model_full::Model, source::judiVector, dObs::judiVector, dm::Union{Array, PhysicalParameter}, nlind::Bool) = lsrtm_objective(model_full, source, dObs, dm, Options(), nlind)

"""
    lsrtm_objective(model, source, dobs, dm; options=Options(), nlind=false)

Evaluate the least-square migration objective function. Returns a tuple with function value and
gradient. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and
observed data of type `judiVector`. The `nlind` parameter decide whether the backround velocity synthetic data should be subtracted
from the observed data effectively computing the gradient as `J'*(J*dm - (d - F*q))``

Example
=======
    function_value, dm = lsrtm_objective(model, source, dobs, dm)
"""
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
    dtComp = convert(Float32, modelPy."critical_dt")

    # Extrapolate input data to computational grid
    qIn = time_resample(source.data[1],source.geometry,dtComp)[1]
    dObserved = time_resample(convert(Matrix{Float32}, dObs.data[1]), dObs.geometry, dtComp)[1]

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(dObs.geometry, model.n)    # shifts rec coordinates by origin

    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    argout1, argout2 = pycall(ac."J_adjoint", Tuple{Float32, PyArray}, modelPy,
                              src_coords, qIn, rec_coords, dObserved, t_sub=options.subsampling_factor,
                              space_order=options.space_order, checkpointing=options.optimal_checkpointing,
                              freq_list=freqs, isic=options.isic, is_residual=false, return_obj=true,
                              dft_sub=options.dft_subsampling_factor[1], f0=options.f0, born_fwd=true)

    argout2 = remove_padding(argout2, modelPy.padsizes; true_adjoint=options.sum_padding)
    if options.limit_m==true
        argout2 = extend_gradient(model_full, model, argout2)
    end

    return Ref{Float32}(argout1), PhysicalParameter(argout2, model_full.d, model_full.o)
end
