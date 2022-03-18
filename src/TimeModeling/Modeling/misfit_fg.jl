
export fwi_objective, lsrtm_objective, fwi_objective!, lsrtm_objective!

"""
    fwi_objective(model, source, dobs; options=Options())

    Evaluate the full-waveform-inversion (reduced state) objective function. Returns a tuple with function value and vectorized \\
gradient. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.

Example
=======
    function_value, gradient = fwi_objective(model, source, dobs)
"""
function fwi_objective(model::Model, q::judiVector, dobs::judiVector; options=Options())
    G = similar(model.m)
    f = fwi_objective!(G, model, q, dobs; options=options)
    f, G
end

"""
    lsrtm_objective(model, source, dobs, dm; options=Options(), nlind=false)

Evaluate the least-square migration objective function. Returns a tuple with function value and \\
gradient. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.

Example
=======
    function_value, gradient = lsrtm_objective(model, source, dobs, dm)
"""
function lsrtm_objective(model::Model, q::judiVector, dobs::judiVector, dm;
                         options=Options(), nlind=false)
    G = similar(model.m)
    f = lsrtm_objective!(G, model, q, dobs, dm; options=options, nlind=nlind)
    f, G
end

"""
    fwi_objective!(G, model, source, dobs; options=Options())

    Evaluate the full-waveform-inversion (reduced state) objective function. Returns a the function value and assigns in-place \\
the gradient to G. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.

Example
=======
    function_value = fwi_objective!(gradient, model, source, dobs)
"""
fwi_objective!(G, model, q, dobs; options=Options()) = multi_src_fg!(G, model, q, dobs, nothing; options=options, nlind=false, lin=false)

"""
    lsrtm_objective!(G, model, source, dobs, dm; options=Options(), nlind=false)

    Evaluate the least-square migration (data-space) objective function. Returns a the function value and assigns in-place \\
the gradient to G. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.

Example
=======
    function_value = lsrtm_objective!(gradient, model, source, dobs, dm; options=Options(), nlind=false)
"""
lsrtm_objective!(G, model, q, dobs, dm; options=Options(), nlind=false) = multi_src_fg!(G, model, q, dobs, dm; options=options, nlind=nlind, lin=true)


function multi_src_fg(model_full::Model, source::judiVector, dObs::judiVector, dm, options::JUDIOptions, nlind::Bool, lin::Bool)
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
        model, dm = limit_model_to_receiver_area(source.geometry, dObs.geometry, model, options.buffer_size; pert=dm)
    else
        model = model_full
    end

    # Set up Python model
    modelPy = devito_model(model, options, dm)
    dtComp = convert(Float32, modelPy."critical_dt")

    # Extrapolate input data to computational grid
    qIn = time_resample(make_input(source), source.geometry, dtComp)[1]
    dObserved = time_resample(make_input(dObs), dObs.geometry, dtComp)[1]

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(dObs.geometry, model.n)    # shifts rec coordinates by origin

    if options.optimal_checkpointing == true
        argout1, argout2 = pycall(ac."J_adjoint_checkpointing", Tuple{Float32, PyArray},
                                  modelPy, src_coords, qIn, rec_coords, dObserved,
                                  is_residual=false, return_obj=true, isic=options.isic,
                                  born_fwd=lin, nlind=nlind, t_sub=options.subsampling_factor, space_order=options.space_order)
    elseif ~isempty(options.frequencies)
        argout1, argout2 = pycall(ac."J_adjoint_freq", Tuple{Float32,  PyArray},
                                  modelPy, src_coords, qIn, rec_coords, dObserved,
                                  is_residual=false, return_obj=true, isic=options.isic,
                                  freq_list=options.frequencies, t_sub=options.subsampling_factor,
                                  space_order=options.space_order, born_fwd=lin, nlind=nlind)
    else
        argout1, argout2 = pycall(ac."J_adjoint_standard", Tuple{Float32, PyArray},
                                  modelPy, src_coords, qIn, rec_coords, dObserved,
                                  is_residual=false, return_obj=true, born_fwd=lin, nlind=nlind,
                                  t_sub=options.subsampling_factor, space_order=options.space_order,
                                  isic=options.isic)
    end
    argout2 = remove_padding(argout2, modelPy.padsizes; true_adjoint=options.sum_padding)
    if options.limit_m==true
        argout2 = extend_gradient(model_full, model, argout2)
    end
    return Ref{Float32}(argout1), PhysicalParameter(argout2, model_full.d, model_full.o)
end


multi_src_fg(t::Tuple{Model, judiVector, judiVector, Any, JUDIOptions, Bool, Bool}) where N =
    multi_src_fg(t[1], t[2], t[3], t[4], t[5], t[6], t[7])
