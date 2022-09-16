
export twri_objective, TWRIOptions


# TWRI options
mutable struct TWRIOptions
    grad_corr::Bool
    comp_alpha::Bool
    weight_fun
    eps
    params
    Invq::String
end

"""
    TWRIOptions
        grad_corr::Bool
        comp_alpha::Bool
        weight_fun
        eps
        params::Symbol
        Invq::String

Options structure for TWRI.

`grad_corr`: Whether to add the gradient correction J'(m0, q)*∇_y

`comp_alpha`: Whether to compute optimal alpha (alpha=1 if not)

`weight_fun`: Whether to apply focusing/weighting function to F(m0)'*y and its norm

`eps`: Epsilon (noise level) value (default=0)

`Invq`: How to compute F'Y, either as full field or as a rank 1 approximation `w(t)*q(x)` using the source wavelet for w

`param`: Which gradient to compute. Choices are `nothing` (objective only), `:m`, `:y` or `:all`

Constructor
==========

All arguments are optional keyword arguments with the following default values:

TWRIOptions(;grad_corr=false, comp_alpha=true, weight_fun=nothing, eps=0, params=:m)
"""
TWRIOptions(;grad_corr=false, comp_alpha=true,
            weight_fun=nothing, eps=0, params=:m, Invq="standard")=
            TWRIOptions(grad_corr, comp_alpha, weight_fun, eps, params, Invq)

function getindex(opt::TWRIOptions, srcnum::Int)
    eloc = length(opt.eps) == 1 ? opt.eps : opt.eps[srcnum]
    return TWRIOptions(opt.grad_corr, opt.comp_alpha, opt.weight_fun, eloc, opt.params, opt.Invq)
end

subsample(opt::TWRIOptions, srcnum::Int) = getindex(opt, srcnum)

"""
    twri_objective(model, source, dobs; options=Options(), optionswri=TWRIOptions())

Evaluate the time domain Wavefield reconstruction inversion objective function. Returns a tuple with function value and
gradient(s) w.r.t to m and/or y. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and 
observed data of type `judiVector`.

Example
=======
    function_value, gradient_m, gradient_y = twri_objective(model, source, dobs; options=Options(), optionswri=TWRIOptions())
"""
function twri_objective(model_full::Model, source::judiVector, dObs::judiVector, y::Union{judiVector, Nothing},
                        options::JUDIOptions, optionswri::TWRIOptions)
    # Load full geometry for out-of-core geometry containers
    dObs.geometry = Geometry(dObs.geometry)
    source.geometry = Geometry(source.geometry)

    # Limit model to area with sources/receivers
    if options.limit_m == true
        model = deepcopy(model_full)
        model = limit_model_to_receiver_area(source.geometry, dObs.geometry, model, options.buffer_size)
    else
        model = model_full
    end

    # Set up Python model structure 
    modelPy = devito_model(model, options)
    dtComp = convert(Float32, modelPy."critical_dt")

    # Extrapolate input data to computational grid
    qIn = time_resample(source.data[1], source.geometry, dtComp)[1]
    dObserved = time_resample(make_input(dObs), dObs.geometry, dtComp)[1]

    isnothing(y) ? Y = nothing : Y = time_resample(make_input(y), y.geometry, dtComp)[1]

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(dObs.geometry, model.n)    # shifts rec coordinates by origin

    ~isempty(options.frequencies) ? freqs = options.frequencies : freqs = nothing
    ~isempty(options.frequencies) ? (wfilt, freqs) =  filter_w(qIn, dtComp, freqs) : wfilt = nothing
    obj, gradm, grady = pycall(ac."wri_func", PyObject,
                               modelPy, src_coords, qIn, rec_coords, dObserved, Y,
                               t_sub=options.subsampling_factor, space_order=options.space_order,
                               grad=optionswri.params, grad_corr=optionswri.grad_corr, eps=optionswri.eps,
                               alpha_op=optionswri.comp_alpha, w_fun=optionswri.weight_fun,
                               freq_list=freqs, wfilt=wfilt, f0=options.f0)

    if (optionswri.params in [:m, :all])
        gradm = remove_padding(gradm, modelPy.padsizes; true_adjoint=options.sum_padding)
        gradm = PhysicalParameter(gradm, model.d, model.o)
    end
    if ~isnothing(grady)
        grady = time_resample(grady, dtComp, dObs.geometry)
        grady = judiVector(dObs.geometry, grady)
    end

    return filter_out(Ref{Float32}(obj), gradm, grady)
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

filter_out(obj, ::Nothing, ::Nothing) = obj
filter_out(obj, m, ::Nothing) = obj, m
filter_out(obj, ::Nothing, y) = obj, y
filter_out(obj, m, y) = obj, m, y


# Parallel
"""
    twri_objective(model, source, dobs; options=Options(), optionswri=TWRIOptions())
Evaluate the time domain Wavefield reconstruction inversion objective function. Returns a tuple with function value and \\
gradient(s) w.r.t to m and/or y. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.
Example
=======
    function_value, gradient = fwi_objective(model, source, dobs)
"""
function twri_objective(model::Model, source::judiVector, dObs::judiVector, y::Union{judiVector, Nothing};
                        options=Options(), optionswri=TWRIOptions())
    pool = _worker_pool()
    if isnothing(y)
        arg_func = j -> (model, source[j], dObs[j], nothing, options[j], optionswri[j])
    else
        arg_func = j -> (model, source[j], dObs[j], y[j], options[j], optionswri[j])
    end
    results = run_and_reduce(twri_objective, pool, source.nsrc, arg_func)
    # Collect and reduce gradients
    out = as_vec(results, Val(options.return_array))
    return out
end
