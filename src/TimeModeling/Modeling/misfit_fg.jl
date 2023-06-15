
export fwi_objective, lsrtm_objective, fwi_objective!, lsrtm_objective!

function multi_src_fg(model_full::AbstractModel, source::judiVector, dObs::judiVector, dm, options::JUDIOptions, nlind::Bool, lin::Bool,
                      misfit::Function)
    # Setup time-domain linear or nonlinear foward and adjoint modeling and interface to devito

    # assert this is for single source LSRTM
    @assert source.nsrc == 1 "Multiple sources are used in a single-source fwi_objective"
    @assert dObs.nsrc == 1 "Multiple-source data is used in a single-source fwi_objective"    

    # Load full geometry for out-of-core geometry containers
    dObs.geometry = Geometry(dObs.geometry)
    source.geometry = Geometry(source.geometry)

    # Compute illumination ?
    illum = compute_illum(model_full, :adjoint_born)

    # Limit model to area with sources/receivers
    if options.limit_m == true
        @juditime "Limit model to geometry" begin
            model = deepcopy(model_full)
            model, dm = limit_model_to_receiver_area(source.geometry, dObs.geometry, model, options.buffer_size; pert=dm)
        end
    else
        model = model_full
    end

    # Set up Python model
    @juditime "Devito Model" begin
        modelPy = devito_model(model, options, dm)
        dtComp = convert(Float32, modelPy."critical_dt")
    end

    # Extrapolate input data to computational grid
    qIn = time_resample(make_input(source), source.geometry, dtComp)
    dObserved = time_resample(make_input(dObs), dObs.geometry, dtComp)

    # Set up coordinates
    @juditime "Sparse coords setup" begin
        src_coords = setup_grid(source.geometry, size(model))  # shifts source coordinates by origin
        rec_coords = setup_grid(dObs.geometry, size(model))    # shifts rec coordinates by origin
    end

    mfunc = pyfunction(misfit, Matrix{Float32}, Matrix{Float32})

    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    IT = illum ? (PyArray, PyArray) : (PyObject, PyObject)
    @juditime "Python call to J_adjoint" begin
        argout = rlock_pycall(ac."J_adjoint", Tuple{Float32, PyArray, IT...}, modelPy,
                src_coords, qIn, rec_coords, dObserved, t_sub=options.subsampling_factor,
                space_order=options.space_order, checkpointing=options.optimal_checkpointing,
                freq_list=freqs, ic=options.IC, is_residual=false, born_fwd=lin, nlind=nlind,
                dft_sub=options.dft_subsampling_factor[1], f0=options.f0, return_obj=true, misfit=mfunc, illum=illum)
    end

    @juditime "Filter empty output" begin
        argout = filter_none(argout)
    end

    @juditime "Remove padding from gradient" begin
        grad = PhysicalParameter(remove_padding(argout[2], modelPy.padsizes; true_adjoint=options.sum_padding),  spacing(model), origin(model))
    end

    fval = Ref{Float32}(argout[1])
    if illum
        @juditime "Process illumination" begin
            illumu = PhysicalParameter(remove_padding(argout[3], modelPy.padsizes; true_adjoint=false), spacing(model), origin(model))
            illumv = PhysicalParameter(remove_padding(argout[4], modelPy.padsizes; true_adjoint=false), spacing(model), origin(model))
        end
        return fval, grad, illumu, illumv
    end
    return fval, grad
end


####### Defaults
multi_src_fg(model_full::AbstractModel, source::judiVector, dObs::judiVector, dm, options::JUDIOptions, nlind::Bool, lin::Bool) =
    multi_src_fg(model_full::AbstractModel, source::judiVector, dObs::judiVector, dm, options::JUDIOptions, nlind::Bool, lin::Bool, mse)


# Find number of experiments
"""
    get_nexp(x)

Get number of experiments given a JUDI type. By default we have only one experiment unless we input
a Vector of judiType such as [model, model] to compute gradient for different cases at once.
"""
get_nexp(x) = 1
for T in [judiVector, AbstractModel, judiWeights, judiWavefield, PhysicalParameter, Vector{Float32}]
    @eval get_nexp(v::Vector{<:$T}) = length(v)
    @eval get_nexp(v::Tuple{N, <:$T}) where N = length(v)
end   

# Filter arguments for given task
"""
    get_exp(x, i)

Filter input `x`` for experiment number `i`. Returns `x` is a constant not depending on experiment.
"""
get_exp(x, i) = x
get_exp(x::Tuple{}, i::Any) = x[i]
for T in [judiVector, AbstractModel, judiWeights, judiWavefield, Array{Float32}, PhysicalParameter]
    @eval get_exp(v::Vector{<:$T}, i) = v[i]
    @eval get_exp(v::NTuple{N, <:$T}, i) where N = v[i]
end

function check_args(args...)
    n = [get_nexp(a) for a in args]
    nexp = maximum(n)
    check = all(ni -> (ni==nexp || ni==1), n)
    check || throw(ArgumentError("Incompatible number of experiements"))
    return nexp
end


# Type of accepted input
Dtypes = Union{<:judiVector, NTuple{N, <:judiVector} where N, Vector{<:judiVector}}
MTypes = Union{<:AbstractModel, NTuple{N, Model} where N, Vector{<:AbstractModel}}
dmTypes = Union{dmType, NTuple{N, dmType} where N, Vector{dmType}}


"""
    fwi_objective(model, source, dobs; options=Options())

    Evaluate the full-waveform-inversion (reduced state) objective function. Returns a tuple with function value and vectorized \\
gradient. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.

Example
=======
    function_value, gradient = fwi_objective(model, source, dobs)
"""
function fwi_objective(model::MTypes, q::Dtypes, dobs::Dtypes; options=Options(), kw...)
    n_exp = check_args(model, q, dobs)
    G = n_exp == 1 ? similar(model.m, model) : [similar(get_exp(model, i).m, get_exp(model, i)) for i=1:n_exp]
    f = fwi_objective!(G, model, q, dobs; options=options, kw...)
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
function lsrtm_objective(model::MTypes, q::Dtypes, dobs::Dtypes, dm::dmTypes; options=Options(), nlind=false, kw...)
    n_exp = check_args(model, q, dobs, dm)
    G = n_exp == 1 ? similar(model.m, model) : [similar(get_exp(model, i).m, get_exp(model, i)) for i=1:n_exp]
    f = lsrtm_objective!(G, model, q, dobs, dm; options=options, nlind=nlind, kw...)
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
function fwi_objective!(G, model::MTypes, q::Dtypes, dobs::Dtypes; options=Options(), kw...)
    n_exp = check_args(G, model, dobs, q)
    return multi_exp_fg!(Val(n_exp), G, model, q, dobs, nothing; options=options, nlind=false, lin=false, kw...)
end

"""
    lsrtm_objective!(G, model, source, dobs, dm; options=Options(), nlind=false)

    Evaluate the least-square migration (data-space) objective function. Returns the function value and assigns in-place \\
the gradient to G. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.

Example
=======
    function_value = lsrtm_objective!(gradient, model, source, dobs, dm; options=Options(), nlind=false)
"""
function lsrtm_objective!(G, model::MTypes, q::Dtypes, dobs::Dtypes, dm::dmTypes; options=Options(), nlind=false, kw...)
    n_exp = check_args(G, model, q, dobs, dm)
    return multi_exp_fg!(Val(n_exp), G, model, q, dobs, dm; options=options, nlind=nlind, lin=true, kw...)
end

multi_exp_fg!(n::Val{1}, ar...; kw...) = multi_src_fg!(ar...; kw...)

function multi_exp_fg!(n::Val{N}, ar...; kw...) where N
    f = zeros(Float32, N)
    @sync for i=1:N
        ai = (get_exp(a, i) for a in ar)
        @async f[i] = multi_src_fg!(ai...; kw...)
    end
    sum(f)
end
