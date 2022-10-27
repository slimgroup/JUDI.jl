export judiIllumination

"""
    judiIllumination

Diagonal approximation of the FWI Hessian as the energy of the wavefield.

"""
struct judiIllumination{DDT, M, K, R} <: joAbstractLinearOperator{DDT, DDT}
    name::String
    illums
    m::Integer
    n::Integer
end

# Real diagonal operator
conj(I::judiIllumination{T}) where T = I
adjoint(I::judiIllumination{T}) where T = I
transpose(I::judiIllumination{T}) where T = I

# Inverse
inv(I::judiIllumination{T, M, K, R}) where {T, M, K, R} = judiIllumination{T, M, -K, R}(I.name, I.illums, I.m, I.n)

# Mul
function *(I::judiIllumination{T, M, K, R}, x::Vector{T}) where {T, M, K, R}
    illum = (.*(values(I.illums)...)).^(K/length(I.illums))
    return (illum[:] .* x) ./ (illum[:].^2 .+ eps(T))
end

function *(I::judiIllumination{T, M, K, R}, x::PhysicalParameter{T}) where {T, M, K, R}
    illum = (.*(values(I.illums)...)).^(K/length(I.illums))
    return (illum .* x) ./ (illum.^2 .+ eps(T))
end

# Functor
function (I::judiIllumination{T, M, K, R})(mode::String) where {T, M, K, R}
    illum = Dict(s=>similar(first(values(I.illums))) for s in split(mode, ""))
    for k ∈ keys(illum)
        if k ∈ keys(I.illums)
            illum[k] = deepcopy(I.illums[k])
        else
            fill!(illum[k], 1)
        end
    end
    judiIllumination{T, Symbol(mode), K, R}(I.name, illum, I.m, I.n)
end

# Assignment
function set_val(I::judiIllumination{T, M, K, R}, mode, v) where {T, M, K, R}
    key = mode ∈ [:forward, :born] ? "u" : "v"
    if key in keys(I.illums)
        I.illums[key] .= v
    end
end

# status
function is_updated(I::judiIllumination{T, M, K, R}) where {T, M, K, R}
    updated = true
    for (k, v) in I.illums
        im, iM = extrema(v)
        updated = (im == iM == 1) && updated
    end
    return ~updated
end

# Constructor
"""
    judiIllumination(model; mode="u", k=1, recompute=true)


# Arguments

- `model`: JUDI Model structure
- `mode`: Type of ilumination, choicees of ("u", "v", "uv")
- `k`: Power of the illumination, real number
- `recompute`: Flag whether to recompute the illumination at each new propagation (Defaults to true)

    judiIllumination(F; mode="u", k=1, recompute=true)

# Arguments

- `F`: JUDI propagator
- `mode`: Type of ilumination, choicees of ("u", "v", "uv")
- `k`: Power of the illumination, real positive number
- `recompute`: Flag whether to recompute the illumination at each new propagation  (Defaults to true)

Diagonal approximation of the FWI Hessian as the energy of the wavefield. The diagonal contains the sum over time
of the wavefield chosen as `mode`.

Options for the mode are "u" for the forward wavefield illumination, "v" for the adjoint wavefield illumination, and 
"uv" for the pointwise product of the forward and adjoint wavefields illuminations. Additionally, the parameter "k" provides control on the scaling of
the daiagonal raising it to the power `k`. 

Example
========

I = judiIllumination(model) 

Construct the diagonal operator such that I*x = x ./ |||u||_2^2


"""
function judiIllumination(model::Model; mode="u", k=1, recompute=true)
    n = prod(model.n)
    # Initialize the illumination as the identity
    illum = Dict(s=>PhysicalParameter(model.n, model.d, model.o, ones(Float32, model.n)) for s in split(mode, ""))
    I = judiIllumination{Float32, Symbol(mode), k, recompute}("Illumination", illum, n, n)
    init_illum(model, I)
    return I
end

judiIllumination(F::judiPropagator; kw...) = judiIllumination(F.model; kw...)

##################  Illumination tracker. ####################
# We carry a global tracker that associate an illumination operator
# with its  model so that we can extract it after propagation

_illums = Dict()

init_illum(model::Model, I::judiIllumination) = (_illums[objectid(model)] = [I, false])

function update_illum(vals::Tuple, F::judiPropagator{D, O}) where {D, O}
    if length(vals) == 3
        update_illum(F.model, vals[2], :forward)
        update_illum(F.model, vals[3], :adjoint)
    else
        update_illum(F.model, vals[2], O)
    end
    return vals[1]
end

update_illum(vals, ::judiPropagator) = vals

function update_illum(vals::Tuple, model::Model, ::Any)
    length(vals) == 2 && (return vals)
    update_illum(model, vals[3], :forward)
    update_illum(model, vals[4], :adjoint)
    return vals[1:2]
end

function update_illum(model::Model, i::PhysicalParameter, mode)
    set_val(_illums[objectid(model)][1], mode, i)
    _illums[objectid(model)][2] = is_updated(_illums[objectid(model)][1])
end

function _compute_illum(::judiIllumination{T, M, K, R}, status, mode) where {T, M, K, R}
    if status && ~R
        return false
    elseif (mode ∈ [:forward, :born] && M ∈ [:u, :uv]) || (mode == :adjoint && M ∈ [:v, :uv]) || (mode == :adjoint_born)
        return true
    else
        return false
    end
end

function compute_illum(model::Model, mode::Symbol)
    objectid(model) ∉ keys(_illums) && (return false)
    return _compute_illum(_illums[objectid(model)]..., mode)
end