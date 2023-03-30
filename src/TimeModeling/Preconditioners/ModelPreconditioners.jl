export judiIllumination, judiDepthScaling, judiTopmute
export DepthScaling, TopMute

"""
    DepthScaling{T, N, K}

Depth scaling operator in `N` dimensions scaling by `depth^K`.


Constructor
===========

    judiDepthScaling(model::AbstractModel; K=.5)

"""
struct DepthScaling{T, N, K} <: ModelPreconditioner{T, T}
    m::Integer
    depth::Array{T, N}
end

function judiDepthScaling(model::AbstractModel; K=.5f0)
    N = length(model.n)
    depth = reshape(range(0f0, stop=(model.n[end] - 1) * model.d[end], length=model.n[end]), ones(Int64, N-1)..., :)
    return DepthScaling{Float32, N, K}(prod(model.n), depth)
end
  
matvec(D::DepthScaling{T, N, K}, x::Vector{T}) where {T, N, K} = vec(reshape(x, :, size(D.depth, N)) .* D.depth[:]'.^K)
matvec(D::DepthScaling{T, N, K}, x::AbstractArray{T, N}) where {T, N, K} = x .* D.depth.^K
matvec(D::DepthScaling{T, N, K}, x::PhysicalParameter{T}) where {T, N, K} = PhysicalParameter{T}(x.n, x.d, x.o, x.data .* D.depth.^K)
matvec(D::DepthScaling{T, N, K}, x::judiWeights{T}) where {T, N, K} = judiWeights{T}(x.nsrc, [matvec(D, x.data[s]) for s=1:x.nsrc])

# Diagonal operator, self-adjoint
matvec_T(D::DepthScaling{T, N, K}, x) where {T, N, K} = matvec(D, x)

# Real diagonal operator
conj(I::DepthScaling{T}) where T = I
adjoint(I::DepthScaling{T}) where T = I
transpose(I::DepthScaling{T}) where T = I
inv(I::DepthScaling{T, N, K}) where {T, N, K} = DepthScaling{Float32, N, -K}(I.depth)

"""
    TopMute{T, N, Nw}

Mute top of the model in `N` dimensions

Constructor
===========
    judiTopmute(model; taperwidht=10)
    judiTopmute(n, wb, taperwidth)   # Legacy
"""
struct TopMute{T, N, Nw} <: ModelPreconditioner{T, T}
    m::Integer
    wb::Array{Int64, Nw}
    taperwidth::Int64
    TopMute(m::Integer, wb::Array{T, Nw}, taperwidth::Integer) where {T, Nw} =  new{Float32, Nw+1, Nw}(m, wb, taperwidth)
end

judiTopmute(n::NTuple{N, Integer}, wb::Array{T, Nw}, taperwidth::Integer) where {T, N, Nw} = TopMute(prod(n), wb, taperwidth)
judiTopmute(n::NTuple{N, Integer}, wb::Integer, taperwidth::Integer) where {N} = TopMute(prod(n), wb*ones(Int64, n[1:end-1]), taperwidth)

function judiTopmute(model::AbstractModel; taperwidth=10)
    wb = find_water_bottom(model.m.data)
    return TopMute(prod(model.n), wb, taperwidth)
end


function matvec(D::TopMute{T, N}, x::Array{T, N}) where {T, N}
    out = 1 .* x
    taper = D.taperwidth < 2 ? 1 : _taper(Val(:reflection), D.taperwidth)
    for i in CartesianIndices(D.wb)
        out[i, 1:D.wb[i]-D.taperwidth] .= 0
        out[i, D.wb[i]-D.taperwidth+1:D.wb[i]] .*= taper
    end
    out
end

matvec(D::TopMute{T, N}, x::PhysicalParameter{T}) where {T, N} = PhysicalParameter(x, matvec(D, x.data))
matvec(D::TopMute{T, N}, x::judiWeights{T}) where {T, N} = judiWeights{T}(x.nsrc, [matvec(D, x.data[s]) for s=1:x.nsrc])
matvec(D::TopMute{T, N}, x::Vector{T}) where {T, N} = vec(matvec(D, reshape(x, size(D.wb)..., :)))
matvec_T(D::TopMute{T, N}, x) where {T, N} = matvec(D, x)

# Real diagonal operator
conj(I::TopMute{T, N}) where {T, N} = I
adjoint(I::TopMute{T, N}) where {T, N} = I
transpose(I::TopMute{T, N}) where {T, N} = I
inv(::TopMute{T, N}) where {T, N} = throw(MethodError(inv, "Topmute masks contains zeros cannot be inverted"))


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
struct judiIllumination{DDT, M, K, R} <: ModelPreconditioner{DDT, DDT}
    name::String
    illums
    m::Integer
end

function judiIllumination(model::AbstractModel; mode="u", k=1, recompute=true)
    n = prod(model.n)
    # Initialize the illumination as the identity
    illum = Dict(s=>PhysicalParameter(model.n, model.d, model.o, ones(Float32, model.n)) for s in split(mode, ""))
    I = judiIllumination{Float32, Symbol(mode), k, recompute}("Illumination", illum, n)
    init_illum(model, I)
    return I
end

judiIllumination(F::judiPropagator; kw...) = judiIllumination(F.model; kw...)


# Real diagonal operator
conj(I::judiIllumination{T}) where T = I
adjoint(I::judiIllumination{T}) where T = I
transpose(I::judiIllumination{T}) where T = I

# Inverse
inv(I::judiIllumination{T, M, K, R}) where {T, M, K, R} = judiIllumination{T, M, -K, R}(I.name, I.illums, I.m)

# Mul
function matvec(I::judiIllumination{T, M, K, R}, x::Vector{T}) where {T, M, K, R}
    illum = (.*(values(I.illums)...)).^(1/length(I.illums))
    inds = findall(illum[:] .> eps(T))
    out = T(0) * x
    out[inds] .= illum[:][inds].^K .* x[inds]
    return out
end

function matvec(I::judiIllumination{T, M, K, R}, x::PhysicalParameter{T}) where {T, M, K, R}
    illum = (.*(values(I.illums)...)).^(1/length(I.illums))
    inds = findall(illum .> eps(T))
    out = T(0) * x
    out[inds] .= illum[inds].^K .* x[inds]
    return out
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
    judiIllumination{T, Symbol(mode), K, R}(I.name, illum, I.m)
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

##################  Illumination tracker. ####################
# We carry a global tracker that associate an illumination operator
# with its  model so that we can extract it after propagation

_illums = Dict()

init_illum(model::AbstractModel, I::judiIllumination) = (_illums[objectid(model)] = [I, false])

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

function update_illum(vals::Tuple, model::AbstractModel, ::Any)
    length(vals) == 2 && (return vals)
    update_illum(model, vals[3], :forward)
    update_illum(model, vals[4], :adjoint)
    return vals[1:2]
end

function update_illum(model::AbstractModel, i::PhysicalParameter, mode)
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

function compute_illum(model::AbstractModel, mode::Symbol)
    objectid(model) ∉ keys(_illums) && (return false)
    return _compute_illum(_illums[objectid(model)]..., mode)
end


function _track_illum(old_m::Model, new_m::Model)
    if (objectid(old_m) ∈ keys(_illums)) && (objectid(new_m) ∉ keys(_illums))
        _illums[objectid(new_m)] = _illums[objectid(old_m)]
    end
end