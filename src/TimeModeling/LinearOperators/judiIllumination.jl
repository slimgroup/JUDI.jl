export judiIllumination

"""
    judiIllumination
Diagonal approximation of the FWI Hessian as the energy of the forward wavefield.

"""
struct judiIllumination{DDT} <: joAbstractLinearOperator{DDT, DDT}
    name::String
    illum::Dict
    m::Integer
    n::Integer
    mode::String
    k::Number
end

# Real diagonal operator
conj(I::judiIllumination{T}) where T = I
adjoint(I::judiIllumination{T}) where T = I
transpose(I::judiIllumination{T}) where T = I
# Inverse
inv(I::judiIllumination{T}) where T = judiIllumination{Float32}(I.name, I.illum, I.m, I.n, I.mode, -I.k)

*(I::judiIllumination{T}, x::AbstractVector{T}) where T = apply_diag(I, x)

(I::judiIllumination{T})(mode::String) where T = judiIllumination{T}(I.name, I.illum, I.m, I.n, mode, I.k)

function apply_diag(I::judiIllumination{T}, x) where T
    out = deepcopy(x)
    if I.mode == "v"
        illum = I.illum["v"]
    elseif I.mode == "uv"
        illum = sqrt.(I.illum["v"]*I.illum["u"])
    else
        illum = I.illum["u"]
    end

    out[illum .> eps(T)] .*= (illum[illum .> eps(T)]).^(I.k)
    out
end

function judiIllumination(model::Model; mode="u", k=1) where T
    n = prod(model.n)
    illum = model.illums
    judiIllumination{Float32}("Illumination", illum, n, n, mode, k)
end

for JT ∈ [judiModeling, judiAbstractJacobian, judiPDEfull, judiPDEextended, judiPDE]
    @eval judiIllumination(F::$(JT); kw...) = judiIllumination(F.model; kw...)
end
