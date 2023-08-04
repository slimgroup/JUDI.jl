
#############################################################################
(F::judiModeling{D, O})(m::AbstractModel{D, N}) where {D, O, N} = judiModeling(m; options=F.options)
(FS::judiDataSourceModeling{D, O})(F::judiModeling{D, O}) where {D, O} = judiDataSourceModeling{D, O}(FS.rInterpolation, F, FS.qInjection)
(FS::judiDataModeling{D, O})(F::judiModeling{D, O}) where {D, O} = judiDataModeling{D, O}(FS.rInterpolation, F)
(FS::judiPointSourceModeling{D, O})(F::judiModeling{D, O}) where {D, O} = judiPointSourceModeling{D, O}(F, FS.qInjection)

(J::judiJacobian{D, O, FT})(F::judiModeling{D, O}) where {D, O, FT} = judiJacobian(J.F(F), J.q)

for FT in [judiPointSourceModeling, judiDataModeling, judiDataSourceModeling, judiJacobian]
    @eval begin
        (F::$(FT))(m::AbstractModel) = F(F.F(m))
    end
end

function (F::judiPropagator)(;kwargs...)
    Fl = deepcopy(F)
    for (k, v) in kwargs
        k in _mparams(Fl.model) && getfield(Fl.model, k) .= reshape(v, size(Fl.model))
    end
    Fl
end

function (F::judiPropagator)(m, q)
    Fm = F(;m=m)
    _track_illum(F.model, Fm.model)
    return Fm*as_src(q)
end

function (F::judiPropagator)(m::AbstractArray)
    @info "Assuming m to be squared slowness for $(typeof(F))"
    return F(;m=reshape(m, size(F.model)))
end

(F::judiPropagator)(m::AbstractModel, q) = F(m)*as_src(q)

function (J::judiJacobian{D, O, FT})(q::judiMultiSourceVector) where {D, O, FT}
    newJ = judiJacobian{D, O, FT}(J.m, J.n, J.F, q)
    _track_illum(J.model, newJ.model)
    return newJ
end

function (J::judiJacobian{D, O, FT})(x::Array{D, N}) where {D, O, FT, N}
    if length(x) == prod(size(J.model))
        return J(;m=reshape(x, size(F.model.n)))
    end
    new_q = _as_src(J.qInjection.op, J.model, x)
    newJ = judiJacobian{D, O, FT}(J.m, J.n, J.F, new_q)
    _track_illum(J.model, newJ.model)
    return newJ
end

  