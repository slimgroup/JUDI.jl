module FluxJUDIExt

import JUDI: LazyPropagation, judiVector, eval_prop
isdefined(Base, :get_extension) ? (using Flux) : (using ..Flux)

Flux.cpu(x::LazyPropagation) = Flux.cpu(eval_prop(x))
Flux.gpu(x::LazyPropagation) = Flux.gpu(eval_prop(x))
Flux.CUDA.cu(F::LazyPropagation) = Flux.CUDA.cu(eval_prop(F))
Flux.CUDA.cu(x::Vector{Matrix{T}}) where T = [Flux.CUDA.cu(x[i]) for i=1:length(x)]
Flux.CUDA.cu(x::judiVector{T, Matrix{T}}) where T = judiVector{T, Flux.CUDA.CuMatrix{T}}(x.nsrc, x.geometry, Flux.CUDA.cu(x.data))

end