module CUDAJUDIExt

import JUDI: LazyPropagation, judiVector, eval_prop
isdefined(Base, :get_extension) ? (using CUDA) : (using ..CUDA)

CUDA.cu(F::LazyPropagation) = CUDA.cu(eval_prop(F))
CUDA.cu(x::Vector{Matrix{T}}) where T = [CUDA.cu(x[i]) for i=1:length(x)]
CUDA.cu(x::judiVector{T, Matrix{T}}) where T = judiVector{T, CUDA.CuMatrix{T}}(x.nsrc, x.geometry, CUDA.cu(x.data))

end