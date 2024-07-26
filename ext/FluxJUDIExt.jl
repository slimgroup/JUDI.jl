module FluxJUDIExt

import JUDI: LazyPropagation
isdefined(Base, :get_extension) ? (using Flux) : (using ..Flux)

Flux.cpu(x::LazyPropagation) = Flux.cpu(eval_prop(x))
Flux.gpu(x::LazyPropagation) = Flux.gpu(eval_prop(x))

end