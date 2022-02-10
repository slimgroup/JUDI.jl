# Create a solver on every worker
function init_local(name::Symbol, model::Model, options::Options)
    pymodel = devito_model(model, options)
    opts = Dict(Symbol(s) => getfield(options, s) for s in fieldnames(JUDI.Options))
    Core.eval(JUDI, :($name = we."WaveSolver"($pymodel; $(opts)...)))
    nothing
end

function init_solver(model::Model, options::Options)
    solver = make_id()
    @sync for p in workers()
        @async remotecall_wait(init_local, p, solver, model, options)
    end
    return solver
end

# Update model on every worker
function update_local(name::Symbol, dm::AbstractArray{T, N}, pad::Vector{<:NTuple{N, Integer}}) where {T, N}
    pysolver = getfield(JUDI, name)
    dm = pad_array(dm, pad)
    pysolver."model"."dm" = dm
    nothing
end

function set_dm!(m::Model, o::Options, s::Symbol, dm::AbstractVector)
    pad = pad_sizes(m, o)
    dm = reshape(dm, m.n)
    @sync for p in workers()
        @async remotecall_wait(update_local, p, s, dm, pad)
    end
    nothing
end

set_dm!(m::Model, o::Options, s::Symbol, dm::PhysicalParameter) = set_dm!(m. o, s, dm.data)