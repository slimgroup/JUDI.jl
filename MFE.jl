
@everywhere using PyCall, Random
@everywhere dv = pyimport("devito")

@everywhere function makeop(size)
    grid = dv.Grid(size)
    u = dv.TimeFunction(name="u", grid=grid, space_order=8, time_order=2)
    op = dv.Operator(dv.Eq(u.forward, 2*u - u.backward + u.laplace))
    op.cfunction
    return op
end

@everywhere function call_op(ids::Symbol)
    # u = dv.TimeFunction(name="u", grid=grid, space_order=8, time_order=2)
    getfield(Main, ids)(h_x=1, h_y=1, time_M=101, dt=.1)
    nothing 
end

function create_op(gridsize)
    ids = Symbol(randstring(10))
    @sync for p in workers()
        @async remotecall_wait(()->Core.eval(Main, :($ids = makeop($gridsize))), p)
    end
    ids
end

ids = create_op((256, 256))

# @sync for p in workers()
#     @async remotecall_fetch(varinfo, p)
# end


# for i=1:10
#     call_op()
# end

@sync for i=1:24
    @async remotecall_fetch(call_op, default_worker_pool(), ids)
end
