# to import MPIManager
using MPIClusterManagers

# need to also import Distributed to use addprocs()
using Distributed

export mpi_devito_interface

function mympi_do(mgr::MPIManager, expr)
    !mgr.initialized && wait(mgr.cond_initialized)
    jpids = keys(mgr.j2mpi)
    refs = Array{Any}(undef, length(jpids))
    out = Array{Any}(undef, length(jpids))
    for (i,p) in enumerate(Iterators.filter(x -> x != myid(), jpids))
        refs[i] = remotecall(expr, p)
    end
    # Execution on local process should be last, since it can block the main
    # event loop
    if myid() in jpids
        refs[end] = remotecall(expr, myid())
    end

    @sync begin
        for (i, r) in enumerate(refs)
            resp = remotecall_fetch(r.where, r) do rr
                wrkr_result = rr[]
                wrkr_result
            end
            out[i] = resp
        end
    end
    out = filter!(x->x!=nothing, out)
    return out[1]
end

macro mympi_do(mgr, expr)
    quote
        # Evaluate expression in Main module
        thunk = () -> (Core.eval(Main, $(Expr(:quote, expr))))
        mympi_do($(esc(mgr)), thunk)
    end
end

function mpi_devito_interface(model, op, args...)
    options = args[end]

    manager = MPIManager(np=options.mpi)
    workers = addprocs(manager)
    # import back JUDI (yeah that's wierd but needed)
    eval(macroexpand(Distributed, quote @everywhere using JUDI, JUDI.TimeModeling end))
    length(model.n) == 3 ? dims = [3,2,1] : dims = [2,1]

    argout = @mympi_do manager begin
        # Init MPI
        using MPI
        comm = MPI.COMM_WORLD
        # Set up Python model structure
        modelPy = devito_model($model, $options)
        update_m(modelPy, $model.m, $dims)
        # Run devito interface
        argout = devito_interface(modelPy, $model, $(args...))
        # Wait for it to finish
        MPI.Barrier(comm)
        # GAther results
        if MPI.Comm_rank(comm) != 0
            MPI.send(argout, 0, MPI.Comm_rank(comm), comm)
        end
        # Return result
        if MPI.Comm_rank(comm) == 0
            for i=1:(MPI.Comm_size(comm)-1)
                out, status = MPI.recv(i, i, comm)
                vcat(argout, out)
            end
        end
        MPI.Barrier(comm)
        if MPI.Comm_rank(comm) == 0
            return argout
        end
    end
    return argout
end
