export set_devito_config, ftp_data, set_serial, set_parallel, set_verbosity
export devito_omp, devito_icx, devito_acc, devito_nvc_host, devito_cuda, devito_sycl, devito_hip

# Logging utilities
set_verbosity(x::Bool) = begin global _verbose = x; end
judilog(msg) = _verbose ? printstyled("JUDI: $(msg) \n", color=:magenta) : nothing

function human_readable_time(t::Float64, decimals=2)
    units = ["ns", "μs", "ms", "s", "min", "hour"]
    scales = [1e-9, 1e-6, 1e-3, 1, 60, 3600]
    if t < 1e-9
        tr = round(t/1e-9; sigdigits=decimals)
        return "$(tr) ns"
    end

    for i=2:6
        if t < scales[i]
            tr = round(t/scales[i-1]; sigdigits=decimals)
            return "$(tr) $(units[i-1])"
        end
    end
    tr1 = div(t, 3600)
    tr2 = round(Int, rem(t, 3600))
    return "$(tr1) h $(tr2) min"
end 


macro juditime(msg, ex)
    return quote
       local t
       t = @elapsed $(esc(ex))
       tr = human_readable_time(t)
       judilog($(esc(msg))*": $(tr)")
    end
end


# Utility for data loading
JUDI_DATA = joinpath(JUDIPATH, "../data")
ftp_data(ftp::String, name::String) = Base.Downloads().download("$(ftp)/$(name)", "$(JUDI.JUDI_DATA)/$(name)")
ftp_data(ftp::String) = Base.Downloads().download(ftp, "$(JUDI.JUDI_DATA)/$(split(ftp, "/")[end])")


# Parallelism
_serial = false
get_serial() = _serial
set_serial(x::Bool) = begin global _serial = x; end
set_serial() = begin global _serial = true; end
set_parallel() = begin global _serial = false; end

function _worker_pool()
    if _serial
        return nothing
    end
    p = default_worker_pool()
    pool = nworkers(p) < 2 ? nothing : p
    return pool
end


# Create a lock for pycall FOR THREAD/TASK SAFETY
# See discussion at
# https://github.com/JuliaPy/PyCall.jl/issues/882

const PYLOCK = Ref{ReentrantLock}()

# acquire the lock before any code calls Python
pylock(f::Function) = Base.lock(PYLOCK[]) do
    prev_gc = GC.enable(false)
    try 
        return f()
    finally
        GC.enable(prev_gc) # recover previous state
    end
end

function rlock_pycall(meth, ::Type{T}, args...; kw...) where T
    out::T = pylock() do
        pycall(meth, T, args...; kw...)
    end
    return out
end

# Devito configuration
set_devito_config(key::String, val::String) = set!(devito."configuration", key, val)
set_devito_config(key::String, val::Bool) = set!(devito."configuration", key, val)

set_devito_config(kw...) = begin
    for (k, v) in kw
        set_devito_config(k, v)
    end
end

# Easy configurations setupes
devito_omp() = set_devito_config("language", "openmp")
devito_icx() = set_devito_config(language="openmp", compiler="icx")
devito_acc() = set_devito_config(language="openacc", compiler="nvc", platform="nvidiaX")
devito_nvc_host() = set_devito_config(language="openmp", compiler="nvc")
devito_cuda() = set_devito_config(language="cuda", platform="nvidiaX")
devito_sycl() = set_devito_config(language="sycl", platform="intelgpuX")
devito_hip() = set_devito_config(language="hip", platform="amdgpuX")
