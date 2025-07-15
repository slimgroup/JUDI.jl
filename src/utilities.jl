export set_devito_config, ftp_data, set_serial, set_parallel, set_verbosity
export devito_omp, devito_icx, devito_acc, devito_nvc_host, devito_cuda, devito_sycl, devito_hip
export devito_apple_arm, devito_apple_x64, devito_C
export default_devito_config

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


# Devito configuration
set_devito_config(key::Symbol, val) = set_devito_config(string(key), val)
set_devito_config(key::String, val::String) = begin devito.configuration[key] = val end
set_devito_config(key::String, val::Bool) =  begin devito.configuration[key] = val end

set_devito_config(;kw...) = begin
    for (k, v) in kw
        set_devito_config(k, v)
    end
end

# Easy configurations setupes
devito_apple_arm() = set_devito_config(language="openmp", platform="m1", compiler="clang")
devito_apple_x64() = set_devito_config(language="openmp", platform="cpu64", compiler="clang")
devito_C(cc::AbstractString="gcc") = set_devito_config(language="C", platform="cpu64", compiler=cc)
devito_omp() = set_devito_config(language="openmp", platform="cpu64", compiler="gcc")
devito_icx() = set_devito_config(language="openmp", compiler="icx", platform="intel64")
devito_acc() = set_devito_config(language="openacc", compiler="nvc", platform="nvidiaX")
devito_nvc_host() = set_devito_config(language="openmp", compiler="nvc", platform="cpu64")
devito_cuda() = set_devito_config(language="cuda", platform="nvidiaX", compiler="cuda")
devito_sycl() = set_devito_config(language="sycl", platform="intelgpuX", compiler="sycl")
devito_hip() = set_devito_config(language="hip", platform="amdgpuX", compiler="hip")


 function supports_openmp(cc::AbstractString="gcc")
    # run the preprocessor with -fopenmp and look for the _OPENMP macro
    out = try
        read(`$cc -fopenmp -dM -E -x c -`, String)
    catch
        return false
    end
    return occursin(r"#define _OPENMP", out)
end


function default_devito_config()
    # Devito already configured, leave it as is
    ("DEVITO_ARCH" in keys(ENV) || "DEVITO_LANGUAGE" in keys(ENV)) && return
    # Are we on apple arm?
    if Sys.isapple()
        # Check that llvm and libomp are installed
        if !supports_openmp("clang")
            @warn """
            llvm and libomp are not installed, please install them via homebrew and follow path instructions:
                `brew install llvm libomp`

            defaulting to C without OpenMP support
            """
            devito_C("clang")
        else
            Sys.ARCH == :aarch64 ? devito_apple_arm() : devito_apple_x64()
        end
    elseif Sys.islinux()
        if !supports_openmp("gcc")
            @warn """
            gcc is not installed or does not support OpenMP, defaulting to C without OpenMP support
            """
            devito_C()
        else
            devito_omp()
        end
    else
        devito_omp()
    end
end
