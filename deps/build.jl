using PythonCall

# Check devito version and update if necessary
struct DevitoException <: Exception
    msg::String
end

if PythonCall.C.CondaPkg.backend() == :Null
    pyexe = PythonCall.python_executable_path()
else
    pyexe = CondaPkg.withenv() do
        condapy = CondaPkg.which("python")
        return condapy
    end
end

end

cmd(x::String) = Cmd(convert(Vector{String}, split(x, " ")))

# Check if can call with '--user' flag
pip = try
    run(Cmd(`$(pyexe) -m pip install --user --no-cache-dir --upgrade pip`))
    "pip install --user --no-cache-dir"
catch e
    "pip install --no-cache-dir"
end

pk = try
    pyimport("pkg_resources")
catch e
    run(cmd("$(pyexe) -m $(pip) setuptools"))
    pyimport("pkg_resources")
end

################## JOLI ##################
try
    pyimport("pywt")
catch e
    run(cmd("$(pyexe) -m $(pip) PyWavelets"))
end

################## Devito ##################
# pip command
dvver = "4.8.14"
dv_cmd = "$(pyexe) -m $(pip) devito[extras,tests]>=$(dvver)"

try
    dv_ver = VersionNumber(split(pk.get_distribution("devito").version, "+")[1])
    if dv_ver < VersionNumber(dvver)
        @info "Devito  version too low, updating to >=$(dvver)"
        run(cmd(dv_cmd))
    end
catch e
    @info "Devito  not installed, installing with PythonCall python"
    run(cmd(dv_cmd))
end

################## Matplotlib ##################
try
    pyimport("matplotlib")
catch e
    run(cmd("$(pyexe) -m $(pip) matplotlib"))
end
