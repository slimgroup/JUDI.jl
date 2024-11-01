using PythonCall

# Check devito version and update if necessary
struct DevitoException <: Exception
    msg::String
end

if PythonCall.C.CondaPkg.backend() == :Null
    pyexe = PythonCall.python_executable_path()
else
    @info "Using $(PythonCall.C.CondaPkg.backend()) as the CondaPkg backend"
    pyexe = PythonCall.C.CondaPkg.withenv() do
        condapy = PythonCall.C.CondaPkg.which("python")
        return condapy
    end
end

pk = try
    pyimport("pkg_resources")
catch e
    run(Cmd(`$(pyexe) -m pip install -U --user --no-cache-dir setuptools`))
    pyimport("pkg_resources")
end

################## JOLI ##################
run(Cmd(`$(pyexe) -m pip install -U --user --no-cache-dir PyWavelets`))

################## Devito ##################
# pip command
dvver = "4.8.10"
cmd = Cmd(`$(pyexe) -m pip install --user --no-cache-dir devito\[extras,tests\]\>\=$(dvver)`)

try
    dv_ver = VersionNumber(split(pk.get_distribution("devito").version, "+")[1])
    if dv_ver < VersionNumber(dvver)
        @info "Devito  version too low, updating to >=$(dvver)"
        run(cmd)
    end
catch e
    @info "Devito  not installed, installing with PythonCall python"
    run(cmd)
end

################## Matplotlib ##################
try
    mpl = pyimport("matplotlib")
catch e
    run(Cmd(`$(pyexe) -m pip install --user --no-cache-dir matplotlib`))
end
