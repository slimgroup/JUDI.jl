using PyCall

# Check devito version and update if necessary
struct DevitoException <: Exception
    msg::String
end

pk = try
    pyimport("pkg_resources")
catch e
    run(PyCall.python_cmd(`-m pip install --user setuptools`))
    pyimport("pkg_resources")
end

################## Devito ##################
# pip command
dvver = "4.8.10"
cmd = PyCall.python_cmd(`-m pip install --user devito\[extras,tests\]\>\=$(dvver)`)

try
    dv_ver = VersionNumber(split(pk.get_distribution("devito").version, "+")[1])
    if dv_ver < VersionNumber(dvver)
        @info "Devito  version too low, updating to >=$(dvver)"
        run(cmd)
    end
catch e
    @info "Devito  not installed, installing with PyCall python"
    run(cmd)
end

################## Matplotlib ##################
try
    mpl = pyimport("matplotlib")
catch e
    run(PyCall.python_cmd(`-m pip install --user matplotlib`))
end
