using PyCall

# Check devito version and update if necessary
struct DevitoException <: Exception
    msg::String
end

pk = pyimport("pkg_resources")
python = PyCall.pyprogramname

################## Devito ##################
# pip command
cmd = Cmd([python, "-m", "pip", "install", "-U", "--user", "devito>=4.4"])

try
    dv_ver = split(pk.get_distribution("devito").versionn, "+")[1]
    if cmp(dv_ver, "4.4") < 0
        @info "Devito  version too low, updating to >=4.4"
        run(cmd)
    end
catch e
    @info "Devito  not installed, installing with PyCall python"
    run(cmd)
end


################## Matplotlib ##################
# pip command
cmd = Cmd([python, "-m", "pip", "install", "--user", "matplotlib"])
try
    mpl = pyimport("matplotlib")
catch e
    run(cmd)
end
