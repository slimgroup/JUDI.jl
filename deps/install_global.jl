using Pkg

Pkg.add(PackageSpec(url="https://github.com/JuliaLang/TOML.jl.git"))

using TOML

for k=keys(TOML.parsefile("./Project.toml")["deps"])
    Pkg.add(k)
end
