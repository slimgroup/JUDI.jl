using Documenter, JUDI, Weave

import JUDI: judiMultiSourceVector
import Base.show

# Some dispatch needed for Weave
show(io::IO, ::MIME, ms::judiMultiSourceVector) = println(io, "$(typeof(ms)) wiht $(ms.nsrc) sources")
show(io::IO, ::MIME, m::Model) = print(io, "Model (n=$(m.n), d=$(m.d), o=$(m.o)) with parameters $(keys(m.params))")
show(io::IO, ::MIME, A::PhysicalParameter) = println(io, "$(typeof(A)) of size $(A.n) with origin $(A.o) and spacing $(A.d)")
show(io::IO, ::MIME, G::Geometry) = println(io, "$(typeof(G)) wiht $(length(G.nt)) sources")

# Convert example to documentation markdown file
ex_path = "$(JUDI.JUDIPATH)/../examples/scripts"
weave("$(ex_path)/modeling_basic_2D.jl"; out_path="src/tutorials/", doctype="github")

# Create documentation
makedocs(sitename="JUDI documentation",
         doctest=false, clean=true,
         authors="Mathias Louboutin",
         pages = Any[
             "Home" => "index.md",
             "About" => "about.md",
             "Installation" => "installation.md",
             "JUDI API" => Any[
                "Abstract vectors" => "abstract_vectors.md",
                "Data Structures" => "data_structures.md",
                "Linear Operators" => "linear_operators.md",
                "Input/Output" => "io.md",
                "Helper Functions" => "helper.md"],
             "Getting Started" => "basics.md",
             "Inversion" => "inversion.md",
             "Tutorials" => map(
                s -> "tutorials/$(s)",
                sort(filter(x->endswith(x, ".md"), readdir(joinpath(@__DIR__, "src/tutorials/"))))),
              "Devito backend reference" => "pysource.md"],
         format = Documenter.HTML(
                # assets = ["assets/slim.css"],
                prettyurls = get(ENV, "CI", nothing) == "true"),
        )

# Deploy documentation
deploydocs(repo="github.com/slimgroup/JUDI.jl")