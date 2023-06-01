using Documenter, JUDI, Weave

# Convert example to documentation markdown file
ex_path = "$(JUDI.JUDIPATH)/../examples/scripts"
doc_path = "$(JUDI.JUDIPATH)/../docs"
weave("$(ex_path)/modeling_basic_2D.jl"; out_path="$(doc_path)/src/tutorials/quickstart.md", doctype="github")
weave("$(ex_path)/imaging_conditions.jl"; out_path="$(doc_path)/src/tutorials/imaging_conditions.md", doctype="github")

# Create documentation
makedocs(sitename="JUDI documentation",
         doctest=false, clean=true,
         authors="Mathias Louboutin",
         pages = Any[
             "Home" => "index.md",
             "About" => "about.md",
             "Installation" => "installation.md",
             "Getting Started" => "basics.md",
             "JUDI API" => Any[
                "Abstract vectors" => "abstract_vectors.md",
                "Data Structures" => "data_structures.md",
                "Linear Operators" => "linear_operators.md",
                "Preconditioners" => "preconditioners.md",
                "Input/Output" => "io.md",
                "Helper Functions" => "helper.md"],
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