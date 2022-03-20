using Documenter, JUDI

makedocs(sitename="JUDI documentation",
         doctest=false, clean=true,
         authors="Mathias Louboutin",
         pages = Any[
             "Home" => "index.md",
             "About" => "about.md",
             "Installation" => "installation.md",
             "Abstract vectors" => "abstract_vectors.md",
             "Data Structures" => "data_structures.md",
             "Linear Operators" => "linear_operators.md",
             "Input/Output" => "io.md",
             "Helper Functions" => "helper.md",
             "Inversion" => "inversion.md",
             "Tutorial" => "tutorials.md",
             "Devito backend reference" => "pysource.md"],
         format = Documenter.HTML(
                # assets = ["assets/slim.css"],
                prettyurls = get(ENV, "CI", nothing) == "true"),
        )

deploydocs(repo="github.com/slimgroup/JUDI.jl")