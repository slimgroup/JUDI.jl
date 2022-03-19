using Documenter, JUDI

makedocs(sitename="JUDI documentation",
         doctest=false, clean=true,
         authors="Mathias Louboutin",
         pages = Any[
             "Home" => "index.md",
             "About" => "about.md",
             "Abstract vectors" => "abstract_vectors.md",
             "Data Structures" => "data_structures.md",
             "Linear Operators" => "linear_operators.md",
             "Input/Output" => "io.md",
             "Helper Functions" => "helper.md",
             "Tutorial" => "tutorials.md",
             "JUDI API reference" => "judiref.md",
         ])

deploydocs(repo="github.com/slimgroup/JUDI.jl")