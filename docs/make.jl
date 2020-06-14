using EmDee
using Documenter

makedocs(;
    modules=[EmDee],
    authors="Charlles Abreu <abreu@eq.ufrj.br> and contributors",
    repo="https://github.com/craabreu/EmDee.jl/blob/{commit}{path}#L{line}",
    sitename="EmDee.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://craabreu.github.io/EmDee.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/craabreu/EmDee.jl",
)
