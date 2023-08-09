using KiwiConstraintSolver
using Documenter

DocMeta.setdocmeta!(KiwiConstraintSolver, :DocTestSetup, :(using KiwiConstraintSolver); recursive=true)

makedocs(;
    modules=[KiwiConstraintSolver],
    authors="Dheepak Krishnamurthy",
    repo="https://github.com/kdheepak/KiwiConstraintSolver.jl/blob/{commit}{path}#{line}",
    sitename="KiwiConstraintSolver.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kdheepak.github.io/KiwiConstraintSolver.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kdheepak/KiwiConstraintSolver.jl",
    devbranch="main",
)
