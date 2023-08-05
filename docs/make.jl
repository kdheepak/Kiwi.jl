using Kiwi
using Documenter

DocMeta.setdocmeta!(Kiwi, :DocTestSetup, :(using Kiwi); recursive=true)

makedocs(;
    modules=[Kiwi],
    authors="Dheepak Krishnamurthy",
    repo="https://github.com/kdheepak/Kiwi.jl/blob/{commit}{path}#{line}",
    sitename="Kiwi.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kdheepak.github.io/Kiwi.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kdheepak/Kiwi.jl",
    devbranch="main",
)
