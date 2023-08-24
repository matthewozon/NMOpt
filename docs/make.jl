using Documenter, DocumenterTools
using NMOpt

makedocs(
    sitename = "NMOpt",
    format = Documenter.HTML(),
    modules = [NMOpt],
    pages = [
    "Home" => "index.md",
    ],
    doctest = true,
)

deploydocs(repo = "github.com/matthewozon/NMOpt.git",branch = "master")

