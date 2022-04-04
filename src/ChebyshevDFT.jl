module ChebyshevDFT

greet() = print("Hello World!")

using ChebyshevQuantum

include("AngMom.jl")
using ..AngMom:fill_dict
fill_dict()

include("LDA.jl")
include("Hartree.jl")
include("SCF.jl")


end # module
