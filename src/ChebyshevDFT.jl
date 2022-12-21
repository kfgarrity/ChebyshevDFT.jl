module ChebyshevDFT

greet() = print("Hello World!")

#using ChebyshevQuantum

#include("AngMom.jl")
#using ..AngMom:construct_real_gaunt_indirect
#using ..AngMom:precalc_sphere
#construct_real_gaunt_indirect()
#precalc_sphere()

include("AngularIntegration.jl")

include("LDA.jl")
include("UseLibxc.jl")

#include("Hartree.jl")
#include("SCF.jl")
#include("Inverse.jl")



#include("Search.jl")

#include("Galerkin.jl")
#include("GalerkinNEW.jl")
include("GalerkinNEW2.jl")

#include("G2.jl")
include("GalerkinDFT.jl")

end # module
