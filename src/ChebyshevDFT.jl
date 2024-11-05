module ChebyshevDFT

greet() = print("Hello World!")

#using ChebyshevQuantum

include("AngMom.jl")
using ..AngMom:construct_real_gaunt_indirect
using ..AngMom:precalc_sphere
construct_real_gaunt_indirect()
precalc_sphere()


include("AngularIntegration.jl")
#include("AngularIntegrationX.jl")

AngularIntegration.fill_gaunt()

include("LDA.jl")
include("UseLibxc.jl")

#include("Hartree.jl")
#include("SCF.jl")
#include("Inverse.jl")


#include("Search.jl")

#include("Galerkin.jl")
#include("GalerkinNEW.jl")

include("GalerkinNEW3.jl")
#include("GalerkinNEWX.jl")

#include("G2.jl")
include("GalerkinDFT.jl")
#include("GalerkinDFTX.jl")

include("CI.jl")

include("GalerkinInverse.jl")

end # module
