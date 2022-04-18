module ChebyshevDFT

greet() = print("Hello World!")

using ChebyshevQuantum

include("AngMom.jl")
#using ..AngMom:fill_dict
#fill_dict()
#using ..AngMom:inddict
#inddict()

using ..AngMom:construct_real_gaunt_indirect
construct_real_gaunt_indirect()


include("LDA.jl")
include("UseLibxc.jl")
include("Hartree.jl")
include("SCF.jl")


end # module
