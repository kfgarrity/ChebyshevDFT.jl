

using ChebyshevQuantum
using ChebyshevDFT
using Test
using Suppressor
using LinearAlgebra
using Arpack

tol_var=1e-5

"note this test takes longer"
@testset "test he CI more" begin

    @suppress begin    
        #see https://pubs.aip.org/aip/jcp/article/71/10/4142/530273/Piecewise-polynomial-configuration-interaction
        # J. Chem. Phys. 71, 41424163 (1979)
        #info on L-dependent Helium CI terms

        gal8 = ChebyshevDFT.Galerkin.makegal(100, 0, 8.0, α = -1.0, M = 400);
        dat_he_p8 = ChebyshevDFT.GalerkinDFT.dft(; fill_str = "2 0 0 1.0 1.0", g = gal8,N = 100, M = 150,
                                                 Z = 2.0, niters = 100, mix = 0.7, exc=:none, exx=1.0, lmax = 1,
                                                 conv_thr = 1e-11, lmaxrho = 0, mix_lm = false, orthogonalize=false);

        arrI,arrJ,arrH,h_sonly = ChebyshevDFT.CI.construct_ham(dat_he_p8, 2, nmax = [40, 0], dense=false, symmetry=true);
        vals, vects = eigs(h_sonly, which=:SR); 
        @test abs(vals[1] - -2.879028758) < 5e-5

        arrI,arrJ,arrH,h_sp = ChebyshevDFT.CI.construct_ham(dat_he_p8, 2, nmax = [40, 35], dense=false, symmetry=true);
        vals, vects = eigs(h_sp, which=:SR); 
        @test abs(vals[1] - -2.900516199) < 5e-5

        he_exact = −2.90372437703411959831115924519440443
        @test abs(vals[1] - he_exact) < 1e-2

        #note, convergence is slow with l and n
    end    
end


