

using ChebyshevQuantum
using ChebyshevDFT
using Test
using Suppressor
using LinearAlgebra
using Arpack

tol_var=1e-5

#note this test takes longer
@testset "test he CI more" begin

    @suppress begin    
        #see https://pubs.aip.org/aip/jcp/article/71/10/4142/530273/Piecewise-polynomial-configuration-interaction
        # J. Chem. Phys. 71, 41424163 (1979)
        #info on L-dependent Helium CI terms

        #see also https://arxiv.org/pdf/1207.7284.pdf table IV

        #see also table 7.1 in MOLECULAR ELECTRONIC-STRUCTURE THEORY
        
        gal8 = ChebyshevDFT.Galerkin.makegal(100, 0, 8.0, α = -1.0, M = 400);
        dat_he_p8 = ChebyshevDFT.GalerkinDFT.dft(; fill_str = "2 0 0 1.0 1.0", g = gal8,N = 100, M = 150,
                                                 Z = 2.0, niters = 100, mix = 0.7, exc=:none, exx=1.0, lmax = 1,
                                                 conv_thr = 1e-11, lmaxrho = 0, mix_lm = false, orthogonalize=false);

        arrI,arrJ,arrH,h_sonly,s_up, s_dn = ChebyshevDFT.CI.construct_ham(dat_he_p8, 2, nmax = [40, 0], dense=false, symmetry=true);
        vals, vects = eigs(h_sonly, which=:SR); 

        #extrapolated, see TABLE IX. "L"-limit energies E L, increments LlEL, and errors, in atomic units.
        
        @test abs(vals[1] - -2.879028765) < 5e-5  #2.879028758 largest basis
#        2.879028765
        
        arrI,arrJ,arrH,h_sp,s_up, s_dn = ChebyshevDFT.CI.construct_ham(dat_he_p8, 2, nmax = [35, 30], dense=false, symmetry=true);
        vals, vects = eigs(h_sp, which=:SR); 
        @test abs(vals[1] - -2.900516220) < 1e-4   #2.900516220 largest basis

        he_exact = -2.90372437703411959831115924519440443
        @test abs(vals[1] - he_exact) < 1e-2

        #note, convergence is slow with l and n
    end    
end


#note this test takes longer as well
@testset "test he CI more" begin

    @suppress begin    
        #see https://pubs.aip.org/aip/jcp/article/71/10/4142/530273/Piecewise-polynomial-configuration-interaction
        # J. Chem. Phys. 71, 41424163 (1979)
        #info on L-dependent Helium CI terms

        #see also https://arxiv.org/pdf/1207.7284.pdf table IV

        #see also table 7.1 in MOLECULAR ELECTRONIC-STRUCTURE THEORY
        
        gal8_orth = ChebyshevDFT.Galerkin.makegal(100, 0, 8.0, α = -1.0, M = 400, orthogonalize = true);
        dat_he_s8 = ChebyshevDFT.GalerkinDFT.dft(; fill_str = "2 0 0 1.0 1.0", g = gal8_orth,N = 100, M = 150,
                                                 Z = 2.0, niters = 100, mix = 0.7, exc=:none, exx=1.0, lmax = 0,
                                                 conv_thr = 1e-11, lmaxrho = 0, mix_lm = false, orthogonalize=false);

        ArrI, ArrJ, ArrH, H, vals, vects, basis_up, basis_dn, s_up, s_dn, denmat, VECTS_new, keep_dict = ChebyshevDFT.CI.run_CI(dat_he_s8, 2, nmax = [40, 0], dense=false, symmetry=true);
        
        #extrapolated, see TABLE IX. "L"-limit energies E L, increments LlEL, and errors, in atomic units.
        
        @test abs(vals[1] - -2.879028765) < 5e-5  #2.879028758 largest basis
    end    
end



#note this test takes longer as well
@testset "test he high spin" begin

    @suppress begin
        gal16_orth = ChebyshevDFT.Galerkin.makegal(200, 0, 16.0, α = -1.0, M = 400, orthogonalize = true);

        dat_l1_s1 = ChebyshevDFT.GalerkinDFT.dft(; fill_str = "1 0 0 1.0 0.0; 2 0 0 1.0 0.0", g = gal16_orth, Z = 2.0, niters = 100, mix = 0.7, exc=:none, exx=1.0, lmax = 1, conv_thr = 1e-10, lmaxrho = 0, mix_lm = false);

        dat_temp, energies, denmat = ChebyshevDFT.CI.run_CI_iterate(dat_l1_s1, 2, nmax = [10,8], thr=1e-11, addnum = 2, niters=30, itertol = 1e-6, symmetry=true);

        he_s1_exact =  -2.17522937823679130573897827820681125  #https://pos.sissa.it/353/060/pdf#:~:text=The%20nonrelativistic%20energy%20levels%20of,number%20of%20basis%20functions%20N.
        #Nonrelativistic energy levels of helium atom D.T. Aznabayev
        @test abs(energies[end] - he_s1_exact) < 1e-3
    end    
    
end

