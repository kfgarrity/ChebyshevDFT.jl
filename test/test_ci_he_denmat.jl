

using ChebyshevQuantum
using ChebyshevDFT
using Test
using Suppressor
using LinearAlgebra
using Arpack

tol_var=1e-5


@testset "test he CI denmat" begin

    @suppress begin    
        #see https://pubs.aip.org/aip/jcp/article/71/10/4142/530273/Piecewise-polynomial-configuration-interaction
        # J. Chem. Phys. 71, 41424163 (1979)
        #info on L-dependent Helium CI terms

        #see also https://arxiv.org/pdf/1207.7284.pdf table IV

        #see also table 7.1 in MOLECULAR ELECTRONIC-STRUCTURE THEORY
        
        gal8 = ChebyshevDFT.Galerkin.makegal(110, 0, 8.0, α = -1.0, M = 400, orthogonalize=true);
        dat_he_s8 = ChebyshevDFT.GalerkinDFT.dft(; fill_str = "2 0 0 1.0 1.0", g = gal8,N = 110, M = 160,
                                                 Z = 2.0, niters = 100, mix = 0.7, exc=:none, exx=1.0, lmax = 0,
                                                 conv_thr = 1e-11, lmaxrho = 0, mix_lm = false, orthogonalize=false);


        ArrI, ArrJ, ArrH, H, vals, vects, basis_up, basis_dn, s_up, s_dn, denmat, VECTS_new_ref, keep_dict = ChebyshevDFT.CI.run_CI(dat_he_s8, 2, nmax = [50, 0], dense=false, symmetry=true);

        dvals, dvects = eigen(Hermitian(denmat[:,:,1,1,1]));
        dat_he_temp2 = deepcopy(dat_he_s8);
        dat_he_temp2.VECTS_small[:,:,1,1,1] = dvects[:,end:-1:1];
        dat_he_temp2.VECTS_small[:,:,2,1,1] = dvects[:,end:-1:1];


        
        ArrI, ArrJ, ArrH, H, vals12, vects, basis_up, basis_dn, s_up, s_dn, denmat12, VECTS_new_ref12, keep_dict = ChebyshevDFT.CI.run_CI(dat_he_temp2, 2, nmax = [12, 0], dense=false, symmetry=true);


        @test abs(vals[1] - -2.879028765) < 5e-5  #2.879028758 largest basis
        @test abs(vals12[1] - -2.879028765) < 5e-5  #2.879028758 largest basis
        @test abs(vals[1] - vals12[1]) < 5e-7  #2.879028758 largest basis
        
        #-------------
        
        dat_he_temp2 = deepcopy(dat_he_s8);
        dat_he_temp2.VECTS_small[:,:,:,:,:] = VECTS_new_ref

        ArrI, ArrJ, ArrH, H, vals12_trun, vects, basis_up, basis_dn, s_up, s_dn, denmat12_trun, VECTS_new_ref12, keep_dict = ChebyshevDFT.CI.run_CI(dat_he_temp2, 2, nmax = [12, 0], dense=false, symmetry=true);

        @test abs(vals[1] - -2.879028765) < 5e-5  #2.879028758 largest basis
        @test abs(vals12_trun[1] - -2.879028765) < 5e-5  #2.879028758 largest basis
        @test abs(vals[1] - vals12_trun[1]) < 5e-7  #2.879028758 largest basis
        

    end    
end


