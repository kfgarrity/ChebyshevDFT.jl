

using ChebyshevQuantum
using ChebyshevDFT
using Test
using Suppressor
using LinearAlgebra

tol_var=1e-5

@testset "test he CI" begin

    #see A.18 in https://iopscience.iop.org/article/10.1088/0143-0807/34/1/111/meta#ejp449896eqn24
    #The spectral decomposition of the helium atom two-electron configuration in terms of hydrogenic orbitals
    #Joel Hutchinson1, Marc Baker1 and Frank Marsiglio3,1,2
    #DOI 10.1088/0143-0807/34/1/111


    #these are exact CI results for hydrogenic orbitals, n=2 or less. Note some are liner combos of the normal Ylm orbitals.
    #href 1s1s , 1s2s+2s1s, 1s2p0+2p01s 2s2s 21-1*211 + 211*21-1 210*210
    href = -0.5* [
    11/2              -32768/64827         -64/729           -448/6561*sqrt(2)          -448/6561 ;
    -32768/64827        2969/729           -4096/84375        2048/28125*sqrt(2)         2048/28125;
    -64/729            -4096/84375           179/128          -15/128*sqrt(2)            -15/128;
    -448/6561*sqrt(2)  2048/28125*sqrt(2) -15/128*sqrt(2)    47/40                    -27/640*sqrt(2);
     -448/6561         2048/28125           -15/128           -27/640*sqrt(2)            779/640] 
    
    vals, vects = eigen(href)
    @test -2.8334051759324277 ≈ vals[1]
    @test  abs(vects[1,1]) - 0.9520 < 1e-3
    
    
    @suppress begin
        #get hydrogen orbitals s,p
        gal50 = ChebyshevDFT.Galerkin.makegal(100, 0, 50.0, α = -1.0, M = 400);

        #this gets numerical hydrogen orbitals
        dat_he_NONE_p = ChebyshevDFT.GalerkinDFT.dft(; fill_str = "2 0 0 1.0 1.0", g = gal50,N = 100, M = 400,  Z = 2.0, niters = 1, mix = 0.0, exc=:hydrogen, exx=1.0, lmax = 1, conv_thr = 1e-11, lmaxrho = 0, mix_lm = false, orthogonalize=false);

        #ci with hydrogen orbitals
        arrI,arrJ,arrH,h = ChebyshevDFT.CI.construct_ham(dat_he_NONE_p, 2, nmax = 2, dense=true);
        valsME, vectsME = eigen(h);
        @test abs(valsME[1] - vals[1]) < 1e-7

    end    

"""
counter 1 Bool[1, 0, 0, 0, 0] Bool[1, 0, 0, 0, 0]  1s1s
counter 2 Bool[1, 0, 0, 0, 0] Bool[0, 1, 0, 0, 0]  1s 2s
counter 3 Bool[1, 0, 0, 0, 0] Bool[0, 0, 1, 0, 0]
counter 4 Bool[1, 0, 0, 0, 0] Bool[0, 0, 0, 1, 0]
counter 5 Bool[1, 0, 0, 0, 0] Bool[0, 0, 0, 0, 1]
counter 6 Bool[0, 1, 0, 0, 0] Bool[1, 0, 0, 0, 0] 2s1s
counter 7 Bool[0, 0, 1, 0, 0] Bool[1, 0, 0, 0, 0]
counter 8 Bool[0, 0, 0, 1, 0] Bool[1, 0, 0, 0, 0]
counter 9 Bool[0, 0, 0, 0, 1] Bool[1, 0, 0, 0, 0]
counter 10 Bool[0, 1, 0, 0, 0] Bool[0, 1, 0, 0, 0]
counter 11 Bool[0, 1, 0, 0, 0] Bool[0, 0, 1, 0, 0]
counter 12 Bool[0, 1, 0, 0, 0] Bool[0, 0, 0, 1, 0]
counter 13 Bool[0, 1, 0, 0, 0] Bool[0, 0, 0, 0, 1]
counter 14 Bool[0, 0, 1, 0, 0] Bool[0, 1, 0, 0, 0]
counter 15 Bool[0, 0, 1, 0, 0] Bool[0, 0, 1, 0, 0]
counter 16 Bool[0, 0, 1, 0, 0] Bool[0, 0, 0, 1, 0]
counter 17 Bool[0, 0, 1, 0, 0] Bool[0, 0, 0, 0, 1]
counter 18 Bool[0, 0, 0, 1, 0] Bool[0, 1, 0, 0, 0]
counter 19 Bool[0, 0, 0, 1, 0] Bool[0, 0, 1, 0, 0]
counter 20 Bool[0, 0, 0, 1, 0] Bool[0, 0, 0, 1, 0]
counter 21 Bool[0, 0, 0, 1, 0] Bool[0, 0, 0, 0, 1]
counter 22 Bool[0, 0, 0, 0, 1] Bool[0, 1, 0, 0, 0]
counter 23 Bool[0, 0, 0, 0, 1] Bool[0, 0, 1, 0, 0]
counter 24 Bool[0, 0, 0, 0, 1] Bool[0, 0, 0, 1, 0]
counter 25 Bool[0, 0, 0, 0, 1] Bool[0, 0, 0, 0, 1]
"""

    #ss
    @test isapprox(h[1,1] , href[1,1], atol=1e-7)
    @test isapprox(1/2*(h[2,2] + h[6,6] + h[2,6]+h[6,2])  , href[2,2], atol=1e-7)
    @test isapprox(h[10,10] , href[3,3], atol=1e-7)
    @test isapprox( h[10,1], href[1,3],  atol=1e-7)
    @test isapprox( 1/sqrt(2)*(h[1,2] + h[1,6]) , href[1,2],  atol=1e-7)
    @test isapprox( 1/sqrt(2)*(h[10,2] + h[10,6]) , href[3,2],  atol=1e-7)

    #sp
    @test  isapprox(  h[end,end], href[end,end],  atol=1e-7)
    @test  isapprox(  h[10,end], href[3,end],  atol=1e-7)
    @test  isapprox( 1/sqrt(2)*(h[2,end]+h[6,end]), href[2,end], atol=1e-7)


end


