using ChebyshevDFT
using Test
using Suppressor

tol_var=1e-8

N=70
M=210

g20 = ChebyshevDFT.Galerkin.makegal(N,0.0, 20.0; α=-1.0, M =M);


@testset "neon hf" begin

    ne_hf = [-32.7724427931715070, -1.9303908799674412, -0.8504096503997881, 0.0137688135347404, 0.0253904652247837]

    
    ret1=missing
    @suppress begin
        ret1 = ChebyshevDFT.GalerkinDFT.dft(; fill_str = "Ne", g = g20,N = N, M = M,  Z = 10.0, niters = 100, mix = 0.3, exc=:none, lmax = 1, conv_thr = 5e-10, lmaxrho = 2, mix_lm = false, exx=1.0);
    end
    v = ret1[1]
    @test sum(abs.(ne_hf -  [v[1,1,1,1], v[2,1,1,1,1],v[1,1,2,2], v[3,1,1,1,1],v[2,1,2,2]])) < tol_var


end




