using ChebyshevDFT
using Test
using Suppressor

tol_var=1e-4

g20_m2_100 = ChebyshevDFT.Galerkin.makegal(100,0.0,20.0; α=-2.0, M = 250);

@testset "boron_lda" begin

    b_lda = [-6.562265e+00, -3.557961e-01 , -1.511434e-01, -1.440930e-01, -6.550154e+00, -3.185737e-01, -1.230608e-01]
    N = 100;M=250;
    ret1=missing
    @suppress begin
        ret1 = ChebyshevDFT.GalerkinDFT.dft(; fill_str = "Be; 2 1 0 1.0 0.0", g = g20_m2_100,N = N, M = M,  Z = 5.0, niters = 100, mix = 0.8, exc=:vwn, lmax = 2, conv_thr = 1e-7, lmaxrho = 6, mix_lm = true);
    end
    vals = ret1.VALS
    @test sum(abs.(b_lda -     [vals[1,1, 1,1],vals[2,1, 1,1], vals[1,1, 2,2], vals[1,1, 2,1], vals[1,2, 1,1],vals[2,2, 1,1], vals[1,2,2,1]]) .< tol_var) == 7


end

@testset "boron_pbe" begin

    b_pbe = [-6.629113e+00, -3.550533e-01, -1.532964e-01, -1.332353e-01, -6.619787e+00, -3.195205e-01, -1.220585e-01]
    N = 90;M=210;
    ret1 = missing
    @suppress begin
        ret1 = ChebyshevDFT.GalerkinDFT.dft(; fill_str = "Be; 2 1 0 1.0 0.0", g = g20_m2_100,N = N, M = M,  Z = 5.0, niters = 100, mix = 0.8, exc=:pbe, lmax = 2, conv_thr = 1e-7, lmaxrho = 8, mix_lm = true);
    end
    vals = ret1.VALS

    @test sum(abs.(b_pbe -     [vals[1,1, 1,1],vals[2,1, 1,1], vals[1,1, 2,2], vals[1,1, 2,1], vals[1,2, 1,1],vals[2,2, 1,1], vals[1,2,2,1]]) .< tol_var) == 7

    
end


@testset "boron_hf" begin

    #hel command ../atomic --Z=5 --Rmax=20.0 --lmax=2 --nela=3 --nelb=2 --symmetry=0 --mmax=6 --nelem=20 --maxit=100 > b_hf_rmax20_lmax2_sym0
    
    b_hf = [ -7.701199806795829,
             -0.5452169252560611,
             -0.3188087451032327,
             -7.686185466519145,
             -0.4461240347636291
             ]
    
    N = 80;M=150;
    ret1 = missing
    @suppress begin
        ret1 = ChebyshevDFT.GalerkinDFT.dft(; fill_str = "Be; 2 1 0 1.0 0.0", g = g20_m2_100,N = N, M = M,  Z = 5.0, niters = 100, mix = 0.8, exc=:none, lmax = 2, conv_thr = 1e-8, lmaxrho = 10, mix_lm = true, exx=1.0);
    end
    vals = ret1.VALS
    

    @test sum(abs.(b_hf -   [vals[1,1, 1,1],vals[2,1, 1,1], vals[1,1, 2,2], vals[1,2, 1,1], vals[2,2, 1,1]]) .< tol_var) == 5

              
end


