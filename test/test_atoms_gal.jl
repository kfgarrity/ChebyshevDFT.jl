

using ChebyshevQuantum
using ChebyshevDFT
using Test
using Suppressor

tol_var=1e-5

@testset "test he" begin

    @suppress begin
        energy_lda_he_nist = -2.834836
        eval_lda_he_nist = -0.570425
        
        energy_he_pbe = -2.8929348668473533
        eval_pbe_he = -5.792907e-01

        energy_non_int = -4.0
        
        galF = ChebyshevDFT.Galerkin.makegal(70, 0, 20.0, α = 0.1, M = 200, orthogonalize=false);

        dat_he_lda = ChebyshevDFT.GalerkinDFT.dft(Z=2.0,  fill_str = "He", g = galF, exc = :lda);

        dat_he_pbe = ChebyshevDFT.GalerkinDFT.dft(Z=2.0,  fill_str = "He", g = galF, exc = :pbe);
        
        @test abs(energy_lda_he_nist - dat_he_lda.etot) < tol_var
        @test abs(eval_lda_he_nist - dat_he_lda.VALS[1]) < tol_var

        @test abs(energy_he_pbe - dat_he_pbe.etot) < tol_var
        @test abs(eval_pbe_he - dat_he_pbe.VALS[1]) < tol_var
        

        dat_he_nonint = ChebyshevDFT.GalerkinDFT.dft(Z=2.0, fill_str = "He", g = galF, exc = :hydrogen);
        
        @test abs(energy_non_int - dat_he_nonint.etot) < tol_var
 
        energy_h_exact = -0.5
        dat_h_hf = ChebyshevDFT.GalerkinDFT.dft(Z=1.0, fill_str = "1 0 0 1.0 0.0", g = galF, exc = :none, exx = 1.0);
        @test abs(energy_h_exact - dat_h_hf.etot) < tol_var
       
        
    end    

    
end


@testset "test he" begin

    @suppress begin
        energy_lda_he_nist = -2.834836
        eval_lda_he_nist = -0.570425
        
        energy_he_pbe = -2.8929348668473533
        eval_pbe_he = -5.792907e-01

        energy_non_int = -4.0
        
        galT = ChebyshevDFT.Galerkin.makegal(70, 0, 20.0, α = 0.1, M = 200, orthogonalize=true);

        dat_he_lda = ChebyshevDFT.GalerkinDFT.dft(Z=2.0,  fill_str = "He", g = galT, exc = :lda);

        dat_he_pbe = ChebyshevDFT.GalerkinDFT.dft(Z=2.0,  fill_str = "He", g = galT, exc = :pbe);
        
        @test abs(energy_lda_he_nist - dat_he_lda.etot) < tol_var
        @test abs(eval_lda_he_nist - dat_he_lda.VALS[1]) < tol_var

        @test abs(energy_he_pbe - dat_he_pbe.etot) < tol_var
        @test abs(eval_pbe_he - dat_he_pbe.VALS[1]) < tol_var
        

        dat_he_nonint = ChebyshevDFT.GalerkinDFT.dft(Z=2.0, fill_str = "He", g = galT, exc = :hydrogen);
        
        @test abs(energy_non_int - dat_he_nonint.etot) < tol_var

        energy_h_exact = -0.5
        dat_h_hf = ChebyshevDFT.GalerkinDFT.dft(Z=1.0, fill_str = "1 0 0 1.0 0.0", g = galT, exc = :none, exx = 1.0);
        @test abs(energy_h_exact - dat_h_hf.etot) < tol_var

    end    

    
end


