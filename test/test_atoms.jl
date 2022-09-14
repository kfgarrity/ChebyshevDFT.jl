

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
        
        energy_lda,converged, vals_r_lda, vects, rho_LM_lda, rall_rs, wall, rhor2_pbe, VH_LM_ref_lda = ChebyshevDFT.SCF.DFT_spin_l_grid_LM(Z=2.0,  fill_str = "He", hydrogen=false, Rmax = 30, N = 70, spherical = true,exc = :lda);

        
        energy_pbe,converged, vals_r_pbe, vects, rho_LM_pbe, rall_rs, wall, rhor2_pbe, VH_LM_ref_pbe = ChebyshevDFT.SCF.DFT_spin_l_grid_LM(Z=2.0,  fill_str = "He", hydrogen=false, Rmax = 30, N = 70, spherical = true,exc = :pbe);

        energy_h,converged, vals_r_h, vects, rho_LM_h, rall_rs, wall, rhor2_h, VH_LM_ref_h = ChebyshevDFT.SCF.DFT_spin_l_grid_LM(Z=2.0,  fill_str = "He", hydrogen=true, Rmax = 30, N = 70, spherical = true,exc = :pbe);

        @test abs(energy_lda_he_nist - energy_lda) < tol_var
        @test abs(vals_r_lda[1] - eval_lda_he_nist) < tol_var

        @test abs(energy_he_pbe - energy_pbe) < tol_var
        @test abs(vals_r_pbe[1] - eval_pbe_he) < tol_var

        @test abs(energy_h - energy_non_int) < tol_var
    end    

    
end


@testset "test ne" begin

    @suppress begin
        energy_lda_ne_nist = -128.233481
        eval_lda_ne_nist = -30.305855
        
        energy_lda,converged, vals_r_lda, vects, rho_LM_lda, rall_rs, wall, rhor2_pbe, VH_LM_ref_lda = ChebyshevDFT.SCF.DFT_spin_l_grid_LM(Z=10.0,  fill_str = "Ne", hydrogen=false, Rmax = 30, N = 70, spherical = true,exc = :lda);

        @test abs(energy_lda_ne_nist - energy_lda) < tol_var
        @test abs(vals_r_lda[1] - eval_lda_ne_nist) < tol_var

        energy_lda,converged, vals_r_lda, vects, rho_LM_lda, rall_rs, wall, rhor2_pbe, VH_LM_ref_lda = ChebyshevDFT.SCF.DFT_spin_l_grid_LM(Z=10.0,  fill_str = "Ne", hydrogen=false, Rmax = 30, N = 70, spherical = false,exc = :lda);

        @test abs(energy_lda_ne_nist - energy_lda) < tol_var
        @test abs(vals_r_lda[1] - eval_lda_ne_nist) < tol_var
    
    end
    
end


@testset "test rn" begin

    @suppress begin
        energy_lda_rn_nist = -21861.346869

        
        energy_lda,converged, vals_r_lda, vects, rho_LM_lda, rall_rs, wall, rhor2_pbe, VH_LM_ref_lda = ChebyshevDFT.SCF.DFT_spin_l_grid_LM(Z=86.0,  fill_str = "Rn", hydrogen=false, Rmax = 40, N = 150, spherical = true,exc = :lda);

        @test abs(energy_lda_rn_nist - energy_lda) < tol_var

    
    end
    
end


@testset "test b spin non-sphere" begin

    @suppress begin
        energy_lda_b_nist = -24.6122090928329
        
        
        ret  = ChebyshevDFT.SCF.DFT_spin_l_grid_LM(Z=5.0,  fill_str = "Be \n 2 1 1 1 0", hydrogen=false, Rmax = 50, N = 50, spherical = false, exc = :pbe, lmax_rho=2);

        @test abs(energy_lda_b_nist - ret[1]) < 1e-4

    end

end
