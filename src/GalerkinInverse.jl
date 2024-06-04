# https://www.nature.com/articles/s41467-019-12467-0

module GInverse

using ..GalerkinDFT:prepare
using ..GalerkinDFT:get_rho
using ..Galerkin:get_gal_rep_matrix
using LinearAlgebra
using Optim

function GalerkinInverseTest(GAL, fill_str)

    Z = 1.0
    lmax = 0
    exc = :hydrogen
    N = GAL.N
    M = GAL.goodM
    lmaxrho = 0
    mix_lm = false
    orthogonalize = false
    soft_coulomb = 0.0
    exx = 0
    
    
    println("time prepare")
    @time Z, nel, filling, nspin, lmax, V_C, V_L, D1, D2, S, invsqrtS, invS, VECTS_start, VALS, funlist, gga, LEB, R, gbvals2, nmax, hf_sym, hf_sym_big, mat_n2m, mat_m2n, R5, LINOP, lm_dict, dict_lm, big_code, Sbig, S5, Sold, sqrtS, exx =
        prepare(Z, fill_str, lmax, exc, N, M, GAL, lmaxrho, mix_lm, orthogonalize, soft_coulomb, exx)
    

    VECTS = deepcopy(VECTS_start)
    ll = 0.0
    
    vals, vects = eigen(Hermitian(D2 + V_C + ll*(ll+1)*V_L),Hermitian( S))
    #h = D2 + V_C + ll*(ll+1)*V_L

    println("vals $(vals[1:3])")
    println("vals diff $(vals[1:3] - VALS[1:3])")

    rho_R2, rho_dR, rho_rs_M, drho_rs_M_LM, MP = get_rho(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, GAL, true, exx, nmax, S5, orthogonalize)
    

    #start search

    MAT = zeros(N, N)
    G = zeros(N)
    rho_target = rho_R2[:,1,1,1]
    rho = zeros(N-1)
    #v_ks = zeros(N-1)
    v_ks = zeros(N-1)

    
    
    function f(v_ks)

        V_KS = get_gal_rep_matrix(v_ks, GAL; M = M)
        vals, vects = eigen(Hermitian(D2 + V_C + V_KS),Hermitian( S))
        VECTS[:,1,1,1,1] = vects[:,1]
        rho_R2, rho_dR, rho_rs_M, drho_rs_M_LM, MP = get_rho(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, GAL, true, exx, nmax, S5, orthogonalize)
        rho .= rho_R2[:,1,1,1]
        ret = 1000*sum((rho_R2[:,1,1,1] - rho_target).^2)
        println("ret $ret vals $(vals[1])")
        return ret
        
    end

    f(v_ks)

    i = 1
    grad_vec = zeros(N-1)
    
    function my_grad!(grad, v_ks)


        println("mygrad input ", v_ks)
        println("mygrad grad  ", grad)
#        println()
        V_KS = get_gal_rep_matrix(v_ks, GAL; M = M)
        MAT .= 0.0
        MAT[1:N-1, 1:N-1] = D2 + V_C + V_KS - I(N-1) * vals[i]
        MAT[1:N-1,N] = 2 * vects[:,i]
        MAT[N, 1:N-1] = 2 *  vects[:,i]'

        G .= 0.0
        G[1:N-1] = 4 * (rho_target - rho) .* vects[:,i]

#        println("MAT")
#        println(MAT)
#        println()
#        println("G")
#        println(G)
        
        grad[:] = 1000*(MAT\G)[1:N-1]
#        println()
#        println("grad")
#        println(grad)
        println("sum abs ", sum(abs.(grad)))
    end

    #ret = optimize(f,  v_ks * 0.9)

    v_ks = rand(N-1) * 0.1
    println("start v_ks $v_ks")
    ret = f(v_ks)
    println("start $ret")
    my_grad!(grad_vec, v_ks)
    println("sum temp ", sum(abs.(grad_vec)))
    
    #    opts = Optim.Options( f_tol = 1e-5, g_tol = 1e-5, iterations = 500, store_trace = true, show_trace = false)
    opts = Optim.Options( f_tol = 1e-9, g_tol = 1e-9, iterations = 500000, store_trace = true, show_trace = false)
    
    ret = optimize(f,  v_ks, opts)

#    ret = optimize(f,my_grad!, v_ks, BFGS(), opts)
    
    #ret = missing
    
    return ret
end





end #end module
