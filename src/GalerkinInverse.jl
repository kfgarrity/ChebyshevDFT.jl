# https://www.nature.com/articles/s41467-019-12467-0

module GInverse

using ..GalerkinDFT:prepare
using ..GalerkinDFT:get_rho
using ..Galerkin:get_gal_rep_matrix
using ..Galerkin:get_gal_rep_matrix_matrix
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
    

    cval = 1e5
    regularization = 1e3
    
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

    rho_target = (rho_rs_M .* R.^2 )[:]

    t = (mat_n2m * vects[:,1])
    rho_target2 = t.^2 / (GAL.b-GAL.a) * 2

#    println("rho ", rho_target2./rho_target)
#    return
    
    #start search

    MAT = zeros(N, N)
    G = zeros(N)
    #rho_target = rho_R2[:,1,1,1]
    #rho_target = vects[:,1].^2
    rho = zeros(N-1)
    #v_ks = zeros(N-1)


    MAT_VKS = get_gal_rep_matrix_matrix( GAL, N,  M);
    V_KS = zeros(N-1, N-1)
    function apply_mat(vec, X)
        for i = 1:N-1
            for j = 1:N-1
                X[i,j] = sum(MAT_VKS[i,j,:] .* vec)
            end
        end
    end

#    r = rand(N-1)
#    @time V_KS = get_gal_rep_matrix(r, GAL; M = M)
#    @time begin 
#        V_KS2 = zeros(N-1, N-1)
#        apply_mat(r, V_KS2)
#    end
#    println("diff ", sum(abs.(V_KS - V_KS2)))
#    return
    
    function f(v_ks)
        
        apply_mat(v_ks, V_KS)
        vals, vects = eigen(Hermitian(D2 + V_C + V_KS),Hermitian( S))
        VECTS[:,1,1,1,1] = vects[:,1]
        rho = (mat_n2m * vects[:,1]).^2 / (GAL.b-GAL.a) * 2
        retX = cval*sum((rho - rho_target).^2)
        retX += regularization * sum(v_ks.^2)
        return retX
        
    end
    
#    f(v_ks)
    
    i = 1
    grad_vec = zeros(N-1)
    X = zeros(N-1, N-1)
    
    function my_grad!(grad, v_ks)


#        println("mygrad input ", v_ks)
#        println("mygrad grad  ", grad)
              #        println()
              #        V_KS = get_gal_rep_matrix(v_ks, GAL; M = M)
        apply_mat(v_ks, V_KS)
              
        #V_KS = diagm(v_ks)
        vals, vects = eigen(Hermitian(D2 + V_C + V_KS),Hermitian( S))
        #        rho = vects[:,1].^2
        MAT .= 0.0
        MAT[1:N-1, 1:N-1] = D2 + V_C + V_KS - I(N-1) * vals[i]
        MAT[1:N-1,N] = 2 * vects[:,i]
        MAT[N, 1:N-1] = 2 *  vects[:,i]'

        G .= 0.0
#        G[1:N-1] = 4 * (rho_target - rho) .* vects[:,i] 

        rho = (mat_n2m * vects[:,1]).^2 / (GAL.b-GAL.a) * 2

#        println("size rho ", size(rho))
#        println("size rho_target ", size(rho_target))
#        println("size mat_m2n ", size(mat_m2n))
#        println("size (mat_n2m * vects[:,1]) ", size((mat_n2m * vects[:,1])))
#        println("size ( (rho_target - rho) ) ", size(rho_target - rho))
#        println("size (rho_target - rho)  .* (mat_n2m * vects[:,1]) ", size((rho_target - rho)  .* (mat_n2m * vects[:,1])))
#        println("size mat_m2n * ((rho_target - rho)  .* (mat_n2m * vects[:,1])) ", size(mat_m2n*((rho_target - rho)  .* (mat_n2m * vects[:,1]))))

        
        G[1:N-1] = mat_n2m'* (4 * (rho_target - rho)  .* (mat_n2m * vects[:,1]) / (GAL.b-GAL.a) * 2)

#        println("MAT")
#        println(MAT)
#        println()
#        println("G")
#        println(G)

        apply_mat(vects[:,i], X)
        
        grad .= X*(cval*(MAT\G)[1:N-1])

        #println("size grad ", size(grad), " size v_ks ", size(v_ks))
        grad .+=  2 * regularization * v_ks

        #        grad .= 1000*(MAT\G)[1:N-1] * vects[:,i]
#        println()
#        println("grad")
#        println(grad)
#        println("sum abs ", sum(abs.(grad)))
    end

    #ret = optimize(f,  v_ks * 0.9)

    
    v_ks = rand(N-1) * 0.1
    #v_ks = zeros(N-1)

    if false
        println("start v_ks $v_ks")
        ret = f(v_ks)
        println("start $ret")
        my_grad!(grad_vec, v_ks)
        println("sum temp ", sum(abs.(grad_vec)))

        
        v_ks_t = deepcopy(v_ks)
        v_ks_t[1] += 0.00001
        b = f(v_ks_t)
        a = f(v_ks)
        println("finite diff ", (b-a) / 0.00001)

        v_ks_t = deepcopy(v_ks)
        v_ks_t[1] += 0.000001
        b = f(v_ks_t)
        a = f(v_ks)
        println("finite diff ", (b-a) / 0.000001)
        v_ks_t = deepcopy(v_ks)
        v_ks_t[1] += 0.0000001
        b = f(v_ks_t)
        a = f(v_ks)
        println("finite diff ", (b-a) / 0.0000001)

        v_ks_t = deepcopy(v_ks)
        v_ks_t[2] += 0.0000001
        b = f(v_ks_t)
        a = f(v_ks)
        println("finite diff2 ", (b-a) / 0.0000001)
        

        grad_vec .= 0.0
        my_grad!(grad_vec, v_ks)
        println("grad_vec ", grad_vec)
        return missing, missing
    end    

    
    
    println("grad_vec ", grad_vec)
    #opts = Optim.Options( f_tol = 1e-5, g_tol = 1e-5, iterations = 500, store_trace = true, show_trace = false)
    opts = Optim.Options( f_tol = 1e-9, g_tol = 1e-9, iterations = 5000, store_trace = true, show_trace = false)

    #v_ks = zeros(N-1)
    v_ks = rand(N-1) * 1.0
    
    ret = optimize(f,  v_ks, opts)
    ret2 = optimize(f,  my_grad!, v_ks, BFGS(), opts)
    #ret2 = missing
    #ret = optimize(f,my_grad!, v_ks, BFGS(), opts)
    
    #ret = missing

    function f_test(v_ks)
        
        apply_mat(v_ks, V_KS)
        vals, vects = eigen(Hermitian(D2 + V_C + V_KS),Hermitian( S))
        VECTS[:,1,1,1,1] = vects[:,1]
        rho = (mat_n2m * vects[:,1]).^2 / (GAL.b-GAL.a) * 2
        retX = 1000*sum((rho - rho_target).^2)
        println("vals[1] ", vals[1])
        println("retX $retX")
        println(rho - rho_target)
        println()
        return retX
        
    end

    println("min f")
    if !ismissing(ret)
        f_test(Optim.minimizer(ret))
    end
    println("min grad")
    if !ismissing(ret2)
        f_test(Optim.minimizer(ret2)) 
    end
    
    return ret , ret2
end





end #end module
