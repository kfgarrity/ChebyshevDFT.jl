# https://www.nature.com/articles/s41467-019-12467-0

module Inverse

using LinearAlgebra
using Base.Threads
import Base.Threads.@spawn
using FastSphericalHarmonics
using Optim

using ..AngMom:real_gaunt_dict

using ..SCF:prepare_dft
using ..SCF:assemble_rho_LM
using ..SCF:VXC_LM

using ..Hartree:V_H3


function get_inv(rho_LM_target; fill_str=missing, N = 40, Z=1.0, Rmax = 10.0, ax = 0.2, bx = 0.5, spherical = false, vext = missing, lmax_rho = missing, hydrogen)

    Z, nel, nspin, lmax, lmax_rho, grid, grid1, grid2, invgrid, invgrid1, r, D, D2, w, rall,wall,H_poisson, D1X, D2X, rall_rs, ig1, H_poisson, D2grid, D2Xgrid, D1Xgrid, VEXT, H0_L, Vc, spin_lm, spin_lm_rho = prepare_dft(Z, N, Rmax, ax, bx, fill_str, spherical, vext, lmax_rho)

    
#    rho_LM_target = rho_LM_target 
#    valsR, vR = eigen(H0_L[1][:,:,1,1,1])
#    rho_LM_target = zeros(size(rho_LM_target))
#    rho_LM_target[2:N,:,:,:] = vR[:,1] .* conj(vR[:,1])

#    return rho_LM_target, vR
    
    #Hartree

    VH_LM = zeros(size(rho_LM_target)[[1,3,4]])
    
    for (spin,l,m) in spin_lm_rho
        if spin == 2
            continue
        end
        d = FastSphericalHarmonics.sph_mode(l,m)

        if nspin == 1
            rho_tot = rho_LM_target[2:N,1,d[1],d[2]] ./ rall_rs[2:N].^2
        elseif nspin == 2
            rho_tot = rho_LM_target[2:N,1,d[1],d[2]] ./ rall_rs[2:N].^2 + rho_LM_target[2:N,2,d[1],d[2]] ./ rall_rs[2:N].^2
        end            
        
        VH_LM[:,d[1],d[2]] = V_H3( rho_tot, rall_rs[2:N],w,H_poisson,Rmax, rall_rs, sum(nel)/sqrt(4*pi), ig1=ig1[2:N], l=l)
    end

#    return VH_LM

    factor = 1.0
    
    VH_LM *= factor
    
    #V_work_LM = zeros(N+1, nspin, lmax_rho+1, (2*lmax_rho+1))

    V_work_LM, elda_LM = VXC_LM(rho_LM_target *4*pi, missing, rall_rs, funlist=missing, D1Xgrid, gga=false)
    V_work_LM = V_work_LM * factor #* 2 * pi

    V_work_LM_start = deepcopy(V_work_LM)
#    return V_work_LM
    ####return VH_LM, V_work_LM
   
    
    vects = zeros(Complex{Float64}, N-1, N-1, nspin, lmax+1, 2*lmax+1)
    HAM = zeros(Complex{Float64}, N-1, N-1, nspin, lmax+1, 2*lmax+1)
    vals_r = zeros(Float64, N-1, nspin, lmax+1,2*lmax+1)    

    
    rho_LM = deepcopy(rho_LM_target)

    f_reg = 1e-5
    #f_reg = 0.0
    
    function go(x)
    
        V_work_LM[:,:,:,:] = reshape(x, N+1, nspin, lmax_rho+1, (2*lmax_rho+1))

        VHt = zeros(N+1)
        V_w_t = zeros(N+1)
        for i = 1:length(spin_lm)
            spin = spin_lm[i][1]
            l = spin_lm[i][2]
            m = spin_lm[i][3]
            d = FastSphericalHarmonics.sph_mode(l,m)
            for ll = 0:(lmax_rho)
                for mm = -ll:ll
                    d2 = FastSphericalHarmonics.sph_mode(ll,mm)
                    VHt += 4*pi*VH_LM[:, d2[1], d2[2]] * real_gaunt_dict[(ll,mm,l,m)] #/ (2*ll+1)
                    V_w_t += V_work_LM[:,spin,d2[1], d2[2]] #* real_gaunt_dict[(ll,mm,l,m)]                    
                end
            end

            VH_mat = diagm(VHt[2:N])
            V_w_mat = diagm(V_w_t[2:N])
            
            if hydrogen
                Ham = H0_L[l+1] + (V_w_mat)
            else
                Ham = H0_L[l+1] + (VH_mat + V_w_mat)
            end
            HAM[:,:,spin,d[1],d[2]] = Ham
            
#            println("size Ham ", size(Ham))
            vals, v = eigen(Ham)

#            println("go vals ", vals[1:3])
            
            vals_r[:,spin,d[1],d[2]] = real.(vals)
            vects[:,:,spin,d[1],d[2]] = v
            
        end

        rho_LMx, filling, rhor2, rhoR, rhoR2 = assemble_rho_LM(vals_r, vects, nel, rall_rs, wall, rho_LM, N, Rmax, D2Xgrid, D1Xgrid, nspin=nspin, lmax=lmax, ig1 = ig1, gga=false)
        rho_LM[:,:,:,:] = rhoR2[:,:,:,:]

        #        println("filling ", filling[1:3])
        
#        rho_LM[2:N,:,:,:] = vects[:,1,:,:,:] .* conj(vects[:,1,:,:,:])
        
#        rho_LM_alt = zeros(size(rho_LM))
#        rho_LM_alt[2:N,:,:,:] = vects[:,1,:,:,:] .* conj(vects[:,1,:,:,:])
#        rho_LM_alt = rho_LM_alt / ( 4.0*pi*sum(rho_LM_alt .* wall .* ig1)) / sqrt(4*pi)
#        println("rho_LM_alt 2 ", rho_LM_alt[2], " " , vects[:,1,:,:,:] .* conj(vects[:,1,:,:,:]))

        
        return rho_LM,vects,filling, vals_r
    end

    rho_LM,vects,filling, vals_r = go( V_work_LM[:])
#    return rho_LM,rho_LM_target
    
    MAT = zeros(Complex{Float64}, N, N)
    B = zeros(Complex{Float64}, N)
    H =  zeros(Complex{Float64}, N-1, N-1)  

    
    function f(x)
        rho_LM,vects,filling, vals_r = go(x)
        retval = sum(weights.*(rho_LM[2:N,:,:,:] - rho_LM_target[2:N,:,:,:]).^2)
        retval += f_reg * sum((x).^2)
        println("retval $retval ", vals_r[1])
        return retval
    end    

    function g(grad, x)

        rho_LM,vects, filling, vals_r = go(x)

        grad_LM .= 0.0
        for i = 1:length(spin_lm)
            spin = spin_lm[i][1]
            l = spin_lm[i][2]
            m = spin_lm[i][3]
            d = FastSphericalHarmonics.sph_mode(l,m)


            H[:,:] = HAM[:,:,spin,d[1],d[2]] 

            for n = 1:(N-1)
                if filling[n, spin, d[1], d[2]] < 1e-8
#                    println("break, ", [n, spin, d[1], d[2]])
                    break
                end
#                println("vals_r ", vals_r[n, spin, d[1], d[2]])

                vnorm = real(vects[:,n,spin,d[1],d[2]]) 

#                println("val ", vals_r[1], " ", vects[:,1]' * H * vects[:,1], " " , vnorm'*H*vnorm)
#                
#                println("vnorm ", sum(vnorm.^2 .* wall[2:N] .* ig1[2:N]))
                
                MAT[1:(N-1), 1:(N-1) ] = H' - I(N-1)*vals_r[n, spin, d[1], d[2]]
                MAT[1:(N-1), N] = 2.0*vnorm
                MAT[N,1:(N-1)] = vnorm'
        

            #        println( size(rho_target), " " , size(rho), "  ", size(vects[:,1]))

#                println("N $N")
#                println("size B ", size(B))
#                println("size rho_LM_target ", size(rho_LM_target))
#                println("vects ", size(vects))
#                println("size weights ", size(weights))
                B[1:(N-1)] =  (4*pi)^-(3/2)  *    4.0*weights[:, spin, d[1], d[2]]  .* (rho_LM_target[2:N, spin, d[1], d[2]] - rho_LM[2:N, spin, d[1], d[2]]  ) .* (
                vnorm / sum(vnorm.^2 .* ig1[2:N] .* wall[2:N]) - 1.0*vnorm.^3  .* ig1[2:N] .* wall[2:N] / sum(vnorm.^2 .* ig1[2:N] .* wall[2:N])^2  ) * filling[n,spin,d[1],d[2]]
                

                VV =  MAT \ B
                
                grad_LM[2:N, spin, d[1], d[2]] += real.(VV[1:N-1] .* vnorm) 
            end
        end
        grad[:] = grad_LM[:]
        grad += 2* f_reg * (x) 
        return grad[:]
    end
    


    weights = 10^3*ones(N-1) .+ 10^1 ./ (rho_LM_target[2:N, :,:,:] .+ 1e-8)

    ff = f(V_work_LM[:])

    


    grad = zeros(size(V_work_LM[:]))
    grad_LM = zeros(size(V_work_LM))    
    
    gg = g(grad, V_work_LM[:])

    return ff, gg, f, g, V_work_LM[:], rho_LM, rho_LM_target
    
#    return missing, missing, VH_LM, rho_LM
    
    f_reg = 1e-5
    
    weights = 10^3*ones(N-1) #.+ 10^1 ./ (rho_LM_target[2:N, :,:,:] .+ 1e-8)
    opts = Optim.Options( f_tol = 1e-5, g_tol = 1e-5, iterations = 50, store_trace = true, show_trace = false)    
    #ret = optimize(f, V_work_LM[:], BFGS(), opts)

    println("v1")
    ret = optimize(f, g, V_work_LM[:], BFGS(), opts)
    #ret = optimize(f, V_work_LM[:], BFGS(), opts)

    if true
        f_reg = 1e-16
        
        weights = 10^3*ones(N-1) .+ 10^8 ./ (rho_LM_target[2:N, :,:,:] .+ 1e-19)
        themin = Optim.minimizer(ret)
        
        opts = Optim.Options( f_tol = 1e-7, g_tol = 1e-7, iterations = 50, store_trace = true, show_trace = false)    
        println("v2")
        #ret = optimize(f, g, themin, BFGS(), opts)
        ret = optimize(f,  themin, BFGS(), opts)
    end
    
    
    println("ret")
    println(ret)
    println("min")
    themin = Optim.minimizer(ret)
    println(themin[1:6])
    
    return ret, themin, VH_LM, V_work_LM_start, rho_LM

end
    
    










end #end module
