module  Search

using Optim
using Suppressor
using ..SCF:DFT_spin_l_grid_LM

function go(E, n, val; oiters = 30, alpha = 1.0, beta = 0.01, fill_str = "1 0 0 1 0\n", Z=1.0,N=100, Rmax=25, niters=80, exc = [:gga_x_pbe, :gga_c_pbe], mix=0.5, spherical=true, hydrogen=true, vstart = missing)

    function fn(vext)
        t = 0.0
        @suppress begin
            t = rms(E, n, val, vext, alpha = alpha,beta = beta,  fill_str = fill_str, Z=Z,N=N, Rmax=Rmax, niters=niters, exc = exc, mix=mix, spherical=spherical, hydrogen=hydrogen)
#            println("t $t")
        end
        println("t $t " )
        return t[1] * 100
    end
    
    opts = Optim.Options(g_tol = 1e-8,f_tol = 1e-8, x_tol = 1e-8,
                         iterations = oiters,
                         store_trace = true,
                         show_trace = false)

    if ismissing(vstart)
        x0 = zeros(N+1)
    else
        x0 = vstart
    end
    
    res = optimize(fn, x0, NelderMead(), opts)

    println("res")
    println(res)
    println()
    minvec = Optim.minimizer(res)    
    println("minvec")
    println(minvec)
    
    return res, minvec
end

    
function rms(E, n, val, vext; alpha = 1.0, beta = 0.01, fill_str = "1 0 0 1 0\n", Z=1.0,N=100, Rmax=25, niters=80, exc = [:gga_x_pbe, :gga_c_pbe], mix=0.5, spherical=true, hydrogen=true)

    energy, converged, vals_rS, vectsN, rho_LMN, rall_rsN, aqqqN, rhor2N, vlda_LMN, drho_LMN,D1N,vaN  =  DFT_spin_l_grid_LM(fill_str = fill_str, Z=Z,N=N, Rmax=Rmax, niters=niters, exc = exc, mix=mix, spherical=spherical, hydrogen=hydrogen, vext = vext);

    nspin = size(rho_LMN)[2]

    if nspin == 1
        
        return sum( (n.(rall_rsN) - rho_LMN[:,1,1,1]).^2) + alpha*( E - energy)^2  + beta * (vals_rS[1] - val)^2, energy, vals_rS[1]
    else
        return sum( (n.(rall_rsN) - sum(rho_LMN[:,:,1,1], dims=2)).^2) + alpha*( E - energy)^2  + beta * (vals_rS[1] - val)^2, energy, vals_rS[1]
    end
        
end



end #end module
