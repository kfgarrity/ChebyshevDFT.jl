"""
libxc stuff

"""
module UseLibxc



using Libxc


function set_functional(exc_list; nspin=1)

    gga = false
    funlist = Functional[]
    for exc in exc_list
        exc = Symbol(exc)
        fun = Functional(exc, n_spin=nspin)
        push!(funlist, fun)
        println("Libxc: $exc, we are adding $(fun.family) functional $(fun.kind) ,  $(fun.name) with nspin=$nspin")
        println(fun.references[1].reference)

        if fun.family != :gga && fun.family != :lda
            throw(ArgumentError, "ERROR, fun $exc family $(fun.family) not supported yet")
        end
        if fun.family == :gga
            gga=true
        end
        
    end
    return funlist, gga
end


function EXC_sp(n, funlist, drho, gga)

    v = zeros(length(n[:,1]), 2)
    e = zeros(length(n[:,1]))
    if gga
        sigma = zeros(length(n[:,1]), 3)
        sigma[:,1] = drho[:,1].^2
        sigma[:,3] = drho[:,2].^2
        sigma[:,2] = drho[:,1] .* drho[:,2]
        sig = collect(sigma')
    end

    rho = collect(n')
    for fun in funlist

        if gga
            ret = evaluate(fun, rho=rho, sigma=sig)
        else
            ret = evaluate(fun, rho=rho)
        end            
        v[:,:] .+= ret.vrho'
        e .+= ret.zk
    end
    return 2.0*e,v
    
end

function EXC(n, funlist, drho, gga, r, D1)

#    println("EXC ", sum(abs.(drho)), " ", gga)
    
    v = zeros(length(n))
    va = zeros(length(n),10)
    e = zeros(length(n))
    if gga
#        kf = (3*pi^2*n).^(1/3)
#        sigma = ((drho./(2* kf .* n) ).^2)
    #    sigma = drho.^2
    end

    drho_s  = drho .* smooth(drho.^2, 1e-6, 1e-10)
    sigma_s = drho_s.^2
    
    n_s = n #.* smooth(n, 1e-12, 1e-15)

    for fun in funlist
        if gga
            ret = evaluate(fun, rho=n_s, sigma=sigma_s )
            vsigma = ret.vsigma[:] 
            
            vrho = ret.vrho[:] #.* smooth(sigma_s, 1e-8, 1e-12)
        else
            ret = evaluate(fun, rho=n)
        end            

        #        println(ret)
        #        println("size r ", size(r), " vs ", size(ret.vsigma))

#        v +=  ret.vrho[:] - 2 ./ r .* ret.vsigma[:] .* drho * 2  #- D1 * ( ret.vsigma[:]  ) 
        
        #        v +=  ret.vrho[:] - 2.0 * r.^(-2) .*  (D1 * (r.^2 .* ret.vsigma[:] .* drho))
        #v +=  ret.vrho[:] - 4.0 * r.^(-1) .*  ret.vsigma[:] .* drho  + 2 * D1 * (ret.vsigma[:] .*  drho) 
        #v +=  ret.vrho[:] - 4.0 * r.^(-1) .*  ret.vsigma[:] .* drho  - 2 * (D1*ret.vsigma').* (drho)

        #        v +=  ret.vrho[:] - 2.0 * r.^(-2) .*  (D1 * (r.^2 .* ret.vsigma[:] .* drho))
        v +=  vrho[:]
        v += - 4.0 * r.^(-1) .*  ( ( vsigma[:] .* drho_s))
        v += -2*D1*(vsigma .* drho_s .* smooth(n_s, 5e-5, 1e-6) ) 

        va[:,1] += drho
        va[:,2] += drho_s
        va[:,3] += n
        va[:,4] += n_s
        va[:,5] += ret.vsigma[:]
        va[:,6] += vsigma
        va[:,7] += ret.vrho[:]
        va[:,8] += vrho
        va[:,9] += -2*D1*(vsigma .* drho_s)
        va[:,10] += -2*D1*(vsigma .* drho_s .* smooth(n_s, 5e-4, 1e-6) ) 
        
#        va[:,1] += ret.vsigma[:]
#        va[:,2] += drho
#        va[:,3] += D1*(ret.vsigma[:] .* drho)
#        va[:,4] += -4.0 * r.^(-1) .*  ( ( ret.vsigma[:] .* drho))
#        va[:,5] += - 2.0 * r.^(-2) .*  D1*(r.^2 .* ret.vsigma[:] .* drho )
#        va[:,6] += ret.vrho[:]
#        va[:,7] += n
#        va[:,8] += vsigma[:]
#        va[:,9] += D1*(vsigma[:] .* drho)
        #v +=  -2.0 * r.^(-1) .* ( D1*( ret.vsigma[:] .* drho))

        #v += (D1*ret.vsigma[:]) ./ r

        #v += - 2.0 * r.^(-2) .*  (D1*(r.^2)) .* ret.vsigma[:] .* drho
        #v += - 2.0 * r.^(-2) .*  r.^2 .* D1*(ret.vsigma[:] .* drho )
        
        #v += - 2.0 * r.^(-2) .*  D1*(r.^2 .* ret.vsigma[:] .* drho )
        #v += - 2.0 * r.^(-2) .*  (D1*(r.^2)) .* ret.vsigma[:] .* drho 
        #v += - 2.0 * r.^(-2) .*  r.^2 .* (D1*(ret.vsigma[:])) .* drho 
        #v += - 2.0 .* ret.vsigma[:] .* (D1*(drho ) )
        #v += - 2.0 * r.^(-2) .*  (D1*(r.^2 .* drho)) .* ret.vsigma[:] 

        
        #v += - 2.0 * r.^(-2) .*  D1*(r.^2 .* ret.vsigma[:])   .* drho

        #        v += - 2.0 * ( D1*ret.vsigma[:]) .* drho
#        v += - 2.0 * ret.vsigma[:] .* (D1*drho)

        
        #v += ret.vsigma[:]  
        e += ret.zk[:]
    end
    
    return 2.0*e,v, va
    
end

function smooth(x, thr1, thr2)

    a = log(thr1)
    b = log(thr2)
    y = log.(x)
    t = (y .- b)/(a-b)
    
    return (y .>= a)*1.0 + (y .< a .&& y .> b) .* (0.0 .+ 10.0 * t.^3 .- 15.0 *  t.^4  .+ 6.0 * t.^5) 

end

function smooth_f(f, r, thr, temp)

    cut = findfirst(f  .< thr)
    return f ./ (exp.( (r .- r[cut] ) / temp) .+ 1)

end



end #end module
