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


function EXC_sp(n, funlist, drho, ddrho, dvsigma, theta, gga,  r, D1)

    v = zeros(length(n[:,1]), 2)
    e = zeros(length(n[:,1]))


    
    if gga
#        sigma = zeros(length(n[:,1]), 3)
#        sigma[:,1] = drho[:,1].^2
#        sigma[:,3] = drho[:,2].^2
        #        sigma[:,2] = drho[:,1] .* drho[:,2]
        sigma = zeros(3,length(drho[:,1,1]))
        
        sigma[1,:] = drho[:,1,1].^2
        sigma[3,:] = drho[:,2,1].^2
        sigma[2,:] = drho[:,1,1].*drho[:,2,1]

        sigma[1 ,2:end] += r[2:end].^(-2) .* ( drho[2:end,1,2].^2               + sin(theta)^(-2)* drho[2:end,1,3].^2)
        sigma[2, 2:end] += r[2:end].^(-2) .* ( drho[2:end,2,2].^2               + sin(theta)^(-2)* drho[2:end,2,3].^2)
        sigma[3, 2:end] += r[2:end].^(-2) .* ( drho[2:end,1,2].*drho[2:end,2,2] + sin(theta)^(-2)* drho[2:end,1,3].*drho[2:end,2,3])
        
    end

    rho = collect(n')
    for fun in funlist

        if gga
            ret = evaluate(fun, rho=rho, sigma=sigma)
        else
            ret = evaluate(fun, rho=rho)
        end            

        if gga
            vsigma = ret.vsigma' 
            
            vrho = ret.vrho

            #1
            v[2:end,1] += - 4.0 * r[2:end].^(-1) .*  ( vsigma[2:end,1] .* drho[2:end,1,1] + 0.5* drho[2:end,2,1].*vsigma[2:end,2])
            v[2:end,2] += - 4.0 * r[2:end].^(-1) .*  ( vsigma[2:end,3] .* drho[2:end,2,1] + 0.5* drho[2:end,1,1].*vsigma[2:end,2])

            v[:,1] += -2.0*calc_D( vsigma[:,1] .* drho[:,1,1] + 0.5*vsigma[:,2] .* drho[:,2,1]   , n[:,1]+n[:,2], D1, r)
            v[:,2] += -2.0*calc_D( vsigma[:,3] .* drho[:,2,1] + 0.5*vsigma[:,2] .* drho[:,1,1]   , n[:,1]+n[:,2], D1, r)


            #2
            v[2:end,1] +=  -2.0/sin(theta) * r[2:end].^(-2) .* (sin(theta) * dvsigma[2:end,1, 2] .* drho[2:end,1, 2] + sin(theta) * vsigma[2:end,1] .* ddrho[2:end,1,2] + cos(theta) * drho[2:end,1, 2] .* vsigma[2:end,1])
            v[2:end,1] +=  -1.0/sin(theta) * r[2:end].^(-2) .* (sin(theta) * dvsigma[2:end,2, 2] .* drho[2:end,2, 2] + sin(theta) * vsigma[2:end,2] .* ddrho[2:end,2,2] + cos(theta) * drho[2:end,2, 2] .* vsigma[2:end,2])

            v[2:end,2] +=  -2.0/sin(theta) * r[2:end].^(-2) .* (sin(theta) * dvsigma[2:end,3, 2] .* drho[2:end,2, 2] + sin(theta) * vsigma[2:end,3] .* ddrho[2:end,2,2] + cos(theta) * drho[2:end,2, 2] .* vsigma[2:end,3])
            v[2:end,2] +=  -1.0/sin(theta) * r[2:end].^(-2) .* (sin(theta) * dvsigma[2:end,2, 2] .* drho[2:end,1, 2] + sin(theta) * vsigma[2:end,2] .* ddrho[2:end,1,2] + cos(theta) * drho[2:end,1, 2] .* vsigma[2:end,2])
            

            
            #3
            
            v[2:end,1] +=  -2.0/sin(theta)^2 * r[2:end].^(-2) .* ( dvsigma[2:end,1,3] .* drho[2:end,1, 3] + vsigma[2:end,1] .* ddrho[2:end,1, 3])
            v[2:end,1] +=  -1.0/sin(theta)^2 * r[2:end].^(-2) .* ( dvsigma[2:end,2,3] .* drho[2:end,2, 3] + vsigma[2:end,2] .* ddrho[2:end,2, 3])

            v[2:end,2] +=  -2.0/sin(theta)^2 * r[2:end].^(-2) .* ( dvsigma[2:end,3,3] .* drho[2:end,2, 3] + vsigma[2:end,3] .* ddrho[2:end,2, 3])
            v[2:end,2] +=  -1.0/sin(theta)^2 * r[2:end].^(-2) .* ( dvsigma[2:end,2,3] .* drho[2:end,1, 3] + vsigma[2:end,2] .* ddrho[2:end,1, 3])
        end        
        
        v[:,:] .+= ret.vrho'
        e .+= ret.zk
    end
    return 2.0*e,v
    
end

function getVsigma(n, funlist, drho, r, theta, nspin)

    if nspin == 1
        sigma = drho[:,1,1].^2
        sigma[2:end] += r[2:end].^(-2) .* ( drho[2:end,1,2].^2 + sin(theta)^(-2)* drho[2:end,1,3].^2)
        v = zeros(length(n))
        for fun in funlist
            ret = evaluate(fun, rho=n[:,1], sigma=sigma)
            v += ret.vsigma[:]
        end    
        return v
    elseif nspin == 2

        sigma = zeros(3,length(drho[:,1,1]))
        
        sigma[1,:] = drho[:,1,1].^2
        sigma[3,:] = drho[:,2,1].^2
        sigma[2,:] = drho[:,1,1].*drho[:,2,1]

        sigma[1,2:end] += r[2:end].^(-2) .* ( drho[2:end,1,2].^2               + sin(theta)^(-2)* drho[2:end,1,3].^2)
        sigma[3,2:end] += r[2:end].^(-2) .* ( drho[2:end,2,2].^2               + sin(theta)^(-2)* drho[2:end,2,3].^2)
        sigma[2,2:end] += r[2:end].^(-2) .* ( drho[2:end,1,2].*drho[2:end,2,2] + sin(theta)^(-2)* drho[2:end,1,3].*drho[2:end,2,3])

        v = zeros(length(n[:,1]),3)
        for fun in funlist
            ret = evaluate(fun, rho=collect(n'), sigma=sigma)
            v += ret.vsigma'
        end    
        return v


    end
        
    
end


function EXC(n, funlist, drho, ddrho, dvsigma, theta, gga, r, D1)

#    println("EXC ", sum(abs.(drho)), " ", gga)
    
    v = zeros(length(n))
    e = zeros(length(n))

    if gga
#        kf = (3*pi^2*n).^(1/3)
#        sigma = ((drho./(2* kf .* n) ).^2)
    #    sigma = drho.^2

        sigma = drho[:,1].^2
        sigma[2:end] += r[2:end].^(-2) .* ( drho[2:end,2].^2 + sin(theta)^(-2)* drho[2:end,3].^2    )

    end

    #    drho_s  = drho 
    #sigma_s = drho[:,1].^2 + drho[:,2].^2 + drho[:,3].^2
    
    for fun in funlist
        if gga
#            println("$fun ")
#            println(typeof(n), " " , typeof(sigma))
            ret = evaluate(fun, rho=collect(n), sigma=sigma )
            vsigma = ret.vsigma[:] 
            
            vrho = ret.vrho

            #1
            v[2:end] += - 4.0 * r[2:end].^(-1) .*  ( ( vsigma[2:end] .* drho[2:end,1]))
            v += -2.0*calc_D((vsigma .* drho[:,1])  , n, D1, r)

            #2
            v[2:end] +=  -2.0/sin(theta) * r[2:end].^(-2) .* (sin(theta) * dvsigma[2:end,2] .* drho[2:end, 2] + sin(theta) * vsigma[2:end] .* ddrho[2:end, 2] + cos(theta) * drho[2:end, 2] .* vsigma[2:end])

            #3
            v[2:end] +=  -2.0/sin(theta)^2 * r[2:end].^(-2) .* ( dvsigma[2:end,3] .* drho[2:end, 3] + vsigma[2:end] .* ddrho[2:end, 3])
            
            
            #v += -2.0 /sin(theta) * vsigma .* (r.^(-1) .* ( cos(theta)  
            
        else
            ret = evaluate(fun, rho=n)
            vrho = ret.vrho
            
        end            

        v +=  vrho[:]

        
        #        println(ret)
        #        println("size r ", size(r), " vs ", size(ret.vsigma))

#        v +=  ret.vrho[:] - 2 ./ r .* ret.vsigma[:] .* drho * 2  #- D1 * ( ret.vsigma[:]  ) 
        
        #        v +=  ret.vrho[:] - 2.0 * r.^(-2) .*  (D1 * (r.^2 .* ret.vsigma[:] .* drho))
        #v +=  ret.vrho[:] - 4.0 * r.^(-1) .*  ret.vsigma[:] .* drho  + 2 * D1 * (ret.vsigma[:] .*  drho) 
        #v +=  ret.vrho[:] - 4.0 * r.^(-1) .*  ret.vsigma[:] .* drho  - 2 * (D1*ret.vsigma').* (drho)

        #        v +=  ret.vrho[:] - 2.0 * r.^(-2) .*  (D1 * (r.^2 .* ret.vsigma[:] .* drho))

#        va[:,1] += drho
#        va[:,2] += drho_s
#        va[:,3] += n
#        va[:,4] += n_s
#        va[:,5] += ret.vsigma[:]
#        va[:,6] += vsigma
#        va[:,7] += ret.vrho[:]
#        va[:,8] += vrho
#        va[:,9] += -2*D1*(vsigma .* drho_s)
#        va[:,10] += calc_D(-2.0*(vsigma .* drho_s)  , n, D1, r) 
        
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
    
    return 2.0*e,v
    
end

function smooth(x, thr1, thr2)

    a = log(thr1)
    b = log(thr2)
    y = log.(abs.(x))
    t = (y .- b)/(a-b)
    
    return (y .>= a)*1.0 + (y .< a .&& y .> b) .* (0.0 .+ 10.0 * t.^3 .- 15.0 *  t.^4  .+ 6.0 * t.^5) 

end

function smooth_f(f, r, thr, temp)

    cut = findfirst(f  .< thr)
    return f ./ (exp.( (r .- r[cut] ) / temp) .+ 1)

end

function calc_D(f, rho, D1, r)

#    thr1 = 5*10^-6
#    thr2 = 5*10^-8
    thr1 = 10^-5
    thr2 = 10^-7
    cutfn = smooth( rho, thr1, thr2)

    dr_fd = zeros(size(f))

    for i = 2:length(dr_fd)-1
        dr_fd[i] = 0.5*(f[i]-f[i-1]) / (r[i] - r[i-1]) + 0.5*(f[i+1]-f[i]) / (r[i+1] - r[i])
    end

    Df = (D1*f) .* ( cutfn ) + dr_fd .* ( 1.0 .- cutfn )

    return Df
    
end



end #end module
