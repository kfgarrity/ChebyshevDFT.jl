"""
libxc stuff

"""
module UseLibxc



using Libxc


function set_functional(exc_list; nspin=1)

    funlist = Functional[]
    for exc in exc_list
        exc = Symbol(exc)
        fun = Functional(exc, n_spin=nspin)
        push!(funlist, fun)

        
        
        println("Libxc: adding $(fun.family) functional $(fun.kind) with nspin=$nspin")
        println(fun.references[1].reference)
    end
    return funlist
end


function EXC_sp(n, funlist)

    v = zeros(length(n[:,1]), 2)
    e = zeros(length(n[:,1]))
    for fun in funlist
        #        ret = evaluate(fun, rho=[nup'; ndn'])
        ret = evaluate(fun, rho=collect(n'))
        v[:,:] .+= ret.vrho'
#        v[:,2] .+= ret.vrho[2,:]
        e .+= ret.zk
    end
    
    return 2.0*e,v
    
end

function EXC(n, funlist)

    v = zeros(length(n))
    e = zeros(length(n))
    for fun in funlist
        ret = evaluate(fun, rho=n)
        v += ret.vrho[:]
        e += ret.zk[:]
    end
    
    return 2.0*e,v
    
end



end #end module
