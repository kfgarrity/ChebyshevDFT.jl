module AngMom

using WignerSymbols
using Base.Threads

gaunt_dict = Dict{NTuple{6,Int64}, Float64}()

function fill_dict(; lmax=6)

    T = []
    for l1 = 0:lmax
        for l2 = 0:lmax
            push!(T, (l1,l2))
        end
    end
    
    @threads for (l1,l2) in T
        for l3 = 0:lmax
            for m1 = -l1:l1
                for m2 = -l2:l2
                    for m3 = -l3:l3
                        #                            println("typeof ", typeof((l1,l2,l3,m1,m2,m3)))
#                        t = gaunt_fn(l1,l2,l3,m1,m2,m3)
#                        if abs(t) > 1e-5
#                            if l1 == l3 && m1 == m3 && l1 == 1 && l3 == 1
#                                println("$l1 $m1 $l2 $m2 $l3 $m3   ", t)
#                            end
#                        end
                        gaunt_dict[(l1,l2,l3,m1,m2,m3)] = gaunt_fn(l1,l2,l3,m1,m2,m3)
                    end
                end
            end
        end
    end

end


function gaunt_fn(l1,l2,l3,m1,m2,m3)

    #antisymmetric 
    
    return real((-1)^(m1) * (-1)^(m2) * sqrt((2*l1+1)*(2*l2+1)*(2*l3+1) / (4*pi)  ) * WignerSymbols.wigner3j(l1,l2,l3,0,0,0) * WignerSymbols.wigner3j(l1,l2,l3,-m1,-m2,m3))
    #return  sqrt((2*l1+1)*(2*l2+1)*(2*l3+1) / (4*pi)  ) * WignerSymbols.wigner3j(l1,l2,l3,0,0,0) * WignerSymbols.wigner3j(l1,l2,l3,m1,m2,-m3)
    
end


function gaunt(l1,l2,l3,m1,m2,m3)
    key = (l1,l2,l3,m1,m2,m3)
    if key in keys(gaunt_dict)
        return gaunt_dict[key]
    end
    return gaunt_fn(l1,l2,l3,m1,m2,m3)
end


end #end module
