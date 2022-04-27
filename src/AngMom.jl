module AngMom

using FastSphericalHarmonics
using SphericalHarmonics

using WignerSymbols
using Base.Threads

gaunt_dict = Dict{NTuple{6,Int64}, Float64}()
real_gaunt_dict = Dict{NTuple{4,Int64}, Float64}()



function construct_real_gaunt_indirect(; lmax=12)

    THETA, PHI = FastSphericalHarmonics.sph_points(lmax+1)
    Ytest = SphericalHarmonics.computeYlm(0.0, 0.0, lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())
    s = length(Ytest)
    YY = zeros(length(THETA), length(PHI), s)
    
    for (i,theta) in enumerate(THETA)
        for (j,phi) in enumerate(PHI)
            Y = SphericalHarmonics.computeYlm(theta, phi, lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())
            YY[i,j,:] = Y[:].^2
        end
    end

    ZZ = zeros(size(YY))
    for ss = 1:s
        ZZ[:,:,ss] = FastSphericalHarmonics.sph_transform(YY[:,:,ss])    
    end

    inds = Dict()
    c=0
    for l = 0:lmax
        for m = -l:l
            c += 1
            inds[(l,m)] = c
        end
    end
    
    for l1 = 0:lmax
        for m1 = -l1:l1
            for l2 = 0:lmax
                for m2 = -l2:l2
                    d = FastSphericalHarmonics.sph_mode(l1,m1)
#                    println("$l1 $m1 $l2 $m2 | $d ", inds[(l2,m2)])
                    real_gaunt_dict[(l1,m1,l2,m2)] = ZZ[d[1], d[2], inds[(l2,m2)]]
                end
            end
        end
    end
    
    return ZZ
    
end






function fill_dict(; lmax=6)

    T = []
    for l1 = 0:lmax
        for l2 = 0:lmax
            push!(T, (l1,l2))
        end
    end

    for (l1,l2) in T
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
    
    return real((-1)^(m1) * sqrt((2*l1+1)*(2*l2+1)*(2*l3+1) / (4*pi)  ) * WignerSymbols.wigner3j(l1,l2,l3,0,0,0) * WignerSymbols.wigner3j(l1,l2,l3,-m1,m2,m3))
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
