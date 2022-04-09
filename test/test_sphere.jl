using FastSphericalHarmonics
using SphericalHarmonics
using ChebyshevQuantum
using Test
using QuadGK
tol_var=1e-8

@testset "test sphere" begin


    lmax=3
    theta,phi = FastSphericalHarmonics.sph_points(lmax+1)
    z = zeros(Float64, length(theta), length(phi))

    function y11(t,p)
        return -(3/(8*pi))^0.5*sin(t)*exp(im*p) 
    end
    function y10(t,p)
        return (3/(4*pi))^0.5*cos(t)
    end

    function px(t,p)
        return -sqrt(2)*real(y11(t,p))
    end
    function py(t,p)
        return -sqrt(2)*imag(y11(t,p))
    end
    function pz(t,p)
        return real(y10(t,p))
    end

    function s(t,p)
        return 1/(4*pi)^0.5
    end
    
    for (i,t) = enumerate(theta)
        for (j,p) = enumerate(phi)
            z[i,j] = pz(t,p)^2
        end
    end
    
    zlm = FastSphericalHarmonics.sph_transform(z)

    println("z")
    display(z)
    println()
    println("zlm")
    display(zlm)

    zlm2 = zeros(size(zlm))
    println("slow transform")
    for (i,t) = enumerate(theta)
        for (j,p) = enumerate(phi)
            Y = SphericalHarmonics.computeYlm(t, p, lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())
            for l = 0:lmax
                for m = -l:l
                    d = FastSphericalHarmonics.sph_mode(l,m)
                    zlm2[d[1],d[2]] += z[i,j]*Y[(l,m)]*sin(t)  / length(theta) / length(pi) * 2 * pi
                end
            end
        end
    end
    println("zlm2")
    display(zlm2)
    
    zz = FastSphericalHarmonics.sph_evaluate(zlm)
    println()
    println("zz")
    display(zz)
    
    
end
