using ChebyshevQuantum
using Test
using QuadGK
tol_var=1e-8

@testset "test sphere" begin



    theta,phi = FastSphericalHarmonics.sph_points(2)
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
            z[i,j] = s(t,p)
        end
    end
    
    zlm = FastSphericalHarmonics.sph_transform(z)

    println("z")
    display(z)
    println()
    println("zlm")
    display(zlm)

    zz = FastSphericalHarmonics.sph_evaluate(zlm)
    println()
    println("zz")
    display(zz)
    
    
end
