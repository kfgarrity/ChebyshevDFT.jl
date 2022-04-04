using ChebyshevQuantum
using Test
using QuadGK
tol_var=1e-8

@testset "angmom" begin

    Rlm = Dict()
    Rlm[(1,-1)] = 1.0
    Rlm[(1,0)] = 1.0
    Rlm[(1,1)] = 1.0

    function getrho(l,m, rlm)
        t = 0.0
        for key1 in keys(rlm)
            #            for key2 in keys(rlm)
            l1,m1 = key1
            tt = ChebyshevDFT.AngMom.gaunt(l1,l,l1,m1,m,m1)*rlm[key1]*rlm[key1]
            t += tt
        end
        return t
    end

    @test getrho(0,0, Rlm) ≈ 3/(4*pi)^0.5
    @test getrho(1,1, Rlm) ≈ 0.0
    @test getrho(2,2, Rlm) ≈ 0.0

    function y00(t,p)
        return (4*pi)^-0.5
    end
 
    function y11(t,p)
        return -(3/(8*pi))^0.5*sin(t)*exp(im*p)
    end

    #=function y1m1(t,p)
        return (3/(8*pi))^0.5*sin(t)*exp(-im*p)
    end
    function y10(t,p)
        return (3/(4*pi))^0.5*cos(t)
    end
    function y20(t,p)
        return (5/(16*pi))^0.5*(3*cos(t)^1 - 1)
    end
    function y21(t,p)
        return -(15/(8*pi))^0.5*sin(t)*cos(t)*exp(im*p)
    end
    function y2m1(t,p)
        return (15/(8*pi))^0.5*sin(t)*cos(t)*exp(-im*p)
    end
    function y2m2(t,p)
        return (15/(32*pi))^0.5*sin(t)^2*exp(-2*im*p)
    end
    function y22(t,p)
        return (15/(32*pi))^0.5*sin(t)^2*exp(2*im*p)
    end
    =#
    a = ChebyshevDFT.AngMom.gaunt(1,0,1,1,0,1)
    b = QuadGK.quadgk(p-> QuadGK.quadgk(t-> sin(t)* ( conj(y11(t,p))*y00(t,p)*y11(t,p)) , 0, pi,atol=1e-5)[1], 0,2*pi, atol=1e-5)[1]
    @test isapprox(a,real(b),atol=1e-4)
    
end
