module Hartree
using LinearAlgebra
using ChebyshevQuantum
using QuadGK

function V_H(N, n, Rmax)

    
#    pts, Dab = getmats(N, a=0.0, b = Rmax)

#    VH0 = (4*pi)*pts.*wts.*n

    VH0 = 4*pi*QuadGK.quadgk(x->n(x)*x, 0, Rmax, atol=1e-10)[1]
    println("VH0 ", VH0)
    function A1(x)
        return 2.0/(x + 1e-4)
    end

    function B(x)
        return n(x)*(-4*pi)
    end
    
    C = ChebyshevQuantum.DiffEq.solve(N = N, bc1=[:a, 0, VH0], bc2=[:b, 0,1/Rmax], a=0.0, b = Rmax, A1 = A1, A2=1.0, B = B)

    
    
end

function V_Ha(N, n, Rmax)

    
#    pts, Dab = getmats(N, a=0.0, b = Rmax)

#    VH0 = (4*pi)*pts.*wts.*n

    VH0 = 4*pi*QuadGK.quadgk(x->n(x)*x, 0, Rmax, atol=1e-10)[1]
    println("VH0 $VH0")
    function A1(x)
        return 2.0/(x + 1e-5)
    end

    function B(x)
        return n(x)*(-4*pi)  + (2.0/(x+1e-5)) * (VH0 / Rmax - 1 / Rmax^2)
    end
    
    C = ChebyshevQuantum.DiffEq.solve(N = N, bc1=[:a, 0, 0.0], bc2=[:b, 0,0.0], a=0.0, b = Rmax, A1 = A1, A2=1.0, B = B)

    
    
end

function V_H2(N, n, Rmax)

    a = 0.0
    b = Rmax
    
    pts = ChebyshevQuantum.Interp.getCpts(N);
    println("N $N pts ", size(pts))
    r = (1 .+ pts) * Rmax / 2    
    D = ChebyshevQuantum.Interp.getD(pts) / ((b-a)/2.0) ;
    D2 = D*D

    VH0 = 4*pi*QuadGK.quadgk(x->n(x)*x, 0, Rmax, atol=1e-10)[1]
    println("VH0 $VH0")

    pts_a = pts[2:N]
    r_a = (1 .+ pts[2:N]) * Rmax / 2    
    D_a = D[2:N, 2:N]
    D2_a = D2[2:N,2:N]

    Ham = D2_a +  diagm( 2 ./ r_a )  * D_a 

    
    B = (-4*pi)*n.(r_a)  .+ (2.0 ./ r_a ) * (VH0 / Rmax - 1.0 / Rmax^2)

    println("H ", Ham[1:2,1:2])
    println("B ", B[1])
    println("n ", n.(r_a)[1])
    
    C = Ham \ B
    
    println("size C ", size(C), " size r ", size(r))
    println("C ", C)
    CC = ChebyshevQuantum.Chebyshev.make_cheb([0.0; C; 0.0], a = a, b = b)
    
    return CC
    
end


function V_H3(rho,r,w,Ham ,Rmax, rall, nel)


    a = 0.0
    b = Rmax
    
    VH0 = 4.0*pi*sum(rho.*r.*w)
    #VH0 = nel
    #println("VH0 H3  $VH0")

    #Ham = D2 +  diagm( 2.0 ./ r )  * D 

    B = (-4.0*pi)*rho  .+ (2.0 ./ r ) * (VH0 / Rmax - nel / Rmax^2)

#    println("H ", Ham[1:2,1:2])
#    println("B ", B[1])
#    println("rho ", rho[1])

    
    
    C = Ham \ B

    C = [0;C;0]  + VH0*(1 .- rall / Rmax) + rall * nel / Rmax^2

    #C = C  + VH0*(1 .- r / Rmax) + r * nel / Rmax^2
    
    return C
    
end



end #end module
