module G2
using Polynomials
using SpecialPolynomials
using Base.Threads
using LinearAlgebra
using ForwardDiff
using QuadGK



function get_cheb_bc(N; vartype=BigFloat)

    B = basis.(Chebyshev{vartype}, 0:N);

    B2 = []
    for n = 3:N+1
        if mod(n,2) == 0
            push!(B2, B[n] - B[2])
        else
            push!(B2, B[n] - B[1])
        end
    end

    if vartype==BigFloat
        println("bigfloat")
        B3 = []
        for b in B2
            push!(B3, b / sqrt(integrate(b^2, -1, 1)))
        end
        return B3
    else
        println("float64")
        B3 = []
        for b in B2
            push!(B3, b / sqrt(QuadGK.quadgk(x->b(x)^2, -1,1)[1]))
        end
        return B3
    end
    
    
end


function overlap(B)

    S = zeros(length(B), length(B))
    for (c1, b1) in enumerate(B)
        for (c2, b2) in enumerate(B)
            S[c1,c2] = integrate(Polynomial(b1*b2), -1,1)
        end
    end

    return S

end

function D1(B)

    d = zeros(length(B), length(B))
    @threads for c1 in 1:length(B)
        println("d1 $c1")
        b1 = B[c1]
        for (c2, b2) in enumerate(B)
            d[c1,c2] = integrate(Polynomial(b1*derivative(b2)), -1,1)
        end
    end

    return d

end

function D2(B)

    d = zeros(length(B), length(B))
    @threads for c1 in 1:length(B)
        println("d2 $c1")
        b1 = B[c1]
#    @threads for (c1, b1) in enumerate(B)
#        println("d2 $c1")
        for (c2, b2) in enumerate(B)
            d[c1,c2] = integrate(Polynomial(b1*derivative(derivative(b2))), -1,1)
        end
    end

    return d

end

function D2_grid(B, r1, r2)

    d = zeros(length(B), length(B))
    @threads for c1 in 1:length(B)
        println("d2 $c1")
        b1 = B[c1]
        b1_1 = x->ForwardDiff.derivative(xx->b1(xx), x)
        b1_2 = x->ForwardDiff.derivative(b1_1, x)
        
        for (c2, b2) in enumerate(B)
            d[c1,c2] = QuadGK.quadgk(x->b1_2(x)*b2(x)/r1(x)*r1(x), -1,1)[1]
        end
    end

    return d

end

function D2_grid1(B, r1)

    d = zeros(length(B), length(B))
    @threads for c1 in 1:length(B)
        println("d2 $c1")
        b1 = B[c1]
        b1_1 = x->ForwardDiff.derivative(xx->b1(xx), x)
        b1_2 = x->ForwardDiff.derivative(b1_1, x)
        for (c2, b2) in enumerate(B)
            d[c1,c2] = QuadGK.quadgk(x->b1_2(x)*b2(x)*r1(x)^3, -1,1)[1]
        end
    end

    return d

end

function D1_grid1(B, r1, r2)

    d = zeros(length(B), length(B))
    @threads for c1 in 1:length(B)
        println("d1 $c1")
        b1 = B[c1]
        b1_1 = x->ForwardDiff.derivative(b1, x)
        for (c2, b2) in enumerate(B)
            d[c1,c2] = QuadGK.quadgk(x->b1_1(x)*b2(x)*r2(x)^1*r1(x), -1,1)[1]
        end
    end

    return d

end


function overlap_grid(B, r1)

    d = zeros(length(B), length(B))
    @threads for c1 in 1:length(B)
        println("d2 $c1")
        b1 = B[c1]
#        b1_1 = x->ForwardDiff.derivative(xx->b1(xx)/r1(xx), x)
#        b1_2 = x->ForwardDiff.derivative(b1_1, x)
        
        for (c2, b2) in enumerate(B)
            d[c1,c2] = QuadGK.quadgk(x->b1(x)*b2(x)*r1(x), -1,1)[1]
        end
    end

    return d

end

function do_integral_direct(V, N)

    B = get_cheb_bc(N, vartype=Float64)
    VV = zeros(length(B), length(B))
    @threads for c1 in 1:length(B)
        println("c1 $c1")
        b1 = B[c1]
        for (c2, b2) in enumerate(B)
            VV[c1,c2] = QuadGK.quadgk(x->b1(x)*b2(x)*V(x), -1,1)[1]
        end
    end

    return VV
end




function get_r(α, rmax)

#=    b = rmax
    a = 0.0
    β = (b-a) / ((exp(α * (b-a))- 1 ) )
    function r(x)
        return β*(exp(α*(x-a)) - 1 )  + a
    end
=#
    function r(x)
        return 2.0 + 1.0*x + 0.0*(x-1)*(x+1)
    end

    
    r1 = x->ForwardDiff.derivative(r, x)
    r2 = x->ForwardDiff.derivative(r1, x)
    r3 = x->ForwardDiff.derivative(r2, x)
    
    
    return r, r1, r2, r3
    
end

function get_r_trivial()

    function r(x)
        return x
    end

    r1 = x->ForwardDiff.derivative(r, x)
    r2 = x->ForwardDiff.derivative(r1, x)
    r3 = x->ForwardDiff.derivative(r2, x)
    
    
    return r, r1, r2, r3
    
end





function hydrogen(Z, N; a = 0.0, b = 20.0)

    
    function V(r)
        return -Z / r
    end

    function X(r)
        return -1.0 + 2 * (a - r)/(a-b)
    end
    function R(x)
        return a .+ (x .- -1.0)*(b-a) / 2.0
    end
    

    
    function Vx(x)
        return V(R(x))
    end

    B = get_cheb_bc(N)
    println("S")
    @time S = overlap(B)
    println("S")
    @time d2 = D2(B)

    println("int")
    @time INT = do_integral_direct(Vx, N)

    vals, vects = eigen(-0.5*(d2[1:N-1, 1:N-1] /  (b-a)^2*2^2  ) + INT , (S[1:N-1, 1:N-1]))

    return vals

end

function hydrogen_g2(Z, N;  α = 0.1, a = 0.0, b = 20.0, M = -1)

    #PTS, W, Bvals,S, d2; 
    
#    if M == -1
#        M =  Int64(round(N * 1.5 + 3))
#    end
#    M = min(M, size(PTS)[1])


    if abs(α) < 1e-10
        println("trivial")
        r, r1, r2, r3 = get_r_trivial()
    else
        println("α $α")
        r, r1, r2, r3 = get_r(α, b)
    end
        
    function V(rr)
        return -Z / rr
    end

    function X(rr)
        return -1.0 + 2 * (a - rr)/(a-b)
    end
    function R(x)
        return a .+ (x .- -1.0)*(b-a) / 2.0
    end
    
    function Vr(xx)
        V(r(xx))
    end
    
    function Vm(xx)

        return (3*r2(xx)^2 - 2*r3(xx)*r1(xx)) / (8*r1(xx)^4)

    end

    return Vm, Vr
    
    function Vx(x)
        return Vr(R(x)) + Vm(R(x))
    end

#    function Vt(x)
#        return Vr(x) + Vm(x)
#    end

#    return Vt
    
    B = get_cheb_bc(N, vartype=Float64)
    println("S")
    @time S = overlap(B)
    println("D2")
    @time d2 = D2_grid(B, r1)

    
    
    #INT = do_integral(Vx, N, PTS, W, Bvals; M = M)
    INT = do_integral_direct(Vx, N)

    #   INT = do_integral(Vx,  PINT, PTS)
    vals, vects = eigen(-0.5*(d2[1:N-1, 1:N-1] /  (b-a)^2*2^2  ) + INT , (S[1:N-1, 1:N-1]))

    return vals

end


function hydrogen_g3(Z, N;  α = 0.1, a = 0.0, b = 20.0, M = -1)

    #PTS, W, Bvals,S, d2; 
    
#    if M == -1
#        M =  Int64(round(N * 1.5 + 3))
#    end
#    M = min(M, size(PTS)[1])


    if abs(α) < 1e-10
        println("trivial")
        r, r1, r2, r3 = get_r_trivial()
    else
        println("α $α")
        r, r1, r2, r3 = get_r(α, b)
    end
        
    function V(rr)
        return -Z / rr
    end

    function X(rr)
        return -1.0 + 2 * (a - rr)/(a-b)
    end
    function R(x)
        return a .+ (x .- -1.0)*(b-a) / 2.0
    end
    
    function Vr(xx)
        V(r(xx))
    end
    


    
    function Vx(x)
        return Vr(R(x)) 
    end

#    function Vt(x)
#        return Vr(x) + Vm(x)
#    end

#    return Vt
    
    B = get_cheb_bc(N, vartype=Float64)
    println("S")
    @time S = overlap(B)
    println("D2")
    @time d2 = D2_grid1(B, r1)
    @time d1 = D1_grid1(B, r2)

    
    
    #INT = do_integral(Vx, N, PTS, W, Bvals; M = M)
    #INT = do_integral_direct(Vx, N)
    INT = 0.0
    #   INT = do_integral(Vx,  PINT, PTS)
    vals, vects = eigen(-0.5*(d2[1:N-1, 1:N-1] /  (b-a)^2*2^2)  + -0.5*d1[1:N-1, 1:N-1]/  (b-a)^2*2^2    .+ INT , (S[1:N-1, 1:N-1]))

    return vals

end


















end #end module
