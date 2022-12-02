module Galerkin
using Polynomials
using SpecialPolynomials
using Base.Threads
using LinearAlgebra
using ForwardDiff
using QuadGK

function hydrogen(Z, N, PTS, W, Bvals,S, d2; a = 0.0, b = 20.0, M = -1)

    if M == -1
        M =  Int64(round(N * 1.5 + 3))
    end
    M = min(M, size(PTS)[1])

    println("old M $M ")
    
#    PINT = pre_do_integral(N, PTS, W, Bvals)    

    function V(r)
        return -Z / r
    end

#=
    function X(r)
        return (r - L) / (r+L)
    end
    function R(x)
        return L*(1 + x) / (1 - x)
    end

    function dR(x)
        return L*(  1.0 / (1 - x) + (1+x)*(1-x)^2)
    end
    =#

    function X(r)
        return -1.0 + 2 * (a - r)/(a-b)
    end
    function R(x)
        return a .+ (x .- -1.0)*(b-a) / 2.0
    end
    

    
    function Vx(x)
        return V(R(x))
    end

    
    INT = do_integral(Vx, N, PTS, W, Bvals; M = M)


    #   INT = do_integral(Vx,  PINT, PTS)
    vals, vects = eigen(-0.5*(d2[1:N-1, 1:N-1] /  (b-a)^2*2^2  ) + INT , (S[1:N-1, 1:N-1]))

    return vals

end

#=function hydrogen_grid(Z, N, PTS, W, Bvals,S, d2; a = 0.0, b = 20.0, M = -1, α = 0.1)

    if M == -1
        M = min(203, Int64(round(N * 1.5 + 3)))
    end
#    PINT = pre_do_integral(N, PTS, W, Bvals)    

  
    function V(r)
        return -Z / r
    end

    β = b / (exp(α * b)-1)
    
    function r(x)
        return β*(exp(α*x) - 1)
    end

    r1 = x -> ForwardDiff.Derivative(r, x)
    r2 = x -> ForwardDiff.Derivative(r1, x)
    r3 = x -> ForwardDiff.Derivative(r2, x)
    
    
#=
    function X(r)
        return (r - L) / (r+L)
    end
    function R(x)
        return L*(1 + x) / (1 - x)
    end

    function dR(x)
        return L*(  1.0 / (1 - x) + (1+x)*(1-x)^2)
    end
    =#

    function X(r)
        return -1.0 + 2 * (a - r)/(a-b)
    end
    function R(x)
        return a .+ (x .- -1.0)*(b-a) / 2.0
    end
    

    function Vm(x)

        return (3 * (r2(r(R(x))))^2 - 2 * r3(r(R(x))) * r1(r(R(x))) )  / (4 * r1(r(R(x)))^4)
        
    end
    
    
    function Vx(x)
        return (V(r(R(x))) + (3 * (r2(r(R(x))))^2 - 2 * r3(r(R(x))) * r1(r(R(x))) )  / (4 * r1(r(R(x)))^4) ) * r1(R(x))
    end

    
    INT = do_integral(Vx, N, PTS, W, Bvals; M = M)


    #   INT = do_integral(Vx,  PINT, PTS)
    vals, vects = eigen(-0.5*(d2[1:N-1, 1:N-1] /  (b-a)^2*2^2  ) + INT , (S[1:N-1, 1:N-1]))

    return vals

end
=#    

function do_integral(fn, N, PTS, W, Bvals; M = 199)

    println("do int $N $M")
    INT = zeros(N-1, N-1)
    f = fn.(@view PTS[M, 2:M+2]) .* (@view W[M, 2:M+2])
    @threads for n1 = 1:(N-1)
        for n2 = 1:(N-1)
            #INT[n1, n2] = sum( (@view W[N, 1:N+3]) .* (@view Bvals[N, n1, 1:N+3]).*(@view Bvals[N, n2,1:N+3]) .* f)

            INT[n1, n2] = sum(  (@view Bvals[M, n1, 2:M+2]).*(@view Bvals[M, n2,2:M+2]) .* f)

        end
    end
    
    return (INT+INT')/2.0

end

#=function do_integral(fn, PINT, PTS)

    N = size(PINT)[1]+1
    f = fn.(@view PTS[N, 1:N+3])
    
    INT = zeros(N-1, N-1)
    @threads for n1 = 1:(N-1)
        for n2 = 1:(N-1)
            INT[n1, n2] = sum( f .* (@view PINT[n1,n2,:]))
        end
    end
    
    return INT

end
=#


function do_integral(fn, PINT, PTS)

    N = size(PINT)[2]+1
    f = fn.(@view PTS[N, 2:N+2])
    #println(size(f))
    
    INT = zeros(N-1, N-1)
    @threads for n1 = 1:(N-1)
        for n2 = n1:(N-1)
            @inbounds INT[n1, n2] = sum( f .* (@view PINT[2:end-1,n1,n2]))
        end
    end
    
    return (INT+INT') - diagm(diag(INT))

end

function pre_do_integral(N, PTS, W, Bvals)

    PINT = zeros(N-1, N-1, N+3)
    @threads for n1 = 1:(N-1)
        for n2 = 1:(N-1)
            PINT[n1, n2, 1:N+3] =  (@view W[N, 1:N+3]) .* (@view Bvals[N, n1, 1:N+3]).*(@view Bvals[N, n2,1:N+3])
        end
    end
    PINT_X = permutedims(PINT, [3,1,2])
    return PINT_X

end


function solve_problem(fn, N, PTS, W, Bvals, d2, S)

    @time INT = do_integral(fn,  N+1, PTS, W, Bvals)
    @time vals, vects = eigen(-0.5*(@view d2[1:N, 1:N]) + INT, (@view S[1:N, 1:N]))

    return vals, vects, -0.5*d2[1:N, 1:N] + INT, S[1:N, 1:N]
                              
end

function solve_problem(fn, N, PTS, PINT,  d2, S)

    @time INT = do_integral(fn,  PINT, PTS)
    @time vals, vects = eigen(-0.5*(@view d2[1:N-1, 1:N-1]) + INT, (@view S[1:N-1, 1:N-1]))

    
    return vals, vects , -0.5*d2[1:N-1, 1:N-1] + INT, S[1:N-1, 1:N-1]
                              
end



function setup_integral_mats(NN, Nmax)

    PTS = zeros(Nmax, Nmax+3)
    W = zeros(Nmax, Nmax+3)

    Bvals = zeros(Nmax,NN,Nmax+3)
    
    B = get_cheb_bc(Nmax);

    nnX = 1
    function get(x)
        return B[nnX](x)
    end
    
    for N = 1:Nmax
        println("N $N")
        w, pts = setup_integrals(N);
        PTS[N,1:N+3] = pts
        W[N,1:N+3] = w
        for nn = 1:(NN-1)
            nnX = nn
            Bvals[N, nn, 1:N+3] = get.(Float64.(pts))
        end
        
    end

    return PTS, W, Bvals
        
end


function setup_integrals(N)

    N = N+2
    
    pts = -cos.( Float64(pi)*(0:N) / N)    
    Nd2 = Int64(floor(N/2))
#    pts = getCpts(N)
    w = zeros(N+1)
    z = zeros(Float64, Nd2*2+1)

    for m = 0:Nd2
        z[:] .= 0.0
        z[2*m+1] = 1.0
        T2m = ChebyshevT(z)

        if m == 0 || m == Nd2
            f2 = 0.5  / (1 - 4*m^2)
        else
            f2 = 1.0  / (1 - 4*m^2)
        end
        
        for j = 1:(N+1)
            
            if j == 1 || j == N+1
                f = 2.0/N
            else
                f = 4.0/N
            end
    
        
            w[j] += f*f2*T2m(pts[j])
            
        end
    end
    return w, pts

end



function get_cheb_bc(N)

    B = basis.(Chebyshev{Float64}, 0:N);

    B2 = []
    for n = 3:N+1
        if mod(n,2) == 0
            push!(B2, B[n] - B[2])
        else
            push!(B2, B[n] - B[1])
        end
    end

    B3 = []
    for b in B2
        push!(B3, b / sqrt(integrate(b^2, -1, 1)))
    end
    return B3
    
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

function testS(B, n)

    return integrate(Polynomial(B[n]*B[n]), -1,1)

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



function f(V, B)

    d = zeros(length(B), length(B))
    for (c1, b1) in enumerate(B)
        for (c2, b2) in enumerate(B)
            d[c1,c2] = integrate(V.*Polynomial(b1*b2), -1,1)
        end
    end

    return d

end


function get_cheb_bc_grid(N, r; vartype=Float64)

    B = basis.(Chebyshev{vartype}, 0:N);

    B2 = []
    for n = 3:N+1
        if mod(n,2) == 0
            push!(B2, B[n] - B[2])
        else
            push!(B2, B[n] - B[1])
        end
    end

    B3 = []
    for (c, b) in enumerate(B2)
        #push!(B3, b / sqrt(integrate(b^2, -1, 1)))
#        println("b $b r(0.1) $(r(0.1))")
        println("c $c")
        function g(x)
            return b(r(x))^2
        end
        push!(B3, b / sqrt(QuadGK.quadgk(g, -1,1, rtol=1e-10)[1]))
        
    end
    return B3
    
end

function get_r(α)

    b = 1.0
    a = -1.0
    β = (b-a) / ((exp(α * (b-a))- 1 ) )
    function r(x)
        if abs(α) < 1e-5
            return x
        else
            return β*(exp(α*(x-a)) - 1 )  + a
        end
    end

    return r
    
end

function stuff(N, α)
    #C = basis.(Chebyshev{Float64}, 0:N)

#    b = 1.0
#    a = -1.0
#    β = (b-a) / ((exp(α * (b-a))- 1 ) )
#    function r(x)
#        return β*(exp(α*(x-a)) - 1 )  + a
#    end

    r = get_r(α)
    
    B = get_cheb_bc_grid(N, r)
    S = zeros(length(B), length(B))

    C1 = 1
    C2 = 1
    
    function over(x)

        return B[C1](r(x))*B[C2](r(x))
    end

    println("overlaps")
    for c1 = 1:length(B)
        for c2 = 1:length(B)

            C1=c1
            C2=c2
#    for (c1, b1) in enumerate(B)
#        for (c2, b2) in enumerate(B)
            S[c1,c2] = QuadGK.quadgk(over, -1,1, rtol=1e-10)[1]
            #p = Polynomial(b1*b2)
            #S[c1,c2] = QuadGK.quadgk(p, -1,1)[1]
        end
    end
    
    d2 = zeros(length(B), length(B))

    B2 = []
    for (c1, b1) in enumerate(B)
        b1_1 = bb -> ForwardDiff.derivative(aa->b1(r(aa)), bb)
        b1_2 = cc -> ForwardDiff.derivative(b1_1, cc)
        push!(B2, b1_2)
    end
    function d2int(x)
        return B[C1](r(x))*B2[C2](x)
    end

    println("derivs")
    for c1 in 1:length(B)
    #    for (c1, b1) in enumerate(B)
#        b1_1 = bb -> ForwardDiff.derivative(aa->b1(r(aa)), bb)
#        b1_2 = cc -> ForwardDiff.derivative(b1_1, cc)
        for (c2, b2) in enumerate(B)
            C1=c1                                                                                     
            C2=c2   
            d2[c1,c2] = QuadGK.quadgk(d2int, -1,1, rtol=1e-10)[1]


            #p = Polynomial(b1*b2)
            #S[c1,c2] = QuadGK.quadgk(p, -1,1)[1]
        end
    end
    
    S = (S+S')/2.0
    d2 = (d2 + d2')/2.0
            

    return S, d2
end



function hydrogen_grid(Z, N, PTS, W, Bvals,S, d2, vint; a = 0.0, b = 20.0, M = -1)
    println("hgrid")
#    M = 35
#    M = N

    if M > size(PTS)[1]
        M = size(PTS)[1]
    end
    
    if M == -1
        M = minimum(Int64(round(N * 1.5 + 3)), size(PTS)[1] )
    end

    println("M $M")
    
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

#    return Vx, R, X, V
    
#    INT = do_integral(
    @time INT = do_integral(Vx, N, PTS, W, Bvals; M = M)

    #@time vals, vects = eigen(-0.5*(d2[1:N-1, 1:N-1] /  (b-a)^2*2^2  ) + Z*vint[1:N-1, 1:N-1] , (S[1:N-1, 1:N-1]))
    @time vals, vects = eigen(-0.5*(d2[1:N-1, 1:N-1] /  (b-a)^2*2^2  ) + INT, (S[1:N-1, 1:N-1]))

    return vals,  Vx

end

function setup_integral_mats_grid(NN, Nmax, α)

    println("setup")
    @time begin
        #Nmax = Int64(round(NN * 2.0 + 3))
        PTS = zeros(Nmax, Nmax+3)
        W = zeros(Nmax, Nmax+3)

        Bvals = zeros(Nmax,NN,Nmax+3)
        
        r = get_r(α)
        
        B = get_cheb_bc_grid(Nmax, r)
    end

    println("loop")
    nnX = 1
    function get(x)
        return B[nnX](r(x))
    end
    
    @time for N = 1:Nmax
    #for N = [Nmax]
        println("N $N")

        @time w, pts = setup_integrals(N);

        PTS[N,1:N+3] = pts
        W[N,1:N+3] = w
        @time for nn = 1:(NN-1)
            nnX = nn
            #Bvals[N, nn, 1:N+3] = B[nn].(r.(Float64.(pts)))
            Bvals[N, nn, 1:N+3] = get.(Float64.(pts))


            #Bvals[N, nn, 1:N+3] = B[nn].(r.(pts))
        end
        
    end

    return PTS, W, Bvals
        
end


function do_integral_direct(V, B, r)

    #B = get_cheb_bc(N, vartype=Float64)
    VV = zeros(length(B), length(B))

    function core(x, a1,a2)
        return B[a1](r(x))*B[a2](r(x))*V(x)
    end
    
    for c1 in 1:length(B)
        println("c1 $c1")
        b1 = B[c1]
        for (c2, b2) in enumerate(B)
            VV[c1,c2] = QuadGK.quadgk(x->b1(r(x))*b2(r(x))*V(x), -1,1)[1]
            #VV[c1,c2] = QuadGK.quadgk(x->core(x, c1,c2), -1, 1, rtol = 1e-10)[1]
        end
    end

    return VV
end

end #end module
