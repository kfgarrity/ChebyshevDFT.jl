module Galerkin
using Polynomials
using SpecialPolynomials
using Base.Threads
using LinearAlgebra
using ForwardDiff
using QuadGK

struct gal
    
    N::Int64
    M::Int64
    a::Float64
    b::Float64
    α::Float64
    bvals::Array{Float64, 3}
    pts::Array{Float64, 2}
    w::Array{Float64, 2}
    s::Array{Float64, 2}
    invs::Array{Float64, 2}
    d1::Array{Float64, 2}
    d2::Array{Float64, 2}
    r
    invr
    R
    X
    B
end

Base.show(io::IO, g::gal) = begin
    println("Galerkin obj")
    println("N=$(g.N) M=$(g.M) a=$(g.a) b=$(g.b) α=$(g.α)")
end

function makegal(N,a,b; α=0.0, M = -1)

    a = Float64(a)
    b = Float64(b)
    
    if M == -1
        M = Int64(round(N*2.5))
    end

    r, invr = get_r(α)
    B = get_cheb_bc_grid(N, r)
    
    s,d1,d2 = get_sd(r, B, α)

    pts, w, bvals = setup_integral_mats_grid_new(N, M, r, invr,  B)
    #    println(size(bvals))

    function R(x)
        return a .+ (x .- -1.0)*(b-a) / 2.0
    end
    function X(r)
        return -1.0 + 2 * (a - r)/(a-b)
    end

    return gal(N,M,a,b,α,bvals,pts,w,s,inv(s),d1,d2,r,invr,R,X,B)

end

function do_1d_integral(f, g::gal; M=-1)
    if M == -1
        M = g.M
    end

    return sum( (@view g.w[2:M+2M]) .* f.(g.R.(@view g.pts[2:M+2,M]))) * (g.b - g.a) / 2.0

end

function do_1d_integral(arr::Vector, g::gal, M=-1)
    if M == -1
        M = g.M
    end
    N = length(arr)
    ret = zero(arr[1])
    for c = 1:N
        ret += arr[c] *  sum(g.bvals[2:M+2, c, M] .* g.w[2:M+2,M])
    end
    return ret * (g.b - g.a)/ 2.0
end

function get_gal_rep(f, g::gal; N=-1, M=-1, invS=missing)

    if M == -1
        M = g.M
    end
    if N == -1
        N = g.N
    end
    if ismissing(invS)
        invS = inv(gal.s[1:N, 1:N])
    end

    nr = f.(g.R.(g.pts[2:M+2,M])) .*g.w[2:M+2,M]
    
    g_rep = zeros(typeof(nr[1]), g.N-1)
    for i = 1:(g.N-1)
        g_rep[i] =   sum( (@view g.bvals[2:M+2,i,M]).*nr )
    end
    
    return invS*g_rep

end

function get_gal_rep(arr::Array{1}, g::gal; M=-1, invS=missing)

    if M == -1
        M = g.M
    end
    N = length(arr)
    if ismissing(invS)
        invS = inv(gal.s[1:N, 1:N])
    end
    
    arrw = arr .* g.w[2:M+2,M]
    
    g_rep = zeros(typeof(arr[1]), g.N-1)
    for i = 1:(g.N-1)
        g_rep[i] =   sum((@view g.bvals[2:M+2,i,M]).*arrw )
    end
    
    return invS*g_rep

end

function get_rho_gal(rho_rs_M_R2,g::gal; N=-1, invS=missing, l = 0)

    if N == -1
        N = g.N
    end
    if ismissing(invS)
        invS = inv(gal.s[1:N-1, 1:N-1])
    end
    
    M = length(rho_rs_M_R2)-1
    
    nrR = rho_rs_M_R2 .* g.w[2:M+2,M]
    nrR_dR = rho_rs_M_R2 .* g.w[2:M+2,M] ./ g.R.(g.pts[2:M+2,M])

    rho_rs_M = rho_rs_M_R2 ./ g.R.(g.pts[2:M+2,M]).^2

    rho_gal = zeros(Float64, N-1)
    rho_gal_dR = zeros(Float64, N-1)
    rho_gal_multipole = zeros(Float64, N-1)

    nrR_mp = rho_rs_M_R2 .* g.w[2:M+2,M] .* g.R.(g.pts[2:M+2,M]).^l
    
    for i = 1:N-1
        rho_gal[i] =   sum( (g.bvals[2:M+2,i,M]).*nrR )
        rho_gal_dR[i] =   sum( (g.bvals[2:M+2,i,M]).*nrR_dR )
        rho_gal_multipole[i] =   sum( (g.bvals[2:M+2,i,M]).*nrR_mp )
    end

    
#    println("size invS ", size(invS), " rho_gal ", size(rho_gal), " ", size(rho_gal_dR))
    return invS*rho_gal ,  rho_gal_dR, rho_rs_M, invS*rho_gal_multipole
    
end


function gal_rep_to_rspace(r, rep, g::gal)
    N = length(rep)
    f = zeros(typeof(rep[1]), size(r))

    for c = 1:N
        f += g.B[c].(g.r.(g.X.(r)))*rep[c]
    end

    return f
end


function gal_rep_to_rspace(rep, g::gal; M = -1)
    
    if M == -1
        M = g.M
    end

    N = length(rep)
    f = zeros(typeof(rep[1]), M+1)
    
    for c = 1:N
        f += g.bvals[2:M+2,c,M]*rep[c]
    end

    return f
end


function gal_rep_to_rspace(r::Number, rep, g::gal)

    if r > g.b
        return 0.0
    elseif r < g.a
        return 0.0
    end
    
    N = length(rep)
    f = zeros(typeof(rep[1]), size(r))

    f = zero(typeof(rep[1]))
    
    for c = 1:N
        f += g.B[c].(g.r.(g.X.(r)))*rep[c]
    end

    return f
end

function get_gal_rep_matrix_R(fn_R, g::gal; N = -1)

    if N == -1
        N = g.N
    end

    M = length(fn_R)-1
    
    INT = zeros(N-1, N-1)
    f = fn_R .* (@view g.w[2:M+2,M])
#    println("get_gal_rep_matrix_R loop fn_R")
    @inbounds for n1 = 1:(N-1)
        for n2 = n1:(N-1)
            INT[n1, n2] = sum(  (@view g.bvals[2:M+2,n1,M]).*(@view g.bvals[2:M+2,n2,M]) .* f)
        end
    end


    return (INT+INT') - diagm(diag(INT))

end


function get_gal_rep_matrix(fn, g::gal; M = -1, N = -1)

    if M == -1
        M = g.M
    end
    if N == -1
        N = g.N
    end
    
    INT = zeros(N-1, N-1)
#    println("fn")
    f = fn.( g.R.(@view g.pts[2:M+2,M])) .* (@view g.w[2:M+2,M])
    #println("get_gal_rep_matrix loop")
    @inbounds for n1 = 1:(N-1)
        for n2 = n1:(N-1)
            INT[n1, n2] = sum(  (@view g.bvals[2:M+2,n1,M]).*(@view g.bvals[2:M+2,n2,M]) .* f)
        end
    end
    
    return (INT+INT')  - diagm(diag(INT))

end

function get_gal_rep_matrix(arr::Vector, g::gal; M = -1)

    if M == -1
        M = g.M
    end
    N = length(arr)+1
    println("N $N")
    INT = zeros(N-1, N-1)


    arr_m = zeros(M+1)
    for n1 = 1:N-1
        arr_m += g.bvals[2:M+2,n1,M] * arr[n1]
    end

    arr_m = arr_m .* (@view g.w[2:M+2,M])

    #println("get_gal_rep_matrix loop arr")
    @inbounds @threads for n1 = 1:(N-1)
        for n2 = n1:(N-1)
            INT[n1, n2] = sum(  (@view g.bvals[2:M+2,n1,M]).*(@view g.bvals[2:M+2,2,M]) .* arr_m)
        end
    end
    
    return (INT+INT') - diagm(diag(INT))

    
    
end


function get_vh_mat(vh_tilde, g::gal, l, m, MP; M = -1)

    if M == -1
        M = g.M
    end
#    println("m l $m $l ")
#    println("m l $m $l ", MP[l+1, m+l+1])
    
    N = length(vh_tilde)+1
    
    INT = zeros(N-1, N-1)

    vh_tilde_vec = zeros(M+1)

    for n1 = 1:N-1
        vh_tilde_vec += g.bvals[2:M+2,n1,M] * vh_tilde[n1]
    end
    
#    f = (vh_tilde_vec  ./ ( g.R.(@view g.pts[2:M+2,M]))  .+ nel/g.b * sqrt(pi)/(2*pi))  .* (@view g.w[2:M+2,M])

    f = (vh_tilde_vec  ./ ( g.R.(@view g.pts[2:M+2,M]))  .+ MP[l+1, m+l+1]/g.b^(l+1) * sqrt(pi)/(2*pi))  .* (@view g.w[2:M+2,M])
    
    @threads for n1 = 1:(N-1)
        for n2 = 1:(N-1)
            INT[n1, n2] = sum(  (@view g.bvals[2:M+2,n1,M]).*(@view g.bvals[2:M+2,n2,M]) .* f)
        end
    end
    
    return (INT+INT')/2.0

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
        #        println("c $c")
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

    function invr(rr)
        if abs(α) < 1e-5
            return rr
        else
            return log( (rr - a)/β + 1)/α + a
        end
    end
    
    return r, invr
    
end



function get_sd(r, B, α)

    N = length(B)
    #    r, invr = get_r(α)
    #    B = get_cheb_bc_grid(N, r)
    S = zeros(length(B), length(B))

    C1 = 1
    C2 = 1
    
    function over(x)

        return B[C1](r(x))*B[C2](r(x))
    end

    #    println("overlaps")
    for c1 = 1:length(B)
        for c2 = 1:length(B)
            C1=c1
            C2=c2
            if abs(α) < 1e-7 && mod(c1,2) != mod(c2,2)
                S[c1,c2] = 0.0
            else
                S[c1,c2] = QuadGK.quadgk(over, -1,1, rtol=0.5e-10)[1]
            end
        end
    end
    
    d2 = zeros(length(B), length(B))
    d1 = zeros(length(B), length(B))

    B2 = []
    B1 = []
    for (c1, b1) in enumerate(B)
        b1_1 = bb -> ForwardDiff.derivative(aa->b1(r(aa)), bb)
        b1_2 = cc -> ForwardDiff.derivative(b1_1, cc)
        push!(B2, b1_2)
        push!(B1, b1_1)
    end
    function d2int(x)
        return B[C1](r(x))*B2[C2](x)
    end
    function d1int(x)
        return B[C1](r(x))*B1[C2](x)
    end

    #   println("derivs")
    for c1 in 1:length(B)
        for (c2, b2) in enumerate(B)
            C1=c1
            C2=c2   
            d2[c1,c2] = QuadGK.quadgk(d2int, -1,1, rtol=0.5e-10)[1]
            #            if c1 == c2
            #                d1[c1,c2] = 0.0
            #            else
            #                d1[c1,c2] = QuadGK.quadgk(d1int, -1,1, rtol=0.5e-10)[1]
            #            end
        end
    end
    
    S = (S+S')/2.0
    d2 = (d2 + d2')/2.0
    d1 = (d1 - d2')/2.0

    return S, d1, d2
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


function setup_integral_mats_grid_new(NN, Nmax, r, invr, B)

    #    println("setup")
    begin
        #Nmax = Int64(round(NN * 2.0 + 3))
        #        println("Nmax $Nmax")
        PTS = zeros(Nmax+3, Nmax)
        W = zeros(Nmax+3, Nmax)

        Bvals = zeros(Nmax+3,NN-1,Nmax)
        

        #        r, invr = get_r(α)
        #        B = get_cheb_bc_grid(Nmax, r)

        dinvr = xx-> ForwardDiff.derivative(invr, xx)
    end

    #    println("loop")
    nnX = 1
    function get(x)
        return B[nnX](x)
    end
    
    for N = 1:Nmax
        #for N = [Nmax]
        #        println("N $N")

        w, pts = setup_integrals(N);

        PTS[1:N+3,N] = invr.(pts)
        W[1:N+3, N] = w .* dinvr.(pts)

        #        PTS[N,1:N+3] = pts
        #        W[N,1:N+3] = w 
        
        for nn = 1:(NN-1)
            nnX = nn
            Bvals[1:N+3, nn, N] = get.(Float64.(pts))

        end
        
    end

    return PTS, W, Bvals
    
end



end #end module
