module SCF


using LinearAlgebra
using ChebyshevQuantum
using Polynomials
using SpecialPolynomials
using ..LDA:v_LDA
using ..Hartree:V_Ha
using ..Hartree:V_H3
using QuadGK

function hydrogen(; N = 40, l = 0, Z=1.0, Rmax = 10.0)


    function V(x)
        return -2.0/ (Rmax*(x+1)) + 4.0/Rmax^2 * l*(l+1) / (x+1)^2
        #return -1.0/x
    end

    vals, vects = ChebyshevQuantum.SolveEig.solve(N=N,A0=V, A2 =-2.0/Rmax^2, dosplit=false)

    #    vals, vects = ChebyshevQuantum.SolveEig.solve(N=N,V=V, A2 =-0.5, dosplit=false, a = 0, b = 50.0)

    
    println("VALS ")
    println(vals)

end


function hydrogen2(; N = 40, l = 0, Z=1.0, Rmax = 10.0)


    function V(x)
        #return -2.0/ (Rmax*(x+1)) + 4.0/Rmax^2 * l*(l+1) / (x+1)^2
        return -1.0/x
    end

    #    vals, vects = ChebyshevQuantum.SolveEig.solve(N=N,V=V, A2 =-2.0/Rmax^2, dosplit=false)

    vals, vects = ChebyshevQuantum.SolveEig.solve(N=N,A0=V, A2 =-0.5, dosplit=false, a = 0, b = Rmax)

    
    println("2 VALS ")
    println(vals)

    return vals, vects
    
end

function hydrogen3(; N = 40, l = 0, Z=1.0, Rmax = 10.0)


    function V(x)
        #return -2.0/ (Rmax*(x+1)) + 4.0/Rmax^2 * l*(l+1) / (x+1)^2
        return -1.0/(x+1)
    end

    #    function D(x)
    #return -2.0/ (Rmax*(x+1)) + 4.0/Rmax^2 * l*(l+1) / (x+1)^2
    #        return -0.5 * x
    #    end
    
    #    vals, vects = ChebyshevQuantum.SolveEig.solve(N=N,V=V, A2 =-2.0/Rmax^2, dosplit=false)

    vals, vects = ChebyshevQuantum.SolveEig.solve(N=N,A0=V, A2 = -0.5, dosplit=false, a = 0, b = Rmax)

    
    println("2 VALS ")
    println(vals)

    return vals, vects
    
end


function hydrogen4(; N = 40, l = 0, Z=1.0, Rmax = 10.0)


    pts = ChebyshevQuantum.Interp.getCpts(N);
    D = ChebyshevQuantum.Interp.getD(pts);
    D2 = D*D

    pts_a = pts[2:N]
    D2_a = D2[2:N,2:N]

    function V(x)
        return -1.0/(x) + l*(l+1)/2.0/x^2
    end

    r = (1 .+ pts_a) * Rmax / 2
    
    Ham = (-0.5*4.0/Rmax^2) * D2_a + diagm(V.(r))

    @time vals, vects = eigen( Ham )
    
    println("vals ", real.(vals[1:4]))
    return real.(vals), vects
    
end


function hydrogen5(; N = 40, l = 0, Z=1.0, Rmax = 10.0)

    function A2(x)
        return -0.5*x
    end

    function B(x)
        return x
    end
    
    
    vals, vects = ChebyshevQuantum.SolveEig.solve(N=N,A0=-1.0, A2 = A2, B= B, dosplit=false, a = 0, b = Rmax)
    println("vals ", real.(vals[1:4]))
    return real.(vals), vects
end

function hydrogen6(; N = 40, l = 0, Z=1.0, Rmax = 10.0)

    function A2(x)
        return -0.5*x^2
    end

    function B(x)
        return x^2
    end

    function V(x)
        return -1.0*x + (l)*(l+1)/2.0
    end
    
    
    vals, vects = ChebyshevQuantum.SolveEig.solve(N=N,A0=V, A2 = A2, B= B, dosplit=false, a = 0, b = Rmax)
    println("vals ", real.(vals[1:4]))
    return vals, vects
end

function setup_mats(;N = N, Rmax=Rmax)
    a = 0.0
    b = Rmax
    
    pts = ChebyshevQuantum.Interp.getCpts(N);
    D = ChebyshevQuantum.Interp.getD(pts) / ((b-a)/2.0) ;
    D2 = D*D

    pts_a = pts[2:N]
    r = (1 .+ pts) * Rmax / 2    
    r_a = (1 .+ pts[2:N]) * Rmax / 2    
    D_a = D[2:N, 2:N]
    D2_a = D2[2:N,2:N]

    w, pts2 = ChebyshevQuantum.Interp.getW(N)

    w_a = w[2:N]*(b - a)/2.0

    H_poisson = D2_a +  diagm( 2.0 ./ r_a )  * D_a
    
    return r_a, D_a, D2_a, w_a, r, w *(b - a)/2.0, H_poisson

    
end

function getVcoulomb(r_a, Z)
    
    return diagm( (-Z)./r_a)
    
end

function getKE(D2_a)
    
    return -0.5*D2_a
    
end

function my_laguerre(α, n; mode=:small)

    if mode == :small
        z = zeros(Int64, n+1)
        z[n+1] = 1
        p = Laguerre{α}(z)
        pp = convert(Polynomial, p)    
    else
        z = zeros(BigInt, n+1)
        z[n+1] = 1
        p = Laguerre{α}(z)
        pp = convert(Polynomial, p)
    end
    
    return  pp

end

function h_rad(n, l;Z=1.0)
    #returns a function
    #radial hydrogen-like wavefunctions
    #units where Bohr radius  a0 = 1
    #note : hydrogen basis not complete unless you add continuum states

    
    if n <= 0 || l < 0 || l >= n
        println("ERROR h_rad n l $n $l")
    end
    
    c= 2.0*Z/n
    norm = sqrt( (2.0*Z  / n )^3 * factorial(n - l - 1) / (2 * n * factorial(n+l) )) * c^l 
    myl = my_laguerre( 2*l+1 ,n-l-1   )

    function f0(r)
        return norm * exp.(-c*r/2.0) .* r.^l  * myl.(c*r)
    end        

    return f0
    
end

function get_initial_rho(rho_init, N, Z, nel, r, w)

    if !ismissing(rho_init)
        if typeof(rho_init) <: Array
            if length(rho_init) != N+1
                println("ERROR, initial rho doesn't match N=$N")
                throw(DimensionMismatch("ERROR, initial rho doesn't match N"))
            else
                rho = copy(rho_init)
            end
        else
            rho = rho_init.(r)
        end

    else #ismissing case

        rho = zeros(N+1)

        # default filling order
        order = [[1,0],[2,0],[2,1],[3,0],[3,1],[4,0],[3,2],[4,1], [5,0],[4,2],[5,1],[6,0],[4,3],[5,2],[6,1],[7,0],[5,3],[6,2],[7,1]]

        nleft = nel
        for (n,l) in order

            if l == 0
                maxfill = 2
            elseif l == 1
                maxfill = 6
            elseif l == 2
                maxfill = 10
            elseif l == 3
                maxfill = 14
            end

            psi = h_rad(n, l;Z=Z)
            if nleft <= maxfill + 1e-15
                println("size rho $(size(rho)) r $(size(r)) psir $(size(psi.(r))) ")
                rho += psi.(r).^2 * nleft
                break
            else
                rho += psi.(r).^2 * maxfill
                nleft -= maxfill
            end
        end
    end        
    
    #check normalization
    rho = rho / (4*pi)
    norm = 4 * pi * sum(rho .* r.^2 .* w)
    println("X check norm $norm")
    #    rho = rho / norm * nel
    
    return rho
    
end


function DFT(; N = 40, nmax = 1, lmax = 0, Z=1.0, Rmax = 10.0, nel = missing, rho_init = missing, mix = 0.5, niters=20)

    Z = Float64(Z)
    if ismissing(nel)
        nel = Float64(Z)
    end

    begin
        r, D, D2, w, rall,wall,H_poisson = setup_mats(;N = N, Rmax=Rmax)
        
        rho = get_initial_rho(rho_init, N, Z, nel,rall,wall)

        Vc = getVcoulomb(r, Z)
        KE = getKE(r)
        
        H0 = -0.5*D2 + Vc
        
        Ham = zeros(size(H0))
        
        vects = zeros(Complex{Float64}, size(H0))
        vals_r = zeros(Float64, size(H0)[1])    
    end

    for iter = 1:niters
        
        VLDA = diagm(v_LDA.(rho[2:N]))
        VH = diagm(V_H3(rho[2:N], r,w,H_poisson,Rmax, rall, nel))

        Ham[:,:] = H0 + (VH + VLDA)

        vals, vects = eigen(Ham)
        vals_r = real.(vals)
        println("$iter vals ", vals_r[1:4])
        
        rho = assemble_rho(vals_r, vects, nel, r, w, rho, N)
        
    end

    return vals_r, vects, rho,r
    
end    

function assemble_rho(vals_r, vects, nel, r, w, rho, N)

    inds = sortperm(vals_r)
    nleft = nel
    rho[:] .= 0.0

#    println(size(rho))
#    println(size(vects))
    
    for i in inds
        vnorm = real.(vects[:,i].*conj(vects[:,i])) ./ r.^2
        vnorm = vnorm / (4.0*pi*sum(vnorm .* w .* r.^2))
        if nleft <= 2.0 + 1e-10

            rho[2:N] += vnorm * nleft
            break
        else

            rho[2:N] += vnorm * 2.0
            nleft -= 2.0
        end
    end    

    norm = 4.0*pi*sum(rho[2:N] .* w .* r.^2)
#    println("check norm ", norm)
    rho = rho/ norm * nel
    return rho
    
end

end #end module
