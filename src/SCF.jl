module SCF


using LinearAlgebra
using ChebyshevQuantum
using Polynomials
using SpecialPolynomials
using ..LDA:v_LDA
using ..LDA:e_LDA

using ..LDA:v_LDA_sp
using ..LDA:e_LDA_sp

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
        rho = rho / (4*pi)
        
    end        
    
    #check normalization
    norm = 4 * pi * sum(rho .* r.^2 .* w)
#    println("X check norm $norm")
    rho = rho / norm * nel
#    norm = 4 * pi * sum(rho .* r.^2 .* w)
    println("X check norm $norm")
    
    return rho
    
end


function DFT(; N = 40, nmax = 1, lmax = 0, Z=1.0, Rmax = 10.0, nel = missing, rho_init = missing, niters=20, mix = 0.7)

    Z = Float64(Z)
    if ismissing(nel)
        nel = Float64(Z)
    end

    begin
        r, D, D2, w, rall,wall,H_poisson = setup_mats(;N = N, Rmax=Rmax)
        
        rho = get_initial_rho(rho_init, N, Z, nel,rall,wall)
        rho_old = deepcopy(rho)
        
        Vc = getVcoulomb(r, Z)
        KE = getKE(r)
        
        H0 = -0.5*D2 + Vc
        
        Ham = zeros(size(H0))
        
        vects = zeros(Complex{Float64}, size(H0))
        vals_r = zeros(Float64, size(H0)[1])    

        VH = zeros(size(rho))
        filling = zeros(size(rho))
        Vtot = zeros(size(rho))
        
    end


    
    for iter = 1:niters
        
        #        VLDA = diagm(v_LDA.(rho[2:N]))
        VLDA = diagm(v_LDA_sp.(rho[2:N], 1.0))
        println("max vlda", maximum(VLDA))
        VH[:] = V_H3(rho[2:N], r,w,H_poisson,Rmax, rall, nel)
        VH_mat = diagm(VH[2:N])
        Vtot = VH[2:N] + diag(VLDA) + diag(Vc)
        
        
        Ham[:,:] = H0 + (VH_mat + VLDA)

        vals, vects = eigen(Ham)
        vals_r = real.(vals)
        println("$iter vals ", vals_r[1:4])
        
        rho, filling = assemble_rho(vals_r, vects, nel, rall, wall, rho, N, Rmax)
        if iter > 1
            rho[:] = mix*rho[:] + (1.0-mix)*rho_old[:]
        end
        rho_old[:] = rho[:]
    end

    energy = calc_energy(rho, N, Rmax, rall, wall, VH, filling, vals_r, Vtot, Z)
    
    return vals_r, vects, rho,rall
    
end    

function calc_energy(rho, N, Rmax, rall, wall, VH, filling, vals_r, Vin, Z)

    ELDA = calc_energy_lda(rho, N, Rmax, rall, wall)
    EH = calc_energy_hartree(rho, N, Rmax, rall, wall, VH)
    KE = calc_energy_ke(rho, N, Rmax, rall, wall, filling, vals_r, Vin)
    ENUC = calc_energy_enuc(rho, N, Rmax, rall, wall,Z)

    println()
    println("ELDA   $ELDA")
    println("EH     $EH")
    println("KE     $KE")
    println("ENUC   $ENUC")

    ENERGY = ELDA + EH + KE + ENUC
    println()
    println("ENERGY $ENERGY ")
    println()
    
end

function calc_energy_enuc(rho, N, Rmax, rall, wall, Z)

    return -Z * 4 * pi * sum(rho .* wall .* rall)

end

function calc_energy_ke(rho, N, Rmax, rall, wall, filling, vals_r, Vin)

    KE = sum(filling .* vals_r)
    KE += -4.0 * pi * sum( Vin .* rho[2:end-1] .* rall[2:end-1].^2 .* wall[2:end-1])

    return KE
    
end

function calc_energy_lda(rho, N, Rmax, rall, wall)

    return 4 * pi * 0.5 * sum( e_LDA_sp.(rho, 1.0) .* rho .* wall .* rall.^2 )  #hartree

end

function calc_energy_hartree(rho, N, Rmax, rall, wall, VH)

    return 0.5 * 4 * pi * sum(VH .* rho .* wall .* rall.^2 )
    
end

function assemble_rho(vals_r, vects, nel, rall, wall, rho, N, Rmax)

    inds = sortperm(vals_r)
    nleft = nel
    rho[:] .= 0.0

#    println(size(rho))
#    println(size(vects))

#    t = zeros(N+1)
    vnorm = zeros(N+1)

    eps =  rall[2]/5.0

    filling = zeros(size(vals_r))
    
    for i in inds

        vnorm[:] = [0; real.(vects[:,i].*conj(vects[:,i])); 0]
        t_cheb = ChebyshevQuantum.Interp.make_cheb(vnorm, a=0.0, b = Rmax)
        vnorm[1] = t_cheb(eps)/(eps)^2
        vnorm[2:N] = vnorm[2:N] ./ rall[2:N].^2

        #limit
        
#        vnorm[:] = [t_cheb(1e-5)/(1e-5)^2;  real.(vects[:,i].*conj(vects[:,i])) ./ r.^2; 0.0]

        vnorm = vnorm / (4.0*pi*sum(vnorm .* wall .* rall.^2))

        if nleft <= 2.0 + 1e-10
            rho += vnorm * nleft
            filling[i] = nleft
            break
        else

            rho += vnorm * 2.0
            nleft -= 2.0
            filling[i] = 2.0
            
        end
    end    

    norm = 4.0*pi*sum(rho .* wall .* rall.^2)
#    println("check norm ", norm)
    rho = rho/ norm * nel
    return rho, filling
    
end

end #end module
