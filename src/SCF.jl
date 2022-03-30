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
#using QuadGK
#using Arpack

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
    r = (1 .+ pts) * (b-a) / 2 .+ a
    r_a = (1 .+ pts[2:N]) * (b-a) / 2 .+ a
    D_a = D[2:N, 2:N]
    D2_a = D2[2:N,2:N]
    D2X = D2

    w, pts2 = ChebyshevQuantum.Interp.getW(N)

    w_a = w[2:N]*(b - a)/2.0

    H_poisson = D2_a +  diagm( 2.0 ./ r_a )  * D_a
    
    return r_a, D_a, D2_a, w_a, r, w *(b - a)/2.0, H_poisson, D2X

    
end

function getVcoulomb(r_a, Z, l)
    
    return diagm( (-Z)./r_a + 0.5*(l)*(l+1)./r_a.^2)
    
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

function get_initial_rho(rho_init, N, Z, nel, r, w; nspin=1, checknorm=true)

    
    if !ismissing(rho_init)
        if typeof(rho_init) <: Array
            if length(rho_init[:,1]) != N+1
                println("make")
                C = ChebyshevQuantum.Interp.make_cheb(rho_init[:], a = 0.0, b = r[end]);
                rho = C.(r)
                if nspin == 2
                    rho = [ rho; rho]
                end
            else
                rho = copy(rho_init)
                if nspin == 2
                    rho = [rho; rho]
                end
            end
        else
            rho = rho_init.(r)
            if nspin == 2
                rho = [ rho; rho]
            end
        end

    else #ismissing case

        rho = zeros(N+1, nspin)

        # default filling order
        order = [[1,0],[2,0],[2,1],[3,0],[3,1],[4,0],[3,2],[4,1], [5,0],[4,2],[5,1],[6,0],[4,3],[5,2],[6,1],[7,0],[5,3],[6,2],[7,1]]
        for spin = 1:nspin

            nleft = sum(nel[:, spin])
            
            if nleft < 1e-20 #deal with completely empty spin channel
                nleft = 1e-20
            end
            
            for (n,l) in order
                
                if l == 0
                    maxfill = 2 / nspin
                elseif l == 1
                    maxfill = 6/ nspin
                elseif l == 2
                    maxfill = 10/ nspin
                elseif l == 3
                    maxfill = 14/ nspin
                end
                
                psi = h_rad(n, l;Z=Z)
                    if nleft <= maxfill/ nspin + 1e-15
                        #                    println("size rho $(size(rho)) r $(size(r)) psir $(size(psi.(r))) ")
                        rho[:,spin] += psi.(r).^2 * nleft
                        break
                    else
                        rho[:,spin] += psi.(r).^2 * maxfill/ nspin
                        nleft -= maxfill/ nspin
                    end
            end
        end
        rho = rho / (4*pi)
    end        
    
    if checknorm
        #check normalization
        for spin = 1:nspin
            norm = 4 * pi * sum(rho[:,spin] .* r.^2 .* w)
            println("spin=$spin INITIAL check norm $norm")
            rho[:,spin] = rho[:,spin] / norm * sum(nel[:,spin])
            println("spin=$spin POST-INITIAL check norm ",  4 * pi * sum(rho[:,spin] .* r.^2 .* w) )
            
        end
    end
    
#    if nspin == 1
#        rho_cheb = [ChebyshevQuantum.Interp.make_cheb(rho[:,1])]
#    else
#        rho_cheb = [ChebyshevQuantum.Interp.make_cheb(rho[:,1]);ChebyshevQuantum.Interp.make_cheb(rho[:,1])]
#    end
        
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
        VLDA = diagm(v_LDA.(rho[2:N]))
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

function calc_energy(rho, N, Rmax, rall, wall, VH, filling, vals_r, Vin, Z; nspin=1, lmax=0)

    rho_tot = sum(rho, dims=2)
    
    ELDA = calc_energy_lda(rho, N, Rmax, rall, wall, nspin=nspin)

#    ELDA_old = calc_energy_lda(rho_tot, N, Rmax, rall, wall, nspin=1)
    

    EH = calc_energy_hartree(rho_tot, N, Rmax, rall, wall, VH)
    KE = calc_energy_ke(rho, N, Rmax, rall, wall, filling, vals_r, Vin)
    ENUC = calc_energy_enuc(rho_tot, N, Rmax, rall, wall,Z)

    println()
    println("ELDA   $ELDA")
#    println("ELDA_old   $ELDA_old")
    println("EH     $EH")
    println("KE     $KE")
    println("ENUC   $ENUC")

    ENERGY = ELDA + EH + KE + ENUC
    println()
    println("ENERGY $ENERGY ")
    println()

    return ENERGY
    
end

function calc_energy2(rho, N, Rmax, rall, wall, VH, filling, vals_r, Vin, Z, rho2, rall2, wall2; nspin=1, lmax=0)

    rho_tot = sum(rho, dims=2)
    rho_tot2 = sum(rho2, dims=2)
    
    ELDA = calc_energy_lda(rho2, N, Rmax, rall2, wall2, nspin=nspin)

#    ELDA_old = calc_energy_lda(rho_tot, N, Rmax, rall, wall, nspin=1)
    

    EH = calc_energy_hartree(rho_tot2, N, Rmax, rall2, wall2, VH)
    
    KE = calc_energy_ke(rho2,2*N, Rmax, rall2, wall2, filling, vals_r, Vin)
    ENUC = calc_energy_enuc(rho_tot2, N, Rmax, rall2, wall2,Z)

    println()
    println("ELDA   $ELDA")
#    println("ELDA_old   $ELDA_old")
    println("EH     $EH")
    println("KE     $KE")
    println("ENUC   $ENUC")

    ENERGY = ELDA + EH + KE + ENUC
    println()
    println("ENERGY $ENERGY ")
    println()

    return ENERGY
    
end

function calc_energy_enuc(rho, N, Rmax, rall, wall, Z)

    return -Z * 4 * pi * sum(rho .* wall .* rall)

end

function calc_energy_ke(rho, N, Rmax, rall, wall, filling, vals_r, Vin)

    KE = sum(filling .* vals_r)
    KE += -4.0 * pi * sum( sum(Vin .* rho[2:end-1,:],dims=2) .* rall[2:end-1].^2 .* wall[2:end-1])

    return KE
    
end

function calc_energy_lda(rho, N, Rmax, rall, wall; nspin=1)

    if nspin == 1
#        println("old ", e_LDA.(rho[end]))
        return 4 * pi * 0.5 * sum( e_LDA.(rho) .* rho .* wall .* rall.^2 )  #hartree

        
    elseif nspin == 2


        rho_tot = rho[:,1] + rho[:,2]
        ζ = (rho[:,1] - rho[:,2])./rho_tot

        return 4 * pi * 0.5 * sum( e_LDA_sp.(rho_tot, ζ) .* rho_tot .* wall .* rall.^2 )  #hartree
    end        
end

function calc_energy_hartree(rho, N, Rmax, rall, wall, VH)

    return 0.5 * 4 * pi * sum(VH .* rho .* wall .* rall.^2 )
    
end

function assemble_rho(vals_r, vects, nel, rall, wall, rho, N, Rmax, D2; nspin=1, lmax=0)

    rho[:] .= 0.0

    #CV = ChebyshevQuantum.Interp.make_cheb([0; real.(vects[:,1,1,1]); 0], a= 0, b = Rmax);
    #nn = sum(CV*CV)*4*pi

    #nn = QuadGK.quadgk(x->CV(x)^2 , 0, Rmax)[1]*4*pi
    
    #t = (sign(real.(vects[5,1,1,1,1]))* CV.(rall[2:N])/sqrt(abs(nn)));
    #rho[2:N] = t.^2 ./ rall[2:N].^2
    
    #CV = ChebyshevQuantum.Interp.make_cheb([0; real.(vects[:,1,1,1]); 0], a= 0, b = Rmax);
    #    nn = QuadGK.quadgk(x->CV(x)^2 , 0, 60)[1]*4*pi;
    #nn = sum(CV*CV)*4*pi
    #t = (sign(real.(vects[5,1,1,1,1]))* CV.(rall)/sqrt(abs(nn))); #plot!(x,  (t ./ x).^2 .* x.^2)
    #rho[2:N] = t[2:N].^2 #./ rall[2:N].^2
    #filling = zeros(size(vals_r))
    #filling[1] = 1.0
    #return rho, filling
    
#    println(size(rho))
#    println(size(vects))

#    t = zeros(N+1)
    vnorm = zeros(N+1)

    eps =  rall[2]/5.0

    filling = zeros(size(vals_r))
    rhor2 = zeros(size(rho))
    
    for spin = 1:nspin
        for l = 0:lmax
            fillval = 2*(2*l+1)
            nleft = nel[l+1, spin]

            if nleft < 1e-20
                nleft = 1e-20
            end
            inds = sortperm(vals_r[:,spin,l+1])
            for i in inds
                vnorm[:] = [0; real.(vects[:,i,spin,l+1].*conj(vects[:,i,spin,l+1])); 0]
#                t_cheb = ChebyshevQuantum.Interp.make_cheb(vnorm, a=0.0, b = Rmax)
#                vnorm[1] = t_cheb(eps)/(eps)^2
                vnorm[2:N] = vnorm[2:N] #./ rall[2:N].^2

#                println("vects ", vects[1:3,i,spin,l+1])
                
                #limit
                
                #        vnorm[:] = [t_cheb(1e-5)/(1e-5)^2;  real.(vects[:,i].*conj(vects[:,i])) ./ r.^2; 0.0]
                
                vnorm = vnorm / (4.0*pi*sum(vnorm .* wall ))# .* rall.^2))

 #               println("vnorm ", vnorm[1:3])

                #                vects[:,i,spin,l+1] = sign.(real.(vects[:,i,spin,l+1])) .* sqrt.(abs.(real.(vnorm[2:N])))
                
                if nleft <= fillval/nspin + 1e-10
                    rho[2:N,spin] += vnorm[2:N]  * nleft  #./ rall[2:N].^2
#                    println("rho ", rho[1:3,spin])
                    filling[i,spin,l+1] = nleft
                    break
                else
                    
                    rho[2:N,spin] += vnorm[2:N] * fillval / nspin #./ rall[2:N].^2
                    nleft -= fillval / nspin
                    filling[i,spin,l+1] = fillval / nspin
                    
                end
            end
        end
        norm = 4.0*pi*sum(rho[:,spin] .* wall ) #.* rall.^2
#        println("check norm spin $spin ", norm)
        rhor2[:,spin] = rho[:,spin] /norm*sum(nel[:,spin])
        #        println("rall 1:3 ", rall[1:3])

        #trick for getting rho at r=0, since we only calculated r^2*rho.   (r^2*rho)'' at r=0 is equal to rho(r=0)*2
        d2r = (D2[:,:] * rho[:,spin])  #/ norm * sum(nel[:,spin])  
        rho[2:end,spin] = (rho[2:end,spin]./rall[2:end].^2  )  / norm * sum(nel[:,spin])
#        println("d2r ", d2r)
        rho[1,spin] = d2r[1]/2.0 / norm * sum(nel[:,spin])
#        println("check norm2 spin $spin ", 4.0*pi*sum(rho[:,spin] .* wall .* rall.^2 ) )
        
#        CR = ChebyshevQuantum.Interp.make_cheb(rho[:,spin], a= 0.0, b = 60.0)
#        CR2 = ChebyshevQuantum.Interp.make_cheb(x -> CR(x) / x^2, a= 1e-1, b = 60.0);

#        rho[2:N,spin] = CR2.(rall[2:N])
#        rho[N+1, spin] =0.0
#        rho[1, spin] = CR2(1e-1)
            
        
        #rho[1,spin] = d2r[1]
        
        #        rho[2:N,spin] = CR.(rall[2:N]) ./ rall[2:N].^2
        
    end
    return rho, filling, rhor2
    
end

function assemble_rho2(vals_r, vects, nel, rall, wall, rho, N, Rmax, NC, rallC, wallC; nspin=1, lmax=0)

    rho[:] .= 0.0

    rho_big = zeros(length(rallC), nspin)
    
#    println(size(rho))
#    println(size(vects))

#    t = zeros(N+1)
    vnorm = zeros(NC+1)

    eps =  rallC[2]/5.0

    filling = zeros(size(vals_r))

#    x_cheb2 = ChebyshevQuantum.Interp.make_cheb(x->x^2, a=0.0, b = Rmax)
    
    for spin = 1:nspin
        for l = 0:lmax
            fillval = 2*(2*l+1)
            nleft = nel[l+1, spin]

            if nleft < 1e-20
                nleft = 1e-20
            end
            inds = sortperm(vals_r[:,spin,l+1])
            for i in inds
                
                v_cheb = ChebyshevQuantum.Interp.make_cheb([0.0; vects[:,i,spin,l+1]; 0.0], a=0.0, b = Rmax)
#                vc = v_cheb.(rallC)
#                vnorm[:] = real.(vc .* conj.(vc))
                
                vc = v_cheb * conj(v_cheb) 
                vnorm = vc.(rallC)

                #                vnorm = vnorm
                t_cheb = ChebyshevQuantum.Interp.make_cheb(vnorm, a=0.0, b = Rmax)
                vnorm[1] = t_cheb(eps) /(eps)^2
                vnorm[2:NC] = vnorm[2:NC] ./ rallC[2:NC].^2
                
                #limit
                
                #        vnorm[:] = [t_cheb(1e-5)/(1e-5)^2;  real.(vects[:,i].*conj(vects[:,i])) ./ r.^2; 0.0]
                
                vnorm = vnorm / (4.0*pi*sum(vnorm .* wallC .* rallC.^2))
                
                if nleft <= fillval/nspin + 1e-10
                    rho_big[:,spin] += vnorm * nleft
                    filling[i,spin,l+1] = nleft
                    break
                else
                    
                    rho_big[:,spin] += vnorm * fillval / nspin
                    nleft -= fillval / nspin
                    filling[i,spin,l+1] = fillval / nspin
                    
                end
            end
        end
        norm = 4.0*pi*sum(rho_big[:,spin] .* wallC .* rallC.^2)
        println("check norm spin $spin ", norm)
        rho_big[:,spin] = rho_big[:,spin]/ norm * sum(nel[:,spin])
        
    end
    
    rho_ret = []
    for spin = 1:nspin
        rho[:] .= 0.0
        x_cheb = ChebyshevQuantum.Interp.make_cheb(rho_big[:,spin], a=0.0, b = Rmax)
        push!(rho_ret, x_cheb)
        rho[:, spin] .= x_cheb.(rall)
        norm = 4.0*pi*sum(rho[:,spin] .* wall .* rall.^2)
        rho[:,spin] = rho[:,spin]/norm
    end

    println("rho_ret1")
    println(rho_ret[1])
    
    return rho, filling, rho_ret
    
end


function DFT_spin(; nspin = 1,  N = 40, nmax = 1, lmax = 0, Z=1.0, Rmax = 10.0, nel = missing, rho_init = missing, niters=20, mix = 0.7)

    Z = Float64(Z)
    if ismissing(nel)
        if nspin == 2
            nel = [Float64(Z)/2.0, Float64(Z)/2.0]
        else
            nel = [Float(Z)]
        end
    end
    if typeof(nel) <: Number
        nel = [nel]
    end

    begin
        r, D, D2, w, rall,wall,H_poisson = setup_mats(;N = N, Rmax=Rmax)
        
        rho = get_initial_rho(rho_init, N, Z, nel,rall,wall, nspin=nspin)
        #        println("rho_ini ", size(rho))

#        return rho
        
        rho_old = deepcopy(rho)
        
        Vc = getVcoulomb(r, Z)
        KE = getKE(r)
        
        H0 = -0.5*D2 + Vc
        
        Ham = zeros(size(H0))
        
        vects = zeros(Complex{Float64}, size(H0)[1], size(H0)[2], nspin)
        vals_r = zeros(Float64, size(H0)[1], nspin)    

        VH = zeros(size(rho)[1],1)

        VLDA = zeros(size(rho))
        
        filling = zeros(size(rho))
        Vtot = zeros(size(rho)[1]-2, nspin)
        
    end


    
    for iter = 1:niters
        println()
        #        VLDA = diagm(v_LDA.(rho[2:N]))
#        println(size(rho))
        if nspin == 1
            VLDA[:,1] = v_LDA.(rho[:, 1])
        elseif nspin == 2
            tt =  v_LDA_sp.(rho[:, 1], rho[:, 2])
            VLDA[:,:] = reshape(vcat(tt...), nspin,N+1)'
        end
        rho_tot = sum(rho, dims=2)
        VH[:] = V_H3( rho_tot[2:N], r,w,H_poisson,Rmax, rall, sum(nel))
        VH_mat = diagm(VH[2:N])

        for spin = 1:nspin
            VLDA_mat = diagm(VLDA[2:N,spin])
            Vtot[:,spin] = VH[2:N] + VLDA[2:N,spin] + diag(Vc)
                 
            Ham[:,:] = H0 + (VH_mat + VLDA_mat)

#            println("VLDA_mat ", VLDA_mat)
#            println()
#            println("VH_mat ", VH_mat)
#            println()
            
            vals, v = eigen(Ham)
            vals_r[:,spin] = real.(vals)
            vects[:,:,spin] = v
            
            println("$iter vals $spin ", vals_r[1:3, spin])
        end
        
        rho, filling = assemble_rho(vals_r, vects, nel, rall, wall, rho, N, Rmax, nspin=nspin)

#        println("filling ", filling)

        
        if iter > 1
            rho[:] = mix*rho[:] + (1.0-mix)*rho_old[:]
        end
        rho_old[:] = rho[:]
    end

    energy = calc_energy(rho, N, Rmax, rall, wall, VH, filling, vals_r, Vtot, Z, nspin=nspin)
    
    return vals_r, vects, rho,rall
    
end    


function assemble_rho3(vals_r, vects, nel, rall, wall, rho, N, Rmax, D2, rall2, wall2; nspin=1, lmax=0)

    rho[:] .= 0.0
    #    vnorm = zeros(N+1)

    eps =  rall[2]/5.0

    filling = zeros(size(vals_r))

    rho2 = zeros(2*N+1, nspin)

    vnorm = zeros(N+1)

    for spin = 1:nspin
        for l = 0:lmax
            fillval = 2*(2*l+1)
            nleft = nel[l+1, spin]

            if nleft < 1e-20
                nleft = 1e-20
            end
            inds = sortperm(vals_r[:,spin,l+1])
            for i in inds
                vnorm[:] = [0; real.(vects[:,i,spin,l+1].*conj(vects[:,i,spin,l+1])); 0]

                v_cheb = ChebyshevQuantum.Interp.make_cheb([0; real.(vects[:,i,spin,l+1]);0] , a=0.0, b = Rmax)

                vnorm2 = v_cheb.(rall2).^2
##                
#                vnorm[2:N] = vnorm[2:N].^2
                vnorm = vnorm / (4.0*pi*sum(vnorm .* wall ))# .* rall.^2))
                vnorm2 = vnorm2 / (4.0*pi*sum(vnorm2 .* wall2 ))# .* rall.^2))
                
                if nleft <= fillval/nspin + 1e-10
                    rho2[2:(2*N),spin] += vnorm2[2:(2*N)]  * nleft  #./ rall[2:N].^2
                    rho[2:(N),spin] += vnorm[2:(N)]  * nleft  #./ rall[2:N].^2
                    println("rho ", rho[1:3,spin])
                    filling[i,spin,l+1] = nleft
                    break
                else
                    
                    rho2[2:(2*N),spin] += vnorm2[2:(2*N)] * fillval / nspin #./ rall[2:N].^2
                    rho[2:(N),spin] += vnorm[2:(N)] * fillval / nspin #./ rall[2:N].^2

                    nleft -= fillval / nspin
                    filling[i,spin,l+1] = fillval / nspin
                    
                end
            end
        end
        norm = 4.0*pi*sum(rho2[:,spin] .* wall2 ) #.* rall.^2
        #trick for getting rho at r=0, since we only calculated r^2*rho.   (r^2*rho)'' at r=0 is equal to rho(r=0)*2
        d2r = (D2[:,:] * rho2[:,spin])  #/ norm * sum(nel[:,spin])  
        rho2[2:end,spin] = (rho2[2:end,spin]./rall2[2:end].^2  )  / norm * sum(nel[:,spin])
        rho2[1,spin] = d2r[1]/2.0 / norm * sum(nel[:,spin])

        norm = 4.0*pi*sum(rho[:,spin] .* wall )
        rho[2:end,spin] = (rho[2:end,spin]./rall[2:end].^2  )  / norm * sum(nel[:,spin])
        rho[1,spin] = d2r[1]/2.0 / norm * sum(nel[:,spin])
        
    end

    return rho, filling, rho2
    
end


function DFT_spin_l(nel; N = 40, nmax = 1, Z=1.0, Rmax = 10.0, rho_init = missing, niters=20, mix = 0.7)

    Z = Float64(Z)
    nel = Float64.(nel)
    
    if length(size(nel)) == 1
        nspin = 1
    else
        nspin = size(nel)[2]
    end
    
    lmax = size(nel)[1] - 1
    println("detected nspin $nspin lmax $lmax Nel=", sum(nel))
    println()

    println("setup")
    @time begin
        r, D, D2, w, rall,wall,H_poisson, D2X = setup_mats(;N = N, Rmax=Rmax)
        
        rho = get_initial_rho(rho_init, N, Z, nel,rall,wall, nspin=nspin)
        #        println("rho_ini ", size(rho))

#        return rho
        
        rho_old = deepcopy(rho)
        rhor2 = zeros(size(rho))

        KE = getKE(r)

        H0_L = []

        for l = 0:lmax
            Vc = getVcoulomb(r, Z, l)
            H0 = -0.5*D2 + Vc
            #H0 = -0.5 * diagm(1.0 ./ r.^2)*D*diagm(r).^2*D + Vc
            push!(H0_L, H0)
        end
        Vc = getVcoulomb(r, Z, 0)

        
        Ham = zeros(N-1, N-1)
        
        vects = zeros(Complex{Float64}, N-1, N-1, nspin, lmax+1)
        vals_r = zeros(Float64, N-1, nspin, lmax+1)    

        VH = zeros(size(rho)[1],1)

        VLDA = zeros(size(rho))
        
        filling = zeros(size(vals_r))
        Vtot = zeros(size(rho)[1]-2, nspin, lmax+1)
        
    end

    eigs_tot_old = sum(filling.*vals_r)
    
    for iter = 1:niters

        println()
        #        VLDA = diagm(v_LDA.(rho[2:N]))
#        println(size(rho))
        println("LDA")
        @time if nspin == 1
            VLDA[:,1] = v_LDA.(rho[:, 1])
        elseif nspin == 2
            tt =  v_LDA_sp.(rho[:, 1], rho[:, 2])
            VLDA[:,:] = reshape(vcat(tt...), nspin,N+1)'
        end
        println("poisson")
        @time begin
            rho_tot = sum(rho, dims=2)
            VH[:] = V_H3( rho_tot[2:N], r,w,H_poisson,Rmax, rall, sum(nel))
            VH_mat = diagm(VH[2:N])
        end
        
        println("eig")
        @time for spin = 1:nspin
            VLDA_mat = diagm(VLDA[2:N,spin])
            Vtot[:,spin, lmax+1] = VH[2:N] + VLDA[2:N,spin] + diag(Vc)
            for l = 0:lmax

                Ham[:,:] = H0_L[l+1] + (VH_mat + VLDA_mat)
                vals, v = eigen(Ham)
                #vals, v = eigs(Ham, which=:SM, nev=4)
                vals_r[:,spin,l+1] = real.(vals)
                vects[:,:,spin,l+1] = v
            
                println("$iter vals $spin $l ", vals_r[1:3, spin, l+1])
            end
        end

        println("ass")
        @time rho, filling, rhor2 = assemble_rho(vals_r, vects, nel, rall, wall, rho, N, Rmax, D2X, nspin=nspin, lmax=lmax)

#        println("filling ", filling)

        eigs_tot_new = sum(filling.*vals_r)
        if iter > 1 && abs(eigs_tot_old-eigs_tot_new) < 1e-8
            println("convergence reached")
            println()
            break
        end
        eigs_tot_old = eigs_tot_new

        
        if iter > 1
            rho[:] = mix*rho[:] + (1.0-mix)*rho_old[:]
        end
        rho_old[:] = rho[:]
    end

    println("energy")
    @time energy = calc_energy(rho, N, Rmax, rall, wall, VH, filling, vals_r, Vtot, Z, nspin=nspin, lmax=lmax)
    
    return energy, vals_r, vects, rho,rall, wall, rhor2
    
end    

function DFT_spin_l_charge(nel; N = 40, nmax = 1, Z=1.0, Rmax = 10.0, rho_init = missing, niters=20, mix = 0.7)

    Z = Float64(Z)
    nel = Float64.(nel)
    
    if length(size(nel)) == 1
        nspin = 1
    else
        nspin = size(nel)[2]
    end
    
    lmax = size(nel)[1] - 1
    println("detected nspin $nspin lmax $lmax Nel=", sum(nel))
    println()

    begin
        r, D, D2, w, rall,wall,H_poisson = setup_mats(;N = N, Rmax=Rmax)

        NC = 2*N
        rC, DC, D2C, wC, rallC,wallC,H_poissonC = setup_mats(;N = NC, Rmax=Rmax)
        
        rho = get_initial_rho(rho_init, N, Z, nel,rall,wall, nspin=nspin)
        #        println("rho_ini ", size(rho))

#        return rho
        
        rho_old = deepcopy(rho)
        

        KE = getKE(r)

        H0_L = []

        for l = 0:lmax
            Vc = getVcoulomb(r, Z, l)
            H0 = -0.5*D2 + Vc
            push!(H0_L, H0)
        end
        Vc = getVcoulomb(r, Z, 0)

        
        Ham = zeros(N-1, N-1)
        
        vects = zeros(Complex{Float64}, N-1, N-1, nspin, lmax+1)
        vals_r = zeros(Float64, N-1, nspin, lmax+1)    

        VH = zeros(size(rho)[1],1)

        VLDA = zeros(size(rho))
        
        filling = zeros(size(rho))
        Vtot = zeros(size(rho)[1]-2, nspin, lmax+1)
        
    end

    rho_big = zeros(NC)
    
    for iter = 1:niters
        println()

        println("LDA")
        @time if nspin == 1
            VLDA[:,1] = v_LDA.(rho[:, 1])
        elseif nspin == 2
            tt =  v_LDA_sp.(rho[:, 1], rho[:, 2])
            VLDA[:,:] = reshape(vcat(tt...), nspin,N+1)'
        end

        println("poisson")
        @time begin
            rho_tot = sum(rho, dims=2)
            VH[:] = V_H3( rho_tot[2:N], r,w,H_poisson,Rmax, rall, sum(nel))
            VH_mat = diagm(VH[2:N])
        end
        
        println("eig")
        @time for spin = 1:nspin
            VLDA_mat = diagm(VLDA[2:N,spin])
            Vtot[:,spin, lmax+1] = VH[2:N] + VLDA[2:N,spin] + diag(Vc)
            for l = 0:lmax

                Ham[:,:] = H0_L[l+1] + (VH_mat + VLDA_mat)
                vals, v = eigen(Ham)
                vals_r[:,spin,l+1] = real.(vals)
                vects[:,:,spin,l+1] = v
            
                println("$iter vals $spin $l ", vals_r[1:3, spin, l+1])
            end
        end

        println("assemble")
        @time rho, filling = assemble_rho(vals_r, vects, nel, rall, wall, rho, N, Rmax, nspin=nspin, lmax=lmax)

#        println("maximum diff ", maximum(abs.(rho - rho1)))
        
#        println("filling ", filling)

        
        if iter > 1
            rho[:] = mix*rho[:] + (1.0-mix)*rho_old[:]
        end
        rho_old[:] = rho[:]
    end

    energy = calc_energy(rho, N, Rmax, rall, wall, VH, filling, vals_r, Vtot, Z, nspin=nspin, lmax=lmax)

    rhoX, filling, rho_ret = assemble_rho2(vals_r, vects, nel, rall, wall, rho, N, Rmax, NC, rallC, wallC, nspin=nspin, lmax=lmax)
    
    
    return energy, vals_r, vects, rho, rho_ret,rall, wall
    
end    

function DFT_spin_l2(nel; N = 40, nmax = 1, Z=1.0, Rmax = 10.0, rho_init = missing, niters=20, mix = 0.7)

    Z = Float64(Z)
    nel = Float64.(nel)
    
    if length(size(nel)) == 1
        nspin = 1
    else
        nspin = size(nel)[2]
    end
    
    lmax = size(nel)[1] - 1
    println("detected nspin $nspin lmax $lmax Nel=", sum(nel))
    println()

    begin
        r, D, D2, w, rall,wall,H_poisson, D2X = setup_mats(;N = N, Rmax=Rmax)

        rA, DA, D2A, wA, rallA,wallA,H_poissonA, D2XA = setup_mats(;N = 2*N, Rmax=Rmax)
        
        rho = get_initial_rho(rho_init, N, Z, nel,rall,wall, nspin=nspin)
        rho2 = get_initial_rho(rho_init, N*2, Z, nel,rallA,wallA, nspin=nspin)

        #        println("rho_ini ", size(rho))

#        return rho
        
        rho_old = deepcopy(rho)
        rho_old2 = deepcopy(rho2)
        

        KE = getKE(r)

        H0_L = []

        for l = 0:lmax
            Vc = getVcoulomb(r, Z, l)
            H0 = -0.5*D2 + Vc
            #H0 = -0.5 * diagm(1.0 ./ r.^2)*D*diagm(r).^2*D + Vc
            push!(H0_L, H0)
        end
        Vc = getVcoulomb(r, Z, 0)

        
        Ham = zeros(N-1, N-1)
        
        vects = zeros(Complex{Float64}, N-1, N-1, nspin, lmax+1)
        vals_r = zeros(Float64, N-1, nspin, lmax+1)    

        VH = zeros(size(rho)[1],1)
        VH2 = zeros(2*N+1,1)

        VLDA = zeros(size(rho))
        
        filling = zeros(size(rho))
        #        Vtot = zeros(size(rho)[1]-2, nspin, lmax+1)
        Vtot = zeros(2*N-1, nspin, lmax+1)
        
    end


    
    for iter = 1:niters
        println()
        #        VLDA = diagm(v_LDA.(rho[2:N]))
#        println(size(rho))

        if nspin == 1
            VLDA[:,1] = v_LDA.(rho[:, 1])
        elseif nspin == 2
            tt =  v_LDA_sp.(rho[:, 1], rho[:, 2])
            VLDA[:,:] = reshape(vcat(tt...), nspin,N+1)'
        end

        if true
            rho_tot = sum(rho2, dims=2)
            VH2[:] = V_H3( rho_tot[2:2*N], rA,wA,H_poissonA,Rmax, rallA, sum(nel))
            println("size VH2 ", size(VH2))
            vh_cheb = ChebyshevQuantum.Interp.make_cheb(VH2[:], a=0.0, b= Rmax)
            VH[:] = vh_cheb.(rall)
            VH_mat = diagm(VH[2:N])
        else
            rho_tot = sum(rho, dims=2)
            VH[:] = V_H3( rho_tot[2:N], r,w,H_poisson,Rmax, rall, sum(nel))
            VH_mat = diagm(VH[2:N])
        end
        
        for spin = 1:nspin
            VLDA_mat = diagm(VLDA[2:N,spin])
            Vtot[:,spin, lmax+1] = VH2[2:2*N] + v_LDA.(rho2[2:2*N, 1]) + (-Z ./ rallA[2:2*N])

            for l = 0:lmax

                Ham[:,:] = H0_L[l+1] + (VH_mat + VLDA_mat)
                vals, v = eigen(Ham)
                vals_r[:,spin,l+1] = real.(vals)
                vects[:,:,spin,l+1] = v
            
                println("$iter vals $spin $l ", vals_r[1:3, spin, l+1])
            end
        end
        
        rho, filling, rho2 = assemble_rho3(vals_r, vects, nel, rall, wall, rho, N, Rmax, D2XA,rallA,wallA, nspin=nspin, lmax=lmax)

#        println("filling ", filling)

        
        if iter > 1
            rho[:] = mix*rho[:] + (1.0-mix)*rho_old[:]
            rho2[:] = mix*rho2[:] + (1.0-mix)*rho_old2[:]
        end
        rho_old[:] = rho[:]
        rho_old2[:] = rho2[:]
    end

    println("new sssssssssssssssssssssssssssss")
    energyNEW = calc_energy2(rho, N, Rmax, rall, wall, VH2, filling, vals_r, Vtot, Z, rho2, rallA, wallA,  nspin=nspin, lmax=lmax)
#    println("old sssssssssssssssssssssssssssss")
#    energy = calc_energy(rho, N, Rmax, rall, wall, VH, filling, vals_r, Vtot, Z,  nspin=nspin, lmax=lmax)
    
    return energyNEW, vals_r, vects, rho,rall, wall, rho2, rallA, energyNEW
    
end    


end #end module
