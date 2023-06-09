

module GalerkinDFT

using Polynomials
using SpecialPolynomials
using Base.Threads
using Printf
using LinearAlgebra
using ForwardDiff
using QuadGK
using FastSphericalHarmonics
using ..Galerkin:get_gal_rep_matrix
using ..Galerkin:get_gal_rep
using ..Galerkin:get_gal_rep_matrix_R
using ..Galerkin:get_rho_gal
using ..Galerkin:gal_rep_to_rspace
using ..Galerkin:get_vh_mat
using ..Galerkin:do_1d_integral
using ..Galerkin:get_r_grid

using ..Galerkin:MAT_M2N
using ..Galerkin:MAT_N2M
    
using Plots
using LoopVectorization

include("Atomlist.jl")

using ..UseLibxc:set_functional
#using ..UseLibxc:EXC
#using ..UseLibxc:EXC_sp
#using ..UseLibxc:smooth
#using ..UseLibxc:getVsigma

using Libxc


using ..LDA:v_LDA
using ..LDA:e_LDA

using ..LDA:v_LDA_sp
using ..LDA:e_LDA_sp



using ..AngularIntegration:real_gaunt_dict
using ..AngularIntegration:makeleb
using JLD

function choose_exc(exc, nspin)

    

    if ismissing(exc) || (typeof(exc) == String && lowercase(exc) == "lda_internal") || (typeof(exc) == Symbol && exc == :lda_internal)

        
        println("Running default LDA (VWN)")
        funlist = :lda_internal
        gga = false

        
    else

        if (typeof(exc) == String && lowercase(exc) == "pbe") || (typeof(exc) == Symbol && exc == :pbe)
            exc = [:gga_x_pbe, :gga_c_pbe]
        elseif (typeof(exc) == String && lowercase(exc) == "pbesol") || (typeof(exc) == Symbol && exc == :pbesol)
            exc = [:gga_x_pbe_sol, :gga_c_pbe_sol]
        elseif (typeof(exc) == String && lowercase(exc) == "lda") || (typeof(exc) == Symbol && exc == :lda)
            exc = [:lda_x, :lda_c_vwn]
        elseif (typeof(exc) == String && lowercase(exc) == "vwn") || (typeof(exc) == Symbol && exc == :vwn)
            exc = [:lda_x, :lda_c_vwn]
        end    

        if (typeof(exc) == String && lowercase(exc) == "hydrogen") || (typeof(exc) == Symbol && exc == :hydrogen) || (typeof(exc) == String && lowercase(exc) == "H") || (typeof(exc) == Symbol && exc == :H)
            funlist = :hydrogen
            println("Running hydrogen-like atom (no hartree, no exc)")
            gga = false
            return funlist, gga

        end

        if (typeof(exc) == String && lowercase(exc) == "none") || (typeof(exc) == Symbol && exc == :none) 
            funlist = :none
            println("Exc is none")
            gga = false
            return funlist, gga
        end
        
        funlist,gga = set_functional(exc, nspin=nspin)
        
    end

    return funlist, gga

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

    if Z < 1; Z=1.0; end
    
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



function prepare(Z, fill_str, lmax, exc, N, M, g, lmaxrho)

    if typeof(Z) == String || typeof(Z) == Symbol
        Z = atoms[String(Z)]
    end

    a = g.a
    b = g.b
    
    Z = Float64(Z)

    for k in keys(atoms)
        if abs(Z - atoms[k]) <=  0.5
            println("Atom name $k")
            if ismissing(fill_str)
                fill_str = k
            end
        end
    end

    println("setup filling")
    @time nel, nspin, lmax, filling = setup_filling(fill_str, N, lmax_init=lmax)

    fs = sum(filling, dims=[2,3,4])
    nmax = findfirst(fs[:] .> 1e-12)
    println("nmax $nmax")

    
    println("choose exc")
    @time funlist, gga = choose_exc(exc, nspin)



    function V_c(r)
        return -Z / r 
    end

    V_C = get_gal_rep_matrix(V_c, g, ; N = N, M = M)



    function V_ang(r)
        return 0.5*1.0/r^2
    end
    #    println("get gal rep matrix ")

    V_L = get_gal_rep_matrix(V_ang, g, ; N = N, M = M)

    #    V_l = []
    #    @time for ll = 0:(lmax+1)
    #        l = ll
    #        vl = get_gal_rep_matrix(V_ang, g, ; N = N, M = M)
    #        push!(V_l, vl)
    #    end

    VECTS = zeros(Complex{Float64}, N-1,N-1, nspin, lmax+1,2*lmax+1 )
    VALS = zeros(N-1,nspin, lmax+1,2*lmax+1 )


    D2 =-0.5* g.d2[1:N-1, 1:N-1] /  (b-a)^2*2^2
    D1 =  g.d1[1:N-1, 1:N-1] /  (b-a)*2

    S = g.s[1:N-1, 1:N-1]

    invsqrtS = inv(sqrt(S))
    invS = inv(S)
    
   #    println("size VECTS ", size(VECTS))

    #    println("initial eig")
    for ll = 0:lmax
        vals, vects = eigen(D2 + V_C + ll*(ll+1)*V_L, S)
        #vals, vects = eigen( invsqrtS*(D2 + VC[ll+1])*invsqrtS)
        
        #        println("vals $ll $(vals[1:3])")
        for spin = 1:nspin
            for m = 1:(2*(ll)+1)
                #                println("spin $spin ll $ll m $m , ", size(vects))
                VECTS[:,:,spin, ll+1, m] = vects
                VALS[:,spin, ll+1, m] = vals
            end
        end
    end

    
    LEB = makeleb(lmaxrho*2+1 , lmax=lmaxrho)

    R = get_r_grid(g, M=M)
    R = R[2:M+2]

    gbvals2 = zeros(M+1, N-1, N-1)
    println("pretrain")
    @time @threads for n1 = 1:(N-1)
        for n2 = 1:(N-1)
            gbvals2[:, n1, n2] =  (@view g.bvals[2:M+2,n1,M]).*(@view g.bvals[2:M+2,n2,M]) 
        end
    end

    
    
    return Z, nel, filling, nspin, lmax, V_C, V_L,D1, D2, S, invsqrtS, invS, VECTS, VALS, funlist, gga, LEB, R, gbvals2, nmax
    
end


function setup_filling(fill_str, N; lmax_init=0)

    T = []
    nspin = 1

    if ismissing(lmax_init)
        lmax = 0
    else
        lmax = lmax_init
    end
    
    
    for line in split(replace(fill_str, ";" => "\n"), "\n")
        sp = split(line)

        if length(sp) == 0
            continue
        end
        
        #starting
        if length(sp) == 1
            tsp = String.(sp[1])
            if tsp in keys(atoms)
                num_el = atoms[tsp]

                for toadd in [ [1,0, 2.0],
                               [2,0, 2.0],
                               [2,1, 6.0],
                               [3,0, 2.0],
                               [3,1, 6.0],
                               [4,0, 2.0],
                               [3,2, 10.0],
                               [4,1, 6.0],
                               [5,0, 2.0],
                               [4,2, 10.0],
                               [5,1, 6.0],
                               [6,0, 2.0],
                               [4,3, 14.0],
                               [5,2, 10.0],
                               [6,1, 6.0],
                               [7,0, 2.0],
                               [5,3, 14.0],
                               [4,2, 10.0],
                               [7,1, 6.0]]
                    
                    
                    if toadd[3]+1e-15 < (num_el)
                        l = toadd[2]
                        n = toadd[3] / (2*l+1)
                        for m = -l:l
                            push!(T,[toadd[1],l, m, n])
                        end
                        num_el -= toadd[3]
                        lmax = max(lmax, Int64(l))
                    else
                        l = toadd[2]
                        n = num_el / (2*l+1)
                        for m = -l:l
                            push!(T,[toadd[1],l, m, n])
                        end
                        num_el -= toadd[3]
                        lmax = max(lmax, Int64(l))
                        
                        
                        #                        push!(T,[toadd[1],toadd[2], toadd[3], num_el])
                        #                        lmax = max(lmax, Int64(toadd[2]))
                        num_el = 0.0
                        break
                    end
                end
            else
                throw(ArgumentError(sp, "setup_filling error"))
            end
            continue
        end

        n = parse(Int64, sp[1])
        if sp[2] == "s" || sp[2] == "0"
            l = 0
        elseif sp[2] == "p" || sp[2] == "1"
            l = 1
        elseif sp[2] == "d" || sp[2] == "2"
            l = 2
        elseif sp[2] == "f" || sp[2] == "3"
            l = 3
        elseif sp[2] == "g" || sp[2] == "4"
            l = 4
        else
            println("err setup_filling $sp")
            throw(ArgumentError(sp, "setup_filling error"))
        end
        lmax = max(lmax, l)
        
        m=0
        for x in -l:l
            if string(x) == String.(sp[3])
                m = x
                break
            end
        end
        
        if length(sp) == 5
            nspin = 2
            push!(T, [n,l,m,parse(Float64, sp[4]), parse(Float64, sp[5])])
        else
            push!(T, [n,l,m, parse(Float64, sp[4])])
        end
    end        
    
    println("lmax $lmax")
    nel = zeros(nspin, lmax+1, (2*lmax+1))
    #    println("Filling")
    for t in T
        #        println(t)
        if length(t) == 4
            n,l,m,num_el = t
            n = Int64(n)
            l = Int64(l)
            m = Int64(m)
            d = [l+1, l+m+1]
            #d = FastSphericalHarmonics.sph_mode(l,m)

            if nspin == 1
                nel[1,d[1],d[2]] += num_el
                #                println("add nel ", [1,d[1],d[2]], num_el)
            else
                nel[1,d[1],d[2]] += num_el/2.0
                nel[2,d[1],d[2]] += num_el/2.0
            end
        elseif length(t) == 5
            n,l,m,num_el_up, num_el_dn = t
            n = Int64(n)
            l = Int64(l)
            m = Int64(m)
            d = [l+1, l+m+1]
            nel[1,d[1],d[2]] += num_el_up
            nel[2,d[1],d[2]] += num_el_dn

        end
    end


    #####
    #    println("nel")
    #    println(nel)
    filling = zeros(N-1, nspin, lmax+1, 2*lmax+1)

    println("FILLING----")
    if nspin == 1
        println(" n  l  m   fill")
    else
        println(" n  l  m   fill_up  fill_dn")
    end
    nmax = zeros(Int64, lmax+1)

    for spin = 1:nspin
        for l = 0:lmax
            for m = (-l):l
                
                #                println("L M $l $m")
                
                fillval = 2.0 / nspin
                nleft = nel[spin, l+1, m+l+1]
                #                println("nleft $l $m $spin $nleft")
                if nleft < 1e-20
                    nleft = 0.0
                end
                if nspin == 1
                    s = 0
                else
                    if spin == 1
                        s = -1
                    elseif spin == 2
                        s = 1
                    end
                end
                
                for n = 1:N-1
                    if nleft <= fillval + 1e-10
                        filling[n,spin,l+1, m+l+1] = nleft
                        
                        #println("$n  $l  $m  $spin ", nleft)
                        #                        @printf "%2i %2i %2i %2i  %4f \n" n l m s nleft
                        if n > nmax[l+1]
                            nmax[l+1] = n
                        end
                        break
                    else
                        nleft -= fillval 
                        filling[n,spin,l+1, m+l+1] = fillval 
                        #                        println("$n  $l  $m  $spin ", fillval )
                        #                        @printf "%2i %2i %2i %2i  %4f \n" n l m s nleft
                        if n > nmax[l+1]
                            nmax[l+1] = n
                        end
                        
                    end
                end
                
                
            end
        end
    end

    #printing
    for l = 0:lmax
        for m = (-l):l
            for n = 1:nmax[l+1]
                #                println("$l $m $n")
                if nspin == 1
                    f = filling[n,1,l+1, m+l+1]
                    @printf "%2i %2i %2i  %4f \n" (n+l) l m f
                elseif nspin == 2
                    f1 = filling[n,1,l+1, m+l+1]
                    f2 = filling[n,2,l+1, m+l+1]
                    @printf "%2i %2i %2i  %4f   %4f \n" (n+l) l m  f1 f2
                end
            end
        end
    end
    println("---")
    
    return nel, nspin, lmax, filling
    
end

function get_rho(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, gga, exx, nmax)
    #    println("get rho")
    #rho = zeros(N-1, nspin, lmax+1, 2*lmax+1)
#    rho_rs_M_R2 = zeros(M+1, nspin)
    S = inv(invS)
    rho_rs_M_R2_LM2 = zeros(M+1, nspin, lmax+1, 2*lmax+1)
    drho_rs_M_R2_LM2 = zeros(M+1, nspin, lmax+1, 2*lmax+1)

    mat_n2m = MAT_N2M(g, N=N, M=M)
    mat_m2n = MAT_M2N(g, N=N, M=M)
    
    for spin = 1:nspin
        for l = 0:lmax
            for m = -l:l
                for n = 1:N-1
                    fillval = filling[n, spin, l+1, l+m+1]
                    #                    println("add fillval $spin $l $m ", fillval)
                    if fillval < 1e-20
                        break
                    end
#                    println("VECTS ", size(VECTS[:, n, spin, l+1,l+1+m]))
#                    t = gal_rep_to_rspace(VECTS[:, n, spin, l+1,l+1+m], g, M=M)
#                    println("size ", size(VECTS))
 #                   println("test ", t - mat_n2m * VECTS[:, n, spin, l+1,l+1+m])

                    t = mat_n2m * VECTS[:, n, spin, l+1,l+1+m]
                    
#                    println("ttttt ", t[1:3], " ts ", real((S*VECTS[:, n, spin, l+1,l+1+m]))[1:3])
#                    t2 = real(t.*conj(t))
#                    rho_rs_M_R2[:,spin] += t2 * fillval
                    rho_rs_M_R2_LM2[:,spin, l+1, m+1+l] += real(t.*conj(t)) * fillval

                    if gga
                    #    vt = VECTS[:, n, spin, l+1,l+1+m]
                        dt = gal_rep_to_rspace(VECTS[:, n, spin, l+1,l+1+m] , g, M=M, deriv=1)
                        drho_rs_M_R2_LM2[:,spin, l+1, m+1+l] += real( t .* conj.(dt) + dt .* conj.(t)  ) * fillval
                    end
                    
                end
            end
        end
    end

    psipsi = zeros(M+1, nspin, nmax, lmax+1, 2*lmax+1, nmax, lmax+1, 2*lmax+1)
    if exx > 1e-12
        for spin = 1:nspin
            for l1 = 0:lmax
                for m1 = -l1:l1
                    for n1 = 1:nmax
                        fillval1 = filling[n1, spin, l1+1, l1+m1+1]
                        #                    println("add fillval $spin $l $m ", fillval)
                        if fillval1 < 1e-20
                            break
                        end
                        t1 = gal_rep_to_rspace(VECTS[:, n1, spin, l1+1,l1+1+m1], g, M=M)
                        
                        for l2 = 0:lmax
                            for m2 = -l2:l2
                                for n2 = 1:nmax
                                    fillval2 = filling[n2, spin, l2+1, l2+m2+1]
                                    
                                    if fillval2 < 1e-20
                                        break
                                    end
                                    t2 = gal_rep_to_rspace(VECTS[:, n2, spin, l2+1,l2+1+m2], g, M=M)
                                    psipsi[:,spin, n1,l1+1, m1+1+l1, n2, l2+1, m2+l2+1] += real(t1.*conj(t1)) * sqrt(fillval1)*sqrt(fillval2)
                                    
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    println("pp ", psipsi[1], " " , rho_rs_M_R2_LM2[1], " " ,  psipsi[1]/rho_rs_M_R2_LM2[1])
    
    
#=    for spin = 1:nspin
        for l = 0:lmax
            for m = -l:l
                grep = get_gal_rep(rho_rs_M_R2_LM2[:,spin, l+1, m+1+l], g, N=N)
                dt = gal_rep_to_rspace(grep  , g, M=M, deriv=1)
                drho_rs_M_R2_LM2[:,spin, l+1, m+1+l] += dt
            end
        end
    end
=#
    

    
    #    println("z ", rho_rs_M_R2[1,1], " " , rho_rs_M_R2_LM2[1,1, 1, 1] )
    
    rho_gal_R2 = zeros(N-1, nspin)
    rho_gal_dR = zeros(N-1, nspin)
    rho_gal_dR2 = zeros(N-1, nspin)
    rho_rs_M = zeros(M+1, nspin)

    lmaxrho = min(lmaxrho, lmax * 2)
    rho_rs_M_R2_LM = zeros(M+1, nspin, lmaxrho+1, 2*lmaxrho+1)
    drho_rs_M_R2_LM = zeros(M+1, nspin, lmaxrho+1, 2*lmaxrho+1)    

    rho_rs_M_LM = zeros(M+1, nspin, lmaxrho+1, lmaxrho*2+1)
    drho_rs_M_LM = zeros(M+1, nspin, lmaxrho+1, lmaxrho*2+1)

    rho_gal_R2_LM = zeros(N-1, nspin, lmaxrho+1, lmaxrho*2+1)
    
    rho_gal_dR_LM = zeros(N-1, nspin, lmaxrho+1, lmaxrho*2+1)

    rho_gal_mulipole_LM = zeros(N-1, nspin, lmaxrho+1, lmaxrho*2+1)

    


    
    for lr = 0:lmaxrho
        for mr = -lr:lr
            for l = 0:lmax
                for m = -l:l
                    gcoef = real_gaunt_dict[(lr,mr,l,m,l,m)]
                    for spin = 1:nspin
                        rho_rs_M_R2_LM[:,spin, lr+1,mr+lr+1] += gcoef * rho_rs_M_R2_LM2[:,spin, l+1, m+1+l]
                        if gga
                            drho_rs_M_R2_LM[:,spin, lr+1,mr+lr+1] += gcoef * drho_rs_M_R2_LM2[:,spin, l+1, m+1+l]
                        end
                    end
                end
            end
        end
    end

    
    rho_rs_M_R2_LM = rho_rs_M_R2_LM * sqrt(4 * pi)
    drho_rs_M_R2_LM = drho_rs_M_R2_LM * sqrt(4 * pi)

   

    
#    println("drho_rs_M_R2_LM  ", drho_rs_M_R2_LM[1])
    println("rho_rs_M_R2_LM  ", rho_rs_M_R2_LM[1])
    

#    for spin = 1:nspin
#        for lr = 0:lmaxrho
#            for mr = -lr:lr
#                if sum(abs.(rho_rs_M_R2_LM[:,spin, lr+1,mr+lr+1])) > 1e-10
#                    println("rho $spin  $lr $mr ", sum(abs.(rho_rs_M_R2_LM[:,spin, lr+1,mr+lr+1])))
#                end
#            end
#        end
#    end
    
    #    println("zz ", rho_rs_M_R2_LM[1,1,1,1])
    #=
    for spin = 1:nspin
    #        println("before1")
    a,b,c,d = get_rho_gal(rho_rs_M_R2[:,spin], g,N=N, invS=invS)
    #        println("length ret ", length(ret))
    rho_gal_R2[:,spin] += a
    rho_gal_dR[:,spin] += b
    rho_rs_M[:,spin]  += c
    end
    =#

    for spin = 1:nspin
        for lr = 0:lmaxrho
            for mr = -lr:lr
                #                for l = 0:lmax
                #                    for m = -l:l
                #                        println("before")
                #                        println("size(rho_rs_M_R2_LM[:,spin, lr+1, mr+lr+1] ", size(rho_rs_M_R2_LM[:,spin, lr+1, mr+lr+1]))
                a,b,c,d,e =  get_rho_gal(rho_rs_M_R2_LM[:,spin, lr+1, mr+lr+1], g,N=N, invS=invS, l = lr, drho_rs_M_R2_LM=drho_rs_M_R2_LM[:,spin, lr+1,mr+lr+1])
                #                        println("size ret ", size(ret))
                rho_gal_R2_LM[:,spin, lr+1, mr+lr+1] += a
                rho_gal_dR_LM[:,spin, lr+1, mr+lr+1 ] += b
                rho_rs_M_LM[:,spin, lr+1, mr+lr+1]  += c
                rho_gal_mulipole_LM[:,spin, lr+1, mr+lr+1]  += d
                if gga
                    drho_rs_M_LM[:,spin, lr+1, mr+lr+1]  += e
                end
                
                #                drho_gal_R2_LM[:,spin, lr+1, mr+lr+1] += D1*a
                #                    end
                #                end
            end
        end
    end

    R = g.R.(g.pts[2:M+2,M])
    t = mat_n2m*real(VECTS[:,1,1,1])
    t_dR = mat_m2n*(t.^2 ./ R  )
#    println("sub ", S*t_dR ./ rho_gal_dR_LM[:,1,1,1])
#    println("rho gal ", mat_m2n*(t.*^2) ./ rho_gal_R2_LM

    psipsi_r_M_LM = zeros(N-1, nspin, nmax, lmaxrho+1, lmaxrho*2+1, nmax, lmaxrho+1, lmaxrho*2+1) 
    if exx > 1e-12
        for spin = 1:nspin
            for n1 = 1:nmax
                for l1 = 0:lmax
                    for m1 = -l1:l1
                        for n2 = 1:nmax
                            for l2 = 0:lmax
                                for m2 = -l2:l2
                                    if filling[n2, spin, l2+1, l2+m2+1] > 1e-12 && filling[n1, spin, l1+1, l1+m1+1] > 1e-12
                                        a,b,c,d,e =  get_rho_gal(psipsi[:,spin, n1,l1+1, m1+1+l1, n2, l2+1, m2+l2+1], g,N=N, invS=invS, l = l1, drho_rs_M_R2_LM=missing)
                                        psipsi_r_M_LM[:,spin, n1,l1+1, m1+1+l1, n2, l2+1, m2+l2+1] = b
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end                                    

    println("psipsi_r_M_LM ", psipsi_r_M_LM[1], " ", rho_gal_dR_LM[1], " " , psipsi_r_M_LM[1]/rho_gal_dR_LM[1])
    
    #for spin = 1:nspin
    #    for lr = 0:lmaxrho
    #        for mr = -lr:lr
    #            drho_rs_M_LM[:,spin, lr+1, mr+lr+1] = gal_rep_to_rspace(drho_gal_R2_LM[:,spin, lr+1, mr+lr+1], g, M=M)
    #        end
    #    end
    #end
    
    #    println("sum rho ", sum(abs.(rho_gal_mulipole_LM[:,1, 1,1])), " " , sum(abs.(rho_gal_mulipole_LM[:,2, 1,1])))
    
    rho_gal_R2 = rho_gal_R2 / (g.b-g.a) * 2
    rho_gal_dR = rho_gal_dR/ (g.b-g.a) * 2
    rho_rs_M = rho_rs_M / (g.b-g.a) * 2

    rho_gal_R2_LM = rho_gal_R2_LM / (g.b-g.a) * 2
    rho_gal_dR_LM = rho_gal_dR_LM / (g.b-g.a) * 2
    rho_rs_M_LM = rho_rs_M_LM  / (g.b-g.a) * 2
    drho_rs_M_LM = drho_rs_M_LM  / (g.b-g.a) * 2
    rho_gal_mulipole_LM = rho_gal_mulipole_LM  / (g.b-g.a) * 2

    psipsi_r_M_LM = psipsi_r_M_LM   / (g.b-g.a) * 2
    

    #    println("SIZE ", size(rho_gal_R2_LM), " " , size(rho_gal_mulipole_LM), " lmax rho ", lmaxrho)
    
    #    println("a ", rho_gal_R2[1]," ", rho_gal_R2_LM[1])
    #    println("b ", rho_gal_dR[1]," ", rho_gal_dR_LM[1])
    #    println("c ", rho_rs_M[1]," ", rho_rs_M_LM[1])
    #    println("d ", rho_gal_R2[1]," ", rho_gal_mulipole_LM[1])

    
    #    println("ChebyshevDFT.Galerkin.do_1d_integral(rho_gal, g) ", do_1d_integral(rho_gal_R2[:,1], g))

    MP = zeros(lmaxrho+1, 2*lmaxrho+1)
    
    for spin = 1:nspin
        for l = 0:lmaxrho
            for m = -l:l
                MP[l+1, l+m+1] += do_1d_integral(rho_gal_mulipole_LM[:,spin,l+1,l+m+1], g)
            end
        end
    end
    #                println("l m $l $m")
    #                println("ChebyshevDFT.Galerkin.do_1d_integral(rho_gal, g) ", do_1d_integral(rho_gal_R2_LM[:,1,l+1,l+m+1], g))
    #                println("ChebyshevDFT.Galerkin.do_1d_integral(rho_muli, g) ", do_1d_integral(rho_gal_mulipole_LM[:,1,l+1,l+m+1], g))
    #            end
    #        end
    #    end
    
    
    if false
        begin
            n0 = xxx->ForwardDiff.derivative(xx-> ForwardDiff.derivative(x->gal_rep_to_rspace(x, rho_gal_R2[:,1,1,1], g), xx), xxx)
            n00 = (n0(0.0)/2.0 )
            #        println("n0 $(n00)")
        end
    end

    x = 1.0
    #println("test gal_rep_to_rspace ", gal_rep_to_rspace(x, rho_gal_R2[:,1], g), " " , gal_rep_to_rspace(x, rho_gal_dR[:,1], g) * x)
    
    return rho_gal_R2_LM, rho_gal_dR_LM, rho_rs_M_LM, MP, drho_rs_M_LM, psipsi_r_M_LM, nmax

end



function get_rho_big(VALS, VALS_big, VECTS_big, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, D2, lm_dict,dict_lm, big_code, gga)

    #    println("SIZE VB ", size(VECTS_big))

    Nbig = size(VECTS_big)[1]

    #    println("Nbig $Nbig M $M")
    
    #rho_rs_M_R2 = zeros(M+1, nspin)
    rho_rs_M_R2_LM2 = zeros(M+1, nspin, lmax+1, 2*lmax+1, lmax+1, 2*lmax+1)
    drho_rs_M_R2_LM2 = zeros(M+1, nspin, lmax+1, 2*lmax+1, lmax+1, 2*lmax+1)


    T = Dict()
    dT = Dict()
    
     for spin = 1:nspin
        for l = 0:lmax
            for m = -l:l
                for (c_n,n) = enumerate(big_code[(spin, l, m)])

                    #println("$c_n, size VALS, ", size(VALS))
                    if c_n > N-1
                        continue
                    end
                    VALS[c_n, spin, l+1, l+1+m] = VALS_big[n, spin]
                    fillval = filling[c_n, spin, l+1, l+m+1]
                    #                    println("add fillval $spin $l $m ", fillval)
                    if fillval < 1e-20
                        continue
                    end
                    for c1 = 1:length(lm_dict)
                        (ll1,mm1) = dict_lm[c1]
#                        t1 = gal_rep_to_rspace(VECTS_big[(1:(N-1)).+(c1-1)*(N-1), n, spin], g, M=M)
                        #                        dt1 = gal_rep_to_rspace(VECTS_big[(1:(N-1)).+(c1-1)*(N-1), n, spin], g, M=M, deriv=1)
                        #t1 = T1[:,c1, spin]
                        #dt1 = DT1[:,c1, spin]

                        if (c1,n,spin) in keys(T)
                            t1 = T[(c1,n,spin)]
                            dt1 = dT[(c1,n,spin)]
                        else
                            t1 = gal_rep_to_rspace(VECTS_big[(1:(N-1)).+(c1-1)*(N-1), n, spin], g, M=M)
                            dt1 = gal_rep_to_rspace(VECTS_big[(1:(N-1)).+(c1-1)*(N-1), n, spin], g, M=M, deriv=1)
                            T[(c1,n,spin)] = t1
                            dT[(c1,n,spin)] = dt1
                        end
                            
                        
                        for c2 = 1:length(lm_dict)
                            (ll2,mm2) = dict_lm[c2]
#                            t2 = gal_rep_to_rspace(VECTS_big[(1:(N-1)).+(c2-1)*(N-1), n, spin], g, M=M)

                            if (c2,n,spin) in keys(T)
                                t2 = T[(c2,n,spin)]
                                dt2 = dT[(c2,n,spin)]
                            else
                                t2 = gal_rep_to_rspace(VECTS_big[(1:(N-1)).+(c2-1)*(N-1), n, spin], g, M=M)
                                dt2 = gal_rep_to_rspace(VECTS_big[(1:(N-1)).+(c2-1)*(N-1), n, spin], g, M=M, deriv=1)
                                T[(c2,n,spin)] = t2
                                dT[(c2,n,spin)] = dt2
                            end


                            rho_rs_M_R2_LM2[:,spin, ll1+1, mm1+1+ll1, ll2+1, mm2+1+ll2] += real(t1.*conj(t2)) * fillval
                            if gga
                                #dt2 = gal_rep_to_rspace(VECTS_big[(1:(N-1)).+(c2-1)*(N-1), n, spin], g, M=M, deriv=1)
                                drho_rs_M_R2_LM2[:,spin,  ll1+1, mm1+1+ll1, ll2+1, mm2+1+ll2] += real(t1 .* conj.(dt2) + dt2 .* conj.(t1)) * fillval 
                            end
                            
                        end
                    end
                end
            end
        end
    end


    
    #    println("z ", rho_rs_M_R2[1,1], " " , rho_rs_M_R2_LM2[1,1, 1, 1] )
    
    rho_gal_R2 = zeros(N-1, nspin)
    rho_gal_dR = zeros(N-1, nspin)
    rho_rs_M = zeros(M+1, nspin)

    lmaxrho = min(lmaxrho, lmax * 2)
    rho_rs_M_R2_LM = zeros(M+1, nspin, lmaxrho+1, 2*lmaxrho+1)
    drho_rs_M_R2_LM = zeros(M+1, nspin, lmaxrho+1, 2*lmaxrho+1)

    rho_rs_M_LM = zeros(M+1, nspin, lmaxrho+1, lmaxrho*2+1)
    rho_gal_R2_LM = zeros(N-1, nspin, lmaxrho+1, lmaxrho*2+1)
    rho_gal_dR_LM = zeros(N-1, nspin, lmaxrho+1, lmaxrho*2+1)

    rho_gal_mulipole_LM = zeros(N-1, nspin, lmaxrho+1, lmaxrho*2+1)

    for lr = 0:2:lmaxrho
        for mr = -lr:lr
            for l1 = 0:lmax
                for m1 = -l1:l1
                    for l2 = 0:lmax
                        for m2 = -l2:l2
                            gcoef = real_gaunt_dict[(lr,mr,l1,m1,l2,m2)]
                            @tturbo for spin = 1:nspin
                                for ind in 1:M+1
                                    rho_rs_M_R2_LM[ind,spin, lr+1,mr+lr+1] += gcoef * rho_rs_M_R2_LM2[ind,spin, l1+1, m1+1+l1, l2+1, m2+1+l2]
                                    drho_rs_M_R2_LM[ind,spin, lr+1,mr+lr+1] += gcoef * drho_rs_M_R2_LM2[ind,spin, l1+1, m1+1+l1, l2+1, m2+1+l2]
                                end
                            end
                        end
                    end
                end
            end
            #println("sum abs $lr $mr ", sum(abs.(rho_rs_M_R2_LM[:,1, lr+1,mr+lr+1])))
        end
    end

    
    rho_rs_M_R2_LM = rho_rs_M_R2_LM * sqrt(4 * pi)
    drho_rs_M_R2_LM = drho_rs_M_R2_LM * sqrt(4 * pi)

#-
#    drho_rs_M_R2_LM = zeros(M+1, nspin, lmaxrho+1, lmaxrho*2+1)
#=    println("nspin $nspin ", size(drho_rs_M_R2_LM), " " , size(rho_rs_M_R2_LM))
    for spin = 1:nspin
        for l = 0:lmaxrho
            for m = -l:l
                grep = get_gal_rep(rho_rs_M_R2_LM[:,spin, l+1, m+1+l], g, N=N)
                dt = gal_rep_to_rspace(grep  , g, M=M, deriv=1)
                drho_rs_M_R2_LM[:,spin, l+1, m+1+l] += dt
            end
        end
    end
=#
#    println("drho_rs_M_R2_LM B ", drho_rs_M_R2_LM[1])
#    println("rho_rs_M_R2_LM B ", rho_rs_M_R2_LM[1])
    
    #for spin = 1:nspin
    #    for lr = 0:lmaxrho
    #        for mr = -lr:lr
    #            if sum(abs.(rho_rs_M_R2_LM[:,spin, lr+1,mr+lr+1])) > 1e-10
    #                println("rho $spin  $lr $mr ", sum(abs.(rho_rs_M_R2_LM[:,spin, lr+1,mr+lr+1])))
    #            end
    #        end
    #    end
    #end
    
    #    println("zz ", rho_rs_M_R2_LM[1,1,1,1])
    #=
    for spin = 1:nspin
    #        println("before1")
    a,b,c,d = get_rho_gal(rho_rs_M_R2[:,spin], g,N=N, invS=invS)
    #        println("length ret ", length(ret))
    rho_gal_R2[:,spin] += a
    rho_gal_dR[:,spin] += b
    rho_rs_M[:,spin]  += c
    end
    =#

    drho_rs_M_LM = zeros(M+1, nspin, lmaxrho+1, lmaxrho*2+1)
    
    for spin = 1:nspin
        for lr = 0:2:lmaxrho
            for mr = -lr:lr
                #                for l = 0:lmax
                #                    for m = -l:l
                #                        println("before")
                #                        println("size(rho_rs_M_R2_LM[:,spin, lr+1, mr+lr+1] ", size(rho_rs_M_R2_LM[:,spin, lr+1, mr+lr+1]))
                a,b,c,d,e =  get_rho_gal(rho_rs_M_R2_LM[:,spin, lr+1, mr+lr+1], g,N=N, invS=invS, l = lr,drho_rs_M_R2_LM=drho_rs_M_R2_LM[:,spin, lr+1,mr+lr+1], gga=gga)
                #                        println("size ret ", size(ret))
                rho_gal_R2_LM[:,spin, lr+1, mr+lr+1] += a
                rho_gal_dR_LM[:,spin, lr+1, mr+lr+1 ] += b
                rho_rs_M_LM[:,spin, lr+1, mr+lr+1]  += c
                rho_gal_mulipole_LM[:,spin, lr+1, mr+lr+1]  += d
                if gga
                    drho_rs_M_LM[:,spin, lr+1, mr+lr+1]  += e
                end
                
                #                    end
                #                end
            end
        end
    end                    

    #    println("sum rho ", sum(abs.(rho_gal_mulipole_LM[:,1, 1,1])), " " , sum(abs.(rho_gal_mulipole_LM[:,2, 1,1])))
    
    rho_gal_R2 = rho_gal_R2 / (g.b-g.a) * 2
    rho_gal_dR = rho_gal_dR/ (g.b-g.a) * 2
    rho_rs_M = rho_rs_M / (g.b-g.a) * 2

    rho_gal_R2_LM = rho_gal_R2_LM / (g.b-g.a) * 2
    rho_gal_dR_LM = rho_gal_dR_LM / (g.b-g.a) * 2
    rho_rs_M_LM = rho_rs_M_LM  / (g.b-g.a) * 2
    drho_rs_M_LM = drho_rs_M_LM  / (g.b-g.a) * 2
    rho_gal_mulipole_LM = rho_gal_mulipole_LM  / (g.b-g.a) * 2
    

    #    println("SIZE ", size(rho_gal_R2_LM), " " , size(rho_gal_mulipole_LM), " lmax rho ", lmaxrho)
    
    #    println("a ", rho_gal_R2[1]," ", rho_gal_R2_LM[1])
    #    println("b ", rho_gal_dR[1]," ", rho_gal_dR_LM[1])
    #    println("c ", rho_rs_M[1]," ", rho_rs_M_LM[1])
    #    println("d ", rho_gal_R2[1]," ", rho_gal_mulipole_LM[1])

    
    #    println("ChebyshevDFT.Galerkin.do_1d_integral(rho_gal, g) ", do_1d_integral(rho_gal_R2[:,1], g))

    MP = zeros(lmaxrho+1, 2*lmaxrho+1)
    
    for spin = 1:nspin
        #for l = 0:lmaxrho
        for l = [0]
            for m = -l:l
                MP[l+1, l+m+1] += do_1d_integral(rho_gal_mulipole_LM[:,spin,l+1,l+m+1], g)
            end
        end
    end
    #                println("l m $l $m")
    #                println("ChebyshevDFT.Galerkin.do_1d_integral(rho_gal, g) ", do_1d_integral(rho_gal_R2_LM[:,1,l+1,l+m+1], g))
    #                println("ChebyshevDFT.Galerkin.do_1d_integral(rho_muli, g) ", do_1d_integral(rho_gal_mulipole_LM[:,1,l+1,l+m+1], g))
    #            end
    #        end
    #    end
    
    
    if false
        begin
            n0 = xxx->ForwardDiff.derivative(xx-> ForwardDiff.derivative(x->gal_rep_to_rspace(x, rho_gal_R2[:,1,1,1], g), xx), xxx)
            n00 = (n0(0.0)/2.0 )
            #        println("n0 $(n00)")
        end
    end

    x = 1.0
    #println("test gal_rep_to_rspace ", gal_rep_to_rspace(x, rho_gal_R2[:,1], g), " " , gal_rep_to_rspace(x, rho_gal_dR[:,1], g) * x)
    
    return rho_gal_R2_LM, rho_gal_dR_LM, rho_rs_M_LM, MP, drho_rs_M_LM

end




#=
function rspace(r, B, arr, rfn, rmax)

a = 0.0
b = rmax
function X(r)
return -1.0 + 2 * (a - r)/(a-b)
end
function R(x)
return a .+ (x .- -1.0)*(b-a) / 2.0
end

f = zeros(typeof(arr[1]), size(r))
for c = 1:length(arr)
f += B[c].(rfn.(X.(r)))*arr[c]
end
return f #/ sqrt(b-a) * sqrt(2)
end


function rspace(r::Number, B, arr , rfn, rmax)

a = 0.0
b = rmax
function X(r)
return -1.0 + 2 * (a - r)/(a-b)
end
function R(x)
return a .+ (x .- -1.0)*(b-a) / 2.0
end

f = zero(r)
for c = 1:length(arr)
f += B[c](rfn(X(r)))*arr[c] #/ (b-a) 
end
return f #/ sqrt(b-a) * sqrt(2)
end


function jkl(f,bvals, pts, M, N, rmax)

a=0.0
b=rmax
function X(r)
return -1.0 + 2 * (a - r)/(a-b)
end
function R(x)
return a .+ (x .- -1.0)*(b-a) / 2.0
end

nr = f.(R.(pts[M, 2:M+2]))

#    println(size(bvals[M,1:N, 2:M+2]), " " , size(nr))

asdf_arr =   bvals[M,1:N,2:M+2]' \ nr

#    println(size(nr), " ", size(bvals[M,1:N,2:M+2]' * asdf_arr))

return asdf_arr, nr, bvals[M,1:N,2:M+2]' * asdf_arr
end

function jkl2(f,bvals, pts, M, N, rmax, w)

a=0.0
b=rmax
function X(r)
return -1.0 + 2 * (a - r)/(a-b)
end
function R(x)
return a .+ (x .- -1.0)*(b-a) / 2.0
end

nr = f.(R.(pts[M, 2:M+2])) .*w[M,2:M+2]

asdf_arr = zeros(N)
for i = 1:N
asdf_arr[i] =   sum( (@view bvals[M,i,2:M+2]).*nr )#.*w[M,2:M+2])
end

#    println(size(nr), " ", size(bvals[M,1:N,2:M+2]' * asdf_arr))

return asdf_arr, nr, bvals[M,1:N,2:M+2]' * asdf_arr
end
=#




#=
function get_rho_gal(arr,bvals, pts, w, M, invS)

N = length(arr)
nr = zeros(typeof(arr[1]), M+1)
for n1 = 1:N
nr += arr[n1]*bvals[M,n1,2:M+2]
end
#nr = nr.*w[M,2:M+2]
nrR = real(nr .* conj(nr)) #.*w[M,2:M+2]

#rho_gal =   bvals[M,1:length(arr),2:M+2]' \ nrR

rho_gal = zeros(typeof(arr[1]), N)
for i = 1:N
rho_gal[i] =   sum( (bvals[M,i,2:M+2]).*nrR.*w[M,2:M+2] )
end

return invS*rho_gal

end
=#

function vxx_LM(psipsi, D2, g, N, M, lmaxrho, lmax, MP, V_L, gbvals2, exx, nmax, nspin, filling, VECTS, S, rho_dR)

#    VX_LM = zeros(N-1, N-1, nspin, lmax+1, 2*lmax+1)

#    for spin = 1:nspin
#        for n1 = 1:nmax
#            for l1 = 0:lmax
#                for m1 = -l1:l1
#                    VX_LM[:, :, spin, l1+1, m1+l1+1] += vhart(psipsi[:,spin, n1,l1+1, m1+l1+1,n1,l1+1, m1+l1+1], D2, V_L, g, M, l1, m1, 0.0*MP, gbvals2)
#                end
#            end
#        end
#    end

    mat_n2m = MAT_N2M(g, N=N, M=M)
    mat_m2n = MAT_M2N(g, N=N, M=M)
    R = g.R.(g.pts[2:M+2,M])
    l=0; m=0
    
#    a, vh_mat2, vt, X = vhart(0.0*zeros(N-1), D2, V_L, g, M, l, m, MP, gbvals2)
    

    t = mat_n2m*real(VECTS[:,1,1,1])

    t_dR = ( mat_m2n*(t.^2 ./ R  )/ (g.b-g.a) * 2 )

    #    println("subVH ", S*t_dR ./ rho_dR[:,1,1,1])
    
#    t = mat_n2m*real(VECTS[:,1,1,1])
#    t_dR = mat_m2n*(t.^2 ./ R  )
#    println("sub ", S*t_dR ./ rho_gal_dR_LM[:,1,1,1])

#    t = mat_n2m*real(VECTS[:,1,1,1]) 
#    t_dR = mat_m2n*(t.^2 ./ R  )

#    di = diagm(t_dR)

#    println("rho test ",  (S*t_dR)[1:3]./ rho_dR[1:3] )

    #vx2_temp =  L * S *mat_m2n*di*t_dR
#    vx2_temp =  L * S *  t_dR

    #@time for ii = 1:N-1
    #    for jj = 1:N-1
    #        for bb = 1:N-1
    #            #                        println("$ii $jj $bb")
    #            vx2[ii,jj] += X[ii,jj,bb]*vx2_temp[bb]
    #        end
    #   end
    #end



#    t_old = mat_n2m*real(VECTS[:,1,1,1])
    
#    t_dR_old = ( mat_m2n*(t_old.^2 ./ R  )/ (g.b-g.a) * 2 )

    
#    t = mat_n2m*real(VECTS[:,1,1,1])
#    
#    t_dR = t.^2 ./ R / (g.b-g.a) * 2  
#    di = diagm(t_dR)

#    println("xxxxxxxxxxx ", (di*t_dR)[1:3] ./ t_dR_old[1:3])
    
    #    temp = L * S * mat_m2n * t_dR

#    L = inv((D2 + l*(l+1)*V_L))


    #=vx2 =  zeros(N-1,N-1)
    VX_LM2 = zeros(N-1, N-1, nspin, lmax+1, 2*lmax+1)

    t = mat_n2m*real(VECTS[:,1,1,1,1])
    
    t_dR =  t.^2 ./ R  / (g.b-g.a) * 2 

    tf1 =  t ./ R  / sqrt(g.b-g.a) * sqrt(2 )
    
    temp = diagm(tf1) * mat_m2n' * S * L * S *  mat_m2n * diagm(tf1)
    vx2[:,:] =     mat_n2m'*temp*mat_n2m * -1*sqrt(pi)/(2*pi) / 2
    VX_LM2[:,:,1,1,1] = vx2
=#


    vx2 =  zeros(N-1,N-1)
    VX_LM2 = zeros(N-1, N-1, nspin, lmax+1, 2*lmax+1)


    #    r5 = mat_n2m'*diagm(1 ./ R.^0.5)*mat_n2m
    r = mat_n2m'*diagm( R)*mat_n2m
    println("size(r) ", size(r))
    L = inv(  (D2 + l*(l+1)*V_L) )
    
    for spin = 1:nspin
        vx2 .= 0.0
        for n = 1:N-1
            f = filling[n,spin,1,1]
#            println("f $spin $n  $f")
            if f < 1e-20
                break
            end
            t = mat_n2m*VECTS[:,n,spin,1,1]

            #tf1 =  t ./ R  / sqrt(g.b-g.a) * sqrt(2 )
            tf1 =  t ./ R  / sqrt(g.b-g.a) * sqrt(2 )
            #tf1 =  t   / sqrt(g.b-g.a) * sqrt(2 )

            #            tf1a =  t ./ R.^2  / sqrt(g.b-g.a) * sqrt(2 )
#            tf1b =  t   / sqrt(g.b-g.a) * sqrt(2 )
           #temp = diagm(conj.(tf1)) * mat_m2n' * S * L * S *  mat_m2n * diagm(tf1)
            temp = diagm(conj.(tf1)) * mat_m2n' * S* L *S  *  mat_m2n * diagm(tf1)

            #temp = diagm(conj.(tf1)) * mat_n2m  * S * L * S *  mat_n2m' * diagm(tf1)
            
            #            vx2[:,:] +=   real(  f *  mat_n2m'*temp*mat_n2m )
            #vx2[:,:] +=   real(  f *   mat_n2m'*diagm(1 ./ R) *temp *  diagm(1 ./ R) *mat_n2m  )
            #vx2[:,:] +=   real(  f *  mat_m2n*temp*mat_m2n')
#            vx2[:,:] +=   real(  f *  mat_n2m'*temp*mat_n2m)


            vx2[:,:] +=   real(  f *  mat_n2m'*   diagm(conj.(tf1)) * mat_m2n' * S * L * S *  mat_m2n * diagm(tf1)  *mat_n2m)
            

            #t = mat_n2m*VECTS[:,1,1,1,1]
            #tf1 =  t ./ R  / (g.b-g.a) * (2 )
            #temp = L * S *  mat_m2n * diagm(tf1) * t
            #correct vh
            #vx2 += f*    mat_n2m'*diagm( (mat_m2n'*S*temp)./R  )*mat_n2m # *sqrt(pi)/(2*pi) / 2
            
            
            
        end
        vx2 = 0.5*(vx2+vx2')
        VX_LM2[:,:,spin,1,1] = vx2 * -1*sqrt(pi)/(2*pi) / 2
    end
#    println("test symm sum(abs.(VX_LM2[:,:,1,1,1] - VX_LM2[:,:,1,1,1]'))   " , sum(abs.(VX_LM2[:,:,1,1,1] - VX_LM2[:,:,1,1,1]'))sum(abs.(VX_LM2[:,:,1,1,1] - VX_LM2[:,:,1,1,1]')))
    
    #correct hartree
    #t = mat_n2m*VECTS[:,1,1,1,1]
    #tf1 =  t ./ R  / (g.b-g.a) * (2 )
    #temp = 2.0*L * S *  mat_m2n * diagm(tf1) * t
    #correct vh
    #vx_temp =     mat_n2m'*diagm( (mat_m2n'*S*temp)./R  )*mat_n2m *sqrt(pi)/(2*pi) / 2
    #println("QRQRQRQRQRQRQ ", sum(abs.(vx_temp - vx2)))

#    println("RRRRRRRRRRRRRRRRRRRRR test ", VECTS[:,1,1,1,1]' *vx_temp * VECTS[:,1,1,1,1])

    
    #temp = mat_n2m' * diagm(tf) * mat_m2n' *  S' * L * S *  mat_m2n * diagm(tf) * mat_n2m
    #println("size temp ", size(temp))
    #vx2[:,:] = temp
    #vx2[:,:] = mat_n2m' * diagm(temp) * mat_n2m
#    @time for ii = 1:N-1
#        for jj = 1:N-1
#            for m1 = 1:M+1
#                for m2 = 1:M+1
#                    vx2[ii,jj] += temp[m1,m2] * g.bvals[m1+1,ii,M] * g.bvals[m2+1,jj,M] * g.w[m1+1,M] * g.w[m2+1,M]
#                end
#            end
#        end
#    end

#    vx2 .= 0.0
#    t = mat_n2m*real(VECTS[:,1,1,1])

#    t_dR = ( (t.^2 ./ R  )/ (g.b-g.a) * 2 ) * sqrt(pi)/(2*pi) / 2

#    println("t_dR ", t_dR[1:3])

    
    #t = mat_n2m*real(VECTS[:,1,1,1]) ./ R  / (g.b-g.a) * 2 
    #t_dR = t  
    #di = diagm(t_dR)

#    println("size(t_dR) ", size(t_dR), ", size(mat_m2n) ", size(mat_m2n))
    
#    temp = L*S*mat_m2n*t_dR # *di*(t_dR)

#    println("temp ", temp[1:3])

    
#    @time for ii = 1:N-1
#        for jj = 1:N-1
#            for bb = 1:N-1
#                vx2[ii,jj] += X[ii,jj,bb]*temp[bb]
#            end
#        end
#    end
#    VX_LM2[:,:,1,1,1] = vx2
               #     for bb = 1:N-1
               #         for cc = 1:N-1
               #             for dd = 1:N-1
               #                 for ee = 1:M+1
               #                     for ff = 1:M+1
               #                         ex2[ii,jj] += X[ii,jj,bb]*invD[bb,cc]*S[cc,dd]*mat_m2n[dd,ee]*di[ee,ff]*t_dR[ff]


    


    #--

#=
    tt = MP[l+1, m+l+1]/g.b^(l+1) * g.w[2:M+2,M]
    INT = zeros(N-1, N-1)
    for n1 = 1:(N-1) #
        for n2 = 1:(N-1)
            for i = 1:(M+1)
                INT[n1, n2] += gbvals2[i,n1,n2] * tt[i]
            end
        end
    end
    VX_LM2 = -VX_LM2 / 4 / sqrt(pi)
    VX_LM2[:, :, 1, l+1, m+l+1] += -INT/sqrt(4*pi)
    println("vxlm2 2 ", VX_LM2[1:3, 1, 2, l+1, m+l+1])
=#

    
#    VX_LM3 = zeros(N-1, N-1, nspin, lmax+1, 2*lmax+1)
#    for n1 = 1:(N-1) #
#        for n2 = 1:(N-1)
#            for i = 1:(M+1)
#                INT[n1, n2] += gbvals2[i,n1,n2] * tt[i]
#            end
#        end
#    end

    
    
    #--    
    
    
#    for spin = 1:nspin
#        for n1 = 1:nmax
#            for l1 = 0:lmax
#                for m1 = -l1:l1
#                    for i = 1:N-1
#                        for j = 1:N-1
#                            VX_LM2[i,j, spin, l1+1, m1+l1+1] +=  VECTS[i,n1, spin, l1+1, l1+1+m1]*L[i,j]*VECTS[j,n1, spin, l1+1, l1+1+m1]
#                        end
#                    end
#                end
#            end
#        end
#    end
#    println("new r ", VX_LM2[1] / VX_LM[1])
    
                    #                    fillval1 = filling[n1, spin, l1+1, l1+m1+1]

#                    diagmat = diagm(VECTS[:,n1, spin, l1+1, l1+1+m1])
#                    VX_LM[:, :, spin, l1+1, m1+l1+1] += real.(conj(diagmat)*S*(D2 + l1*(l1+1)*V_L)^-1*S*diagmat)
#                end
#            end
#        end
#    end
#                    for n2 = 1:nmax
#                        for l2 = 0:lmax
#                            for m2 = -l1:l1
#                                VX_LM[:, :, spin, l1+1, m1+l1+1] += vhart(psipsi[:,spin, n1,l1+1, m1+l1+1,n2,l2+1, m2+l2+1], D2, V_L, g, M, l1, m1, MP, gbvals2)
#                            end
#                        end
#                    end
#                end
#            end
#        end
#    end
    if nspin == 1
        return VX_LM2*exx/2.0
    else
        return VX_LM2*exx
    end
    
end



function vhart_LM(rho_dR, D2, g, N, M, lmaxrho, lmax, MP, V_L, gbvals2, S, VECTS)


    #    println("size rho_dR ", size(rho_dR))
    #    println("lmax rho ", lmaxrho)
    loopmax = min(lmax, lmaxrho)
    #loopmax = lmaxrho
    VH_LM = zeros(N-1, N-1, lmaxrho+1, 2*lmaxrho+1)
    VH_LM2 = zeros(N-1, N-1, lmaxrho+1, 2*lmaxrho+1)
    #    println("size VH_LM ", size(VH_LM))

    mat_n2m = MAT_N2M(g, N=N, M=M)
    mat_m2n = MAT_M2N(g, N=N, M=M)

    invD = D2^-1
    R = g.R.(g.pts[2:M+2,M])

    
    for l = 0:loopmax
        for m = -l:l
            #            println("vhart $l $m ")
            VH_LM[:, :, l+1, m+l+1], vh_mat2, vt, X = vhart(rho_dR[:,1,l+1, m+l+1], D2, V_L, g, M, l, m, MP, gbvals2)



            
#            a = diagm(vt)
#            b = S*diagm(vt)
#            c = diagm(vt)*S
#            d = (D2*S) \ rho_dR[:,1,l+1, m+l+1]
#            e = (D2) \ (S*rho_dR[:,1,l+1, m+l+1])
#            f = inv(D2) * (diagm(rho_dR[:,1,l+1, m+l+1]))
#            gx = inv(D2) * (S*diagm(rho_dR[:,1,l+1, m+l+1]))
#            h = S^0.5*diagm(vt)*S^0.5

            #diagmat = diagm(rho_dR[:,1,l+1, m+l+1].^0.5)
#            ex2 = zeros(N-1, N-1)
#            ex3 = zeros(N-1, N-1)

#            ex = zeros(N-1, N-1)
            t = invD*(@view rho_dR[:,1,l+1,m+l+1])
#            println("size t ", size(t), " size  X ", size(X), " size ex ", size(ex))
            for ii = 1:N-1
                for jj = 1:N-1
                    for bb = 1:N-1
                        VH_LM2[ii, jj, l+1, m+l+1] += X[ii,jj,bb]*t[bb]
#                    for bb = 1:N-1
#                        for cc = 1:N-1                        
#                            ex[ii,jj] += X[ii,jj,bb]*invD[bb,cc]*rho_dR[cc,1,l+1, m+l+1]
#                        end
#                    end
                    end
                end
            end
#            ex = ex / 4 / sqrt(pi)
#            println("test ", VH_LM[1:3, 1, l+1, m+l+1] ./ ex[1:3])
            #=
            R = g.R.(g.pts[2:M+2,M])
            t = mat_n2m*real(VECTS[:,1,1,1]) ./ R.^0.5
            t_dR = t  
            di = diagm(t_dR)
                  
#=            @time for ii = 1:2
                for jj = 1:2
                    for bb = 1:N-1
                        for cc = 1:N-1
                            for dd = 1:N-1
                                for ee = 1:M+1
                                    for ff = 1:M+1
                                        ex2[ii,jj] += X[ii,jj,bb]*invD[bb,cc]*S[cc,dd]*mat_m2n[dd,ee]*di[ee,ff]*t_dR[ff]
                                    end
                                end
                            end
                        end                            
                    end
                end
            end
=#
            temp = invD*S*mat_m2n*di*t_dR
            temp2 = zeros(N-1)
            for bb = 1:N-1
                for cc = 1:N-1
                    for dd = 1:N-1
                        for ee = 1:M+1
                            for ff = 1:M+1                        
                                temp2[bb] += invD[bb,cc]*S[cc,dd]*mat_m2n[dd,ee]*di[ee,ff]*t_dR[ff]
                            end
                        end
                    end
                end                            
            end


            vx1_temp = di *  mat_m2n' * S* invD * S*  mat_m2n * di 
            vx1 =  mat_n2m' * vx1_temp * mat_n2m

            vx1a = zeros(N-1, N-1)
            for i = 1:N-1
                for j = 1:N-1
                    for m1 = 1:M+1
                        for m2 = 1:M+1
                            vx1a[i,j] += vx1_temp[m1, m2] * g.bvals[m1+1,i,M] * g.bvals[m2+1,j,M] * g.w[m1+1,M] * g.w[m2+1,M]
                        end
                    end
                end
            end

#            vx2_temp = t_dR'  *  mat_m2n' * S* invD * S*  mat_m2n * di
            vx2_temp =  invD * S *mat_m2n*di*t_dR
            vx2 =  zeros(N-1,N-1)

            @time for ii = 1:2
                for jj = 1:2
                    for bb = 1:N-1
#                        println("$ii $jj $bb")
                        vx2[ii,jj] += X[ii,jj,bb]*vx2_temp[bb]
                    end
                end
            end

            
            
            
            
            println("size vx1_temp ", size(vx1_temp), " sub ", sum(abs.(vx1_temp - vx1_temp')))
            
            println("size vx1 ", size(vx1), " sub ", sum(abs.(vx1 - vx1')))
            println("size vx1a ", size(vx1), " sub ", sum(abs.(vx1a - vx1a')))

            
            
            println("temp check ", temp - temp2)
            println("size temp ", size(temp), " X " , size(X), " ex3 ", size(ex3), " mat_m2n ", size(mat_m2n), " S ", size(S), " invD ", size(invD), " di ", size(di))
            @time for ii = 1:2
                for jj = 1:2
                    for bb = 1:N-1
#                        println("$ii $jj $bb")
                        ex3[ii,jj] += X[ii,jj,bb]*temp[bb]
                    end
                end
            end
            println("zero ", VH_LM[:, :, l+1, m+l+1][1:2], " a ", a[1:2], " b ", b[1:2])
#            println([c[1:2], d[1:2], e[1:2], f[1:2], g[1:2]])
            for aa in [a[1:2], b[1:2], c[1:2], d[1:2], e[1:2], f[1:2], gx[1:2], h[1:2], vh_mat2[1:2], ex[1:2], ex2[1:2] ,ex3[1:2], vx1[1:2], vx1a[1:2], vx2[1:2]]
                println(aa ./  VH_LM[:, :, l+1, m+l+1][1:2])
            end
            println("sym ", sum(abs.(vh_mat2 - vh_mat2')))
            =#
        end
    end



    l=0; m=0

    a, vh_mat2, vt, X = vhart(0.0*rho_dR[:,1,l+1, m+l+1], D2, V_L, g, M, l, m, MP, gbvals2)


    tt = MP[l+1, m+l+1]/g.b^(l+1) * g.w[2:M+2,M]


    INT = zeros(N-1, N-1)
    for n1 = 1:(N-1) #
        for n2 = 1:(N-1)
            for i = 1:(M+1)
                INT[n1, n2] += gbvals2[i,n1,n2] * tt[i]
            end
        end
    end

    
    VH_LM2 = VH_LM2 / 4 / sqrt(pi)
    VH_LM2[:, :, l+1, m+l+1] += INT/sqrt(4*pi)
    
    println("test  ", VH_LM2[1:3, 1, l+1, m+l+1] ./ VH_LM[1:3, 1, l+1, m+l+1])
    println("test2 ", INT[1:3,1] ./ VH_LM[1:3, 1, l+1, m+l+1] / sqrt(4*pi))

    
    
    return VH_LM
    
end


function vhart(rhor2, D2, V_L, g, M, l, m, MP, gbvals2)


    vh_tilde = (D2 + l*(l+1)*V_L) \ rhor2
    vh_tilde = vh_tilde /(4*pi)*sqrt(pi)

#    println("vh_tilde ", sum(abs.(vh_tilde)), " rhor2 ", sum(abs.(rhor2)))
    
    vh_tilde_copy = deepcopy(vh_tilde)

#    println("size vh_tilde ", size(vh_tilde))
    vh_mat, vh_mat2, X = get_vh_mat(vh_tilde, g, l, m, MP, gbvals2, M=M)
    
#    println("$l $m size D2 ", size(D2), " size(V_L) " , size(V_L), " size(rhor2) ", size(rhor2), " size vh_mat ", size(vh_mat), " size vh_tilde", size(vh_tilde))
    return vh_mat, vh_mat2,vh_tilde_copy, X
    
end

function vxc_LM(rho_rs_M, drho_rs_M, g, M, N, funlist, gga, nspin, lmax, lmaxrho, LEB, R, invS, gbvals2)

    begin
        #get vxc in r space

        rho = zeros( M+1, nspin, LEB.N)
        drho = zeros(M+1, nspin,3, LEB.N)
        #    drho_tp = zeros(M+1, nspin,2)
        ddrho_omega = zeros(M+1, nspin,LEB.N)    
        
        if nspin == 1
            VSIGMA_tp = zeros( M+1, 1, LEB.N)
        elseif nspin == 2
            VSIGMA_tp = zeros( M+1, 3, LEB.N)
        end
        
        VRHO_tp = zeros( M+1, nspin, LEB.N)
        ERHO_tp = zeros( M+1, nspin, LEB.N)
        
        
        loopmax = min(lmax*2, lmaxrho)
    end
        
    #get rho and then VXC in theta / phi / r space
#    println("get rho ntp and v")
#=
    @time for ntp = 1:LEB.N
        theta = LEB.θ[ntp]
        for spin = 1:nspin
            for l = 0:2:loopmax
                for m = -l:l
                    rho[:, spin, ntp] += rho_rs_M[:,spin, l+1, m+l+1] * LEB.Ylm[(l,m)][ntp]
                    if gga
                        drho[:, spin, 1, ntp] += drho_rs_M[:,spin, l+1, m+l+1] * LEB.Ylm[(l,m)][ntp]
                        drho[:, spin,2, ntp] += rho_rs_M[:,spin, l+1, m+l+1] * LEB.dYlm_theta[(l,m)][ntp]
                        drho[:, spin,3, ntp] += rho_rs_M[:,spin, l+1, m+l+1] * LEB.dYlm_phi_sin[(l,m)][ntp]
                        ddrho_omega[:, spin, ntp] += -(l)*(l+1)*rho_rs_M[:,spin, l+1, m+l+1] * LEB.Ylm[(l,m)][ntp] #spherical laplace eq
                    end
                end
            end
        end
    end
=#
    for l = 0:2:loopmax
        for m = -l:l
            @turbo for spin = 1:nspin
                for ntp = 1:LEB.N
                    for r = 1:M+1
                        theta = LEB.θ[ntp]
                        rho[r, spin, ntp] += rho_rs_M[r,spin, l+1, m+l+1] * LEB.Ylm[(l,m)][ntp]
                        drho[r, spin, 1, ntp] += drho_rs_M[r,spin, l+1, m+l+1] * LEB.Ylm[(l,m)][ntp]
                        drho[r, spin,2, ntp] += rho_rs_M[r,spin, l+1, m+l+1] * LEB.dYlm_theta[(l,m)][ntp]
                        drho[r, spin,3, ntp] += rho_rs_M[r,spin, l+1, m+l+1] * LEB.dYlm_phi_sin[(l,m)][ntp]
                        ddrho_omega[r, spin, ntp] += -(l)*(l+1)*rho_rs_M[r,spin, l+1, m+l+1] * LEB.Ylm[(l,m)][ntp] #spherical laplace eq
                    end
                end
            end
        end
    end
    
    #println("EXC_get_ve")
    @inbounds @threads for ntp = 1:LEB.N
        theta = LEB.θ[ntp]
        if !ismissing(funlist) && funlist != :lda_internal
            vrho, vsigma, erho =  EXC_get_ve( (@view rho_rs_M[:,:, 1, 1]), (@view rho[:,1:nspin,ntp])/sqrt(4*pi)  , funlist, (@view drho[:,1:nspin,:,ntp])/sqrt(4*pi), theta, gga, R)
            VSIGMA_tp[:,:,ntp] = vsigma 
            VRHO_tp[:,:,ntp] = vrho
        else
            if nspin == 1
                v = v_LDA.(rho[:, 1, ntp] * sqrt(4*pi)/ (4*pi))
                VRHO_tp[:,1,ntp] = v
            elseif nspin == 2
                v = v_LDA_sp.(rho[:, 1, ntp][:] * sqrt(4*pi)/ (4*pi) ,  rho[:, 2, ntp][:] * sqrt(4*pi)/ (4*pi))
                VRHO_tp[:,1:2,ntp] = reshape(vcat(v...), nspin,M+1)'
            else
                println("nspin must be 1 or 2 : $nspin")
            end
        end
    end

    #now transform vsigma to LM space
    #println("more vars")
    if nspin == 1
        VSIGMA_LM = zeros(M+1, 1, lmaxrho+1, 2*lmaxrho+1)
        nind = 1
    elseif nspin == 2
        VSIGMA_LM = zeros(M+1, 3, lmaxrho+1, 2*lmaxrho+1)
        nind = 3
    end
    
    #println("FT vsigma")
    if gga
        for l = 0:2:lmaxrho
            for m = -l:l
                @tturbo for r = 1:M+1
                    for ind = 1:nind
                        for ind2 = 1:LEB.N
                            VSIGMA_LM[r, ind, l+1, l+m+1] +=  4*pi*LEB.Ylm[(l,m)][ind2] * VSIGMA_tp[r, ind, ind2] * LEB.w[ind2]
                        end
                    end
                end
            end
        end
    end
    
    #now transform back derivatives wrt t, pi
    #println("more more vars")
    begin
        dvsigma_theta = zeros(M+1, nind, nthreads())
        dvsigma_phi = zeros(M+1, nind, nthreads())
        VXC_tp = zeros( M+1, nspin, LEB.N, nthreads())
    end
    
    #    EXC_tp = zeros( M+1, nspin, LEB.N)
    
    #println("FT back, EXC_gal")
    for ntp = 1:LEB.N
        id = threadid()
        dvsigma_theta[:,:,id] .= 0.0
        dvsigma_phi[:,:,id] .= 0.0
#        vsigma_test .= 0.0
        if gga
            @inbounds for l = 0:2:lmaxrho
                for m = -l:l
                    dvsigma_theta[:,:,id] += (@view VSIGMA_LM[:, :, l+1, l+m+1]) * LEB.dYlm_theta[(l,m)][ntp]
                    dvsigma_phi[:,:,id] += (@view VSIGMA_LM[:, :, l+1, l+m+1]) * LEB.dYlm_phi_sin[(l,m)][ntp]
                end
            end
        end

        
        theta = LEB.θ[ntp]

        if !ismissing(funlist) && funlist != :lda_internal
            v = EXC_gal( (@view rho[:, :, ntp]) / sqrt(4*pi), funlist, (@view drho[:,:,:,ntp])/sqrt(4*pi), (@view ddrho_omega[:,:,ntp])/sqrt(4*pi), (@view VRHO_tp[:,:, ntp]), (@view VSIGMA_tp[:, :, ntp]), (@view dvsigma_theta[:,:,id]), (@view dvsigma_phi[:,:,id]), theta, gga, R,  g, invS, N, M)
            VXC_tp[:,:,ntp, id] += v
        else
            VXC_tp[:,:,ntp,id] += VRHO_tp[:,:,ntp]
        end

    end
    VXC_tp = sum(VXC_tp, dims=4)
    
    
    
    #now transform to LM space again
    #println("VXC to LM")
    VXC_LM = zeros(M+1, nspin, lmaxrho+1, 2*lmaxrho+1)
    for spin = 1:nspin
        for l = 0:2:lmaxrho
            for m = -l:l
                @tturbo for r = 1:M+1
                    for ind = 1:LEB.N
                        VXC_LM[r, spin, l+1, l+m+1] +=  4*pi*LEB.Ylm[(l,m)][ind] * VXC_tp[r,spin,ind] * LEB.w[ind]
                    end
                end
            end
        end
    end


    #VXC_LM = VXC_LM #/ sqrt(4*pi)

    
    VXC_LM_MAT = zeros(N-1,N-1,nspin, lmaxrho+1, 2*lmaxrho+1)
    #println("get matrix")
    for spin = 1:nspin
        for l = 0:2:lmaxrho
            for m = -l:l
                VXC_LM_MAT[:,:,spin,l+1,l+m+1] = get_gal_rep_matrix_R(VXC_LM[:,spin,l+1,l+m+1], g,gbvals2 ; N = N)

            end
        end
    end
    EXC_tp = zeros(N-1,N-1,nspin, lmaxrho+1, 2*lmaxrho+1)
    
    return VXC_LM_MAT, VXC_tp, EXC_tp, VSIGMA_tp
    
end


#=
function vxc(rho,drho,ddrho, g, M, N, funlist, gga, nspin, theta, R, invS)

    #    println("size rho $(size(rho)) vs ", nspin)

    #    println("FUNLIST $funlist")

    #    ddrho = zeros(size(drho))
    dvsigma = zeros(size(drho))
    
    VLDA = zeros(size(rho,1), nspin)
    #println("size R ", size(R))
    #    DRHO = zeros(length(drho), 3)
    #    DRHO[:,1] = drho[:,1]
    if nspin == 1
        if !ismissing(funlist) && funlist != :lda_internal
            #e,v = EXC( (@view rho[:,1), funlist, (@view drho[:,1,l,m,:]), (@view ddrho[:,1,l,m,:]), (@view dvsigma[:,1,l,m,:]), THETA[l], gga, r, D1)
            if gga
                vrho, vsigma, erho =  EXC_get_ve( (rho[:,1]/(4*pi)  ), funlist, (drho/(4*pi)), theta, gga, R)
                #                e,v = EXC_gal( (rho[:,1]/(4*pi)  ), funlist, (drho/(4*pi)), ddrho, (dvsigma[:,1]), theta, gga, R, g, invS, N, M)
            else
                println("USE EXTERNAL LDA?")
                e,v = EXC_gal( (rho[:,1]/(4*pi)  ), funlist, missing,       missing,       missing,    theta, gga, R,  g, invS, N, M)
            end
            VLDA[:,1] = v
        else
            rho_col = reshape(rho , length(rho),1)
            VLDA[:,1] = v_LDA.(rho[:, 1]/ (4*pi))
        end
    elseif nspin == 2
        #=
        if !ismissing(funlist) && funlist != :lda_internal
        if gga
        e,v = EXC_sp( ( rho[:,1:2] / (4*pi) ), funlist, (@view drho[:,1:2,:]), (@view ddrho[:,1:2,:]), (@view dvsigma[:,1:3,:]), THETA[l], gga, r, D1)
        else
        e,v = EXC_sp( ( rho[:,1:2] /  (4*pi)), funlist, missing, missing, missing, missing, gga, missing, missing)
        end                        
        
        else
        tt =  v_LDA_sp.(rho[:, 1]/ (4*pi), rho[:, 2]/ (4*pi))
        VLDA[:,:] = reshape(vcat(tt...), nspin,M+1)'
        end
        =#
    end

    VLDA = VLDA * sqrt(4*pi)

    #    println("size VLDA ", size(VLDA))
    #VLDA_mat = get_gal_rep_matrix_R(VLDA[:,1,1,1], g, ; N = N)
    
    #return VLDA_mat #, VLDA

    return VLDA
end
=#

function display_eigs(VALS, nspin,lmax)
    println()
    for spin = 1:nspin
        for l = 0:lmax
            for m = -l:l
                if m < 0
                    println("vals spin =  $spin l = $l m = $m   ", VALS[1:min(6, size(VALS)[1]), spin, l+1,m+l+1])
                else
                    println("vals spin =  $spin l = $l m =  $m   ", VALS[1:min(6, size(VALS)[1]), spin, l+1,m+l+1])

                end
            end
        end
        println()
    end

end

function solve_small(V_C, V_L, VH_LM, VXC_LM, VX_LM, D2, S, nspin, lmax, lmaxrho, funlist, VECTS, VALS,exx)

    VLM = zeros(size(V_C))

    #    println("sub abs VXC_LM ",sum(abs.( VXC_LM)))

    
    println("ratio ", VH_LM[1,1,1,1] / VX_LM[1,1,1,1,1])

    Sh = Hermitian(S)

    for spin = 1:nspin 
        for l = 0:(lmax)
            V = V_C + V_L*l*(l+1)
            for m = -l:l

                if funlist != :hydrogen  #VHART AND VXC_LM
                    VLM .= 0.0
                    for lr = 0:lmaxrho
                        for mr = -lr:lr
                            gcoef = real_gaunt_dict[(lr,mr,l,m,l,m)]
                            VLM += gcoef * (4*pi*VH_LM[:,:,lr+1,lr+mr+1] + VXC_LM[:,:,spin,lr+1,lr+mr+1] + 4*pi*VX_LM[:,:,spin, lr+1,lr+mr+1])

                            
                        end
                    end
                end
                
                #println("eigen")

#                println("sum abs Ham ", sum(abs.(D2 + V + VLM)), " ", sum(abs.(S)))
                Hh = Hermitian(D2 + V + VLM)
                vals, vects = eigen(Hh, Sh)
                #S5 = S^-0.5
                #vals, vects = eigen(S5*(D2 + V + VLM)*S5)
                VECTS[:,:,spin, l+1, l+1+m] = vects
                VALS[:,spin, l+1, l+1+m] = vals
            end
        end
    end
    
    return VALS, VECTS
    
end


function solve_big(V_C, V_L, VH_LM, VXC_LM, D2, S, nspin, lmax, lmaxrho, funlist, lm_dict)

    VLM = zeros(size(V_C))

    #    println("sub abs VXC_LM ",sum(abs.( VXC_LM)))

    Nsmall = size(V_C)[1]

    c=length(lm_dict)
    Hbig = zeros(Nsmall * c, Nsmall * c)
    Sbig = zeros(Nsmall * c, Nsmall * c)

    N = Nsmall * c

    VECTS_BIG = zeros(Float64, N,N,nspin)
    VALS_BIG = zeros(Float64, N,nspin)

#    println("lmax $lmax ")
    big_code = Dict()

    for spin = 1:nspin
        Hbig .= 0.0
        Sbig .= 0.0
        #println("prepare mat")
        for l = 0:(lmax)
            for m = -l:l
                ind1 = (1:Nsmall) .+ Nsmall*lm_dict[(l,m)]
                for l2 = 0:(lmax)
                    for m2 = -l2:l2
                        ind2 = (1:Nsmall) .+ Nsmall*lm_dict[(l2,m2)]

                        #println("core")
                        if l == l2 && m == m2
                            Hbig[ind1,ind2] += V_C + V_L*l*(l+1)
                            Hbig[ind1,ind2] += D2
                            Sbig[ind1,ind2] += S                            
                            #                            println("add S $l $m  $l2 $m2")
                        end

                        

                        #println("more")
                        if funlist != :hydrogen  #VHART AND VXC_LM
                            VLM .= 0.0
                            for lr = 0:2:lmaxrho
                                for mr = -lr:lr
                                    gcoef = real_gaunt_dict[(lr,mr,l,m,l2,m2)]
                                    VLM += gcoef * (4*pi* (@view VH_LM[:,:,lr+1,lr+mr+1]) + (@view VXC_LM[:,:,spin,lr+1,lr+mr+1]))
                                    #t = gcoef * (4*pi*VH_LM[:,:,lr+1,lr+mr+1] + VXC_LM[:,:,spin,lr+1,lr+mr+1])
                                    #if maximum(abs.(t)) > 1e-10
                                    #    Hbig[ind1,ind2] +=  t
                                    #end
                                    #                                    temp = sum(abs.(gcoef * (4*pi*VH_LM[:,:,lr+1,lr+mr+1] + VXC_LM[:,:,spin,lr+1,lr+mr+1])))
                                    #                                    if abs(temp) > 1e-10
                                    #                                        println("ham sum $lr $mr  $l $m    $l2 $m2 ", sum((gcoef * (4*pi*VH_LM[:,:,lr+1,lr+mr+1] + VXC_LM[:,:,spin,lr+1,lr+mr+1]))))
                                    #                                    end
                                end
                            end
                            Hbig[ind1,ind2] +=  VLM
                            
                        end
#                        if sum(abs.(Hbig[ind1,ind2])) > 1e-10
#                            println("HAM $spin $l $m $l2 $m2 ", sum(abs.(Hbig[ind1,ind2])))
#                        end
                        #println("eigen")
                        #                        vals, vects = eigen(D2 + V + VLM, S)
                        #                        VECTS[:,:,spin, l+1, l+1+m] = vects
                        #                        VALS[:,spin, l+1, l+1+m] = vals
                    end
                end
            end
            
        end
        #        println("sum check ", sum(Hbig - Hbig'), " ", sum(Sbig - Sbig'))
        #        println("eigen big")

        #println("herm")
        begin
            Hh = Hermitian(Hbig)
            Sh = Hermitian(Sbig)
        end
        
        #println("eig")
        vals, vects = eigen(Hh, Sh)

        println("stuff")
        begin
            VALS_BIG[:,spin] = vals
            VECTS_BIG[:,:,spin] =vects
            
            COUNT = zeros(lmax+1, 2*lmax+1)
            
            Sv = Sh*vects
        end

        #println("sort vals")
        for n = 1:(N-1)
            tmax = 0.0
            l_m = 0
            m_m = 0
            for l = 0:(lmax)
                for m = -l:l
                    ind = (1:Nsmall) .+ Nsmall*lm_dict[(l,m)]
                    t = sum(abs.( real.(vects[ind,n] .* conj(Sv[ind,n])))) - COUNT[l+1, m+l+1]*0.001
                    if t > tmax
                        tmax = t
                        l_m = l
                        m_m = m
                    end
                end
            end
            COUNT[l_m+1, m_m+l_m+1] += 1.0
            
            #println("val $n  $spin   $l_m $m_m  $(vals[n])")
            if ! ((spin,l_m, m_m) in keys(big_code))
                big_code[(spin,l_m, m_m)] = Int64[]
                #                println("add big_code ", (spin,l_m, m_m))
            end
            push!(big_code[(spin,l_m, m_m)], n)
            #            println("spin assign $n $(VALS_BIG[n,spin]) to $((spin,l_m, m_m))")
        end
        #        println()
        
        #        println("vals spin $spin")
        #        println(vals[1:12])
        #        println("Hh")
        #        println(Hh)
        #        println("Sh")
        #        println(Sh)
        
    end
    
    return VALS_BIG, VECTS_BIG, big_code
    
end

function getsigma(rho, drho, nspin, r, theta)
    if nspin == 1
        sigma = zeros(size(drho)[1])
        @tturbo for i = 1:length(r)
            sigma[i] += drho[i,1,1]^2 #+ r[i]^(-2) * drho[i,1,2]^2
            sigma[i] +=  r[i]^(-2) * ( drho[i,1,2]^2 + drho[i,1,3]^2 )
        end
        
    elseif nspin == 2
        sigma = zeros(3,size(drho)[1])
        @turbo for i = 1:length(r)
            sigma[1,i] += drho[i,1,1]^2
            sigma[3,i] += drho[i,2,1]^2
            sigma[2,i] += drho[i,1,1]*drho[i,2,1]

            sigma[1, i] +=  r[i]^(-2) * ( drho[i,1,2]^2 + drho[i,1,3]^2 )
            sigma[3, i] +=  r[i]^(-2) * ( drho[i,2,2]^2 + drho[i,2,3]^2 )
            sigma[2, i] +=  r[i]^(-2) * ( drho[i,1,2]*drho[i,2,2] + drho[i,1,3]*drho[i,2,3])
        end
    else
        println("nspin err $nspin ")
    end
    return sigma #, drho_temp
end
function EXC_get_ve(nmean, n, funlist, drho, theta, gga, r)

    nspin = size(n)[2]

#    println("nspin $nspin")
    if gga
        sigma = getsigma(n, drho, nspin, r, theta)    #, drho_temp
    end

    if nspin == 1
        vsigma = zeros(size(n)[1], 1)        
    elseif nspin ==2
        vsigma = zeros(size(n)[1], 3)
    end

    vrho = zeros(size(n)[1], nspin)
    erho = zeros(size(n)[1], 1) 

    rho = collect(n')
   
    for (c_fun,fun) in enumerate(funlist)
        if gga
            ret = evaluate(fun, rho=rho, sigma=collect(sigma) )
            vsigma += collect(ret.vsigma)'
        else
            ret = evaluate(fun, rho=rho)
        end

        vrho += ret.vrho'
        erho += ret.zk[:]
        
    end

    ns = sum(nmean, dims=2)
    for i = 1:size(n,1)
        if ns[i] < 1e-5
            vsigma[i,:] *= cutoff_fn( abs(log(ns[i])), abs(log(1e-5)),abs(log(1e-7)))
        end
    end

    
    return vrho, vsigma, erho
    
end


#=function cutoff_fn(num, min_c, max_c)

#    if num < 1e-4
#        return 0.0
#    end
    if num < min_c
        return 1.0
    elseif num > max_c
        return 0.0
    else
        t = (num - min_c)/(max_c-min_c)
        return 1.0 - 10.0 * t^3 + 15.0 *  t^4  - 6.0 * t^5
    end
end
=#

function cutoff_fn(num, min_c, max_c)
    
    t = (num - min_c)/(max_c-min_c)
    return max.(min.(1.0 - 10.0 * t^3 + 15.0 *  t^4  - 6.0 * t^5, 1.0), 0.0)
    
end

function EXC_gal( n, funlist, drho, ddrho_omega, vrho, vsigma, dvsigma_theta, dvsigma_phi, theta, gga, r, g, invS, N, M)

    #    println("EXC ", sum(abs.(drho)), " ", gga)
    nspin = size(n)[2]

#    println("mem")
    begin
        v = zeros(size(n)[1], nspin)
        e = zeros(size(n)[1])
    end

    #println("get sig")
    if gga
        println("pre sigma ", [n[1], drho[1]])
        sigma = getsigma(n, drho, nspin, r, theta)  #, drho_temp
    end

    #println("sigma ", sigma[1])
    #println("terms")
    if gga
        if nspin == 1
            arr =  r .* drho[:,1,1] .* vsigma #* sqrt(4*pi)
            rep = get_gal_rep(arr[:], g, N=N)
            my_deriv = gal_rep_to_rspace(rep, g, M=M, deriv=1)
            v += (-2.0*r.^-1 .* my_deriv +  -2.0 *r.^-1 .* drho[:,1,1] .* vsigma  )   

            v += -2.0 * r.^(-2) .* ( dvsigma_theta .* drho[:,1, 2] + dvsigma_phi .* drho[:,1, 3] + vsigma .* ddrho_omega[:,1] )  

            println("asdf $(drho[1]) $(vsigma[1]) $(my_deriv[1])")
            
        elseif nspin ==2


            v[:,1] += - 2.0 * r[:].^(-1) .*  ( vsigma[:,1] .* drho[:,1,1] + 0.5* drho[:,2,1].*vsigma[:,2])  
            v[:,2] += - 2.0 * r[:].^(-1) .*  ( vsigma[:,3] .* drho[:,2,1] + 0.5* drho[:,1,1].*vsigma[:,2])  

            
            arr1 =  r .* (drho[:,1,1] .* vsigma[:,1] + 0.5*vsigma[:,2].* drho[:,2,1])
            rep1 = get_gal_rep(arr1, g, N=N)
            my_deriv1 = gal_rep_to_rspace(rep1, g, M=M, deriv=1) 
            
            arr2 =  r .* (drho[:,2,1] .* vsigma[:,3] + 0.5*vsigma[:,2].* drho[:,1,1])
            rep2 = get_gal_rep(arr2, g, N=N)
            my_deriv2 = gal_rep_to_rspace(rep2, g, M=M, deriv=1) 


            v[:,1] +=  (-2.0*r.^-1 .* my_deriv1 )
            v[:,2] +=  (-2.0*r.^-1 .* my_deriv2 )

            v[:,1] += -2.0 * r.^(-2) .* ( dvsigma_theta[:,1] .* drho[:,1, 2] + dvsigma_phi[:,1] .* drho[:,1, 3] + vsigma[:,1] .* ddrho_omega[:,1] )  
            v[:,2] += -2.0 * r.^(-2) .* ( dvsigma_theta[:,3] .* drho[:,2, 2] + dvsigma_phi[:,3] .* drho[:,2, 3] + vsigma[:,3] .* ddrho_omega[:,2] )  
            
            v[:,1] += -1.0 * r.^(-2) .* ( dvsigma_theta[:,2] .* drho[:,2, 2] + dvsigma_phi[:,2] .* drho[:,2, 3] + vsigma[:,2] .* ddrho_omega[:,2] )  
            v[:,2] += -1.0 * r.^(-2) .* ( dvsigma_theta[:,2] .* drho[:,1, 2] + dvsigma_phi[:,2] .* drho[:,1, 3] + vsigma[:,2] .* ddrho_omega[:,1] )  

        end
        


    else
        println("EXC_gal nspin $nspin")
    end
    

    v +=  vrho

    return v

end


function dft(; fill_str = missing, g = missing, N = -1, M = -1, Z = 1.0, niters = 50, mix = 0.5, mixing_mode=:pulay, exc = missing, lmax = missing, conv_thr = 1e-7, lmaxrho = 0, mix_lm = false, exx = 0.0, VECTS=missing)

    if M == -1
        M = g.M
    end
    if N == -1
        N = g.N
    end

    
    
   # println("time prepare")
    Z, nel, filling, nspin, lmax, V_C, V_L, D1, D2, S, invsqrtS, invS, VECTS_start, VALS, funlist, gga, LEB, R, gbvals2, nmax = prepare(Z, fill_str, lmax, exc, N, M, g, lmaxrho)
    if ismissing(VECTS)
        VECTS = VECTS_start
    end

    lm_dict = Dict()
    dict_lm = Dict()
    c=0
    for l = 0:(lmax)
        for m = -l:l
            lm_dict[(l,m)] = c
            c+=1
            dict_lm[c] = (l,m)
        end
    end

    
    println("lmaxrho $lmaxrho")
    
    #println("vChebyshevDFT.Galerkin.do_1d_integral(VECTS[:,1,1,1], g) ", do_1d_integral(real.(VECTS[:,1,1,1,1]).^2, g))
    
    #println("initial get rho time")
    rho_R2, rho_dR, rho_rs_M, MP, drho_rs_M_LM,psipsi = get_rho(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, true, exx, nmax)
#    println("MP")
#    println(MP)

    #   println("done rho")
    #    println("rho_R2 ", rho_R2[1])
    println("ChebyshevDFT.Galerkin.do_1d_integral(rho[:,1,1,1], g) ", do_1d_integral(rho_R2[:,1,1,1], g))

    #    println("M $M size rho_rs_M ", size(rho_rs_M))
    
    #    println("start rho ", rho[1:5])
    
    VALS_1 = deepcopy(VALS)

    VH_LM = zeros(N-1,N-1,lmaxrho+1, lmaxrho*2+1)
    VX_LM = zeros(N-1,N-1,nspin, lmaxrho+1, lmaxrho*2+1)
    VXC_LM = zeros(N-1,N-1,nspin, lmaxrho+1, lmaxrho*2+1)

    vxc_tp = zeros(M+1,nspin, lmaxrho+1, lmaxrho*2+1)
    VSIGMA_tp = zeros(M+1,3, lmaxrho+1, lmaxrho*2+1)
                   
    println("iters")
#    H1 = missing
#    H2 = missing
    for iter = 1:niters

        VALS_1[:,:,:,:] = VALS


        if funlist != :hydrogen
            NEL = sum(nel)
            ex_factor = exx * (NEL-1)/NEL + (1-exx)
#            println("NEL $nel exx $exx ex_factor $ex_factor")
            
            VH_LM = vhart_LM( sum(rho_dR, dims=2), D2, g, N, M, lmaxrho, lmax, MP*0.0, V_L,gbvals2, S, VECTS) #ex_factor*
            VH_LM0 = vhart_LM(0.0* sum(rho_dR, dims=2), D2, g, N, M, lmaxrho, lmax, MP, V_L,gbvals2, S, VECTS) #ex_factor* 
        else
            VH_LM = zeros(N-1,N-1,lmaxrho+1, lmaxrho*2+1)
        end

        if funlist != :none && funlist != :hydrogen
            VXC_LM, vxc_tp, exc_tp, VSIGMA_tp = vxc_LM( rho_rs_M, drho_rs_M_LM, g, M, N, funlist, gga, nspin, lmax, lmaxrho, LEB, R, invS, gbvals2)
        else
            VXC_LM = zeros(N-1,N-1,nspin, lmaxrho+1, lmaxrho*2+1)
        end

        if exx > 1e-12
            println("calculating exact exchange")
            VX_LM = vxx_LM( psipsi, D2, g, N, M, lmaxrho, lmax, MP, V_L,gbvals2, exx, nmax, nspin, filling, VECTS, S, sum(rho_dR, dims=2))

#            println("testVX/VH    ", VX_LM[1:3,1,1,:,:] +  VH_LM[1:3,1,1,:,:])
#            println("testVX/VH2   ", VX_LM[1:3,1,2,1,1])

#            println("testVX/VH spin2  ", VX_LM[:,:,2,:,:] +  VH_LM[:,:,1,:,:])

#            println("size ", size(VX_LM), " ", size(VH_LM))
#            println("test sum ", sum(abs.(VX_LM[:,:,1,:,:] + VH_LM[:,:,1,:,:])))
        end




        #        println("funlist $funlist")

        if mix_lm == false
            #println("solve small time")

#            println("subtract ", sum(abs.(VX_LM[:,:,1,1,1] + VH_LM[:,:,1,1])))
#            println("subtract ", sum(abs.(VX_LM[:,:,2,1,1] + VH_LM[:,:,1,1])))
#            println("before")
#            println("VC  ", VECTS[:,1,1,1,1]' * V_C * VECTS[:,1,1,1,1])
#            println("D2  ", VECTS[:,1,1,1,1]' * D2 * VECTS[:,1,1,1,1])
#            println("VH  ", VECTS[:,1,1,1,1]' * 4*pi*VH_LM[:,:,1,1] * VECTS[:,1,1,1,1])
#            println("VH0 ", VECTS[:,1,1,1,1]' * 4*pi*VH_LM0[:,:,1,1] * VECTS[:,1,1,1,1])
#            println("VX  ", VECTS[:,1,1,1,1]' * 4*pi*VX_LM[:,:,1,1,1] * VECTS[:,1,1,1,1])
#            println("ratio  ", VECTS[:,1,1,1,1]' * 4*pi*VX_LM[:,:,1,1,1] * VECTS[:,1,1,1,1] / (VECTS[:,1,1,1,1]' * 4*pi*VH_LM[:,:,1,1] * VECTS[:,1,1,1,1]))
#            println()
#            println("size VH ", size(VH_LM), " size VX ", size(VX_LM))
#            println("QQQQQQQQQQQQQQQQQQQQQ test 1 1 ", VECTS[:,1,1,1,1]' * 4*pi*VX_LM[:,:,1,1,1] * VECTS[:,1,1,1,1])
#            println("QQQQQQQQQQQQQQQQQQQQQ test 1 2 ", VECTS[:,1,1,1,1]' * 4*pi*VX_LM[:,:,1,1,1] * VECTS[:,2,1,1,1])
#            println("QQQQQQQQQQQQQQQQQQQQQ test 2 1 ", VECTS[:,2,1,1,1]' * 4*pi*VX_LM[:,:,1,1,1] * VECTS[:,1,1,1,1])
#            println("QQQQQQQQQQQQQQQQQQQQQ test 2 2 ", VECTS[:,2,1,1,1]' * 4*pi*VX_LM[:,:,1,1,1] * VECTS[:,2,1,1,1])
            
#            println()
#            println("RRRRRRRRRRRRRRRRRRRRR test 1 1 ", VECTS[:,1,1,1,1]' * 4*pi*VH_LM[:,:,1,1] * VECTS[:,1,1,1,1])
#            println("RRRRRRRRRRRRRRRRRRRRR test 1 2 ", VECTS[:,1,1,1,1]' * 4*pi*VH_LM[:,:,1,1] * VECTS[:,2,1,1,1])
#            println("RRRRRRRRRRRRRRRRRRRRR test 2 1 ", VECTS[:,2,1,1,1]' * 4*pi*VH_LM[:,:,1,1] * VECTS[:,1,1,1,1])
#            println("RRRRRRRRRRRRRRRRRRRRR test 2 2 ", VECTS[:,2,1,1,1]' * 4*pi*VH_LM[:,:,1,1] * VECTS[:,2,1,1,1])
#            println()
#            ttt = 4*pi*VH_LM[:,:,1,1] + 4*pi*VX_LM[:,:,1,1,1]
#            println("AAAAAAAAAAAAAAAAAAAAA test 1 1 ", VECTS[:,1,1,1,1]' * ttt * VECTS[:,1,1,1,1])
#            println("AAAAAAAAAAAAAAAAAAAAA test 1 2 ", VECTS[:,1,1,1,1]' * ttt * VECTS[:,2,1,1,1])
#            println("AAAAAAAAAAAAAAAAAAAAA test 2 1 ", VECTS[:,2,1,1,1]' * ttt * VECTS[:,1,1,1,1])
#            println("AAAAAAAAAAAAAAAAAAAAA test 2 2 ", VECTS[:,2,1,1,1]' * ttt * VECTS[:,2,1,1,1])
#            println("AAAAAAAAAAAAAAAAAAAAA test 1 3 ", VECTS[:,1,1,1,1]' * ttt * VECTS[:,3,1,1,1])
#            println("AAAAAAAAAAAAAAAAAAAAA test 2 3 ", VECTS[:,2,1,1,1]' * ttt * VECTS[:,3,1,1,1])
#            println("AAAAAAAAAAAAAAAAAAAAA test 3 3 ", VECTS[:,3,1,1,1]' * ttt * VECTS[:,3,1,1,1])


#            println()
#            half = 0.5
#            println("half vh               test 1 1 ", VECTS[:,1,1,1,1]' * half*4*pi*VH_LM[:,:,1,1] * VECTS[:,1,1,1,1])
#            println("half vh               test 1 2 ", VECTS[:,1,1,1,1]' * half*4*pi*VH_LM[:,:,1,1] * VECTS[:,2,1,1,1])
#            println("half vh               test 2 1 ", VECTS[:,2,1,1,1]' * half*4*pi*VH_LM[:,:,1,1] * VECTS[:,1,1,1,1])
#            println("half vh               test 2 2 ", VECTS[:,2,1,1,1]' * half*4*pi*VH_LM[:,:,1,1] * VECTS[:,2,1,1,1])
#            println("half vh               test 1 3 ", VECTS[:,1,1,1,1]' * half*4*pi*VH_LM[:,:,1,1] * VECTS[:,3,1,1,1])
#            println("half vh               test 2 3 ", VECTS[:,2,1,1,1]' * half*4*pi*VH_LM[:,:,1,1] * VECTS[:,3,1,1,1])
#            println("half vh               test 3 3 ", VECTS[:,3,1,1,1]' * half*4*pi*VH_LM[:,:,1,1] * VECTS[:,3,1,1,1])
            
#            println("VXC ", VECTS[:,1,1,1,1]' * VXC_LM[:,:,1,1,1] * VECTS[:,1,1,1,1])            
#            println("test xxxxxxxx ", sum(abs.( VH_LM[:,:,1,1]  + VH_LM0[:,:,1,1]*ex_factor + VX_LM[:,:,1,1,1] - 0.5*(VH_LM[:,:,1,1]  + VH_LM0[:,:,1,1]))))

            solve_small(V_C, V_L, VH_LM  + VH_LM0*ex_factor, VXC_LM, VX_LM, D2, S, nspin, lmax, lmaxrho, funlist, VECTS, VALS, exx)

            #            solve_small(V_C, V_L, VH_LM  + VH_LM0*sqrt(2), VXC_LM, VX_LM, D2, S, nspin, lmax, lmaxrho, funlist, VECTS, VALS, exx)

#            solve_small(V_C, V_L, 0.5*(VH_LM  + VH_LM0) , VXC_LM, 0.0*VX_LM, D2, S, nspin, lmax, lmaxrho, funlist, VECTS, VALS, exx)

#            println("ex_factor $ex_factor")
#            solve_small(V_C, V_L, VH_LM  + VH_LM0*ex_factor + VX_LM[:,:,1,:,:][:,:,:,:] , VXC_LM, 0.0*VX_LM, D2, S, nspin, lmax, lmaxrho, funlist, VECTS, VALS, exx)
#            t1 = deepcopy(0.5*VH_LM  + VH_LM0*ex_factor)
#            t2 = deepcopy(VH_LM  + VH_LM0*ex_factor + VX_LM[:,:,1,:,:][:,:,:,:])
#            println("test yyyyyyyyy re ", sum(abs.(real.(t1 - t2))))
#            println("test yyyyyyyyy im ", sum(abs.(imag.(t1 - t2))))
            #            solve_small(V_C, V_L, 0.5*VH_LM  + VH_LM0*ex_factor , VXC_LM, 0.0*VX_LM, D2, S, nspin, lmaxx, lmaxrho, funlist, VECTS, VALS, exx)
#            solve_small(V_C, V_L, t2 , VXC_LM, 0.0*VX_LM, D2, S, nspin, lmax, lmaxrho, funlist, VECTS, VALS, exx)
#            println("VALS ", VALS[1:3])
#            println("sumV ", sum(abs.(VECTS[:,1,1,1,1])))
#            solve_small(V_C, V_L, t1 , VXC_LM, 0.0*VX_LM, D2, S, nspin, lmax, lmaxrho, funlist, VECTS, VALS, exx)
#            println("VALS ", VALS[1:3])
#            println("sumV ", sum(abs.(VECTS[:,1,1,1,1])))
#            println("test AFTER H1 H2 ", sum(abs.(H1 - H2)))
#            v1,vects1 = eigen(H1, S)
#            v2,vects2 = eigen(H2, S)
#            println("sum abs vals ", sum(abs.(v1-v2)))
#            println("sum abs vects ", sum(abs.(vects1-vects2)))
            
#            println("after")
#            println("VC  ", VECTS[:,1,1,1,1]' * V_C * VECTS[:,1,1,1,1])
#            println("D2  ", VECTS[:,1,1,1,1]' * D2 * VECTS[:,1,1,1,1])
#            println("VH  ", VECTS[:,1,1,1,1]' * 4*pi*VH_LM[:,:,1,1] * VECTS[:,1,1,1,1])
#            println("VX  ", VECTS[:,1,1,1,1]' * 4*pi*VX_LM[:,:,1,1,1] * VECTS[:,1,1,1,1])
#            println("ratio  ", VECTS[:,1,1,1,1]' * 4*pi*VX_LM[:,:,1,1,1] * VECTS[:,1,1,1,1] / (VECTS[:,1,1,1,1]' * 4*pi*VH_LM[:,:,1,1] * VECTS[:,1,1,1,1]))

#            println("VXC ", VECTS[:,1,1,1,1]' * VXC_LM[:,:,1,1,1] * VECTS[:,1,1,1,1])            
        else
            #println("solve big time")            
            VALS_BIG, VECTS_BIG, big_code = solve_big(V_C, V_L, VH_LM, VXC_LM, D2, S, nspin, lmax, lmaxrho, funlist, lm_dict)
        end
 #       VALS_BIG, VECTS_BIG, big_code = solve_big(V_C, V_L, VH_LM, VXC_LM, D2, S, nspin, lmax, lmaxrho, funlist, lm_dict)


#        rho_R2_new, rho_dR_new, rho_rs_M_new, MP_new, drho_rs_M_LM = get_rho(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, gga)
#        rho_R2_newB, rho_dR_newB, rho_rs_M_newB, MP_newB  = get_rho_big(VALS, VALS_BIG, VECTS_BIG, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, D2, lm_dict, dict_lm, big_code) 

        
#        println("big_eig")
#        display_eigs(VALS, nspin, lmax)
#        println()

        #println("SIZE VECTS ", size(VECTS))
        #println("SIZE VECTS BIG ", size(VECTS_BIG))
        #        return VECTS_BIG, VALS_BIG, big_code
        
        #        display_eigs(VALS, nspin, lmax)
        
        if mix_lm == false
            #println("get rho small time")
            rho_R2_new, rho_dR_new, rho_rs_M_new, MP_new, drho_rs_M_LM, psipsi = get_rho(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, gga, exx, nmax)
        else
            #println("get rho big time")
            rho_R2_new, rho_dR_new, rho_rs_M_new, MP_new, drho_rs_M_LM  = get_rho_big(VALS, VALS_BIG, VECTS_BIG, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, D2, lm_dict, dict_lm, big_code, gga) 
        end

        #println("ChebyshevDFT.Galerkin.do_1d_integral(rho[:,1,1,1], g) ", do_1d_integral(rho_R2_new[:,1,1,1], g))

        
#        VALS_BIG, VECTS_BIG, big_code = solve_big(V_C, V_L, VH_LM, VXC_LM, D2, S, nspin, lmax, lmaxrho, funlist, lm_dict)
#        rho_R2_newB, rho_dR_newB, rho_rs_M_newB, MP_newB, drho_rs_M_LMB  = get_rho_big(VALS, VALS_BIG, VECTS_BIG, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, D2, lm_dict, dict_lm, big_code) 
#        println("sum abs ", sum(abs.(rho_R2_new - rho_R2_newB)), " " , sum(abs.(drho_rs_M_LM - drho_rs_M_LMB)), " ", sum(abs.(rho_rs_M_new - rho_rs_M_newB)))
#        println("d ", drho_rs_M_LM[1] , " " ,  drho_rs_M_LMB[1])
        #        println("small_eig")
#        display_eigs(VALS, nspin, lmax)
#        println()
        
#        rho_R2_newB, rho_dR_newB, rho_rs_M_newB, MP_newB  = get_rho_big(VALS, VALS_BIG, VECTS_BIG, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, D2, lm_dict, dict_lm, big_code) 

#        println("big_eig")
#        display_eigs(VALS_BIG, nspin, lmax)
#        println()

        
        #mix
        #println("mix time")
        begin 
            rho_R2 = rho_R2_new * mix + rho_R2 *(1-mix)
            rho_dR = rho_dR_new * mix + rho_dR * (1-mix)
            rho_rs_M = rho_rs_M_new * mix + rho_rs_M * (1-mix)

            MP = MP_new*mix + MP_new*(1-mix) #this is approximation for higher multipoles.

                
            eigval_diff = maximum(abs.(filling.*(VALS - VALS_1)))
        end
        println("iter $iter eigval_diff $eigval_diff ")

        
        #if maximum(abs.(filling.*(VALS - VALS_1))) < conv_thr
        #    break
        #end
        
        display_eigs(VALS, nspin, lmax)
        println()
            
        
    end
    println("done iters")
    
    display_eigs(VALS, nspin, lmax)
    println()

    #    println("size rho_rs_M", size(rho_rs_M))
    return VALS, VECTS, rho_R2, VH_LM , VXC_LM, rho_rs_M, drho_rs_M_LM, vxc_tp, VSIGMA_tp
    
end #end dft




end #end module
