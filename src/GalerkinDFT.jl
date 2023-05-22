

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

    
    LEB = makeleb(lmaxrho*2+1, lmax=lmaxrho)

    R = get_r_grid(g, M=M)
    R = R[2:M+2]
    return Z, nel, filling, nspin, lmax, V_C, V_L,D1, D2, S, invsqrtS, invS, VECTS, VALS, funlist, gga, LEB, R
    
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

function get_rho(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, gga)
    #    println("get rho")
    #rho = zeros(N-1, nspin, lmax+1, 2*lmax+1)
    rho_rs_M_R2 = zeros(M+1, nspin)
    rho_rs_M_R2_LM2 = zeros(M+1, nspin, lmax+1, 2*lmax+1)
    drho_rs_M_R2_LM2 = zeros(M+1, nspin, lmax+1, 2*lmax+1)
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
                    t = gal_rep_to_rspace(VECTS[:, n, spin, l+1,l+1+m], g, M=M)
                    rho_rs_M_R2[:,spin] += real(t.*conj(t)) * fillval
                    rho_rs_M_R2_LM2[:,spin, l+1, m+1+l] += real(t.*conj(t)) * fillval

                    if true
                        dt = gal_rep_to_rspace(VECTS[:, n, spin, l+1,l+1+m], g, M=M, deriv=1)
                        drho_rs_M_R2_LM2[:,spin, l+1, m+1+l] += real( t .* conj.(dt) + dt .* conj.(t)  ) * fillval
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
                        if true
                            drho_rs_M_R2_LM[:,spin, lr+1,mr+lr+1] += gcoef * drho_rs_M_R2_LM2[:,spin, l+1, m+1+l]
                        end
                    end
                end
            end
        end
    end

    
    rho_rs_M_R2_LM = rho_rs_M_R2_LM * sqrt(4 * pi)
    drho_rs_M_R2_LM = drho_rs_M_R2_LM * sqrt(4 * pi)
    

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
                if true
                    drho_rs_M_LM[:,spin, lr+1, mr+lr+1]  += e
                end
                
                #                drho_gal_R2_LM[:,spin, lr+1, mr+lr+1] += D1*a
                #                    end
                #                end
            end
        end
    end                    
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
    
    return rho_gal_R2_LM, rho_gal_dR_LM, rho_rs_M_LM, MP, drho_rs_M_LM

end



function get_rho_big(VALS, VALS_big, VECTS_big, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, D2, lm_dict,dict_lm, big_code)

    #    println("SIZE VB ", size(VECTS_big))

    Nbig = size(VECTS_big)[1]

    #    println("Nbig $Nbig M $M")
    
    #rho_rs_M_R2 = zeros(M+1, nspin)
    rho_rs_M_R2_LM2 = zeros(M+1, nspin, lmax+1, 2*lmax+1, lmax+1, 2*lmax+1)
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
                        t1 = gal_rep_to_rspace(VECTS_big[(1:(N-1)).+(c1-1)*(N-1), n, spin], g, M=M)
                        for c2 = 1:length(lm_dict)
                            (ll2,mm2) = dict_lm[c2]
                            t2 = gal_rep_to_rspace(VECTS_big[(1:(N-1)).+(c2-1)*(N-1), n, spin], g, M=M)
                            
                            #rho_rs_M_R2[:,spin] += real(t.*conj(t)) * fillval
                            rho_rs_M_R2_LM2[:,spin, ll1+1, mm1+1+ll1, ll2+1, mm2+1+ll2] += real(t1.*conj(t2)) * fillval
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

    rho_rs_M_LM = zeros(M+1, nspin, lmaxrho+1, lmaxrho*2+1)
    rho_gal_R2_LM = zeros(N-1, nspin, lmaxrho+1, lmaxrho*2+1)
    rho_gal_dR_LM = zeros(N-1, nspin, lmaxrho+1, lmaxrho*2+1)

    rho_gal_mulipole_LM = zeros(N-1, nspin, lmaxrho+1, lmaxrho*2+1)
    
    for lr = 0:lmaxrho
        for mr = -lr:lr
            for l1 = 0:lmax
                for m1 = -l1:l1
                    for l2 = 0:lmax
                        for m2 = -l2:l2
                            gcoef = real_gaunt_dict[(lr,mr,l1,m1,l2,m2)]
                            for spin = 1:nspin
                                rho_rs_M_R2_LM[:,spin, lr+1,mr+lr+1] += gcoef * rho_rs_M_R2_LM2[:,spin, l1+1, m1+1+l1, l2+1, m2+1+l2,]
                            end
                        end
                    end
                end
            end
        end
    end

    
    rho_rs_M_R2_LM = rho_rs_M_R2_LM * sqrt(4 * pi)

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

    for spin = 1:nspin
        for lr = 0:lmaxrho
            for mr = -lr:lr
                #                for l = 0:lmax
                #                    for m = -l:l
                #                        println("before")
                #                        println("size(rho_rs_M_R2_LM[:,spin, lr+1, mr+lr+1] ", size(rho_rs_M_R2_LM[:,spin, lr+1, mr+lr+1]))
                a,b,c,d =  get_rho_gal(rho_rs_M_R2_LM[:,spin, lr+1, mr+lr+1], g,N=N, invS=invS, l = lr)
                #                        println("size ret ", size(ret))
                rho_gal_R2_LM[:,spin, lr+1, mr+lr+1] += a
                rho_gal_dR_LM[:,spin, lr+1, mr+lr+1 ] += b
                rho_rs_M_LM[:,spin, lr+1, mr+lr+1]  += c
                rho_gal_mulipole_LM[:,spin, lr+1, mr+lr+1]  += d
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
    rho_gal_mulipole_LM = rho_gal_mulipole_LM  / (g.b-g.a) * 2
    

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
    
    return rho_gal_R2_LM, rho_gal_dR_LM, rho_rs_M_LM, MP

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

function vhart_LM(rho_dR, D2, g, N, M, lmaxrho, lmax, MP, V_L)


    #    println("size rho_dR ", size(rho_dR))
    #    println("lmax rho ", lmaxrho)
    loopmax = min(lmax*2, lmaxrho)
    #loopmax = lmaxrho
    VH_LM = zeros(N-1, N-1, lmaxrho+1, 2*lmaxrho+1)
    #    println("size VH_LM ", size(VH_LM))
    
    for l = 0:loopmax
        for m = -l:l
            #            println("$l $m size vhart ", size(rho_dR), " ", size(rho_dR[:,1, l+1, m+l+1]))
            #            println(size(vhart(rho_dR[:,1,l+1, m+l+1], D2, V_L, g, M, l, m, MP)))

            VH_LM[:, :, l+1, m+l+1] = vhart(rho_dR[:,1,l+1, m+l+1], D2, V_L, g, M, l, m, MP)
        end
    end

    return VH_LM
    
end


function vhart(rhor2, D2, V_L, g, M, l, m, MP)

    vh_tilde = (D2 + l*(l+1)*V_L) \ rhor2

    vh_tilde = vh_tilde /(4*pi)*sqrt(pi)
    
    #    println("MP ", size(MP))
    #    println(MP)
    #    println()
    

    vh_mat = get_vh_mat(vh_tilde, g, l, m, MP, M=M)
    
    #    function vh_f(r)
    #        if r < 1e-7
    #            r = 1e-7
    #        end
    #        return gal_rep_to_rspace(r, vh_tilde, g) / r  .+ MP[l+1, l+m+1]/g.b^(l+1) * sqrt(pi)/(2*pi) / (2*l+1)
    #    end
    
    #    return vh_mat, vh_f

    return vh_mat
    
end

function vxc_LM(rho_rs_M, drho_rs_M, g, M, N, funlist, gga, nspin, lmax, lmaxrho, LEB, R, invS)

    #get vxc in r space

    rho = zeros( M+1, nspin, LEB.N)
    drho = zeros(M+1, nspin,3, LEB.N)
    #    drho_tp = zeros(M+1, nspin,2)
    ddrho = zeros(M+1, nspin,3, LEB.N)    
    
    VSIGMA_tp = zeros( M+1, nspin, LEB.N)
    VRHO_tp = zeros( M+1, nspin, LEB.N)
    ERHO_tp = zeros( M+1, nspin, LEB.N)
    

    
    loopmax = min(lmax*2, lmaxrho)

    #get rho and then VXC in theta / phi / r space
    for ntp = 1:LEB.N
        for spin = 1:nspin
            for l = 0:loopmax
                for m = -l:l
                    rho[:, spin, ntp] += rho_rs_M[:,spin, l+1, m+l+1] * LEB.Ylm[(l,m)][ntp]
                    if gga
                        drho[:, spin, 1, ntp] += drho_rs_M[:,spin, l+1, m+l+1] * LEB.Ylm[(l,m)][ntp]

                        #angular deriv
                        drho[:, spin,2, ntp] += drho_rs_M[:,spin, l+1, m+l+1] * LEB.dYlm_theta[(l,m)][ntp]
                        drho[:, spin,3, ntp] += drho_rs_M[:,spin, l+1, m+l+1] * LEB.dYlm_phi[(l,m)][ntp]

                        # 2nd angular derivs
                        ddrho[:, spin,2, ntp] += drho_rs_M[:,spin, l+1, m+l+1] * LEB.ddYlm_theta[(l,m)][ntp]
                        ddrho[:, spin,3, ntp] += drho_rs_M[:,spin, l+1, m+l+1] * LEB.ddYlm_phi[(l,m)][ntp]

                        
                        
                    end
                end
            end
        end
        
        #        println("$ntp rho_rs_M ", rho[1:5,1]* sqrt(4*pi), " xxxxxxxxxxxxxxx")

        theta = LEB.θ[ntp]
        
        if !ismissing(funlist) && funlist != :lda_internal
            for spin = 1:nspin
                vrho, vsigma, erho =  EXC_get_ve( (rho[:,spin,ntp]/sqrt(4*pi)  ), funlist, (drho[:,spin,:,ntp]/sqrt(4*pi)), theta, gga, R)
                VSIGMA_tp[:,spin,ntp] = vsigma #/ sqrt(4*pi)
                VRHO_tp[:,spin,ntp] = vrho #/ sqrt(4*pi)
                ERHO_tp[:,spin,ntp] = erho
#                println("vrho[1:3] ", vrho[1:3])
                
            end
        else
            for spin = 1:nspin
                v = v_LDA.(rho[:, spin, ntp] * sqrt(4*pi)/ (4*pi))
#                println("v[1:3] ", v[1:3])
                VRHO_tp[:,spin,ntp] = v
            end
        end

    end

    #now transform vsigma to LM space
    VSIGMA_LM = zeros(M+1, nspin, lmaxrho+1, 2*lmaxrho+1)
    if gga
        for spin = 1:nspin
            for l = 0:lmaxrho
                for m = -l:l
                    for r = 1:M+1
                        VSIGMA_LM[r, spin, l+1, l+m+1] =  4*pi*sum(LEB.Ylm[(l,m)][:] .* VSIGMA_tp[r, spin, :] .* LEB.w)
                    end
                end
            end
        end
    end
    
    #now transform back derivatives wrt t, pi
    dvsigma_theta = zeros(M+1)
    dvsigma_phi = zeros(M+1)

    VXC_tp = zeros( M+1, nspin, LEB.N)
    #    EXC_tp = zeros( M+1, nspin, LEB.N)

    for ntp = 1:LEB.N
        dvsigma_theta .= 0.0
        dvsigma_phi .= 0.0
        for spin = 1:nspin

            if gga
                for l = 0:loopmax
                    for m = -l:l
                        dvsigma_theta += VSIGMA_LM[:, spin, l+1, l+m+1] * LEB.dYlm_theta[(l,m)][ntp]
                        dvsigma_phi += VSIGMA_LM[:, spin, l+1, l+m+1] * LEB.dYlm_phi[(l,m)][ntp]
                    end
                end
            end
            theta = LEB.θ[ntp]
            if !ismissing(funlist) && funlist != :lda_internal
                v = EXC_gal( rho[:, spin, ntp] / sqrt(4*pi), funlist, (drho[:,spin,:,ntp]/sqrt(4*pi)), (ddrho[:,spin,:,ntp]/sqrt(4*pi)), VRHO_tp[:,spin, ntp], VSIGMA_tp[:, spin, ntp], dvsigma_theta, dvsigma_phi, theta, gga, R,  g, invS, N, M)
                VXC_tp[:,spin,ntp] += v
                #println("in $(VRHO_tp[1:3,spin, ntp]) out $(v[1:3])")
                
            else
                VXC_tp[:,spin,ntp] += VRHO_tp[:,spin,ntp]
            end
                
        end
    end
    
    
    
    #now transform to LM space again
    VXC_LM = zeros(M+1, nspin, lmaxrho+1, 2*lmaxrho+1)
    for spin = 1:nspin
        for l = 0:lmaxrho
            for m = -l:l
                for r = 1:M+1
                    VXC_LM[r, spin, l+1, l+m+1] =  4*pi*sum(LEB.Ylm[(l,m)][:] .* VXC_tp[r,spin,:] .* LEB.w)
                end
            end
        end
    end


    VXC_LM = VXC_LM #/ sqrt(4*pi)

    
    VXC_LM_MAT = zeros(N-1,N-1,nspin, lmaxrho+1, 2*lmaxrho+1)
    for spin = 1:nspin
        for l = 0:lmaxrho
            for m = -l:l
                VXC_LM_MAT[:,:,spin,l+1,l+m+1] = get_gal_rep_matrix_R(VXC_LM[:,spin,l+1,l+m+1], g, ; N = N)

            end
        end
    end
    EXC_tp = zeros(N-1,N-1,nspin, lmaxrho+1, 2*lmaxrho+1)
    
    return VXC_LM_MAT, VXC_tp, EXC_tp
    
end


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

function solve_small(V_C, V_L, VH_LM, VXC_LM, D2, S, nspin, lmax, lmaxrho, funlist, VECTS, VALS)

    VLM = zeros(size(V_C))

    #    println("sub abs VXC_LM ",sum(abs.( VXC_LM)))

    
    
    for spin = 1:nspin 
        for l = 0:(lmax)
            V = V_C + V_L*l*(l+1)
            for m = -l:l

                if funlist != :hydrogen  #VHART AND VXC_LM
                    VLM .= 0.0
                    for lr = 0:lmaxrho
                        for mr = -lr:lr
                            gcoef = real_gaunt_dict[(lr,mr,l,m,l,m)]
                            VLM += gcoef * (4*pi*VH_LM[:,:,lr+1,lr+mr+1] + VXC_LM[:,:,spin,lr+1,lr+mr+1])
                        end
                    end
                end
                
                #println("eigen")
                vals, vects = eigen(D2 + V + VLM, S)
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

    println("lmax $lmax ")
    big_code = Dict()
    for spin = 1:nspin
        Hbig .= 0.0
        Sbig .= 0.0
        for l = 0:(lmax)
            for m = -l:l
                ind1 = (1:Nsmall) .+ Nsmall*lm_dict[(l,m)]
                for l2 = 0:(lmax)
                    for m2 = -l2:l2
                        ind2 = (1:Nsmall) .+ Nsmall*lm_dict[(l2,m2)]

                        if l == l2 && m == m2
                            Hbig[ind1,ind2] += V_C + V_L*l*(l+1)
                            Hbig[ind1,ind2] += D2
                            Sbig[ind1,ind2] += S                            
                            #                            println("add S $l $m  $l2 $m2")
                        end

                        

                        
                        if funlist != :hydrogen  #VHART AND VXC_LM
                            VLM .= 0.0
                            for lr = 0:lmaxrho
                                for mr = -lr:lr
                                    gcoef = real_gaunt_dict[(lr,mr,l,m,l2,m2)]
                                    VLM += gcoef * (4*pi*VH_LM[:,:,lr+1,lr+mr+1] + VXC_LM[:,:,spin,lr+1,lr+mr+1])
                                    t = gcoef * (4*pi*VH_LM[:,:,lr+1,lr+mr+1] + VXC_LM[:,:,spin,lr+1,lr+mr+1])
                                    if maximum(abs.(t)) > 1e-10
                                        Hbig[ind1,ind2] +=  t
                                    end
                                    #                                    temp = sum(abs.(gcoef * (4*pi*VH_LM[:,:,lr+1,lr+mr+1] + VXC_LM[:,:,spin,lr+1,lr+mr+1])))
                                    #                                    if abs(temp) > 1e-10
                                    #                                        println("ham sum $lr $mr  $l $m    $l2 $m2 ", sum((gcoef * (4*pi*VH_LM[:,:,lr+1,lr+mr+1] + VXC_LM[:,:,spin,lr+1,lr+mr+1]))))
                                    #                                    end
                                end
                            end
                        end
                        if sum(abs.(Hbig[ind1,ind2])) > 1e-10
                            println("HAM $spin $l $m $l2 $m2 ", sum(abs.(Hbig[ind1,ind2])))
                        end
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
        Hh = Hermitian(Hbig)
        Sh = Hermitian(Sbig)
        save("data.$spin.jld", "Hh", Hh, "Sh", Sh)
        vals, vects = eigen(Hh, Sh)
        
        VALS_BIG[:,spin] = vals
        VECTS_BIG[:,:,spin] =vects

        COUNT = zeros(lmax+1, 2*lmax+1)

        Sv = Sh*vects
        
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

function EXC_get_ve(n, funlist, drho, theta, gga, r)

    if gga
        sigma = drho[:,1].^2 + r.^(-2) .* (  drho[:,2].^2  )
#        if abs(theta) > 1e-10
#            sigma += sin(theta)^(-2)* r.^(-2).*drho[:,3].^2
#        end
        
    end

    vsigma = zeros(length(n))
    vrho = zeros(length(n))
    erho = zeros(length(n))
    
    for (c_fun,fun) in enumerate(funlist)
        if gga
#            println("size n ", size(n))
#            println("size sigma ", size(sigma))            
            ret = evaluate(fun, rho=collect(n), sigma=sigma )
            vsigma += ret.vsigma[:]

            #vrho = ret.vrho
            #arr =  r .* drho[:,1] .* vsigma #* sqrt(4*pi)
            #rep = get_gal_rep(arr, g, N=N)
            #my_deriv = gal_rep_to_rspace(rep, g, M=M, deriv=1)
        else
            ret = evaluate(fun, rho=collect(n))
        end
        vrho += ret.vrho[:]
        erho += ret.zk[:]
        
    end
    
    return vrho, vsigma, erho
    
end


function EXC_gal(n, funlist, drho, ddrho, vrho, vsigma, dvsigma_theta, dvsigma_phi, theta, gga, r, g, invS, N, M)

    #    println("EXC ", sum(abs.(drho)), " ", gga)
    
    v = zeros(length(n))
    e = zeros(length(n))

    if gga
        sigma = drho[:,1].^2 + r.^(-2) .* (  drho[:,2].^2   )
#        if abs(theta) > 1e-10
#            sigma += sin(theta)^(-2)* r.^(-2).*drho[:,3].^2
#        end
    end

    #    for (c_fun,fun) in enumerate(funlist)
    #        println("c_fun $c_fun")
    if gga
        arr =  r .* drho[:,1] .* vsigma #* sqrt(4*pi)
        rep = get_gal_rep(arr, g, N=N)
        my_deriv = gal_rep_to_rspace(rep, g, M=M, deriv=1)

        v += (-2.0*r.^-1 .* my_deriv + -2.0 *r.^-1 .* drho[:,1] .* vsigma  )   #* sqrt(4*pi)

        #1
        #v[2:end] += - 4.0 * r[2:end].^(-1) .*  ( ( vsigma[2:end] .* drho[2:end,1]))
        #v += -2.0*calc_D((vsigma .* drho[:,1])  , n, D1, r)

        #2
        if abs(theta) > 1e-10
            v +=  -2.0* r.^(-2) .* ( dvsigma_theta .* drho[:, 2] +  vsigma .* ddrho[:, 2] + cos(theta)/sin(theta) * drho[:, 2] .* vsigma)

            
        #3
            #v +=  -2.0 / sin(theta)^2 * r.^(-2) .* ( dvsigma_phi .* drho[:, 3] + vsigma .* ddrho[:, 3])
        end
        
        #else
        #ret = evaluate(fun, rho=collect(n))
        #vrho = ret.vrho
        
    end            

    v +=  vrho
    
#    end
    
    return v
    
end


function dft(; fill_str = missing, g = missing, N = -1, M = -1, Z = 1.0, niters = 50, mix = 0.5, mixing_mode=:pulay, exc = missing, lmax = missing, conv_thr = 1e-7, lmaxrho = 0, mix_lm = false)

    if M == -1
        M = g.M
    end
    if N == -1
        N = g.N
    end

    
    
    println("prepare")
    Z, nel, filling, nspin, lmax, V_C, V_L, D1, D2, S, invsqrtS, invS, VECTS, VALS, funlist, gga, LEB, R = prepare(Z, fill_str, lmax, exc, N, M, g, lmaxrho)


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
    
    #    println("get rho")
    rho_R2, rho_dR, rho_rs_M, MP, drho_rs_M_LM = get_rho(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, gga)
    println("MP")
    println(MP)

    #   println("done rho")

    println("ChebyshevDFT.Galerkin.do_1d_integral(rho[:,1,1,1], g) ", do_1d_integral(rho_R2[:,1,1,1], g))

    #    println("M $M size rho_rs_M ", size(rho_rs_M))
    
    #    println("start rho ", rho[1:5])
    
    VALS_1 = deepcopy(VALS)

    VH_LM = zeros(N-1,N-1,lmaxrho+1, lmaxrho*2+1)
    VXC_LM = zeros(N-1,N-1,nspin, lmaxrho+1, lmaxrho*2+1)

    
    println("iters")
    for iter = 1:niters

        VALS_1[:,:,:,:] = VALS


        if funlist != :hydrogen
            VH_LM = vhart_LM( sum(rho_dR, dims=2), D2, g, N, M, lmaxrho, lmax, MP, V_L)
            VXC_LM, vxc_tp, exc_tp = vxc_LM( rho_rs_M, drho_rs_M_LM, g, M, N, funlist, gga, nspin, lmax, lmaxrho, LEB, R, invS)
        else
            VH_LM = zeros(N-1,N-1,lmaxrho+1, lmaxrho*2+1)
            VXC_LM = zeros(N-1,N-1,nspin, lmaxrho+1, lmaxrho*2+1)
        end

        #VH_LM = zeros(N-1,N-1,lmaxrho+1, lmaxrho*2+1)
        #VXC_LM = zeros(N-1,N-1,nspin, lmaxrho+1, lmaxrho*2+1)
        


        #        println("funlist $funlist")

        if mix_lm == false
            solve_small(V_C, V_L, VH_LM, VXC_LM, D2, S, nspin, lmax, lmaxrho, funlist, VECTS, VALS)
        else
            VALS_BIG, VECTS_BIG, big_code = solve_big(V_C, V_L, VH_LM, VXC_LM, D2, S, nspin, lmax, lmaxrho, funlist, lm_dict)
        end


        #println("SIZE VECTS ", size(VECTS))
        #println("SIZE VECTS BIG ", size(VECTS_BIG))
        #        return VECTS_BIG, VALS_BIG, big_code
        
        #        display_eigs(VALS, nspin, lmax)
        
        if mix_lm == false
            rho_R2_new, rho_dR_new, rho_rs_M_new, MP_new, drho_rs_M_LM = get_rho(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, gga)
        else
            println("big")
            rho_R2_new, rho_dR_new, rho_rs_M_new, MP_new  = get_rho_big(VALS, VALS_BIG, VECTS_BIG, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, D2, lm_dict, dict_lm, big_code) 
        end
        
        
        #mix
        rho_R2 = rho_R2_new * mix + rho_R2 *(1-mix)
        rho_dR = rho_dR_new * mix + rho_dR * (1-mix)
        rho_rs_M = rho_rs_M_new * mix + rho_rs_M * (1-mix)

        MP = MP_new*mix + MP_new*(1-mix) #this is approximation for higher multipoles.

        eigval_diff = maximum(abs.(filling.*(VALS - VALS_1)))
        println("iter $iter eigval_diff $eigval_diff ")

        display_eigs(VALS, nspin, lmax)
        println()
        
        if maximum(abs.(filling.*(VALS - VALS_1))) < conv_thr
            break
        end
        
    end
    println("done iters")
    
    display_eigs(VALS, nspin, lmax)
    println()

    #    println("size rho_rs_M", size(rho_rs_M))
    return VALS, VECTS, rho_R2, VH_LM , VXC_LM, rho_rs_M, drho_rs_M_LM
    
end #end dft




end #end module
