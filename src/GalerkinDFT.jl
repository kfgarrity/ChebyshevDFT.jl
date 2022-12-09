

module GalerkinDFT

using Polynomials
using SpecialPolynomials
using Base.Threads
using LinearAlgebra
using ForwardDiff
using QuadGK
using FastSphericalHarmonics
using ..Galerkin:get_gal_rep_matrix
using ..Galerkin:get_gal_rep_matrix_R
using ..Galerkin:get_rho_gal
using ..Galerkin:gal_rep_to_rspace
using ..Galerkin:get_vh_mat
using ..Galerkin:do_1d_integral

include("Atomlist.jl")

using ..UseLibxc:set_functional
using ..UseLibxc:EXC
using ..UseLibxc:EXC_sp
using ..UseLibxc:smooth
using ..UseLibxc:getVsigma

using ..LDA:v_LDA
using ..LDA:e_LDA

using ..LDA:v_LDA_sp
using ..LDA:e_LDA_sp


function choose_exc(exc, nspin)

        

    if !ismissing(exc)

        #convenience functions
        if (typeof(exc) == String && lowercase(exc) == "pbe") || (typeof(exc) == Symbol && exc == :pbe)
            exc = [:gga_x_pbe, :gga_c_pbe]
        elseif (typeof(exc) == String && lowercase(exc) == "pbesol") || (typeof(exc) == Symbol && exc == :pbesol)
            exc = [:gga_x_pbe_sol, :gga_c_pbe_sol]
        elseif (typeof(exc) == String && lowercase(exc) == "lda") || (typeof(exc) == Symbol && exc == :lda)
            exc = [:lda_x, :lda_c_vwn]
        elseif (typeof(exc) == String && lowercase(exc) == "vwn") || (typeof(exc) == Symbol && exc == :vwn)
            exc = [:lda_x, :lda_c_vwn]
        end    

        if (typeof(exc) == String && lowercase(exc) == "hydrogen") || (typeof(exc) == Symbol && exc == :hydrogen)
            funlist = :hydrogen
            println("Running hydrogen-like atom (no hartree, no exc)")
            gga = false
            return funlist, gga

        end
        
        funlist,gga = set_functional(exc, nspin=nspin)
    else
        println("Running default LDA (VWN)")
        funlist = :lda
        gga = false
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

    

function prepare(Z, fill_str, lmax, exc, N, M, g)

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
    
    nel, nspin, lmax = setup_filling(fill_str, lmax_init=lmax)

    funlist, gga = choose_exc(exc, nspin)

    l = 0


    function V(r)
        return -Z / r + 0.5*l*(l+1)/r^2
    end
    
    VC = []
    for ll = 0:(lmax+1)
        l = ll
        Vc = get_gal_rep_matrix(V, g, ; N = N, M = M)
        push!(VC, Vc)
    end

    VECTS = zeros(Complex{Float64}, N-1,N-1, nspin, lmax+1,2*lmax+1 )
    VALS = zeros(N-1,nspin, lmax+1,2*lmax+1 )


    D2 =-0.5* g.d2[1:N-1, 1:N-1] /  (b-a)^2*2^2
    S = g.s[1:N-1, 1:N-1]

    invsqrtS = inv(sqrt(S))
    invS = inv(S)
    
#    println("size VECTS ", size(VECTS))

    for ll = 0:lmax
        vals, vects = eigen(D2 + VC[ll+1], S)
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
  
    return Z, nel, nspin, lmax, VC, D2, S, invsqrtS, invS, VECTS, VALS, funlist, gga
    
end


function setup_filling(fill_str; lmax_init=0)

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
            d = FastSphericalHarmonics.sph_mode(l,m)

            if nspin == 1
                nel[1,d[1],d[2]] += num_el
            else
                nel[1,d[1],d[2]] += num_el/2.0
                nel[2,d[1],d[2]] += num_el/2.0
            end
        elseif length(t) == 5
            n,l,m,num_el_up, num_el_dn = t
            n = Int64(n)
            l = Int64(l)
            m = Int64(m)
            d = FastSphericalHarmonics.sph_mode(l,m)
            nel[1,d[1],d[2]] += num_el_up
            nel[2,d[1],d[2]] += num_el_dn
        end
    end

    return nel, nspin, lmax
    
end

function get_rho(VALS, VECTS, nel, nspin, lmax, N, M, invS, g, D2)

    rho = zeros(N-1, nspin, lmax+1, 2*lmax+1)
    filling = zeros(size(VALS))
    
    for spin = 1:nspin
        for l = 0:lmax
            for m = -l:l
                fillval = 2.0
                nleft = nel[spin, l+1, m+l+1]
                if nleft < 1e-20
                    nleft = 0.0
                    break
                end
                
                for n = 1:N-1
                    #println("add rho $spin $l $m $n ", fillval/nspin)
                    if nleft <= fillval/nspin + 1e-10

                        rho[:,spin, l+1, m+l+1] += sqrt(nleft) *VECTS[:, n, spin, l+1, m+l+1]
                        filling[n,spin,l+1, m+l+1] = nleft
                        break
                    else
                        nleft -= fillval / nspin
                        filling[n,spin,l+1, m+l+1] = fillval / nspin

                        #t = fillval/nspin* real( (VECTS[:, n, spin, l+1, m+l+1]).*conj(VECTS[:, n, spin, l+1, m+l+1]))
                        rho[:,spin, l+1, m+l+1] += sqrt(fillval/nspin)*VECTS[:, n, spin, l+1, m+l+1]
                    end
                end
                    
                
            end
        end
    end

#    println("rho")
#    println(rho)
    
    rho_gal = zeros(N-1, nspin, lmax+1, 2*lmax+1)
    rho_gal_R2 = zeros(N-1, nspin, lmax+1, 2*lmax+1)
    rho_rs_M = zeros(M+1, nspin, lmax+1, 2*lmax+1)
    
    for spin = 1:nspin
        for l = 0:lmax
            for m = -l:l
                rho_gal[:,spin, l+1, m+l+1], rho_gal_R2[:,spin, l+1, m+l+1], rho_rs_M[:,spin, l+1, m+l+1] = get_rho_gal(rho[:,spin, l+1,m+l+1], g, M=M, invS=invS)
            end
        end
    end
    rho_gal = rho_gal / (g.b-g.a) * 2

#    println(size(rho_gal))
#    println(size(D2))
#    n0 = gal_rep_to_rspace(1e-7, D2 * rho_gal[:,1,1,1], g) / 2.0
#    println("n0 $n0")

    if false
    begin
        n0 = xxx->ForwardDiff.derivative(xx-> ForwardDiff.derivative(x->gal_rep_to_rspace(x, rho_gal[:,1,1,1], g), xx), xxx)
        n00 = (n0(0.0)/2.0/ (g.b-g.a) * 2)
#        println("n0 $(n00)")
    end
    end
    
    return rho_gal, rho_gal_R2/ (g.b-g.a) * 2, filling, rho_rs_M / (g.b-g.a) * 2

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

function vhart(rhor2, D2, nel, g, M)

    vh_tilde = D2 \ rhor2

    vh_tilde = vh_tilde /(4*pi)*sqrt(pi)
    
    
    vh_mat = get_vh_mat(vh_tilde, g, nel, M=M)
    
    function vh_f(r)
        if r < 1e-7
            r = 1e-7
        end
        return gal_rep_to_rspace(r, vh_tilde, g) / r  .+ nel/g.b * sqrt(pi)/(2*pi)
    end
    
    return vh_mat, vh_f
end


function vxc(rho, g, M, N, funlist, gga, nspin)


    VLDA = zeros(length(rho), nspin)
    if nspin == 1
        rho_col = reshape(rho , length(rho),1)
        VLDA[:,1] = v_LDA.(rho[:, 1]/ (4*pi))
    elseif nspin == 2
        tt =  v_LDA_sp.(rho[:, 1], rho[:, 2])
        VLDA[:,:] = reshape(vcat(tt...), nspin,N+1)'
    end

    VLDA = VLDA * sqrt(4*pi)

#    println("size VLDA ", size(VLDA))
    VLDA_mat = get_gal_rep_matrix_R(VLDA[:,1,1,1], g, ; N = N)
    
    return VLDA_mat, VLDA
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
    end

end

function dft(; fill_str = missing, g = missing, N = -1, M = -1, Z = 1.0, niters = 50, mix = 0.5, mixing_mode=:pulay, exc = missing, lmax = missing, conv_thr = 1e-7)

    if M == -1
        M = g.M
    end
    if N == -1
        N = g.N
    end

    Z, nel, nspin, lmax, VC, D2, S, invsqrtS, invS, VECTS, VALS, funlist, gga = prepare(Z, fill_str, lmax, exc, N, M, g)

    
    rho, rhor2, filling, rho_rs_M = get_rho(VALS, VECTS, nel, nspin, lmax, N, M, invS, g, D2)

#    println("start rho ", rho[1:5])
    
    VALS_1 = deepcopy(VALS)

    for iter = 1:niters

        VALS_1[:,:,:,:] = VALS
        
        for spin = 1:nspin 
            for l = 0:(lmax)

                V = deepcopy(VC[l+1])
                if funlist != :hydrogen

                    vh_mat, vh_f = vhart(rhor2[:,1,1,1], D2, sum(nel), g, M)
                    vlda_mat, vlda = vxc(rho_rs_M[:,1,1,1], g, M, N, funlist, gga, nspin)

                    V += (4*pi*vh_mat/sqrt(4*pi) + vlda_mat/2 /sqrt(pi))
                    
                end
                
                vals, vects = eigen(D2 + V, S)
                for m = -l:l

                    VECTS[:,:,spin, l+1, l+1+m] = vects
                    VALS[:,spin, l+1, l+1+m] = vals
                end
                
                
            end
        end

#        display_eigs(VALS, nspin, lmax)
        
        rho_new, rhor2_new, filling, rho_rs_M_new = get_rho(VALS, VECTS, nel, nspin, lmax, N, M, invS, g, D2)

        println("ChebyshevDFT.Galerkin.do_1d_integral(rho[:,1,1,1], g) ", do_1d_integral(rho[:,1,1,1], g))


        
        rhor2 = rhor2_new * mix + rhor2 *(1-mix)
        rho = rho_new * mix + rho * (1-mix)
        rho_rs_M = rho_rs_M_new * mix + rho_rs_M * (1-mix)

        eigval_diff = maximum(abs.(filling.*(VALS - VALS_1)))
        println("iter $iter eigval_diff $eigval_diff ")
        
        if maximum(abs.(filling.*(VALS - VALS_1))) < conv_thr
            break
        end
        
    end

    
    display_eigs(VALS, nspin, lmax)
    println()
    return VALS, VECTS, rho, rhor2, filling, rho_rs_M

    
end #end dft


end #end module
