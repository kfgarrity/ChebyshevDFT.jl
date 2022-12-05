

module GalerkinDFT

using Polynomials
using SpecialPolynomials
using Base.Threads
using LinearAlgebra
using ForwardDiff
using QuadGK
using FastSphericalHarmonics
using ..Galerkin:do_integral

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

    

function prepare(Z, fill_str, lmax, exc, rmax, N, M, PTS, W, Bvals, d2, s)

    if typeof(Z) == String || typeof(Z) == Symbol
        Z = atoms[String(Z)]
    end
    
    
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

    a = 0.0
    b = rmax
    function V(r)
        return -Z / r + 0.5*l*(l+1)/r^2
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

    VC = []
    for ll = 0:(lmax+1)
        l = ll
        Vc = do_integral(Vx, N, PTS, W, Bvals; M = M)
        push!(VC, Vc)
    end

    VECTS = zeros(Complex{Float64}, N-1,N-1, nspin, lmax+1,2*(lmax)+1 )
    VALS = zeros(N-1,nspin, lmax+1,2*(lmax)+1 )


    D2 =-0.5* d2[1:N-1, 1:N-1] /  (b-a)^2*2^2
    S = s[1:N-1, 1:N-1]

    invsqrtS = inv(sqrt(S))
    
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
  
    return Z, nel, nspin, lmax, VC, D2, S, invsqrtS, VECTS, VALS, funlist, gga
    
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
    println("Filling")
    for t in T
        println(t)
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

function get_rho(VALS, VECTS, nel, nspin, lmax, N, M, ptw, w, bvals, S)

    rho = zeros(N-1, nspin, lmax+1, 2*(lmax+1))
    filling = zeros(size(VALS))
    
    for spin = 1:nspin
        for l = 0:lmax
            for m = -l:l
                fillval = 2.0
                nleft = nel[spin, l+1, m+lmax+1]
                if nleft < 1e-20
                    nleft = 1e-20
                end
                
                for n = 1:N-1
                    if nleft <= fillval/nspin + 1e-10
                        t = nleft * real( (S*VECTS[n, :, l+1, m+lmax+1]).*conj(VECTS[n, :, l+1, m+lmax+1]))
                        rho[:,spin, l+1, m+lmax+1] += t
                        filling[i,spin,d1[1], d1[2]] = nleft
                        break
                    else
                        nleft -= fillval / nspin
                        filling[i,spin,d1[1], d1[2]] = fillval / nspin

                        t = fillval/nspin* real( (S*VECTS[n, :, l+1, m+lmax+1]).*conj(VECTS[n, :, l+1, m+lmax+1]))
                        rho[:,spin, l+1, m+lmax+1] += t
                    end
                end
                    
                
            end
        end
    end
    
    return rho, filling

end

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
    @threads for i = 1:N
        asdf_arr[i] =   sum( (@view bvals[M,i,2:M+2]).*nr )#.*w[M,2:M+2])
    end

#    println(size(nr), " ", size(bvals[M,1:N,2:M+2]' * asdf_arr))

    return asdf_arr, nr, bvals[M,1:N,2:M+2]' * asdf_arr
end


function asdf(arr,bvals, pts, w, M)

#    asdf_arr = zeros(length(arr))
    nr = zeros(typeof(arr[1]), M+1)
    for n1 = 1:length(arr)
        nr += arr[n1]*bvals[M,n1,2:M+2]
    end

    nr = real(nr .* conj(nr))
    #nr = real(nr)
    
    println(size(bvals[M,1:length(arr), 2:M+2]), " " , size(nr))
    
    asdf_arr =   bvals[M,1:length(arr),2:M+2]' \ nr

    println(size(nr), " ", size(bvals[M,1:length(arr),2:M+2]' * asdf_arr))

    return asdf_arr
#    return nr, bvals[M,1:length(arr),2:M+2]' * asdf_arr
#    println([bvals[M,1:length(arr),2:M+2]' * asdf_arr  nr])
    
        #        for n2 = 1:length(arr)
#            asdf_arr[n2] += sum(nr .* bvals[M,n2,2:M+2].* w[M,2:M+2])
#        end
#    end
#    return asdf_arr
end


function dft(; fill_str = missing, N = 50, M = 300, Z = 1.0, rmax = 20.0, pts=missing , w=missing, bvals=missing, d2=missing, s=missing, niters = 50, mix = 0.5, mixing_mode=:pulay, exc = missing, lmax = missing, conv_thr = 1e-7)

    Z, nel, nspin, lmax, VC, D2, S, invsqrtS, VECTS, VALS, funlist, gga = prepare(Z, fill_str, lmax, exc, rmax, N, M, pts, w, bvals, d2, s)


    return VALS, VECTS
    
    rho, filling = get_rho(VALS, VECTS, nel, nspin, lmax, N, M, ptw, w, bvals)
    
    VALS_1 = deepcopy(VALS)

    for iter = 1:niters

        VALS_1[:,:,:,:] = VALS
        
        for spin = 1:nspin 
            for l = 0:(lmax)
                vals, vects = eigen(D2 + VC[ll+1], S)

                VECTS[:,:,spin, ll+1, 1] = vects
                VALS[:,spin, ll+1, m] = vals

                rho, filling = get_rho(VALS, VECTS, nel, nspin, lmax, N, M, ptw, w, bvals)
                
            end
        end

        if maximum(abs(nel*(VALS - VALS_1))) < conv_thr
            break
        end
        
    end


    end #end dft

end #end module
