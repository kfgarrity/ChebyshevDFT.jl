

module GalerkinDFT

#using Trapz
using Polynomials
using SpecialPolynomials
using Base.Threads
using Printf
using LinearAlgebra
using ForwardDiff
using QuadGK
using WignerSymbols
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

using ..Galerkin:gal
using ..AngularIntegration:leb

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
using ..AngularIntegration:real_gaunt_arr
using ..AngularIntegration:makeleb
using JLD

struct scf_data

    g::gal
    N::Int64
    M::Int64
    rmax::Float64
    α::Float64
    Z::Float64
    nspin::Int64
    lmax::Int64
    lmaxrho::Int64
    mix_lm::Bool
    niters::Int64
    mix::Float64
    mixing_mode::Symbol
    conv_thr::Float64
    fill_str::String
    nel::Array{Float64,3}
    exc
    funlist
    gga::Bool
    LEB::leb
    exx::Float64
    filling::Array{Float64,4}
    VALS::Array{Float64,4}
    VECTS_small::Array{Float64,5}
    VECTS_big::Array{Float64,3}    
    rho_R2::Array{Float64,4}
    big_code
    R::Array{Float64}
    D1::Array{Float64,2}
    D2::Array{Float64,2}
    S::Array{Float64}
    V_C::Array{Float64,2}
    V_L::Array{Float64,2}
    mat_n2m::Array{Float64,2}
    mat_m2n::Array{Float64,2}
    dict_lm
    lm_dict
    etot::Float64
    e_vxc::Float64
    e_hart::Float64
    e_exx::Float64
    e_ke::Float64
    e_nuc::Float64
end

Base.show(io::IO, d::scf_data) = begin

    println(io, "SCF input data")
    println(io, "N=$(d.N) ; M=$(d.M) ; rmax=$(d.rmax) ; α=$(d.α)")
    println(io, "Z=$(d.Z) ; nspin=$(d.nspin) ; lmax=$(d.lmax) ; lmaxrho=$(d.lmaxrho) ; mix_lm=$(d.mix_lm) ; fill_str=$(d.fill_str); net_charge=$(sum(d.nel) - d.Z)")
    println(io, "mix=$(d.mix) ; mising_mode=:$(d.mixing_mode) ; conv_thr=$(d.conv_thr) ; niters=$(d.niters)")
    println(io, "Functional: $(d.exc) ; gga=$(d.gga) ; exx=$(d.exx) " )
    println(io)
    NEL=sum(d.nel)
    if NEL <= 2
        nmax=2*ones(Int64, 8)
    elseif NEL <= 10
        nmax=3*ones(Int64, 8)
    elseif NEL <= 18
        nmax=4*ones(Int64, 8)
    elseif NEL <= 36
        nmax=5 *ones(Int64, 8)
    else
        nmax=6*ones(Int64, 8)
    end
    
    display_filling(d.filling, d.lmax, nmax, d.nspin)

    println()
    println(io, "SCF output data")
    display_eigs(d.VALS, d.nspin, d.lmax)
    println()
    display_energy(d.etot, d.e_vxc, d.e_hart, d.e_exx, d.e_ke, d.e_nuc)
end

function get_vect_gal(d, n, l, m, spin=1)
    if d.mix_lm == false
        return d.VECTS_small[:,n,spin,l+1,l+m+1]
    else
        vnum = d.big_code[(spin, l,m)]
        v = d.VECTS_big[:,vnum[n], spin]
        vv = zeros(d.N-1, length( d.lm_dict))
        c = 0
        for l = 0:d.lmax
            for m = -l:l
                c+=1
                ind = ( 1:(d.N-1)) .+ (d.N-1)*d.lm_dict[(l,m )]
                vv[:,d.lm_dict[(l,m )]+1 ] = v[ind]
            end
        end
        return vv

    end

        

end

function get_vect_r(r,d,n,l,m,spin=1; deriv=0)
    vv = get_vect_gal(d, n, l, m, spin)
    vr = zeros(length(r), size(vv,2))
#    println("s, ", size(vv))
    for col in 1:size(vv,2)
        vr[:, col] .= gal_rep_to_rspace(r, vv[:,col][:], d.g, deriv=deriv)

    end
    return vr
    
end

#    println(io, 
#    println(io, 
#    println(io, 
#    println(io, 
#    println(io, 
#    println(io, 
#    println(io, 
    
#end


function make_scf_data(    g::gal,    N::Int64,    M::Int64,    rmax::Float64,  α::Float64,  Z::Float64,    nspin::Int64,    lmax::Int64,    lmaxrho::Int64,    mix_lm::Bool,    niters::Int64,    mix::Float64,    mixing_mode::Symbol,    conv_thr::Float64,    fill_str::String,    nel::Array{Float64,3},    exc,    funlist,    gga::Bool,    LEB::leb,    exx::Float64,    filling::Array{Float64,4},    VALS::Array{Float64,4},    VECTS_small::Array{Float64,5},    VECTS_big::Array{Float64,3}    ,    rho_R2::Array{Float64,4},    big_code,    R::Array{Float64},    D1::Array{Float64,2},    D2::Array{Float64,2},    S::Array{Float64},    V_C::Array{Float64,2},V_L::Array{Float64,2},    mat_n2m::Array{Float64,2},    mat_m2n::Array{Float64,2},    dict_lm,lm_dict,     etot::Float64,    e_vxc::Float64,    e_hart::Float64,    e_exx::Float64,    e_ke::Float64,    e_nuc::Float64)

    return scf_data(    g,    N,    M,    rmax,  α,  Z,    nspin,    lmax,    lmaxrho,    mix_lm,    niters,    mix,    mixing_mode,    conv_thr,    fill_str,    nel,    exc,    funlist,    gga,    LEB,    exx,    filling,    VALS,    VECTS_small,    VECTS_big    ,    rho_R2,    big_code,    R,    D1,    D2,    S,    V_C, V_L,   mat_n2m,    mat_m2n,    dict_lm,lm_dict, etot, e_vxc, e_hart, e_exx, e_ke, e_nuc)


end
    
function calc_energy(rho_rs_M_LM, EXC_LM, funlist, g, N, M, R, Vin, filling, vals, VH, Z, VX_LM, lmax, VECTS, exx, mix_lm, nspin, big_code, lm_dict)

    e_vxc = 0.0
    if funlist != :hydrogen && funlist != :none
        e_vxc = calc_energy_vxc(rho_rs_M_LM, EXC_LM, g, N, M, R)
    end
    
    #println("------------------------------------")
    #println("ENERGY XC FUN:  $e_vxc")
    e_hart = 0.0
    if funlist != :hydrogen 
        e_hart = calc_energy_vh(rho_rs_M_LM, VH , g, N, M, R)
    end
    
    #println("ENERGY HARTREE: $e_hart")
    e_exx = 0.0
    if exx > 1e-12
        e_exx = calc_energy_exx(VX_LM, VECTS, filling, g, N, M, mix_lm, lmax, nspin, big_code, lm_dict)
    end
    #println("ENERGY EXX:     $e_exx")
    e_ke = 0.0
    e_ke = calc_energy_ke(rho_rs_M_LM, Vin , g, N, M, R, filling, vals, e_exx)
    #println("ENERGY KINETIC: $e_ke")
    e_nuc = 0.0
    e_nuc = calc_energy_nuc(rho_rs_M_LM, Z , g, N, M, R)
    #println("ENERGY NUCLEAR: $e_nuc")
    #println()
    etot = e_vxc+e_hart+e_exx+e_ke+e_nuc
    #println("ETOT:           $etot")
    #println("------------------------------------")

    display_energy(etot, e_vxc, e_hart, e_exx, e_ke, e_nuc)
    
    return etot, e_vxc, e_hart, e_exx, e_ke, e_nuc
end

function display_energy(etot, e_vxc, e_hart, e_exx, e_ke, e_nuc)
    println("------------------------------------")
    println("ENERGY XC FUN:  $e_vxc")
    println("ENERGY HARTREE: $e_hart")
    println("ENERGY EXX:     $e_exx")
    println("ENERGY KINETIC: $e_ke")
    println("ENERGY NUCLEAR: $e_nuc")
    println("ETOT:           $etot")
    println("------------------------------------")

end

function calc_energy_vxc(rho_rs_M_LM, EXC_LM, g, N, M, R)
    a,nspin,lmax1,mmax1 = size(rho_rs_M_LM)
    e_vxc = sum( sum(rho_rs_M_LM, dims=2)[:,1,:,:] .* EXC_LM[:,1:lmax1,1:mmax1].* g.w[2:M+2,M] .* R.^2  )  /sqrt(4*pi)/2 * (g.b - g.a)/2.0
    #println("norm1 ", sum(rho_rs_M_LM[:,1,1,1] .* g.w[2:M+2,M] .* R.^2 ) .* (g.b - g.a)/2.0)
    #println("norm2 ", sum(rho_rs_M_LM[:,2,1,1] .* g.w[2:M+2,M] .* R.^2 ) .* (g.b - g.a)/2.0)
    #println("exc1 ", EXC_LM[1])
    
    #elda += 2*pi*sum(elda_LM[:,l,m].*rho_LM_tot[:,1,l,m].*rall.^2 .* wall)
    return e_vxc
    
end

function calc_energy_ke(rho_rs_M_LM, Vin , g, N, M, R, filling, vals, e_exx)

    KE = sum(filling .* vals)
    #KE = 0.0
    #    println("ke $KE")
    nspin = size(rho_rs_M_LM, 2)
    for spin = 1:nspin
#        println("add ", -sum( sum( Vin[:,spin,:,:] .* rho_rs_M_LM[:,spin,:,:], dims=[2,3]) .*  g.w[2:M+2,M] .* R.^2)/ sqrt(4*pi)/2 * (g.b - g.a)/2.0)
        KE += -sum( sum( Vin[:,spin,:,:] .* rho_rs_M_LM[:,spin,:,:], dims=[2,3]) .*  g.w[2:M+2,M] .* R.^2)/ sqrt(4*pi)/2 * (g.b - g.a)/2.0
    end
    KE += e_exx * -2.0
    #println("KE $KE sum ", sum(abs.( Vin[:,1,:,:])))
    return KE 
end

function calc_energy_vh(rho_rs_M_LM, VH , g, N, M, R)

    
    return 2.0*0.5*4*pi* sum((sum( sum(rho_rs_M_LM, dims=2)[:,1,:,:] .* VH, dims=[2,3]) .* g.w[2:M+2,M] .* R.^2  ))  /sqrt(4*pi)/2 * (g.b - g.a)/2.0
    
    ####return 0.5 * 4 * pi * sum(VH .* rho .* wall .* rall.^2 )
    
end

function calc_energy_nuc(rho_rs_M_LM, Z , g, N, M, R)

    return -Z * 4 * pi * sum(sum(rho_rs_M_LM[:,:,1,1], dims=2)  .* g.w[2:M+2,M] .* R)  /sqrt(4*pi)/2 * (g.b - g.a)/2.0 / sqrt(pi)
    
end

function calc_energy_exx(VX_LM, VECTS, filling, g, N, M, mix_lm, lmax, nspin, big_code, lm_dict)

    e_exx = 0.0
    if mix_lm == false
        for spin = 1:nspin
            for l1 = 0:lmax
                for m1 = -l1:l1
                    v = VX_LM[:,:,spin, l1+1,l1+m1+1, l1+1, l1+m1+1]
                    for n = 1:(N-1)
                        f = filling[n,spin,l1+1,m1+l1+1]
                        
                        fillval = filling[n, spin, l1+1, l1+m1+1]
                        if fillval < 1e-20
                            break
                        end
                        e_exx += 0.5*fillval*VECTS[:,n,spin, l1+1, l1+m1+1]' * v *  VECTS[:,n,spin, l1+1, l1+m1+1]
                    end
                end
            end
        end                

    else
        Nsmall = N-1
        for spin = 1:nspin
            for l = 0:(lmax)
                for m = -l:l
                    nlist = big_code[(spin,l, m)]
                    for (n_count, n_code) in enumerate(nlist)
                        fillval = filling[n_count,spin,l+1,m+l+1]
                        if fillval < 1e-20
                            continue
                        end
                        for l1 = 0:(lmax)
                            for m1 = -l1:l1
                                ind1 = (1:Nsmall) .+ Nsmall*lm_dict[(l1,m1)]
                                for l2 = 0:(lmax)
                                    for m2 = -l2:l2
                                        ind2 = (1:Nsmall) .+ Nsmall*lm_dict[(l2,m2)]
                                        v = VX_LM[:,:,spin, l1+1,l1+m1+1, l2+1, l2+m2+1]
                                        e_exx += 0.5*fillval*VECTS[ind1,n_code,spin]' * v *  VECTS[ind2,n_code,spin]
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end        
    end    
    

    return e_exx

end

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

        if (typeof(exc) == String && lowercase(exc) == "hydrogen") || (typeof(exc) == Symbol && exc == :hydrogen) || (typeof(exc) == String && lowercase(exc) == "h") || (typeof(exc) == Symbol && (exc == :H || exc == :h))
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

function mix_vects(VECTS, VECTS_new, mix, filling, S, nspin, lmax, N; mixall=false)

    for spin = 1:nspin
        for l = 0:lmax #this is the sum over b
            for m = -l:l
                for n = 1:N-1 #this is the sum over b as well
                    
                    f = filling[n,spin,l+1,m+l+1]
                    #a =  real(abs(sum(VECTS[:,n,spin,l+1, m+l+1] -  VECTS_new[:,n,spin,l+1, m+l+1])))
                    #b =  real(abs(sum(VECTS[:,n,spin,l+1, m+l+1] +  VECTS_new[:,n,spin,l+1, m+l+1])))
                    #if a < b
                    #    sign = +1.0
                    #else
                    #    sign = -1.0
                    #end
#                    println("t $spin $l $m $n ", real(VECTS[:,n,spin,l+1, m+l+1]'*S*VECTS_new[:,n,spin,l+1, m+l+1]))
                    s = sign(real(VECTS[:,n,spin,l+1, m+l+1]'*S*VECTS_new[:,n,spin,l+1, m+l+1]))
                    
                    VECTS[:,n,spin,l+1, m+l+1] = VECTS[:,n,spin,l+1, m+l+1]*(1-mix) + s*mix*VECTS_new[:,n,spin,l+1, m+l+1]

                    norm = real(VECTS[:,n,spin,l+1, m+l+1]'*S*VECTS[:,n,spin,l+1, m+l+1])
                    VECTS[:,n,spin,l+1, m+l+1] = VECTS[:,n,spin,l+1, m+l+1] ./ norm^0.5

                    if mixall == false
                        if f < 1e-20 && n > 4
                            break
                        end
                    end
                    
                end
            end
        end
    end

    #no return
    
end


function mix_vects_big(VECTS, VECTS_new, mix, filling, Sbig, nspin, lmax, N, big_code, big_code_new, VALS, VALS_big; mixall = false)
    #    println("mix_vects_big")
    Nbig = size(Sbig,1)
    Snh = Float64.(Sbig)
    for spin = 1:nspin
        for l = 0:lmax #this is the sum over b
            for m = -l:l
                nlist_old = big_code[(spin,l, m)]
                nlist_new = big_code_new[(spin,l, m)]
#                println("l $l m $m nlist_old ", nlist_old)
#                println("l $l m $m nlist_new ", nlist_new)
                for (n_count, (n_old, n_new)) in enumerate(zip(nlist_old, nlist_new))
                    f = filling[n_count,spin,l+1,m+l+1]
                    VALS[n_count, spin, l+1, l+m+1] = VALS_big[n_new,spin]                    

                    s = 0.0
                    @tturbo for i = 1:Nbig
                        for j = 1:Nbig
                            s += VECTS[i,n_old,spin]*Snh[i,j]*VECTS_new[j,n_new,spin]
                        end
                    end
                    s = sign(s)
                    VECTS[:,n_old,spin] .= (@view VECTS[:,n_old,spin])*(1-mix) + s*mix*(@view VECTS_new[:,n_new,spin])
                    norm = 0.0
                    @tturbo for i = 1:Nbig
                        for j = 1:Nbig
                            norm += VECTS[i,n_old,spin]*Snh[i,j]*VECTS[j,n_old,spin]
                        end
                    end
                    VECTS[:,n_old,spin] = (@view VECTS[:,n_old,spin]) ./ norm^0.5

                    #                    s = sign(real( ( @view VECTS[:,n_old,spin])'*Sbig*(@view VECTS_new[:,n_new,spin])))
#                    VECTS[:,n_old,spin] = (@view VECTS[:,n_old,spin])*(1-mix) + s*mix*(@view VECTS_new[:,n_new,spin])
                        
#                    norm = real( (@view VECTS[:,n_old,spin])'*Sbig*(@view VECTS[:,n_old,spin]))
#                    VECTS[:,n_old,spin] = (@view VECTS[:,n_old,spin]) ./ norm^0.5

                    if mixall != false
                        if f < 1e-20 && n_count > 4
                            break
                        end
                    end


                    
                end
            end
        end
    end
    
    #no return
    
end

function prepare(Z, fill_str, lmax, exc, N, M, g, lmaxrho, mix_lm)

    if typeof(Z) == String || typeof(Z) == Symbol
        Z = atoms[String(Z)]
    end

    a = g.a
    b = g.b
    
    Z = Float64(Z)

    for k in keys(atoms)
        if abs(Z - atoms[k]) <=  0.5
#            println("Atom name $k")
            if ismissing(fill_str)
                fill_str = k
            end
        end
    end
    
#    println("setup filling")
    nel, nspin, lmax, filling = setup_filling(fill_str, N, lmax_init=lmax)

    fs = sum(filling, dims=[2,3,4])
    nmax = findfirst(fs[:] .> 1e-12)
#    println("nmax $nmax")

    
    println("choose exc")
    funlist, gga = choose_exc(exc, nspin)


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

    if mix_lm
        VECTS = zeros(Float64, length(lm_dict) * (N-1), length(lm_dict) * (N-1), nspin)
    else
        VECTS = zeros(Float64, N-1,N-1, nspin, lmax+1,2*lmax+1 )
    end
    
    VALS = zeros(N-1,nspin, lmax+1,2*lmax+1 )


    D2 =-0.5* g.d2[1:N-1, 1:N-1] /  (b-a)^2*2^2
    D1 =  g.d1[1:N-1, 1:N-1] /  (b-a)*2

    S = g.s[1:N-1, 1:N-1]

    invsqrtS = inv(sqrt(S))
    invS = inv(S)
    
   #    println("size VECTS ", size(VECTS))

    #    println("initial eig")

#    println("lm_dict")
#    println(lm_dict)
    big_code = Dict()
    counter = 1
    for ll = 0:lmax
        vals, vects = eigen(D2 + V_C + ll*(ll+1)*V_L, S)

        h = D2 + V_C + ll*(ll+1)*V_L
        #println("ham ", sum(abs.(h - h')), " ", sum(abs.(S - S')))
        #println("ham")
        #println(h)
        #println()
        #println(S)
        #vals, vects = eigen( invsqrtS*(D2 + VC[ll+1])*invsqrtS)
        
#        println("vals $ll $(vals[1:3])")
        for m = 1:(2*(ll)+1)
            for spin = 1:nspin
                big_code[(spin,ll, m-ll-1)] = counter:(counter+N-2)
#                println("add to big code ", (spin,ll, m-ll-1))
                if mix_lm == true
                    ind = ( 1:(N-1)) .+ (N-1)*lm_dict[(ll,m - ll - 1)]
#                    println("l $ll m $m spin $spin")
#                    println("ind ", ind)
#                    println("big_code ", big_code[(spin,ll, m-ll-1)] )
                    
                    VECTS[ind,big_code[(spin,ll, m-ll-1)],spin] = vects
                else
                    VECTS[:,:,spin, ll+1, m] = vects
                end
                VALS[:,spin, ll+1, m] = vals
#                println("vals prepare $ll $m ", vals[1:3])
            end
            counter += N-1
        end
    end
    
    
    LEB = makeleb(lmaxrho*2+1 , lmax=max(lmaxrho, lmax*2) )

    R = get_r_grid(g, M=M)
    R = R[2:M+2]

    gbvals2 = zeros(M+1, N-1, N-1)
#    println("pretrain")
    @time @threads for n1 = 1:(N-1)
        for n2 = 1:(N-1)
            gbvals2[:, n1, n2] =  (@view g.bvals[2:M+2,n1,M]).*(@view g.bvals[2:M+2,n2,M]) 
        end
    end


    hf_sym = zeros(lmax+1, lmax*2+1, lmax+1, lmax*2+1, lmax+1, lmax*2+1, lmax*2+1)
    for l = 0:lmax
        for m = -l:l
            for l1 = 0:lmax
                for m1 = -l1:l1
                    for l2 = 0:lmax
                        for m2 = -l2:l2
                            for L = 0:lmax*2
                                ta = 0;
                                for MM = -L:L;
                                    ta +=   4*pi/(2*L+1) * real_gaunt_dict[(L,MM,l,m,l1,m1)]*real_gaunt_dict[(L,MM,l,m,l2,m2)]
                                end
                                hf_sym[l+1, m+l+1, l1+1, l1+m1+1, l2+1, l2+m2+1, L+1] = ta
                            end
                        end
                    end
                end
            end
        end
    end

    hf_sym_big = zeros(lmax+1, lmax*2+1, lmax+1, lmax*2+1,lmax+1, lmax*2+1, lmax+1, lmax*2+1, lmax*2+1)
    for l = 0:lmax
        for m = -l:l
            for la = 0:lmax
                for ma = -la:la
                    for l1 = 0:lmax
                        for m1 = -l1:l1
                            for l2 = 0:lmax
                                for m2 = -l2:l2
                                    for L = 0:lmax*2
                                        ta = 0;
                                        for MM = -L:L;
                                            ta +=   4*pi/(2*L+1) * real_gaunt_dict[(L,MM,l,m,l1,m1)]*real_gaunt_dict[(L,MM,la,ma,l2,m2)]
                                        end
                                        hf_sym_big[l+1, m+l+1, la+1, ma+la+1, l1+1, l1+m1+1, l2+1, l2+m2+1, L+1] = ta
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    mat_n2m = MAT_N2M(g, N=N, M=M)
    mat_m2n = MAT_M2N(g, N=N, M=M)

#    r5 = mat_n2m'*diagm( (R/g.b).^(L/2.0) ) * mat_m2n'#
#
    R5 = Dict()
    
    LINOP = Dict()
    for L = 0:lmax*2
        Linv = inv((D2 + L*(L+1)*V_L))  
        LINOP[L] = mat_m2n' * S* Linv *S  *  mat_m2n

        #        println("typeof R ", typeof((R/g.b).^(L)))

        function V_R(r)
            return (r/g.b).^L
        end

        #V_C = get_gal_rep_matrix(V_c, g, ; N = N, M = M)


        
        MAT = get_gal_rep_matrix(V_R, g, N=N, M=M)



        #        R5[L] = deepcopy(mat_m2n' *S*MAT*S*  mat_m2n)
        R5[L] = MAT

        #R5[L] = mat_m2n' * S* MAT  *S  *  mat_m2n

        #@println("MAT $L ", sum(MAT), " size MAT ", size(MAT), " size R5 ", size(R5[L]) )        
        
        #R5[L] = mat_m2n' * MAT   *  mat_m2n
        #R5[L] = mat_m2n' *  MAT   *  mat_m2n

#        if L == 0
#            println("R5[L] $L ", sum(abs.(R5[L] - S)))
#        end
        
        #R5[L] = mat_n2m'*diagm( (R/g.b).^(L/2) ) * mat_m2n'
        #R5[L] = mat_m2n*diagm( (R/g.b).^(L/2 ) ) * mat_n2m
    end

    if mix_lm
        Sbig = zeros( (N-1)*length(lm_dict), (N-1)*length(lm_dict))
        for l = 0:(lmax)
            for m = -l:l
                ind1 = (1: (N-1)) .+ (N-1)*lm_dict[(l,m)]
                for l2 = 0:(lmax)
                    for m2 = -l2:l2
                        ind2 = (1:(N-1)) .+ (N-1)*lm_dict[(l2,m2)]
                        if l == l2 && m == m2
                            Sbig[ind1,ind2] += S                            
                        end
                    end
                end
            end
        end
        Sbig= Hermitian(Sbig)
    else
        Sbig = Hermitian(zeros(1,1))
    end
        
    
    
    return Z, nel, filling, nspin, lmax, V_C, V_L,D1, D2, S, invsqrtS, invS, VECTS, VALS, funlist, gga, LEB, R, gbvals2, nmax, hf_sym, hf_sym_big,mat_n2m, mat_m2n, R5, LINOP, lm_dict, dict_lm, big_code, Sbig
    
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

    display_filling(filling, lmax, nmax, nspin)
    
    return nel, nspin, lmax, filling
    
end

function display_filling(filling, lmax, nmax, nspin)
    println("FILLING----")
    if nspin == 1
        println(" n  l  m   fill")
    else
        println(" n  l  m   fill_up  fill_dn")
    end
    
    #printing
    for l = 0:lmax
        for m = (-l):l
            for n = 1:nmax[l+1]
                #                println("$l $m $n")
                if nspin == 1
                    f = filling[n,1,l+1, m+l+1]
                    if abs(f) > 1e-20 
                        @printf "%2i %2i %2i  %4f \n" (n+l) l m f
                    end
                elseif nspin == 2
                    f1 = filling[n,1,l+1, m+l+1]
                    f2 = filling[n,2,l+1, m+l+1]
                    if abs(f1) > 1e-20 ||  abs(f2) > 1e-20
                        @printf "%2i %2i %2i  %4f   %4f \n" (n+l) l m  f1 f2
                    end
                end
            end
        end
    end
    println("---")

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

#    psipsi = zeros(M+1, nspin, nmax, lmax+1, 2*lmax+1, nmax, lmax+1, 2*lmax+1)
    if false
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
#                                        psipsi[:,spin, n1,l1+1, m1+1+l1, n2, l2+1, m2+l2+1] += real(t1.*conj(t1)) * sqrt(fillval1)*sqrt(fillval2)
                                        
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
#    println("pp ", psipsi[1], " " , rho_rs_M_R2_LM2[1], " " ,  psipsi[1]/rho_rs_M_R2_LM2[1])
    
    
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
#                        if sum(abs.(gcoef * rho_rs_M_R2_LM2[:,spin, l+1, m+1+l])) > 1e-8
#                            println("add rho spin $spin, lmr $lr $mr,  $l $m  ", gcoef, " sum ", sum(rho_rs_M_R2_LM2[:,spin, l+1, m+1+l]))
#                        end
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
#    println("rho_rs_M_R2_LM  ", rho_rs_M_R2_LM[1])
    

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

#    psipsi_r_M_LM = zeros(N-1, nspin, nmax, lmaxrho+1, lmaxrho*2+1, nmax, lmaxrho+1, lmaxrho*2+1) 
#=
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
=#
#    println("psipsi_r_M_LM ", psipsi_r_M_LM[1], " ", rho_gal_dR_LM[1], " " , psipsi_r_M_LM[1]/rho_gal_dR_LM[1])
    
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

#    psipsi_r_M_LM = psipsi_r_M_LM   / (g.b-g.a) * 2
    

    #    println("SIZE ", size(rho_gal_R2_LM), " " , size(rho_gal_mulipole_LM), " lmax rho ", lmaxrho)
    
    #    println("a ", rho_gal_R2[1]," ", rho_gal_R2_LM[1])
    #    println("b ", rho_gal_dR[1]," ", rho_gal_dR_LM[1])
    #    println("c ", rho_rs_M[1]," ", rho_rs_M_LM[1])
    #    println("d ", rho_gal_R2[1]," ", rho_gal_mulipole_LM[1])

    
    println("ChebyshevDFT.Galerkin.do_1d_integral(rho_gal, g)    ", do_1d_integral(rho_gal_R2_LM[:,1,1,1], g))
    println("ChebyshevDFT.Galerkin.do_1d_integral(rho_gal, g) dR ", do_1d_integral(rho_gal_dR_LM[:,1,1,1], g))

    MP = zeros(lmaxrho+1, 2*lmaxrho+1)
    
        for l = 0:lmaxrho
            for m = -l:l
                for spin = 1:nspin

                #MP[l+1, l+m+1] += do_1d_integral(rho_gal_mulipole_LM[:,spin,l+1,l+m+1], g) / (2*l +1)
                #println("MP $l $m $spin ", do_1d_integral(rho_gal_mulipole_LM[:,spin,l+1,l+m+1], g) / (2*l +1) , " alt ", sum((rho_rs_M_R2_LM[:,spin,l+1,l+m+1]) .* g.w[2:M+2,M] .* R.^l) / (2*l+1) )
                    MP[l+1, l+m+1] += real(sum((rho_rs_M_R2_LM[:,spin,l+1,l+m+1]) .* g.w[2:M+2,M] .* R.^l) / (2*l+1))
                    #println("typeof " , typeof(rho_rs_M_R2_LM[1,spin,l+1,l+m+1]))
                end
                if abs(MP[l+1, l+m+1]) > 1e-5
                    println("rho MP $l $m ", MP[l+1, l+m+1])
                end
            end
        end
#    println()
#    for spin = 1:nspin
#        for l = 0:lmaxrho
#            for m = -l:l
#                MP[l+1, l+m+1] += do_1d_integral(rho_gal_mulipole_LM[:,spin,l+1,l+m+1], g) / (2*l +1)
#                println("MP2 $l $m $spin ", MP[l+1, l+m+1])
#            end
#        end
#    end
#    println()

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
    
    return rho_gal_R2_LM, rho_gal_dR_LM, rho_rs_M_LM,  drho_rs_M_LM, MP

end



function get_rho_big(VALS, VECTS_big, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, D2, lm_dict,dict_lm, big_code, gga, R; VALS_big = missing)

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
#    for MM = [100, 200, 300]
#        for l = 0:lmaxrho
#            #for l = [0]
#            for m = -l:l
#                mp = 0.0
#                for spin = 1:nspin
#                    mp += do_1d_integral(rho_gal_mulipole_LM[:,spin,l+1,l+m+1], g, M = MM) / (2*l+1)
#                end
#                if abs(mp ) > 1e-6
#                    println("MP big MM $MM $l $m  $mp")
#                    t = 0.0
#                    for spin = 1:nspin
#                        t += trapz(R, rho_rs_M_LM[:,spin,l+1,l+m+1] .* R.^(2+l))/(2*l+1)
#                    end
#                    println("trapz ", t)
#                end
#            end
#        end
#    end
    
    MP = zeros(lmaxrho+1, 2*lmaxrho+1)
    
    for l = 0:lmaxrho
        #for l = [0]
        for m = -l:l
            for spin = 1:nspin
                MP[l+1, l+m+1] += do_1d_integral(rho_gal_mulipole_LM[:,spin,l+1,l+m+1], g) / (2*l+1)
            end
#            if abs(MP[l+1, l+m+1] ) > 1e-6
#                println("MP big $l $m  $(MP[l+1, l+m+1])")
#                t = 0.0
#                for spin = 1:nspin
#                    t += trapz(R, rho_rs_M_LM[:,spin,l+1,l+m+1] .* R.^(2+l))/(2*l+1)
#                end
#                println("trapz ", t)
#            end
        end
    end
#    MP .= 0.0
#    MP[1,1] = 5.0
#    MP[3,3] = 1.103897865555666
#    MP[5,5] = 0.004816230021690286
    

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
    
    return rho_gal_R2_LM, rho_gal_dR_LM, rho_rs_M_LM, drho_rs_M_LM, MP

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

function vxx_LM( D2, g, N, M, lmaxrho, lmax, MP, V_L, gbvals2, exx, nmax, nspin, filling, VECTS, S, rho_dR)


    mat_n2m = MAT_N2M(g, N=N, M=M)
    mat_m2n = MAT_M2N(g, N=N, M=M)
    R = g.R.(g.pts[2:M+2,M])

    vx2 =  zeros(N-1,N-1)
    VX_LM2 = zeros(N-1, N-1, nspin, lmax+1, 2*lmax+1)


    #    r5 = mat_n2m'*diagm(1 ./ R.^0.5)*mat_n2m
    r = mat_n2m'*diagm( R)*mat_n2m
    println("size(r) ", size(r))

    LINOP = Dict()
    for L = 0:lmax*2
        Linv = inv((D2 + L*(L+1)*V_L))
        LINOP[L] = mat_m2n' * S* Linv *S  *  mat_m2n
    end



    #okay, there is a sum for L, a sum over b, and then sum over lright and lleft.
    
    for spin = 1:nspin
    
        for l = 0:lmax #this is the sum over b
            for m = -l:l
                for n = 1:N-1 #this is the sum over b as well
                    
                    f = filling[n,spin,l+1,m+l+1]
                    if f < 1e-20
                        break
                    end
                    t = mat_n2m*VECTS[:,n,spin,l+1, m+l+1]
                    tf1 =  t ./ R  / sqrt(g.b-g.a) * sqrt(2 )

                    for l1 = 0:lmax
                        for m1 = -l1:l1
                            
                            for L =  0:lmax*2
                                symfactor_old = 0.0
                                for M = -L:L
                                    symfactor_old += real_gaunt_dict[(L,M,l,m,l1,m1)]*sqrt(4*pi)
#                                    symfactor += real_gaunt_dict[(L,0,l,0,l1,0)]
                                end
                                symfactor = Float64(wigner3j(l,l1,L,0,0,0))^2 * (2*L+1) 

                                ta = 0;
                                for M = -L:L;
                                    
                                    for m11 = -l1:l1 #this one is wierd. 

                                        for ma = -l:l  #this one is fine. actually, As is this one.

                                            ta += 4*pi/(2*l+1)/(2*l1+1)/(2*L+1) * real_gaunt_dict[(L,M,l,ma,l1,m11)]^2;
                                        end
                                    end
                                end
                                
                                println("$l $m   $l1 $m1   $L  symfactor $(round(symfactor, digits=5))  old  $(round(symfactor_old,digits=5))  new    ", round(ta *(2*L+1), digits=5))
#                                println("$l $m   $l1 $m1   $L  symfactor $(round(symfactor, digits=5))  old  $(round(symfactor_old,digits=5))  new    ", round(ta *(2*L+1)), digits=5))
#                                println("$l $m   $l1 $m1   $L  symfactor $(round(symfactor, digits=5))  old  $(round(symfactor_old,digits=5))  new    ", round(ta *(2*l+1)*(2*l1+1), digits=5))
#                                println("$l $m   $l1 $m1   $L  symfactor $(round(symfactor, digits=5))  old  $(round(symfactor_old,digits=5))  new    ", round(ta *(2*l+1)*(2*L+1), digits=5))
                                println()

                                temp = diagm(conj.(tf1)) * LINOP[L] * diagm(tf1)
                                
                                VX_LM2[:,:,spin,l1+1,m1+l1+1] += -symfactor*f* real(   mat_n2m'*  temp * mat_n2m)/2 
                            end
                        end
                    end
                end
            end
        end
    end
    for spin = 1:nspin
        for l1 = 0:lmax
            for m1 = -l1:l1
                VX_LM2[:,:,spin,l1+1,m1+l1+1] = 0.5 *(VX_LM2[:,:,spin,l1+1,m1+l1+1] + VX_LM2[:,:,spin,l1+1,m1+l1+1]') 
            end
        end
    end    
    if nspin == 1
        return VX_LM2*exx/2.0
    else
        return VX_LM2*exx
    end
    
end


function vxx_LM4( D2, g, N, M, lmaxrho, lmax, MP, V_L, gbvals2, exx, nmax, nspin, filling, VECTS, S, rho_dR)


    mat_n2m = MAT_N2M(g, N=N, M=M)
    mat_m2n = MAT_M2N(g, N=N, M=M)
    R = g.R.(g.pts[2:M+2,M])

    vx2 =  zeros(N-1,N-1)
    VX_LM2 = zeros(N-1, N-1, nspin, lmax+1, 2*lmax+1, lmax+1, 2*lmax+1)
    #VX_LM2 = zeros(N-1, N-1, nspin, lmax+1, 2*lmax+1)


    #    r5 = mat_n2m'*diagm(1 ./ R.^0.5)*mat_n2m
    r = mat_n2m'*diagm( R)*mat_n2m
    #println("size(r) ", size(r))

    LINOP = Dict()
    for L = 0:lmax*2
        Linv = inv((D2 + L*(L+1)*V_L))
        LINOP[L] = mat_m2n' * S* Linv *S  *  mat_m2n
    end



    #okay, there is a sum for L, a sum over b, and then sum over lright and lleft.
    
    for spin = 1:nspin
    
        for l = 0:lmax #this is the sum over b
            for m = -l:l
                for n = 1:N-1 #this is the sum over b as well
                    
                    f = filling[n,spin,l+1,m+l+1]
                    if f < 1e-20
                        break
                    end
                    t = mat_n2m*VECTS[:,n,spin,l+1, m+l+1]
                    tf1 =  t ./ R  / sqrt(g.b-g.a) * sqrt(2 )
                    for L =  0:lmax*2
                        temp = diagm(conj.(tf1)) * LINOP[L] * diagm(tf1)

                        for l1 = 0:lmax
                            for m1 = -l1:l1
                                for l2 = 0:lmax
                                    for m2 = -l2:l2
                                        

                                        #                                        symfactor_old = 0.0

                                        #for M = -L:L
                                        #   symfactor_old += real_gaunt_dict[(L,M,l,m,l1,m1)]*sqrt(4*pi)
                                        #                                    symfactor += real_gaunt_dict[(L,0,l,0,l1,0)]
                                        #                                 end
                                        #symfactor = Float64(wigner3j(l,l1,L,0,0,0))^2 * (2*L+1) 

                                        ta = 0;
                                        for M = -L:L;
#                                            for m11 = -l1:l1 #this one is wierd. 
#                                                for ma = -l:l  #this one is fine. actually, As is this one.
#                                                    ta += 4*pi/(2*l+1)/(2*l1+1)/(2*L+1) * real_gaunt_dict[(L,M,l,ma,l1,m11)]^2;
#                                                end
                                            #                                            end
                                            ta += 4*pi/(2*L+1) * real_gaunt_dict[(L,M,l,m,l1,m1)]*real_gaunt_dict[(L,M,l,m,l2,m2)];
                                        end
                                        
                                        #                                        println("$l $m   $l1 $m1   $L  symfactor $(round(symfactor, digits=5))  old  $(round(symfactor_old,digits=5))  new    ", round(ta *(2*L+1), digits=5))
                                        #                                println("$l $m   $l1 $m1   $L  symfactor $(round(symfactor, digits=5))  old  $(round(symfactor_old,digits=5))  new    ", round(ta *(2*L+1)), digits=5))
                                        #                                println("$l $m   $l1 $m1   $L  symfactor $(round(symfactor, digits=5))  old  $(round(symfactor_old,digits=5))  new    ", round(ta *(2*l+1)*(2*l1+1), digits=5))
                                        #                                println("$l $m   $l1 $m1   $L  symfactor $(round(symfactor, digits=5))  old  $(round(symfactor_old,digits=5))  new    ", round(ta *(2*l+1)*(2*L+1), digits=5))
                                        #println()

                                        
                                        VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -ta *(2*L+1)*f* real(   mat_n2m'*  temp * mat_n2m)/2 
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    for spin = 1:nspin
        for l1 = 0:lmax
            for m1 = -l1:l1
                for l2 = 0:lmax
                    for m2 = -l2:l2
                        t1 = VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1]
                        t2 = VX_LM2[:,:,spin,l2+1,m2+l2+1,l1+1,m1+l1+1]
                        VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] = 0.5*(t1 + t2')
                        VX_LM2[:,:,spin,l2+1,m2+l2+1,l1+1,m1+l1+1] = 0.5*(t1' + t2)
                    end
                end
            end
        end
    end    

#=    
    #okay, there is a sum for L, a sum over b, and then sum over lright and lleft.
    
    for spin = 1:nspin
    
        for l = 0:lmax #this is the sum over b
            for m = -l:l
                for n = 1:N-1 #this is the sum over b as well
                    
                    f = filling[n,spin,l+1,m+l+1]
                    if f < 1e-20
                        break
                    end
                    t = mat_n2m*VECTS[:,n,spin,l+1, m+l+1]
                    tf1 =  t ./ R  / sqrt(g.b-g.a) * sqrt(2 )

                    for L = 0:lmax*2
                        temp = mat_n2m' *diagm(conj.(tf1)) * LINOP[L] * diagm(tf1) * mat_n2m / 2.0


                        for l1 = 0:lmax
                            for m1 = -l1:l1
                                #                                for l2 = 0:lmax
                                #                                    for m2 = -l2:l2
                                ta = 0.0

                                
                                for M = -L:L;
                                    
                                    for m11 = -l1:l1 #this one is wierd. 

                                        for ma = -l:l  #this one is fine. actually, As is this one.

                                            ta += 4*pi/(2*l+1)/(2*l1+1)/(2*L+1) * real_gaunt_dict[(L,M,l,ma,l1,m11)]^2;
                                        end
                                    end
                                end

                                #=for M = -L:L
                                
                                for m11 = -l1:l1 #this one is wierd. 

                                for ma = -l:l  #this one is fine. actually, As is this one.

                                ta += 4*pi/(2*l+1)/(2*l1+1)/(2*L+1) * real_gaunt_dict[(L,M,l,ma,l1,m11)]^2;
                                end
                                end
                                
                                #                                            for m11 = -l1:l1 #this one is wierd.
                                #                                                for m22 = -l2:l2 #this one is wierd. 
                                #                                                    for ma = -l:l  #this one is fine. actually, As is this one.#

                                #                                            for ma = -l:l
                                #                                                for mb = -l:l
                                #                                                    ta += 4*pi/(2*l+1)/(2*l1+1)/(2*L+1) * real_gaunt_dict[(L,M,l,ma,l1,m1)]*real_gaunt_dict[(L,M,l,m11,l2,m2)]
                                #                                                end
                                #                                            end
                                end=#
                                #VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -f*ta* real( temp)
                                VX_LM2[:,:,spin,l1+1,m1+l1+1,l1+1,m1+l1+1] += -f*ta* real( temp)
                            end
                        end
#                            end
#                        end
                    end
                end
            end
        end
    end
    for spin = 1:nspin
        for l1 = 0:lmax
            for m1 = -l1:l1
#                for l2 = 0:lmax
                #                    for m2 = -l2:l2
                #VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] = 0.5 *(VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] + VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1]')

                VX_LM2[:,:,spin,l1+1,m1+l1+1,l1+1,m1+l1+1] = 0.5 *(VX_LM2[:,:,spin,l1+1,m1+l1+1,l1+1,m1+l1+1] + VX_LM2[:,:,spin,l1+1,m1+l1+1,l1+1,m1+l1+1]')                
#                    end
#                end
            end
        end
    end    

    =#

    if nspin == 1
        return VX_LM2*exx/2.0
    else
        return VX_LM2*exx
    end
    
end

function vxx_LM3( D2, g, N, M, lmaxrho, lmax, MP, V_L, gbvals2, exx, nmax, nspin, filling, VECTS, S, rho_dR)


    mat_n2m = MAT_N2M(g, N=N, M=M)
    mat_m2n = MAT_M2N(g, N=N, M=M)
    R = g.R.(g.pts[2:M+2,M])

    vx2 =  zeros(N-1,N-1)
    VX_LM2 = zeros(N-1, N-1, nspin, lmax+1, 2*lmax+1, lmax+1, 2*lmax+1)
    #VX_LM2 = zeros(N-1, N-1, nspin, lmax+1, 2*lmax+1)


    #    r5 = mat_n2m'*diagm(1 ./ R.^0.5)*mat_n2m
    r = mat_n2m'*diagm( R)*mat_n2m
    r2 = mat_n2m'*diagm( R.^2 .* g.w[2:M+2,M] )*mat_n2m
    #println("size(r) ", size(r))

    LINOP = Dict()
    for L = 0:lmax*2
        Linv = inv((D2 + L*(L+1)*V_L))  
        LINOP[L] = mat_m2n' * S* Linv *S  *  mat_m2n 
    end

    nel = sum(filling, dims=[1,3,4])
    if nspin == 1
        nel = nel / 2
    end
    println("nel $nel")

    #invS = S^-1
    
    #okay, there is a sum for L, a sum over b, and then sum over lright and lleft.
    
    for spin = 1:nspin
    
        for l = 0:lmax #this is the sum over b
            for m = -l:l
                for n = 1:N-1 #this is the sum over b as well
                    
                    f = filling[n,spin,l+1,m+l+1]
                    if f < 1e-20
                        break
                    end
                    t = mat_n2m*VECTS[:,n,spin,l+1, m+l+1]
                    println("test $l $m $n $f $(sum(abs.(t)))")
                    
                    tf1 =  t ./ R  / sqrt(g.b-g.a) * sqrt(2 )

                    
                    MP_x = zeros(lmax*2*2+1)
                    
                    #for L = [0]
                    for L = 0:lmax*2
                        #MP_x[L+1] = f * sum( ( t.*conj(t) .*R.^L .* g.w[2:M+2,M])) / (2*L +1)
                        #rho_gal_multipole_M = f * (t.*conj(t) .*R.^L ) / (2*L+1) / (g.b-g.a) * 2
                        #rho_gal_multipole_N = mat_m2n * rho_gal_multipole_M

                        #MP_x[L+1] = do_1d_integral(rho_gal_multipole_N, g)
                        #MP_x[L+1] = sum( (mat_n2m*rho_gal_multipole_N) .* g.w[2:M+2,M]  ) * (g.b-g.a)/2
                        MP_x[L+1] = f * sum( ( t.*conj(t) .*R.^L .* g.w[2:M+2,M])) / (2*L +1)
                        println("MP x $l $m $n,   $L ", MP_x[L+1])
                        #println("MP_x L $L ", MP_x[L+1], " alt ", f * sum( ( t.*conj(t) .*R.^L .* g.w[2:M+2,M])) / (2*L +1))
                    end
                    #println()
                    
                    S5 = S^0.5

                    MAT1 = sqrt(4*pi)*real.(S*VECTS[:,n,spin,l+1, m+l+1]* (VECTS[:,n,spin,l+1, m+l+1])')  #sqrt(4*pi)*


                    #MAT1 = sqrt(4*pi)*S*VECTS[:,n,spin,l+1, m+l+1]* (VECTS[:,n,spin,l+1, m+l+1])'

                    
                    for L =  0:lmax*2

                        #temp = diagm(conj.(tf1)) * (LINOP[L]  + 0.5*diagm(R.*MP_x[L+1]/g.b^(L+1) * sqrt(pi)/(2*pi)  .* (@view g.w[2:M+2,M]))) * diagm(tf1)

                        temp = diagm(conj.(tf1)) * LINOP[L] * diagm(tf1)

#                        temp = diagm(conj.(t)) *diagm( 1 ./ R) * LINOP[L]  *diagm( 1 ./ R)* diagm(t) * 2 / (g.b-g.a)
                        

                        
                        #println("size ", size(tf1), size(LINOP[L]))
                        INT = zeros(N-1, N-1)
                        toadd = zeros(M+1)
#                        if L == 0
#                            toadd[:] =  MP_x[L+1]/g.b^(L+1) * sqrt(pi)/(2*pi)  .* (@view g.w[2:M+2,M]) 
#                        else
#                            toadd[:] =  MP_x[L+1]/g.b^(L+1) * sqrt(pi)/(2*pi) *  (R / g.b).^L   .* g.w[2:M+2,M]  #* (2*L+1)
#                        end

                        if L == 0
                            toadd[:] =  MP_x[L+1]/g.b^(L+1) * sqrt(pi)/(2*pi) * (R / g.b).^L .* (@view g.w[2:M+2,M]) 
                        else
                            toadd[:] =  MP_x[L+1]/g.b^(L+1) * sqrt(pi)/(2*pi) *  (R / g.b).^L  .* g.w[2:M+2,M]  * (2*L+1) 
                        end

#                        temp += diagm(conj.(t)) * diagm(toadd .* t.^-2)   * diagm(t) * 2 / (g.b-g.a)
                        

                        #                        temp = diagm(conj.(tf1)) * (LINOP[L] + diagm( 20.0 *R.^2 .* toadd .* t.^-2 .* g.w[2:M+2,M] ))  * diagm(tf1)


                        #temp = diagm(conj.(tf1)) * (LINOP[L] )  * diagm(tf1) + 1.0*diagm(toadd) / (4*pi) * sqrt(pi) * 2 * pi /g.b * 0.625 / f
                        
#                        asdf = mat_m2n*diagm(toadd)*mat_m2n'
#                        println("size asdf ", size(asdf))
#                        println("s ", size(real.(S*VECTS[:,n,spin,l+1, m+l+1])* (S*VECTS[:,n,spin,l+1, m+l+1])'))

                        #MAT1 = sqrt(4*pi)*(mat_n2m'*diagm(toadd)*mat_m2n') *real.(S*VECTS[:,n,spin,l+1, m+l+1])* (VECTS[:,n,spin,l+1, m+l+1])'  #sqrt(4*pi)*
                        #MAT1 = real(sqrt(4*pi)*diagm(S*VECTS[:,n,spin,l+1, m+l+1])*(mat_m2n*diagm(toadd)*mat_m2n')*diagm((S*VECTS[:,n,spin,l+1, m+l+1])))

#                        MAT1 = real( 1/20 * 2 * (S*VECTS[:,n,spin,l+1, m+l+1])* (S*VECTS[:,n,spin,l+1, m+l+1])')

                        #MAT = 0.0
#                        MAT = MAT1
                        
                        for n1 = 1:(N-1) #
                            for n2 = 1:(N-1)
                                for i = 1:(M+1)
                                    INT[n1,n2] += gbvals2[i,n1,n2] * toadd[i]
                                end
                            end
                        end
#
                        MAT =  0.5*(MAT1 *INT  + (MAT1 *INT )') 
                        #mat_m2n' * S* INT *S  *  mat_m2n

#                        temp2 = diagm(conj.(tf1)) * mat_m2n' * S* INT *S  *  mat_m2n * diagm(tf1)  
#                        MAT = mat_n2m'*  temp2 * mat_n2m / 2
 #                       MAT1 = sqrt(4*pi)*diagm(VECTS[:,n,spin,l+1, m+l+1])*INT*diagm(VECTS[:,n,spin,l+1, m+l+1])

                        
                        #MAT =  (MAT1 *INT)

  #                      MAT = MAT1

                        #MAT =  (INT * MAT1)
                        #MAT = sqrt(4*pi)*(2*L+1)* diagm(VECTS[:,n,spin,l+1, m+l+1])* invS * INT *invS* diagm(VECTS[:,n,spin,l+1, m+l+1])
                        
                        for l1 = 0:lmax
                            for m1 = -l1:l1
                                for l2 = 0:lmax
                                    for m2 = -l2:l2
                                        

                                        #                                        symfactor_old = 0.0

                                        #for M = -L:L
                                        #   symfactor_old += real_gaunt_dict[(L,M,l,m,l1,m1)]*sqrt(4*pi)
                                        #                                    symfactor += real_gaunt_dict[(L,0,l,0,l1,0)]
                                        #                                 end
                                        #symfactor = Float64(wigner3j(l,l1,L,0,0,0))^2 * (2*L+1) 

                                        ta = 0;
                                        for MM = -L:L;
#                                            for m11 = -l1:l1 #this one is wierd. 
#                                                for ma = -l:l  #this one is fine. actually, As is this one.
#                                                    ta += 4*pi/(2*l+1)/(2*l1+1)/(2*L+1) * real_gaunt_dict[(L,M,l,ma,l1,m11)]^2;
#                                                end
                                            #                                            end
                                            ta +=   4*pi/(2*L+1) * real_gaunt_dict[(L,MM,l,m,l1,m1)]*real_gaunt_dict[(L,MM,l,m,l2,m2)] 
                                        end
                                        
                                        #                                        println("$l $m   $l1 $m1   $L  symfactor $(round(symfactor, digits=5))  old  $(round(symfactor_old,digits=5))  new    ", round(ta *(2*L+1), digits=5))
                                        #                                println("$l $m   $l1 $m1   $L  symfactor $(round(symfactor, digits=5))  old  $(round(symfactor_old,digits=5))  new    ", round(ta *(2*L+1)), digits=5))
                                        #                                println("$l $m   $l1 $m1   $L  symfactor $(round(symfactor, digits=5))  old  $(round(symfactor_old,digits=5))  new    ", round(ta *(2*l+1)*(2*l1+1), digits=5))
                                        #                                println("$l $m   $l1 $m1   $L  symfactor $(round(symfactor, digits=5))  old  $(round(symfactor_old,digits=5))  new    ", round(ta *(2*l+1)*(2*L+1), digits=5))
                                        #println()
                                        
                                        
                                        #VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -ta *(2*L+1)*f* real(   mat_n2m'*  temp * mat_n2m)/2  .- ta*MAT #/2*nspin


                                        #VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -ta *(2*L+1)*f* real(   mat_n2m'*  temp * mat_n2m)/2 - ta*S* VECTS[:,n,spin,l+1, m+l+1]* (VECTS[:,n,spin,l+1, m+l+1])'*S* MP_x[L+1] / g.b^(L+1) * sqrt(pi)/(2*pi) *sqrt(4*pi) *(2*L+1)

                                        #r5 = mat_m2n*diagm( (R/g.b).^(L/2.0) ) * mat_n2m
                                        r5 = mat_n2m'*diagm( (R/g.b).^(L/2.0) ) * mat_m2n'
                                        #println("size r5 ", size(r5))
                                        
                                        #                                        VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -ta *(2*L+1)*f* real(   mat_n2m'*  temp * mat_n2m)/2 - ta*r5'*S* VECTS[:,n,spin,l+1, m+l+1]* (VECTS[:,n,spin,l+1, m+l+1])'*S*r5 * MP_x[L+1] / g.b^(L+1) * sqrt(pi)/(2*pi) *sqrt(4*pi) *(2*L+1)
                                        VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -ta *(2*L+1)*f* real(   mat_n2m'*  temp * mat_n2m)/2 - r5*S* VECTS[:,n,spin,l+1, m+l+1]* (VECTS[:,n,spin,l+1, m+l+1])'*S'*r5' * MP_x[L+1] / g.b^(L+1) * sqrt(pi)/(2*pi)  *(2*L+1) * sqrt(4*pi) *ta 
                                        
                                            



                                        #VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -ta *(2*L+1)*f* real(   mat_n2m'*  temp * mat_n2m)/2  #- f*ta*MAT/2*nspin

#                                        VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -0.2 * VECTS[:,n,spin,l+1, m+l+1]*VECTS[:,n,spin,l+1, m+l+1]'

#                                        VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -ta *(2*L+1)*f* real(   mat_n2m'*  temp * mat_n2m)/2  - (2*L+1)*ta*INT  * sqrt(4*pi) * diagm(VECTS[:,n,spin,l+1, m+l+1])*diagm(VECTS[:,n,spin,l+1, m+l+1])
                                        #println("INT L $L  ", ta*(2*L+1)*sqrt(4*pi)*INT[1:3,1], " ta $ta $(2*L+1) " )
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    for spin = 1:nspin
        for l1 = 0:lmax
            for m1 = -l1:l1
                for l2 = 0:lmax
                    for m2 = -l2:l2
                        t1 = VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1]
                        t2 = VX_LM2[:,:,spin,l2+1,m2+l2+1,l1+1,m1+l1+1]
                        println("t1t2 $l1 $m1 $l2 $m2 ", sum(abs.(t1 - t2)), " diff  " , sum(abs.(t1 - t2')), " tot ", sum(abs.(t1)))
                        VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] = 0.5*(t1 + t2')
                        VX_LM2[:,:,spin,l2+1,m2+l2+1,l1+1,m1+l1+1] = 0.5*(t1' + t2)
                    end
                end
            end
        end
    end    

#=    
    #okay, there is a sum for L, a sum over b, and then sum over lright and lleft.
    
    for spin = 1:nspin
    
        for l = 0:lmax #this is the sum over b
            for m = -l:l
                for n = 1:N-1 #this is the sum over b as well
                    
                    f = filling[n,spin,l+1,m+l+1]
                    if f < 1e-20
                        break
                    end
                    t = mat_n2m*VECTS[:,n,spin,l+1, m+l+1]
                    tf1 =  t ./ R  / sqrt(g.b-g.a) * sqrt(2 )

                    for L = 0:lmax*2
                        temp = mat_n2m' *diagm(conj.(tf1)) * LINOP[L] * diagm(tf1) * mat_n2m / 2.0


                        for l1 = 0:lmax
                            for m1 = -l1:l1
                                #                                for l2 = 0:lmax
                                #                                    for m2 = -l2:l2
                                ta = 0.0

                                
                                for M = -L:L;
                                    
                                    for m11 = -l1:l1 #this one is wierd. 

                                        for ma = -l:l  #this one is fine. actually, As is this one.

                                            ta += 4*pi/(2*l+1)/(2*l1+1)/(2*L+1) * real_gaunt_dict[(L,M,l,ma,l1,m11)]^2;
                                        end
                                    end
                                end

                                #=for M = -L:L
                                
                                for m11 = -l1:l1 #this one is wierd. 

                                for ma = -l:l  #this one is fine. actually, As is this one.

                                ta += 4*pi/(2*l+1)/(2*l1+1)/(2*L+1) * real_gaunt_dict[(L,M,l,ma,l1,m11)]^2;
                                end
                                end
                                
                                #                                            for m11 = -l1:l1 #this one is wierd.
                                #                                                for m22 = -l2:l2 #this one is wierd. 
                                #                                                    for ma = -l:l  #this one is fine. actually, As is this one.#

                                #                                            for ma = -l:l
                                #                                                for mb = -l:l
                                #                                                    ta += 4*pi/(2*l+1)/(2*l1+1)/(2*L+1) * real_gaunt_dict[(L,M,l,ma,l1,m1)]*real_gaunt_dict[(L,M,l,m11,l2,m2)]
                                #                                                end
                                #                                            end
                                end=#
                                #VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -f*ta* real( temp)
                                VX_LM2[:,:,spin,l1+1,m1+l1+1,l1+1,m1+l1+1] += -f*ta* real( temp)
                            end
                        end
#                            end
#                        end
                    end
                end
            end
        end
    end
    for spin = 1:nspin
        for l1 = 0:lmax
            for m1 = -l1:l1
#                for l2 = 0:lmax
                #                    for m2 = -l2:l2
                #VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] = 0.5 *(VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] + VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1]')

                VX_LM2[:,:,spin,l1+1,m1+l1+1,l1+1,m1+l1+1] = 0.5 *(VX_LM2[:,:,spin,l1+1,m1+l1+1,l1+1,m1+l1+1] + VX_LM2[:,:,spin,l1+1,m1+l1+1,l1+1,m1+l1+1]')                
#                    end
#                end
            end
        end
    end    

    =#

    if nspin == 1
        return VX_LM2*exx/2.0
    else
        return VX_LM2*exx
    end
    
end

function vxx_LM5(VX_LM2, mat_n2m, mat_m2n, R, LINOP, g, N, M, lmaxrho, lmax, gbvals2, exx, nspin, filling, VECTS, S, hf_sym, R5)

    VX_LM2 .= 0.0

    MP_x = zeros(lmax*2+1)

    temp = zeros(N-1,N-1)

    MAT = zeros(N-1, N-1)
    MATL = zeros(N-1, N-1)
    t = zeros(M+1)
    tf1 = zeros(M+1)
    mt = zeros(N-1, M+1)
    #mt1 = zeros(N-1, M+1)
    mt1 = zeros(N-1)

    RMAT = []
    for L = 0:lmax*2
        push!(RMAT, get_gal_rep_matrix_R(R.^L / (2*L+1), g, gbvals2, N = N))
    end
    
    #println("vxx")

#    S5 = S^-0.5

#=    for spin = 1:nspin
        for l = 0:lmax #this is the sum over b
            for m = -l:l
                for n = 1:N-1 #this is the sum over b as well
                    
                    f = filling[n,spin,l+1,m+l+1]
                    if f < 1e-20
                        break
                    end
                    for l1 = 0:lmax #this is the sum over b
                        for m1 = -l1:l1
                            for n1 = 1:N-1 #this is the sum over b as well
                                
                                f1 = filling[n1,spin,l1+1,m1+l1+1]
                                if f1 < 1e-20
                                    break
                                end
#                                for L = 0:lmax*2
#                                    println("test L $L,  $n $l $m , $n1 $l1 $m1 ", (VECTS[:,n,spin, l+1, m+l+1]'*R5[L]*VECTS[:,n1,spin,l1+1, m1+l1+1]))
#                                end
#                                println()
                            end
                        end
                    end
                end
            end
        end
    end
=#
                                    
    
    for spin = 1:nspin
        for l = 0:lmax #this is the sum over b
            for m = -l:l
                for n = 1:N-1 #this is the sum over b as well
                    
                    f = filling[n,spin,l+1,m+l+1]
                    if f < 1e-20
                        break
                    end
                    t .= real(mat_n2m*(@view VECTS[:,n,spin,l+1, m+l+1]))
                    tf1 .=  t ./ R  / sqrt(g.b-g.a) * sqrt(2 )
                    mt .= mat_n2m'*diagm(tf1)
#                    mt1 .= mat_n2m'*diagm(t) / sqrt(g.b-g.a) * sqrt(2 )
                    MP_x .= 0.0
                    #MP_x[1] = nspin
                    #   for L = 0:0
                    
                    for L = 0:lmax*2
                    
                        MP_x[L+1] = f * sum( ( t.*conj(t) .*R.^L .* g.w[2:M+2,M])) / (2*L +1)
                    end


#                    MAT .= S* (@view VECTS[:,n,spin,l+1, m+l+1])* (@view VECTS[:,n,spin,l+1, m+l+1])'*S'
#                    MAT .= S* (@view VECTS[:,n,spin,l+1, m+l+1])* (@view VECTS[:,n,spin,l+1, m+l+1])'*S'                    

                    for L =  0:lmax*2

#                        MP_x = f * sum( ( t.*conj(t) .*R.^L .* g.w[2:M+2,M])) / (2*L +1)  #* factor

                        temp .=  mt * LINOP[L] * mt' *( (2*L+1)*f/2.0 ) #+ MATL 

#                        println("test s$spin   $l $m $n, $L ", MP_x * (2*L+1)/f / g.b^L, " ", (t'*diagm(R.^L .* g.w[2:M+2,M])*t/g.b^L ), " ", (VECTS[:,n,spin, l+1, m+l+1]'*R5[L]*VECTS[:,n,spin,l+1, m+l+1]))
                        
                        #                        MATL .= (VECTS[:,n,spin, l+1, m+l+1]'*R5[L]*VECTS[:,n,spin,l+1, m+l+1]) * S*VECTS[:,n,spin,l+1, m+l+1]*(S*VECTS[:,n,spin,l+1, m+l+1])'    *(  MP_x / g.b^(L+1) * sqrt(pi)/(2*pi)  *(2*L+1) * sqrt(4*pi)  )

                        #MATL .= (VECTS[:,n,spin, l+1, m+l+1]'*R5[L]*VECTS[:,n,spin,l+1, m+l+1]) * S*VECTS[:,n,spin,l+1, m+l+1]*(S*VECTS[:,n,spin,l+1, m+l+1])'    *(  MP_x[L+1] / g.b^(L+1) * sqrt(pi)/(2*pi)  *(2*L+1) * sqrt(4*pi)  )

                        #MATL .= (VECTS[:,n,spin, l+1, m+l+1]'*R5[L]*VECTS[:,n,spin,l+1, m+l+1]) * S*VECTS[:,n,spin,l+1, m+l+1]*(S*VECTS[:,n,spin,l+1, m+l+1])'    *(  MP_x / g.b^(L+1) * sqrt(pi)/(2*pi)  *(2*L+1) * sqrt(4*pi)  )                        

#                        if l == L

                        #                        MATL .=  real(S*VECTS[:,n,spin,l+1, m+l+1]*(S*VECTS[:,n,spin,l+1, m+l+1])')    *(  MP_x[L+1]*MP_x[L+1] *(2*L+1)/f/g.b^L / g.b^(L+1) * sqrt(pi)/(2*pi)  *(2*L+1) * sqrt(4*pi)  )

                        #                        MATL .=  real(S*VECTS[:,n,spin,l+1, m+l+1]*(S*VECTS[:,n,spin,l+1, m+l+1])')    *(  MP_x[L+1]*MP_x[L+1] *(2*L+1)/f/g.b^L / g.b^(L+1) * sqrt(pi)/(2*pi)  *(2*L+1) * sqrt(4*pi)  )







                        #best guess
#                        MATL .=  real(S*VECTS[:,n,spin,l+1, m+l+1]*(S*VECTS[:,n,spin,l+1, m+l+1])')    *(  MP_x[L+1]*MP_x[L+1] *(2*L+1)/f/g.b^L / g.b^(L+1) * sqrt(pi)/(2*pi)  *(2*L+1) * sqrt(4*pi)  )

                        #new
                        MATL .= real(RMAT[L+1]*VECTS[:,n,spin,l+1, m+l+1]*(RMAT[L+1]*VECTS[:,n,spin,l+1, m+l+1])') * f / g.b^L / g.b^(L+1) * sqrt(pi)/(2*pi) * sqrt(4*pi) * (2*L+1)^2
                        
#                        else
#                            MATL .=  real(S*VECTS[:,n,spin,l+1, m+l+1]*(S*VECTS[:,n,spin,l+1, m+l+1])')    *(  MP_x[L+1]^2 *(2*L+1)/f/g.b^L / g.b^(L+1) * sqrt(pi)/(2*pi)  *(2*L+1) * sqrt(4*pi)  ) 
#                        end
                        
                        #MATL .=  diagm(S*VECTS[:,n,spin,l+1, m+l+1])*R5[L]*diagm(S*VECTS[:,n,spin,l+1, m+l+1])    *(  MP_x[L+1]  / g.b^(L+1) * sqrt(pi)/(2*pi)  *(2*L+1) * sqrt(4*pi)  ) 

                        #println("go spin $spin,  $n, $l $m, L $L ", sum(abs.(MATL)))

                        #MATL .=  real(S*VECTS[:,n,spin,l+1, m+l+1]*(S*VECTS[:,n,spin,l+1, m+l+1])')    *(  MP_x[L+1]^2 *(2*L+1)^1/(2*L+1)/f/g.b^L / g.b^(L+1) * sqrt(pi)/(2*pi)  *(2*L+1) * sqrt(4*pi)  )
                        
                        #MATL .= (t'*diagm(R.^L .* g.w[2:M+2,M])*t/g.b^L ) * S*VECTS[:,n,spin,l+1, m+l+1]*(S*VECTS[:,n,spin,l+1, m+l+1])'    *(  MP_x / g.b^(L+1) * sqrt(pi)/(2*pi)  *(2*L+1) * sqrt(4*pi)  )

                        
                        #MATL .=  S*VECTS[:,n,spin,l+1, m+l+1]*(S*VECTS[:,n,spin,l+1, m+l+1])'    *(  MP_x / g.b^(L+1) * sqrt(pi)/(2*pi)  *(2*L+1) * sqrt(4*pi)  )        
                        for l1 = 0:lmax
                            for m1 = -l1:l1
                                for l2 = 0:lmax
                                    for m2 = -l2:l2
                                        ta = hf_sym[l+1, l+m+1,l1+1,m1+l1+1,l2+1,m2+l2+1, L+1]
                                        for ii = 1:N-1
                                            for jj = 1:N-1
                                                #                                                VX_LM2[ii,jj,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -ta*(temp[ii,jj])
                                                VX_LM2[ii,jj,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -ta*temp[ii,jj] - ta*MATL[ii,jj] #* pi^L #/7.9314220444898185 
                                            end
                                        end
#                                        if l1 == l2 && m1 == m2
#                                            println("factor $l $m, $l1 $m1,  $l2 $m2  L  $L  ", (  MP_x[L+1] / g.b^(L+1) * sqrt(pi)/(2*pi)    ) ,  "  ta $ta  MP_x $(MP_x[L+1]) tot ", sum(abs.(ta*MATL)))
#                                        end
                                        
                                    end
                                end
                            end
                        end
#                        println()
                        
                    end
                    
                end
            end
        end
    end

    
    t1 = zeros(N-1,N-1)
    t2 = zeros(N-1,N-1)
    
    for spin = 1:nspin
        for l1 = 0:lmax
            for m1 = -l1:l1
                for l2 = 0:lmax
                    for m2 = -l2:l2
                        t1 .= (@view VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1])
                        t2 .= (@view VX_LM2[:,:,spin,l2+1,m2+l2+1,l1+1,m1+l1+1])
#                        println("t1t2 spin $spin,  $l1 $m1,  $l2 $m2    ", sum(abs.(t1 - t2)), "     diff     " , sum(abs.(t1 - t2')), "     tot      ", sum(abs.(t1)))
                        VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] .= 0.5*(t1 + t2')
                        VX_LM2[:,:,spin,l2+1,m2+l2+1,l1+1,m1+l1+1] .= 0.5*(t1' + t2)
                    end
                end
            end
        end
    end    
    
    if nspin == 1
        VX_LM2 .= VX_LM2 * exx/2.0
    else
        VX_LM2 .= VX_LM2 * exx
    end


end


function vxx_LM5_big(VX_LM2, mat_n2m, mat_m2n, R, LINOP, g, N, M, lmaxrho, lmax, gbvals2, exx, nspin, filling, VECTS, S, hf_sym_big, R5, big_code, lm_dict)

    #println("declare")
    begin
        VX_LM2 .= 0.0

        #    MP_x = zeros(lmax*2+1)

        temp = zeros(N-1,N-1)

        MAT = zeros(N-1, N-1)
        MATL = zeros(N-1, N-1)
        t = zeros(M+1)
        tf1 = zeros(M+1)
        mt = zeros(N-1, M+1)

        ta = zeros(M+1)
        tf1a = zeros(M+1)
        mta = zeros(N-1, M+1)

        #mt1 = zeros(N-1, M+1)
        mt1 = zeros(N-1)
        mt1a = zeros(N-1)

        RMAT = []
        for L = 0:lmax*2
            push!(RMAT, get_gal_rep_matrix_R(R.^L / (2*L+1), g, gbvals2, N = N))
        end
    end

    #println("loop1")
    for spin = 1:nspin
        for lX = 0:lmax #this is the sum over b
            for mX = -lX:lX
                nlist = big_code[(spin,lX, mX)]
                for (n_count, n) = enumerate(nlist) #this is the sum over b as well
                    f = filling[n_count,spin,lX+1,mX+lX+1]
                    if f < 1e-20
                        break
                    end

                    for l = 0:(lmax)
                        for m = -l:l
                            
                            ind = (1:(N-1)) .+ (N-1)*lm_dict[(l,m)]

                            temptemp = VECTS[ind,n,spin]' * S * VECTS[ind,n,spin]
#                            if abs(temptemp) > 1e-8
#                                println("VECT $spin, n $n_count $n, $lX $mX, $l $m ", temptemp, " ind $#ind")
#                            end
                            
                            if abs.(VECTS[ind,n,spin]' * S * VECTS[ind,n,spin]) < 1e-10
                                continue
                            end
                            
                            t .= real(mat_n2m*(@view VECTS[ind,n,spin]))
                            tf1 .=  t ./ R  / sqrt(g.b-g.a) * sqrt(2 )
                            mt .= mat_n2m'*diagm(tf1)

                                
#                            for la = [l]
#                                for ma = [m]
                            
                            for la = 0:(lmax)
                                for ma = -la:la
                                    #la = l
                                    #ma = m

                                    inda = (1:(N-1)) .+ (N-1)*lm_dict[(la,ma)]

                                    if abs.(VECTS[inda,n,spin]' * S * VECTS[inda,n,spin]) < 1e-10
                                        continue
                                    end

                                    
                                    #println("spin $spin, x $lX $mX, n ($n, $n_count), $l $m $la $ma      ", round(VECTS[ind,n,spin]'*S * VECTS[inda,n,spin], digits=5)  )
                                    
                                    ta .= real(mat_n2m*(@view VECTS[inda,n,spin]))
                                    tf1a .=  ta ./ R  / sqrt(g.b-g.a) * sqrt(2 )
                                    mta .= mat_n2m'*diagm(tf1a)
                                    
                                    #                            MP_x .= 0.0
                                    
                                    #                            for L = 0:lmax*2
                                    #                                
                                    #                                MP_x[L+1] = f * sum( ( t.*conj(t) .*R.^L .* g.w[2:M+2,M])) / (2*L +1)
                                    #                            end


                                    for L =  0:lmax*2


                                        temp .=  mt * LINOP[L] * mta' *( (2*L+1)*f/2.0 ) #+ MATL 
                                        
                                        MATL .= real(RMAT[L+1]*VECTS[ind,n,spin]*(RMAT[L+1]*VECTS[inda,n,spin])') * f / g.b^L / g.b^(L+1) * sqrt(pi)/(2*pi) * sqrt(4*pi) * (2*L+1)^2
                                        
                                        for l1 = 0:lmax
                                            for m1 = -l1:l1
                                                for l2 = 0:lmax
                                                    for m2 = -l2:l2
                                                        tb = hf_sym_big[l+1, l+m+1,la+1, la+ma+1,l1+1,m1+l1+1,l2+1,m2+l2+1, L+1]
                                                        @tturbo for ii = 1:N-1
                                                            for jj = 1:N-1
                                                                #                                                VX_LM2[ii,jj,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -ta*(temp[ii,jj])
                                                                VX_LM2[ii,jj,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] += -tb*temp[ii,jj] - tb*MATL[ii,jj] #* pi^L #/7.9314220444898185 
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    t1 = zeros(N-1,N-1)
    t2 = zeros(N-1,N-1)
    #println("loop2")
    for spin = 1:nspin
        for l1 = 0:lmax
            for m1 = -l1:l1
                for l2 = 0:lmax
                    for m2 = -l2:l2
                        

                        t1 .= (@view VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1])
                        t2 .= (@view VX_LM2[:,:,spin,l2+1,m2+l2+1,l1+1,m1+l1+1])

#                        println("t1t2 spin $spin,  $l1 $m1,  $l2 $m2    ", sum(abs.(t1 - t2)), "     diff     " , sum(abs.(t1 - t2')), "     tot      ", sum(abs.(t1)))
                        VX_LM2[:,:,spin,l1+1,m1+l1+1,l2+1,m2+l2+1] .= 0.5*(t1 + t2')
                        VX_LM2[:,:,spin,l2+1,m2+l2+1,l1+1,m1+l1+1] .= 0.5*(t1' + t2)
                    end
                end
            end
        end
    end    

                        
    #println("spin")
    if nspin == 1
        factor =  exx/2.0
    else
        factor =  exx
    end

    @tturbo for i = 1:length(VX_LM2)
        VX_LM2[i] = VX_LM2[i]*factor
    end


end


function vxx_LM2( D2, g, N, M, lmaxrho, lmax, MP, V_L, gbvals2, exx, nmax, nspin, filling, VECTS, S, rho_dR)


    mat_n2m = MAT_N2M(g, N=N, M=M)
    mat_m2n = MAT_M2N(g, N=N, M=M)
    R = g.R.(g.pts[2:M+2,M])
    l=0; m=0
    

    t = mat_n2m*real(VECTS[:,1,1,1,1])

    t_dR = ( mat_m2n*(t.^2 ./ R  )/ (g.b-g.a) * 2 )


    vx2 =  zeros(N-1,N-1)
    VX_LM2 = zeros(N-1, N-1, nspin, lmax+1, 2*lmax+1)



    r = mat_n2m'*diagm( R)*mat_n2m

    L = inv(  (D2 + l*(l+1)*V_L) )
    
    for spin = 1:nspin
        vx2 .= 0.0
        for n = 1:N-1
            f = filling[n,spin,1,1]

            if f < 1e-20
                break
            end
            t = mat_n2m*VECTS[:,n,spin,1,1]

            tf1 =  t ./ R  / sqrt(g.b-g.a) * sqrt(2 )
            #temp = diagm(conj.(tf1)) * mat_m2n' * S* L *S  *  mat_m2n * diagm(tf1)

            vx2[:,:] +=   real(  f *  mat_n2m'*   diagm(conj.(tf1)) * mat_m2n' * S * L * S *  mat_m2n * diagm(tf1)  *mat_n2m) 
            
        end
        vx2 = 0.5*(vx2+vx2')
        VX_LM2[:,:,spin,1,1] = vx2 * -1*sqrt(pi)/(2*pi) / 2
    end


    if nspin == 1
        return VX_LM2*exx/2.0
    else
        return VX_LM2*exx
    end
    
end



function vhart_LM(rho_dR, D2, g, N, M, lmaxrho, lmax, MP, V_L, gbvals2, S, VECTS, loopmax)


    #    println("size rho_dR ", size(rho_dR))
    #    println("lmax rho ", lmaxrho)
    #loopmax = min(lmax*2, lmaxrho)
    #loopmax = lmaxrho
    #loopmax = lmaxrho
    VH_LM = zeros(N-1, N-1, lmaxrho+1, 2*lmaxrho+1)
    VH_LM2 = zeros(N-1, N-1, lmaxrho+1, 2*lmaxrho+1)
    #    println("size VH_LM ", size(VH_LM))

    mat_n2m = MAT_N2M(g, N=N, M=M)
    mat_m2n = MAT_M2N(g, N=N, M=M)

    invD = D2^-1
    R = g.R.(g.pts[2:M+2,M])

    VTILDE = zeros(M+1,  loopmax+1, loopmax*2+1)

    for l = 0:loopmax
        for m = -l:l
            println("size ", size(VH_LM), " ", size(VTILDE), " " , size(rho_dR))
            VH_LM[:, :, l+1, m+l+1], VTILDE[:,l+1,m+l+1] = vhart(rho_dR[:,1,l+1, m+l+1], D2, V_L, g, M, l, m, MP, gbvals2)
#            println("VHART $l $m ", sum(abs.(VH_LM[:, :, l+1, m+l+1])),  "  " ,sum(abs.(rho_dR[:,1,l+1, m+l+1])))
            #, vh_mat2, vt, X
            
            #t = invD*(@view rho_dR[:,1,l+1,m+l+1])
#            println("size t ", size(t), " size  X ", size(X), " size ex ", size(ex))
            #for ii = 1:N-1
            #    for jj = 1:N-1
            #        for bb = 1:N-1
            #            VH_LM2[ii, jj, l+1, m+l+1] += X[ii,jj,bb]*t[bb]
#                    for bb = 1:N-1
#                        for cc = 1:N-1                        
#                            ex[ii,jj] += X[ii,jj,bb]*invD[bb,cc]*rho_dR[cc,1,l+1, m+l+1]
#                        end
#                    end
            #        end
            #    end
            #end
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



#    l=0; m=0
#
#    a, vh_mat2, vt, X = vhart(0.0*rho_dR[:,1,l+1, m+l+1], D2, V_L, g, M, l, m, MP, gbvals2)
###
#
#    tt = MP[l+1, m+l+1]/g.b^(l+1) * g.w[2:M+2,M]


#    INT = zeros(N-1, N-1)
#    for n1 = 1:(N-1) #
#        for n2 = 1:(N-1)
 #           for i = 1:(M+1)
 #               INT[n1, n2] += gbvals2[i,n1,n2] * tt[i]
 #           end
 #       end
 #   end

    
  #  VH_LM2 = VH_LM2 / 4 / sqrt(pi)
  #  VH_LM2[:, :, l+1, m+l+1] += INT/sqrt(4*pi)
    
#    println("test  ", VH_LM2[1:3, 1, l+1, m+l+1] ./ VH_LM[1:3, 1, l+1, m+l+1])
#    println("test2 ", INT[1:3,1] ./ VH_LM[1:3, 1, l+1, m+l+1] / sqrt(4*pi))

    
    
    return VH_LM, VTILDE
    
end


function vhart(rhor2, D2, V_L, g, M, l, m, MP, gbvals2)

    println("check rho 1D ", do_1d_integral(rhor2[:,1,1,1], g))
    vh_tilde = (D2 + l*(l+1)*V_L) \ rhor2
    vh_tilde = vh_tilde /(4*pi)*sqrt(pi)

#    println("vh_tilde ", sum(abs.(vh_tilde)), " rhor2 ", sum(abs.(rhor2)))
    
    vh_tilde_copy = deepcopy(vh_tilde)

#    println("size vh_tilde ", size(vh_tilde))
    vh_mat, vt = get_vh_mat(vh_tilde, g, l, m, MP, gbvals2, M=M)

    println("sum vt ", sum(vt[:,1,1,1]))
#    println("size1 ", size(inv(D2 + l*(l+1)*V_L)) )
#    println("size2 ", size(rhor2))
#    println("size3 ", size( diagm( 1.0 ./ g.R.(@view g.pts[2:M+2,M]) )))

    mat_n2m = MAT_N2M(g, N=size(D2)[1]+1, M=M)
    mat_m2n = MAT_M2N(g, N=size(D2)[1]+1, M=M)

    R = g.R.(@view g.pts[2:M+2,M])


    #vt2 =  diagm( 1.0 ./g.R.(@view g.pts[2:M+2,M])) *mat_n2m* (inv(D2 + l*(l+1)*V_L) + 0.0*MP[l+1, m+l+1]/g.b^(l+1) *diagm(1.0 ./ rhor2   ))  * rhor2 /(4*pi)*sqrt(pi)
    #vt2 =  (      diagm( 1.0 ./R ) * mat_n2m* inv(D2 + l*(l+1)*V_L)*0.0
    #              + MP[l+1, m+l+1]/g.b^(l+1) *diagm(1.0 ./ rhor2   ))  * rhor2 /(4*pi)*sqrt(pi)


    t = mat_n2m * rhor2
    vt2 = diagm( 1.0 ./g.R.(@view g.pts[2:M+2,M])) *mat_n2m* (inv(D2 + l*(l+1)*V_L)) * rhor2 /(4*pi)*sqrt(pi) + 2.0*MP[l+1, m+l+1]/g.b^(l+1) * diagm(t.^-1) *  mat_n2m * rhor2 /(4*pi)*sqrt(pi)
    
    
    #    vt2 =  diagm( 1.0 ./g.R.(@view g.pts[2:M+2,M])) *mat_n2m* (0.0*inv(D2 + l*(l+1)*V_L) + MP[l+1, m+l+1]/g.b^(l+1) *diagm(1.0 ./ rhor2   ))  * rhor2 /(4*pi)*sqrt(pi)

    #vt2 =  (inv(D2 + l*(l+1)*V_L) + 0.0*MP[l+1, m+l+1]/g.b^(l+1)*diagm(1.0 ./ rhor2   ))  * rhor2 /(4*pi)*sqrt(pi)

    #+ sqrt(pi)/(2*pi)*MP[l+1, m+l+1]/g.b^(l+1) * sqrt(pi)/(2*pi) * diagm(1.0 ./ rhor2 .* R))

    #println("size vt  ", size(vt))
    #println("size vt2", size(vt2))
    #println("vt diff ", vt[1:20] - vt2[1:20])
    #println()
    #println("vt div ", vt[1:20] ./ vt2[1:20])
    #println()
    #println("vt1 ", vt[1:10])
    #println()
    #println("vt2 ", vt2[1:10])
    #println()
            
#    println("$l $m size D2 ", size(D2), " size(V_L) " , size(V_L), " size(rhor2) ", size(rhor2), " size vh_mat ", size(vh_mat), " size vh_tilde", size(vh_tilde))
    return vh_mat, vt   # , vh_mat2,vh_tilde_copy, X
    
end

function vxc_LM(rho_rs_M, drho_rs_M, g, M, N, funlist, gga, nspin, lmax, lmaxrho, LEB, R, invS, gbvals2, loopmax)

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
        ERHO_tp = zeros( M+1, LEB.N)
        
        
#        loopmax = min(lmax*2, lmaxrho)
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
            ERHO_tp[:,ntp] = erho
        else
            if nspin == 1
                v = v_LDA.(rho[:, 1, ntp] * sqrt(4*pi)/ (4*pi))
                VRHO_tp[:,1,ntp] = v
                e = e_LDA.(rho[:, 1, ntp] * sqrt(4*pi)/ (4*pi))
                ERHO_tp[:,ntp] = e
            elseif nspin == 2
                v = v_LDA_sp.(rho[:, 1, ntp][:] * sqrt(4*pi)/ (4*pi) ,  rho[:, 2, ntp][:] * sqrt(4*pi)/ (4*pi))
                VRHO_tp[:,1:2,ntp] = reshape(vcat(v...), nspin,M+1)'

                r = rho[:,1,ntp] + rho[:,2,ntp]
                ζ = (rho[:,1,ntp] - rho[:,2,ntp]) ./ r
                
                #                e = e_LDA_sp.( rho[:, 1, ntp][:] * sqrt(4*pi)/ (4*pi) ,  rho[:, 2, ntp][:] * sqrt(4*pi)/ (4*pi))

                e = e_LDA_sp.( r * sqrt(4*pi)/ (4*pi) ,  ζ)
                ERHO_tp[:,ntp] = e
                
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
        EXC_tp = zeros( M+1, LEB.N, nthreads())
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
            EXC_tp[:,ntp,id] += ERHO_tp[:,ntp] *2.0
        else
            VXC_tp[:,:,ntp,id] += VRHO_tp[:,:,ntp]
            EXC_tp[:,ntp,id] += ERHO_tp[:,ntp]
        end

    end
    VXC_tp = sum(VXC_tp, dims=4)
    EXC_tp = sum(EXC_tp, dims=3)
    
    
    
    #now transform to LM space again
    #println("VXC to LM")
    VXC_LM = zeros(M+1, nspin, lmaxrho+1, 2*lmaxrho+1)
    EXC_LM = zeros(M+1, lmaxrho+1, 2*lmaxrho+1)
    for l = 0:2:lmaxrho
        for m = -l:l
            @tturbo for spin = 1:nspin
                for r = 1:M+1
                    for ind = 1:LEB.N
                        VXC_LM[r, spin, l+1, l+m+1] +=  4*pi*LEB.Ylm[(l,m)][ind] * VXC_tp[r,spin,ind] * LEB.w[ind]
                    end
                end
            end
        end
    end

    for l = 0:2:lmaxrho
        for m = -l:l
       @tturbo      for r = 1:M+1
                for ind = 1:LEB.N
                    EXC_LM[r, l+1, l+m+1] +=  4*pi*LEB.Ylm[(l,m)][ind] * EXC_tp[r,ind] * LEB.w[ind]
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
    #    EXC_tp = zeros(N-1,N-1,nspin, lmaxrho+1, 2*lmaxrho+1)
    
    return VXC_LM_MAT, VXC_tp, EXC_tp, VSIGMA_tp, EXC_LM , VXC_LM
    
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

    
#    println("ratio ", VH_LM[1,1,1,1] / VX_LM[1,1,1,1,1])

    Sh = Hermitian(S)
    VECTS_new = zeros(eltype(VECTS), size(VECTS))

    Nsmall = size(Sh,1)
    
    for spin = 1:nspin 
        for l = 0:(lmax)
            V = V_C + V_L*l*(l+1)
            for m = -l:l

                if funlist != :hydrogen  #VHART AND VXC_LM
                    VLM .= 0.0
                    for lr = 0:lmaxrho
                        for mr = -lr:lr
                            gcoef = real_gaunt_dict[(lr,mr,l,m,l,m)]
                            @tturbo for i = 1:Nsmall
                                for j = 1:Nsmall
                                    VLM[i,j] += gcoef * (4*pi*VH_LM[i,j,lr+1,lr+mr+1] + VXC_LM[i,j,spin,lr+1,lr+mr+1])
                                end
                            end
                        end
                    end


                    #                    VLM += 4*pi*VX_LM[:,:,spin, l+1,l+m+1, l+1,l+m+1]/(4*pi)
                    if exx > 1e-12
                        #VLM += 4*pi*VX_LM[:,:,spin, l+1,l+m+1]/(4*pi)
#                        VLM_old = deepcopy(VLM)
                        VLM += VX_LM[:,:,spin, l+1,l+m+1, l+1, l+m+1]

                        #                        println("size VLM_old ", size(VLM_old ), " size VX ", size(VX_LM[:,:,spin, l+1,l+m+1, l+1, l+m+1]))
#                        println("sum abs  XX $l $m  ", sum(abs.( VX_LM[:,:,spin, l+1,l+m+1, l+1, l+m+1])))
#                        println("vh vx diff $l $m  ", sum(abs.(VX_LM[:,:,spin, l+1,l+m+1, l+1, l+m+1] - VLM_old)), " VH_tot $VLM_old VX $(VX_LM[:,:,spin, l+1,l+m+1, l+1, l+m+1]) ")
                        #println("vh vx div ", VX_LM[1:2,1:2,spin, l+1,l+m+1, l+1, l+m+1] ./ (VLM_old[1:2,1:2] .+ 1e-20))
                    end


                    
                end
                
                #println("eigen")

#                println("sum abs Ham ", sum(abs.(D2 + V + VLM)), " ", sum(abs.(S)))

                ##-  real( (Sh*VECTS[:,1,1,1,1])*(Sh*VECTS[:,1,1,1,1])')
                
                Hh = Hermitian(D2 + V + VLM )
                vals, vects = eigen(Hh, Sh)
                #S5 = S^-0.5
                #vals, vects = eigen(S5*(D2 + V + VLM)*S5)
                VECTS_new[:,:,spin, l+1, l+1+m] = vects
                VALS[:,spin, l+1, l+1+m] = vals
            end
        end
    end
    
    return VALS, VECTS_new
    
end


function solve_big(V_C, V_L, VH_LM, VXC_LM, VX_LM, D2, S, nspin, lmax, lmaxrho, funlist, lm_dict, VECTS, VALS,exx, Sbig, symmetry_list)

    VLM = zeros(size(V_C))

    #    println("sub abs VXC_LM ",sum(abs.( VXC_LM)))

    Nsmall = size(V_C)[1]

    c=length(lm_dict)
    Hbig = zeros(Nsmall * c, Nsmall * c)
#    Sbig = zeros(Nsmall * c, Nsmall * c)

    Hh = Hermitian(Hbig)
    Sh = Hermitian(Sbig)
    
    N = Nsmall * c

    VECTS_BIG = zeros(Float64, N,N,nspin)
    VALS_BIG = zeros(Float64, N,nspin)
    VECTS_new = zeros(eltype(VECTS), size(VECTS))

    
#    println("lmax $lmax ")
    big_code = Dict()

    for spin = 1:nspin
        Hbig .= 0.0
#        Sbig .= 0.0
#        println()
        #println("assemble spin $spin")
        for l = 0:(lmax)
            for m = -l:l
                ind1 = (1:Nsmall) .+ Nsmall*lm_dict[(l,m)]
                for l2 = 0:(lmax)
                    for m2 = -l2:l2
                        ind2 = (1:Nsmall) .+ Nsmall*lm_dict[(l2,m2)]
                        #println("l $l m $m l2 $l2 m2 $m2 ----")
                        #println("core")
                        if l == l2 && m == m2
                            Hbig[ind1,ind2] += V_C + V_L*l*(l+1)
                            Hbig[ind1,ind2] += D2
#                            Sbig[ind1,ind2] += S                            
                            #                            println("add S $l $m  $l2 $m2")
                        end

                        
                        
                        #println("more")
                        if funlist != :hydrogen  #VHART AND VXC_LM
                            VLM .= 0.0
#                            println("vlm")
                            for lr = 0:lmaxrho
                                for mr = -lr:lr
                                    gcoef = real_gaunt_arr[lr+1,lr+mr+1,l+1,m+l+1,l2+1,l2+m2+1]
                                    @tturbo for i = 1:Nsmall
                                        for j = 1:Nsmall
                                            VLM[i,j] += gcoef * ((VXC_LM[i,j,spin,lr+1,lr+mr+1])) + gcoef * (4*pi* (VH_LM[i,j,lr+1,lr+mr+1]))
                                        end
                                    end
                                end
                            end
                            #println("vxx")
                            if exx > 1e-12
                                @tturbo for i = 1:Nsmall
                                    for j = 1:Nsmall
                                        VLM[i,j] += (VX_LM[i,j,spin, l+1,l+m+1, l2+1, l2+m2+1])
                                    end
                                end
                            end
#                            if abs( sum(abs.(VLM))) > 1e-10
#                                println("$spin sum abs $l $m   $l2 $m2   ", sum(abs.(VLM)))
#                            end
                            
                            #println("add")
                            @tturbo for i = 1:Nsmall
                                for j = 1:Nsmall
                                    Hbig[ind1[i],ind2[j]] +=  VLM[i,j]
                                end
                            end

                            
                        end
#                        println("--------------")                        
#                        if sum(abs.(Hbig[ind1,ind2])) > 1e-6
#                            println("HAM $spin,   $l $m,  $l2 $m2 ", sum(abs.(Hbig[ind1,ind2])))
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

        begin
            Hh = Hermitian(Hbig)
            Sh = Hermitian(Sbig)
        end

        n = length(lm_dict)
#        println("n $n")



        #        if ismissing(symmetry_list)
        #
        symmetry_list = zeros(Bool, n,n)
        #println("get sym list")
        for n1 = 1:n
            for n2 = 1:n
#                println("n1 $n1 n2 $n2 ", (1:(Nsmall)) .+ (n1-1)*(Nsmall), " x ", (1:(Nsmall)) .+ (n2-1)*(Nsmall))
                
                if maximum(abs.(Hbig[ (1:(Nsmall)) .+ (n1-1)*(Nsmall), (1:(Nsmall)) .+ (n2-1)*(Nsmall)])) > 1e-6
                    symmetry_list[n1, n2] = 1
                end
            end
        end
        #        end

#        println("symmetry_list")
#        println(symmetry_list)

#        println("Nsmall $Nsmall")
#        println("do eig")
        if true
            done_list = []
            VALS_LIST = []
            VECTS_LIST = []
            counter = 0
            for nstart = 1:n
#                println("nstart $nstart")
                if nstart in done_list
                    continue
                end
                ind = symmetry_list[nstart,:]

                indind = Int64[]
                for ii = (1:n)[ind]
                    for i = 1:(Nsmall)
                        push!(indind, i+(ii-1)*(Nsmall))
                    end
                end
#                println("indind ", indind)
                
                Hht = Hermitian(Hbig[indind,indind])
                Sht = Hermitian(Sbig[indind,indind])

                vals, vects = eigen(Hht, Sht)

                VALS_BIG[1+counter:counter+length(indind) ,spin] = vals
                VECTS_BIG[indind,1+counter:counter+length(indind),spin] =vects[:,:]
                counter += length(indind)

#                println("counter $counter")
                
                for i in (1:n)[ind]
                    push!(done_list, i)
                end
#                println("$nstart, donelist ", done_list)
                
            end
#            println("counter final , ",  counter)
        end                    

        
#        #println("eig")


        #vals, vects = eigen(Hh, Sh)

        #println("HAMTEST spin $spin h ", sum(abs.(Hh)), " s ", sum(abs.(Sh)))
        
#        println("store stuff")
        begin

            indperm = sortperm(VALS_BIG[:,spin])
            VALS_BIG[:,spin] = VALS_BIG[indperm,spin]
            VECTS_BIG[:,:,spin] = VECTS_BIG[:,indperm,spin]

#            VALS_BIG[:,spin] = vals
#            VECTS_BIG[:,:,spin] =vects
            
            COUNT = zeros(lmax+1, 2*lmax+1)
            
            Sv = Sh*(@view VECTS_BIG[:,:,spin])
        end

#        println("sort vals")
        vects = VECTS_BIG[:,:,spin]
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
#    println("--------")
    return VALS_BIG, VECTS_BIG, big_code, Hh, Sh, symmetry_list
    
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
        


    end
    

    v +=  vrho

    return v

end

function get_t2(VECTS, mat_n2m, mat_m2n, N, M, nspin, lmax)

    t2 = zeros(M+1, N-1, nspin, lmax+1, 2*lmax+1)
    for spin = 1:nspin
        for l = 0:lmax
            for m = -l:l
                for n = 1:N-1
                    fillval = filling[n, spin, l+1, l+m+1]
                    #                    println("add fillval $spin $l $m ", fillval)
                    if fillval < 1e-20
                        break
                    end
                    t = mat_n2m * VECTS[:, n, spin, l+1,l+1+m]
                    t2[:,n,spin, l+1, l+m+1] = t*conj(t) * fillval
                end
            end
        end
    end
    return t2

end


function dft(; fill_str = missing, g = missing, N = -1, M = -1, Z = 1.0, niters = 50, mix = 0.5, mixing_mode=:pulay, exc = missing, lmax = missing, conv_thr = 1e-7, lmaxrho = 0, mix_lm = false, exx = 0.0, VECTS=missing)

    if M > g.M
        M = g.M
        println("warning, M too large, set to $M")
    end
    if N > g.N
        N = g.N
        println("warning, N too large, set to $N")
    end
    
    if M == -1 
        M = g.M
    end
    if N == -1
        N = g.N
    end

    
    
    println("time prepare")
    @time Z, nel, filling, nspin, lmax, V_C, V_L, D1, D2, S, invsqrtS, invS, VECTS_start, VALS, funlist, gga, LEB, R, gbvals2, nmax, hf_sym, hf_sym_big, mat_n2m, mat_m2n, R5, LINOP, lm_dict, dict_lm, big_code, Sbig = prepare(Z, fill_str, lmax, exc, N, M, g, lmaxrho, mix_lm)

    VECTS_big = zeros(Float64, length(lm_dict) * (N-1), length(lm_dict) * (N-1), nspin)
    VECTS_small = zeros(Float64, N-1,N-1, nspin, lmax+1,2*lmax+1 )


    if ismissing(VECTS)
        VECTS = VECTS_start
    end
    VECTS_new = deepcopy(VECTS)

    big_code_new = deepcopy(big_code)
    VALS_big = zeros(Float64, size(VECTS_new,1) ,nspin)
    
    symmetry_list = missing
    
    Hh = missing
    Sh = missing
    
#    println("lmaxrho $lmaxrho")
    
    #println("vChebyshevDFT.Galerkin.do_1d_integral(VECTS[:,1,1,1], g) ", do_1d_integral(real.(VECTS[:,1,1,1,1]).^2, g))
    
    #println("initial get rho time")
    if mix_lm
        rho_R2, rho_dR, rho_rs_M, drho_rs_M_LM, MP  = get_rho_big(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, D2, lm_dict, dict_lm, big_code, gga, R) 

    else
        rho_R2, rho_dR, rho_rs_M, drho_rs_M_LM, MP = get_rho(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, true, exx, nmax)
    end
#    println("MP")
#    println(MP)

    #   println("done rho")
    #    println("rho_R2 ", rho_R2[1])

    println()
    println("This test should be near integer")
    println("ChebyshevDFT.Galerkin.do_1d_integral(rho[:,1,1,1], g) ", do_1d_integral(rho_R2[:,1,1,1], g))
    println()

    #    println("M $M size rho_rs_M ", size(rho_rs_M))
    
    #    println("start rho ", rho[1:5])
    
    VALS_1 = deepcopy(VALS)

    VH_LM = zeros(N-1,N-1,lmaxrho+1, lmaxrho*2+1)
   #VX_LM = zeros(N-1,N-1,nspin, lmaxrho+1, lmaxrho*2+1)
    l2=lmax*2
    VX_LM = zeros(N-1,N-1,nspin, l2+1, l2*2+1, l2+1, l2*2+1)
    #VX_LM = zeros(1)
    VXC_LM = zeros(N-1,N-1,nspin, lmaxrho+1, lmaxrho*2+1)


    

    vxc_tp = zeros(M+1,nspin, lmaxrho+1, lmaxrho*2+1)
    VSIGMA_tp = zeros(M+1,3, lmaxrho+1, lmaxrho*2+1)

    EXC_LM = zeros(M+1, lmaxrho+1, lmaxrho*2+1)
                   
#    println("iters")
#    H1 = missing
    #    H2 = missing

    #VH_LM0 = vhart_LM(0.0* sum(rho_dR, dims=2), D2, g, N, M, lmaxrho, lmax, MP, V_L,gbvals2, S, VECTS) #ex_factor*

    ex_factor = 1.0
    NEL = sum(nel)
    #ex_factor = exx * (NEL-1)/NEL + (1-exx)
    #ex_factor = 1.0
    #ex_factor = exx*9/10
#    println()
#    println("NEL $nel exx $exx ex_factor $ex_factor")
#    println()

    loopmax = min(lmax*2, lmaxrho)
    Vin = zeros(M+1,nspin, loopmax+1, loopmax*2+1)    
        
    VTILDE = zeros(M+1,  loopmax, loopmax*2+1)

    for iter = 1:niters

        Vin .= 0.0
        Vin[:,1,1,1] = -Z ./ R * 4 *pi / sqrt(pi) 
        if nspin == 2
            Vin[:,2,1,1] = -Z ./ R * 4 *pi / sqrt(pi) 
        end
        
        VALS_1[:,:,:,:] = VALS

#        println("iter $iter")
#        println("vhart")
        if funlist != :hydrogen

            #MP_temp = MP[1]
            #MP .= 0.0
            #MP[1] = MP_temp
            VH_LM_old = deepcopy(VH_LM)
            VH_LM, VTILDE = vhart_LM( sum(rho_dR, dims=2), D2, g, N, M, lmaxrho, lmax, MP*ex_factor, V_L,gbvals2, S, VECTS, loopmax) #ex_factor*

            println("vtilde ", sum( VTILDE[:,1,1,1]))
            Vin[:,1,:,:] += 4 * pi * VTILDE[:,1:loopmax+1, 1:loopmax*2+1] * 2.0 
            if nspin == 2
                Vin[:,2,:,:] += 4 * pi * VTILDE[:,1:loopmax+1, 1:loopmax*2+1] * 2.0
            end            
            #            println()
#            println("sum diff VH_LM ", sum(abs.(VH_LM - VH_LM_old)))
#            println()
            
        else
            VH_LM = zeros(N-1,N-1,lmaxrho+1, lmaxrho*2+1)
        end

        #println("vxc")
        if funlist != :none && funlist != :hydrogen
            VXC_LM, vxc_tp, exc_tp, VSIGMA_tp, EXC_LM, VXC_LM_M = vxc_LM( rho_rs_M, drho_rs_M_LM, g, M, N, funlist, gga, nspin, lmax, lmaxrho, LEB, R, invS, gbvals2, loopmax)

            println("VXC_LM_M ", VXC_LM_M[1])
            Vin += VXC_LM_M[:,:,1:loopmax+1, 1:loopmax*2+1] * 2.0
            
        end

        #println("vxx")
        if exx > 1e-12
            #            println("calculating exact exchange")
            #VX_LM = vxx_LM( psipsi, D2, g, N, M, lmaxrho, lmax, MP, V_L,gbvals2, exx, nmax, nspin, filling, VECTS, S, sum(rho_dR, dims=2))
            #VX_LM_old = deepcopy(VX_LM)
            if mix_lm
                #println("big")
                vxx_LM5_big(VX_LM, mat_n2m, mat_m2n, R, LINOP, g, N, M, lmaxrho, lmax, gbvals2, exx, nspin, filling, VECTS, S, hf_sym_big, R5, big_code, lm_dict)
            else
                vxx_LM5(VX_LM, mat_n2m, mat_m2n, R, LINOP, g, N, M, lmaxrho, lmax, gbvals2, exx, nspin, filling, VECTS, S, hf_sym, R5)
            end
#            println("sum abs.(VX_LM) ", sum(abs.(VX_LM)))
#            println("sum diff VX_LM ", sum(abs.(VX_LM - VX_LM_old)))
            #VX_LM = vxx_LM3( [], D2, g, N, M, lmaxrho, lmax, MP, V_L,gbvals2, exx, nmax, nspin, filling, VECTS, S, sum(rho_dR, dims=2))

        end




        #        println("funlist $funlist")
        #println("solve")
        if mix_lm == false
            VALS, VECTS_new = solve_small(V_C, V_L, VH_LM  , VXC_LM, VX_LM, D2, S, nspin, lmax, lmaxrho, funlist, VECTS, VALS, exx) #+ VH_LM0*ex_factor

        else
            #println("solve big time")            
            VALS_big, VECTS_new, big_code_new, Hh, Sh,symmetry_list = solve_big(V_C, V_L, VH_LM, VXC_LM, VX_LM, D2, S, nspin, lmax, lmaxrho, funlist, lm_dict, VECTS, VALS, exx, Sbig ,symmetry_list)
        end

#        println("mix_lm ", mix_lm)
#        println("rho1")
#        @time if mix_lm == false
            #println("get rho small time")
#            rho_R2_new, rho_dR_new, rho_rs_M_new,  drho_rs_M_LM, MP_new = get_rho(VALS, VECTS_new, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, gga, exx, nmax)

            
#        else
            #println("get rho big time")
#            rho_R2_new, rho_dR_new, rho_rs_M_new, MP_new, drho_rs_M_LM  = get_rho_big(VALS, VALS_BIG, VECTS_new, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, D2, lm_dict, dict_lm, big_code, gga, R) 
        #        end

        
        begin 
#            rho_R2_old = deepcopy(rho_R2)

            #rho_R2 = rho_R2_new * mix + rho_R2 *(1-mix)
            #rho_dR = rho_dR_new * mix + rho_dR * (1-mix)
            #rho_rs_M = rho_rs_M_new * mix + rho_rs_M * (1-mix)

            #MP = MP_new*mix + MP*(1-mix) #this is approximation for higher multipoles.
            

#            t2_new = get_t2(VECTS_new, mat_n2m, mat_m2n, N, M, nspin, lmax)
            
 #           t2 = t2*mix + t2_new * (1-mix)

            #println("mix")
            if mix_lm
                VECTS_old = deepcopy(VECTS)
                big_code_old = deepcopy(big_code)
                mix_vects_big(VECTS, VECTS_new, mix, filling, Sbig, nspin, lmax, N, big_code, big_code_new, VALS, VALS_big)
            else
                mix_vects(VECTS, VECTS_new, mix, filling, S, nspin, lmax, N)
            end                
            #VECTS = VECTS*mix + VECTS_new*(1-mix)

            rho_R2_old = deepcopy(rho_R2)
            #println("get rho")
            if mix_lm == false
                rho_R2, rho_dR, rho_rs_M,  drho_rs_M_LM, MP  = get_rho(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, gga, exx, nmax)
            else
                rho_R2, rho_dR, rho_rs_M,  drho_rs_M_LM, MP  = get_rho_big(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, D2, lm_dict, dict_lm, big_code, gga, R) 
            end                

            #            rho_R2, rho_dR, rho_rs_M,  drho_rs_M_LM, MP                = get_rho(VALS, VECTS, nel, filling, nspin, lmax, lmaxrho, N, M, invS, g, gga, exx, nmax)

#            println("test diff rho ", sum(abs.(rho_R2 - rho_R2_old)))
            
            #            VECTS = VECTS_new
            
            
            
            eigval_diff = maximum(abs.(filling.*(VALS - VALS_1)))
        end
        println("iter $iter eigval_diff $eigval_diff ")

        
        if maximum(abs.(filling.*(VALS - VALS_1))) < conv_thr
            break
        end
        
#        display_eigs(VALS, nspin, lmax)
        println()
            
        
    end
    println("done iters")

#    println("VALS after l0 ", VALS[1:2, 1, 1,1])
#    println("VALS after l1 ", VALS[1:2, 1, 2,2])
    
    if mix_lm
        VECTS_old = deepcopy(VECTS)
        big_code_old = deepcopy(big_code)
        mix_vects_big(VECTS, VECTS_new, mix, filling, Sbig, nspin, lmax, N, big_code, big_code_new, VALS, VALS_big, mixall=true)
    else
        mix_vects(VECTS, VECTS_new, mix, filling, S, nspin, lmax, N, mixall=true)
    end                

    display_eigs(VALS, nspin, lmax)
    println()

    
    etot, e_vxc, e_hart, e_exx, e_ke, e_nuc = calc_energy(rho_rs_M, EXC_LM, funlist, g, N, M, R, Vin, filling, VALS, VTILDE, Z, VX_LM, lmax, VECTS, exx, mix_lm, nspin, big_code, lm_dict)
    println()
    
    #    println("size rho_rs_M", size(rho_rs_M))
    println("size(VECTS ", size(VECTS))
    if mix_lm

        VECTS_big .= VECTS
    else
        VECTS_small .= VECTS
    end

    println(size(VECTS_small))
    println(size(VECTS_big))
    
    dat = make_scf_data(   g,    N,    M,    g.b, g.α,    Z,    nspin,    lmax,    lmaxrho,    mix_lm,    niters,    mix,    mixing_mode,    conv_thr,    fill_str,    nel,    exc,    funlist,    gga,    LEB,    exx,    filling,    VALS,    VECTS_small,    VECTS_big    ,    rho_R2,    big_code,    R,    D1,    D2,    S,    V_C,  V_L,  mat_n2m,    mat_m2n,    dict_lm,lm_dict, etot, e_vxc, e_hart, e_exx, e_ke, e_nuc)
    
    return dat
    
end #end dft




end #end module
