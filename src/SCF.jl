module SCF

using ForwardDiff
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

using ..AngMom:gaunt
using ..AngMom:real_gaunt_dict
using ..AngMom:Y_dict
using ..AngMom:Ytheta_dict
using ..AngMom:Yphi_dict
using ..AngMom:Ytheta2_dict
using ..AngMom:Yphi2_dict

using Base.Threads
import Base.Threads.@spawn

using SphericalHarmonics
using FastSphericalHarmonics
using SphericalHarmonicModes

using ..UseLibxc:set_functional
using ..UseLibxc:EXC
using ..UseLibxc:EXC_sp
using ..UseLibxc:smooth
using ..UseLibxc:getVsigma

include("Atomlist.jl")

#using QuadGK
#using Arpack


function setup_mats(;N = N, a = 0.0, Rmax=Rmax)

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
    D1X = D

    w, pts2 = ChebyshevQuantum.Interp.getW(N)

    w_a = w[2:N]*(b - a)/2.0

    H_poisson = D2_a +  diagm( 2.0 ./ r_a )  * D_a
    
    return r_a, D_a, D2_a, w_a, r, w *(b - a)/2.0, H_poisson, D1X, D2X

    
end

function getVcoulomb(r_a, Z, l)
    
    return diagm( (-Z)./r_a + 0.5*(l)*(l+1)./r_a.^2)
    
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


function get_initial_rho(rho_init, N, Z, nel, r, w; nspin=1, checknorm=true, ig1=missing)

    if ismissing(ig1)
        ig1 = ones(size(r))
    end
    
    
    if !ismissing(rho_init)
        if typeof(rho_init) <: Array
            println("a")
            if length(rho_init[:,1]) != N+1
                println("make")
                C = ChebyshevQuantum.Interp.make_cheb(rho_init[:], a = 0.0, b = r[end]);
                rho = C.(r)
                if nspin == 2
                    rho = [ rho  rho]
                end
            else
                rho = copy(rho_init)
                if nspin == 2
                    rho = [rho  rho]
                end
            end
        else
            rho = rho_init.(r)
            if nspin == 2
                rho = [ rho  rho]

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
                elseif l == 4
                    maxfill = 18/ nspin
                end
#                println("fill $n $l $maxfill")
                psi = h_rad(n, l;Z=Z)
                if nleft <= maxfill + 1e-15
                    #                    println("size rho $(size(rho)) r $(size(r)) psir $(size(psi.(r))) ")
                    rho[:,spin] += psi.(r).^2 * nleft
                    break
                else
                    rho[:,spin] += psi.(r).^2 * maxfill
                    nleft -= maxfill
                end
            end
        end
        rho = rho / (4*pi)
    end        
    
    if checknorm
        #check normalization
        for spin = 1:nspin
            norm = 4 * pi * sum(rho[:,spin] .* r.^2 .* w .* ig1)
            
            if abs(norm - sum(nel[:,spin])) > 1e-2
                println("spin=$spin INITIAL rho check norm $norm, we renormalize")
            end
            rho[:,spin] = rho[:,spin] / norm * sum(nel[:,spin])
#            println("spin=$spin POST-INITIAL check norm ",  4 * pi * sum(rho[:,spin] .* r.^2 .* w .* ig1) )
            
        end
    end
    
#    if nspin == 1
#        rho_cheb = [ChebyshevQuantum.Interp.make_cheb(rho[:,1])]
#    else
#        rho_cheb = [ChebyshevQuantum.Interp.make_cheb(rho[:,1]);ChebyshevQuantum.Interp.make_cheb(rho[:,1])]
#    end
        
    return rho
    
end



function calc_energy(rho, N, Rmax, rall, wall, VH, filling, vals_r, Vin, Z; nspin=1, lmax=0, ig1=missing)

    if ismissing(ig1)
        ig1 = ones(size(rall))
    end
    wall = wall .* ig1
    
    rho_tot = sum(rho, dims=2)
    
    ELDA = calc_energy_lda(rho, N, Rmax, rall, wall,  nspin=nspin)

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

function calc_energy_LM(rho_LM, drho_LM, N, Rmax, rall, wall, VH, filling, vals_r, Vin, Z, D1Xgrid; nspin=1, lmax=0, ig1=missing, funlist=funlist, gga=gga, verbose=true, vext = missing, hydrogen=false)

    if ismissing(ig1)
        ig1 = ones(size(rall))
    end
    wall = wall .* ig1

    ELDA = 0.0
    EH = 0.0
    KE = 0.0
    ENUC = 0.0
    
    a = @spawn begin
        if hydrogen == false
            ELDA = calc_energy_lda_LM(rho_LM, drho_LM, N, Rmax, rall, wall, D1Xgrid, nspin=nspin, funlist=funlist, gga=gga)
        end
    end
    b = @spawn begin
        if hydrogen == false
            EH = calc_energy_hartree_LM(rho_LM, N, Rmax, rall, wall, VH)
        end
    end
    c = @spawn begin
        KE = calc_energy_ke_LM(rho_LM, N, Rmax, rall, wall, filling, vals_r, Vin)
    end
    d = @spawn begin
        ENUC = calc_energy_enuc_LM(rho_LM, N, Rmax, rall, wall,Z)
    end
    EEXT = 0.0
    e = @spawn begin
        if !ismissing(vext)
            EEXT = calc_energy_vext_LM(rho_LM, N, Rmax, rall, wall,vext)
        else
            EEXT = 0.0
        end
    end

    wait(b)
    wait(c)
    wait(d)
    wait(a)
    wait(e)

    if verbose
        println("ELDA   $ELDA")
        println("EH     $EH")
        println("KE     $KE")
        println("ENUC   $ENUC")
        if !ismissing(vext)
            println("EEXT   $EEXT")
        end
    end
    
    ENERGY = ELDA + EH + KE + ENUC + EEXT
    println()
    println("ENERGY $ENERGY ")
    println()

    return ENERGY

    
    #=    rho_tot = sum(rho, dims=2)
    
    ELDA = calc_energy_lda(rho, N, Rmax, rall, wall,  nspin=nspin)

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
    =#
    
end


function calc_energy_enuc(rho, N, Rmax, rall, wall, Z)

    return -Z * 4 * pi * sum(rho .* wall .* rall)

end

function calc_energy_enuc_LM(rho_LM, N, Rmax, rall, wall, Z)

    return -Z * 4 * pi * sum(sum(rho_LM[:,:,1,1],dims=2) .* wall .* rall) * sqrt(4*pi)

end

function calc_energy_vext_LM(rho_LM, N, Rmax, rall, wall, vext)

    return   4 * pi * sum(sum(rho_LM[:,:,1,1],dims=2) .* vext .* wall .* rall.^2) * sqrt(4*pi)

end


function calc_energy_ke(rho, N, Rmax, rall, wall, filling, vals_r, Vin)

    KE = sum(filling .* vals_r)
    println("calc_energy_ke KE start ", KE)
    KE += -4.0 * pi * sum( sum(Vin .* rho[2:end-1,:],dims=2) .* rall[2:end-1].^2 .* wall[2:end-1])

    return KE
    
end

function calc_energy_ke_LM(rho_LM, N, Rmax, rall, wall, filling, vals_r, Vin)

#    println("filling ")
#    println(filling)
#    println("vals")
#    println(vals_r)
    
    KE = sum(filling .* vals_r)
#    println("KE start $KE")
    for l = 1:size(rho_LM)[3]
        for m = 1:size(rho_LM)[4]
            KE += -4.0 * pi * sum( sum(Vin[2:end-1,:,l,m] .* rho_LM[2:end-1,:,l,m],dims=2) .* rall[2:end-1].^2 .* wall[2:end-1])
        end
    end
    return KE
    
end

function calc_energy_lda(rho, N, Rmax, rall, wall; nspin=1)

    if nspin == 1
#        println("old ", e_LDA.(rho[end]))
        return 4 * pi * 0.5 * sum( e_LDA.(rho) .* rho .* wall .* rall.^2  )  #hartree

        
    elseif nspin == 2


        rho_tot = rho[:,1] + rho[:,2]
        ζ = (rho[:,1] - rho[:,2])./rho_tot

        return 4 * pi * 0.5 * sum( e_LDA_sp.(rho_tot, ζ) .* rho_tot .* wall .* rall.^2 )  #hartree
    end        
end

function calc_energy_lda_LM(rho_LM, drho_LM, N, Rmax, rall, wall, D1Xgrid; nspin=1, funlist=funlist, gga=gga)

    vlda_LM, elda_LM = VXC_LM(rho_LM *4*pi, drho_LM*4*pi, rall, D1Xgrid, get_elm=true, funlist=funlist,  gga=gga)
    elda = 0.0
    rho_LM_tot = sum(rho_LM, dims=2)
    for l = 1:size(rho_LM)[3]
        for m = 1:size(rho_LM)[4]
            elda += 2*pi*sum(elda_LM[:,l,m].*rho_LM_tot[:,1,l,m].*rall.^2 .* wall)
        end        
    end

    return elda
    
end


function calc_energy_hartree(rho, N, Rmax, rall, wall, VH)

    return 0.5 * 4 * pi * sum(VH .* rho .* wall .* rall.^2 )
    
end

function calc_energy_hartree_LM(rho_LM, N, Rmax, rall, wall, VH)
    eh = 0.0
    rho_LM_tot = sum(rho_LM, dims=2)
    for l = 1:size(rho_LM)[3]
        for m = 1:size(rho_LM)[4]
            eh +=  sum(VH[:,l,m] .* rho_LM_tot[:,1,l,m] .* wall .* rall.^2 )
        end
    end
    return 0.5 * (4 * pi)^2 * eh
    
end

function assemble_rho(vals_r, vects, nel, rall, wall, rho, N, Rmax, D2; nspin=1, lmax=0, ig1=missing)

    rho[:] .= 0.0

    if ismissing(ig1)
        ig1= ones(size(rall))
    end
    
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
                
                vnorm = vnorm / (4.0*pi*sum(vnorm .* wall .* ig1))# .* rall.^2))

                println("vnorm ", vnorm[1:3])

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
        norm = 4.0*pi*sum(rho[:,spin] .* wall .*ig1 ) #.* rall.^2
#        println("check norm spin $spin ", norm)
        rhor2[:,spin] = rho[:,spin] /norm*sum(nel[:,spin])
        #        println("rall 1:3 ", rall[1:3])

        #trick for getting rho at r=0, since we only calculated r^2*rho.   (r^2*rho)'' at r=0 is equal to rho(r=0)*2
        d2r = (D2[:,:] * rho[:,spin])  #/ norm * sum(nel[:,spin])  
        rho[2:end,spin] = (rho[2:end,spin]./rall[2:end].^2  )  / norm * sum(nel[:,spin])
#        println("d2r ", d2r)
        rho[1,spin] = d2r[1]/2.0 / norm * sum(nel[:,spin])

#        println("check ", rho[1:3,spin])

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


##

function assemble_rho_LM(vals_r, vects, nel, rall, wall, rho, N, Rmax, D2, D1; nspin=1, lmax=0, ig1=missing, gga = false)

    rho[:] .= 0.0

    if ismissing(ig1)
        ig1= ones(size(rall))
    end
    
    vnorm = zeros(N+1)

    eps =  rall[2]/5.0

    filling = zeros(size(vals_r))
    rhor2 = zeros(size(rho)[1:2])
    rhoR = zeros(size(rho))
    rhoR2 = zeros(size(rho))
    
    for spin = 1:nspin
        for l = 0:lmax
            for m = -l:l
#                println("nel $spin $l $m ", nel[spin, l+1, m+l+1])
                fillval = 2.0
                d1 = FastSphericalHarmonics.sph_mode(l,m)
                nleft = nel[spin, d1[1], d1[2]]
#                println("$l $m nleft $nleft $nleft")
                if nleft < 1e-20
                    nleft = 1e-20
                end
                inds = sortperm(vals_r[:,spin,d1[1], d1[2]])
                for i in inds
                    vnorm[:] = [0; real.(vects[:,i,spin,d1[1], d1[2]].*conj(vects[:,i,spin,d1[1],d1[2]])); 0]
#                    println("vn 2 a ", vnorm[2])
                    #                    vnorm[2:N] = vnorm[2:N] #./ rall[2:N].^2
                    vnorm = vnorm / (4.0*pi*sum(vnorm .* wall .* ig1))# .* rall.^2))

#                    println("vn 2 b ", vnorm[2])
                    
#                    println("vnormLM ", vnorm[1:3])
                    
                    if nleft <= fillval/nspin + 1e-10
                        t = vnorm[2:N]  * nleft 
                        for ll = 0:(lmax*2)
                            for mm = -ll:ll
                                #                                rho[2:N,spin, ll+1, mm+ll+1] += t  * gaunt(l,ll,l,m,-mm,m)*(-1)^mm
                                d = FastSphericalHarmonics.sph_mode(ll,mm)

                                rho[2:N,spin, d[1], d[2]] += t  * real_gaunt_dict[(ll,mm,l,m)]

                                #                                if sum(abs.(t  * real_gaunt_dict[(ll,mm,l,m)])) > 1e-5
#                                    println("$nleft | $l $m $ll $mm  rho $d , ", sum(abs.(t  * real_gaunt_dict[(ll,mm,l,m)])), " " , real_gaunt_dict[(ll,mm,l,m)])
#                                end
                            end
                        end

                        filling[i,spin,d1[1], d1[2]] = nleft
                        break
                    else
                        t = vnorm[2:N]  * fillval / nspin
                        for ll = 0:(lmax*2)
                            for mm = -ll:ll
                                #rho[2:N,spin, ll+1, mm+ll+1] += t * gaunt(l,ll,l,m,-mm,m)*(-1)^mm
                                d = FastSphericalHarmonics.sph_mode(ll,mm)

                                rho[2:N,spin, d[1], d[2]] += t * real_gaunt_dict[(ll,mm,l,m)]


                                #                                if sum(abs.(t  * real_gaunt_dict[(ll,mm,l,m)])) > 1e-5
#                                    println("$nleft | $l $m $ll $mm  rho $d , ", sum(abs.(t  * real_gaunt_dict[(ll,mm,l,m)])), " " , real_gaunt_dict[(ll,mm,l,m)])
#                                end                                
                            end
                        end
                        
                        #                        rho[2:N,spin] += vnorm[2:N] * fillval / nspin #./ rall[2:N].^2
                        nleft -= fillval / nspin
                        filling[i,spin,d1[1], d1[2]] = fillval / nspin
                        
                    end

                    
                    
                end
            end
        end




        
        norm = 4.0*pi*sum(rho[:,spin,1,1] .* wall .*ig1 ) #/ (4*pi)^0.5  #.* rall.^2
#        println("norm ", norm)
        
        rhor2[:,spin] = rho[:,spin,1,1] #/norm*sum(nel[spin,:,:])


        
        t = zeros(size(rho)[1])
        t[2:end] = rho[2:end,spin,1,1] ./ rall[2:end]
        d1r = D1 * t

        rhoR2 = deepcopy(rho)

#        println("rhoR2 2 ", rhoR2[2])
        
#        d2r = (D2[:,:] * rho[:,spin,1,1])  #/ norm * sum(nel[:,spin])  

        for ll = 0:(lmax*2)
            for mm = -ll:ll
                d = FastSphericalHarmonics.sph_mode(ll,mm)
                rhoR[2:end,spin,d[1],d[2]] = rho[2:end,spin, d[1], d[2]] ./ rall[2:end]
                
                if ll == 0 && mm == 0
                    #                     println("norm $norm")
                    rho[2:end,spin, 1,1] = (rho[2:end,spin, 1, 1]./rall[2:end].^2  ) #/ norm * sum(nel[spin,:,:]) #* sqrt(4*pi)

                    #rho[1,spin,1,1] = d2r[1]/2.0 # / norm * sum(nel[spin,:,:])

                    rho[1,spin,1,1] = d1r[1] # / norm * sum(nel[spin,:,:])

                     #                     println("check ", rho[1:3,spin,1,1])
                    #                     println("check ", sum(rho[:,spin,1,1] .* rall.^2 .* wall .* ig1)*4*pi*sqrt(4*pi) )
                     
                else
                    #d = FastSphericalHarmonics.sph_mode(ll,mm)
                    rho[2:end,spin, d[1], d[2]] = (rho[2:end,spin, d[1], d[2]]./rall[2:end].^2  ) 
                end
            end
        end
        
        
    end
        

    #
#    println("filling")
#    println(filling)
    
    return rho, filling, rhor2, rhoR, rhoR2
    
end



####################

function DFT_spin_l_grid(nel; N = 40, nmax = 1, Z=1.0, Rmax = 10.0, rho_init = missing, niters=20, mix = 1.0, mixing_mode=:pulay, ax = 0.02)

    Z = Float64(Z)
    nel = Float64.(nel)
    
    if length(size(nel)) == 1
        nspin = 1
    else
        nspin = size(nel)[2]
    end
    
    lmax = size(nel)[1] - 1
    println()
    println("Running Z= $Z with nspin= $nspin lmax= $lmax Nel= ", sum(nel))
    println()

#    println("setup")


    
    function grid(x)
        return log(x + ax)
    end

    function grid1(x)
        return 1.0/(x+ax)
    end
    function grid2(x)
        return -1.0 / (x+ax)^2
    end

    
#    function grid1(x)
#        return ForwardDiff.derivative(grid, x)
#    end
#    function grid2(x)
#        return ForwardDiff.derivative(grid1,x)
#    end
    
    function invgrid(x)
        return exp(x) - ax
    end

    function invgrid1(x)
        return grid1(x).^-1
    end

    
    
#=    println("grid check ", grid(invgrid(1.0)), " " , 1.0)
    println("grid check ", grid(invgrid(0.0)), " " , 0.0)
    println("grid check ", invgrid(grid(1.0)), " " , 1.0)
    println("grid check ", invgrid(grid(0.0)), " " , 0.0)

    println("grid check ", grid(invgrid(1.1)), " " , 1.1)
    println("grid check ", invgrid(grid(1.1)), " " , 1.1)

    println("grid1 check ", grid1(1.1), " " , grid1a(1.1))
    println("grid2 check ", grid2(1.1), " " , grid2a(1.1))
=#

    
    begin
        r, D, D2, w, rall,wall,H_poisson, D1X, D2X = setup_mats(;N = N, a=grid(0.0), Rmax=grid(Rmax))


        
#        r, D, D2, w, rall_orig,wall,H_poisson, D2X = setup_mats(;N = N, a=0.0, Rmax=Rmax)
#
#        println(rall[1], " " , rall_orig[1], " ", invgrid(rall[1]))
#        
        rall_rs = invgrid.(rall)

        ig1 = invgrid1.(rall_rs)

        H_poisson = diagm(grid1.(rall_rs[2:N]).^2)*D2 + diagm(grid2.(rall_rs[2:N])) *D +  diagm( 2.0 ./ rall_rs[2:N] )*diagm(grid1.(rall_rs[2:N]))  * D
        D2grid = ( diagm(grid1.(rall_rs[2:N]).^2)*D2 + diagm(grid2.(rall_rs[2:N])) *D  )
        D2Xgrid = ( diagm(grid1.(rall_rs).^2)*D2X + diagm(grid2.(rall_rs)) *D1X  )

        
#        return rall_orig, rall2
        
        rho = get_initial_rho(rho_init, N, Z, nel,rall_rs,wall, nspin=nspin, ig1=ig1)
        #        println("rho_ini ", size(rho))

#        return rho
        
        rho_old = deepcopy(rho)
#        rho_veryold = deepcopy(rho)

        n1in = deepcopy(rho)
        n1out = deepcopy(rho)
        n2in = deepcopy(rho)
        n2out = deepcopy(rho)
        
        rhor2 = zeros(size(rho))

#        KE = getKE(r)

        H0_L = []

        
        for l = 0:lmax
            Vc = getVcoulomb(rall_rs[2:N], Z, l)


            
            H0 = -0.5*( diagm(grid1.(rall_rs[2:N]).^2)*D2 + diagm(grid2.(rall_rs[2:N])) *D  ) + Vc

                   
            
            #H0 = -0.5 * diagm(1.0 ./ r.^2)*D*diagm(r).^2*D + Vc
            push!(H0_L, H0)
        end
        Vc = getVcoulomb(rall_rs[2:N], Z, 0)

        
#        Ham = zeros(N-1, N-1)
        
        vects = zeros(Complex{Float64}, N-1, N-1, nspin, lmax+1)
        vals_r = zeros(Float64, N-1, nspin, lmax+1)    

        VH = zeros(size(rho)[1],1)

        VLDA = zeros(size(rho))
        
        filling = zeros(size(vals_r))
        Vtot = zeros(N-1, nspin)
        
    end

    eigs_tot_old = sum(filling.*vals_r)

    spin_l = []
    for spin = 1:nspin
        for l = 0:lmax
            push!(spin_l,  [spin, l])
        end
    end

    
    for iter = 1:niters

        println()
        #        VLDA = diagm(v_LDA.(rho[2:N]))
#        println(size(rho))
#        println("LDA")

        if nspin == 1
            VLDA[:,1] = v_LDA.(rho[:, 1])
        elseif nspin == 2
            tt =  v_LDA_sp.(rho[:, 1], rho[:, 2])
            VLDA[:,:] = reshape(vcat(tt...), nspin,N+1)'
        end

#        println("poisson")        
        begin
            rho_tot = sum(rho, dims=2)
            VH[:] = V_H3( rho_tot[2:N], rall_rs[2:N],w,H_poisson,Rmax, rall_rs, sum(nel), ig1=ig1[2:N])
            VH_mat = diagm(VH[2:N])
        end

#       return VLDA[:,1], VH, rall_rs
#
        
#        println("eig")




        for i = 1:length(spin_l)
            spin = spin_l[i][1]
            l = spin_l[i][2]
            
            VLDA_mat = diagm(VLDA[2:N,spin])
            if l == 0
                Vtot[:,spin] = (VH[2:N] + VLDA[2:N,spin]) + diag(Vc)
            end
            #            for l = 0:lmax

            Ham = H0_L[l+1] + (VH_mat + VLDA_mat)
            vals, v = eigen(Ham)
            #vals, v = eigs(Ham, which=:SM, nev=4)
            vals_r[:,spin,l+1] = real.(vals)
            vects[:,:,spin,l+1] = v
            
#            end
        end
        
#        for spin = 1:nspin
#            for l = 0:lmax
#                println("iter $iter vals spin =  $spin l = $l ", vals_r[1:min(5, size(vals_r)[1]), spin, l+1])
#            end
#        end
        
#        println("ass")
        rho_new, filling, rhor2 = assemble_rho(vals_r, vects, nel, rall_rs, wall, rho, N, Rmax, D2Xgrid, nspin=nspin, lmax=lmax, ig1 = ig1)

#        println("filling ", filling)

        eigs_tot_new = sum(filling.*vals_r)
        if iter > 1 && abs(eigs_tot_old-eigs_tot_new) < 1e-8
            println()
            println("eigval-based convergence reached ", abs(eigs_tot_old-eigs_tot_new))
            println()
            break
        end
        eigs_tot_old = eigs_tot_new

        
        n1in[:] = n2in[:]
        n1out[:] = n2out[:]

        n2in[:] = rho_old[:]
        n2out[:] = rho_new[:]
        
        
        if iter > 1 && mixing_mode == :pulay 
            for spin = 1:nspin
                rho[:,spin] = pulay_mix(n1in[:,spin], n2in[:,spin], n1out[:,spin], n2out[:,spin], rho_old[:,spin], mix)
            end
        else
            rho[:] = mix*rho[:] + (1.0-mix)*rho_old[:]
        end
        
#        rho_veryold[:] = rho_old[:]
        rho_old[:] = rho[:]
    end

    #println("energy")
    energy = calc_energy(rho, N, Rmax, rall_rs, wall, VH, filling, vals_r, Vtot, Z, nspin=nspin, lmax=lmax, ig1=ig1)
    
    return energy, vals_r, vects, rho, rall_rs, wall, rhor2, VH
    
end    

function pulay_mix(n1in, n2in, n1out, n2out, rho_old, mix)

    if sum(rho_old) < 1e-5 #fixes the case a spin channel is totally empty
        return rho_old
    end
    
    R1 = n1out - n1in
    R2 = n2out - n2in
    
    B = zeros(3,3)
    B[1,3] = 1
    B[2,3] = 1
    B[3,3] = 0.0
    B[3,1] = 1
    B[3,2] = 1
    
    R = zeros(3,1)
    R[3,1] = 1.0
    
                    
    B[1,1] = R1' * R1
    B[2,2] = R2' * R2
    B[1,2] = R1' * R2
    B[2,1] = R2' * R1
    
    c = B \ R
    
    n_pulay = n1out * c[1,1] + n2out * c[2,1]
    t = (1-mix) * rho_old + (mix) * n_pulay
    #t =  n_pulay 
    
    return t

end

#######################
"""
1 s 2.0
2 s 2.0
2 p 2.0
3 p 2.0
3 p 2.0
"""
function setup_filling(fill_str)

    T = []
    nspin = 1
    lmax = 0
    for line in split(fill_str, "\n")
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
                    
                               
#=                for toadd in   [ [1,0,0,2.0],
                                 [2,0, 0,2.0],
                                 [2,1, 1,2.0],
                                 [2,1, 0,2.0],
                                 [2,1,-1,2.0],
                                 [3,0, 0,2.0],
                                 [3,1, 1,2.0],
                                 [3,1, 0,2.0],
                                 [3,1,-1,2.0],
                                 [4,0, 0,2.0],
                                 [3,2, 2,2.0],
                                 [3,2, 1,2.0],
                                 [3,2, 0,2.0],
                                 [3,2,-1,2.0],
                                 [3,2,-2,2.0],
                                 [4,1, 1,2.0],
                                 [4,1, 0,2.0],
                                 [4,1,-1,2.0],
                                 [5,0, 0,2.0],
                                 [4,2, 2,2.0],
                                 [4,2, 1,2.0],
                                 [4,2, 0,2.0],
                                 [4,2,-1,2.0],
                                 [4,2,-2,2.0],
                                 [5,1, 1,2.0],
                                 [5,1, 0,2.0],
                                 [5,1,-1,2.0],
                                 [6,0, 0,2.0],
                                 [4,3, 3,2.0],
                                 [4,3, 2,2.0],
                                 [4,3, 1,2.0],
                                 [4,3, 0,2.0],
                                 [4,3,-1,2.0] ,
                                 [4,3,-2,2.0],
                                 [4,3,-3,2.0],
                                 [5,2, 2,2.0],
                                 [5,2, 1,2.0],
                                 [5,2, 0,2.0],
                                 [5,2,-1,2.0],
                                 [5,2,-2,2.0],
                                 [6,1, 1,2.0],
                                 [6,1, 0,2.0],
                                 [6,1,-1,2.0] ]
                    =#
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


function prepare_dft(Z, N, Rmax, ax, bx, fill_str, spherical, vext, lmax_rho)
                     
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
    
    nel, nspin, lmax = setup_filling(fill_str)

    if ismissing(lmax_rho)
        lmax_rho = min(lmax*2+4, 10)
    end
    
    if spherical
        lmax_rho = 0
    end

    #=
    begin
        function grid(x)
            return log(bx*x + ax)
        end
        
        function grid1(x)
            return bx/(bx*x+ax)
        end
        function grid2(x)
            return -bx^2 / (bx*x+ax)^2
        end

        function invgrid(x)
            return (exp(x) - ax)/bx
        end
        
        function invgrid1(x)
            return grid1(x).^-1
        end
    end
=#
    begin

        println("linear grid")
        function grid(x)
            return x
        end

        function grid1(x)
            return 1.0
        end
        function grid2(x)
            return 0.0
        end

        function invgrid(x)
            return x
        end

        function invgrid1(x)
            return 1.0
        end
    end
    
    
#    println("fakegrid")
    

    
    r, D, D2, w, rall,wall,H_poisson, D1X, D2X = setup_mats(;N = N, a=grid(0.0), Rmax=grid(Rmax))

    
    rall_rs = invgrid.(rall)

    
    ig1 = invgrid1.(rall_rs)

    H_poisson = diagm(grid1.(rall_rs[2:N]).^2)*D2 + diagm(grid2.(rall_rs[2:N])) *D +  diagm( 2.0 ./ rall_rs[2:N] )*diagm(grid1.(rall_rs[2:N]))  * D
    D2grid = ( diagm(grid1.(rall_rs[2:N]).^2)*D2 + diagm(grid2.(rall_rs[2:N])) *D  )
    D2Xgrid = ( diagm(grid1.(rall_rs).^2)*D2X + diagm(grid2.(rall_rs)) *D1X  )

    D1Xgrid = diagm(grid1.(rall_rs))*D1X

    if ismissing(vext)
        VEXT = zeros(N+1)
    elseif typeof(vext) != Vector{Float64} || typeof(vext) != Array{Float64, 1}
        println("vect ", typeof(vext))
        VEXT = vext.(rall_rs)
    else
        VEXT = vext
    end

    H0_L = []
    
    for l = 0:lmax
        Vc = getVcoulomb(rall_rs[2:N], Z, l)

        Vc += diagm(VEXT[2:N])
        
        H0 = -0.5*( diagm(grid1.(rall_rs[2:N]).^2)*D2 + diagm(grid2.(rall_rs[2:N])) *D  ) + Vc

        #            println("H0 $l ")
        #            println(H0)
        #            println()

        
        #H0 = -0.5 * diagm(1.0 ./ r.^2)*D*diagm(r).^2*D + Vc
        push!(H0_L, H0)
        
    end
    Vc = getVcoulomb(rall_rs[2:N], Z, 0)

    spin_lm = []
    for spin = 1:nspin
        for l = 0:lmax
            for m = -l:l
                push!(spin_lm,  [spin, l,m])
            end
        end
    end

    spin_lm_rho = []
    for spin = 1:nspin
        for l = 0:(lmax*2)
            for m = -l:l
                push!(spin_lm_rho,  [spin, l,m])
            end
        end
    end

    
    return Z, nel, nspin, lmax, lmax_rho, grid, grid1, grid2, invgrid, invgrid1, r, D, D2, w, rall,wall,H_poisson, D1X, D2X, rall_rs, ig1, H_poisson, D2grid, D2Xgrid, D1Xgrid, VEXT, H0_L, Vc, spin_lm, spin_lm_rho
    
end
    
function choose_exc(exc)

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

        
        funlist,gga = set_functional(exc, nspin=nspin)
    else
        println("Running default LDA (VWN)")
        funlist = missing
        gga = false
    end

    return funlist, gga

end


function DFT_spin_l_grid_LM(; fill_str=missing, N = 40, Z=1.0, Rmax = 10.0, rho_init = missing, niters=50, mix = 0.5, mixing_mode=:pulay, ax = 0.2, exc=missing, bx = 0.5, vext = missing, spherical = false, hydrogen=false, lmax_rho = missing)


    Z, nel, nspin, lmax, lmax_rho, grid, grid1, grid2, invgrid, invgrid1, r, D, D2, w, rall,wall,H_poisson, D1X, D2X, rall_rs, ig1, H_poisson, D2grid, D2Xgrid, D1Xgrid, VEXT, H0_L, Vc, spin_lm, spin_lm_rho = prepare_dft(Z, N, Rmax, ax, bx, fill_str, spherical, vext, lmax_rho)

    funlist, gga = choose_exc(exc)
    
        
    
    println()
    println("Running Z= $Z with nspin= $nspin lmax= $lmax Nel= ", sum(nel))
    println()

    
    begin

        
        if nspin == 1
            nel_t = [sum(nel)]
        else
            nel_t = [sum(nel[1,:,:,:]) sum(nel[2,:,:,:])]
        end
            
        rho = get_initial_rho(rho_init, N, Z, nel_t,rall_rs,wall, nspin=nspin, ig1=ig1)

        #                R  SPIN       L            m
        rho_LM = zeros(N+1, nspin, lmax_rho+1, (2*lmax_rho+1))

        for spin in 1:nspin
            rho_LM[:,spin,1,1] = rho[:,spin]/(4*pi)^0.5
        end
        
        
        rho_old_LM = deepcopy(rho_LM)
        rho_new_LM = deepcopy(rho_LM)


        n1in = deepcopy(rho_LM)
        n1out = deepcopy(rho_LM)
        n2in = deepcopy(rho_LM)
        n2out = deepcopy(rho_LM)
        
        rhor2 = zeros(size(rho))

        drho_LM = zeros(size(rho_LM))

        vects = zeros(Complex{Float64}, N-1, N-1, nspin, lmax+1, 2*lmax+1)
        vals_r = zeros(Float64, N-1, nspin, lmax+1,2*lmax+1)    

        VH_LM = zeros(size(rho_LM)[[1,3,4]])


        vlda_LM = zeros(size(rho_LM))
        VLDA = zeros(size(rho)[1])
        
        filling = zeros(size(vals_r))
        Vtot = zeros(size(rho_LM))

        va = zeros(1)

        rhoR = zeros(size(rho_LM))
        rhoR2 = zeros(size(rho_LM))
        
    end

    eigs_tot_old = sum(filling.*vals_r)

    
    
    converged = false
    println()
    for iter = 1:niters


#        println()
        #        VLDA = diagm(v_LDA.(rho[2:N]))
        
        
        lda_task = @spawn  begin
            vlda_LM, elda_LM = VXC_LM(rho_LM *4*pi, 4*pi*drho_LM, rall_rs, funlist=funlist, D1Xgrid, gga=gga)
        end

        poisson_task = @spawn         begin
            #rho_tot = sum(rho, dims=2)
            for (spin,l,m) in spin_lm_rho
                if spin == 2
                    continue
                end
                d = FastSphericalHarmonics.sph_mode(l,m)
                rho_tot = sum(rho_LM[:,:,d[1],d[2]], dims=2)
#                println("$spin $l $m  rho_tot ", sum(abs.(rho_tot)))
                VH_LM[:,d[1],d[2]] = V_H3( rho_tot[2:N], rall_rs[2:N],w,H_poisson,Rmax, rall_rs, sum(nel)/sqrt(4*pi), ig1=ig1[2:N], l=l)
            end
        end

        wait(lda_task)
        wait(poisson_task)

#        println("VH_LM " )
#        println(VH_LM[:,1,1])
#        println()

#       return VLDA[:,1], VH, rall_rs
#
        
#        println("eig")


#        println("size VH_LM ", size(VH_LM))
#        println("sum VH")
#        println(sum(abs.(VH_LM), dims=[1]))
#        println()

#        println("size vlda_LM ", size(vlda_LM))
#        println("sum vlda_LM")
#        println(sum(abs.(vlda_LM), dims=[1,2]))
#        println()
        
        Vtot .= 0.0


        if hydrogen == false
            Vtot[:,1,:,:] .= (4*pi*VH_LM[:,:,:]  + vlda_LM[:,1,:,:])
        end
        
        Vtot[2:N,1, 1,1] += diag(Vc) * sqrt(4*pi)
        if !ismissing(vext)
            Vtot[2:N,1, 1,1] += VEXT[2:N] * sqrt(4*pi)
        end
        #Vtot[2:N,1, 1,1] += diag(Vc) * sqrt(4*pi)

        if nspin == 2             
            if hydrogen == false
                Vtot[:,2,:,:] .= (4*pi*VH_LM[:, :,:]  + vlda_LM[:,2,:,:])
            end
            
            Vtot[2:N,2, 1,1] += diag(Vc) * sqrt(4*pi)
        end
        if !ismissing(vext) && nspin == 2
            Vtot[2:N,2, 1,1] += VEXT[2:N] * sqrt(4*pi)
        end

        

        
        #        @threads 
        @threads for i = 1:length(spin_lm)
            spin = spin_lm[i][1]
            l = spin_lm[i][2]
            m = spin_lm[i][3]

            
            #            VLDA_mat = diagm(VLDA[2:N,spin])

            #VLDA_mat = diagm(vlda_LM[2:N,spin,1,1])

            #println("vlda before ", vlda_LM[2:5,1,1,1])

            d = FastSphericalHarmonics.sph_mode(l,m)
            
            VLDAt = zeros(N+1)
            VHt = zeros(N+1)
            for ll = 0:(lmax_rho)
                for mm = -ll:ll
                    #                    VH += 4*pi*VH_LM[:, ll+1, mm+ll+1] * real_gaunt_dict[(ll,mm,l,m)]
                    d2 = FastSphericalHarmonics.sph_mode(ll,mm)

                    VHt += 4*pi*VH_LM[:, d2[1], d2[2]] * real_gaunt_dict[(ll,mm,l,m)] #/ (2*ll+1)
                    VLDAt += vlda_LM[:,spin,d2[1], d2[2]] * real_gaunt_dict[(ll,mm,l,m)] #/ (2*ll+1)

#                    VHt += 4*pi*VH_LM[:, d2[1], d2[2]] * real_gaunt_dict[(l,m,ll,mm)] #/ (2*ll+1)
#                    VLDAt += vlda_LM[:,spin,d2[1], d2[2]] * real_gaunt_dict[(l,m,ll,mm)] #/ (2*ll+1)

                    
#                    println("vlda zzz $ll $mm $l $m ", sum(abs.(vlda_LM[:,spin, d[1], d[2]])), " g ", real_gaunt_dict[(ll,mm,l,m)])
                    
                end
            end
            #println("vlda after ", VLDA[2:5])
            

            
            VH_mat = diagm(VHt[2:N])
            VLDA_mat = diagm(VLDAt[2:N])


            #            Vtot[:,spin, d[1], d[2]] = (VHt + VLDAt) * sqrt(4*pi)


            
            #            if l == 0
            
            #            end
            
            #            for l = 0:lmax

            if hydrogen
                Ham = H0_L[l+1] #+ (VH_mat + VLDA_mat)
            else
                Ham = H0_L[l+1] + (VH_mat + VLDA_mat)
            end                


            #            println("Ham $l spin $spin")
#            println(Ham)
            vals, v = eigen(Ham)
            #vals, v = eigs(Ham, which=:SM, nev=4)
#            println("spin $spin l $l m $m")
            
            vals_r[:,spin,d[1],d[2]] = real.(vals)
            vects[:,:,spin,d[1],d[2]] = v
            
#            end
        end
        
#        println("rho_LM ", rho_LM[1:3,1,1,1])
        #println("ass")

        rho_new_LM, filling, rhor2, rhoR, rhoR2 = assemble_rho_LM(vals_r, vects, nel, rall_rs, wall, rho_LM, N, Rmax, D2Xgrid, D1Xgrid, nspin=nspin, lmax=lmax, ig1 = ig1, gga=gga)

        
        #return filling
        
#        if true
            #                calc_drho(rho_new_LM, drho_LM, D1Xgrid, lmax)
            #calc_drho(rho_new_LM, drho_LM, D1Xgrid, lmax)
#            calc_drho3(rho_LM, drho_LM, D1Xgrid, lmax, rhoR, rall_rs)
            
#        end

        
#        println("size rho_new_LM ", size(rho_new_LM))
#        println("sum rho")
#        println(sum(abs.(rho_LM), dims=[1,2]))
#        println()
        
#        println("rho_new_LM ", rho_LM[1:3,1,1,1])

#        println("filling ", filling)

#            for spin = 1:nspin
#                for l = 0:lmax
#                    for m = -l:l
#                        d = FastSphericalHarmonics.sph_mode(l,m)
#                        println("iter $iter vals spin =  $spin l = $l m = $m   ", vals_r[1:min(5, size(vals_r)[1]), spin, d[1],d[2]])
#                    end
#                end
#            end
        
#        energy = calc_energy_LM(rho_LM, drho_LM, N, Rmax, rall_rs, wall, VH_LM, filling, vals_r, Vtot, Z, D1Xgrid, nspin=nspin, lmax=lmax, ig1=ig1, funlist=funlist, gga=gga, verbose=false)
        
        eigs_tot_new = sum(filling.*vals_r)
        println("conv check ", abs(eigs_tot_old-eigs_tot_new))
        if iter > 1 && abs(eigs_tot_old-eigs_tot_new) < 1e-6
            println()
            println("eigval-based convergence reached ", abs(eigs_tot_old-eigs_tot_new))
            converged = true
            for spin = 1:nspin
                for l = 0:lmax
                    for m = -l:l
                        d = FastSphericalHarmonics.sph_mode(l,m)
                        println("iter $iter vals spin =  $spin l = $l m = $m   ", vals_r[1:min(6, size(vals_r)[1]), spin, d[1],d[2]])
                    end
                end
            end

            println()
            if true
                calc_drho3(rho_LM, drho_LM, D1Xgrid, lmax, rhoR, rall_rs)
            end
            break
        end
        eigs_tot_old = eigs_tot_new

        
        n1in[:] = n2in[:]
        n1out[:] = n2out[:]

        n2in[:] = rho_old_LM[:]
        n2out[:] = rho_new_LM[:]
        
        
        if iter > 6 && mixing_mode == :pulay
        #if 
            for spin = 1:nspin

                temp = pulay_mix(n1in[:,spin, :, :][:], n2in[:,spin, :, :][:], n1out[:,spin, :, :][:], n2out[:,spin, :, :][:], rho_old_LM[:,spin, :, :][:], mix) 
                rho_LM[:,spin, :, :] = reshape(temp, size(rho_LM)[1], 1, size(rho_LM)[3], size(rho_LM)[4])
                #                for l = 1:size(rho_LM)[3]
#                    for m = 1:size(rho_LM)[4]
#                        rho_LM[:,spin, l, m] = pulay_mix(n1in[:,spin, l, m], n2in[:,spin, l, m], n1out[:,spin, l, m], n2out[:,spin, l, m], rho_old_LM[:,spin, l, m], mix)
#                    end
#                end
            end
        elseif iter > 50
            mm = min(0.25, mix)
            rho_LM[:] = mm*rho_new_LM[:] + (1.0-mm)*rho_old_LM[:]
        else
            mm = min(0.7, mix)
            rho_LM[:] = mm*rho_new_LM[:] + (1.0-mm)*rho_old_LM[:]
        end
        
#        rho_veryold[:] = rho_old[:]
        rho_old_LM[:] = rho_LM[:]

        if gga
            calc_drho3(rho_LM, drho_LM, D1Xgrid, lmax, rhoR, rall_rs)
        end

        
    end

    println("energy")
    energy = 0.0
    energy = calc_energy_LM(rho_LM, drho_LM, N, Rmax, rall_rs, wall, VH_LM, filling, vals_r, Vtot, Z, D1Xgrid, nspin=nspin, lmax=lmax, ig1=ig1, funlist=funlist, gga=gga, vext=VEXT, hydrogen=hydrogen)

    #    return energy,converged, vals_r, vects, rho_LM, rall_rs, wall.*ig1, rhor2, vlda_LM, drho_LM, D1Xgrid, va, rhoR, rhoR2

    return energy,converged, vals_r, vects, rho_LM, rall_rs, wall.*ig1, rhoR2, VH_LM
    
end    

function calc_drho(rho, drho, D1, lmax)
    println("calc_drho1")
    nspin = size(rho)[2]
    for spin = 1:nspin
        for ll = 0:(2*lmax)
            for mm = -ll:ll
                d = FastSphericalHarmonics.sph_mode(ll,mm)
                drho[:,spin,d[1], d[2]] = D1*rho[:,spin,d[1], d[2]]
            end
        end
    end
end

function calc_drho2(rho, drho, D1, lmax, rhoR, r)
    println("calc_drho2")
    nspin = size(rho)[2]
    for spin = 1:nspin
        for ll = 0:(2*lmax)
            for mm = -ll:ll
                d = FastSphericalHarmonics.sph_mode(ll,mm)

                t = D1 * rhoR[:,spin,d[1], d[2]]
                
                drho[2:end,spin,d[1], d[2]] = (t[2:end] - rho[2:end,spin,d[1], d[2]]) ./ r[2:end]
                drho[1,spin,d[1], d[2]] = D1[1,:]'*rho[:,spin,d[1], d[2]]
            end
        end
    end
end

function calc_drho3(rho, drho, D1, lmax, rhoR, r)
#    println("calc_drho3")


    
    thr1 = 10^-5
    thr2 = 10^-7
    nspin = size(rho)[2]
    rho_t = sum(rho, dims=2)[:,1,:,:]
    for spin = 1:nspin
        for ll = 0:(2*lmax)
            for mm = -ll:ll
                d = FastSphericalHarmonics.sph_mode(ll,mm)

                t = D1 * (rho[:,spin,d[1], d[2]] .* r)

                dr_spect = zeros(size(t))
                dr_spect[2:end] = (t[2:end] - rho[2:end,spin,d[1], d[2]]) ./ r[2:end]
                dr_spect[1] = D1[1,:]'*rho[:,spin,d[1], d[2]]

                dr_fd = zeros(size(t))

                a = rho[:,spin,d[1], d[2]]
                
                for i = 2:length(dr_fd)-1
                    dr_fd[i] = 0.5*(a[i]-a[i-1]) / (r[i] - r[i-1]) + 0.5*(a[i+1]-a[i]) / (r[i+1] - r[i])
                end
                #                println("dr ", dr_fd[10:12])

                cutfn = smooth( rho_t[:,d[1], d[2]] , thr1, thr2)
                
                drho[:,spin,d[1], d[2]] = dr_spect .* ( cutfn ) + dr_fd .* ( 1.0 .- cutfn )

            end
        end
    end
end

function VXC_LM(rho_LM, drho_LM, r, D1; get_elm=false, funlist=missing, gga=false)


    nspin = size(rho_LM)[2]
#    println("nspin $nspin")
    nr = size(rho_LM)[1]
#    rho = zeros(size(rho_LM))
    vlda = zeros(size(rho_LM))
    VLDA = zeros(nr,nspin)

    vlda_lm = zeros(size(rho_LM))

    if get_elm
        elda = zeros(nr,size(rho_LM)[3],size(rho_LM)[4])
        elda_lm = zeros(nr,size(rho_LM)[3],size(rho_LM)[4])
    else
        elda = zeros(1)
        elda_lm = zeros(1)
    end


    
    #convert to real (theta phi) space
#    println("convert")


#    rho_test = zeros(size(rho_LM))
#    for r in 1:nr
#        for spin = 1:nspin
#            rho_test[r,spin,:,:] = FastSphericalHarmonics.sph_evaluate(rho_LM[r,spin,:,:])
#        end
#    end


    
    #rho2 = deepcopy(rho)

    #println("get rho")

    rho, drho, ddrho, vsigma, dvsigma,THETA,PHI = get_rho_rs(rho_LM, drho_LM, r, funlist, gga=gga) 

#    println("1rho      ", rho[50:55, 1,1,1])
#    println("1rho_test ", rho_test[50:55, 1,1,1])
#    println()
#    println("2rho      ", rho[50:55, 1,2,1])
#    println("2rho_test ", rho_test[50:55, 1,2,1])
#    println()
    #    println(rho2[1:5,1,1,1])
#    println(rho[1:5,1,1,1])
    
    #test
    #test = FastSphericalHarmonics.sph_evaluate(ones(size(rho_LM[r,spin,:,:])))

    rho = max.(rho, 1e-30)
    va = zeros(1)
    
#    println("xxxxxxxxxxxxxxxxxxxxxxxxx")
    #evaluate vxc in (theta phi) space
    #println("eval")
    for l = 1:size(rho_LM)[3]
        for m = 1:size(rho_LM)[4]
        
            if nspin == 1


                if !ismissing(funlist)
                    e,v = EXC( (@view rho[:,1,l,m]), funlist, (@view drho[:,1,l,m,:]), (@view ddrho[:,1,l,m,:]), (@view dvsigma[:,1,l,m,:]), THETA[l], gga, r, D1)
                    vlda[:,1,l, m] .= v
                    if get_elm
                        elda[:,l,m] .= e
                    end
                else
                    vlda[:,1,l, m] = v_LDA.(@view rho[:,1,l,m])
                    if get_elm
                        elda[:,l,m] = e_LDA.(@view rho[:,1,l,m])
                    end
                end
                
                
            elseif nspin == 2

                if !ismissing(funlist)
#                    e,v = EXC_sp( rho[:,1:2,l,m], funlist, drho[:,1:2,l,m], gga)
                    if gga
                        e,v = EXC_sp( (@view rho[:,1:2,l,m]), funlist, (@view drho[:,1:2,l,m,:]), (@view ddrho[:,1:2,l,m,:]), (@view dvsigma[:,1:3,l,m,:]), THETA[l], gga, r, D1)
                    else
                        e,v = EXC_sp( (@view rho[:,1:2,l,m]), funlist, missing, missing, missing, THETA[l], gga, r, D1)
                    end                        
                    vlda[:,1,l, m] .= v[:,1]
                    vlda[:,2,l, m] .= v[:,2]
                    if get_elm
                        elda[:,l,m] .= e
                    end

                else
                    tt =  v_LDA_sp.( (@view rho[:,1,l,m]), (@view rho[:, 2,l,m]))
                    vlda[:,1:2,l, m] = reshape(vcat(tt...), nspin,nr)'
                    if get_elm
                        r = rho[:,1,l,m] + rho[:,2,l,m]
                        ζ = (rho[:,1,l,m] - rho[:,2,l,m]) ./ r
                        elda[:,l,m] = e_LDA_sp.(r, ζ)
                    end
                end
                
                #                VLDA[:,:] = reshape(vcat(tt...), nspin,nr)'
#                vlda[:,1,l, m] = VLDA[:,1]
#                vlda[:,2,l, m] = VLDA[:,2]                
            end
        end
    end
#    println("vlda tp ", vlda[1:3,1,1,1])
    #convert back to LM space
    #println("transform")
    
    for r in 1:nr
        for spin = 1:nspin
            vlda_lm[r,spin,:,:] = FastSphericalHarmonics.sph_transform(( vlda[r,spin,:,:]))
            if get_elm && spin == 1
                elda_lm[r,:,:] = FastSphericalHarmonics.sph_transform(( elda[r,:,:]))
            end
                
        end
    end

    
#    vlda_lm2 = zeros(size(vlda))
    #vlda_lm2 = copy(vlda_lm)
#
 #   vlda_lm2 = get_v_lm(vlda_lm2, vlda)    

#    println("check 1 1 ", vlda_lm[1:4,1,1, 1])
#    println("check 1 2 ", vlda_lm[1:4,1,1, 2])
#    println("check 2 1 ", vlda_lm[1:4,1,2, 1])
#    println("check 2 2 ", vlda_lm[1:4,1,2, 2])
#    println("check v2 ", vlda_lm2[1:4,1,1, 1])

    
    return vlda_lm, elda_lm
    
    

end



function get_rho_rs(rho_LM, drho_LM, r, funlist; gga=false)

    nspin = size(rho_LM)[2]
    nr = size(rho_LM)[1]

    
    lmax = size(rho_LM)[3]-1
    THETA,PHI = FastSphericalHarmonics.sph_points(lmax+1)


    rho_thr = zeros(nr, nspin, lmax+1, size(rho_LM)[4], nthreads())
    drho_thr = zeros(nr, nspin, lmax+1, size(rho_LM)[4], 3, nthreads())
    ddrho_thr = zeros(nr, nspin, lmax+1, size(rho_LM)[4], 3, nthreads())

    
    phi = 0.0
    theta = 0.0

    #this stuff is to compute derivatives of spherical harmonics. This could be cached.
    #println("begin")
    begin
#=        function ftheta(x)
            SphericalHarmonics.computeYlm(x, phi, lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())
        end

        function fphi(x)
            SphericalHarmonics.computeYlm(theta, x, lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())
        end

        dftheta = x-> ForwardDiff.derivative(ftheta, x)
        dfphi = x-> ForwardDiff.derivative(fphi, x)

        ddftheta = x-> ForwardDiff.derivative(dftheta, x)
        ddfphi = x-> ForwardDiff.derivative(dfphi, x)
  =#      
    end
    
#    ml = SphericalHarmonicModes.ML(0:lmax, -lmax:lmax)
    
    
#    println("size ", size(rho_LM), " " , size(drho_LM))
    
    #    @threads for (t,theta) in enumerate(THETA)

#    println("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    srho = sum(abs.(rho_LM), dims=[1,2])[1,1,:,:]
    
    
    for l = 0:lmax
        for m = -l:l
            d = FastSphericalHarmonics.sph_mode(l,m)
            if srho[d[1], d[2]]  < 1e-7
                continue
            end
            
            for t in 1:length(THETA)
                theta = THETA[t]
                id = threadid()
                
                for (p,phi) in enumerate(PHI) 
                    #                        rho_thr[:,:,t, p, id] += ( rho_LM[:,:, d[1], d[2]]) * Y[(l,m)]
                    rho_thr[:,:,t, p, id] .+= (@view rho_LM[:,:, d[1], d[2]]) * Y_dict[(lmax, t,p,l,m)]
                    if gga
                        drho_thr[:,:,t, p,1, id] .+= ( @view drho_LM[:,:, d[1], d[2]]) * Y_dict[(lmax, t,p,l,m)]
                        drho_thr[:,:,t, p,2, id] .+= ( @view rho_LM[:,:, d[1], d[2]]) *  Ytheta_dict[(lmax, t,p,l,m)]
                        drho_thr[:,:,t, p,3, id] .+= ( @view rho_LM[:,:, d[1], d[2]]) *  Yphi_dict[(lmax, t,p,l,m)]
                        
                        ddrho_thr[:,:,t, p,2, id] .+= ( @view  rho_LM[:,:, d[1], d[2]]) *  Ytheta2_dict[(lmax,t,p,l,m)]
                        ddrho_thr[:,:,t, p,3, id] .+= ( @view  rho_LM[:,:, d[1], d[2]]) *  Yphi2_dict[(lmax,t,p,l,m)]
                    end
                end
                
            end
            
        end
        
    end
    
    rho = sum(rho_thr, dims=5)[:,:,:,:,1]
    drho = sum(drho_thr, dims=6)[:,:,:,:,:,1]
    ddrho = sum(ddrho_thr, dims=6)[:,:,:,:,:,1]


    #figure out vsigma and derivatives wrt angle.
    #current method is to evaluate vsigma in real (theta phi) space, transform to LM space, take the derivative, and then transform backwards.

    #println("if gga")
    if gga

        #real space
        if nspin == 2
            vsigma = zeros(nr, 3, lmax+1, size(rho_LM)[4])
        else
            vsigma = zeros(nr, 1, lmax+1, size(rho_LM)[4])
        end            

        #println("getVsigma")
        for t in 1:length(THETA)
            theta = THETA[t]
            for (p,phi) in enumerate(PHI) 
                vsigma[:,:, t,p] = getVsigma( (@view rho[:,:,t,p]), funlist, (@view drho[:,:,t,p,:]), r, theta, nspin)
            end
        end

        #LM space
        vsigma_lm = zeros(size(vsigma))
#        rho_lm_test = zeros(size(vsigma))

        #println("LM space")
        begin
            if nspin == 1
                for r in 1:nr
                    vsigma_lm[r,1,:,:] .= FastSphericalHarmonics.sph_transform( vsigma[r,1,:,:])
                end
            elseif nspin == 2
                for r in 1:nr
                    vsigma_lm[r,1,:,:] .= FastSphericalHarmonics.sph_transform( vsigma[r,1,:,:])
                    vsigma_lm[r,2,:,:] .= FastSphericalHarmonics.sph_transform( vsigma[r,2,:,:])
                    vsigma_lm[r,3,:,:] .= FastSphericalHarmonics.sph_transform( vsigma[r,3,:,:])
                end
            end
        end
        
            
#        println("vsigma_lm 50")
#        println(vsigma_lm[30,1,:,:])
#        println("rho_lm_test 50")
#        println(rho_lm_test[30,1,:,:])
        

        # real space again

        if nspin == 1
            dvsigma_thr = zeros(nr, 1, lmax+1, size(rho_LM)[4], 3, nthreads())
        elseif nspin == 2
            dvsigma_thr = zeros(nr, 3, lmax+1, size(rho_LM)[4], 3, nthreads())
        end
        
        #        vsigma_test = zeros(nr, nspin, lmax+1, size(rho_LM)[4])
#        vsigma_test2 = zeros(nr, nspin, lmax+1, size(rho_LM)[4])

#        rho_test = zeros(size(rho_lm_test))

 #       for r in 1:nr
 #           for spin = 1:nspin
 #               vsigma_test2[r,spin,:,:] = FastSphericalHarmonics.sph_evaluate(vsigma_lm[r,spin,:,:])
 #           end
 #       end
        
        lmax = lmax 
        ml = SphericalHarmonicModes.ML(0:lmax, -lmax:lmax)
        #println("realspace again")

        sv = sum(abs.(vsigma_lm), dims=[1,2])[1,1,:,:]
        
    
        for l = 0:lmax
            for m = -l:l
                d = FastSphericalHarmonics.sph_mode(l,m)
                if sv[d[1], d[2]]  < 1e-8
                    continue
                end

                for t in 1:length(THETA)
                    id = threadid()
                    for (p,phi) in enumerate(PHI) 
                        #                for l = 0:(lmax)
                        #                    for m = -l:l
                        #                        d = FastSphericalHarmonics.sph_mode(l,m)
                        if nspin == 1
                            dvsigma_thr[:,1,t, p,2,id] .+=   0.5*( vsigma_lm[:,1, d[1], d[2]]) * Ytheta_dict[(lmax, t,p,l,m)] #dtheta[modeindex(ml, (l,m))]
                            dvsigma_thr[:,1,t, p,3,id] .+=   0.5*( vsigma_lm[:,1, d[1], d[2]]) * Yphi_dict[(lmax, t,p,l,m)] #dphi[modeindex(ml, (l,m))]
                            elseif nspin == 2
                            for ii = 1:3
                                dvsigma_thr[:,ii,t, p,2,id] .+=   0.5*( vsigma_lm[:,ii, d[1], d[2]]) * Ytheta_dict[(lmax, t,p,l,m)]# dtheta[modeindex(ml, (l,m))]
                                dvsigma_thr[:,ii,t, p,3,id] .+=   0.5*( vsigma_lm[:,ii, d[1], d[2]]) * Yphi_dict[(lmax, t,p,l,m)]# dphi[modeindex(ml, (l,m))]
                            end
                        end
                    end
                    #                   end
                end
            end
        end
        dvsigma = sum(dvsigma_thr, dims=6)[:,:,:,:,:,1]
        
#        println("1vsigma       ", vsigma[50:55, 1,1,1])
#        println("1vsigma_test  ", vsigma_test[50:55,1,1,1])
#        println("1vsigma_test2 ", vsigma_test2[50:55,1,1,1])
#        println()
#        d = FastSphericalHarmonics.sph_mode(1,0)
#        println("2vsigma       ", vsigma[50:55, 1,d[1], d[2]])
#        println("2vsigma_test  ", vsigma_test[50:55,1,d[1], d[2]])
#        println("2vsigma_test2 ", vsigma_test2[50:55,1,d[1], d[2]])
 #       println()

#        println("rtest      ", rho[50:55, 1,d[1], d[2]])
#        println("rtest_test ", rho_test[50:55, 1,d[1], d[2]])

        
        #        println("max abs diff vsigma ", maximum(abs.(vsigma - vsigma_test)))
        
    else

        if nspin == 2
            vsigma = zeros(nr, 3, lmax+1, size(rho_LM)[4])
            dvsigma = zeros(nr, 3, lmax+1, size(rho_LM)[4])
            
        else
            vsigma = zeros(nr, 1, lmax+1, size(rho_LM)[4])
            dvsigma = zeros(nr, 1, lmax+1, size(rho_LM)[4])            
        end

    end
    
    return rho, drho, ddrho, vsigma, dvsigma, THETA,PHI

end

#=
This doesn't work i'm pretty sure

function get_v_lm(vlda_lm, vlda)

    nspin = size(vlda_lm)[2]
    
    lmax = size(vlda_lm)[3]-1
    THETA,PHI = FastSphericalHarmonics.sph_points(lmax+1)

    println(size(vlda_lm))
    vlda_lm[:] .= 0.0
    for (t,theta) in enumerate(THETA)
        for (p,phi) in enumerate(PHI) 
            Y = SphericalHarmonics.computeYlm(theta, phi, lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())
#           Y = SphericalHarmonics.computeYlm(theta, phi, lmax=lmax)
            for spin = 1:nspin
                for l = 0:lmax
                    for m = -l:l
                        d = FastSphericalHarmonics.sph_mode(l,m)
                        vlda_lm[:,spin,d[1],d[2]] += 2*pi*sin(theta) * vlda[:,spin, t, p] * Y[(l,m)] / length(THETA)/length(PHI)
                    end
                end
            end
        end
    end
            
    return vlda_lm

end
=#

#=
function calc_energy_lda_LM_realspace(rho_LM, N, Rmax, rall, wall; nspin=1)

    if nspin == 1
#        println("old ", e_LDA.(rho[end]))
#        return 4 * pi * 0.5 * sum( e_LDA.(rho) .* rho .* wall .* rall.^2  )  #hartree

        nspin = size(rho_LM)[2]
        nr = size(rho_LM)[1]

    
        lmax = size(rho_LM)[3]-1
        THETA,PHI = FastSphericalHarmonics.sph_points(40)
        
        nthreads()
        e_thr = zeros( nthreads())
        
        #    @threads for (t,theta) in enumerate(THETA)
        for r = 1:nr
            id = threadid()
            for t in 1:length(THETA)
                theta = THETA[t]
                domega = 2*pi*sin(theta)
                for (p,phi) in enumerate(PHI) 
                    Y = SphericalHarmonics.computeYlm(theta, phi, lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())
                
 #                   for spin = 1:nspin
                    rho = 0.0
                    for l = 0:lmax
                        for m = -l:l
                            d = FastSphericalHarmonics.sph_mode(l,m)
                            rho +=  4*pi*rho_LM[r,1, d[1], d[2]] * Y[(l,m)]
                        end
                    end
                    e_thr[id] +=  e_LDA(rho) * rho * rall[r]^2 * wall[r] * domega * pi/2

#                    end
                end
            end
        end
            
        return sum(e_thr) / length(THETA) / length(PHI)

        
        
    elseif nspin == 2


#        rho_tot = rho[:,1] + rho[:,2]
#        ζ = (rho[:,1] - rho[:,2])./rho_tot

#        return 4 * pi * 0.5 * sum( e_LDA_sp.(rho_tot, ζ) .* rho_tot .* wall .* rall.^2 )  #hartree
    end        
end
=#

function srl(Z, N, Rmax)

    r, D, D2, w, rall,wall,H_poisson, D1X, D2X = setup_mats(;N = N, a=0.0, Rmax=Rmax)

    V = diagm(-Z ./ ( rall))
#
    alpha = 1/ 137.0359895
    #alpha = 0.0

    c= (1/alpha)
#    c= 0.0
    #    c = 10000.0
    H2 = -0.5*D2 .+ V[2:end-1, 2:end-1]
    vals = eigvals(-0.5*D2 .+ V[2:end-1, 2:end-1])
    println("0 vals ", vals[1:5])
    E = deepcopy(H2)
    for i = 1:2
        #        E = diagm(vals)
        #E = vals[1] * I(N-1)
        #E = diagm(min.(vals, vals[1]))
        
        #E = I*-1/9/2
        #        E = I(N-1)*-0.5
#        E = -0.5
#        E = min.(E, -0.5)
        A = alpha^2 / 2 * (E .- V[2:end-1, 2:end-1])
        M = I + A
#        return M
        dVdr = diagm(+Z ./ rall[2:end-1].^2)
        dMdr = alpha^2/2 *(-1) * dVdr
        println("size M ", size(M), " A " , size(A), " dMdr ", size(dMdr) , " D ", size(D), " 1./r ", size(-1 ./ rall[2:end-1]))
#        println(size.([-0.5*D2 , M*V ])) # , - A, E, + 0.5* ( inv(M) .* dMdr), (D + diagm(-1 ./ rall[2:end-1] ))]))
        #H2 = -0.5*D2 + M*V[2:end-1, 2:end-1] - A*E + 0.5* ( inv(M) .* dMdr)*(D + diagm(-1 ./ rall[2:end-1] ))
        VV =  diagm(-Z ./ ( rall))[2:end-1, 2:end-1]
        H2 = 0.5* ( -D2 + 2*VV - alpha^2*VV*(VV-E) + alpha^2*E*(VV - E) - alpha^2 *   inv(2*I - alpha^2*(VV-E))* dVdr  * (D - diagm(1 ./ rall[2:end-1])))
        
        #        H2 = -0.5*D2 - D * (c^2* inv(2*c^2*I - V)) * D + V


        #        t = c^2* inv(2*c^2*I - V)
#        t = diagm(c^2 * (2*c^2 .- diag(V)).^-1)
#        return t
#        t = collect(0.5*I(N+1))
#        t[1,1] = 10^6
#        H2 = -(t*D1X* D1X)[2:end-1, 2:end-1] + V[2:end-1, 2:end-1]
#        H2 = -0.5*D*D + V
        
        vals = eigvals(H2)
        println("$i vals ", vals[1:5])
    end
    return vals
end




function srl2(Z, N, Rmax)

    r, D, D2, w, rall,wall,H_poisson, D1X, D2X = setup_mats(;N = N, a=0.0, Rmax=Rmax)

    V = diagm(-Z ./ ( rall))
#
    alpha = 1/ 137.0359895

    c= (1/alpha)
    vals, vects = eigen(-0.5*D2 .+ V[2:end-1, 2:end-1])
    println("pert ", vects[:,2]' * (- 1/(8*c^2) * (D1X^4)[2:end-1, 2:end-1] - 1.0/(4*c^2) * diagm(1 ./ rall)[2:end-1, 2:end-1] * (D1X^3)[2:end-1, 2:end-1]) * vects[:,2])
    println("no pert ", vects[:,1]' * (-0.5*D2 .+ V[2:end-1, 2:end-1])*vects[:,1])
    println("0 vals ", vals[1:5])
    for i = 1:1
        H2 = (-0.5*D1X* D1X)[2:end-1, 2:end-1] + V[2:end-1, 2:end-1] - 1/(8*c^2) * (D1X^4)[2:end-1, 2:end-1] - 1/(4*c^2) * diagm(1 ./ rall)[2:end-1, 2:end-1] * (D1X^3)[2:end-1, 2:end-1]
        vals = eigvals(H2)
        println("$i vals ", vals[1:5])
#
    end
    return vals
end



end #end module
