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
using Base.Threads

using SphericalHarmonics
using FastSphericalHarmonics

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
        return 4 * pi * 0.5 * sum( e_LDA.(rho) .* rho .* wall .* rall.^2  )  #hartree

        
    elseif nspin == 2


        rho_tot = rho[:,1] + rho[:,2]
        ζ = (rho[:,1] - rho[:,2])./rho_tot

        return 4 * pi * 0.5 * sum( e_LDA_sp.(rho_tot, ζ) .* rho_tot .* wall .* rall.^2 )  #hartree
    end        
end

function calc_energy_hartree(rho, N, Rmax, rall, wall, VH)

    return 0.5 * 4 * pi * sum(VH .* rho .* wall .* rall.^2 )
    
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

        println("check ", rho[1:3,spin])

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

function assemble_rho_LM(vals_r, vects, nel, rall, wall, rho, N, Rmax, D2; nspin=1, lmax=0, ig1=missing)

    rho[:] .= 0.0

    if ismissing(ig1)
        ig1= ones(size(rall))
    end
    
    vnorm = zeros(N+1)

    eps =  rall[2]/5.0

    filling = zeros(size(vals_r))
    rhor2 = zeros(size(rho)[1:2])
    
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
                    vnorm[2:N] = vnorm[2:N] #./ rall[2:N].^2

                    vnorm = vnorm / (4.0*pi*sum(vnorm .* wall .* ig1))# .* rall.^2))

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

                        filling[i,spin,l+1,l+m+1] = nleft
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
                        filling[i,spin,l+1,l+m+1] = fillval / nspin
                        
                    end

                    
                    
                end
            end
        end


        norm = 4.0*pi*sum(rho[:,spin,1,1] .* wall .*ig1 ) #/ (4*pi)^0.5  #.* rall.^2
#        println("norm ", norm)
        
        rhor2[:,spin] = rho[:,spin,1,1] #/norm*sum(nel[spin,:,:])
        d2r = (D2[:,:] * rho[:,spin,1,1])  #/ norm * sum(nel[:,spin])  

        for ll = 0:(lmax*2)
             for mm = -ll:ll
                 if ll == 0 && mm == 0
#                     println("norm $norm")
                     rho[2:end,spin, 1,1] = (rho[2:end,spin, 1, 1]./rall[2:end].^2  ) #/ norm * sum(nel[spin,:,:]) #* sqrt(4*pi)
                     rho[1,spin,1,1] = d2r[1]/2.0 # / norm * sum(nel[spin,:,:])
#                     println("check ", rho[1:3,spin,1,1])
#                     println("check ", sum(rho[:,spin,1,1] .* rall.^2 .* wall .* ig1)*4*pi*sqrt(4*pi) )
                     
                 else
                     d = FastSphericalHarmonics.sph_mode(ll,mm)
                     rho[2:end,spin, d[1], d[2]] = (rho[2:end,spin, d[1], d[2]]./rall[2:end].^2  ) 
                 end
             end
        end
        
        
    end
#
#    println("filling")
#    println(filling)
    
    return rho, filling, rhor2
    
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
        Vtot = zeros(size(rho)[1]-2, nspin, lmax+1)
        
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




        @threads for i = 1:length(spin_l)
            spin = spin_l[i][1]
            l = spin_l[i][2]
            
            VLDA_mat = diagm(VLDA[2:N,spin])
            if l == 0
                Vtot[:,spin, lmax+1] = VH[2:N] + VLDA[2:N,spin] + diag(Vc)
            end
            #            for l = 0:lmax

            Ham = H0_L[l+1] +0.0* (VH_mat + VLDA_mat)
            vals, v = eigen(Ham)
            #vals, v = eigs(Ham, which=:SM, nev=4)
            vals_r[:,spin,l+1] = real.(vals)
            vects[:,:,spin,l+1] = v
            
#            end
        end
        
        for spin = 1:nspin
            for l = 0:lmax
                println("iter $iter vals spin =  $spin l = $l ", vals_r[1:min(5, size(vals_r)[1]), spin, l+1])
            end
        end
        
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
                for toadd in   [ [1,0,0,2.0],
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
                    if toadd[4]+1e-15 < (num_el)
                        push!(T,toadd)
                        num_el -= 2.0
                        lmax = max(lmax, Int64(toadd[2]))
                    else
                        push!(T,[toadd[1],toadd[2], toadd[3], num_el])
                        lmax = max(lmax, Int64(toadd[2]))
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
    for t in T
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


function DFT_spin_l_grid_LM(fill_str; N = 40, nmax = 1, Z=1.0, Rmax = 10.0, rho_init = missing, niters=20, mix = 1.0, mixing_mode=:pulay, ax = 0.02)

    Z = Float64(Z)
    nel, nspin, lmax = setup_filling(fill_str)


    
    lmax_rho =  6
    
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

    
    
    function invgrid(x)
        return exp(x) - ax
    end

    function invgrid1(x)
        return grid1(x).^-1
    end

    
    
    begin
        r, D, D2, w, rall,wall,H_poisson, D1X, D2X = setup_mats(;N = N, a=grid(0.0), Rmax=grid(Rmax))


        
        rall_rs = invgrid.(rall)

        ig1 = invgrid1.(rall_rs)

        H_poisson = diagm(grid1.(rall_rs[2:N]).^2)*D2 + diagm(grid2.(rall_rs[2:N])) *D +  diagm( 2.0 ./ rall_rs[2:N] )*diagm(grid1.(rall_rs[2:N]))  * D
        D2grid = ( diagm(grid1.(rall_rs[2:N]).^2)*D2 + diagm(grid2.(rall_rs[2:N])) *D  )
        D2Xgrid = ( diagm(grid1.(rall_rs).^2)*D2X + diagm(grid2.(rall_rs)) *D1X  )

        
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


#        n1in = deepcopy(rho)
#        n1out = deepcopy(rho)
#        n2in = deepcopy(rho)
#        n2out = deepcopy(rho)
        
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
        
        vects = zeros(Complex{Float64}, N-1, N-1, nspin, lmax+1, 2*lmax+1)
        vals_r = zeros(Float64, N-1, nspin, lmax+1,2*lmax+1)    

        VH_LM = zeros(size(rho_LM)[[1,3,4]])
        VH = zeros(size(rho)[1])

        vlda_LM = zeros(size(rho_LM))
        VLDA = zeros(size(rho)[1])
        
        filling = zeros(size(vals_r))
        Vtot = zeros(size(rho)[1]-2, nspin, lmax+1)
        
    end

    eigs_tot_old = sum(filling.*vals_r)

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
    
    
    for iter = 1:niters

        println()
        #        VLDA = diagm(v_LDA.(rho[2:N]))
        
        #println("LDA")


        vlda_LM = VXC_LM(rho_LM *4*pi)
#        println("size vlda_LM ", size(vlda_LM))
        #vlda_LM = vlda_LM #/ sqrt(4*pi)
        
#        rho[:,:] = rho_LM[:,:,1,1] * sqrt(4*pi)

#        println("rho 11 ", rho[1:3, 1])
        
#        println("lda old")
#        @time if nspin == 1
#            VLDA[:,1] = v_LDA.(rho[:, 1])
#        elseif nspin == 2
#            tt =  v_LDA_sp.(rho[:, 1], rho[:, 2])
#            VLDA[:,:] = reshape(vcat(tt...), nspin,N+1)'
#        end

#        println("vlda  ", vlda_LM[1:3,1,1,1])
#        println("vldaO ", VLDA[1:3,1])
        #        println("vlda check ", sum(abs.(VLDA - vlda_LM[:,:,1,1])))
#        println("vlda check2 ", sum(abs.(VLDA))/ sum(abs.( vlda_LM[:,:,1,1])))
#        println(VLDA[1:5,1])
#        println(vlda_LM[1:5,1,1,1])
        
#        t = zeros(size(rho_LM))
#        for r in 1:size(rho_LM)[1]
#            t[r,1,:,:] = FastSphericalHarmonics.sph_evaluate(rho_LM[r,1,:,:])
#        end

#        println(t[1:5,1,1,1] * sqrt(4*pi))
#        println(rho_LM[1:5,1,1,1])
        
#        println("check rho ", sum(abs.(rho)) / sum(abs.(t)))
        
#        println("poisson")        
        begin
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
        
        

        #        @threads 
        for i = 1:length(spin_lm)
            spin = spin_lm[i][1]
            l = spin_lm[i][2]
            m = spin_lm[i][3]

            
            #            VLDA_mat = diagm(VLDA[2:N,spin])

            #VLDA_mat = diagm(vlda_LM[2:N,spin,1,1])

            #println("vlda before ", vlda_LM[2:5,1,1,1])

            d = FastSphericalHarmonics.sph_mode(l,m)
            
            VLDA[:] .= 0.0
            VH[:] .= 0.0
            for ll = 0:(lmax_rho)
                for mm = -ll:ll
                    #                    VH += 4*pi*VH_LM[:, ll+1, mm+ll+1] * real_gaunt_dict[(ll,mm,l,m)]
                    d2 = FastSphericalHarmonics.sph_mode(ll,mm)

                    VH += 4*pi*VH_LM[:, d2[1], d2[2]] * real_gaunt_dict[(ll,mm,l,m)] #/ (2*ll+1)
                    VLDA += vlda_LM[:,spin,d2[1], d2[2]] * real_gaunt_dict[(ll,mm,l,m)] #/ (2*ll+1)

#                    println("vlda zzz $ll $mm $l $m ", sum(abs.(vlda_LM[:,spin, d[1], d[2]])), " g ", real_gaunt_dict[(ll,mm,l,m)])
                    
                end
            end
            #println("vlda after ", VLDA[2:5])
            

            
            VH_mat = diagm(VH[2:N])
            VLDA_mat = diagm(VLDA[2:N])

            
            #            if l == 0
#                Vtot[:,spin, lmax+1] = VH[2:N] + VLDA[2:N,spin] + diag(Vc)
            #            end
            
            #            for l = 0:lmax

            Ham = H0_L[l+1] + (VH_mat + VLDA_mat)
            vals, v = eigen(Ham)
            #vals, v = eigs(Ham, which=:SM, nev=4)
#            println("spin $spin l $l m $m")
            
            vals_r[:,spin,d[1],d[2]] = real.(vals)
            vects[:,:,spin,d[1],d[2]] = v
            
#            end
        end
        
        for spin = 1:nspin
            for l = 0:lmax
                for m = -l:l
                    d = FastSphericalHarmonics.sph_mode(l,m)
                    println("iter $iter vals spin =  $spin l = $l m = $m   ", vals_r[1:min(5, size(vals_r)[1]), spin, d[1],d[2]])
                end
            end
        end
#        println("rho_LM ", rho_LM[1:3,1,1,1])
#        println("ass")
        rho_new_LM, filling, rhor2 = assemble_rho_LM(vals_r, vects, nel, rall_rs, wall, rho_LM, N, Rmax, D2Xgrid, nspin=nspin, lmax=lmax, ig1 = ig1)

#        println("size rho_new_LM ", size(rho_new_LM))
#        println("sum rho")
#        println(sum(abs.(rho_LM), dims=[1,2]))
#        println()
        
#        println("rho_new_LM ", rho_LM[1:3,1,1,1])

#        println("filling ", filling)

        eigs_tot_new = sum(filling.*vals_r)
        if iter > 1 && abs(eigs_tot_old-eigs_tot_new) < 1e-8
            println()
            println("eigval-based convergence reached ", abs(eigs_tot_old-eigs_tot_new))
            println()
            break
        end
        eigs_tot_old = eigs_tot_new

        
#        n1in[:] = n2in[:]
#        n1out[:] = n2out[:]

#        n2in[:] = rho_old[:]
#        n2out[:] = rho_new[:]
        
        
        #        if iter > 1 && mixing_mode == :pulay
        if false
            for spin = 1:nspin
                rho[:,spin] = pulay_mix(n1in[:,spin], n2in[:,spin], n1out[:,spin], n2out[:,spin], rho_old[:,spin], mix)
            end
        else
            rho_LM[:] = mix*rho_new_LM[:] + (1.0-mix)*rho_old_LM[:]
        end
        
#        rho_veryold[:] = rho_old[:]
        rho_old_LM[:] = rho_LM[:]
    end

    #println("energy")
#    energy = calc_energy(rho, N, Rmax, rall_rs, wall, VH, filling, vals_r, Vtot, Z, nspin=nspin, lmax=lmax, ig1=ig1)
    energy = 0.0
    return energy, vals_r, vects, rho_LM, rall_rs, wall.*ig1, rhor2, vlda_LM
    
end    


function VXC_LM(rho_LM)


    nspin = size(rho_LM)[2]
    nr = size(rho_LM)[1]
    rho = zeros(size(rho_LM))
    vlda = zeros(size(rho_LM))
    VLDA = zeros(nr,nspin)

    vlda_lm = zeros(size(rho_LM))

    #convert to real (theta phi) space
#    println("convert")

#    for r in 1:nr
#        for spin = 1:nspin
#            rho[r,spin,:,:] = FastSphericalHarmonics.sph_evaluate(rho_LM[r,spin,:,:])
#        end
#    end

    #rho2 = deepcopy(rho)
#    println("rho v2")

    rho = get_rho_rs(rho, rho_LM) 

    #    println(rho2[1:5,1,1,1])
#    println(rho[1:5,1,1,1])
    
    #test
    #test = FastSphericalHarmonics.sph_evaluate(ones(size(rho_LM[r,spin,:,:])))
    
    rho = max.(rho, 1e-30)
    
#    println("xxxxxxxxxxxxxxxxxxxxxxxxx")
    #evaluate vxc in (theta phi) space
    for l = 1:size(rho_LM)[3]
        for m = 1:size(rho_LM)[4]
            if nspin == 1
                VLDA[:,1] = v_LDA.(rho[:,1,l,m])
                vlda[:,1,l, m] = VLDA[:,1]

#                println("l $l m $m")
#                if l == 1 && m == 1
#                    println("rho11xc ", rho[1:3,1,1,1])
#                end
                
            elseif nspin == 2
                tt =  v_LDA_sp.(rho[:,1,l,m], rho[:, 2,l,m])
                VLDA[:,:] = reshape(vcat(tt...), nspin,nr)'
                vlda[:,1,l, m] = VLDA[:,1]
                vlda[:,2,l, m] = VLDA[:,2]                
            end
        end
    end
#    println("vlda tp ", vlda[1:3,1,1,1])
    #convert back to LM space
    for r in 1:nr
        for spin = 1:nspin
            vlda_lm[r,spin,:,:] = FastSphericalHarmonics.sph_transform(vlda[r,spin,:,:])
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

    
    return vlda_lm
    
    

end


function get_rho_rs(rho, rho_LM)

    nspin = size(rho)[2]
    
    lmax = size(rho_LM)[3]-1
    THETA,PHI = FastSphericalHarmonics.sph_points(lmax+1)

    rho[:,:,:,:] .= 0.0

    for (t,theta) in enumerate(THETA)
        for (p,phi) in enumerate(PHI) 
            Y = SphericalHarmonics.computeYlm(theta, phi, lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())
#           Y = SphericalHarmonics.computeYlm(theta, phi, lmax=lmax)
            for spin = 1:nspin
                for l = 0:lmax
                    for m = -l:l
                        d = FastSphericalHarmonics.sph_mode(l,m)
                        rho[:,spin,t, p] += rho_LM[:,spin, d[1], d[2]] * Y[(l,m)]
                    end
                end
            end
        end
    end
            
    return rho

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


end #end module
