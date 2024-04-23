
module CI

using ..GalerkinDFT:get_vect_gal
using ..GalerkinDFT:get_Svect_gal
using ..Galerkin:do_1d_integral
using ..AngularIntegration:real_gaunt_dict
using ..AngularIntegration:makeleb
using Combinatorics
using SparseArrays
using LoopVectorization
using Suppressor
using Base.Threads
using LinearAlgebra
using Arpack

function matrix_coulomb_small_old(dat, nlms1, nlms2, nlms3, nlms4)

    
    v1=get_Svect_gal(dat, nlms1[1], nlms1[2], nlms1[3], nlms1[4])
    v2=get_Svect_gal(dat, nlms2[1], nlms2[2], nlms2[3], nlms2[4])
    v3=get_Svect_gal(dat, nlms3[1], nlms3[2], nlms3[3], nlms3[4])
    v4=get_Svect_gal(dat, nlms4[1], nlms4[2], nlms4[3], nlms4[4])

    V1 = dat.mat_n2m * v1
    V2 = dat.mat_n2m * v2
    V3 = dat.mat_n2m * v3
    V4 = dat.mat_n2m * v4

    r12m = (V1.*V2)
    r34m = (V3.*V4)
    r12n = dat.mat_m2n*r12m
    r12n_dR = dat.S * dat.mat_m2n*(r12m ./ dat.R)  *  (2.0/dat.rmax)
    r34n = dat.mat_m2n*r34m

    println(" 1d int    ", do_1d_integral(r12n, dat.g))
    println(" 1d int dR ", do_1d_integral(r12n_dR, dat.g))
    println(" 1d int    ", do_1d_integral(r34n, dat.g))
    
    
    l = 0

    println("size r12 m ", size(r12m), " n ", size(r12n))
    vh = (dat.D2 + l*(l+1)*dat.V_L) \ r12n_dR
    vhv = dat.mat_n2m * vh 
    MP = sum( r12m .* dat.g.w[2:dat.M+2,dat.M] .* dat.R.^l) / (2*l+1)
    println("MP $MP")
    println("size vh ", size(vh))
    println("dat R ", size(dat.R))
    println("sum vh ", sum(vhv)/(4*pi)*sqrt(pi) / (20/2)^2 )
    vhx = ( vhv ./ dat.R / sqrt(4*pi) /2   .+ MP / dat.g.b^(l+1) * sqrt(pi)/(2*pi) ) #.* dat.g.w[2:dat.M+2, dat.M]

    println("sum 1 ", sum(vhv ./ dat.R.* dat.g.w[2:dat.M+2, dat.M] / sqrt(4*pi) /2 ))
    println("sum 2 ", sum(MP / dat.g.b^(l+1) * sqrt(pi)/(2*pi) *  dat.g.w[2:dat.M+2, dat.M]))
    println("sum vh ", sum(vhx))
    println("sum r34m ", sum(r34m))
    
    return sum(r34m.*vhx.* dat.g.w[2:dat.M+2,dat.M]) * (2.0*0.5*4*pi/ sqrt(4*pi)/2)


end

function precalc_twobody(dat, s, Z0)

    ns = size(s)[1]
    H_2bdy = zeros(ns, ns)
    for i1 in 1:ns
        #        n1,l1,m1,s1 = s[i1,:]
        for i2 in 1:ns
            #            n2,l2,m2,s2 = s[i2,:]
            H_2bdy[i1,i2] = matrix_twobody_small(dat, s[i1,:], s[i2,:], Z0)[1]
        end
    end
    return H_2bdy

end

function dist2(R1,theta1,phi1, R2, theta2, phi2)
    return (R1^2 + R2 ^2 - 2 * R1 * R2 *(sin(theta1)*sin(theta2)*cos(phi1-phi2) + cos(theta1)*cos(theta2)))
end

#function dist2(R1,theta1,phi1, R2, theta2, phi2)
#    return (R1^2 + R2 ^2 - 2 * R1 * R2 *(sin(theta1)*sin(theta2)*cos(phi1-phi2) + cos(theta1)*cos(theta2)))
#end

function convolve_gaussian(dat, a, lmax, lint,nlms)

    l = nlms[2]
    m = nlms[3]    
    vv = get_vect_gal(dat, nlms[1], nlms[2], nlms[3], nlms[4])
    VV = dat.mat_n2m * vv

    R = dat.R

    #VV2 = VV.^2 ./ R.^2

    
    @time LEB = makeleb(lint , lmax=lmax)
    M = dat.M
    N = dat.N
    
    stheta = sin.(LEB.θ[:])
    sphi = sin.(LEB.ϕ[:])

    ctheta = cos.(LEB.θ[:])
    cphi = cos.(LEB.ϕ[:])


    RA = zeros(M+1,LEB.N )

    f = 1/2/a^2
    f2 = 1/sqrt(2*pi)^3 / a^3 * (dat.g.b-dat.g.a)/2 * 4*pi

    VV2 = exp.(-R.^2 / 2)
    
    @turbo for ntp1 = 1:LEB.N
#        theta1 = LEB.θ[ntp1]
        #        phi1 = LEB.ϕ[ntp1]

        for ntp2 = 1:LEB.N
#            theta2 = LEB.θ[ntp2]
#            phi2 = LEB.ϕ[ntp2]
            for r1 = 1:M+1
                R1 = dat.R[r1]
                for r2 = 1:M+1
                    R2 = dat.R[r2]                    

                    dist2 = (R[r1]^2 + R[r2]^2 - 2*R[r1]*R[r2]*(stheta[ntp1]*stheta[ntp2]*(cphi[ntp1]*cphi[ntp2] + sphi[ntp1]*sphi[ntp2]) + ctheta[ntp1]*ctheta[ntp2]))
                    RA[r1, ntp1] += f2 * VV2[r2]*LEB.Ylm_arr[l+1,m+l+1, ntp2]*exp(-f * dist2 ) * R2^2 * dat.g.w[r2+1,M] * LEB.w[ntp2]
                end
            end
        end
    end

    return R, RA, VV2
    
end

function soft_coulomb(dat, a, lmax, lint)

    #    LEB = dat.LEB
    @time LEB = makeleb(lint , lmax=lmax )
    @time LEB2 = makeleb(5 , lmax=lmax )


    M = dat.M
    N = dat.N
    
    stheta = sin.(LEB.θ[:])
    sphi = sin.(LEB.ϕ[:])

    ctheta = cos.(LEB.θ[:])
    cphi = cos.(LEB.ϕ[:])

    R = dat.R
    println("ra")
#    @time RA = zeros(LEB.N, LEB.N, M+1, M+1)
#=
    @time @turbo for ntp1 = 1:LEB.N

        
        #        theta1 = LEB.θ[ntp1]
#        phi1 = LEB.ϕ[ntp1]

        for ntp2 = 1:LEB.N
#            theta2 = LEB.θ[ntp2]
#            phi2 = LEB.ϕ[ntp2]
            for r1 = 1:M+1
#                R1 = dat.R[r1]
                for r2 = 1:M+1
#                    R2 = dat.R[r2]                    
                    #                    RA[ntp1, ntp2, r1,r2] =  1/ sqrt(a^2 + dist2(R1,theta1,phi1, R2, theta2, phi2))
                    RA[ntp1, ntp2, r1,r2] =  a^2 + R[r1]^2 + R[r2]^2 - 2*R[r1]*R[r2]*(stheta[ntp1]*stheta[ntp2]*(cphi[ntp1]*cphi[ntp2] + sphi[ntp1]*sphi[ntp2]) + ctheta[ntp1]*ctheta[ntp2])
                    #RA[ntp1, ntp2, r1,r2] =  1.0
                end
            end
        end
    end

    RA = 1 ./ sqrt.(RA)
=#
    #lmaxrho=max(dat.lmaxrho, dat.lmax*2)
    lmaxrho = lmax
    println("mem2")
    @time RA_lm_lm = zeros(lmaxrho+1, 2*lmaxrho+1, lmaxrho+1, 2*lmaxrho+1, M+1, M+1)
    t=0.0
#=    @time for l1 = 0:lmaxrho
        for m1 = -l1:l1
            for l2 = 0:lmaxrho
                for m2 = -l2:l2
                    @turbo for r1 = 1:M+1
                        for r2 = 1:M+1
                            for ntp1 = 1:LEB.N
                                for ntp2 = 1:LEB.N
                                    RAT = 1/sqrt( a^2 + R[r1]^2 + R[r2]^2 - 2*R[r1]*R[r2]*(stheta[ntp1]*stheta[ntp2]*(cphi[ntp1]*cphi[ntp2] + sphi[ntp1]*sphi[ntp2]) + ctheta[ntp1]*ctheta[ntp2]))
                                    #RA_lm_lm[l1+1, l1+m1+1, l2+1, l2+m2+1, r1,r2] += (4*pi)*LEB.Ylm_arr[l2+1,m2+l2+1, ntp2]*LEB.Ylm_arr[l1+1,m1+l1+1, ntp1]*RA[ntp1, ntp2, r1,r2] * LEB.w[ntp1] * LEB.w[ntp2]
                                    #t += RA[ntp1, ntp2, r1,r2] * LEB.w[ntp1] * LEB.w[ntp2] * exp(-R[r1])^2 * exp(-R[r2])^2 * dat.g.w[r1+1,M] * dat.g.w[r2+1,M] * R[r1]^2 *  R[r2]^2
                                    t += RAT * LEB.w[ntp1] * LEB.w[ntp2] * exp(-R[r1])^2 * exp(-R[r2])^2 * dat.g.w[r1+1,M] * dat.g.w[r2+1,M] * R[r1]^2 *  R[r2]^2
                                    #t +=  LEB.w[ntp1] * LEB.w[ntp2] * dat.g.w[r1+1,M] * dat.g.w[r2+1,M] * R[r1]^2 *  R[r2]^2

                                    
                                end
                            end
                        end
                    end
                end
            end
        end
    end
=#
    t2 = 0.0
    ntp2 = 1
    t1 = 0.0
    #    @time for l1 = 0:lmaxrho
#        for m1 = -l1:l1
#            for l2 = 0:lmaxrho
#                for m2 = -l2:l2
  #                  @tturbo for r1 = 1:M+1
  #                      for r2 = 1:M+1
  #                          for ntp1 = 1:LEB.N
#                                for ntp2 = 1:LEB2.N
                                #RAT = 1/sqrt( a^2 + R[r1]^2 + R[r2]^2 - 2*R[r1]*R[r2]*(stheta[ntp1]*stheta[ntp2]*(cphi[ntp1]*cphi[ntp2] + sphi[ntp1]*sphi[ntp2]) + ctheta[ntp1]*ctheta[ntp2]))
                                    #RA_lm_lm[l1+1, l1+m1+1, l2+1, l2+m2+1, r1,r2] += (4*pi)*LEB.Ylm_arr[l2+1,m2+l2+1, ntp2]*LEB.Ylm_arr[l1+1,m1+l1+1, ntp1]*RA[ntp1, ntp2, r1,r2] * LEB.w[ntp1] * LEB.w[ntp2]
                                    #t += RA[ntp1, ntp2, r1,r2] * LEB.w[ntp1] * LEB.w[ntp2] * exp(-R[r1])^2 * exp(-R[r2])^2 * dat.g.w[r1+1,M] * dat.g.w[r2+1,M] * R[r1]^2 *  R[r2]^2
#                                    t2 += RAT * LEB.w[ntp1]  * exp(-R[r1])^2 * exp(-R[r2])^2 * dat.g.w[r1+1,M] * dat.g.w[r2+1,M] * R[r1]^2 *  R[r2]^2 
                                    #t +=  LEB.w[ntp1] * LEB.w[ntp2] * dat.g.w[r1+1,M] * dat.g.w[r2+1,M] * R[r1]^2 *  R[r2]^2
#                                dist = R[r1]^2 + R[r2]^2 - 2*R[r1]*R[r2]*(stheta[ntp1]*stheta[ntp2]*(cphi[ntp1]*cphi[ntp2] + sphi[ntp1]*sphi[ntp2]) + ctheta[ntp1]*ctheta[ntp2])
#                                t += LEB.w[ntp1]* dat.g.w[r1+1,M] * dat.g.w[r2+1,M] * R[r1]^2 *  R[r2]^2 * exp(-dist^2 / a^2)
#                                t += LEB.w[ntp1]* dat.g.w[r1+1,M] * dat.g.w[r2+1,M] *  R[r1]^2 * exp(-dist^2 / a^2) 
 #                               end
 #                           end
#                        end
#                    end
#                end
#            end
#        end
#    end
    t = 0.0
    @tturbo for r1 = 1:M+1
        for ntp1 = 1:LEB.N

            t += LEB.w[ntp1]* dat.g.w[r1+1,M] *  R[r1]^2 * exp(-0.5*R[r1]^2 / a^2) / a^3 / sqrt(2*pi)^3
        end
    end


    t2 = 0.0
    @tturbo for r1 = 1:M+1
        for ntp1 = 1:LEB.N
            for r2 = 1:M+1
                for ntp2 = 1:LEB.N
                    dist2 = R[r1]^2 + R[r2]^2 - 2*R[r1]*R[r2]*(stheta[ntp1]*stheta[ntp2]*(cphi[ntp1]*cphi[ntp2] + sphi[ntp1]*sphi[ntp2]) + ctheta[ntp1]*ctheta[ntp2])
                    #t2 += LEB.w[ntp1]* dat.g.w[r1+1,M] *  R[r1]^2 * exp(-0.5*R[r1]^2 / a^2) / a^3 / sqrt(2*pi)^3 * LEB.w[ntp2]* dat.g.w[r2+1,M] *  R[r2]^2 * exp(-0.5*R[r2]^2 / a^2) / a^3 / sqrt(2*pi)^3
                    t2 += LEB.w[ntp1]* dat.g.w[r1+1,M] *  R[r1]^2 * exp(-0.5*R[r1]^2 / a^2) / a^3 / sqrt(2*pi)^3 * LEB.w[ntp2]* dat.g.w[r2+1,M] *  R[r2]^2 * exp(-0.5*dist2 / a^2) / a^3 / sqrt(2*pi)^3
                end
            end
        end
    end
    
    return t * ( (dat.g.b - dat.g.a) / 2 * (4*pi)), t2* ( (dat.g.b - dat.g.a) / 2 * (4*pi))^2
#    return RA, RA_lm_lm
    #   println("test ", RA_lm_lm[1] - RA[1], " " , [RA_lm_lm[1] , RA[1]])
    
    
    RA_lm_lm_NN = zeros(lmaxrho+1, 2*lmaxrho+1, lmaxrho+1, 2*lmaxrho+1, N-1, N-1)
    t = 0.0
    @time for l1 = 0:lmaxrho
        for m1 = -l1:l1
            for l2 = 0:lmaxrho
                for m2 = -l2:l2
                    @tturbo for n1 = 1:(N-1)
                        for n2 = 1:(N-1)
                            for r1 = 1:M+1
                                for r2 = 1:M+1
                                    #                                    RA_lm_lm_NN[l1+1, l1+m1+1, l2+1, l2+m2+1, n1, n2] += dat.g.bvals[r1+1,n1,M] * dat.g.bvals[r2+1,n2,M] * RA_lm_lm[l1+1, l1+m1+1, l2+1, l2+m2+1, r1,r2] * dat.g.w[r1+1,M] * dat.g.w[r2+1,M]
#                                    t += RA_lm_lm[l1+1, l1+m1+1, l2+1, l2+m2+1, r1,r2] * dat.g.w[r1+1,M] * dat.g.w[r2+1,M] * exp(-R[r1])^2 * exp(-R[r2])^2
                                    
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    println("t $t")
    return t
#    RA = RA  / (dat.g.b- dat.g.a)^4 * 2^2
#    RA_lm_lm = RA_lm_lm  / (dat.g.b- dat.g.a)^4 * 2^4
#    RA_lm_lm_NN = RA_lm_lm_NN  / (dat.g.b- dat.g.a)^4 * 2^4
    
    return RA, RA_lm_lm, RA_lm_lm_NN
    
end

function precalc_coulomb(dat, s)

    ns = size(s)[1]
    LMAX = maximum(s[:,2])
    VH = zeros(Float64, ns, ns, 2*LMAX+1, dat.M+1)
    RR = zeros(Float64, ns, ns, dat.M+1)

    MAT = zeros(dat.N-1, dat.N-1, LMAX*2+1)
    for L = 0:(LMAX*2)
        MAT[:,:,L+1] = inv(dat.D2 + L*(L+1)*dat.V_L)
    end

    basis_dict = Dict{Array{ Int64}, Int64}()
    
    for i1 in 1:ns
        n1,l1,m1,s1 = s[i1,:]


        basis_dict[s[i1,:]] = i1
        
        v1=get_vect_gal(dat, n1,l1,m1,s1)
        V1 = dat.mat_n2m * v1

        for i2 in 1:ns

            n2,l2,m2,s2 = s[i2,:]
            if s1 != s2
                continue
            end

            v2=get_vect_gal(dat, n2,l2,m2,s2)
            V2 = dat.mat_n2m * v2

            r = V1 .* V2
            #rn = dat.mat_m2n*r
            RR[i1, i2, :] = r .* dat.g.w[2:dat.M+2,dat.M]
            rn_dR = dat.S * dat.mat_m2n*(r ./ dat.R)  *  (2.0/dat.rmax)
            for L = 0:(LMAX*2)
                vh = MAT[:,:,L+1] *  rn_dR
                vhv = dat.mat_n2m * vh 
                MP = sum( r .* dat.g.w[2:dat.M+2,dat.M] .* dat.R.^L) / (2*L+1) 
                vhx = ( vhv ./ dat.R / sqrt(4*pi) /2   .+ dat.R.^L ./ dat.g.b^L  / dat.g.b^(L+1) * MP * sqrt(pi)/(2*pi) )   #.* dat.g.w[2:dat.M+2, dat.M]
                
                
                VH[i1,i2,L+1, :] =  vhx * ( sqrt(pi) * (2*L+1) )
            end
        end
    end

    return VH, RR, basis_dict
    
end

function matrix_coulomb_small_precalc(dat, nlms1, nlms2, nlms3, nlms4, basis_dict, VH, RR)
    #    println("a")
    begin 
        i1 = basis_dict[nlms1]
        i2 = basis_dict[nlms2]
        i3 = basis_dict[nlms3]
        i4 = basis_dict[nlms4]
        
        l1 = nlms1[2]
        l2 = nlms2[2]
        l3 = nlms3[2]
        l4 = nlms4[2]

        m1 = nlms1[3]
        m2 = nlms2[3]
        m3 = nlms3[3]
        m4 = nlms4[3]

        ret = 0.0
        LMAX = max(l1,l2,l3,l4)*2
        Mp1=dat.M+1
        sym_factor = @view dat.hf_sym_big[l1+1, l1+m1+1,l3+1, l3+m3+1,l2+1,m2+l2+1,l4+1,m4+l4+1,:]
    end
    #println("b")

    @turbo for L = 0:LMAX
        ###sym_factor = dat.hf_sym_big[l1+1, l1+m1+1,l3+1, l3+m3+1,l2+1,m2+l2+1,l4+1,m4+l4+1, L+1]


        for ind = 1:Mp1
            ret += VH[i1,i2,L+1,ind] * RR[i3,i4,ind] * sym_factor[L+1]
        end

        #ret += 1.0
        
    end


    return ret

end


function matrix_coulomb_small(dat, nlms1, nlms2, nlms3, nlms4)

    #    return 0.0
    
    l1 = nlms1[2]
    l2 = nlms2[2]
    l3 = nlms3[2]
    l4 = nlms4[2]

    m1 = nlms1[3]
    m2 = nlms2[3]
    m3 = nlms3[3]
    m4 = nlms4[3]
    
    v1=get_vect_gal(dat, nlms1[1], nlms1[2], nlms1[3], nlms1[4])
    v2=get_vect_gal(dat, nlms2[1], nlms2[2], nlms2[3], nlms2[4])
    v3=get_vect_gal(dat, nlms3[1], nlms3[2], nlms3[3], nlms3[4])
    v4=get_vect_gal(dat, nlms4[1], nlms4[2], nlms4[3], nlms4[4])

    V1 = dat.mat_n2m * v1
    V2 = dat.mat_n2m * v2
    V3 = dat.mat_n2m * v3
    V4 = dat.mat_n2m * v4

    r12m = (V1.*V2)
    r34m = (V3.*V4)
    r12n = dat.mat_m2n*r12m
    if dat.orthogonalize
        r12n_dR = inv(dat.S5*dat.S5) * dat.mat_m2n*(r12m ./ dat.R)  *  (2.0/dat.rmax)
    else
        r12n_dR = dat.S * dat.mat_m2n*(r12m ./ dat.R)  *  (2.0/dat.rmax)
    end        
    r34n = dat.mat_m2n*r34m

    ret = 0.0

    for L = 0: (max(l1,l2,l3,l4)*2)
        #for L = 0:max(l1+l2, l3+l4)
        #for L = 0:(max[l1+l2,l3+l4]

        #for L = 0:2
        #        gc = 0.0
        #        for M = -L:L
        #            gc += real_gaunt_dict[(L,M,l1,m1,l2,m2)]
        #        end
        
        
        vh = inv(dat.S5)*(dat.D2 + L*(L+1)*dat.V_L)*inv(dat.S5) \ r12n_dR
        vhv = dat.mat_n2m * vh 
        MP = sum( r12m .* dat.g.w[2:dat.M+2,dat.M] .* dat.R.^L) / (2*L+1) 
        #        println("MP L $L  $MP")
        
        vhx = ( vhv ./ dat.R / sqrt(4*pi) /2   .+ dat.R.^L ./ dat.g.b^L  / dat.g.b^(L+1) * MP * sqrt(pi)/(2*pi) )   #.* dat.g.w[2:dat.M+2, dat.M]

        sym_factor = dat.hf_sym_big[l1+1, l1+m1+1,l3+1, l3+m3+1,l2+1,m2+l2+1,l4+1,m4+l4+1, L+1]
        #        if abs(sym_factor) > 1e-10
        #            println("ret ( $l1 $l2 $l3 $l4 ) ( $m1 $m2 $m3 $m4 ) L $L sym $sym_factor  :  ", sym_factor * sum(r34m.*vhx.* dat.g.w[2:dat.M+2,dat.M]) * sqrt(pi)  * (2*L+1) , "    MP : ", MP )
        #        end
        ret += sym_factor * sum(r34m.*vhx.* dat.g.w[2:dat.M+2,dat.M]) * sqrt(pi) * (2*L+1) 
        
        
    end
    
    return ret


end

function matrix_coulomb_small_fast(dat, nlms1, nlms2, nlms3, nlms4)

    l1 = nlms1[2]
    l2 = nlms2[2]
    l3 = nlms3[2]
    l4 = nlms4[2]

    m1 = nlms1[3]
    m2 = nlms2[3]
    m3 = nlms3[3]
    m4 = nlms4[3]
    
    v1=get_Svect_gal(dat, nlms1[1], nlms1[2], nlms1[3], nlms1[4])
    v2=get_Svect_gal(dat, nlms2[1], nlms2[2], nlms2[3], nlms2[4])
    v3=get_Svect_gal(dat, nlms3[1], nlms3[2], nlms3[3], nlms3[4])
    v4=get_Svect_gal(dat, nlms4[1], nlms4[2], nlms4[3], nlms4[4])

    V1 = dat.mat_n2m * v1
    V2 = dat.mat_n2m * v2
    V3 = dat.mat_n2m * v3
    V4 = dat.mat_n2m * v4

    r12m = (V1.*V2)
    #    r34m = (V3.*V4)
    #    r12n = dat.mat_m2n*r12m
    r12n_dR = dat.S * dat.mat_m2n*(r12m ./ dat.R)  *  (2.0/dat.rmax)
    #    r34n = dat.mat_m2n*r34m

    ret = 0.0
    
    for L = 0:max(l1+l2, l3+l4)
        sym_factor = dat.hf_sym_big[l1+1, l1+m1+1,l3+1, l3+m3+1,l2+1,m2+l2+1,l4+1,m4+l4+1, L+1]
        #        if sym_factor < 1e-10
        #            continue
        #        end
        #for L = 0:(max[l1+l2,l3+l4]

        #for L = 0:2
        #        gc = 0.0
        #        for M = -L:L
        #            gc += real_gaunt_dict[(L,M,l1,m1,l2,m2)]
        #        end
        
        
        #        vh = (dat.D2 + L*(L+1)*dat.V_L) \ r12n_dR
        #        vhv = dat.mat_n2m * vh 
        MP = sum( (r12m) .* (@view dat.g.w[2:dat.M+2,dat.M]) .* dat.R.^L) / (2*L+1) 
        #        println("MP L $L  $MP")
        
        vhx = ( (dat.mat_n2m * ((dat.D2 + L*(L+1)*dat.V_L) \ r12n_dR) ) ./ dat.R / sqrt(4*pi) /2   .+ dat.R.^L ./ dat.g.b^L  / dat.g.b^(L+1) * MP * sqrt(pi)/(2*pi) )   #.* dat.g.w[2:dat.M+2, dat.M]

        
        ret += sym_factor * sum(  (V3.*V4) .*vhx.* (@view dat.g.w[2:dat.M+2,dat.M])) * sqrt(pi) * (2*L+1) 
        
        
    end
    
    return ret


end

function matrix_twobody_small_precalc(nlms1, nlms2, basis_dict, H_2bdy)
    i1 = basis_dict[nlms1]
    i2 = basis_dict[nlms2] 
    return H_2bdy[i1,i2]
end


function matrix_twobody_small(dat, nlms1, nlms2, Z0)

    l1 = nlms1[2]
    l2 = nlms2[2]

    m1 = nlms1[3]
    m2 = nlms2[3]
    
    s1 = nlms1[4]
    s2 = nlms2[4]

    if l1 != l2 || m1 != m2  || s1 != s2
        return 0.0
    end

    v1=get_vect_gal(dat, nlms1[1], nlms1[2], nlms1[3], nlms1[4])
    v2=get_vect_gal(dat, nlms2[1], nlms2[2], nlms2[3], nlms2[4])

    #    if l1 == 0 && l2 == 0
    #        e_nuc = v2'*( dat.V_C ) * v1
    #    else
    #        e_nuc = 0.0
    #    end

    #    println("contr ", nlms1, nlms2, " ", v2'*(real_gaunt_dict[(0,0,l1,m1,l1,m2)]*sqrt(4*pi)*dat.V_C  +  dat.D2*0.0 + 0.0*dat.V_L*l1*(l1+1)) * v1)

    #    if s1 == s2
    #        ke = 1.0
    #    else
    #        ke = 0.0
    #    end

    #real_gaunt_dict[(0,0,l1,m1,l1,m2)]*sqrt(4*pi)*
    
    return v2'*(dat.V_C*Z0  +  dat.D2 + dat.V_L*l1*(l1+1)) * v1, v2'*dat.S*v1

end

function gen_basis(dat,  energy_max, nmax)

    
    nup = Int64(sum( dat.nel[1,:,:]))
    ndn = Int64(sum( dat.nel[2,:,:]))

    ground_state = Int64.(dat.filling)
    
    n_occ_max = findlast(sum(ground_state, dims=[2,3,4]) .> 0)[1]

    l_gs = 0
    m_gs = 0
    s_gs = 0
    for n = 1:n_occ_max
        for l = 0:dat.lmax
            for m = -l:l
                for spin = 1:dat.nspin
                    fill = ground_state[n, spin, l+1,m+l+1]
                    if fill == 1
                        if spin == 1
                            s_gs += 1
                            l_gs += l
                            m_gs += m
                            
                        else
                            s_gs += -1
                            l_gs += -l
                            m_gs += -m
                            
                        end
                    end
                end
            end
        end
    end
    println("ground state quantum numbers l $l_gs m $m_gs s $s_gs")
    
    
    val = zeros(Int64, 0,4)
    cond = zeros(Int64, 0,4)

    val_up = zeros(Int64, 0,4)
    cond_up = zeros(Int64, 0,4)

    val_dn = zeros(Int64, 0,4)
    cond_dn = zeros(Int64, 0,4)

    println("nmax $nmax")
    
    for spin = 1:dat.nspin
        for l = 0:dat.lmax
            for m = -l:l
                #for n = 1:max(min(dat.N-1, nmax[l+1]), n_occ_max)
#                println("add $l $m $(nmax[l+1])")
                for n = 1:min(dat.N-1,nmax[l+1])
                    
#                    println("add basis spin $spin l $l m $m n $n")
                    #                    println("add ", [n, spin, l+1, l+m+1], " ", dat.VALS[n, spin, l+1, l+m+1], " " , ground_state[n,spin, l+1, l+m+1])
                    if dat.VALS[n, spin, l+1, l+m+1] > energy_max
                        continue
                    end
                    if ground_state[n,spin, l+1, l+m+1] == 1
                        #                       println("occ")
                        vcat(val , [n l m spin])
                        if spin == 1
                            val_up = vcat(val_up , [n l m spin])
                            #                          println("CAT val_up ", val_up)
                            
                        else
                            val_dn = vcat(val_dn , [n l m spin])
                        end
                    else
                        #                     println("UNocc")                        
                        vcat(cond , [n l m spin])
                        if spin == 1
                            cond_up = vcat(cond_up , [n l m spin])
                        else
                            cond_dn = vcat(cond_dn , [n l m spin])
                        end
                    end
                end
            end
        end
    end

    return val_up, val_dn, cond_up, cond_dn, nup, ndn, [l_gs, m_gs, s_gs]
end

#function generate_excitations(dat, nexcite, energy_max, nmax)
function generate_excitations(nexcite, nup, ndn, val_up, val_dn, cond_up, cond_dn, qnums_gs, symmetry, s_up, s_dn)

    println("nexcite $nexcite")
    nval_up = size(val_up,1)
    nval_dn = size(val_dn,1)
    ncond_up = size(cond_up,1)
    ncond_dn = size(cond_dn,1)

    println("size basis $nval_up nval_up nval_dn $nval_dn ncond_up $ncond_up ncond_dn $ncond_dn")
    println()

    counter = 0
    for nex = 0:nexcite
        for nex_up = 0:nex
            nex_dn = nex - nex_up
            if nex_up > nup || nex_dn > ndn
                continue
            end
            counter += binomial(nval_up, nex_up) * binomial(ncond_up, nex_up) * binomial(nval_dn, nex_dn) * binomial(ncond_dn, nex_dn)
        end
    end
    println("Number of states: $counter")
    
    v_up = zeros(Bool, counter, nval_up+ncond_up)
    v_dn = zeros(Bool, counter, nval_dn+ncond_dn)
    counter = 0
    for nex = 0:nexcite
        for nex_up = 0:nex
            nex_dn = nex - nex_up
            if nex_up > nup || nex_dn > ndn
                continue
            end
            

            
            v_up_t =  gen_combos(nex_up, nval_up, ncond_up)
            v_dn_t =  gen_combos(nex_dn, nval_dn, ncond_dn)

            #            println("$counter nex $nex up $nex_up size $(size(v_up_t)) | dn $nex_dn  $(size(v_dn_t)) : $(size(v_up_t,1) * size(v_dn_t,1))")

            
            #println("up ", [nex_up, nval_up, ncond_up])
            #println(v_up_t)
            #println()
            #println("dn ", [nex_dn, nval_dn, ncond_dn])
            #println(v_dn_t)
            
            for i = 1:size(v_up_t,1)
                for j = 1:size(v_dn_t,1)
                    #                    if symmetry == false
                    counter += 1
                    v_up[counter,:] .= v_up_t[i,:]
                    v_dn[counter,:] .= v_dn_t[j,:]
                    #                    else
                    #                        l_up, m_up, spin_up = get_qn(v_up_t[i,:], s_up)
                    #                        l_dn, m_dn, spin_dn = get_qn(v_dn_t[j,:], s_dn)
                    #                        println("add $(v_up_t[i,:]) lms $l_up $m_up $spin_up $(v_dn_t[j,:]) lms $l_dn $m_dn $spin_dn")
                    #                        if m_up-m_dn == qnums_gs[2] && l_up-l_dn == qnums_gs[1]
                    #                            println("keep")
                    #                            counter += 1
                    #                            v_up[counter,:] .= v_up_t[i,:]
                    #                            v_dn[counter,:] .= v_dn_t[j,:]
                    #                        end                            
                    
                    #                    println("counter $counter $(v_up[counter,:]) $(v_dn[counter,:])")
                    #                    v_up = vcat(v_up,v_up_t[i,:]')
                    #                    v_dn = vcat(v_dn,v_dn_t[j,:]')
                end
            end
            #            println("-")
        end
    end
    v_up = v_up[1:counter,:]
    v_dn = v_dn[1:counter,:]
    #    println()
    #    println("total ", size(v_up))
    
    
    return v_up, v_dn
    
end

function get_qn(v,s_b)
    l=0
    m=0
    s=0
    c=0
    #    println("s_b")
    #    println(s_b)
    
    for xx in v
        #        println("s_b2")
        #        println(s_b)
        c+=1
        #       println("xx $xx")
        if xx
            #          println("s_b3")
            #          println(s_b)
            #          println(["c", c, l])
            #          println("typeof ", typeof(s_b))
            #          println("s_b[c,2], ", s_b[c,2])
            l += s_b[c,2]
            m += s_b[c,3]
            if s_b[c,4] == 1
                s += 1
            else
                s += -1
            end
        end
    end
    return l, m, s
end

function gen_combos( nex, nval, ncond)

    vground = zeros(Bool, 1, nval+ncond)
    vground[1:nval] .= true
    vground[nval+1:end] .= false

    if nex == 0
        return vground
    end
    
    ncombos = length(combinations(1:nval, nex)) * length(combinations(1:ncond, nex))

    v = zeros(Bool, ncombos, nval+ncond)

    counter = 1

    for (i_v, c_v) in enumerate(combinations(1:nval, nex))
        for (i_c, c_c) in enumerate(combinations( (nval+1):(nval+ncond), nex))

            v[counter,:] = vground[1,:]

            for cc_v in c_v
                v[counter,cc_v] = false
            end
            for cc_c in c_c
                v[counter,cc_c] = true
            end
            counter += 1
        end
    end

    return v
    
end


function construct_ham(dat, nexcite; energy_max = 1000000000000.0, nmax = 1000000, dense=false, symmetry=false, Z0=1.0)

    if dat.nspin == 1
        println("ERROR - need 2 spins")
        return nothing
    end
    if abs(sum(dat.nel) - sum(round.(dat.nel))) > 1e-10
        println("ERROR - need integer number of electrons")
        return nothing
    end

    nel = Int64(sum(dat.nel))
    if  nexcite > nel 
        nexcite=nel
        println("set nexcite to $nexcite")
    end

    #put qualfied states into a list
    println("gen_basis")
    @time val_up, val_dn, cond_up, cond_dn, nup, ndn, qnums_gs = gen_basis(dat, energy_max, nmax)

    s_up = [val_up; cond_up]
    s_dn = [val_dn; cond_dn]

    println("precalc")
    @time VH, RR, basis_dict = precalc_coulomb(dat, [s_up;s_dn])
    println("precalc 2bdy")
    @time H_2bdy = precalc_twobody(dat, [s_up;s_dn], Z0)
    #return H_2bdy
    #println("done precalc")

    if false
        aa1 = [1,0,0,1] #,[1,0,0,1], [1,0,0,1], [1,0,0,1]
        println("before xxx")
        @time    ret = matrix_coulomb_small_precalc(dat,aa1,aa1,aa1,aa1, basis_dict, VH, RR)
        println("after xxx")
        @time    ret2 =  matrix_coulomb_small(dat, [1,0,0,1] ,[1,0,0,1], [1,0,0,1], [1,0,0,1])
        println("test $ret vs $ret2")
        
        @time    ret = matrix_coulomb_small_precalc(dat,[1,0,0,1] ,[1,0,0,1], [2,0,0,1], [2,0,0,1], basis_dict, VH, RR)
        
        @time ret2 =  matrix_coulomb_small(dat, [1,0,0,1] ,[1,0,0,1], [2,0,0,1], [2,0,0,1])
        
        println("test $ret vs $ret2")

    end
    
    #    println( size(s_up) )
    #    println( size(val_up) )
    #    println( size(cond_up) )
    #    println("basis fn $nup $ndn")
    #    println("val_up")
    #    println(val_up)
    #    println("cond_up")
    #    println(cond_up)
    #    for i = 1:size(s_up,1)
    #        println("i $i $(s_up[i,:])     $(s_dn[i,:])  ")
    #    end
    #    println("--")
    
    #generate excitations
    println("generate ex")
    @time basis_up, basis_dn = generate_excitations(nexcite,nup, ndn, val_up, val_dn, cond_up, cond_dn, qnums_gs, symmetry, s_up, s_dn)

#        println("basis_up dn")
#        for i = 1:size(basis_up,1)
#            println("b $i   $(basis_up[i,:])  $(basis_dn[i,:])")
#        end
    
    #    ham = missing
    #    @time @suppress begin
    #        ham = make_ham(dat, s_up, s_dn, nup, ndn, basis_up, basis_dn,  dense=dense)
    #    end
    ham_pre = missing 
    println("make_ham_precalc")
    @time  begin #@suppress
        ArrI, ArrJ, ArrH, H = make_ham_precalc(dat, s_up, s_dn, nup, ndn, basis_up, basis_dn, basis_dict, VH, RR, H_2bdy, dense=dense, symmetry=symmetry, Z0=Z0)
    end
    #return ham[end], ham_pre[end]
    return ArrI, ArrJ, ArrH, H, basis_up, basis_dn, s_up, s_dn
end

function make_ham(dat, s_up, s_dn, nup, ndn, basis_up, basis_dn; dense=false)

    N = size(basis_up,1)
    ArrI = Int64[]
    ArrJ = Int64[]
    ArrH = Float64[]
    sign = 1.0
    h = 0.0
    @time for i = 1:N
        for j = 1:N
            #            println()
            count_up, locs_up = find_diff(basis_up[i,:], basis_up[j,:])
            count_dn, locs_dn = find_diff(basis_dn[i,:], basis_dn[j,:])

            
            sign = 1.0
            #            println("i $i j $j count $([count_up, count_dn]) $(basis_up[i,:])$(basis_dn[i,:]) ,   $(basis_up[j,:])$(basis_dn[j,:]) lup $locs_up ldn $locs_dn")
            #            continue
            h=0.0

            if count_up == 0 && count_dn == 0
                h = matrix_el_0(dat, s_up, s_dn, basis_up[i,:], basis_dn[i,:])
            end

            if count_up == 2 && count_dn == 0
                h = matrix_el_1_samespin(dat, s_up, basis_up[i,:], s_dn, basis_dn[i,:], locs_up)
                sign = (-1.0)^sum(basis_up[i,:][minimum(locs_up)+1:maximum(locs_up)-1])

            end
            if count_up == 0 && count_dn == 2
                h = matrix_el_1_samespin(dat, s_dn, basis_dn[i,:], s_up, basis_up[i,:], locs_dn)
                sign = (-1.0)^sum(basis_dn[i,:][minimum(locs_dn)+1:maximum(locs_dn)-1])
            end
            if count_up == 2 && count_dn == 2
                
                h = matrix_el_2_diffspin(dat, s_up, s_dn, locs_up, locs_dn)
                sign =      (-1.0)^sum( basis_up[i,:][minimum(locs_up)+1:maximum(locs_up)-1])
                sign = sign*(-1.0)^sum( basis_dn[j,:][minimum(locs_dn)+1:maximum(locs_dn)-1])

#                if (i == 2 || i == 6 ) && j == 25
#                    println("IJ $i $j $h $sign")
#                end
                
            end
            if false                
                if count_up == 4 && count_dn == 0
                    h = matrix_el_2_samespin(dat, s_up, locs_up)
                end
                if count_up == 0 && count_dn == 4
                    h = matrix_el_2_samespin(dat, s_dn, locs_dn)                
                end
            end
            if abs(h) > 1e-10
                #                if (i == 2 || i == 6 ) && j == 25
                #                    println("add IJ $i $j $h $sign")
                #                end

                push!(ArrI, i)
                push!(ArrJ, j)
                push!(ArrH, h*sign)

            end
            
        end
        #        println()
        
    end

    #    for (h,i,j) in zip(ArrH, ArrI, ArrJ)
    #        println("test $h    $i $j")
    #    end
    
    H = sparse(ArrI, ArrJ, ArrH)
    #    println(["IJ ",H[2,25], H[6,25]])
    #    println(["IJ ",H[25,2], H[25,6]])
    H = 0.5*(H + H')
    if dense
        H = collect(H)
    end

    return     ArrI, ArrJ, ArrH, H

end



function make_ham_precalc(dat, s_up, s_dn, nup, ndn, basis_up, basis_dn, basis_dict, VH, RR, H_2bdy; dense=false, symmetry=false, Z0=1.0)



    
    N = size(basis_up,1)

    #ArrI = Int64[]
    #ArrJ = Int64[]
    #ArrH = Float64[]

    ArrI_threads = []
    ArrJ_threads = []
    ArrH_threads = []

    for n = 1:nthreads()
        push!(ArrI_threads, Int64[])
        push!(ArrJ_threads, Int64[])
        push!(ArrH_threads, Float64[])
    end


    
    #    sign = 1.0
    #    h = 0.0
    #    count_up = 0
    #    count_dn = 0

    #    locs_tmp_up = zeros(Int64,4)
    #    locs_tmp_dn = zeros(Int64,4)

    locs_up_threads = zeros(Int64,4, nthreads())
    locs_dn_threads = zeros(Int64,4, nthreads())
    locs_up = zeros(Int64,4)
    locs_dn = zeros(Int64,4)
    W=size(basis_up,2)
    basis_tmp_threads = zeros(Int64, W, nthreads())
    basis_tmp_threads2 = zeros(Int64, W, nthreads())    


    println("symmetry part")
    @time if symmetry==false
        good_list = 1:N
    else
        i_ind = 1
        good_list =  [1 ] #start with ground state
        for j = 2:N
            id = 1
            locs_up = @view locs_up_threads[:,id]
            locs_dn = @view locs_dn_threads[:,id]
            basis_tmp = @view basis_tmp_threads[:,id]

            count_up, sign_up = find_diff_fast(basis_up, i_ind, j, basis_tmp, locs_up, W)
            count_dn, sign_dn = find_diff_fast(basis_dn, i_ind, j, basis_tmp, locs_dn, W)
            check = false
            if count_up == 2 && count_dn == 0
                check = matrix_el_1_samespin_precalc_sym(dat, s_up, (@view basis_up[i_ind,:]), s_dn, (@view basis_dn[i_ind,:]), locs_up, basis_dict, VH, RR, H_2bdy)
            elseif count_up == 0 && count_dn == 2
                check = matrix_el_1_samespin_precalc_sym(dat, s_dn, (@view basis_dn[i_ind,:]), s_up, (@view basis_up[i_ind,:]), locs_dn, basis_dict, VH, RR, H_2bdy)
            elseif count_up == 2 && count_dn == 2
                htemp = matrix_el_2_diffspin_precalc(dat, s_up, s_dn, locs_up, locs_dn, basis_dict, VH, RR)
                if abs(htemp) > 1e-8
                    check = true
                end
            elseif count_up == 4 && count_dn == 0
                #println("same spin up $i $j")
                htemp = matrix_el_2_samespin_precalc(dat, s_up, locs_up, basis_dict, VH, RR)
                if abs(htemp) > 1e-8
                    check = true
                end
            elseif count_up == 0 && count_dn == 4
                #println("same spin dn $i $j")
                htemp = matrix_el_2_samespin_precalc(dat, s_dn, locs_dn, basis_dict, VH, RR)
                if abs(htemp) > 1e-8
                    check = true
                end
            end
            if check == true
                push!(good_list, j)
            end
        end
    end
    if length(good_list) < 10
        println("symmetry $symmetry ", good_list)
    else
        println("symmetry $symmetry keep ", length(good_list))
    end        
    #@THREADS
    println("second part")
    locs_i = zeros(Int64, 2)
    locs_j = zeros(Int64, 2)
    @time for ii = 1:length(good_list) #1:N
        i = good_list[ii]
        id = threadid()
        htemp = 0.0 #for some reason the variable name h was causing problems with threads
        sign = 1.0
        for jj = 1:length(good_list) #1:N
            j = good_list[jj]

            #            println("thread id $id")
            #locs_up = locs_up_threads[:,id]
            #locs_dn = locs_dn_threads[:,id]
            basis_tmp = @view basis_tmp_threads[:,id]
            basis_tmp2 = @view basis_tmp_threads2[:,id]
            
            #count_up, sign_up = find_diff_fast(basis_up, i, j, basis_tmp, locs_up, W)
            #count_dn, sign_dn = find_diff_fast(basis_dn, i, j, basis_tmp, locs_dn, W)

#            count_up = 0
#            count_dn = 0
            count_up, sign_up = find_diff_fast2(basis_up, i, j, basis_tmp2, locs_up, W, locs_i, locs_j)
            count_dn, sign_dn = find_diff_fast2(basis_dn, i, j, basis_tmp2, locs_dn, W, locs_i, locs_j)
            #continue
            #println("compare $ii $jj  $([count_up, count_dn]):  ", [count_up - count_up2, count_dn - count_dn2, sign_up - sign_up2, sign_dn - sign_dn2])
            #
            #if sign_up != sign_up2
            #    println("up $sign_up $sign_up2")
            #    println(basis_up[i,:])
            #    println(basis_up[j,:])
            #    println()
            #end
#            count_up = 0
#            count_dn = 0
            
            #            println()

            #find_diff
            #            count_up, locs_up = find_diff(basis_up[i,:], basis_up[j,:])
            #            count_dn, locs_dn = find_diff(basis_dn[i,:], basis_dn[j,:])

            #            push!(ArrH_threads[id], h)#*sign)
            sign = 1.0
            if true
                #            push!(ArrH_threads[id], h)#*sign)

                #            continue

                #            count_up = find_diff_fast(basis_up, i, j, basis_tmp, locs_up, W)
                #            count_dn = find_diff_fast(basis_dn, i, j, basis_tmp, locs_dn, W)

                #            if count_up == 2
                #                if locs_up[1] == 0 || locs_up[1] == 0
                #            continue
                #            for xx in 1:length(locs_up)
                #                println("test dn  ", locs_tmp_up[xx] - locs_up[xx])
                #            end
                #            for xx in 1:length(locs_dn)
                #                println("test up  ", locs_tmp_dn[xx] - locs_dn[xx])
                #            end
                #            println("count $(count_up - count_up_tmp)   $(count_dn - count_dn_tmp)")
                #            println("locs_up")
                #            println(locs_up_old)
                #            println(locs_up)


                #            println("locs_dn")
                #            println(locs_dn)
                #            println(locs_tmp_dn)
                #count_dn, locs_dn = find_diff(basis_dn[i,:], basis_dn[j,:])
                
                htemp = 0.0
                sign = 1.0
                #            println("i $i j $j count $([count_up, count_dn]) $(basis_up[i,:])$(basis_dn[i,:]) ,   $(basis_up[j,:])$(basis_dn[j,:]) lup $locs_up ldn $locs_dn")
                #            continue
                if count_up == 0 && count_dn == 0
                    
                    htemp = matrix_el_0_precalc(dat, s_up, s_dn, (@view basis_up[i,:]), (@view basis_dn[i,:]), basis_dict, VH, RR, H_2bdy, Z0)
                    
                elseif count_up == 2 && count_dn == 0

                    htemp = matrix_el_1_samespin_precalc(dat, s_up, (@view basis_up[i,:]), s_dn, (@view basis_dn[i,:]), locs_up, basis_dict, VH, RR, H_2bdy, Z0=Z0)
                    sign = sign_up
                    
                    
                elseif count_up == 0 && count_dn == 2
                    htemp = matrix_el_1_samespin_precalc(dat, s_dn, (@view basis_dn[i,:]), s_up, (@view basis_up[i,:]), locs_dn, basis_dict, VH, RR, H_2bdy, Z0=Z0)
                    sign = sign_dn
                    
                elseif count_up == 2 && count_dn == 2
                    htemp = matrix_el_2_diffspin_precalc(dat, s_up, s_dn, locs_up, locs_dn, basis_dict, VH, RR)
                    sign = sign_up * sign_dn

                elseif count_up == 4 && count_dn == 0
                    htemp = matrix_el_2_samespin_precalc(dat, s_up, locs_up, basis_dict, VH, RR)
                    sign = sign_up
                elseif count_up == 0 && count_dn == 4
                    htemp = matrix_el_2_samespin_precalc(dat, s_dn, locs_dn, basis_dict, VH, RR)
                    sign = sign_dn
                    
                end
                #h = i + j

                #if false
            end
            #println(["h ", i, basis_up[i,:], basis_up[j,:], j, basis_dn[i,:], basis_dn[j,:], count_up, count_dn], "   sign $sign  ", locs_up, locs_dn)

            if abs(htemp) > 1e-10
                #                if (i == 2 || i == 6 ) && j == 25
                #                    println("add IJ $i $j $h $sign")
                #                end

                #                push!(ArrI, i)
                #                push!(ArrJ, j)
                #                push!(ArrH, h*sign)

                push!(ArrI_threads[id], ii)
                push!(ArrJ_threads[id], jj)
                push!(ArrH_threads[id], htemp * sign )#* sign)#*sign)
                #                if !( h ≈ Float64(i+j))
                #                    println("h $h $(i+j)")
                #                end
                #println(h - (i+j))

                
            end

        end
        #        println()
        
    end

    ArrI = collect(Iterators.flatten(ArrI_threads))
    ArrJ = collect(Iterators.flatten(ArrJ_threads))
    ArrH = collect(Iterators.flatten(ArrH_threads))



    if false                
        if count_up == 4 && count_dn == 0
            h = matrix_el_2_samespin(dat, s_up, locs_up)
        end
        if count_up == 0 && count_dn == 4
            h = matrix_el_2_samespin(dat, s_dn, locs_dn)                
        end
    end

    
    #    for (h,i,j) in zip(ArrH, ArrI, ArrJ)
    #        println("test $h    $i $j")
    #    end

    println("sparse")
    @time H = sparse(ArrI, ArrJ, ArrH)
    #    println(["IJ ",H[2,25], H[6,25]])
    #    println(["IJ ",H[25,2], H[25,6]])
    H = 0.5*(H + H')
    if dense
        H = collect(H)
    end

    return     ArrI, ArrJ, ArrH, H

end

function find_diff(v1,v2)

    
    count = sum(abs.(v1 - v2))
    if count >= 5
        return count, Int64[]
    else
        
        N = length(v1)
        count = 0
        locs = Int64[]
        for i = 1:N
            if v1[i] != v2[i]
                count += 1
                push!(locs, i)
            end
        end
        
        return count, locs
    end

    
end

function find_diff_fast(basis, i,j, basis_tmp, locs_tmp,W)
    
    count = 0
    for c = 1:W
        basis_tmp[c] = abs(basis[i,c] - basis[j,c])
        count += basis_tmp[c]
    end
    #count = sum(abs.(v1 - v2))
    sign = 1.0
    if count >= 5
        return count, 0.0
    else
        locs_tmp .= 0
        ind = 0


        locs_i = []
        locs_j = []
        
        for c = 1:W
            if basis_tmp[c] == 1
                ind += 1
                locs_tmp[ind] = c
            end
            if basis[i,c] == 1 && basis[j,c] == 0
                push!(locs_i, c)
            elseif basis[i,c] == 0 && basis[j,c] == 1
                push!(locs_j, c)
            end
                
        end
        if count == 2
            sign =  (-1.0)^sum( basis[j,min(locs_i[1], locs_j[1])+1:max(locs_j[1], locs_i[1])-1])

        elseif count == 4
            basis_t = basis[j,:]

            basis_t[locs_i[1]] = 1
            basis_t[locs_j[1]] = 0
            sign =      (-1.0)^sum( basis[j,min(locs_i[1], locs_j[1])+1:max(locs_j[1], locs_i[1])-1])
            sign =  sign*(-1.0)^sum( basis_t[min(locs_i[2], locs_j[2])+1:max(locs_j[2], locs_i[2])-1])

            basis_t[locs_i[2]] = 1
            basis_t[locs_j[2]] = 0
        end
    end
    return count, sign

    
end

function find_diff_fast2(basis, i,j, basis_tmp, locs_tmp,W, locs_i, locs_j)
    
    count = 0
    for c = 1:W
        basis_tmp[c] = abs(basis[i,c] - basis[j,c])
        count += basis_tmp[c]
    end
    #count = sum(abs.(v1 - v2))
    sign = 1.0
    if count >= 5
        return count, 0.0
    else
        #locs_tmp .= 0
        ind = 0


#        locs_i = []
#        locs_j = []

#        println("basis")
#        println(basis[i,:])
#        println(basis[j,:])
#        println("size locs_tmp $(size(locs_tmp))")

        

        c_i = 0
        c_j = 0
        
        for c = 1:W
            if basis_tmp[c] == 1
                ind += 1
                locs_tmp[ind] = c
            end
            if basis[i,c] == 1 && basis[j,c] == 0
                c_i += 1
                locs_i[c_i] = c
            elseif basis[i,c] == 0 && basis[j,c] == 1
                c_j += 1
                locs_j[c_j] = c
            end
        end
        if count == 2
            #sign =  (-1.0)^sum( basis[j,min(locs_i[1], locs_j[1])+1:max(locs_j[1], locs_i[1])-1])
            c = 0
            for ii = min(locs_i[1], locs_j[1])+1:max(locs_j[1], locs_i[1])-1
                c += basis[j,ii]
            end
            sign = (-1.0)^c
            
        elseif count == 4
            basis_t = basis[j,:]

            basis_t[locs_i[1]] = 1
            basis_t[locs_j[1]] = 0
            c=0
            for ii = min(locs_i[1], locs_j[1])+1:max(locs_j[1], locs_i[1])-1
                c += basis[j,ii]
            end
            sign = (-1.0)^c
            c=0
            for ii = min(locs_i[2], locs_j[2])+1:max(locs_j[2], locs_i[2])-1
                c += basis[j,ii]
            end
            sign = sign*(-1.0)^c
            
            #sign =      (-1.0)^sum( basis[j,min(locs_i[1], locs_j[1])+1:max(locs_j[1], locs_i[1])-1])
            #sign =  sign*(-1.0)^sum( basis_t[min(locs_i[2], locs_j[2])+1:max(locs_j[2], locs_i[2])-1])

            #basis_t[locs_i[2]] = 1
            #basis_t[locs_j[2]] = 0
        end
    end
    return count, sign

    
end

function matrix_el_0(dat, s_up, s_dn,b_up, b_dn)

    #    println("b_up ", b_up)
    #    println("b_dn ", b_dn)
    #    println()
    #    return 0.0, 0.0
    h = 0.0
    for (c,v) = enumerate(b_up)
        #        println("c $c v $v $(s_up[c,:]) ")
        if v == 1
            #            println("add $c $(s_up[c,:])  ", matrix_twobody_small(dat, s_up[c,:], s_up[c,:])[1])
            
            h += matrix_twobody_small(dat, s_up[c,:], s_up[c,:])[1]
        end
    end
    for (c,v) = enumerate(b_dn)
        if v == 1
            h += matrix_twobody_small(dat, s_dn[c,:], s_dn[c,:])[1]
        end
    end

    #    println("h0 $h")
    #    h=0.0#
    if true
        #println("h $h")
        hh1 = 0.0
        hh2 = 0.0
        hh3 = 0.0
        hh4 = 0.0
        
        hex1 = 0.0
        hex2 = 0.0
        for (c1,v1) = enumerate(b_up)
            if v1 == 1
                for (c2,v2) = enumerate(b_up)
                    if v2 == 1
                        #                        println("c1 $c1 c2 $c2")
                        #h += ( matrix_coulomb_small(dat, s_up[c1,:], s_up[c1,:], s_up[c2,:], s_up[c2,:]) - matrix_coulomb_small(dat, s_up[c1,:], s_up[c2,:], s_up[c1,:], s_up[c2,:]))

                        #hartree only
                        hh1 += ( matrix_coulomb_small(dat, s_up[c1,:], s_up[c1,:], s_up[c2,:], s_up[c2,:]) )
                        #                        println("h1 c1 $c1 c2 $c2 sup $(s_up[c1,:]) $(s_up[c2,:])")
                        #ex only
                        hex1 += ( - matrix_coulomb_small(dat, s_up[c1,:], s_up[c2,:], s_up[c1,:], s_up[c2,:]))

                    end
                end
            end
        end
        
        #        println("h $h")
        
        for (c1,v1) = enumerate(b_dn)
            if v1 == 1
                for (c2,v2) = enumerate(b_dn)
                    if v2 == 1

                        #h += ( matrix_coulomb_small(dat, s_dn[c1,:], s_dn[c1,:], s_dn[c2,:], s_dn[c2,:]) - matrix_coulomb_small(dat, s_dn[c1,:], s_dn[c2,:], s_dn[c1,:], s_dn[c2,:]))

                        #hartree only
                        hh2 += ( matrix_coulomb_small(dat, s_dn[c1,:], s_dn[c1,:], s_dn[c2,:], s_dn[c2,:]))

                        #ex only
                        hex2 += (  - matrix_coulomb_small(dat, s_dn[c1,:], s_dn[c2,:], s_dn[c1,:], s_dn[c2,:]))
                    end
                end
            end
        end
        #        println("h $h")

        if true
            for (c1,v1) = enumerate(b_up)
                if v1 == 1
                    for (c2,v2) = enumerate(b_dn)
                        if v2 == 1
                            hh3 += ( matrix_coulomb_small(dat, s_up[c1,:], s_up[c1,:], s_dn[c2,:], s_dn[c2,:])) #- matrix_coulomb_small(dat, s_up[c1,:], s_dn[c2,:], s_up[c1,:], s_dn[c2,:])
                        end
                    end
                end
            end
            #            println("h $h")
            for (c1,v1) = enumerate(b_up)
                if v1 == 1
                    for (c2,v2) = enumerate(b_dn)
                        if v2 == 1
                            hh4 += ( matrix_coulomb_small(dat, s_dn[c2,:], s_dn[c2,:], s_up[c1,:], s_up[c1,:])) #- ( matrix_coulomb_small(dat, s_dn[c2,:], s_dn[c2,:], s_up[c1,:], s_up[c1,:]))
                        end
                    end
                end
            end
        end
        h += (hh1 + hh2 + hh3 + hh4 + hex1 + hex2)
        #        println("hh $hh1 $hh2 $hh3 $hh4  hex $hex1 $hex2  tot $h ")
        
    end
    
    return h
end

function matrix_el_0_precalc(dat, s_up, s_dn,b_up, b_dn, basis_dict, VH, RR, H_2bdy, Z0)

    #    println("b_up ", b_up)
    #    println("b_dn ", b_dn)
    #    println()
    #    return 0.0, 0.0
    h = 0.0
#    h_test = 0.0

    for (c,v) = enumerate(b_up)
        #        println("c $c v $v $(s_up[c,:]) ")
        if v == 1
            #            println("add $c $(s_up[c,:])  ", matrix_twobody_small(dat, s_up[c,:], s_up[c,:])[1])
            h += matrix_twobody_small(dat, s_up[c,:], s_up[c,:], Z0)[1]
#            h_test += matrix_twobody_small_precalc((@view s_up[c,:]), (@view s_up[c,:]), basis_dict, H_2bdy)
        end
    end

    #    println("h $h h_test $h_test      $(h - h_test)")
    
    for (c,v) = enumerate(b_dn)
        if v == 1
            h += matrix_twobody_small(dat, s_dn[c,:], s_dn[c,:], Z0)[1]
        end
    end

    #    println("h0 $h")
    #    h=0.0#
    if true
        #println("h $h")
        hh1 = 0.0
        hh2 = 0.0
        hh3 = 0.0
        hh4 = 0.0
        
        hex1 = 0.0
        hex2 = 0.0
        for (c1,v1) = enumerate(b_up)
            if v1 == 1
                for (c2,v2) = enumerate(b_up)
                    if v2 == 1
                        #                        println("c1 $c1 c2 $c2")
                        #h += ( matrix_coulomb_small(dat, s_up[c1,:], s_up[c1,:], s_up[c2,:], s_up[c2,:]) - matrix_coulomb_small(dat, s_up[c1,:], s_up[c2,:], s_up[c1,:], s_up[c2,:]))

                        #hartree only
                        hh1 += ( matrix_coulomb_small_precalc(dat, (@view s_up[c1,:]), (@view s_up[c1,:]), (@view s_up[c2,:]), (@view s_up[c2,:]), basis_dict, VH, RR ))
                        #                        println("h1 c1 $c1 c2 $c2 sup $(s_up[c1,:]) $(s_up[c2,:])")
                        #ex only
                        hex1 += ( - matrix_coulomb_small_precalc(dat, (@view s_up[c1,:]), (@view s_up[c2,:]), (@view s_up[c1,:]),(@view s_up[c2,:]), basis_dict, VH, RR ))

                    end
                end
            end
        end
        
        #        println("h $h")
        
        for (c1,v1) = enumerate(b_dn)
            if v1 == 1
                for (c2,v2) = enumerate(b_dn)
                    if v2 == 1

                        #h += ( matrix_coulomb_small(dat, s_dn[c1,:], s_dn[c1,:], s_dn[c2,:], s_dn[c2,:]) - matrix_coulomb_small(dat, s_dn[c1,:], s_dn[c2,:], s_dn[c1,:], s_dn[c2,:]))

                        #hartree only
                        hh2 += ( matrix_coulomb_small_precalc(dat, (@view s_dn[c1,:]), (@view s_dn[c1,:]), (@view s_dn[c2,:]), (@view s_dn[c2,:]), basis_dict, VH, RR ))

                        #ex only
                        hex2 += (  - matrix_coulomb_small_precalc(dat, (@view s_dn[c1,:]), (@view s_dn[c2,:]), (@view s_dn[c1,:]), (@view s_dn[c2,:]), basis_dict, VH, RR ))
                    end
                end
            end
        end
        #        println("h $h")

        if true
            for (c1,v1) = enumerate(b_up)
                if v1 == 1
                    for (c2,v2) = enumerate(b_dn)
                        if v2 == 1
                            hh3 += ( matrix_coulomb_small_precalc(dat, (@view s_up[c1,:]), (@view s_up[c1,:]), (@view s_dn[c2,:]), (@view s_dn[c2,:]), basis_dict, VH, RR )) #- matrix_coulomb_small(dat, s_up[c1,:], s_dn[c2,:], s_up[c1,:], s_dn[c2,:])
                        end
                    end
                end
            end
            #            println("h $h")
            for (c1,v1) = enumerate(b_up)
                if v1 == 1
                    for (c2,v2) = enumerate(b_dn)
                        if v2 == 1
                            hh4 += ( matrix_coulomb_small_precalc(dat, (@view s_dn[c2,:]), (@view s_dn[c2,:]), (@view s_up[c1,:]), (@view s_up[c1,:]), basis_dict, VH, RR )) #- ( matrix_coulomb_small(dat, s_dn[c2,:], s_dn[c2,:], s_up[c1,:], s_up[c1,:]))
                        end
                    end
                end
            end
        end
        h += (hh1 + hh2 + hh3 + hh4 + hex1 + hex2)
        #        println("hh $hh1 $hh2 $hh3 $hh4  hex $hex1 $hex2  tot $h ")
        
    end
    
    return h
end

function matrix_el_1_samespin(dat, s, b, s2, b2, loc)

    h = matrix_twobody_small(dat, s[loc[1],:], s[loc[2],:])[1]
    #    println("h0 $h")
    for (c1,v1) = enumerate(b)
        if v1 == 1
            h += 2.0*( matrix_coulomb_small(dat, (@view s[c1,:]), (@view s[c1,:]), (@view s[loc[1],:]), (@view s[loc[2],:]) ) )
        end
    end
    #    println("h same hartree $h")
    for (c1,v1) = enumerate(b)
        if v1 == 1
            h += 2.0*(  - matrix_coulomb_small(dat, (@view s[loc[1],:]), (@view s[c1,:]), (@view s[loc[2],:]), (@view s[c1,:]) ))
        end
    end
    #    println("h same x $h")

    for (c2,v2) = enumerate(b2)
        if v2 == 1
            h += 2.0*( matrix_coulomb_small(dat, (@view s[loc[1],:]), (@view s[loc[2],:]), (@view s2[c2,:]), (@view s2[c2,:]))) 
        end
    end
    #    println("h diff x $h")
    #    println()
    #   println("h1 $h")

    return h
    
end

function matrix_el_1_samespin_precalc(dat, s, b, s2, b2, loc, basis_dict, VH, RR, H_2bdy; sym_check=false, Z0=1.0 )

    h = matrix_twobody_small_precalc( s[loc[1],:], s[loc[2],:], basis_dict, H_2bdy)

    for (c1,v1) = enumerate(b)
        if v1 == 1
            h += 2.0*( matrix_coulomb_small_precalc(dat, (@view s[c1,:]), (@view s[c1,:]), (@view s[loc[1],:]), (@view s[loc[2],:]), basis_dict, VH, RR ) )
        end
    end

    for (c1,v1) = enumerate(b)
        if v1 == 1
            h += 2.0*(  - matrix_coulomb_small_precalc(dat, (@view s[loc[1],:]), (@view s[c1,:]), (@view s[loc[2],:]), (@view s[c1,:]), basis_dict, VH, RR ))
        end
    end


    for (c2,v2) = enumerate(b2)
        if v2 == 1
            h += 2.0*( matrix_coulomb_small_precalc(dat, (@view s[loc[1],:]), (@view s[loc[2],:]), (@view s2[c2,:]), (@view s2[c2,:]), basis_dict, VH, RR )) 
        end
    end
    
    return h
    
end

function matrix_el_1_samespin_precalc_sym(dat, s, b, s2, b2, loc, basis_dict, VH, RR, H_2bdy)

    h = matrix_twobody_small_precalc( s[loc[1],:], s[loc[2],:], basis_dict, H_2bdy)
    if abs(h) > 1e-8
        return true
    else
        h = 0.0
        for (c1,v1) = enumerate(b)
            if v1 == 1
                h += 2.0*( matrix_coulomb_small_precalc(dat, (@view s[c1,:]), (@view s[c1,:]), (@view s[loc[1],:]), (@view s[loc[2],:]), basis_dict, VH, RR ) )
            end
        end

        for (c1,v1) = enumerate(b)
            if v1 == 1
                h += 2.0*(  - matrix_coulomb_small_precalc(dat, (@view s[loc[1],:]), (@view s[c1,:]), (@view s[loc[2],:]), (@view s[c1,:]), basis_dict, VH, RR ))
            end
        end
        
        for (c2,v2) = enumerate(b2)
            if v2 == 1
                h += 2.0*( matrix_coulomb_small_precalc(dat, (@view s[loc[1],:]), (@view s[loc[2],:]), (@view s2[c2,:]), (@view s2[c2,:]), basis_dict, VH, RR )) 
            end
        end
        if abs(h) > 1e-8
            return true
        end
    end
    return false
    
end

#function matrix_el_1_diffspin(dat, s, b, s2, b2, loc)
#
#    #h = matrix_twobody_small(dat, s[loc[1],:], s[loc[2],:])[1]
#    h = 0.0
#    println("h1b $h")
#    for (c1,v1) = enumerate(b)
#        if v1 == 1
#            h += 2.0*( matrix_coulomb_small(dat, s[loc[1],:], s[loc[2],:], s[c1,:], s[c1,:])) 
#        end
#    end
#    println("hsame $h")
#    for (c2,v2) = enumerate(b2)
#        if v2 == 1
#            h += 2.0*( matrix_coulomb_small(dat, s[loc[1],:], s[loc[2],:], s[c1,:], s[c2,:]))
#        end
#    end
#    println("diff $h")    
#    return h
#    
#end

function  matrix_el_2_diffspin(dat, s_up, s_dn, locs_up, locs_dn)

    #    println(s_up[locs_up[1],:])
    #    println(s_up[locs_up[2],:])
    #    println(s_dn[locs_dn[1],:])
    #    println(s_dn[locs_dn[2],:])

    h = 2.0*( matrix_coulomb_small(dat, s_up[locs_up[1],:], s_up[locs_up[2],:], s_dn[locs_dn[1],:], s_dn[locs_dn[2],:]))

    #h = 2.0*( matrix_coulomb_small(dat, s_up[locs_up[1],:], s_dn[locs_dn[1],:], s_up[locs_up[2],:], s_dn[locs_dn[2],:]))
    
    #    if locs_up[1] == 1 && locs_dn[1] == 1 && locs_up[2] == 5 && locs_dn[2] == 5 && abs(h) > 1e-10
    #        println("diff 2_diffspin $locs_up $locs_dn  s $s_up $s_dn   h $h")
    #    end
    return h
    
end

function  matrix_el_2_diffspin_precalc(dat, s_up, s_dn, locs_up, locs_dn, basis_dict, VH, RR )

    #    println(s_up[locs_up[1],:])
    #    println(s_up[locs_up[2],:])
    #    println(s_dn[locs_dn[1],:])
    #    println(s_dn[locs_dn[2],:])

    #    h = 2.0*( matrix_coulomb_small_precalc(dat, (@view s_up[locs_up[1],:]), (@view s_up[locs_up[2],:]), (@view s_dn[locs_dn[1],:]),( @view s_dn[locs_dn[2],:]), basis_dict, VH, RR ))
    h = 2.0*( matrix_coulomb_small_precalc(dat, ( s_up[locs_up[1],:]), ( s_up[locs_up[2],:]), ( s_dn[locs_dn[1],:]),(  s_dn[locs_dn[2],:]), basis_dict, VH, RR ))

    #h = 2.0*( matrix_coulomb_small(dat, s_up[locs_up[1],:], s_dn[locs_dn[1],:], s_up[locs_up[2],:], s_dn[locs_dn[2],:]))
    
    #    if locs_up[1] == 1 && locs_dn[1] == 1 && locs_up[2] == 5 && locs_dn[2] == 5 && abs(h) > 1e-10
    #        println("diff 2_diffspin $locs_up $locs_dn  s $s_up $s_dn   h $h")
    #    end
    return h
    
end

function  matrix_el_2_samespin_precalc(dat, s, locs, basis_dict, VH, RR )

#    println("locs $locs")

    h = 2.0*( matrix_coulomb_small_precalc(dat, ( s[locs[1],:]), ( s[locs[2],:]), ( s[locs[3],:]),(  s[locs[4],:]), basis_dict, VH, RR ))
    h += - 2.0*( matrix_coulomb_small_precalc(dat, ( s[locs[1],:]), ( s[locs[3],:]), ( s[locs[2],:]),(  s[locs[4],:]), basis_dict, VH, RR ))

    return h
end

#function  matrix_el_2_samespin(dat, s, locs)
#    println("same $locs")
#    h =  2.0*( matrix_coulomb_small(dat, s[locs[1],:], s[locs[3],:], s[locs[2],:], s[locs[4],:]))
    #h -= 2.0*( matrix_coulomb_small(dat, s[locs[1],:], s[locs[2],:], s[locs[3],:], s[locs[4],:]))
#    return h
#    
#end


function get_denmat(vects, H, basis_up, basis_dn, s_up, s_dn, dat; thr=1e-12)

    println("get_denmat $(size(vects))")

    lmax = dat.lmax
    M = dat.M
    nspin = dat.nspin
    N = dat.N
#    mat_m2n = dat.mat_m2n
#    mat_n2m = dat.mat_n2m

    nmax_up = size(s_up, 1)
    nmax_dn = size(s_dn, 1)

    nmax = max(nmax_up, nmax_dn)
    
    v_NN = zeros(N-1, N-1, nspin, nmax)

    bw = max(size(basis_up)[2], size(basis_dn)[2])
#    println("bw $bw")
    s_updn = [s_up, s_dn]

    denmat = zeros(N-1,N-1,nspin,lmax+1,lmax*2+1)

    N_basis = size(basis_up)[1]
    W_basis = size(basis_up)[2]

    println("N_basis $N_basis W_basis $W_basis")
    
    basis_tmp = zeros(Int64, W_basis)
    locs_i = zeros(Int64, 2)
    locs_j = zeros(Int64, 2)

    locs_up = zeros(Int64,4)
    locs_dn = zeros(Int64,4)
    
    for i = 1:N_basis
        for j = 1:N_basis
            
            weight = real(vects[i,1]*conj(vects[j,1]))

            b_up1 = basis_up[i,:]
            b_dn1 = basis_dn[i,:]

            b_up2 = basis_up[j,:]
            b_dn2 = basis_dn[j,:]

            
            count_up, sign_up = find_diff_fast2(basis_up, i, j, basis_tmp, locs_up, W_basis, locs_i, locs_j)
            count_dn, sign_dn = find_diff_fast2(basis_dn, i, j, basis_tmp, locs_dn, W_basis, locs_i, locs_j)

            bb1 = (b_up1, b_dn1)
            bb2 = (b_up2, b_dn2)

            if count_up == 0 && count_dn == 0

                for spin in [1,2]
#                    println("add onsite spin $spin     $i $j  $b1 $b2")

                    s = s_updn[spin]

                    b = bb1[spin]
                    for ind = 1:length(bb1[spin])
                        if b[ind] == 1

                            n1,l1,m1,spin_t1 = s[ind,:]
                            n2,l2,m2,spin_t2 = s[ind,:]

#                            println("add spin $spin ij $i $j  n1n2 $n1 $n2    spins $spin_t1 $spin_t2  $weight $(sum(0.5*weight*dat.VECTS_small[:,n1,spin_t1, l1+1, l1+m1+1]*dat.VECTS_small[:,n2,spin_t2, l2+1, l2+m2+1]'))")
                            
                            denmat[:,:,spin,l1+1,l1+m1+1] += 0.5*weight*dat.VECTS_small[:,n1,spin_t1, l1+1, l1+m1+1]*dat.VECTS_small[:,n2,spin_t2, l2+1, l2+m2+1]'
                            denmat[:,:,spin,l1+1,l1+m1+1] += 0.5*weight*dat.VECTS_small[:,n2,spin_t2, l2+1, l2+m2+1]*dat.VECTS_small[:,n1,spin_t1, l1+1, l1+m1+1]'
                        end
#                        if b[ind] == 1##

#                            n1,l1,m1,spin_t1 = s[ind,:]
#                            n2,l2,m2,spin_t2 = s[ind,:]

#                            println("add spin $spin ij $i $j  n1n2 $n1 $n2    spins $spin_t1 $spin_t2")
                            
#                            denmat[:,:,spin,l1+1,l1+m1+1] += 0.5*weight*dat.VECTS_small[:,n1,spin_t1, l1+1, l1+m1+1]*dat.VECTS_small[:,n2,spin_t2, l2+1, l2+m2+1]'
#                            denmat[:,:,spin,l1+1,l1+m1+1] += 0.5*weight*dat.VECTS_small[:,n2,spin_t2, l2+1, l2+m2+1]*dat.VECTS_small[:,n1,spin_t1, l1+1, l1+m1+1]'
#                        end

                    end
                end
            elseif count_up == 2 && count_dn == 0

                s = s_updn[1]
                b1 = bb1[1]
                b2 = bb2[1]
                n1,l1,m1,spin_t1 = s[locs_up[1],:]
                n2,l2,m2,spin_t2 = s[locs_up[2],:]
                
#                println("add $i $j , $n1 $n2     $locs_up   $b1  $b2    sign $sign_up")
                denmat[:,:,1,l1+1,l1+m1+1] += 0.5*sign_up*weight*dat.VECTS_small[:,n1,spin_t1, l1+1, l1+m1+1]*dat.VECTS_small[:,n2,spin_t2, l2+1, l2+m2+1]'
                denmat[:,:,1,l1+1,l1+m1+1] += 0.5*sign_up*weight*dat.VECTS_small[:,n2,spin_t2, l2+1, l2+m2+1]*dat.VECTS_small[:,n1,spin_t1, l1+1, l1+m1+1]'
                
            elseif count_up == 0 && count_dn == 2

                s = s_updn[2]
                b1 = bb1[2]
                b2 = bb2[2]
                n1,l1,m1,spin_t1 = s[locs_dn[1],:]
                n2,l2,m2,spin_t2 = s[locs_dn[2],:]

#                println("add dn $i $j , $n1 $n2     $locs_up   $b1  $b2    sign $sign_up")
                denmat[:,:,2,l1+1,l1+m1+1] += 0.5*sign_dn*weight*dat.VECTS_small[:,n1,spin_t1, l1+1, l1+m1+1]*dat.VECTS_small[:,n2,spin_t2, l2+1, l2+m2+1]'
                denmat[:,:,2,l1+1,l1+m1+1] += 0.5*sign_dn*weight*dat.VECTS_small[:,n2,spin_t2, l2+1, l2+m2+1]*dat.VECTS_small[:,n1,spin_t1, l1+1, l1+m1+1]'
                

                
                
            end
        end
    end
    
    #count_up, sign_up = find_diff_fast(basis_up, i_ind, j, basis_tmp, locs_up, W)
    #count_dn, sign_dn = find_diff_fast(basis_dn, i_ind, j, basis_tmp, locs_dn, W)

    
#=
    for c1 in 1:size(basis_up)[1]
        for c2 in 1:size(basis_up)[1]

            b_up1 = basis_up[c1,:]
            b_dn1 = basis_dn[c1,:]

            b_up2 = basis_up[c2,:]
            b_dn2 = basis_dn[c2,:]
            
            weight = real(vects[c1,1]*conj(vects[c2,1]))

            #        println("weight $c $weight")
            bb1 = (b_up1, b_dn1)
            bb2 = (b_up2, b_dn2)

            for spin in [1]
                b1 = bb1[spin]
                b2 = bb2[spin]
                f = 1.0
                for ind = 2:length(b1)
                    f = f * (b1[ind] == b2[ind])
                end
                l1 = 0
                m1 = 0

                s = s_updn[spin]
                n1,l1,m1,spin_t1 = s[ind,:]
                n2,l2,m2,spin_t2 = s[ind,:]

                denmat[:,:,spin,l1+1,l1+m1+1] += weight*dat.VECTS_small[:,n1,spin_t1, l1+1, l1+m1+1]*dat.VECTS_small[:,n2,spin_t2, l2+1, l2+m2+1]'
            end
        end
    end
=#    
#            for spin = 1:nspin
#                b1 = bb1[spin]
#                b2 = bb2[spin]
#                s = s_updn[spin]
#                for ind1 = 1:length(b1)
#                    for ind2 = 1:length(b2)
#                       if b1[ind1] == 1
#                            if b2[ind2] == 1
#                                n1,l1,m1,spin_t1 = s[ind1,:]
#                                n2,l2,m2,spin_t2 = s[ind2,:]
#                                if spin_t1 != spin
#                                    println("error $spin $spin_t1 $spin_t2")
#                                end
#                                if spin_t1 != spin_t2
#                                    continue
#                                end
#                                if l1 != l2
#                                    continue
#                                end
#                                if m1 != m2
#                                    continue
#                                end
#                                if spin == 1
#                                    #                                    denmat[:,:,spin,l1+1,l1+m1+1] += weight*dat.VECTS_small[:,n1,spin_t1, l1+1, l1+m1+1]*dat.VECTS_small[:,n2,spin_t2, l2+1, l2+m2+1]'
#                                    if c1 == c2
#                                        println("add c $c1 $c2  ind $ind1 $ind2  n  $n1 $n2  l $l1 $l2 m  $m1 $m2  s $spin_t1 $spin_t2    $weight")
#                                        denmat[:,:,spin,l1+1,l1+m1+1] += weight*dat.VECTS_small[:,n1,spin_t1, l1+1, l1+m1+1]*dat.VECTS_small[:,n2,spin_t2, l2+1, l2+m2+1]'
#                                    end
#                                end
#                                    
#                            end
#                        end
#                    end
#                    println("add denmat $c $spin")
#                end
#            end
#        end
#    end

    
#=
    for spin = 1:nspin
        s = s_updn[spin]
        for ss in 1:min(size(s,1), bw)
            n, l, m, spin_t = s[ss,:]
            v_NN[:,:,spin,ss] = dat.VECTS_small[:,n,spin_t, l+1, l+m+1]*dat.VECTS_small[:,n,spin_t, l+1, l+m+1]'
#            println("add v_NN $n ")
        end
    end

    denmat = zeros(N-1,N-1,nspin,lmax+1,lmax*2+1)
    for c in 1:size(basis_up)[1]

        b_up = basis_up[c,:]
        b_dn = basis_dn[c,:]

        weight = real(vects[c,1]*conj(vects[c,1]))

#        println("weight $c $weight")
        bb = (b_up, b_dn)
        for spin = 1:nspin
            b = bb[spin]
            s = s_updn[spin]
            for ind = 1:length(b)
                if b[ind] == 1
                    n,l,m,spin_t = s[ind,:]
                    denmat[:,:,spin,l+1,l+m+1] += weight*v_NN[:,:,spin,ind]
#                    println("add denmat $c $spin")
                end
            end
        end
    end
=#
#    sqrtS
    
    return denmat


end
    

function truncate_denmat(denmat, dat, nmax; thr=1e-8)

    lmax = dat.lmax
    M = dat.M
    nspin = dat.nspin
    N = dat.N

    sqrtS_inv = inv(dat.sqrtS)
#    VECTS_small = deepcopy(dat.VECTS_small)

    keep_dict = Dict()
    nkeep_max = 0
    VECTS_new = zeros(N-1, N-1, nspin, lmax+1, 2*lmax+1)
    
    #    for spin = 1:nspin
    for spin = 1:1
        for l = 0:lmax
            for m = -l:l
                #                valsX, vectsX = eigen(Hermitian(dat.sqrtS * denmat[:,:,spin,l+1,l+m+1] * dat.sqrtS))
                #                valsX, vectsX = eigen(Hermitian( denmat[:,:,spin,l+1,l+m+1]) , dat.S)
                #valsX, vectsX = eigen(Hermitian( denmat[:,:,spin,l+1,l+m+1]) , Hermitian(inv(dat.S)))

                valsX, vectsX = eigen(Hermitian( denmat[:,:,spin,l+1,l+m+1]))

                #for ii = 1:10
                #    println("valsX start $ii ", valsX[ii])
                #end
                #println()
                #for ii = 10:-1:1
                #    println("valsX end   $ii ", valsX[end-ii+1])
                #end
                #println()
#                vects = sqrtS_inv * vectsX
#                valsX, vects = eigen(denmat[:,:,spin,l+1,l+m+1])
#                println()
#                a = sum(abs.(vectsX[:,N-1:-1:1] - dat.VECTS_small[:,:,spin,l+1,l+m+1]), dims=1)
#                b = sum(abs.(vectsX[:,N-1:-1:1] + dat.VECTS_small[:,:,spin,l+1,l+m+1]), dims=1)
                #for ii = 1:30
                #    println("v $ii $(min(a[ii],b[ii]))")
                #end
                #                println("VECTS TEST ", sum(abs.(vects[:,N-1:-1:1] - dat.VECTS_small[:,:,spin,l+1,l+m+1]), dims=1))
#                println()
#                println("VECTS TEST2 ", sum(abs.(vects[:,N-1:-1:1] + dat.VECTS_small[:,:,spin,l+1,l+m+1]), dims=1))
#                println()

                keep = Int64[]
                c=0
                for n = (N-1):-1:(N-1 - nmax[l+1])
#                    println("spin $spin n $n  lm $l $m occ: $(valsX[n])")
                    if valsX[n] > thr
                        c+= 1
                        #VECTS_new[:,c,spin, l+1, l+m+1] = vects[:,n]   * sqrt(valsX[n])
                        VECTS_new[:,c,spin, l+1, l+m+1] = vectsX[:,n]
                        push!(keep, n)                        
                    else
                        #extra = n:(N-1)
                        #extra = (nmax[l+1]+1):(N-1)
#                        println("done c $c l $l m $m nmax[l+1] $(nmax[l+1]) N $N length(extra) $(length(extra))  $((c+1):(c+1+length(extra))) extra $(nmax[l+1]+1) $(N-1) ")
                        #                        println("update vects_new $(c+1) $(c+length(extra))")
                        #VECTS_new[:,(c+1):(c+length(extra)),spin,l+1,l+m+1] = dat.VECTS_small[:,extra,spin,l+1,l+m+1] 
                        break
                    end
                end
                keep_dict[(spin, l, m)] = keep
#                println("KEEP spin $spin n $c  lm $l $m : $(length(keep))")
                nkeep_max= max(length(keep), nkeep_max)
            end
        end
    end

    
    return VECTS_new, keep_dict
    
    
    
end



function run_CI(dat, nexcite; energy_max = 1000000000000.0, nmax = 1000000, dense=false, symmetry=false, Z0=1.0, thr = 1e-12)

    if typeof(nmax) == Int64
        nmax = nmax * ones(Int64, dat.lmax+1)
        for n in 1:dat.lmax
            nmax[n+1] = nmax[n+1] - 1
        end
    end

    ham =  construct_ham(dat, nexcite; energy_max = energy_max, nmax = nmax, dense=dense, symmetry=symmetry, Z0=Z0)
    if isnothing(ham)
        return nothing
    end
    ArrI, ArrJ, ArrH, H, basis_up, basis_dn, s_up, s_dn = ham
    if size(H)[1] < 100
        vals, vects = eigen(collect(H));
    else
        vals, vects = eigs(H, which=:SR);
    end
    println("ENERGY $(vals[1]) ECORR $(vals[1] - dat.etot)")

    println("get_denmat")
    @time denmat = get_denmat(vects, H, basis_up, basis_dn, s_up, s_dn, dat, thr=thr)
    println("end get_denmat")

    
    println("truncate_denmat")
    @time VECTS_new, keep_dict = truncate_denmat(denmat, dat, nmax, thr=thr)
    println("end truncate_denmat")
    
    return ArrI, ArrJ, ArrH, H, vals, vects, basis_up, basis_dn, s_up, s_dn, denmat, VECTS_new, keep_dict
    
end
    




end #end module
