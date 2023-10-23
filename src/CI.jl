
module CI

using ..GalerkinDFT:get_vect_gal
using ..GalerkinDFT:get_Svect_gal
using ..Galerkin:do_1d_integral
using ..AngularIntegration:real_gaunt_dict
using Combinatorics
using SparseArrays
using LoopVectorization
using Suppressor

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

function precalc(dat, s)

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

    @tturbo for L = 0:LMAX
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


function matrix_twobody_small(dat, nlms1, nlms2)

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
    
    return v2'*(dat.V_C  +  dat.D2 + dat.V_L*l1*(l1+1)) * v1, v2'*dat.S*v1

end

function gen_basis(dat,  energy_max, nmax)

    nup = Int64(sum( dat.nel[1,:,:]))
    ndn = Int64(sum( dat.nel[2,:,:]))

    ground_state = Int64.(dat.filling)

    n_occ_max = findlast(sum(ground_state, dims=[2,3,4]) .> 0)[1]
    
    val = zeros(Int64, 0,4)
    cond = zeros(Int64, 0,4)

    val_up = zeros(Int64, 0,4)
    cond_up = zeros(Int64, 0,4)

    val_dn = zeros(Int64, 0,4)
    cond_dn = zeros(Int64, 0,4)

    for spin = 1:dat.nspin
        for l = 0:dat.lmax
            for m = -l:l
                for n = 1:max(min(dat.N-1, nmax-l), n_occ_max)
                    println("add basis spin $spin l $l m $m n $n")
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

    return val_up, val_dn, cond_up, cond_dn, nup, ndn
end

#function generate_excitations(dat, nexcite, energy_max, nmax)
function generate_excitations(nexcite, nup, ndn, val_up, val_dn, cond_up, cond_dn)

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
                    counter += 1
                    v_up[counter,:] .= v_up_t[i,:]
                    v_dn[counter,:] .= v_dn_t[j,:]
#                    println("counter $counter $(v_up[counter,:]) $(v_dn[counter,:])")
                    #                    v_up = vcat(v_up,v_up_t[i,:]')
                    #                    v_dn = vcat(v_dn,v_dn_t[j,:]')
                end
            end
#            println("-")
        end
    end
#    println()
#    println("total ", size(v_up))

    
    return v_up, v_dn

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


function construct_ham(dat, nexcite; energy_max = 1000000000000.0, nmax = 1000000, dense=false)

    if dat.nspin == 1
        println("ERROR - need 2 spins")
        return Nothing
    end
    if abs(sum(dat.nel) - sum(round.(dat.nel))) > 1e-10
        println("ERROR - need integer number of electrons")
        return Nothing
    end

    nel = Int64(sum(dat.nel))
    if  nexcite > nel 
        nexcite=nel
        println("set nexcite to $nexcite")
    end

    #put qualfied states into a list
    val_up, val_dn, cond_up, cond_dn, nup, ndn = gen_basis(dat, energy_max, nmax)

    s_up = [val_up; cond_up]
    s_dn = [val_dn; cond_dn]

    #println("precalc")
    @time VH, RR, basis_dict = precalc(dat, [s_up;s_dn])
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
    basis_up, basis_dn = generate_excitations(nexcite,nup, ndn, val_up, val_dn, cond_up, cond_dn)

#    println("basis_up ")
#    for i = 1:size(basis_up,1)
#        println("b $i   $(basis_up[i,:])  $(basis_dn[i,:])")
#    end
    
#    ham = missing
#    @time @suppress begin
#        ham = make_ham(dat, s_up, s_dn, nup, ndn, basis_up, basis_dn,  dense=dense)
#    end
    ham_pre = missing 
    @time @suppress begin
        ham_pre = make_ham_precalc(dat, s_up, s_dn, nup, ndn, basis_up, basis_dn, basis_dict, VH, RR, dense=dense)
    end
    #return ham[end], ham_pre[end]
    return ham_pre
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

                    if (i == 2 || i == 6 ) && j == 25
                        println("IJ $i $j $h $sign")
                    end
                    
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

function make_ham_precalc(dat, s_up, s_dn, nup, ndn, basis_up, basis_dn, basis_dict, VH, RR; dense=false)

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
                    h = matrix_el_0_precalc(dat, s_up, s_dn, (@view basis_up[i,:]), (@view basis_dn[i,:]), basis_dict, VH, RR)
                end

                if count_up == 2 && count_dn == 0
                    h = matrix_el_1_samespin_precalc(dat, s_up, (@view basis_up[i,:]), s_dn, (@view basis_dn[i,:]), locs_up, basis_dict, VH, RR)
                    sign = (-1.0)^sum(basis_up[i,:][minimum(locs_up)+1:maximum(locs_up)-1])

                end
                if count_up == 0 && count_dn == 2
                    h = matrix_el_1_samespin_precalc(dat, s_dn, (@view basis_dn[i,:]), s_up, (@view basis_up[i,:]), locs_dn, basis_dict, VH, RR)
                    sign = (-1.0)^sum(basis_dn[i,:][minimum(locs_dn)+1:maximum(locs_dn)-1])
                end
                if count_up == 2 && count_dn == 2
                    
                    h = matrix_el_2_diffspin_precalc(dat, s_up, s_dn, locs_up, locs_dn, basis_dict, VH, RR)
                    sign =      (-1.0)^sum( basis_up[i,:][minimum(locs_up)+1:maximum(locs_up)-1])
                    sign = sign*(-1.0)^sum( basis_dn[j,:][minimum(locs_dn)+1:maximum(locs_dn)-1])

                    if (i == 2 || i == 6 ) && j == 25
                        println("IJ $i $j $h $sign")
                    end
                    
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

function find_diff(v1,v2)

    
    count = sum(abs.(v1 - v2))
    if count >= 4
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

function matrix_el_0_precalc(dat, s_up, s_dn,b_up, b_dn, basis_dict, VH, RR)

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

function matrix_el_1_samespin_precalc(dat, s, b, s2, b2, loc, basis_dict, VH, RR )

    h = matrix_twobody_small(dat, s[loc[1],:], s[loc[2],:])[1]
#    println("h0 $h")
    for (c1,v1) = enumerate(b)
        if v1 == 1
            h += 2.0*( matrix_coulomb_small_precalc(dat, (@view s[c1,:]), (@view s[c1,:]), (@view s[loc[1],:]), (@view s[loc[2],:]), basis_dict, VH, RR ) )
        end
    end
#    println("h same hartree $h")
    for (c1,v1) = enumerate(b)
        if v1 == 1
            h += 2.0*(  - matrix_coulomb_small_precalc(dat, (@view s[loc[1],:]), (@view s[c1,:]), (@view s[loc[2],:]), (@view s[c1,:]), basis_dict, VH, RR ))
        end
    end
#    println("h same x $h")

    for (c2,v2) = enumerate(b2)
        if v2 == 1
            h += 2.0*( matrix_coulomb_small_precalc(dat, (@view s[loc[1],:]), (@view s[loc[2],:]), (@view s2[c2,:]), (@view s2[c2,:]), basis_dict, VH, RR )) 
        end
    end
#    println("h diff x $h")
#    println()
 #   println("h1 $h")

    return h
    
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

    h = 2.0*( matrix_coulomb_small_precalc(dat, (@view s_up[locs_up[1],:]), (@view s_up[locs_up[2],:]), (@view s_dn[locs_dn[1],:]),( @view s_dn[locs_dn[2],:]), basis_dict, VH, RR ))

    #h = 2.0*( matrix_coulomb_small(dat, s_up[locs_up[1],:], s_dn[locs_dn[1],:], s_up[locs_up[2],:], s_dn[locs_dn[2],:]))
    
    #    if locs_up[1] == 1 && locs_dn[1] == 1 && locs_up[2] == 5 && locs_dn[2] == 5 && abs(h) > 1e-10
    #        println("diff 2_diffspin $locs_up $locs_dn  s $s_up $s_dn   h $h")
    #    end
    return h
    
end

function  matrix_el_2_samespin(dat, s, locs)
    println("same $locs")
    h =  2.0*( matrix_coulomb_small(dat, s[locs[1],:], s[locs[3],:], s[locs[2],:], s[locs[4],:]))
    #h -= 2.0*( matrix_coulomb_small(dat, s[locs[1],:], s[locs[2],:], s[locs[3],:], s[locs[4],:]))
    return h
    
end



end #end module
