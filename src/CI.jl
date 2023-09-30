
module CI

using ..GalerkinDFT:get_vect_gal
using ..Galerkin:do_1d_integral

function matrix_coulomb_small(dat, nlms1, nlms2, nlms3, nlms4)

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










end #end module
