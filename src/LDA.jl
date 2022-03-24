"""
    module LDA

local density approximation module 
"""
module LDA

using QuadGK
using Base.Threads
using ForwardDiff
using FastGaussQuadrature
#using ..ThomasFermi:gettf

function Y(y)

    b = 3.72744
    c = 12.9352
    return y^2 + b*y + c
    
end

function f(ζ)
    return ((1+ζ)^(4/3) + (1 - ζ)^(4/3) - 2 ) / (2 * (2^(1/3) - 1))
end

function F(rs,A,x0,b,c)

    x = rs^0.5
    X = x^2+b*x+c
    X0 = x0^2+b*x0+c
    Q = (4*c-b^2)^0.5
    
    return A*(log(x^2/X) + 2 * b / Q *atan(Q / (2*x + b)) - b * x0 / X0 * (log( (x-x0)^2 / X) + 2*(b + 2 * x0) / Q * atan(Q / (2*x+b))))

    
end

function e_LDA_sp(n, ζ)

#    n=rho_up+rho_dn
    if n < 1e-100
        return 0.0
    end

    #exchange see https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations/atomic-reference-data-electronic-6-3
    
 #   ζ = (rho_up - rho_dn)/n
#    
    rs = (3.0/(4.0*pi*n))^(1.0/3.0)

    e_x_p = -3.0 * (9.0/ (32.0*pi^2))^(1.0/3.0) * rs^(-1.0)
    e_x_f =  e_x_p / (2.0^(-1.0/3.0))

    f = ((1+ζ)^(4/3) + (1 - ζ)^(4/3) - 2 ) / (2 * (2^(1/3) - 1))

#    println("f $f")
    
    ex =  e_x_p  + (e_x_f - e_x_p) * f

    #correlation

    ff2_0 = 1.709920934161365

#    para = [0.0310907, -0.10498, 3.72744, 12.9352]
#    ferro = [0.01554535, -0.32500, 7.06042, 18.0578]
#    stiff = [-1/(6*pi^2), -0.00475840, 1.13107, 13.0045]

    Ap = 0.0310907
    xp = -0.10498
    bp = 3.72744
    cp = 12.9352

    Af = 0.01554535
    xf = -0.32500
    bf = 7.06042
    cf = 18.0578

    As = -1/(6*pi^2)
    xs = -0.00475840
    bs = 1.13107
    cs = 13.0045

    e_c_p = F(rs, Ap,xp,bp,cp)
    e_c_f = F(rs, Af,xf,bf,cf)
    e_c_s = F(rs, As,xs,bs,cs)

    de1 = e_c_f - e_c_p
    
    β = ff2_0 * de1 / e_c_s - 1
    
    ec = e_c_p + e_c_s * (f/ff2_0) * (1 + β * ζ^4)

    #print([2*ex, 2*ec])
    return (ex + 2.0*ec)  #2 for rydberg !?
end


function e_LDA(rho)

#    println("rho $rho")
    
    #Vosko-Wilk-Nusair (VWN)  Vosko, S. H., Wilk, L., and Nusair, M., Can. J. Phys. 58 (1980).
    #non-rel, non spin-polarized

    if rho < 1e-100
        return 0.0
    end
    
    eps_x = -3.0 / ( 4.0 *pi) *(3.0 *pi^2*rho)^(1.0/3.0)

    A = 0.0621814
    b = 3.72744
    c = 12.9352

    y0 = -0.10498

    rs = (3.0 /(4.0*pi*rho))^(1.0/3.0)    
    y = sqrt(rs)
    Q = sqrt(4*c - b^2)
    

    
    eps_c = A/2 * (log(y^2/Y(y)) + 2*b/Q * atan(Q / (2*y+b)) - b * y0 / Y(y0) * (log( (y-y0)^2/Y(y)) + 2*(b+2*y0)/Q *atan(Q /(2*y+b))))
    
    #2 for rydberg units

    #println([2*eps_x, 2*eps_c])
    
    return 2*(eps_x + eps_c)
    
end

function v_LDA_sp(rho_up, rho_dn)

        
    function eup(x)

        rhot = max(x+rho_dn, 1e-100)
        ζ = (x - rho_dn)/rhot

        return rhot*e_LDA_sp(rhot, ζ) / 2.0
    end

    function edn(x)

        rhot = max(rho_up+x, 1e-100)
        ζ = (rho_up - x)/rhot

        return rhot*e_LDA_sp(rhot, ζ) / 2.0
    end
    
    vup = ForwardDiff.derivative(eup ,rho_up)
    vdn = ForwardDiff.derivative(edn ,rho_dn)

    #println("lda ", rho, " ", ONE, " " , TWO)

    return [vup, vdn]
    
end

function v_LDA(rho)
    rho = max(rho, 1e-100)
    
    ONE = e_LDA(rho)

    TWO = rho * ForwardDiff.derivative(e_LDA , rho)

    #println("lda ", rho, " ", ONE, " " , TWO)
    
    return (ONE + TWO)/2.0
    
end

function E_LDA(rho_r; atol=1e-7)

    EN = 4*pi*QuadGK.quadgk( x-> rho_r(x)* e_LDA( rho_r(x) ) * x^2, 0.0, 70.0, atol=atol )[1]

    return EN
    
end

function V_LDA_TOT(rho_r; atol=1e-7)

    EN = 4*pi*QuadGK.quadgk( x-> rho_r(x)* v_LDA( rho_r(x) ) * x^2, 0.0, 70.0, atol=atol )[1]

    return EN
    
end

function V_LDA_TOT(rho_r, rho_r2; atol=1e-7)

    EN = 4*pi*QuadGK.quadgk( x-> rho_r(x)* v_LDA( rho_r2(x) ) * x^2, 0.0, 70.0, atol=atol )[1]

    return EN
    
end

function V_LDA(rho_r, h_arr, inds; atol=1e-7)

    IND, RIND, dim = inds

    dict = Dict()

    H = zeros(Float64, dim, dim)
    
    @threads for i1 = 1:dim
        n1,l1,m1 = RIND[i1]
        for i2 = i1:dim
            n2,l2,m2 = RIND[i2]

            if l1 == l2 && m1 == m2

                if (n1,l1,n2,l2) in keys(dict)
                    v = dict[(n1,l1,n2,l2)]
                else
                
#                    h1 = h_arr[i1]
#                    h2 = h_arr[i2]

                    v = QuadGK.quadgk( x-> h_arr[i1](x)*h_arr[i2](x)*v_LDA( rho_r(x) ) * x^2, 0.0, 70.0, atol=atol )[1]                
                    dict[(n1,l1,n2,l2)] = v

                end
                
                H[i1,i2] = v
                H[i2,i1] = v
                
            end            
        end
    end

    return H
    
end



function V_LDA_gl(h_arr_gl, inds, x_gl, w_gl, rho_rs_gl,  Z)

    
    IND, RIND, dim = inds

    dict = Dict()

    H = zeros(Float64, dim, dim)
    


    f_gl = w_gl .* v_LDA.( rho_rs_gl ) ./ (exp.(-x_gl) )

    
    for i1 = 1:dim
        n1,l1,m1 = RIND[i1]
        for i2 = i1:dim
            n2,l2,m2 = RIND[i2]

            if l1 == l2 && m1 == m2

#                if (n1,l1,n2,l2) in keys(dict)
#                    v = dict[(n1,l1,n2,l2)]
#                else
                
#                    h1 = h_arr[i1]
#                    h2 = h_arr[i2]

#                    v = QuadGK.quadgk( x-> h_arr[i1](x)*h_arr[i2](x)*v_LDA( rho_r(x) ) * x^2, 0.0, 70.0, atol= 1e-8 )[1]

                @inbounds v = sum(f_gl .* h_arr_gl[i1,:] .* h_arr_gl[i2,:]  ) / Z^3
#                v = 0.0
                    #v = sum(f_gl .* h_arr_gl[:, i1] .* h_arr_gl[:, i2]  ) / Z^3

#                    println("v, v0 ", v, "  ", v0, "  ", v0/v)
                    
#                    dict[(n1,l1,n2,l2)] = v

#                end
                
                H[i1,i2] = v
                H[i2,i1] = v
                
            end            
        end
    end

    return H
    
end




function E_LDA_gauss(rho_gl, x, w, c)

    #t = rho_r.(x/c)
    EN = 4 * pi * sum(w .* rho_gl .* e_LDA.( rho_gl )  ./ (exp.(-x) ) ) / c^3
    return EN
    
end

function V_LDA_TOT_gauss(rho_r, rho_r2, x, w, c)

    EN = 4 * pi * sum(w .* rho_r .* v_LDA.(rho_r2 )  ./ (exp.(-x) ) ) / c^3
    #    EN = 4*pi*QuadGK.quadgk( x-> rho_r(x)* v_LDA( rho_r2(x) ) * x^2, 0.0, 70.0, atol=atol )[1]

    return EN
    
end


end #end module
