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

#    println([eps_x, eps_c])
    
    return 2*(eps_x + eps_c)
    
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
