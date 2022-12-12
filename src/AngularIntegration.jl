
module AngularIntegration
using SphericalHarmonics
using Lebedev
using Base.Threads


struct leb

    n::Int64
    N::Int64
    x::Array{Float64, 1}
    y::Array{Float64,1}
    z::Array{Float64,1}

    θ::Array{Float64,1}
    ϕ::Array{Float64,1}
    
    w::Array{Float64,1}
    
    
end

Base.show(io::IO, l::leb) = begin
    println("Lebedev obj")
    println("n=$(l.n) N=$(l.N)")
end

real_gaunt_dict = Dict{NTuple{6,Int64}, Float64}()


function makeleb(n)
    
    for nn = n:n+11
        if Lebedev.isavailable(nn)
            n = nn
            break
        end
    end

    x,y,z,w = Lebedev.lebedev_by_order(n)
    N = length(x)
    println("order $n , length $N")

    θ, ϕ = get_tp(x,y,z)

    return leb(n,N,x,y,z,θ,ϕ,w)

end

function integrate(f, l::leb)

    return 4*pi*sum(f.(l.θ, l.ϕ) .*l.w)

end


function integrate(f, n::Integer)

    l = makeleb(n)

    return integrate(f, l)
    
end

function get_tp(x,y,z)

    θ = zeros(length(x))
    ϕ  = zeros(length(x))    
    for i = 1:length(x)
#        println([x[i],y[i],z[i]])
        if abs(z[i]) < 1e-12 && abs(x[i]^2 + y[i]^2) > 1e-12
            θ[i] = pi/2
        elseif z[i] > 0
            θ[i] = atan( sqrt(x[i]^2+y[i]^2) / z[i])
        elseif z[i] < 0
            θ[i] = pi + atan( sqrt(x[i]^2+y[i]^2) / z[i])
        else
            println("something broke θ")
            θ[i] = 0.0
        end

        if abs(x[i]) < 1e-12 && abs(y[i]) < 1e-12
            ϕ[i] = 0.0
        elseif abs(x[i]) < 1e-12 && y[i] < 0
            ϕ[i] = -pi/2
        elseif abs(x[i]) < 1e-12 && y[i] > 0
            ϕ[i] = pi/2
        elseif x[i] < 0.0  && y[i] < 0
            ϕ[i] = atan(y[i]/x[i]) - pi
        elseif x[i] < 0.0  && y[i] >= 0
            ϕ[i] = atan(y[i]/x[i]) + pi
        elseif x[i] > 0.0
            ϕ[i] = atan(y[i]/x[i])
        else
            println("something broke ϕ else ")
            ϕ[i] = 0.0
        end
    end            
            

        
    
    #    ϕ = sign.(y).*acos.(x ./ sqrt.(x.^2 + y.^2))

    return θ, ϕ

end

function fill_gaunt(;lmax=4, lmax_rho=12, n=23)

    l = makeleb(n)


    
    function y(t,p)
        return SphericalHarmonics.computeYlm(t, p, lmax=lmax_rho, SHType = SphericalHarmonics.RealHarmonics())
    end

    data = y.(l.θ, l.ϕ)

    data_reshape = Dict{NTuple{2,Int64}, Vector{Float64}}()

    for l1 = 0:lmax_rho
        for m1 = -l1:l1
            data_reshape[(l1,m1)] = zeros(l.N)
            for n = 1:l.N
                data_reshape[(l1,m1)][n] = data[n][(l1,m1)]
            end                
        end
    end
            
    
    
#    return data_reshape
    for l1 = 0:lmax_rho
        for m1 = -l1:l1
            for l2 = 0:lmax
                for m2 = -l2:l2
                    for l3 = 0:lmax
                        for m3 = -l3:l3
                            @inbounds real_gaunt_dict[(l1,m1,l2,m2,l3,m3)] = sum(data_reshape[(l1,m1)].*data_reshape[(l2,m2)].*data_reshape[(l3,m3)].*l.w)*4*pi
                        end
                    end
                end
            end
        end
    end

    #    return real_gaunt_dict

end



end #end module
