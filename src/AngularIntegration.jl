
module AngularIntegration

using Lebedev


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

    return sum(f.(l.θ, l.ϕ) .*l.w)

end


function integrate(f, n)

    
    for nn = n:n+11
        if Lebedev.isavailable(nn)
            n = nn
            break
        end
    end

    x,y,z,w = Lebedev.lebedev_by_order(n)

    println("x ", x)
    println("y ", y)
    println("z ", z)

    
    N = length(x)
    println("order $n , length $N")

    θ, ϕ = get_tp(x,y,z)

    println("θ ", θ)
    println("ϕ ", ϕ)

    ftp = f.(θ, ϕ)

    println("ftp ", ftp)


    
    return sum(ftp.*w)
    
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
    

end #end module
