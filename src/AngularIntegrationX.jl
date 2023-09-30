
module AngularIntegration
using SphericalHarmonics
using Lebedev
using Base.Threads
using ForwardDiff

struct leb

    n::Int64
    N::Int64
    x::Array{Float64, 1}
    y::Array{Float64,1}
    z::Array{Float64,1}

    θ::Array{Float64,1}
    ϕ::Array{Float64,1}
    
    w::Array{Float64,1}
    lmax::Int64
    Ylm::Dict{NTuple{2,Int64}, Vector{Float64}}

    dYlm_theta::Dict{NTuple{2,Int64}, Vector{Float64}}
    dYlm_phi::Dict{NTuple{2,Int64}, Vector{Float64}}
    ddYlm_theta::Dict{NTuple{2,Int64}, Vector{Float64}}
    ddYlm_phi::Dict{NTuple{2,Int64}, Vector{Float64}}
    
end

Base.show(io::IO, l::leb) = begin
    println("Lebedev obj")
    println("n=$(l.n) N=$(l.N) lmax=$(l.lmax)")
end

real_gaunt_dict = Dict{NTuple{6,Int64}, Float64}()


function makeleb(n; lmax = 12 )

    println("makeleb")
    
    if n == 0 || n == 1 #special case
        
        x = zeros(1)
        y = zeros(1)
        z = zeros(1)
        w = ones(1)
        θ = zeros(1)
        ϕ = zeros(1)
        Ylm = Dict{NTuple{2,Int64}, Vector{Float64}}()
        Ylm[(0,0)] = [1/sqrt(4*pi)]

        dYlm_theta = Dict{NTuple{2,Int64}, Vector{Float64}}()
        dYlm_phi = Dict{NTuple{2,Int64}, Vector{Float64}}()
        
        ddYlm_theta = Dict{NTuple{2,Int64}, Vector{Float64}}()
        ddYlm_phi = Dict{NTuple{2,Int64}, Vector{Float64}}()

        dYlm_theta[(0,0)] = [0.0]
        dYlm_phi[(0,0)] = [0.0]
        ddYlm_theta[(0,0)] = [0.0]
        ddYlm_phi[(0,0)] = [0.0]
        
        return leb(1,1,x,y,z,θ,ϕ,w, 0, Ylm, dYlm_phi,ddYlm_theta, ddYlm_phi )
        
    end

    #normal case
    
    for nn = n:n+11
        if Lebedev.isavailable(nn)
            n = nn
            break
        end
    end

    x,y,z,w = Lebedev.lebedev_by_order(n)
    N = length(x)
    println("Lebedev integration order $n , length $N")

    θ, ϕ = get_tp(x,y,z)

    #println("length tp ", length(θ), " ", length(ϕ))
    
    ####

    
    function yfn(t,p)
        return SphericalHarmonics.computeYlm(t, p, lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())
    end

    data = yfn.(θ, ϕ)

    pp=0.0
    tt=0.0

    f_theta = tx->SphericalHarmonics.computeYlm(tx, pp, lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())[:]
    f_phi   = px->SphericalHarmonics.computeYlm(tt, px, lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())[:]

    function f_dtheta(ttx)
        ForwardDiff.derivative(f_theta, ttx)
    end
    function f_dphi(ppx)
        ForwardDiff.derivative(f_phi, ppx)
    end

    function f_ddtheta(ttx)
        ForwardDiff.derivative(f_dtheta, ttx)
    end
    function f_ddphi(ppx)
        ForwardDiff.derivative(f_dphi, ppx)
    end

    DP = []
    DT = []

    DDP = []
    DDT = []
    for (t,p) in zip(θ, ϕ)
        pp=p
        tt=t
        
        push!(DT, f_dtheta(tt))
        push!(DDT, f_ddtheta(tt))

        push!(DP, f_dphi(pp))
        push!(DDP, f_ddphi(pp))
        
    end

    data_theta = zeros(N)
    
#    println("t")
#    println(typeof(data))
#    println("size(data) ", size(data))
    
    Ylm = Dict{NTuple{2,Int64}, Vector{Float64}}()

#    f_dtheta = tt->ForwardDiff.gradient(t->yfn(t, ϕ), θ)
#    data_dtheta
    
    #    f_ddtheta = ForwardDiff.derivative(tt->ForwardDiff.derivative(t->yfn(t, ϕ), tt), θ)
#    f_dphi = ForwardDiff.derivative(p->yfn(θ, p), ϕ)
#    f_ddphi = ForwardDiff.derivative(pp->ForwardDiff.derivative(p->yfn(θ, p), pp), ϕ)
    
                           
    dYlm_theta = Dict{NTuple{2,Int64}, Vector{Float64}}()
    dYlm_phi = Dict{NTuple{2,Int64}, Vector{Float64}}()

    ddYlm_theta = Dict{NTuple{2,Int64}, Vector{Float64}}()
    ddYlm_phi = Dict{NTuple{2,Int64}, Vector{Float64}}()

    counter = 0
    for l1 = 0:lmax
        for m1 = -l1:l1
            counter += 1
            
            Ylm[(l1,m1)] = zeros(N)

            dYlm_theta[(l1,m1)] = zeros(N)
            ddYlm_theta[(l1,m1)] = zeros(N)           

            dYlm_phi[(l1,m1)] = zeros(N)
            ddYlm_phi[(l1,m1)] = zeros(N)           
            for nn = 1:N
                Ylm[(l1,m1)][nn] = data[nn][(l1,m1)]

                dYlm_theta[(l1,m1)][nn] = DT[nn][counter]
                ddYlm_theta[(l1,m1)][nn] = DT[nn][counter]
                dYlm_phi[(l1,m1)][nn] = DT[nn][counter]
                ddYlm_theta[(l1,m1)][nn] = DT[nn][counter]
                
#                dYlm_theta[(l1,m1)][nn] = data_dtheta[nn][(l1,m1)]
#                ddYlm_theta[(l1,m1)][nn] = data_ddtheta[nn][(l1,m1)]

#                dYlm_phi[(l1,m1)][nn] = data_dphi[nn][(l1,m1)]
#                ddYlm_phi[(l1,m1)][nn] = data_ddphi[nn][(l1,m1)]
                
            end                
        end
    end

    ####

#    println("typeof ")
#    for (c,x) =enumerate( [n,N,x,y,z,θ,ϕ,w,lmax,Ylm])
#        println("$c typeof ", typeof(x))
#    end
    
    return leb(n,N,x,y,z,θ,ϕ,w,lmax,Ylm,dYlm_theta, dYlm_phi,ddYlm_theta, ddYlm_phi )

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
