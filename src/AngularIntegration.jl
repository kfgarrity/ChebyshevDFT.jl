
module AngularIntegration
using SphericalHarmonics
using Lebedev
using Base.Threads
using ForwardDiff

θ0 = 0.0


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

    dYlm_phi_sin::Dict{NTuple{2,Int64}, Vector{Float64}}
    Ylm_arr::Array{Float64,3}
end

Base.show(io::IO, l::leb) = begin
    println("Lebedev obj")
    println("n=$(l.n) N=$(l.N) lmax=$(l.lmax)")
end

real_gaunt_dict = Dict{NTuple{6,Int64}, Float64}()

real_gaunt_arr = zeros(14+1, 2*14+1, 6+1, 6*2+1, 6+1, 6*2+1)


function makeleb(n; lmax = 12 )

#    println("makeleb")

    
    if n == 0 || n == 1 #special case
        
        x = zeros(1)
        y = zeros(1)
        z = zeros(1)
        w = ones(1)
        #θ = zeros(1)
        θ = [θ0]
        ϕ = zeros(1)
        Ylm = Dict{NTuple{2,Int64}, Vector{Float64}}()
        Ylm[(0,0)] = [1/sqrt(4*pi)]

        dYlm_theta = Dict{NTuple{2,Int64}, Vector{Float64}}()
        dYlm_phi = Dict{NTuple{2,Int64}, Vector{Float64}}()
        
        ddYlm_theta = Dict{NTuple{2,Int64}, Vector{Float64}}()
        ddYlm_phi = Dict{NTuple{2,Int64}, Vector{Float64}}()

        dYlm_phi_sin = Dict{NTuple{2,Int64}, Vector{Float64}}()

        dYlm_theta[(0,0)] = [0.0]
        dYlm_phi[(0,0)] = [0.0]
        ddYlm_theta[(0,0)] = [0.0]
        ddYlm_phi[(0,0)] = [0.0]

        dYlm_phi_sin[(0,0)] = [0.0]

        Ylm_arr = ones(1,1,1) * 1/sqrt(4*pi)
        
        return leb(1,1,x,y,z,θ,ϕ,w, 0, Ylm, dYlm_theta, dYlm_phi, ddYlm_theta, ddYlm_phi, dYlm_phi_sin, Ylm_arr )
        
    end

    #normal case
    
    for nn = n:n+11
        if Lebedev.isavailable(nn)
            n = nn
            break
        end
    end

    x,y,z,w = Lebedev.lebedev_by_order(n)

    θ, ϕ, w = get_tp(x,y,z, w)

    N = length(w)
#    println("Lebedev integration order $n , length $N")
    
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

    function f_dphi_sin(ppx)
        ForwardDiff.derivative(f_phi, ppx) / sin(tt)
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

    DP_sin = []
    for (t,p) in zip(θ, ϕ)
        pp=p
        tt=t
            
        push!(DT, f_dtheta(tt))
        push!(DDT, f_ddtheta(tt))
        
        push!(DP, f_dphi(pp))
        push!(DDP, f_ddphi(pp))

        
        #this deals with issues related to dividing by sin(θ=0), which appears to diverge but can give finite nonzero answers in the limit at θ → 0 . This is important for gradients of things expanded in Ylm.
        if abs(t) < 1e-10 || abs(t-pi) < 1e-10
            if abs(t) < 1e-10
                tt = 1e-12
                temp = f_dphi(pp)/sin(tt)
                for ii = 1:length(temp)
                    if abs(temp[ii]) < 1e-6
                        temp[ii] = 0.0
                    end
                end
                push!(DP_sin, temp)
            elseif abs(t) < pi
                tt = pi + 1e-12
                temp = f_dphi(pp)/sin(tt)
                for ii = 1:length(temp)
                    if abs(temp[ii]) < 1e-6
                        temp[ii] = 0.0
                    end
                end
                push!(DP_sin, f_dphi(pp)/sin(tt))
            end
        else    
            push!(DP_sin, f_dphi(pp)/sin(tt))
        end
        
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

    dYlm_phi_sin = Dict{NTuple{2,Int64}, Vector{Float64}}()
    
    ddYlm_theta = Dict{NTuple{2,Int64}, Vector{Float64}}()
    ddYlm_phi = Dict{NTuple{2,Int64}, Vector{Float64}}()

    counter = 0

    Ylm_arr = zeros(lmax+1,2*lmax+1,N)

    for l1 = 0:lmax
        for m1 = -l1:l1
            counter += 1
            
            Ylm[(l1,m1)] = zeros(N)

            dYlm_theta[(l1,m1)] = zeros(N)
            ddYlm_theta[(l1,m1)] = zeros(N)           

            dYlm_phi[(l1,m1)] = zeros(N)
            ddYlm_phi[(l1,m1)] = zeros(N)           

            dYlm_phi_sin[(l1,m1)] = zeros(N)           
            for nn = 1:N
                Ylm[(l1,m1)][nn] = data[nn][(l1,m1)]
                Ylm_arr[l1+1,l1+m1+1,nn] = data[nn][(l1,m1)]
                
                dYlm_theta[(l1,m1)][nn] = DT[nn][counter]
                ddYlm_theta[(l1,m1)][nn] = DDT[nn][counter]
                dYlm_phi[(l1,m1)][nn] = DP[nn][counter]
                ddYlm_phi[(l1,m1)][nn] = DDP[nn][counter]

                dYlm_phi_sin[(l1,m1)][nn] = DP_sin[nn][counter]
                
                
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
    
    return leb(n,N,x,y,z,θ,ϕ,w,lmax,Ylm,dYlm_theta, dYlm_phi,ddYlm_theta, ddYlm_phi, dYlm_phi_sin, Ylm_arr)

end

function integrate(f, l::leb)

    return 4*pi*sum(f.(l.θ, l.ϕ) .*l.w)

end


function integrate(f, n::Integer)

    l = makeleb(n)

    return integrate(f, l)
    
end

function get_tp(x,y,z,w)

    θ = zeros(eltype(x), length(x))
    ϕ  = zeros(eltype(x), length(x))    
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


#        if abs(θ[i]) < 1e-12
#            θ[i] = θ0
#            ϕ[i] = 0.0
#        end
#        if abs(θ[i] - pi) < 1e-12
#            θ[i] = pi - θ0
#            ϕ[i] = pi/2
#        end

    end            

    #this is supposed to fix issues with the north/south poles which have undefined phi
    #it doesn't matter for integrals of Ylm, but integrals of the gradient of Ylm get weird because px and py have different gradients at θ=0 for ϕ=0 and ϕ=ϕ/2 even though those are the same point.
    θ2 = eltype(x)[]
    ϕ2 = eltype(x)[]
    w2 = eltype(x)[]
    for i = 1:length(x)
        if abs(θ[i]) < 1e-12
            push!(θ2, 0.0)
            push!(ϕ2, 0.0)

            push!(θ2, 0.0)
            push!(ϕ2, pi/2)

            push!(w2, w[i]/2)
            push!(w2, w[i]/2)
        elseif abs(θ[i] - pi) < 1e-12
            push!(θ2, pi)
            push!(ϕ2, 0.0)

            push!(θ2, pi)
            push!(ϕ2, pi/2)

            push!(w2, w[i]/2)
            push!(w2, w[i]/2)
        else            
            push!(θ2, θ[i])
            push!(ϕ2, ϕ[i])
            push!(w2, w[i])
           
        end
    end
    return θ2, ϕ2, w2

        
    
    #    ϕ = sign.(y).*acos.(x ./ sqrt.(x.^2 + y.^2))


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
    @inbounds for l1 = 0:lmax_rho
        for m1 = -l1:l1
            for l2 = 0:lmax
                for m2 = -l2:l2
                    for l3 = 0:lmax
                        for m3 = -l3:l3
                            temp = sum(data_reshape[(l1,m1)].*data_reshape[(l2,m2)].*data_reshape[(l3,m3)].*l.w)*4*pi
                            real_gaunt_dict[(l1,m1,l2,m2,l3,m3)] = temp
                            real_gaunt_arr[l1+1,l1+m1+1,l2+1,l2+m2+1,l3+1,l3+m3+1] = temp
                        end
                    end
                end
            end
        end
    end

    #    return real_gaunt_dict

end



end #end module
