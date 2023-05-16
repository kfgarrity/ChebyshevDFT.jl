

function smooth(x, thr1, thr2)

    a = log(thr1)
    b = log(thr2)
    y = log.(x)
    t = (y .- b)/(a-b)
    
    return (y .>= a)*1.0 + (y .< a .&& y .> b) .* (0.0 .+ 10.0 * t.^3 .- 15.0 *  t.^4  .+ 6.0 * t.^5) 

end


function U(x)
    return exp(cos(x))
end

function iR(RR)
    return 1/α*log( (RR/β + 1)) 
end

function iR1(RR)
    return  ForwardDiff.derivative(iR, RR)
end

function iR2(RR)
    return  ForwardDiff.derivative(iR1, RR)
end


function R(r)
    return β*(exp.(α * r) .- 1)
    #return r
end

function R1(r)
    return ForwardDiff.derivative(R, r)
end

function R2(r)
    return ForwardDiff.derivative(R1, r)
end

function R3(r)
    return ForwardDiff.derivative(R2, r)
end

function U1(x)
    return ForwardDiff.derivative(U, x)
end

function U2(x)
    return ForwardDiff.derivative(U1, x)
end

function UR(x)
    return U(R(x))
end

function UR1(x)
    return ForwardDiff.derivative(UR, x) / R1(x)
end

function UR2(x)
    a = ForwardDiff.derivative(xxx->ForwardDiff.derivative(UR, xxx), x) / (R1(x))
#    b = ForwardDiff.derivative(UR, x) / R1(x)^2 * R2(x)
    #    return a -  b

    #    return ForwardDiff.derivative(UR1, x) / R1(x)

#    a = ForwardDiff.derivative(xx-> ForwardDiff.derivative(UR, xx), x) / R1(x)
    
    #b = ForwardDiff.derivative(UR, x) * ForwardDiff.derivative(xx->1/R1(xx), x)

    b = ForwardDiff.derivative(UR, x) * (-1)/R1(x)^2 * R2(x)
    
    return (a + b)  / R1(x)
    
end
