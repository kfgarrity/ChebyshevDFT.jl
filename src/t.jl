

function smooth(x, thr1, thr2)

    a = log(thr1)
    b = log(thr2)
    y = log.(x)
    t = (y .- b)/(a-b)
    
    return (y .>= a)*1.0 + (y .< a .&& y .> b) .* (0.0 .+ 10.0 * t.^3 .- 15.0 *  t.^4  .+ 6.0 * t.^5) 

end
