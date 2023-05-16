

function F(x)

#    return sum(sin.(x) .* sum(x))#  / sum(sin.(x).^2 .* x))

    return sum(x.^2 * sum(x))
    
end


function G(x)

    return 2 * x *sum(x) + x.^3
    
    #    return 2*sin.(x).*cos.(x) / sum(sin.(x).^2 .* x) - sin.(x).^2 / sum(sin.(x).^2 .* x).^2 .* (2 .* sin.(x) .* cos.(x) .* x + sin.(x).^2 )

#    return 2*sin.(x).*cos.(x) / sum(sin.(x).^2 .* x) - sin.(x).^2  .* (2*sin.(x) .* cos.(x) .* x + sin.(x).^2 .* x) / sum(sin.(x).^2 .* x)^2 

end
