using ForwardDiff

function f(x)

    return x^4 + 2.0
    
end


function f1(x)

    return ForwardDiff.derivative(f, x)

end

function f2(x)

    return ForwardDiff.derivative(f1, x)

end


function grid(x)

    return exp(x) - 0.1
    
end

function invgrid(x)

    return log(x + 0.1)
    
end

function invgrid1(x)

    return ForwardDiff.derivative(invgrid, x)
    
end

function grid1(x)

    return ForwardDiff.derivative(grid, x)
    
end


function fg(x)

    return f(grid(x))

end

function f1g(x)

    return f1(grid(x))

end

function g(x)

    return fg(invgrid(x))

end

function g1(x)

    #    return f1g(invgrid(x))*grid1(x)
    return f1g(invgrid(x))

end


function a(x)
    return sin(x^2)
end

function a1(x)
    return cos(x^2)*2*x
end

