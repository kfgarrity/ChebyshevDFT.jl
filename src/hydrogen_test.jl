
#using QuadGK
#using Plots
#using ForwardDiff

Z = 10.0

atol = 1e-11

function psi_1s(r)
    return r/sqrt(pi) * exp(-r * Z) * Z^(3/2)
end
function psi_2s(r)
    return r/4/sqrt(2*pi) * exp(-r/2* Z) * (2-r*Z)* Z^(3/2)
end
function psi_2p(r)
    return r/4/sqrt(2*pi)* Z*r* exp(-r/2 * Z)*Z^(3/2)
end

function VX(psi, L)

    Rmax = 200.0
    return r->-1.0*QuadGK.quadgk(rp->psi_1s(rp)*psi(rp) * (4*pi) * min(r, rp)^L/max(r,rp)^(L+1) , 0, Rmax, atol=atol)[1] * psi_1s(r) / 2.0
    
end

function D2(psi)

    return y -> ForwardDiff.derivative(x->ForwardDiff.derivative(psi, x), y)
    
end

function top()

    Rmax = 200.0
    
    println("norm")
    println("1s 1s ", QuadGK.quadgk(r->psi_1s(r).^2 *4*pi  , 0, Rmax, atol=atol))
    println("1s 2s ", QuadGK.quadgk(r->psi_1s(r)*psi_2s(r) *4*pi  , 0, Rmax, atol=atol))
    println("2s 1s ", QuadGK.quadgk(r->psi_2s(r)*psi_1s(r) *4*pi  , 0, Rmax, atol=atol))
    println("2s 2s ", QuadGK.quadgk(r->psi_2s(r)*psi_2s(r) *4*pi  , 0, Rmax, atol=atol))
    println()
    println("2p 2p ", QuadGK.quadgk(r->psi_2p(r)*psi_2p(r) *2*pi  , 0, Rmax, atol=atol)[1] * QuadGK.quadgk(t->cos(t)^2*sin(t), 0, pi)[1])
    #println("2p 2p ", QuadGK.quadgk(r->psi_2p(r)*psi_2p(r) *2*pi * r^2 , 0, Rmax, atol=atol)[1] * QuadGK.quadgk(t->cos(t)^2*sin(t), 0, pi)[1])
#    println(QuadGK.quadgk(r->psi_2p(r)*psi_2p(r) *2*pi * r^2 , 0, Rmax, atol=atol))
#    println(QuadGK.quadgk(t->cos(t)^2*sin(t), 0, pi))

    println("KE")
    println("1s 1s ", QuadGK.quadgk(r->-0.5*psi_1s(r)*D2(psi_1s)(r) *4*pi  , 0, Rmax, atol=atol))
    println("1s 2s ", QuadGK.quadgk(r->-0.5*psi_1s(r)*D2(psi_2s)(r) *4*pi  , 0, Rmax, atol=atol))
    println("2s 1s ", QuadGK.quadgk(r->-0.5*psi_2s(r)*D2(psi_1s)(r) *4*pi  , 0, Rmax, atol=atol))
    println("2s 2s ", QuadGK.quadgk(r->-0.5*psi_2s(r)*D2(psi_2s)(r) *4*pi  , 0, Rmax, atol=atol))
    println()
    println("PE")
    println("1s 1s ", QuadGK.quadgk(r->-Z/r*psi_1s(r)*psi_1s(r) *4*pi  , 0, Rmax, atol=atol))
    println("1s 2s ", QuadGK.quadgk(r->-Z/r*psi_1s(r)*psi_2s(r) *4*pi  , 0, Rmax, atol=atol))
    println("2s 1s ", QuadGK.quadgk(r->-Z/r*psi_2s(r)*psi_1s(r) *4*pi  , 0, Rmax, atol=atol))
    println("2s 2s ", QuadGK.quadgk(r->-Z/r*psi_2s(r)*psi_2s(r) *4*pi  , 0, Rmax, atol=atol))
    println()

    println("KE")
    factor =  QuadGK.quadgk(t->cos(t)^2*sin(t), 0, pi)[1]
    println("2p 2p ", QuadGK.quadgk(r->-0.5*psi_2p(r)*D2(psi_2p)(r) *2*pi + 0.5*(1)(1+1)/r^2 * psi_2p(r)*psi_2p(r) *2*pi  , 0, Rmax, atol=atol)[1]  * factor)
    
end
    
function test()

    top()

    
    Rmax = 200.0
    

    vh = r->QuadGK.quadgk(rp->1/2 * psi_1s(rp)*psi_1s(rp) * (4*pi) * 1/max(r,rp) , 0, Rmax, atol=atol)[1] 

    #return vh
    
    println("Hartree 1s 1s   ", QuadGK.quadgk(r->vh(r)*psi_1s(r)*psi_1s(r)*4*pi, 0, Rmax, atol=atol)[1])
    println("Hartree 2s 1s   ", QuadGK.quadgk(r->vh(r)*psi_2s(r)*psi_1s(r)*4*pi, 0, Rmax, atol=atol)[1])
    println("Hartree 1s 2s   ", QuadGK.quadgk(r->vh(r)*psi_1s(r)*psi_2s(r)*4*pi, 0, Rmax, atol=atol)[1])
    println("Hartree 2s 2s   ", QuadGK.quadgk(r->vh(r)*psi_2s(r)*psi_2s(r)*4*pi, 0, Rmax, atol=atol)[1])

    println()
    L = 0.0
    println("Vx 1s 1s   ", QuadGK.quadgk(r->VX(psi_1s, L)(r)*psi_1s(r)*4*pi, 0, Rmax, atol=atol)[1])
    println("Vx 1s 2s   ", QuadGK.quadgk(r->VX(psi_1s, L)(r)*psi_2s(r)*4*pi, 0, Rmax, atol=atol)[1])
    println("Vx 2s 1s   ", QuadGK.quadgk(r->VX(psi_2s, L)(r)*psi_1s(r)*4*pi, 0, Rmax, atol=atol)[1])
    println("Vx 2s 2s   ", QuadGK.quadgk(r->VX(psi_2s, L)(r)*psi_2s(r)*4*pi, 0, Rmax, atol=atol)[1])


    factor =  QuadGK.quadgk(t->cos(t)^2*sin(t), 0, pi)[1]
    
    println("Hartree 2p 2p   ", QuadGK.quadgk(r->vh(r)*psi_2p(r)*psi_2p(r)*2*pi, 0, Rmax, atol=atol)[1] * factor)

    L=0.0
    println("Vx 2p 2p L $L  ", QuadGK.quadgk(r->VX(psi_2p, 0.0)(r)*psi_2p(r)*2*pi, 0, Rmax, atol=atol)[1] * factor)

    L=1.0
    println("Vx 2p 2p L $L  ", QuadGK.quadgk(r->VX(psi_2p, 1.0)(r)*psi_2p(r)*2*pi, 0, Rmax, atol=atol)[1] * factor)

    L=2.0
    println("Vx 2p 2p L $L  ", QuadGK.quadgk(r->VX(psi_2p, 2.0)(r)*psi_2p(r)*2*pi, 0, Rmax, atol=atol)[1] * factor)
    
    
    return vh
    
end
