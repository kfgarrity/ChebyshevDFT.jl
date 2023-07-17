
using QuadGK
using Plots
using ForwardDiff

Z = 10.0

atol = 1e-10

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

    println("KE 2p")
    factor =  QuadGK.quadgk(t->cos(t)^2*sin(t), 0, pi)[1]
    println("2p 2p ", QuadGK.quadgk(r->-0.5*psi_2p(r)*D2(psi_2p)(r) *2*pi + 0.5*(1)(1+1)/r^2 * psi_2p(r)*psi_2p(r) *2*pi  , 0, Rmax, atol=atol)[1]  * factor)
    println()
    
end
    
function test()

#    top()

    
    Rmax = 200.0
    

    vh = r->QuadGK.quadgk(rp->1/2 * psi_1s(rp)*psi_1s(rp) * (4*pi) * 1/max(r,rp) , 0, Rmax, atol=atol)[1] 

    factor =  QuadGK.quadgk(t->cos(t)^2*sin(t), 0, pi, atol=atol)[1]
    println("factor $factor")

    #=
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

    println()
    println("Hartree 2p 2p   ", QuadGK.quadgk(r->vh(r)*psi_2p(r)*psi_2p(r)*2*pi, 0, Rmax, atol=atol)[1] * factor)
    println()
    L=0.0
    println("Vx 2p 2p L $L  ", QuadGK.quadgk(r->VX(psi_2p, 0.0)(r)*psi_2p(r)*2*pi, 0, Rmax, atol=atol)[1] * factor)

    L=1.0
    println("Vx 2p 2p L $L  ", QuadGK.quadgk(r->VX(psi_2p, 1.0)(r)*psi_2p(r)*2*pi, 0, Rmax, atol=atol)[1] * factor)

    L=2.0
    println("Vx 2p 2p L $L  ", QuadGK.quadgk(r->VX(psi_2p, 2.0)(r)*psi_2p(r)*2*pi, 0, Rmax, atol=atol)[1] * factor)
    
    println("hartree 1s 1s bigfloat ", hcubature(r->1/2 * psi_1s(r[1])*psi_1s(r[1]) * (BigFloat(4)*pi) * 1/max(r[1],r[2]) *psi_1s(r[2])*psi_1s(r[2])*4*pi, [0.0,0.0], [Rmax, Rmax], reltol=1e-11))
=#
    println("hartree 2p 2p bigfloat ", hcubature(r->1/2 * psi_1s(r[1])*psi_1s(r[1]) * (BigFloat(4)*pi) * 1/max(r[1],r[2]) *psi_2p(r[2])*psi_2p(r[2])*4*pi, [0.0,0.0], [Rmax, Rmax], reltol=1e-11)[1] * factor)
    L = 0.0
    println("hartree 2p 2p VX = L $L bigfloat ", hcubature(r-> psi_1s(r[1])*psi_2p(r[1]) * (4*pi) * min(r[2], r[1])^L/max(r[2],r[1])^(L+1) * psi_1s(r[2]) / 2.0* psi_2p(r[2])*BigFloat(2)*pi  , [0.0,0.0], [Rmax, Rmax], reltol=1e-11)[1] * factor)
    L = 1.0
    println("hartree 2p 2p VX = L $L bigfloat ", hcubature(r-> psi_1s(r[1])*psi_2p(r[1]) * (4*pi) * min(r[2], r[1])^L/max(r[2],r[1])^(L+1) * psi_1s(r[2]) / 2.0* psi_2p(r[2])*BigFloat(2)*pi  , [0.0,0.0], [Rmax, Rmax], reltol=1e-11)[1] * factor)
    L = 2.0
    println("hartree 2p 2p VX = L $L bigfloat ", hcubature(r-> psi_1s(r[1])*psi_2p(r[1]) * (4*pi) * min(r[2], r[1])^L/max(r[2],r[1])^(L+1) * psi_1s(r[2]) / 2.0* psi_2p(r[2])*BigFloat(2)*pi  , [0.0,0.0], [Rmax, Rmax], reltol=1e-11)[1] * factor)

    
    return vh
    
end


#=

norm
1s 1s (1.0, 5.353535364597598e-11)
1s 2s (-2.4069288229178198e-17, 9.972397395297541e-11)
2s 1s (-2.4069288229178198e-17, 9.972397395297541e-11)
2s 2s (1.0, 4.389966342231663e-11)

2p 2p 0.9999999999999998
KE
1s 1s (50.0, 9.887570431024624e-12)
1s 2s (20.951312035156963, 3.406419463464356e-11)
2s 1s (20.951312035156963, 6.990710329794456e-11)
2s 2s (12.500000000000002, 4.535891510795544e-11)

PE
1s 1s (-100.0, 6.172675320404562e-12)
1s 2s (-20.95131203515697, 2.3472844733402143e-11)
2s 1s (-20.951312035156967, 2.347681033837328e-11)
2s 2s (-25.000000000000007, 7.63538026237258e-11)

KE 2p
2p 2p 12.500000000000002

Hartree 1s 1s   3.1250000629590624
Hartree 2s 1s   0.44677518193882765
Hartree 1s 2s   0.44677518193882765
Hartree 2s 2s   1.0493827207730162

Vx 1s 1s   -3.125000062982126
Vx 1s 2s   -0.446775181945741
Vx 2s 1s   -0.4467751819564384
Vx 2s 2s   -0.10973937376736473

factor 0.6666666666666666
Hartree 2p 2p   1.2139917750937927

Vx 2p 2p L 0.0  -0.4023776918925356
Vx 2p 2p L 1.0  -0.25605854445779563
Vx 2p 2p L 2.0  -0.18289897635858904
hartree 1s 1s bigfloat (3.125000000004513, 3.124945758476997e-11)

factor 0.6666666666666666
hartree 2p 2p bigfloat 2.4279835390979674
hartree 2p 2p VX = L 0.0 bigfloat 0.4023776863288856
hartree 2p 2p VX = L 1.0 bigfloat 0.2560585276638591
hartree 2p 2p VX = L 2.0 bigfloat 0.1828989483313258


=#
