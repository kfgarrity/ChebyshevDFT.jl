using ChebyshevDFT
using Test
using QuadGK
using ForwardDiff

tol_var=1e-12

@testset "gal" begin

    function f(x); return sin(x/4*pi); end
    gal = ChebyshevDFT.Galerkin.makegal(30, 0, 20.0, α = 0.1, M = 100)

    q = QuadGK.quadgk(f,0,  20.0)[1] * pi
    g = ChebyshevDFT.Galerkin.do_1d_integral(f, gal) * pi

#    println("q $q g $g")
    
    @test abs(q - 8.0) < tol_var
    @test abs(g - 8.0) < tol_var
    

    rep = ChebyshevDFT.Galerkin.get_gal_rep(f, gal)

    g2 = ChebyshevDFT.Galerkin.do_1d_integral(rep, gal) * pi

    @test abs(g2 - 8.0) < tol_var
    
    fr0 = ChebyshevDFT.Galerkin.gal_rep_to_rspace(0.1, rep, gal; deriv=0)

    @test abs( fr0 - f(0.1)) < tol_var

    df = x->ForwardDiff.derivative(f, x)

    function df_analytic(x); return cos(x/4*pi)/4*pi ; end

    @test abs( df_analytic(0.1) - df(0.1)) < tol_var

    fr1 = ChebyshevDFT.Galerkin.gal_rep_to_rspace(0.1, rep, gal; deriv=1)

    @test abs( fr1 - df(0.1)) < tol_var


    #---- Lower convergence model
    
    tol_var=1e-3

    gal = ChebyshevDFT.Galerkin.makegal(15, 0, 20.0, α = 0.1, M = 30)

    q = QuadGK.quadgk(f,0,  20.0)[1] * pi
    g = ChebyshevDFT.Galerkin.do_1d_integral(f, gal) * pi

#    println("q $q g $g")
    
    @test abs(q - 8.0) < tol_var
    @test abs(g - 8.0) < tol_var
    

    rep = ChebyshevDFT.Galerkin.get_gal_rep(f, gal)

    g2 = ChebyshevDFT.Galerkin.do_1d_integral(rep, gal) * pi

    @test abs(g2 - 8.0) < tol_var
    
    fr0 = ChebyshevDFT.Galerkin.gal_rep_to_rspace(0.1, rep, gal; deriv=0)

    @test abs( fr0 - f(0.1)) < tol_var

    df = x->ForwardDiff.derivative(f, x)

    function df_analytic(x); return cos(x/4*pi)/4*pi ; end

    @test abs( df_analytic(0.1) - df(0.1)) < tol_var

    fr1 = ChebyshevDFT.Galerkin.gal_rep_to_rspace(0.1, rep, gal; deriv=1)

    @test abs( fr1 - df(0.1)) < tol_var
    
end
