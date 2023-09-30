

    vx2 =  zeros(N-1,N-1)
    VX_LM2 = zeros(N-1, N-1, nspin, lmax+1, 2*lmax+1)

    for spin = 1:nspin
        vx2 .= 0.0
        for n = 1:N-1
            f = filling[n,spin,1,1]
            println("f $spin $n  $f")
            if f < 1e-20
                break
            end
            t = mat_n2m*real(VECTS[:,n,nspin,1,1])
            tf1 =  t ./ R  / sqrt(g.b-g.a) * sqrt(2 )
            temp = diagm(tf1) * mat_m2n' * S * L * S *  mat_m2n * diagm(tf1)
            vx2[:,:] +=   real(  f * mat_n2m'*temp*mat_n2m)
            println("test s $spin n $n VX", VECTS[:,1,1,1,1]' * 4*pi*vx2* VECTS[:,1,1,1,1])
        end
        VX_LM2[:,:,spin,1,1] = vx2
    end


function f(rho_LM)

    
    lmax = size(rho_LM)[3] - 1
    nspin = size(rho_LM)[2]
    nr = size(rho_LM)[1]

    println("begin")
    @time begin
    Y_dict = ChebyshevDFT.AngMom.Y_dict
    Ytheta_dict = ChebyshevDFT.AngMom.Ytheta_dict
    Yphi_dict = ChebyshevDFT.AngMom.Yphi_dict

    Ytheta2_dict = ChebyshevDFT.AngMom.Ytheta2_dict
    Yphi2_dict = ChebyshevDFT.AngMom.Yphi2_dict
    
    THETA,PHI = FastSphericalHarmonics.sph_points(lmax+1)

    
#    rho_thr = zeros(lmax+1, size(rho_LM)[4],nr, nspin)

        rho_thr = zeros(nr, nspin, lmax+1, size(rho_LM)[4])#, nthreads())
    drho_thr = zeros(nr, nspin, lmax+1, size(rho_LM)[4], 3, nthreads())
    ddrho_thr = zeros(nr, nspin, lmax+1, size(rho_LM)[4], 3, nthreads())


    phi = 0.0
    theta = 0.0

    end
    println("loop")
    @time for t in 1:length(THETA)
        theta = THETA[t]

        for (p,phi) in enumerate(PHI) 

            
            for l = 0:lmax
#                    id = threadid()
                    for m = -l:l
#                        y = Y_dict[(lmax, t,p,l,m)]
                        d = FastSphericalHarmonics.sph_mode(l,m)
                        #for spin = 1:nspin

                        #                        rho_thr[:,:,t, p] .+= (@view rho_LM[:,:, d[1], d[2]]) * Y_dict[(lmax, t,p,l,m)]

                        #rho_thr[:,:,t, p] .+= (@view rho_LM[:,:, d[1], d[2]]) * Y_dict[(lmax, t,p,l,m)]
                        rho_thr[:,:,t, p] .+=  (@view rho_LM[:,:, d[1], d[2]]) #* Y_dict[(lmax, t,p,l,m)]

                        if false

                        rho_thr[:,spin,t, p, id] += (@view rho_LM[:,spin, d[1], d[2]]) * Y_dict[(lmax, t,p,l,m)]

                        if true

                            drho_thr[:,spin,t, p,1, id] += ( @view drho_LM[:,spin, d[1], d[2]]) * Y_dict[(lmax, t,p,l,m)]
                            drho_thr[:,spin,t, p,2, id] += ( @view rho_LM[:,spin, d[1], d[2]]) *  Ytheta_dict[(lmax, t,p,l,m)]
                            drho_thr[:,spin,t, p,3, id] += ( @view rho_LM[:,spin, d[1], d[2]]) *  Yphi_dict[(lmax, t,p,l,m)]

                            ddrho_thr[:,spin,t, p,2, id] += ( @view  rho_LM[:,spin, d[1], d[2]]) *  Ytheta2_dict[(lmax,t,p,l,m)]
                            ddrho_thr[:,spin,t, p,3, id] += ( @view  rho_LM[:,spin, d[1], d[2]]) *  Yphi2_dict[(lmax,t,p,l,m)]

#                        end
                        end
                    end
                end
            end
            
        end
        
    end
end
